"""v3 residual-on-v1 recurrent wavelet front-end.

The base sequence is produced by ``WaveletTSEncoder._encode_channel`` itself:
coarse band, global detail top-k, value projection, band embedding, Fourier
positions and padding are therefore exactly v1.  Its already-computed pyramid
is reused by a coarse-to-fine recurrent reader.  The reader's context is added
through a learned scalar::

    output = v1_tokens + context_scale * token_gate * context

The class default remains zero so identity tests and old residual-on-v1
checkpoints retain their exact semantics.  Run configs use a small non-zero
initial value to avoid the zero-scale gradient deadlock.  The loop can only add
context; no normalization or learned transform touches the base-token path.
Base-token positions and bands are retained for the real-position locality
prior.
"""

from __future__ import annotations

import math
import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.models.wavelet_encoder import WaveletTSEncoder


class _LoopBlock(nn.Module):
    """One iteration of the loop: gated cross-attention read of the current
    level, self-attention among residual tokens, MLP.  Pre-LN blocks."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: int,
        dropout: float,
        self_attn_scale_init: float,
        mlp_scale_init: float,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln_s = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim)
        )
        self.self_attn_scale = nn.Parameter(torch.tensor(float(self_attn_scale_init)))
        self.mlp_scale = nn.Parameter(torch.tensor(float(mlp_scale_init)))

    def forward(
        self,
        z: torch.Tensor,
        query: torch.Tensor,
        e: torch.Tensor,
        gate: torch.Tensor | None,
        position_bias: torch.Tensor,
        reader_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, kv = self.ln_q(query), self.ln_kv(e)
        attn_mask = position_bias.repeat_interleave(self.n_heads, dim=0)
        read, _ = self.cross(
            q, kv, kv, attn_mask=attn_mask, need_weights=False
        )
        if gate is not None:
            read = read * gate
        z = z + read
        if reader_mode == "cross_only":
            zero = torch.zeros_like(read)
            return z, read, zero, zero
        s = self.ln_s(z)
        mix, _ = self.self_attn(s, s, s, need_weights=False)
        mix = self.self_attn_scale.to(mix.dtype) * mix
        z = z + mix
        mlp = self.mlp_scale.to(z.dtype) * self.mlp(self.ln_m(z))
        return z + mlp, read, mix, mlp


class RecurrentWaveletTSEncoder(WaveletTSEncoder):
    multi_variable_default = False

    def __init__(
        self,
        output_dim: int = 512,
        num_tokens: int = 256,
        max_levels: int = 10,
        coarse_blocks: int = 128,
        n_pos_freqs: int = 16,
        multi_variable: bool = False,
        freeze: bool = False,
        level_topm: int = 1024,
        tied: bool = True,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        level_gate: bool = True,
        depth_mode: str = "full",
        sdepth_min: int = 1,
        noise_tau: float = 1.0,
        pos_dtype: str = "input",
        locality_sigma: float = 0.0078125,
        context_scale_init: float = 0.0,
        reader_mode: str = "full",
        context_mode: str = "full",
        self_attn_scale_init: float = 1.0,
        mlp_scale_init: float = 1.0,
        token_gate: bool = False,
        wavelet: str = "haar",
        use_coarse: bool = True,
        **_,
    ):
        if depth_mode not in ("full", "sdepth", "noise_gate"):
            raise ValueError(f"depth_mode must be full|sdepth|noise_gate, got {depth_mode!r}")
        if pos_dtype not in ("input", "fp32"):
            raise ValueError(f"pos_dtype must be input|fp32, got {pos_dtype!r}")
        if reader_mode not in ("full", "cross_only"):
            raise ValueError(
                f"reader_mode must be full|cross_only, got {reader_mode!r}"
            )
        if context_mode not in ("full", "local", "global"):
            raise ValueError(
                f"context_mode must be full|local|global, got {context_mode!r}"
            )
        if int(output_dim) <= 0 or int(num_tokens) <= 0:
            raise ValueError("output_dim and num_tokens must be positive")
        if int(max_levels) < 1 or int(coarse_blocks) < 1:
            raise ValueError("max_levels and coarse_blocks must be >= 1")
        if int(n_pos_freqs) < 1 or int(level_topm) < 1:
            raise ValueError("n_pos_freqs and level_topm must be >= 1")
        if int(n_heads) < 1 or int(output_dim) % int(n_heads) != 0:
            raise ValueError("n_heads must be positive and divide output_dim")
        if int(mlp_ratio) < 1:
            raise ValueError("mlp_ratio must be >= 1")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if int(sdepth_min) < 1:
            raise ValueError("sdepth_min must be >= 1")
        if float(noise_tau) < 0.0 or not math.isfinite(float(noise_tau)):
            raise ValueError("noise_tau must be finite and >= 0")
        if float(locality_sigma) <= 0.0 or not math.isfinite(float(locality_sigma)):
            raise ValueError("locality_sigma must be finite and > 0")
        if not math.isfinite(float(context_scale_init)):
            raise ValueError("context_scale_init must be finite")
        if not math.isfinite(float(self_attn_scale_init)):
            raise ValueError("self_attn_scale_init must be finite")
        if not math.isfinite(float(mlp_scale_init)):
            raise ValueError("mlp_scale_init must be finite")

        # This creates the complete v1 token path with the exact same module
        # names, parameter initialization and dtype behavior as v1.
        super().__init__(
            output_dim=output_dim,
            num_tokens=num_tokens,
            max_levels=max_levels,
            coarse_blocks=coarse_blocks,
            n_pos_freqs=n_pos_freqs,
            multi_variable=multi_variable,
            freeze=freeze,
            wavelet=wavelet,
            use_coarse=use_coarse,
        )
        self.level_topm = int(level_topm)
        self.tied = bool(tied)
        self.level_gate = bool(level_gate)
        self.depth_mode = depth_mode
        self.sdepth_min = max(1, int(sdepth_min))
        self.noise_tau = float(noise_tau)
        self.pos_dtype = pos_dtype
        self.locality_sigma = float(locality_sigma)
        self.reader_mode = reader_mode
        self.context_mode = context_mode
        self.token_gate_enabled = bool(token_gate)
        # The audit found a large token-shared mode.  Cross-only removes it;
        # controlled full readers approach the same local baseline smoothly,
        # while the historical full/1.0 default has strength zero and is
        # therefore bit-for-bit unchanged.
        if self.reader_mode == "cross_only":
            self.local_center_strength = 1.0
        else:
            transform_strength = max(
                abs(float(self_attn_scale_init)), abs(float(mlp_scale_init))
            )
            self.local_center_strength = max(0.0, 1.0 - min(1.0, transform_strength))
        D = self.output_dim
        n_steps = self.max_levels + 1  # coarse + up to max_levels detail levels

        # --- the loop -------------------------------------------------------
        self.step_embed = nn.Embedding(n_steps, D)  # tells the shared block which iteration it is
        nn.init.normal_(self.step_embed.weight, mean=0.0, std=0.02)
        blocks = 1 if self.tied else n_steps
        self.blocks = nn.ModuleList(
            [
                _LoopBlock(
                    D, n_heads, mlp_ratio, dropout,
                    self_attn_scale_init, mlp_scale_init,
                )
                for _ in range(blocks)
            ]
        )
        # gate in [0,1] from [mean base query, mean level tokens, level stats]
        self.gate_head = nn.Linear(2 * D + 4, 1) if self.level_gate else None
        if self.gate_head is not None:
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, 2.0)  # sigmoid(2) ~ 0.88: starts almost open
        # Only the residual is normalized.  The v1 base tokens bypass this
        # layer entirely and are added unchanged at the final line.
        self.context_norm = nn.LayerNorm(D)
        self.context_scale = nn.Parameter(torch.tensor(float(context_scale_init)))
        self.context_token_gate = nn.Linear(2 * D, 1) if self.token_gate_enabled else None
        if self.context_token_gate is not None:
            nn.init.zeros_(self.context_token_gate.weight)
            nn.init.zeros_(self.context_token_gate.bias)

        # Diagnostics stay out of state_dict.  Logging is opt-in because it
        # belongs to the run harness, not to the model architecture.
        try:
            self.gate_log_every = int(
                os.environ.get("TSLM_V3_GATE_LOG_EVERY", "0")
            )
        except ValueError as exc:
            raise ValueError("TSLM_V3_GATE_LOG_EVERY must be an integer >= 0") from exc
        if self.gate_log_every < 0:
            raise ValueError("TSLM_V3_GATE_LOG_EVERY must be >= 0")
        self._forward_calls = 0
        self.last_depth = 0
        self.last_steps = 0
        self.last_gates: list[float] = []
        self.last_active: list[float] = []
        self.last_context_stats: dict[str, object] = {}
        self.last_base_positions: torch.Tensor | None = None
        self.last_base_bands: torch.Tensor | None = None
        self.last_base_valid: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def _block(self, step: int) -> _LoopBlock:
        return self.blocks[0] if self.tied else self.blocks[step]

    def _local_context(self, z: torch.Tensor) -> torch.Tensor:
        if self.local_center_strength == 0.0:
            return z
        return z - self.local_center_strength * z.mean(dim=-2, keepdim=True)

    def _context_components(
        self,
        z: torch.Tensor,
        base_bands: torch.Tensor,
        base_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split valid slots into within-band variation and exact-band means.

        Padding is excluded from every mean.  It remains in the local
        component so ``local + global == full`` over the complete tensor,
        while all semantic invariants are defined on ``base_valid`` slots.
        """
        if base_bands.shape != z.shape[:-1] or base_valid.shape != z.shape[:-1]:
            raise ValueError("context metadata shape must match context token axes")
        global_context = torch.zeros_like(z)
        for band in range(self.max_levels + 1):
            mask = base_valid & (base_bands == band)
            count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
            mean = (
                (z * mask.unsqueeze(-1)).sum(dim=-2, keepdim=True)
                / count.unsqueeze(-1)
            )
            global_context = global_context + mean * mask.unsqueeze(-1)
        return z - global_context, global_context

    def _context_for_mode(
        self,
        z: torch.Tensor,
        base_bands: torch.Tensor,
        base_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Select the configured component before the context LayerNorm."""
        full = self._local_context(z)
        if self.context_mode == "full":
            # Deliberately return the historical tensor directly: the default
            # forward must remain bit-for-bit identical.
            return full
        local, global_context = self._context_components(
            full, base_bands, base_valid
        )
        return local if self.context_mode == "local" else global_context

    @staticmethod
    def _rms(value: torch.Tensor) -> torch.Tensor:
        return value.detach().float().square().mean().sqrt()

    def _context_snapshot(
        self,
        z: torch.Tensor,
        cross: torch.Tensor | None = None,
        self_attn: torch.Tensor | None = None,
        mlp: torch.Tensor | None = None,
    ) -> dict[str, float | None]:
        """Detached context stats over every token and feature dimension."""
        with torch.no_grad():
            z_all = z.detach().float()
            h = self.context_norm(z.detach()).float()
            if self.last_base_bands is None or self.last_base_valid is None:
                raise RuntimeError("context diagnostics require base token metadata")
            bands = self.last_base_bands
            valid = self.last_base_valid
            if bands.shape != h.shape[:-1] or valid.shape != h.shape[:-1]:
                raise RuntimeError("context diagnostic metadata shape mismatch")
            eps = torch.finfo(h.dtype).eps

            def masked_rms(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                weight = mask.unsqueeze(-1).float()
                count = (mask.sum() * value.shape[-1]).clamp_min(1)
                return (value.float().square() * weight).sum().div(count).sqrt()

            def group_stats(mask: torch.Tensor) -> dict[str, float | None]:
                h_rms = masked_rms(h, mask).clamp_min(eps)
                count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
                mean = (
                    (h * mask.unsqueeze(-1)).sum(dim=-2, keepdim=True)
                    / count.unsqueeze(-1)
                )
                variable = masked_rms(h - mean, mask) / h_rms
                unit = F.normalize(h, dim=-1, eps=eps) * mask.unsqueeze(-1)
                pair_count = count.squeeze(-1) * (count.squeeze(-1) - 1)
                pair_sum = unit.sum(dim=-2).square().sum(dim=-1) - count.squeeze(-1)
                usable = pair_count > 0
                if bool(usable.any()):
                    cosine_value: float | None = float(
                        (pair_sum[usable].sum() / pair_count[usable].sum()).cpu()
                    )
                else:
                    cosine_value = None
                if h.shape[0] > 1:
                    pair_mask = mask[1:] & mask[:-1]
                    if bool(pair_mask.any()):
                        series_value: float | None = float(
                            (masked_rms(h[1:] - h[:-1], pair_mask) / h_rms).cpu()
                        )
                    else:
                        series_value = None
                else:
                    series_value = None
                return {
                    "z_rms": float(masked_rms(z_all, mask).cpu()),
                    "h_rms": float(h_rms.cpu()),
                    "token_variable_fraction": float(variable.cpu()),
                    "mean_token_cosine": cosine_value,
                    "input_series_relative": series_value,
                }

            masks = {
                "total": valid,
                "coarse": valid & (bands == 0),
                "detail": valid & (bands >= 1),
            }
            grouped = {name: group_stats(mask) for name, mask in masks.items()}
            total = grouped["total"]
            n_tokens = h.shape[-2]
            shift = max(1, n_tokens // 3)
            rolled_valid = valid.roll(shift, dims=-1)
            translation_mask = valid & rolled_valid
            token_translation = (
                masked_rms(h - h.roll(shift, dims=-2), translation_mask)
                / max(float(total["h_rms"]), eps)
            )

            def component_rms(value: torch.Tensor | None) -> float:
                if value is None:
                    return 0.0
                return float(masked_rms(value.detach(), valid).cpu())

            result: dict[str, float | None] = {
                # Backward-compatible aliases now explicitly mean "total".
                "z_rms": total["z_rms"],
                "h_rms": total["h_rms"],
                "token_variable_fraction": total["token_variable_fraction"],
                "mean_token_cosine": total["mean_token_cosine"],
                "input_series_relative": total["input_series_relative"],
                "token_translation_relative": float(token_translation.cpu()),
                "cross_rms": component_rms(cross),
                "self_attn_rms": component_rms(self_attn),
                "mlp_rms": component_rms(mlp),
            }
            for group, values in grouped.items():
                for key, value in values.items():
                    result[f"{key}_{group}"] = value
            return result

    def _merge_context(
        self,
        base_tokens: torch.Tensor,
        context: torch.Tensor,
        token_gate_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the output boundary; override exists for exact gate tests/audits."""
        if token_gate_override is not None:
            gate = token_gate_override.to(device=context.device, dtype=context.dtype)
        elif self.context_token_gate is not None:
            base_normalized = F.layer_norm(base_tokens, (base_tokens.shape[-1],))
            gate_input = torch.cat([base_normalized, context], dim=-1)
            gate = 2.0 * torch.sigmoid(self.context_token_gate(gate_input))
        else:
            gate = torch.ones_like(context[..., :1])
        scale = self.context_scale.to(context.dtype)
        if os.environ.get("TSLM_V3_CONTEXT_OFF") == "1":
            scale = scale * 0.0
        return base_tokens + scale * gate * context, gate

    def _tokens(self, v: torch.Tensor, pos: torch.Tensor, band: int) -> torch.Tensor:
        """v, pos: (B, m) -> (B, m, D); same featurisation as v1."""
        if self.pos_dtype == "fp32":
            ang = pos.float().unsqueeze(-1) * self.pos_freqs.float()
            sc = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).to(v.dtype)
        else:  # exactly v1: positions in the input dtype, buffer dtype = module dtype
            ang = pos.unsqueeze(-1) * self.pos_freqs
            sc = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).to(v.dtype)
        feat = torch.cat([v.unsqueeze(-1).clamp(-20, 20), sc], dim=-1)
        b = torch.full(v.shape, band, dtype=torch.long, device=v.device)
        return self.value_proj(feat) + self.band_embed(b)

    def _requested_depth(self, n_steps: int) -> int:
        env = os.environ.get("TSLM_V3_DEPTH")
        if env is not None and env != "":
            try:
                requested = int(env)
            except ValueError as exc:
                raise ValueError(
                    f"TSLM_V3_DEPTH must be an integer >= 1, got {env!r}"
                ) from exc
            if requested < 1:
                raise ValueError(
                    f"TSLM_V3_DEPTH must be >= 1, got {requested}"
                )
            return min(n_steps, requested)
        if self.depth_mode == "sdepth" and self.training:
            lo = min(self.sdepth_min, n_steps)
            return int(torch.randint(lo, n_steps + 1, (1,)).item())
        return n_steps

    def _v1_tokens_and_pyramid(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Run v1's exact method while recording its single DWT evaluation.

        A private copy of the function globals lets us wrap the pyramid call
        without mutating the imported v1 module (important for concurrency).
        The executed bytecode is WaveletTSEncoder._encode_channel itself, not a
        fork of its token-selection implementation.
        """
        method = WaveletTSEncoder._encode_channel
        fn_globals = method.__globals__.copy()
        captured: dict[str, tuple[list[torch.Tensor], torch.Tensor]] = {}

        original_haar = fn_globals["haar_pyramid"]
        original_filter = fn_globals["filter_pyramid"]

        def capture_haar(values: torch.Tensor, levels: int):
            result = original_haar(values, levels)
            captured["pyramid"] = result
            return result

        def capture_filter(values: torch.Tensor, levels: int, lo: torch.Tensor, hi: torch.Tensor):
            result = original_filter(values, levels, lo, hi)
            captured["pyramid"] = result
            return result

        fn_globals["haar_pyramid"] = capture_haar
        fn_globals["filter_pyramid"] = capture_filter
        exact_v1 = types.FunctionType(
            method.__code__, fn_globals, method.__name__, method.__defaults__, method.__closure__
        )
        exact_v1.__kwdefaults__ = method.__kwdefaults__
        base = exact_v1(self, x)
        if "pyramid" not in captured:
            raise RuntimeError("v1 encoder did not evaluate a wavelet pyramid")
        details, coarse = captured["pyramid"]
        return base, details, coarse

    def _base_metadata(
        self,
        details: list[torch.Tensor],
        coarse: torch.Tensor,
        input_length: int,
        position_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Positions/bands matching v1's selected tokens; padded slots use -1."""
        B = coarse.shape[0]
        dev = coarse.device
        levels = len(details)
        bands: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        nc = 0
        if self.use_coarse:
            nc = coarse.shape[-1]
            block = 2 ** levels
            cpos = (
                torch.arange(nc, device=dev, dtype=position_dtype) * block + block / 2
            ) / input_length
            positions.append(cpos.expand(B, nc))
            bands.append(torch.zeros(B, nc, dtype=torch.long, device=dev))

        n_detail = max(0, self.num_tokens - nc)
        if n_detail > 0:
            all_bands, all_positions, all_scores = [], [], []
            for li, detail in enumerate(details, start=1):
                nd = detail.shape[-1]
                sigma = 1.4826 * detail.abs().median(dim=-1, keepdim=True).values + 1e-8
                dblock = 2 ** li
                pos = (
                    torch.arange(nd, device=dev, dtype=position_dtype) * dblock
                    + dblock / 2
                ) / input_length
                all_bands.append(torch.full((B, nd), li, dtype=torch.long, device=dev))
                all_positions.append(pos.expand(B, nd))
                all_scores.append(detail.abs() / sigma)
            cat_bands = torch.cat(all_bands, dim=-1)
            cat_positions = torch.cat(all_positions, dim=-1)
            cat_scores = torch.cat(all_scores, dim=-1)
            k = min(n_detail, cat_scores.shape[-1])
            top = cat_scores.topk(k, dim=-1).indices.sort(dim=-1).values
            bands.append(torch.gather(cat_bands, -1, top))
            positions.append(torch.gather(cat_positions, -1, top))

        band = torch.cat(bands, dim=-1)[..., : self.num_tokens]
        position = torch.cat(positions, dim=-1)[..., : self.num_tokens]
        n_valid = position.shape[-1]
        valid = torch.zeros(B, self.num_tokens, dtype=torch.bool, device=dev)
        valid[:, :n_valid] = True
        if n_valid < self.num_tokens:
            # v1 padding has no source coefficient.  Keep an explicit sentinel
            # band and a deterministic position solely for locality queries.
            pad_n = self.num_tokens - n_valid
            pad_pos = (
                torch.arange(n_valid, self.num_tokens, device=dev, dtype=position_dtype) + 0.5
            ) / self.num_tokens
            position = torch.cat([position, pad_pos.expand(B, pad_n)], dim=-1)
            band = torch.cat([
                band,
                torch.full((B, pad_n), -1, dtype=torch.long, device=dev),
            ], dim=-1)
        return position, band, valid

    def _iteration(
        self,
        z: torch.Tensor,
        base_tokens: torch.Tensor,
        base_positions: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
        stats: torch.Tensor,
        band: int,
        step: int,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Build one level's tokens and apply one recurrent iteration.

        Token construction stays inside this function so activation
        checkpointing recomputes both the value MLP and the attention block;
        otherwise all <=11 level embeddings remain live until backward.
        """
        e = self._tokens(values, positions, band)
        query = base_tokens + self.step_embed.weight[step].to(z.dtype)
        gate = None
        if self.gate_head is not None:
            g_in = torch.cat([query.mean(dim=1), e.mean(dim=1), stats], dim=-1)
            gate = torch.sigmoid(self.gate_head(g_in)).unsqueeze(-1)
        # Every query is centered on the real coefficient position carried by
        # its v1 base token, not on the ordinal slot i/num_tokens.
        qpos = base_positions.float().unsqueeze(-1)
        kpos = positions.float().unsqueeze(1)
        position_bias = -torch.log1p(((qpos - kpos) / self.locality_sigma).square())
        position_bias = position_bias.to(e.dtype)
        z_new, cross, self_attn, mlp = self._block(step)(
            z, query, e, gate, position_bias, self.reader_mode
        )
        gate_out = gate if gate is not None else z_new.new_empty((0,))
        return z_new, gate_out, cross, self_attn, mlp

    def _num_levels(self, input_length: int) -> int:
        return max(
            1,
            min(
                self.max_levels,
                math.ceil(math.log2(max(2, input_length / self.coarse_blocks))),
            ),
        )

    # ------------------------------------------------------------------
    def _encode_channel(
        self,
        x: torch.Tensor,
        depth_override: int | None = None,
    ) -> torch.Tensor:
        """x: (B, L) -> (B, num_tokens, output_dim)."""
        B, L = x.shape
        if B < 1 or L < 1:
            raise ValueError(f"expected non-empty (B, L), got {tuple(x.shape)}")
        if not torch.is_floating_point(x):
            raise TypeError(f"expected floating-point input, got {x.dtype}")
        dev = x.device
        levels = self._num_levels(L)
        n_steps = levels + 1
        depth = (
            self._requested_depth(n_steps)
            if depth_override is None
            else depth_override
        )
        if not 1 <= depth <= n_steps:
            raise ValueError(
                f"depth_override must be in [1, {n_steps}], got {depth}"
            )

        if not bool(torch.isfinite(x).all()):
            raise ValueError("input contains NaN/inf")
        base_tokens, details, coarse = self._v1_tokens_and_pyramid(x)
        pyramid_finite = torch.stack(
            [torch.isfinite(coarse).all()]
            + [torch.isfinite(detail).all() for detail in details]
        ).all()
        if not bool(pyramid_finite):
            raise ValueError("wavelet pyramid produced NaN/inf")

        base_positions, base_bands, base_valid = self._base_metadata(
            details, coarse, L, x.dtype
        )
        self.last_base_positions = base_positions.detach()
        self.last_base_bands = base_bands.detach()
        self.last_base_valid = base_valid.detach()

        # With a zero scale and gradients disabled, evaluating the loop cannot
        # affect the result.  This makes the mandatory 16x131072 equality gate
        # cost the same as v1 while training still builds the residual graph.
        context_is_off = os.environ.get("TSLM_V3_CONTEXT_OFF") == "1"
        if (
            not torch.is_grad_enabled()
            and (self.context_scale.detach().item() == 0.0 or context_is_off)
        ):
            self.last_depth, self.last_steps = depth, n_steps
            self.last_gates, self.last_active = [], []
            self.last_context_stats = {}
            return base_tokens

        # ---- per-level token sets, coarse-to-fine ----------------------------
        # Keep only scalars/selected coefficients here.  The large (B,m,D)
        # token embeddings are built inside each loop iteration so encoder
        # activation checkpointing can discard and recompute them.
        # entry: (values, positions, stats, band, has_signal)
        pdt = x.dtype if self.pos_dtype == "input" else torch.float32
        schedule = []
        nc = coarse.shape[-1]
        block = 2 ** levels
        cpos = ((torch.arange(nc, device=dev, dtype=pdt) * block + block / 2) / L).expand(B, nc)
        cstats = torch.stack([torch.ones(B, device=dev), torch.ones(B, device=dev),
                              torch.zeros(B, device=dev),
                              torch.zeros(B, device=dev)], dim=-1).to(x.dtype)
        schedule.append((coarse, cpos, cstats, 0,
                         torch.ones(B, dtype=torch.bool, device=dev)))
        for li in range(levels, 0, -1):  # D_L (coarsest detail) ... D_1 (finest)
            if len(schedule) >= depth:
                break
            d = details[li - 1]
            nd = d.shape[-1]
            sigma = 1.4826 * d.abs().median(dim=-1, keepdim=True).values + 1e-8
            score = d.abs() / sigma
            if not bool(torch.isfinite(score).all()):
                raise ValueError(
                    f"detail level {li} produced non-finite MAD scores"
                )
            dblock = 2 ** li
            pos = ((torch.arange(nd, device=dev, dtype=pdt) * dblock + dblock / 2) / L).expand(B, nd)
            m = min(self.level_topm, nd)
            idx = score.topk(m, dim=-1).indices.sort(dim=-1).values
            v = torch.gather(d, -1, idx)
            p = torch.gather(pos, -1, idx)
            s = torch.gather(score, -1, idx)
            smax = s.max(dim=-1).values.float()
            stats = torch.stack([
                (score > 2.0).float().mean(dim=-1),
                (score > 3.0).float().mean(dim=-1),
                torch.log1p(smax),
                torch.full((B,), li / self.max_levels, device=dev),
            ], dim=-1).to(x.dtype)
            # noise floor of the level: universal threshold sigma*sqrt(2 ln n)
            # (Donoho-Johnstone); pure noise stays below it, an anomaly does not.
            thr = self.noise_tau * math.sqrt(2.0 * math.log(max(2, nd)))
            schedule.append((v, p, stats, li, smax > thr))

        # ---- the loop --------------------------------------------------------
        z = torch.zeros(
            B, self.num_tokens, self.output_dim, dtype=x.dtype, device=dev
        )
        gate_means: list[torch.Tensor] = []
        active_means: list[torch.Tensor] = []
        iteration_stats: list[dict[str, float]] = []
        next_call = self._forward_calls + 1
        try:
            trainer_ctx_every = int(os.environ.get("TSLM_V3_CTX_LOG_EVERY", "0"))
        except ValueError as exc:
            raise ValueError("TSLM_V3_CTX_LOG_EVERY must be an integer >= 0") from exc
        if trainer_ctx_every < 0:
            raise ValueError("TSLM_V3_CTX_LOG_EVERY must be >= 0")
        collect_stats = (
            not torch.is_grad_enabled()
            or (self.gate_log_every and next_call % self.gate_log_every == 0)
            or (trainer_ctx_every and next_call % trainer_ctx_every == 0)
        )
        use_checkpoint = (
            self.training
            and torch.is_grad_enabled()
            and os.environ.get("TSLM_GRAD_CKPT") == "1"
        )
        for step in range(depth):
            values, positions, stats, band, has_signal = schedule[step]
            if use_checkpoint:
                # Bind loop indices as defaults: checkpoint invokes this
                # function again during backward, after the loop has advanced.
                def run_step(z_, v_, p_, s_, _band=band, _step=step):
                    return self._iteration(
                        z_, base_tokens, base_positions, v_, p_, s_, _band, _step
                    )

                z_new, gate_out, cross, self_attn, mlp = checkpoint(
                    run_step, z, values, positions, stats,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                z_new, gate_out, cross, self_attn, mlp = self._iteration(
                    z, base_tokens, base_positions, values, positions, stats, band, step
                )
            if self.gate_head is not None:
                gate_means.append(gate_out.detach().float().mean())
            if self.depth_mode == "noise_gate" and band > 0:
                # series whose level is below the noise floor do not read it
                keep = has_signal.view(B, 1, 1)
                z = torch.where(keep, z_new, z)
                cross = torch.where(keep, cross, torch.zeros_like(cross))
                self_attn = torch.where(keep, self_attn, torch.zeros_like(self_attn))
                mlp = torch.where(keep, mlp, torch.zeros_like(mlp))
                active_means.append(keep.float().mean())
            else:
                z = z_new
                active_means.append(torch.ones((), device=dev))
            if collect_stats:
                snapshot = self._context_snapshot(
                    self._context_for_mode(z, base_bands, base_valid),
                    cross, self_attn, mlp,
                )
                snapshot.update({"step": step, "band": band})
                iteration_stats.append(snapshot)

        self.last_depth, self.last_steps = depth, n_steps
        self.last_gates = (
            torch.stack(gate_means).cpu().tolist() if gate_means else []
        )
        self.last_active = torch.stack(active_means).cpu().tolist()
        self._forward_calls += 1
        z = self._context_for_mode(z, base_bands, base_valid)
        context = self.context_norm(z)
        output, token_gate = self._merge_context(base_tokens, context)
        if collect_stats:
            final_stats = self._context_snapshot(z)
            gate_sample = token_gate.detach().float()[base_valid].flatten()
            final_stats["gate_mean"] = float(gate_sample.mean().cpu())
            final_stats["gate_min"] = float(gate_sample.min().cpu())
            final_stats["gate_max"] = float(gate_sample.max().cpu())
            final_stats["gate_p10"] = float(torch.quantile(gate_sample, 0.1).cpu())
            final_stats["gate_p50"] = float(torch.quantile(gate_sample, 0.5).cpu())
            final_stats["gate_p90"] = float(torch.quantile(gate_sample, 0.9).cpu())
            final_stats["gate_saturation_low"] = float((gate_sample < 0.05).float().mean().cpu())
            final_stats["gate_saturation_high"] = float((gate_sample > 1.95).float().mean().cpu())
            self.last_context_stats = {
                "reader_mode": self.reader_mode,
                "context_mode": self.context_mode,
                "context_grouping": "exact_base_band",
                "iterations": iteration_stats,
                "final": final_stats,
            }
        if (
            self.gate_log_every
            and self._forward_calls % self.gate_log_every == 0
        ):
            gates_text = ",".join(f"{value:.4f}" for value in self.last_gates)
            active_text = ",".join(f"{value:.4f}" for value in self.last_active)
            final = self.last_context_stats.get("final", {})
            def logged(key: str) -> float:
                value = final.get(key)
                return float(value) if isinstance(value, (int, float)) else float("nan")
            print(
                f"[V3Ctx] call={self._forward_calls} depth={depth}/{n_steps} "
                f"reader={self.reader_mode} context_mode={self.context_mode} "
                f"z_rms={logged('z_rms'):.6g} "
                f"h_rms={logged('h_rms'):.6g} "
                f"token_variable_fraction_total={logged('token_variable_fraction_total'):.6g} "
                f"mean_token_cosine_total={logged('mean_token_cosine_total'):.6g} "
                f"input_series_relative_total={logged('input_series_relative_total'):.6g} "
                f"token_variable_fraction_coarse={logged('token_variable_fraction_coarse'):.6g} "
                f"mean_token_cosine_coarse={logged('mean_token_cosine_coarse'):.6g} "
                f"input_series_relative_coarse={logged('input_series_relative_coarse'):.6g} "
                f"token_variable_fraction_detail={logged('token_variable_fraction_detail'):.6g} "
                f"mean_token_cosine_detail={logged('mean_token_cosine_detail'):.6g} "
                f"input_series_relative_detail={logged('input_series_relative_detail'):.6g} "
                f"token_gate_mean={logged('gate_mean'):.6g} "
                f"gates=[{gates_text}] active=[{active_text}]",
                flush=True,
            )
        return output

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        legacy = [
            prefix + name
            for name in ("latents", "ln_out.weight", "content_scale")
            if prefix + name in state_dict
        ]
        if legacy:
            error_msgs.append(
                "legacy latent-only v3 checkpoint is incompatible with residual-on-v1 v3; "
                f"found {legacy}"
            )
        # Only an actual residual-on-v1 checkpoint has context_scale.  The v1
        # bootstrap in WaveletV3TSModel intentionally lacks it and must retain
        # constructor-configured scale initials instead of being "migrated".
        if prefix + "context_scale" in state_dict:
            for block_index, block in enumerate(self.blocks):
                for name in ("self_attn_scale", "mlp_scale"):
                    key = f"{prefix}blocks.{block_index}.{name}"
                    if key not in state_dict:
                        state_dict[key] = torch.ones_like(getattr(block, name))
            if self.context_token_gate is not None:
                for name, value in self.context_token_gate.state_dict().items():
                    key = f"{prefix}context_token_gate.{name}"
                    if key not in state_dict:
                        state_dict[key] = value.detach().clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multi_variable:
            if x.dim() != 3:
                raise ValueError(f"expected (B, L, V), got {tuple(x.shape)}")
            # Resolve stochastic depth once per public forward call so all
            # variables in the same example see the same scale horizon.
            n_steps = self._num_levels(x.shape[1]) + 1
            depth = self._requested_depth(n_steps)
            chans = [
                self._encode_channel(x[..., v], depth_override=depth)
                for v in range(x.shape[-1])
            ]
            return torch.stack(chans, dim=1)
        if x.dim() != 2:
            raise ValueError(f"expected (B, L), got {tuple(x.shape)}")
        return self._encode_channel(x)


@torch.no_grad()
def context_input_dependence(
    encoder: RecurrentWaveletTSEncoder,
    x: torch.Tensor,
    different: torch.Tensor,
) -> dict[str, float]:
    """Measure context-only response to a different and a translated input.

    This explicit probe is intentionally outside the normal forward: training
    never pays for two extra encodes.  Results are also attached to
    ``last_context_stats['final']`` for checkpoint audits.
    """
    if x.shape != different.shape:
        raise ValueError("x and different must have the same shape")
    captured: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        captured.append(output.detach().float())

    was_training = encoder.training
    handle = encoder.context_norm.register_forward_hook(capture)
    time_dim = -2 if encoder.multi_variable else -1
    try:
        encoder.eval()
        captured.clear()
        encoder(x)
        primary_contexts = list(captured)
        primary = primary_contexts[-1]
        primary_stats = encoder.last_context_stats
        captured.clear()
        encoder(different)
        other_contexts = list(captured)
        other = other_contexts[-1]
        captured.clear()
        shift = max(1, x.shape[time_dim] // 3)
        encoder(torch.roll(x, shifts=shift, dims=time_dim))
        translated_contexts = list(captured)
        translated = translated_contexts[-1]
    finally:
        handle.remove()
        encoder.train(was_training)
    denominator = encoder._rms(primary).clamp_min(torch.finfo(primary.dtype).eps)
    result = {
        "different_series_relative": float(
            (encoder._rms(primary - other) / denominator).cpu()
        ),
        "translated_input_relative": float(
            (encoder._rms(primary - translated) / denominator).cpu()
        ),
    }
    encoder.last_context_stats = primary_stats
    if primary_stats:
        primary_stats.setdefault("final", {}).update(result)
        iterations = primary_stats.get("iterations", [])
        if not (
            len(primary_contexts) == len(other_contexts) == len(translated_contexts)
        ):
            raise RuntimeError("context diagnostic captures changed across paired inputs")
        for index, row in enumerate(iterations):
            source = primary_contexts[index]
            source_rms = encoder._rms(source).clamp_min(
                torch.finfo(source.dtype).eps
            )
            row["different_series_relative"] = float(
                (encoder._rms(source - other_contexts[index]) / source_rms).cpu()
            )
            row["translated_input_relative"] = float(
                (encoder._rms(source - translated_contexts[index]) / source_rms).cpu()
            )
    return result


@torch.no_grad()
def token_diversity(
    encoder: RecurrentWaveletTSEncoder,
    x: torch.Tensor,
    comparison: torch.Tensor | None = None,
) -> dict[str, float]:
    """Measure slot diversity and input dependence without changing mode.

    ``comparison`` may be a genuinely different series or the same series
    with an anomaly moved elsewhere.  By default the time axis is rolled by
    one third, which makes the function directly useful in deployment smoke
    tests while preserving the input shape and marginal distribution.
    """
    if not isinstance(encoder, RecurrentWaveletTSEncoder):
        raise TypeError("encoder must be a RecurrentWaveletTSEncoder")
    expected_dim = 3 if encoder.multi_variable else 2
    if x.dim() != expected_dim:
        expected = "(B, L, V)" if encoder.multi_variable else "(B, L)"
        raise ValueError(f"expected {expected}, got {tuple(x.shape)}")
    time_dim = -2 if encoder.multi_variable else -1
    if comparison is None:
        shift = max(1, x.shape[time_dim] // 3)
        comparison = torch.roll(x, shifts=shift, dims=time_dim)
    if comparison.shape != x.shape:
        raise ValueError(
            f"comparison shape {tuple(comparison.shape)} != input shape {tuple(x.shape)}"
        )

    was_training = encoder.training
    try:
        encoder.eval()
        output = encoder(x).float()
        other = encoder(comparison).float()
    finally:
        encoder.train(was_training)

    # Token axis is penultimate for both public output contracts.  Flatten
    # batch and optional variable axes so each series/channel contributes one
    # cosine matrix, rather than mixing tokens belonging to different inputs.
    flat = output.reshape(-1, output.shape[-2], output.shape[-1])
    if flat.shape[1] < 2:
        raise ValueError("token diversity requires num_tokens >= 2")
    unit = F.normalize(flat, dim=-1, eps=1e-12)
    cosine = unit @ unit.transpose(-1, -2)
    off_diagonal = ~torch.eye(
        flat.shape[1], dtype=torch.bool, device=flat.device
    )
    mean_cosine = cosine[:, off_diagonal].mean()
    between_token_std = flat.std(dim=1, unbiased=False).mean()
    total_std = flat.std(unbiased=False)
    eps = torch.finfo(output.dtype).eps
    output_norm = torch.linalg.vector_norm(output).clamp_min(eps)
    relative_difference = torch.linalg.vector_norm(output - other) / output_norm
    return {
        "mean_token_cosine": float(mean_cosine.cpu()),
        "between_token_std": float(between_token_std.cpu()),
        "total_std": float(total_std.cpu()),
        "between_total_std_ratio": float(
            (between_token_std / total_std.clamp_min(eps)).cpu()
        ),
        "input_relative_difference": float(relative_difference.cpu()),
    }


@torch.no_grad()
def position_decodability(
    encoder: nn.Module,
    n: int = 288,
    seed: int = 20260907,
    ridge: float = 3.0,
    batch_pairs: int = 24,
) -> dict[str, float | dict[str, float]]:
    """CPU ridge probe for anomaly position, presence, and magnitude.

    ``n`` paired noise/anomaly cases are spread over lengths 4096, 8192 and
    32768 and over spike, level-shift and variance-change anomalies.  The
    probe sees each encoded series independently; pairing is used only to keep
    the positive/negative nuisance distribution identical.  Its features are
    the maximum over slots for each channel plus the RMS of each slot.  The
    former lets v1 expose a selected large coefficient; the latter lets a
    fixed-slot reader expose where its content energy changed.  This is a
    small, fixed readout rather than a trainable feature normaliser: absolute
    content scale therefore remains part of what "decodable" means.

    IDs divisible by four form the held-out split.  Ridge is solved in the
    dual, so the function remains practical when encoder output_dim is 512.
    The encoder's training mode is restored on return.
    """
    if n < 36 or n % 36:
        raise ValueError("n must be a multiple of 36 and >= 36")
    if not math.isfinite(float(ridge)) or ridge <= 0:
        raise ValueError("ridge must be finite and > 0")
    if int(batch_pairs) < 1:
        raise ValueError("batch_pairs must be >= 1")
    try:
        parameter = next(encoder.parameters())
    except StopIteration as exc:
        raise ValueError("encoder must have parameters") from exc
    if parameter.device.type != "cpu":
        raise ValueError("position_decodability is an offline CPU probe")
    if getattr(encoder, "multi_variable", False):
        raise ValueError("position_decodability expects a single-variable encoder")

    lengths = (4096, 8192, 32768)
    kinds = ("spike", "level_shift", "variance_change")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    groups: dict[int, list[tuple[torch.Tensor, torch.Tensor, int, float, float, str]]] = {}
    for sample_id in range(n):
        length = lengths[sample_id % len(lengths)]
        kind = kinds[(sample_id // len(lengths)) % len(kinds)]
        position = 0.05 + 0.90 * torch.rand((), generator=generator).item()
        magnitude = 0.25 + 0.75 * torch.rand((), generator=generator).item()
        clean = torch.randn(length, generator=generator)
        anomalous = clean.clone()
        start = min(length - 2, max(1, int(position * length)))
        if kind == "spike":
            anomalous[start] += 30.0 + 60.0 * magnitude
        elif kind == "level_shift":
            anomalous[start:] += 4.0 + 9.0 * magnitude
        else:
            width = max(32, int(0.08 * length))
            start = min(start, length - width)
            position = start / length
            anomalous[start:start + width] *= 3.0 + 8.0 * magnitude
        groups.setdefault(length, []).append(
            (clean, anomalous, sample_id, position, magnitude, kind)
        )

    feature_batches = []
    labels: list[tuple[int, int, float, float, str]] = []
    was_training = encoder.training
    try:
        encoder.eval()
        for rows in groups.values():
            for offset in range(0, len(rows), int(batch_pairs)):
                chunk = rows[offset:offset + int(batch_pairs)]
                batch = torch.stack([
                    series
                    for clean, anomalous, *_ in chunk
                    for series in (clean, anomalous)
                ]).to(dtype=parameter.dtype)
                tokens = encoder(batch).float()
                if tokens.dim() != 3 or tokens.shape[1] != 256:
                    raise ValueError(
                        "position_decodability requires output shaped (B, 256, D), "
                        f"got {tuple(tokens.shape)}"
                    )
                features = torch.cat([
                    tokens.amax(dim=1),
                    tokens.square().mean(dim=-1).sqrt(),
                ], dim=-1)
                # Keep the ridge strength comparable for compact test encoders
                # and D=512 without fitting a data-dependent scaler.
                features = features * math.sqrt(512.0 / features.shape[-1])
                feature_batches.append(features)
                labels.extend([
                    (sample_id, present, position, magnitude, kind)
                    for _, _, sample_id, position, magnitude, kind in chunk
                    for present in (0, 1)
                ])
    finally:
        encoder.train(was_training)

    raw_features = torch.cat(feature_batches, dim=0)
    order = sorted(range(len(labels)), key=lambda i: (labels[i][0], labels[i][1]))
    features = raw_features[order]
    labels = [labels[i] for i in order]
    train_ids = [i for i, label in enumerate(labels) if label[0] % 4]
    test_ids = [i for i, label in enumerate(labels) if not label[0] % 4]

    def fit_predict(target_index: int, positives_only: bool) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        train = [
            i for i in train_ids if not positives_only or labels[i][1] == 1
        ]
        test = [
            i for i in test_ids if not positives_only or labels[i][1] == 1
        ]
        x_train = features[train]
        x_test = features[test]
        x_mean = x_train.mean(dim=0)
        x_train = x_train - x_mean
        x_test = x_test - x_mean
        y_train = torch.tensor(
            [labels[i][target_index] for i in train], dtype=torch.float32
        )
        y_test = torch.tensor(
            [labels[i][target_index] for i in test], dtype=torch.float32
        )
        y_mean = y_train.mean()
        gram = x_train @ x_train.T
        dual = torch.linalg.solve(
            gram + float(ridge) * torch.eye(gram.shape[0]),
            y_train - y_mean,
        )
        prediction = x_test @ x_train.T @ dual + y_mean
        return prediction, y_test, test

    pos_pred, pos_true, positive_test = fit_predict(2, positives_only=True)
    mag_pred, mag_true, _ = fit_predict(3, positives_only=True)
    presence_pred, presence_true, _ = fit_predict(1, positives_only=False)
    by_kind = {}
    for kind in kinds:
        mask = torch.tensor([labels[i][4] == kind for i in positive_test])
        by_kind[kind] = float((pos_pred[mask] - pos_true[mask]).abs().mean())
    return {
        "position_mae": float((pos_pred - pos_true).abs().mean()),
        "position_mae_by_type": by_kind,
        "presence_accuracy": float(
            ((presence_pred >= 0.5) == presence_true.bool()).float().mean()
        ),
        "magnitude_mae": float((mag_pred - mag_true).abs().mean()),
        "n_pairs": float(n),
        "ridge": float(ridge),
    }
