"""v3 — recurrent (looped) wavelet perceiver front-end for TSLM variants.

Same external contract as WaveletTSEncoder (v1): forward (B, L) -> (B, N, D),
or (B, L, V) -> (B, V, N, D) with multi_variable=True; attributes output_dim,
multi_variable, freeze, patch_size, get_num_output_patches.  N = num_tokens is
fixed (256): Qwen's sequence is IDENTICAL to v1, whatever the depth.

What changes vs v1 is only how wavelet coefficients become the 256 tokens:

  v1  hard selection — the 256 slots are the coarse band plus the global top-k
      detail coefficients (MAD-normalised magnitude), placed directly in Qwen's
      sequence.
  v3  recurrent reading — 256 learned latents read the pyramid ONE LEVEL PER
      ITERATION, coarse-to-fine (A_L, D_L, ..., D_1), through a block that is
      shared across iterations (tied weights = one "looped layer").  Each level
      exposes at most `level_topm` coefficients (per-level top-m by the SAME
      MAD score as v1) so the finest levels never flood the reader; a per-level
      gate lets the loop down-weight a level it finds uninformative.  Only the
      latents reach the projector/Qwen: the detail coefficients are consulted,
      never inserted.

Depth ("when to stop", see README):
  depth_mode = "full"        every level, coarse -> D_1 (default);
               "sdepth"      training: random truncation depth per forward
                             (Huginn-style), so the model is usable at any
                             depth and metric-vs-depth curves are free at eval;
               "noise_gate"  a detail level whose largest MAD-normalised
                             coefficient is below the universal threshold
                             noise_tau * sqrt(2 ln n) (Donoho-Johnstone: the
                             expected maximum of n pure-noise coefficients) is
                             skipped for that series: the latents are not
                             updated by that level.  noise_tau = 1.0 default.
  TSLM_V3_DEPTH=r (env)      inference override: run only the first r
                             iterations (r >= 1) — the eval sweep for the curve.

Token featurisation (value clamp, Fourier positions, band embedding) is copied
from v1; only the conditioning embedding's initial scale is reduced so it does
not hide value/position differences inside a level.
Each iteration also has an identity local-read highway, so value/position
features do not depend on random attention projections to reach the output.
At the boundary, content is normalised before adding the positional anchor;
learned content/context scales replace the old hand-tuned output mixture.
`pos_dtype="input"` keeps v1's dtype behaviour for positions; "fp32" is the
v2 fix and is NOT enabled by default (it would confound the comparison).

Diagnostics for logs/tests: forward sets last_depth, last_steps, last_gates
(mean gate per iteration), and last_active (fraction of series that actually
read each iteration); token_diversity measures slot diversity/input dependence.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.models.wavelet_encoder import haar_pyramid  # v1 file, untouched


class _LoopBlock(nn.Module):
    """One iteration of the loop: gated cross-attention read of the current
    level, self-attention among latents, MLP.  Pre-LN residual blocks."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int, dropout: float):
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

    def forward(
        self,
        z: torch.Tensor,
        query: torch.Tensor,
        e: torch.Tensor,
        gate: torch.Tensor | None,
        position_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ``query`` carries the positional/step anchors, whereas ``z`` is the
        # content residual.  Keeping them separate prevents a fixed learned
        # query from dominating the norm of every emitted token.
        q, kv = self.ln_q(query), self.ln_kv(e)
        # MultiheadAttention accepts a per-example additive mask only in
        # (B * heads, target, source) form.
        attn_mask = position_bias.repeat_interleave(self.n_heads, dim=0)
        read, _ = self.cross(
            q, kv, kv, attn_mask=attn_mask, need_weights=False
        )
        # Identity content highway at initialisation: the positional prior
        # already defines a meaningful local interpolation, while a freshly
        # initialised V/out projection is an unnecessary random bottleneck.
        # Cross-attention remains a learned correction and can move/reshape
        # the read; the highway guarantees that local coefficient features
        # reach the recurrent state at their native scale from step zero.
        local_weights = torch.softmax(position_bias.float(), dim=-1).to(e.dtype)
        local_read = torch.bmm(local_weights, e)
        read = read + local_read
        if gate is not None:
            read = read * gate
            local_read = local_read * gate
        z = z + read
        s = self.ln_s(z)
        mix, _ = self.self_attn(s, s, s, need_weights=False)
        z = z + mix
        return z + self.mlp(self.ln_m(z)), local_read


class RecurrentWaveletTSEncoder(nn.Module):
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
        **_,
    ):
        super().__init__()
        if depth_mode not in ("full", "sdepth", "noise_gate"):
            raise ValueError(f"depth_mode must be full|sdepth|noise_gate, got {depth_mode!r}")
        if pos_dtype not in ("input", "fp32"):
            raise ValueError(f"pos_dtype must be input|fp32, got {pos_dtype!r}")
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
        self.output_dim = int(output_dim)
        self.num_tokens = int(num_tokens)
        self.max_levels = int(max_levels)
        self.coarse_blocks = int(coarse_blocks)
        self.n_pos_freqs = int(n_pos_freqs)
        self.multi_variable = bool(multi_variable)
        self.freeze = bool(freeze)
        self.patch_size = 1
        self.level_topm = int(level_topm)
        self.tied = bool(tied)
        self.level_gate = bool(level_gate)
        self.depth_mode = depth_mode
        self.sdepth_min = max(1, int(sdepth_min))
        self.noise_tau = float(noise_tau)
        self.pos_dtype = pos_dtype
        D = self.output_dim
        n_steps = self.max_levels + 1  # coarse + up to max_levels detail levels

        # --- token featurisation: identical to v1 ---------------------------
        in_dim = 1 + 2 * self.n_pos_freqs
        self.value_proj = nn.Sequential(nn.Linear(in_dim, D), nn.GELU(), nn.Linear(D, D))
        self.band_embed = nn.Embedding(n_steps, D)  # 0 = coarse, li = detail level li
        # A band label is conditioning, not the token content itself.  Keep it
        # at the same small scale as the step label so position/value features
        # remain visible to cross-attention within a level.
        nn.init.normal_(self.band_embed.weight, mean=0.0, std=0.02)
        freqs = 2.0 ** torch.arange(self.n_pos_freqs)
        self.register_buffer("pos_freqs", freqs * math.pi, persistent=False)
        latent_positions = (
            torch.arange(self.num_tokens, dtype=torch.float32) + 0.5
        ) / self.num_tokens
        self.register_buffer(
            "latent_positions", latent_positions, persistent=False
        )

        # --- the loop -------------------------------------------------------
        # Give every output slot a positional identity before the first read.
        # The old N(0, .02) latents were swamped by nn.Embedding's default
        # O(1) step vector in ``z + step_embed[step]``; after ln_q all queries
        # were consequently almost equal.  A centred Fourier basis, projected
        # isometrically to D and RMS-normalised, keeps the slots separated at
        # O(1) scale while retaining exactly the position frequencies used by
        # the level-token featurisation.
        latent_ang = self.latent_positions.unsqueeze(-1) * self.pos_freqs.float()
        latent_features = torch.cat(
            [torch.sin(latent_ang), torch.cos(latent_ang)], dim=-1
        )
        latent_features = latent_features - latent_features.mean(dim=0, keepdim=True)
        latent_projection = torch.empty(latent_features.shape[-1], D)
        nn.init.orthogonal_(latent_projection)
        latent_init = latent_features @ latent_projection
        latent_init = latent_init / latent_init.square().mean(
            dim=-1, keepdim=True
        ).sqrt().clamp_min(1e-6)
        self.latents = nn.Parameter(latent_init)
        self.step_embed = nn.Embedding(n_steps, D)  # tells the shared block which iteration it is
        nn.init.normal_(self.step_embed.weight, mean=0.0, std=0.02)
        blocks = 1 if self.tied else n_steps
        self.blocks = nn.ModuleList(
            [_LoopBlock(D, n_heads, mlp_ratio, dropout) for _ in range(blocks)]
        )
        # gate in [0,1] per (series, iteration) from [mean latents, mean level tokens, level stats]
        self.gate_head = nn.Linear(2 * D + 4, 1) if self.level_gate else None
        if self.gate_head is not None:
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, 2.0)  # sigmoid(2) ~ 0.88: starts almost open
        self.ln_out = nn.LayerNorm(D)
        # Content and position leave the encoder through two explicit paths.
        # Normalising content before adding the anchor keeps both at O(1)
        # without a hand-tuned mixture coefficient; the learned scalar lets
        # training change their ratio from the neutral 1:1 initialisation.
        self.content_scale = nn.Parameter(torch.ones(()))
        # Depth-averaged residual initialisation: the deterministic local read
        # is the base representation, while the sum of randomly initialised
        # recurrent corrections starts at its per-step mean scale.
        self.context_scale = nn.Parameter(
            torch.tensor(1.0 / n_steps)
        )

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

    # ------------------------------------------------------------------
    def get_num_output_patches(self, input_length: int) -> int:
        return self.num_tokens

    def _block(self, step: int) -> _LoopBlock:
        return self.blocks[0] if self.tied else self.blocks[step]

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

    def _iteration(
        self,
        z: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
        stats: torch.Tensor,
        band: int,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build one level's tokens and apply one recurrent iteration.

        Token construction stays inside this function so activation
        checkpointing recomputes both the value MLP and the attention block;
        otherwise all <=11 level embeddings remain live until backward.
        """
        e = self._tokens(values, positions, band)
        query = (
            z
            + self.latents.to(z.dtype).unsqueeze(0)
            + self.step_embed.weight[step].to(z.dtype)
        )
        gate = None
        if self.gate_head is not None:
            g_in = torch.cat([query.mean(dim=1), e.mean(dim=1), stats], dim=-1)
            gate = torch.sigmoid(self.gate_head(g_in)).unsqueeze(-1)
        # A narrow heavy-tailed locality prior makes slot i read near
        # (i+.5)/N from the first forward.  Its logarithmic tails keep distant
        # coefficients reachable, so learned content logits can zoom out.
        qpos = self.latent_positions.to(
            device=positions.device, dtype=torch.float32
        ).view(1, self.num_tokens, 1)
        kpos = positions.float().unsqueeze(1)
        locality_sigma = 2.0 / self.num_tokens
        position_bias = -torch.log1p(((qpos - kpos) / locality_sigma).square())
        position_bias = position_bias.to(e.dtype)
        z_new, local_read = self._block(step)(z, query, e, gate, position_bias)
        gate_out = gate if gate is not None else z_new.new_empty((0,))
        return z_new, gate_out, local_read

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

        med = x.median(dim=-1, keepdim=True).values
        mad = (x - med).abs().median(dim=-1, keepdim=True).values
        scale = 1.4826 * mad + 1e-8
        xn = (x - med) / scale
        if not bool(torch.isfinite(xn).all()):
            raise ValueError(
                "input or robust normalization produced NaN/inf; "
                "check the time-series data and input dtype"
            )
        details, coarse = haar_pyramid(xn, levels)
        pyramid_finite = torch.stack(
            [torch.isfinite(coarse).all()]
            + [torch.isfinite(detail).all() for detail in details]
        ).all()
        if not bool(pyramid_finite):
            raise ValueError("Haar pyramid produced NaN/inf")

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
        # The learned Fourier latents condition the queries and form the
        # separate positional output path.  Both the recurrent state and the
        # direct local-content accumulator start empty.
        z = torch.zeros(
            B, self.num_tokens, self.output_dim, dtype=x.dtype, device=dev
        )
        direct = torch.zeros_like(z)
        gate_means: list[torch.Tensor] = []
        active_means: list[torch.Tensor] = []
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
                    return self._iteration(z_, v_, p_, s_, _band, _step)

                z_new, gate_out, local_read = checkpoint(
                    run_step, z, values, positions, stats,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                z_new, gate_out, local_read = self._iteration(
                    z, values, positions, stats, band, step
                )
            if self.gate_head is not None:
                gate_means.append(gate_out.detach().float().mean())
            if self.depth_mode == "noise_gate" and band > 0:
                # series whose level is below the noise floor do not read it
                keep = has_signal.view(B, 1, 1)
                z = torch.where(keep, z_new, z)
                direct = torch.where(keep, direct + local_read, direct)
                active_means.append(keep.float().mean())
            else:
                z = z_new
                direct = direct + local_read
                active_means.append(torch.ones((), device=dev))

        self.last_depth, self.last_steps = depth, n_steps
        self.last_gates = (
            torch.stack(gate_means).cpu().tolist() if gate_means else []
        )
        self.last_active = torch.stack(active_means).cpu().tolist()
        self._forward_calls += 1
        if (
            self.gate_log_every
            and self._forward_calls % self.gate_log_every == 0
        ):
            gates_text = ",".join(f"{value:.4f}" for value in self.last_gates)
            active_text = ",".join(f"{value:.4f}" for value in self.last_active)
            print(
                f"[V3Gates] call={self._forward_calls} depth={depth}/{n_steps} "
                f"gates=[{gates_text}] active=[{active_text}]",
                flush=True,
            )
        # Split the emitted representation into an input-dependent local
        # residual and a fixed positional anchor.  Exact centring removes the
        # broadcast mode introduced by shared biases; self-attention has
        # already placed global context inside each local residual.  Crucially
        # the final normalisation is applied to content *before* the anchor is
        # added, so it cannot shrink the input-dependent path to a few percent.
        context = z - direct
        content = direct + self.context_scale.to(z.dtype) * context
        content = content - content.mean(dim=1, keepdim=True)
        content = self.ln_out(content)
        anchors = self.latents.to(z.dtype)
        anchors = anchors - anchors.mean(dim=0, keepdim=True)
        return self.content_scale.to(z.dtype) * content + anchors.unsqueeze(0)

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
