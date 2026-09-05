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
from v1 on purpose so that ablation 1 (v3 vs v1) changes exactly one thing.
`pos_dtype="input"` keeps v1's dtype behaviour for positions; "fp32" is the
v2 fix and is NOT enabled by default (it would confound the comparison).

Diagnostics for logs/tests (set on every forward): last_depth, last_steps,
last_gates (mean gate per iteration), last_active (fraction of series that
actually read each iteration).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from src.models.wavelet_encoder import haar_pyramid  # v1 file, untouched


class _LoopBlock(nn.Module):
    """One iteration of the loop: gated cross-attention read of the current
    level, self-attention among latents, MLP.  Pre-LN residual blocks."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln_s = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln_m = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim)
        )

    def forward(self, z: torch.Tensor, e: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
        q, kv = self.ln_q(z), self.ln_kv(e)
        read, _ = self.cross(q, kv, kv, need_weights=False)
        if gate is not None:
            read = read * gate
        z = z + read
        s = self.ln_s(z)
        mix, _ = self.self_attn(s, s, s, need_weights=False)
        z = z + mix
        return z + self.mlp(self.ln_m(z))


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
        freqs = 2.0 ** torch.arange(self.n_pos_freqs)
        self.register_buffer("pos_freqs", freqs * math.pi, persistent=False)

        # --- the loop -------------------------------------------------------
        self.latents = nn.Parameter(torch.randn(self.num_tokens, D) * 0.02)
        self.step_embed = nn.Embedding(n_steps, D)  # tells the shared block which iteration it is
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

        # diagnostics
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
        if env:
            return max(1, min(n_steps, int(env)))
        if self.depth_mode == "sdepth" and self.training:
            lo = min(self.sdepth_min, n_steps)
            return int(torch.randint(lo, n_steps + 1, (1,)).item())
        return n_steps

    # ------------------------------------------------------------------
    def _encode_channel(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) -> (B, num_tokens, output_dim)."""
        B, L = x.shape
        dev = x.device
        levels = max(1, min(self.max_levels,
                            math.ceil(math.log2(max(2, L / self.coarse_blocks)))))

        med = x.median(dim=-1, keepdim=True).values
        mad = (x - med).abs().median(dim=-1, keepdim=True).values
        scale = 1.4826 * mad + 1e-8
        xn = (x - med) / scale
        details, coarse = haar_pyramid(xn, levels)

        # ---- per-level token sets, coarse-to-fine ----------------------------
        # each entry: (tokens (B, m, D), stats (B, 4), band, has_signal (B,) bool)
        pdt = x.dtype if self.pos_dtype == "input" else torch.float32
        schedule = []
        nc = coarse.shape[-1]
        block = 2 ** levels
        cpos = ((torch.arange(nc, device=dev, dtype=pdt) * block + block / 2) / L).expand(B, nc)
        cstats = torch.stack([torch.ones(B, device=dev), torch.ones(B, device=dev),
                              torch.zeros(B, device=dev),
                              torch.zeros(B, device=dev)], dim=-1).to(x.dtype)
        schedule.append((self._tokens(coarse, cpos, 0), cstats, 0,
                         torch.ones(B, dtype=torch.bool, device=dev)))
        for li in range(levels, 0, -1):  # D_L (coarsest detail) ... D_1 (finest)
            d = details[li - 1]
            nd = d.shape[-1]
            sigma = 1.4826 * d.abs().median(dim=-1, keepdim=True).values + 1e-8
            score = d.abs() / sigma
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
            schedule.append((self._tokens(v, p, li), stats, li, smax > thr))

        n_steps = len(schedule)
        depth = self._requested_depth(n_steps)

        # ---- the loop --------------------------------------------------------
        z = self.latents.to(x.dtype).unsqueeze(0).expand(B, -1, -1)
        gates, active = [], []
        for step in range(depth):
            e, stats, band, has_signal = schedule[step]
            z_in = z + self.step_embed.weight[step].to(z.dtype)
            gate = None
            if self.gate_head is not None:
                g_in = torch.cat([z_in.mean(dim=1), e.mean(dim=1), stats], dim=-1)
                gate = torch.sigmoid(self.gate_head(g_in)).unsqueeze(-1)  # (B,1,1)
                gates.append(float(gate.detach().float().mean()))
            z_new = self._block(step)(z_in, e, gate)
            if self.depth_mode == "noise_gate" and band > 0:
                # series whose level is below the noise floor do not read it
                keep = has_signal.view(B, 1, 1)
                z = torch.where(keep, z_new, z)
                active.append(float(keep.float().mean()))
            else:
                z = z_new
                active.append(1.0)

        self.last_depth, self.last_steps = depth, n_steps
        self.last_gates, self.last_active = gates, active
        return self.ln_out(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multi_variable:
            if x.dim() != 3:
                raise ValueError(f"expected (B, L, V), got {tuple(x.shape)}")
            chans = [self._encode_channel(x[..., v]) for v in range(x.shape[-1])]
            return torch.stack(chans, dim=1)
        if x.dim() != 2:
            raise ValueError(f"expected (B, L), got {tuple(x.shape)}")
        return self._encode_channel(x)
