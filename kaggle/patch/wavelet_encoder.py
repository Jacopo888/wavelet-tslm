"""Multiresolution wavelet front-end for TSLM variants.

Drop-in alternative to Chronos2TSEncoder (same contract: forward (B,L)->(B,N,D),
or (B,L,V)->(B,V,N,D) with multi_variable=True; attributes output_dim,
multi_variable, freeze, patch_size, get_num_output_patches).

Stage A: fixed Haar analysis (computed in torch, differentiable by construction so
Stage B can unfreeze/replace the filterbank later) + trainable token embedding.
Each channel is decomposed into detail bands D1..DL plus a coarse block-mean band;
the token sequence is [all coarse tokens] + [top-k detail coefficients by
MAD-normalized magnitude], each token embedding (value, scale, position). The
number of output tokens is fixed (`num_tokens`), independent of series length —
adaptive resolution allocation under a constant perception budget, in contrast to
uniform patching.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_SQRT2 = math.sqrt(2.0)

# Daubechies-4 (4 vanishing moments, 8 taps) decomposition low-pass, PyWavelets
# convention; sums to sqrt(2), unit l2 norm. High-pass via QMF below.
_DB4_LO = [
    -0.010597401784997278, 0.032883011666982945, 0.030841381835986965,
    -0.18703481171888114, -0.02798376941698385, 0.6308807679295904,
    0.7148465705525415, 0.23037781330885523,
]


def haar_pyramid(x: torch.Tensor, levels: int):
    """x: (B, L) -> (details list[(B, L/2^l)], coarse_means (B, L/2^levels)).

    Odd lengths at any level are edge-padded. Coarse band is rescaled to plain
    block means so values stay on the input scale.
    """
    details = []
    a = x
    for _ in range(levels):
        if a.shape[-1] % 2 == 1:
            a = torch.cat([a, a[..., -1:]], dim=-1)
        ev, od = a[..., 0::2], a[..., 1::2]
        details.append((ev - od) / _SQRT2)
        a = (ev + od) / _SQRT2
    coarse = a / (_SQRT2 ** levels)
    return details, coarse


def filter_pyramid(x: torch.Tensor, levels: int, lo: torch.Tensor,
                   hi: torch.Tensor):
    """Generic DWT pyramid via stride-2 conv with edge padding (db4 ablation).

    Same output contract as haar_pyramid; per-level lengths ~L/2 (edge padding
    of flen-2 keeps halving exact for even lengths). Coarse rescaled by
    2^{-levels/2} to stay on the input scale, mirroring the Haar path.
    """
    flen = lo.shape[-1]
    pad = flen - 2
    w = torch.stack([lo, hi]).unsqueeze(1)  # (2,1,flen)
    details = []
    a = x
    for _ in range(levels):
        if a.shape[-1] % 2 == 1:
            a = torch.cat([a, a[..., -1:]], dim=-1)
        ap = F.pad(a.unsqueeze(1), (pad // 2, pad - pad // 2), mode="replicate")
        out = F.conv1d(ap, w.to(a.dtype), stride=2)
        a, d = out[:, 0], out[:, 1]
        details.append(d)
    coarse = a / (_SQRT2 ** levels)
    return details, coarse


class WaveletTSEncoder(nn.Module):
    """Perception budget of `num_tokens` tokens per channel: coarse band first,
    remaining slots filled by the largest MAD-normalized detail coefficients."""

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
        wavelet: str = "haar",
        use_coarse: bool = True,
        **_,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.num_tokens = int(num_tokens)
        self.max_levels = int(max_levels)
        self.coarse_blocks = int(coarse_blocks)
        self.n_pos_freqs = int(n_pos_freqs)
        self.multi_variable = bool(multi_variable)
        self.freeze = bool(freeze)
        self.patch_size = 1

        if wavelet not in ("haar", "db4"):
            raise ValueError(f"wavelet must be 'haar' or 'db4', got {wavelet!r}")
        self.wavelet = wavelet
        self.use_coarse = bool(use_coarse)
        if wavelet == "db4":
            lo = torch.tensor(_DB4_LO, dtype=torch.float32)
            # QMF: hi[k] = (-1)^k * lo[flen-1-k]
            sign = torch.tensor([(-1.0) ** k for k in range(lo.shape[-1])])
            hi = sign * lo.flip(-1)
            self.register_buffer("dec_lo", lo, persistent=False)
            self.register_buffer("dec_hi", hi, persistent=False)

        # value scalar + fourier positions -> token, plus a scale (band) embedding.
        in_dim = 1 + 2 * self.n_pos_freqs
        self.value_proj = nn.Sequential(
            nn.Linear(in_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        # bands: 0 = coarse, 1..max_levels = detail levels (1 = finest)
        self.band_embed = nn.Embedding(self.max_levels + 1, self.output_dim)
        self.pad_token = nn.Parameter(torch.zeros(self.output_dim))
        freqs = 2.0 ** torch.arange(self.n_pos_freqs)
        self.register_buffer("pos_freqs", freqs * math.pi, persistent=False)

    def get_num_output_patches(self, input_length: int) -> int:
        return self.num_tokens

    def _encode_channel(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) -> (B, num_tokens, output_dim)."""
        B, L = x.shape
        levels = max(1, min(self.max_levels,
                            math.ceil(math.log2(max(2, L / self.coarse_blocks)))))

        med = x.median(dim=-1, keepdim=True).values
        mad = (x - med).abs().median(dim=-1, keepdim=True).values
        scale = 1.4826 * mad + 1e-8
        xn = (x - med) / scale

        if self.wavelet == "db4":
            details, coarse = filter_pyramid(xn, levels, self.dec_lo, self.dec_hi)
        else:
            details, coarse = haar_pyramid(xn, levels)

        vals, bands, poss = [], [], []
        nc = 0
        if self.use_coarse:
            nc = coarse.shape[-1]
            block = 2 ** levels
            vals.append(coarse)
            bands.append(torch.zeros(B, nc, dtype=torch.long, device=x.device))
            cpos = (torch.arange(nc, device=x.device, dtype=x.dtype) * block
                    + block / 2) / L
            poss.append(cpos.expand(B, nc))

        n_detail = max(0, self.num_tokens - nc)
        if n_detail > 0:
            dv, db, dp, dscore = [], [], [], []
            for li, d in enumerate(details, start=1):
                nd = d.shape[-1]
                sigma = 1.4826 * d.abs().median(dim=-1, keepdim=True).values + 1e-8
                dblock = 2 ** li
                pos = (torch.arange(nd, device=x.device, dtype=x.dtype) * dblock
                       + dblock / 2) / L
                dv.append(d)
                db.append(torch.full((B, nd), li, dtype=torch.long, device=x.device))
                dp.append(pos.expand(B, nd))
                dscore.append(d.abs() / sigma)
            dv = torch.cat(dv, dim=-1)
            db = torch.cat(db, dim=-1)
            dp = torch.cat(dp, dim=-1)
            dscore = torch.cat(dscore, dim=-1)
            k = min(n_detail, dv.shape[-1])
            top = dscore.topk(k, dim=-1).indices
            top = top.sort(dim=-1).values
            vals.append(torch.gather(dv, -1, top))
            bands.append(torch.gather(db, -1, top))
            poss.append(torch.gather(dp, -1, top))

        v = torch.cat(vals, dim=-1)[..., : self.num_tokens]
        b = torch.cat(bands, dim=-1)[..., : self.num_tokens]
        p = torch.cat(poss, dim=-1)[..., : self.num_tokens]

        ang = p.unsqueeze(-1) * self.pos_freqs
        feat = torch.cat([v.unsqueeze(-1).clamp(-20, 20),
                          torch.sin(ang), torch.cos(ang)], dim=-1)
        tok = self.value_proj(feat) + self.band_embed(b)

        n = tok.shape[1]
        if n < self.num_tokens:
            pad = self.pad_token.expand(B, self.num_tokens - n, self.output_dim)
            tok = torch.cat([tok, pad.to(tok.dtype)], dim=1)
        return tok

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multi_variable:
            if x.dim() != 3:
                raise ValueError(f"expected (B, L, V), got {tuple(x.shape)}")
            chans = [self._encode_channel(x[..., v]) for v in range(x.shape[-1])]
            return torch.stack(chans, dim=1)
        if x.dim() != 2:
            raise ValueError(f"expected (B, L), got {tuple(x.shape)}")
        return self._encode_channel(x)
