"""Perceptual mixer with two-axis RoPE, the perceptual memory of views, and the projector (DESIGN §7.1-7.3).

Every mixer head has 64 dimensions: the first 32 are rotated by the memory ordinal ``p`` (acquisition order,
immutable at insertion) and the last 32 by the physical anchor time ``tau = (t - t_origin) / (64 * dt_ref)``
shared by all channels and all crops of an episode.  Each axis has 16 pairs with
``omega_r = 10000^(-2r/32)``; V is never rotated.

Block 1 attends from the current view's descriptors to the perceptual memory ``H0`` (which for the first view
is the view itself, i.e. bidirectional self-attention over all channels); block 2 attends from the updated
queries to the same memory.  Memory never contains a future observation; it is rebuilt every optimizer step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

MIXER_DIM = 512
MIXER_HEADS = 8
MIXER_HEAD_DIM = 64
MIXER_FFN = 1536
AXIS_DIM = 32  # 16 pairs per axis
AXIS_THETA = 10000.0
TIME_UNIT_SAMPLES = 64.0
PROJECTOR_OUT = 2560


def axis_inv_freq(device=None) -> Tensor:
    r = torch.arange(0, AXIS_DIM // 2, dtype=torch.float32, device=device)
    return AXIS_THETA ** (-2.0 * r / AXIS_DIM)


def tau_from_anchor(anchor_time: Tensor, t_origin: float, dt_ref: float) -> Tensor:
    """Physical time unit shared by the whole episode (DESIGN 7.2), fp64 in -> fp32 out."""
    if dt_ref <= 0:
        raise ValueError("dt_ref must be positive")
    return ((anchor_time.to(torch.float64) - float(t_origin)) / (TIME_UNIT_SAMPLES * float(dt_ref))).to(torch.float32)


def _rotate_axis(x: Tensor, pos: Tensor, inv_freq: Tensor) -> Tensor:
    """x [B,H,N,32] rotated pairwise (r, r+16) by pos [B,N] * inv_freq[16]."""
    half = AXIS_DIM // 2
    angle = pos.to(torch.float32)[:, None, :, None] * inv_freq  # [B,1,N,16]
    cos, sin = angle.cos(), angle.sin()
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def apply_two_axis_rope(x: Tensor, p: Tensor, tau: Tensor, inv_freq: Tensor) -> Tensor:
    """Rotate q or k [B,H,N,64]: dims [0:32] by the ordinal p, dims [32:64] by the physical time tau."""
    pos_part = _rotate_axis(x[..., :AXIS_DIM], p, inv_freq)
    time_part = _rotate_axis(x[..., AXIS_DIM:], tau, inv_freq)
    return torch.cat((pos_part, time_part), dim=-1).to(x.dtype)


class MixerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(MIXER_DIM)
        self.ln_kv = nn.LayerNorm(MIXER_DIM)
        self.q_proj = nn.Linear(MIXER_DIM, MIXER_DIM)
        self.k_proj = nn.Linear(MIXER_DIM, MIXER_DIM)
        self.v_proj = nn.Linear(MIXER_DIM, MIXER_DIM)
        self.o_proj = nn.Linear(MIXER_DIM, MIXER_DIM)
        self.ln_ffn = nn.LayerNorm(MIXER_DIM)
        self.ffn = nn.Sequential(nn.Linear(MIXER_DIM, MIXER_FFN), nn.GELU(), nn.Linear(MIXER_FFN, MIXER_DIM))
        self.register_buffer("inv_freq", axis_inv_freq(), persistent=False)

    def forward(self, x: Tensor, p: Tensor, tau: Tensor, mem: Tensor, p_mem: Tensor, tau_mem: Tensor,
                mem_valid: Tensor | None = None) -> Tensor:
        batch, n_q, _ = x.shape
        n_k = mem.shape[1]
        q = self.q_proj(self.ln_q(x)).view(batch, n_q, MIXER_HEADS, MIXER_HEAD_DIM).transpose(1, 2)
        kv = self.ln_kv(mem)
        k = self.k_proj(kv).view(batch, n_k, MIXER_HEADS, MIXER_HEAD_DIM).transpose(1, 2)
        v = self.v_proj(kv).view(batch, n_k, MIXER_HEADS, MIXER_HEAD_DIM).transpose(1, 2)
        q = apply_two_axis_rope(q, p, tau, self.inv_freq)
        k = apply_two_axis_rope(k, p_mem, tau_mem, self.inv_freq)
        mask = None
        if mem_valid is not None and not bool(mem_valid.all()):
            if not bool(mem_valid.any(dim=1).all()):
                raise ValueError("every batch row needs at least one valid memory token")
            mask = mem_valid[:, None, None, :].expand(batch, MIXER_HEADS, n_q, n_k)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn = attn.transpose(1, 2).reshape(batch, n_q, MIXER_DIM)
        x = x + self.o_proj(attn)
        return x + self.ffn(self.ln_ffn(x))


class PerceptualMixer(nn.Module):
    """Two blocks sharing the same protocol for the first view and for later zooms (DESIGN 7.3)."""

    def __init__(self) -> None:
        super().__init__()
        self.block1 = MixerBlock()
        self.block2 = MixerBlock()

    def forward(self, h_cur: Tensor, p_cur: Tensor, tau_cur: Tensor, h_mem: Tensor, p_mem: Tensor,
                tau_mem: Tensor, mem_valid: Tensor | None = None) -> Tensor:
        z = self.block1(h_cur, p_cur, tau_cur, h_mem, p_mem, tau_mem, mem_valid)
        return self.block2(z, p_cur, tau_cur, h_mem, p_mem, tau_mem, mem_valid)


class Projector(nn.Module):
    """LayerNorm -> Linear 512->2560 -> GELU -> Linear 2560->2560, then a PARAMETER-FREE per-token RMS normalisation
    to ``target_rms`` (Astra 15/09, sentinella D2).

    Every soft token leaves with exactly the backbone text-embedding RMS per dimension: the norm cannot drift and no
    gain is learned; training shapes directions only.  The second linear has no activation and a zero-initialised
    bias so that the positive common component of a GELU-terminal projector can be unlearned.  ``target_rms=None``
    (unit tests without a backbone) skips the normalisation.
    """

    def __init__(self, out_dim: int = PROJECTOR_OUT, *, target_rms: float | None = None) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(MIXER_DIM)
        self.proj = nn.Linear(MIXER_DIM, out_dim)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(out_dim, out_dim)
        nn.init.zeros_(self.proj_out.bias)
        self.target_rms = float(target_rms) if target_rms is not None else None

    def unscaled(self, z: Tensor) -> Tensor:
        return self.proj_out(self.act(self.proj(self.norm(z))))

    def forward(self, z: Tensor) -> Tensor:
        y = self.unscaled(z)
        if self.target_rms is None:
            return y
        rms = y.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return (y.float() / rms * self.target_rms).to(y.dtype)


def soft_token_decorrelation(softs: Sequence[Tensor], *, max_tokens: int = 2048) -> Tensor:
    """Mean SQUARED off-diagonal cosine among the soft tokens of each view, averaged over views (Astra 15/09,
    sentinella D2: the signed mean would let opposite directions cancel).  Views with more than ``max_tokens``
    tokens are strided down deterministically to bound the Gram matrix.  Differentiable."""

    terms = []
    for soft in softs:
        s = soft.float()
        n = int(s.shape[0])
        if n > max_tokens:
            s = s[:: math.ceil(n / max_tokens)]
            n = int(s.shape[0])
        if n < 2:
            continue
        unit = s / s.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        gram = unit @ unit.T
        terms.append((gram.pow(2).sum() - gram.diagonal().pow(2).sum()) / (n * (n - 1)))
    if not terms:
        return torch.zeros(())
    return torch.stack(terms).mean()


@dataclass
class MemoryView:
    descriptors: Tensor  # [N, 512] pre-mixer descriptors (valid tokens only, sorted)
    p: Tensor  # [N] float32 ordinals
    tau: Tensor  # [N] float32
    anchor_time: Tensor  # [N] float64
    kind: str
    start: int
    end: int
    channel_ids: tuple[str, ...]


@dataclass
class PerceptualMemory:
    """Descriptors of every acquired view with immutable ordinals; rebuilt from parameters each step."""

    t_origin: float
    dt_ref: float
    views: list[MemoryView] = field(default_factory=list)

    @property
    def size(self) -> int:
        return sum(int(v.descriptors.shape[0]) for v in self.views)

    def append(self, descriptors: Tensor, anchor_time: Tensor, *, kind: str, start: int, end: int,
               channel_ids: tuple[str, ...]) -> MemoryView:
        if descriptors.ndim != 2 or descriptors.shape[1] != MIXER_DIM:
            raise ValueError("descriptors must be [N, 512]")
        if anchor_time.shape != (descriptors.shape[0],):
            raise ValueError("anchor_time must be [N]")
        offset = self.size
        n = descriptors.shape[0]
        p = torch.arange(offset, offset + n, dtype=torch.float32, device=descriptors.device)
        tau = tau_from_anchor(anchor_time, self.t_origin, self.dt_ref).to(descriptors.device)
        view = MemoryView(descriptors, p, tau, anchor_time, kind, start, end, tuple(channel_ids))
        self.views.append(view)
        return view

    def stacked(self) -> tuple[Tensor, Tensor, Tensor]:
        if not self.views:
            raise ValueError("memory is empty")
        h = torch.cat([v.descriptors for v in self.views], dim=0)
        p = torch.cat([v.p for v in self.views], dim=0)
        tau = torch.cat([v.tau for v in self.views], dim=0)
        return h, p, tau


__all__ = [
    "MIXER_DIM",
    "MemoryView",
    "MixerBlock",
    "PerceptualMemory",
    "PerceptualMixer",
    "Projector",
    "TIME_UNIT_SAMPLES",
    "apply_two_axis_rope",
    "axis_inv_freq",
    "tau_from_anchor",
]
