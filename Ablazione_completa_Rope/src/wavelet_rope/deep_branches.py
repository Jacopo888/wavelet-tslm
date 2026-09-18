"""Deep per-branch temporal trunks for the perceptual front-end (DESIGN 6.1, 8).

Three trunks (A, D, raw) process dense coefficient or sample sequences before
any token selection.  Every trunk shares its weights across channels: channels
are run as the batch dimension and channel identity enters only later through
the token channel embedding.  The D trunk shares one weight set across Haar
levels and is conditioned on a level embedding through FiLM.  Each branch also
owns two local adapter blocks that are applied only to local views, after the
shared trunk (DESIGN 8).

Masking follows Astra decision 2: invalid positions are zeroed before every
convolution and after every block, so no bias or convolution can recreate a
valid slot from an invalid one.  The mask is always passed explicitly.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


WIDTH = 64
KERNEL_SIZE = 5
EXPANSION = 2
RESIDUAL_SCALE = 0.1  # fixed constant, deliberately not a parameter (DESIGN 6.1)
MAX_LEVEL_EMBEDDINGS = 13  # levels 1..12; index 0 reserved (never used by D)

A_INPUT_DIM = 9
D_INPUT_DIM = 11
RAW_INPUT_DIM = 7

A_DILATIONS: tuple[int, ...] = (1, 2, 4)
D_DILATIONS: tuple[int, ...] = (1, 2, 4, 8)
RAW_DILATIONS: tuple[int, ...] = (1, 2)
LOCAL_ADAPTER_DILATIONS: tuple[int, ...] = (1, 2)

FILM_INIT_STD = 0.02


def mask_positions(h: Tensor, valid: Tensor) -> Tensor:
    """Zero every feature vector whose position is invalid (out of place)."""

    if valid.dtype != torch.bool:
        raise TypeError("valid mask must be bool")
    if tuple(valid.shape) != tuple(h.shape[:-1]):
        raise ValueError(
            f"valid mask shape {tuple(valid.shape)} does not match positions "
            f"{tuple(h.shape[:-1])}"
        )
    return h.masked_fill(~valid.unsqueeze(-1), 0.0)


class ResidualTemporalBlock(nn.Module):
    """One DESIGN 6.1 block: LN -> (FiLM) -> depthwise conv -> MLP, residual 0.1.

    ``u = LayerNorm(h)`` per position; optional FiLM ``u = u * (1 + gamma) + beta``
    with ``gamma``/``beta`` of shape ``[B, 1, width]``; invalid positions are
    zeroed before the bidirectional depthwise convolution (symmetric padding);
    ``u = Linear(width, 2*width) -> SiLU -> Linear(2*width, width)``;
    ``h_next = h + 0.1 * u``, zeroed again on invalid positions.
    """

    def __init__(
        self,
        width: int = WIDTH,
        kernel_size: int = KERNEL_SIZE,
        dilation: int = 1,
        expansion: int = EXPANSION,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd for symmetric padding")
        if dilation < 1:
            raise ValueError("dilation must be >= 1")
        self.width = int(width)
        self.dilation = int(dilation)
        self.residual_scale = float(RESIDUAL_SCALE)
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(
            width,
            width,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
            groups=width,
        )
        self.up = nn.Linear(width, width * expansion)
        self.down = nn.Linear(width * expansion, width)

    def forward(
        self,
        h: Tensor,
        valid: Tensor,
        film: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if h.ndim != 3 or h.shape[-1] != self.width:
            raise ValueError(
                f"h must have shape [B, N, {self.width}], got {tuple(h.shape)}"
            )
        u = self.norm(h)
        if film is not None:
            gamma, beta = film
            u = u * (1.0 + gamma) + beta
        u = mask_positions(u, valid)
        u = self.conv(u.transpose(1, 2)).transpose(1, 2)
        u = self.down(F.silu(self.up(u)))
        h = h + self.residual_scale * u
        return mask_positions(h, valid)


class BranchStem(nn.Module):
    """``Linear(n_in -> width)`` -> SiLU -> LayerNorm per position (Astra 15/09); output masked on invalid slots."""

    def __init__(self, n_in: int, width: int = WIDTH) -> None:
        super().__init__()
        self.n_in = int(n_in)
        self.proj = nn.Linear(n_in, width)
        self.norm = nn.LayerNorm(width)  # Astra 15/09: per-position LN after the stem, before the residual blocks

    def forward(self, feats: Tensor, valid: Tensor | None = None) -> Tensor:
        if feats.ndim != 3 or feats.shape[-1] != self.n_in:
            raise ValueError(
                f"stem expects features [B, N, {self.n_in}], got {tuple(feats.shape)}"
            )
        h = self.norm(F.silu(self.proj(feats)))
        if valid is not None:
            h = mask_positions(h, valid)
        return h


class _BlockStack(nn.Module):
    """A sequence of residual blocks, optionally FiLM-conditioned per block."""

    def __init__(
        self,
        dilations: Sequence[int],
        *,
        width: int,
        kernel_size: int,
        expansion: int,
        film: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(width, kernel_size, int(d), expansion)
            for d in dilations
        )
        self.film: nn.ModuleList | None = None
        if film:
            self.film = nn.ModuleList(nn.Linear(width, 2 * width) for _ in dilations)
            for layer in self.film:
                nn.init.normal_(layer.weight, std=FILM_INIT_STD)
                nn.init.zeros_(layer.bias)

    def forward(
        self, h: Tensor, valid: Tensor, level_vec: Tensor | None = None
    ) -> Tensor:
        if (self.film is None) != (level_vec is None):
            raise ValueError("level conditioning must be given exactly when FiLM exists")
        for index, block in enumerate(self.blocks):
            film = None
            if self.film is not None:
                assert level_vec is not None
                gamma, beta = self.film[index](level_vec).chunk(2, dim=-1)
                shape = (h.shape[0], 1, h.shape[-1])
                film = (
                    gamma.reshape(1, 1, -1).expand(shape),
                    beta.reshape(1, 1, -1).expand(shape),
                )
            h = block(h, valid, film)
        return h


class DeepBranches(nn.Module):
    """A, D and raw trunks plus per-branch local adapters (DESIGN 6.1 and 8).

    * ``a_stem`` + ``a_blocks`` (dilations 1, 2, 4); ``a_adapter`` (1, 2).
    * ``d_stem`` + ``d_blocks`` (dilations 1, 2, 4, 8) with ONE weight set shared
      across levels, conditioned per block by FiLM from ``level_embedding``;
      ``d_adapter`` (1, 2) with its own weights and its own FiLM layers.
    * ``raw_stem`` + ``raw_blocks`` (dilations 1, 2); ``raw_adapter`` (1, 2).

    Adapters run only when ``is_local`` is true, after the shared trunk.
    """

    def __init__(self, spec: dict | None = None) -> None:
        super().__init__()
        cfg = dict(spec["branches"]) if spec is not None and "branches" in spec else {}
        width = int(cfg.get("width", WIDTH))
        kernel = int(cfg.get("kernel_size", KERNEL_SIZE))
        expansion = int(cfg.get("expansion", EXPANSION))
        residual = float(cfg.get("residual_scale", RESIDUAL_SCALE))
        if residual != RESIDUAL_SCALE or cfg.get("residual_scale_trainable", False):
            raise ValueError("the residual scale is a fixed 0.1 constant here")
        a_dil = tuple(int(d) for d in cfg.get("A_dilations", A_DILATIONS))
        d_dil = tuple(int(d) for d in cfg.get("D_dilations", D_DILATIONS))
        raw_dil = tuple(int(d) for d in cfg.get("raw_dilations", RAW_DILATIONS))
        local_dil = tuple(
            int(d)
            for d in cfg.get("local_adapter_dilations_per_branch", LOCAL_ADAPTER_DILATIONS)
        )
        self.width = width
        common = dict(width=width, kernel_size=kernel, expansion=expansion)

        self.a_stem = BranchStem(A_INPUT_DIM, width)
        self.a_blocks = _BlockStack(a_dil, film=False, **common)
        self.a_adapter = _BlockStack(local_dil, film=False, **common)

        self.level_embedding = nn.Embedding(MAX_LEVEL_EMBEDDINGS, width)
        self.d_stem = BranchStem(D_INPUT_DIM, width)
        self.d_blocks = _BlockStack(d_dil, film=True, **common)
        self.d_adapter = _BlockStack(local_dil, film=True, **common)

        self.raw_stem = BranchStem(RAW_INPUT_DIM, width)
        self.raw_blocks = _BlockStack(raw_dil, film=False, **common)
        self.raw_adapter = _BlockStack(local_dil, film=False, **common)

    @staticmethod
    def _check(feats: Tensor, valid: Tensor) -> None:
        if feats.ndim != 3:
            raise ValueError("branch features must have shape [B, N, n_in]")
        if valid.shape != feats.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be a bool mask of shape [B, N]")
        if valid.device != feats.device:
            raise ValueError("features and mask must share a device")

    def forward_A(self, feats: Tensor, valid: Tensor, is_local: bool) -> Tensor:
        self._check(feats, valid)
        h = self.a_stem(feats, valid)
        h = self.a_blocks(h, valid)
        if is_local:
            h = self.a_adapter(h, valid)
        return h

    def forward_D(
        self, feats: Tensor, valid: Tensor, level: int, is_local: bool
    ) -> Tensor:
        self._check(feats, valid)
        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError("level must be an integer")
        if not 1 <= level < MAX_LEVEL_EMBEDDINGS:
            raise ValueError(
                f"level must be in [1, {MAX_LEVEL_EMBEDDINGS - 1}], got {level}"
            )
        level_vec = self.level_embedding(
            torch.tensor([level], dtype=torch.int64, device=feats.device)
        )
        h = self.d_stem(feats, valid)
        h = self.d_blocks(h, valid, level_vec)
        if is_local:
            h = self.d_adapter(h, valid, level_vec)
        return h

    def forward_raw(self, feats: Tensor, valid: Tensor, is_local: bool) -> Tensor:
        self._check(feats, valid)
        h = self.raw_stem(feats, valid)
        h = self.raw_blocks(h, valid)
        if is_local:
            h = self.raw_adapter(h, valid)
        return h


__all__ = [
    "A_DILATIONS",
    "A_INPUT_DIM",
    "BranchStem",
    "D_DILATIONS",
    "D_INPUT_DIM",
    "DeepBranches",
    "EXPANSION",
    "KERNEL_SIZE",
    "LOCAL_ADAPTER_DILATIONS",
    "MAX_LEVEL_EMBEDDINGS",
    "RAW_DILATIONS",
    "RAW_INPUT_DIM",
    "RESIDUAL_SCALE",
    "ResidualTemporalBlock",
    "WIDTH",
    "mask_positions",
]
