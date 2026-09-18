"""Perceptual tokenizer: quotas, pooling, salience, fusion, provenance (DESIGN 6.2, 6.3).

The tokenizer consumes one :class:`SignalView` together with its Haar pyramids,
runs the deep branches over every coefficient and sample, and emits a fixed
grid of ``C * 4 * Q`` token slots (four families per channel, ``Q`` slots per
family).  Slots that cannot be filled with observed data are left invalid; no
token is ever duplicated to reach a quota (Astra decision 2).

All coordinates are absolute: supports are half-open sample intervals of the
original recording and anchor times are recording-clock FP64 timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .deep_branches import (
    A_INPUT_DIM,
    D_INPUT_DIM,
    MAX_LEVEL_EMBEDDINGS,
    RAW_INPUT_DIM,
    WIDTH,
    DeepBranches,
    mask_positions,
)
from .pyramid import HaarBand, HaarPyramid, build_haar_pyramid
from .signal_contract import (
    MAD_TO_SIGMA,
    MAX_CHANNELS,
    RELATIVE_SCALE_FLOOR,
    SignalView,
    ViewKind,
)


FAMILY_A, FAMILY_D_COVERAGE, FAMILY_D_SALIENT, FAMILY_RAW = 0, 1, 2, 3
N_FAMILIES = 4
FAMILY_NAMES = ("A", "D_coverage", "D_salient", "raw")

FUSION_INPUT_DIM = 4 * WIDTH  # 256
DESCRIPTOR_DIM = 512
META_DIM = 5
SALIENT_PER_BIN = 2
_LOG2_MAX_LEVELS = 12.0
# Astra 15/09 (scala). DESIGN §6 r.141 presupposes a HARD clip of the linear stem columns next to the unclipped
# asinh columns; the band scale gets an absolute floor in normalised units; the fused content is LayerNorm-ed
# before the additive identifiers, whose scale is calibrated (meta gain fixed, embeddings init std 0.5).
LINEAR_CLIP = 16.0
BAND_SCALE_FLOOR = 1e-3
META_GAIN = 0.25
EMBEDDING_INIT_STD = 0.5


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def _median(values: Tensor) -> Tensor:
    """Conventional median (mean of the two middle values for even counts)."""

    ordered = values.sort().values
    mid = ordered.numel() // 2
    if ordered.numel() % 2:
        return ordered[mid]
    return ordered[mid - 1] + (ordered[mid] - ordered[mid - 1]) * 0.5


def robust_band_scale(values: Tensor, usable: Tensor) -> Tensor:
    """Per-channel robust scale of a Haar band over USABLE coefficients only.

    Astra decision 2 applied to a band: ``m = median(V)``, ``MAD = median(|V - m|)``,
    ``eps = max(1e-6 * max|V|, finfo(float32).tiny)``, ``s = max(1.4826 * MAD, eps)``.
    A band with no usable coefficient gets ``s = 1`` (its coefficients are all
    invalid anyway).  Returns FP32 ``[C]``.
    """

    if values.ndim != 2 or usable.shape != values.shape:
        raise ValueError("band values and usable mask must both have shape [C, N]")
    channels = values.shape[0]
    tiny = torch.finfo(torch.float32).tiny
    scales: list[Tensor] = []
    with torch.autocast(device_type=values.device.type, enabled=False):
        for channel in range(channels):
            observed = values[channel][usable[channel]].to(torch.float32)
            if observed.numel() == 0:
                scales.append(torch.ones((), dtype=torch.float32, device=values.device))
                continue
            median = _median(observed)
            mad = _median((observed - median).abs())
            max_abs = observed.abs().amax().to(torch.float64)
            epsilon = torch.clamp_min(max_abs * RELATIVE_SCALE_FLOOR, tiny)
            scale = torch.maximum((mad * MAD_TO_SIGMA).to(torch.float64), epsilon)
            scale = torch.clamp_min(scale, BAND_SCALE_FLOOR)  # Astra 15/09: bands are in normalised units
            scales.append(scale.to(torch.float32))
    return torch.stack(scales)


def _lexsort(keys: Sequence[Tensor], descending: Sequence[bool] | None = None) -> Tensor:
    """Stable lexicographic argsort; ``keys[0]`` is the most significant key."""

    if not keys:
        raise ValueError("at least one sort key is required")
    n = keys[0].numel()
    if descending is None:
        descending = [False] * len(keys)
    perm = torch.arange(n, dtype=torch.int64, device=keys[0].device)
    for key, desc in reversed(list(zip(keys, descending))):
        if key.numel() != n:
            raise ValueError("all sort keys must have the same length")
        order = torch.sort(key[perm], stable=True, descending=bool(desc)).indices
        perm = perm[order]
    return perm


def uniform_bin_edges(start: int, end: int, n_bins: int, device: torch.device) -> Tensor:
    """Absolute ``[n_bins + 1]`` int64 edges of uniform bins over ``[start, end)``."""

    if n_bins < 1 or end <= start:
        raise ValueError("bins require n_bins >= 1 and a non-empty interval")
    length = end - start
    grid = torch.arange(n_bins + 1, dtype=torch.int64, device=device)
    return start + (grid * length) // n_bins


def _bin_of(edges: Tensor, positions: Tensor) -> Tensor:
    """Bin id of each position given nondecreasing edges (empty bins never win)."""

    if edges.dtype != positions.dtype:
        edges = edges.to(positions.dtype)
    n_bins = edges.numel() - 1
    return (torch.searchsorted(edges, positions, right=True) - 1).clamp_(0, n_bins - 1)


def _support_centre(band: HaarBand) -> Tensor:
    """FP64 centre sample index of each coefficient's support (channel independent)."""

    return (band.support_start + band.support_end - 1).to(torch.float64) * 0.5


def pool_uniform_bins(
    features: Tensor, valid: Tensor, bins: Tensor, n_bins: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Masked mean and max of ``features [B, N, D]`` per bin ``[B, Q]``.

    Invalid positions are routed to a dummy bin that is discarded, so they never
    contribute.  Returns ``(mean, amax, present)``; empty bins carry zeros.
    """

    if features.ndim != 3:
        raise ValueError("features must have shape [B, N, D]")
    batch, length, dim = features.shape
    if valid.shape != (batch, length) or bins.shape != (batch, length):
        raise ValueError("valid and bins must have shape [B, N]")
    stride = n_bins + 1
    routed = torch.where(valid, bins, torch.full_like(bins, n_bins))
    flat = (
        torch.arange(batch, dtype=torch.int64, device=bins.device)[:, None] * stride + routed
    ).reshape(-1)
    source = features.reshape(-1, dim)
    sums = torch.zeros((batch * stride, dim), dtype=source.dtype, device=source.device)
    sums = sums.index_add(0, flat, source)
    counts = torch.bincount(flat, minlength=batch * stride).reshape(batch, stride)[:, :n_bins]
    maxes = torch.zeros((batch * stride, dim), dtype=source.dtype, device=source.device)
    maxes = maxes.scatter_reduce(
        0, flat.unsqueeze(-1).expand(-1, dim), source, reduce="amax", include_self=False
    )
    sums = sums.reshape(batch, stride, dim)[:, :n_bins]
    maxes = maxes.reshape(batch, stride, dim)[:, :n_bins]
    present = counts > 0
    mean = sums / counts.clamp_min(1).to(sums.dtype).unsqueeze(-1)
    mean = mask_positions(mean, present)
    maxes = mask_positions(maxes, present)
    return mean, maxes, present


def select_salient_coefficients(
    score: Tensor,
    level: Tensor,
    index: Tensor,
    bin_id: Tensor,
    quota: int,
    per_bin: int = SALIENT_PER_BIN,
) -> tuple[Tensor, int]:
    """Deterministic salience selection over usable coefficients of one channel.

    First pass: within each salience bin, the best ``per_bin`` coefficients by
    ``score`` (ties broken by ``(level, index)``).  Refill: the best remaining
    coefficients of the whole view in the same order, without duplicates.
    Returns ``(positions, n_first_pass)``; ``positions`` indexes the inputs and
    has ``min(quota, len(score))`` entries, first-pass entries first.
    """

    count = score.numel()
    device = score.device
    if count == 0 or quota <= 0:
        return torch.empty(0, dtype=torch.int64, device=device), 0
    by_score = _lexsort([score, level, index], descending=[True, False, False])
    by_bin = by_score[torch.sort(bin_id[by_score], stable=True).indices]
    bins_sorted = bin_id[by_bin]
    change = torch.ones(count, dtype=torch.bool, device=device)
    change[1:] = bins_sorted[1:] != bins_sorted[:-1]
    group_start = change.nonzero().squeeze(1)
    group_id = torch.cumsum(change.to(torch.int64), dim=0) - 1
    rank = torch.arange(count, dtype=torch.int64, device=device) - group_start[group_id]
    chosen = torch.zeros(count, dtype=torch.bool, device=device)
    chosen[by_bin[rank < per_bin]] = True
    first = by_score[chosen[by_score]][:quota]
    n_first = int(first.numel())
    remaining = by_score[~chosen[by_score]][: max(0, quota - n_first)]
    return torch.cat((first, remaining)), n_first


def _interpolate_in_time(
    positions: Tensor, features: Tensor, valid: Tensor, query: Tensor
) -> Tensor:
    """Linear interpolation of ``features [N, D]`` (at ``positions``) at ``query [T]``.

    Only valid rows take part; queries outside the valid range are clamped to the
    nearest valid row; a channel without valid rows yields zeros.
    """

    keep = valid.nonzero().squeeze(1)
    dim = features.shape[-1]
    if keep.numel() == 0:
        return torch.zeros((query.numel(), dim), dtype=features.dtype, device=features.device)
    xa = positions[keep]
    fa = features[keep]
    if keep.numel() == 1:
        return fa.expand(query.numel(), dim)
    upper = torch.searchsorted(xa, query, right=True).clamp_(1, keep.numel() - 1)
    lower = upper - 1
    x0 = xa[lower]
    x1 = xa[upper]
    weight = ((query - x0) / (x1 - x0)).clamp_(0.0, 1.0).to(features.dtype).unsqueeze(-1)
    return fa[lower] * (1.0 - weight) + fa[upper] * weight


# --------------------------------------------------------------------------- #
# Branch inputs
# --------------------------------------------------------------------------- #


@dataclass
class BranchInputs:
    """Stem features for the three branches of one view (all FP32, masked)."""

    is_local: bool
    log_scale_ratio: Tensor  # [C] log(local scale / global scale), 0 for global views
    a_features: Tensor  # [C, N_A, 9]
    a_valid: Tensor  # [C, N_A]
    d_features: tuple[Tensor, ...]  # per level j: [C, N_j, 11]
    d_valid: tuple[Tensor, ...]  # per level j: [C, N_j]
    d_band_scale: tuple[Tensor, ...]  # per level j: [C] robust band scale
    raw_features: Tensor  # [C, L, 7]
    raw_valid: Tensor  # [C, L]


def _log_scale_ratio(view: SignalView) -> Tensor:
    ratio = view.local_to_global_scale
    if ratio is None:
        return torch.zeros(len(view.channel_ids), dtype=torch.float32, device=view.values.device)
    safe = torch.where(ratio > 0, ratio, torch.ones_like(ratio))
    return torch.where(ratio > 0, torch.log(safe), torch.zeros_like(ratio)).to(torch.float32)


def _band_features(
    band: HaarBand,
    local_band: HaarBand | None,
    log_ratio: Tensor,
    is_local: bool,
    band_scale: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """The nine common stem features of a band (plus two band-scaled ones for D)."""

    valid = band.usable
    z = band.values.to(torch.float32)
    channels, count = z.shape
    ones = torch.ones_like(z)
    z_local = local_band.values.to(torch.float32) if local_band is not None else torch.zeros_like(z)
    columns = [
        z.clamp(-LINEAR_CLIP, LINEAR_CLIP),
        torch.asinh(z),
        valid.to(torch.float32),
        band.observed_fraction_of_nominal.to(torch.float32),
        ones * (math.log2(band.nominal_width) / _LOG2_MAX_LEVELS),
        z_local.clamp(-LINEAR_CLIP, LINEAR_CLIP),
        torch.asinh(z_local),
        ones * log_ratio.to(torch.float32)[:, None],
        ones * (1.0 if is_local else 0.0),
    ]
    if band_scale is not None:
        z_band = z / band_scale.to(torch.float32)[:, None]
        columns.extend([z_band.clamp(-LINEAR_CLIP, LINEAR_CLIP), torch.asinh(z_band)])
    feats = torch.stack(columns, dim=-1)
    return mask_positions(feats, valid), valid


def build_branch_inputs(
    view: SignalView,
    pyramid_global: HaarPyramid,
    pyramid_local: HaarPyramid | None = None,
) -> BranchInputs:
    """Assemble stem features for A, every D level, and raw samples."""

    if pyramid_global.trace != "global":
        raise ValueError("pyramid_global must carry the global normalization trace")
    if (pyramid_global.start, pyramid_global.end) != (view.start, view.end):
        raise ValueError("pyramid_global does not belong to this view interval")
    is_local = view.kind is ViewKind.LOCAL
    if is_local:
        if pyramid_local is None:
            raise ValueError("a local view requires its local-trace pyramid")
        if pyramid_local.trace != "local":
            raise ValueError("pyramid_local must carry the local normalization trace")
        if (pyramid_local.start, pyramid_local.end) != (view.start, view.end):
            raise ValueError("pyramid_local does not belong to this view interval")
        if pyramid_local.levels != pyramid_global.levels:
            raise ValueError("global and local pyramids must share the same depth")
    elif pyramid_local is not None:
        raise ValueError("a global view has no local-trace pyramid")

    with torch.autocast(device_type=view.values.device.type, enabled=False):
        log_ratio = _log_scale_ratio(view)
        a_feats, a_valid = _band_features(
            pyramid_global.approximation,
            pyramid_local.approximation if pyramid_local is not None else None,
            log_ratio,
            is_local,
        )
        d_feats: list[Tensor] = []
        d_valid: list[Tensor] = []
        d_scale: list[Tensor] = []
        for level in range(1, pyramid_global.levels + 1):
            band = pyramid_global.detail(level)
            scale = robust_band_scale(band.values, band.usable)
            feats, valid = _band_features(
                band,
                pyramid_local.detail(level) if pyramid_local is not None else None,
                log_ratio,
                is_local,
                band_scale=scale,
            )
            d_feats.append(feats)
            d_valid.append(valid)
            d_scale.append(scale)

        z_global = view.global_normalized.to(torch.float32)
        z_local = (
            view.local_normalized.to(torch.float32)
            if view.local_normalized is not None
            else torch.zeros_like(z_global)
        )
        ones = torch.ones_like(z_global)
        raw = torch.stack(
            [
                z_global.clamp(-LINEAR_CLIP, LINEAR_CLIP),
                torch.asinh(z_global),
                view.valid.to(torch.float32),
                z_local.clamp(-LINEAR_CLIP, LINEAR_CLIP),
                torch.asinh(z_local),
                ones * log_ratio[:, None],
                ones * (1.0 if is_local else 0.0),
            ],
            dim=-1,
        )
        raw = mask_positions(raw, view.valid)

    assert a_feats.shape[-1] == A_INPUT_DIM
    assert all(f.shape[-1] == D_INPUT_DIM for f in d_feats)
    assert raw.shape[-1] == RAW_INPUT_DIM
    return BranchInputs(
        is_local=is_local,
        log_scale_ratio=log_ratio,
        a_features=a_feats,
        a_valid=a_valid,
        d_features=tuple(d_feats),
        d_valid=tuple(d_valid),
        d_band_scale=tuple(d_scale),
        raw_features=raw,
        raw_valid=view.valid,
    )


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #


@dataclass
class ViewTokens:
    """Fixed-slot perceptual tokens of one view with absolute provenance.

    Slot layout: index ``(channel * 4 + family) * Q + q``.  Invalid slots carry
    zero descriptors, zero anchor times and empty supports.
    """

    descriptors: Tensor  # [N, 512]
    valid: Tensor  # [N] bool
    anchor_time: Tensor  # [N] float64 recording-clock anchor
    support_start: Tensor  # [N] int64 absolute sample index (inclusive)
    support_end: Tensor  # [N] int64 absolute sample index (exclusive)
    channel_index: Tensor  # [N] int64 channel index in the EPISODE
    family: Tensor  # [N] int64 in {0, 1, 2, 3}
    # A tokens carry the pyramid depth J of their view, salient D tokens the level
    # j of the selected coefficient.  D coverage summaries mix every present band
    # inside one bin and raw patches have no Haar level, so both carry 0.
    level: Tensor  # [N] int64 Haar level (A: J, D salient: j, 0 otherwise)
    view_kind: ViewKind
    start: int
    end: int
    raw_bin_features: Tensor  # [C, Q, 64] raw uniform-bin summaries before fusion
    raw_bin_valid: Tensor  # [C, Q] bool
    raw_bin_edges: Tensor  # [Q + 1] int64 absolute sample edges
    sort_order: Tensor  # [N] int64 permutation; valid tokens by (t, ch, fam, lvl, k)
    coefficient_index: Tensor  # [N] int64 bin index (uniform) or coefficient k (salient)

    @property
    def n_tokens(self) -> int:
        return int(self.valid.shape[0])

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #


class PerceptualTokenizer(nn.Module):
    """Quotas, uniform pooling, salience, anchored fusion (DESIGN 6.2 / 6.3)."""

    def __init__(self, branches: DeepBranches, spec: dict) -> None:
        super().__init__()
        self.branches = branches
        tok = spec["tokenizer"]
        self._quota = {
            ViewKind.GLOBAL: self._read_quota(tok["global_per_channel"]),
            ViewKind.LOCAL: self._read_quota(tok["local_per_channel"]),
        }
        self._salience_bins = {
            ViewKind.GLOBAL: int(tok["global_salience_bins"]),
            ViewKind.LOCAL: int(tok["local_salience_bins"]),
        }
        width = int(branches.width)
        fusion_in = int(tok.get("fusion_input_dim", FUSION_INPUT_DIM))
        fusion_out = int(tok.get("fusion_output_dim", DESCRIPTOR_DIM))
        if fusion_in != 4 * width:
            raise ValueError("fusion input must concatenate four branch-width vectors")
        self.width = width
        self.descriptor_dim = fusion_out

        self.a_pool = nn.Linear(2 * width, width)
        self.d_pool = nn.Linear(2 * width, width)
        self.raw_pool = nn.Linear(2 * width, width)
        self.fuse_in = nn.Linear(fusion_in + 2 * int(tok.get("position_fourier_freqs", 0)), fusion_out)
        self.fuse_out = nn.Linear(fusion_out, fusion_out)
        self.fuse_norm = nn.LayerNorm(fusion_out)  # Astra 15/09: per-token LN of the fused content
        # Astra 16/09: the normalised position enters meta as position/L.  Under bf16 autocast that scalar stops
        # resolving anything finer than ~257 samples at index 65.536 of a 131.072-sample series, and ~585 at the
        # end -- against a median anomaly width of 137.  With this flag the metadata projection runs in fp32 so
        # the position survives; the default keeps the behaviour every run so far was trained with.
        self.meta_proj_fp32 = bool(tok.get("meta_proj_fp32", False))
        self.family_embedding = nn.Embedding(N_FAMILIES, fusion_out)
        self.level_embedding = nn.Embedding(MAX_LEVEL_EMBEDDINGS, fusion_out)
        self.channel_embedding = nn.Embedding(MAX_CHANNELS, fusion_out)
        self.view_embedding = nn.Embedding(2, fusion_out)
        self.meta_proj = nn.Linear(META_DIM, fusion_out)
        # Fable 18/09: la posizione arriva al LM solo come uno scalare a guadagno 0,25 piu' una fase RoPE che la
        # wildcard cancella nelle coppie testo-segnale, quindi non esiste una forma COPIABILE di "dove".  va1 --
        # l'unico sistema che abbia localizzato in vista globale -- costruisce il token dalle feature di Fourier
        # della posizione insieme al valore.  Qui si riproduce quella costruzione concatenandole all'ingresso
        # della fusione (256 -> 256 + 2*freqs), non sommandole al contenuto gia' normalizzato.
        self.position_fourier_freqs = int(tok.get("position_fourier_freqs", 0))
        if self.position_fourier_freqs > 0:
            self.register_buffer("position_freqs",
                                 (2.0 ** torch.arange(self.position_fourier_freqs, dtype=torch.float64))
                                 * math.pi, persistent=False)
        # Astra 15/09: the four categorical embeddings start at std 0.5 so that their SUM has norm ~22 (one
        # LayerNorm unit) instead of ~44 with the default N(0, 1); they stay trainable.
        for embedding in (self.family_embedding, self.level_embedding, self.channel_embedding, self.view_embedding):
            nn.init.normal_(embedding.weight, mean=0.0, std=EMBEDDING_INIT_STD)

    @staticmethod
    def _read_quota(entry: dict) -> int:
        values = {int(entry[name]) for name in FAMILY_NAMES}
        if len(values) != 1:
            raise ValueError("this tokenizer requires one quota shared by all four families")
        return values.pop()

    def quota(self, kind: ViewKind) -> int:
        return self._quota[ViewKind(kind)]

    def salience_bins(self, kind: ViewKind) -> int:
        return self._salience_bins[ViewKind(kind)]

    # ------------------------------------------------------------------ #

    def _project_meta(self, meta: Tensor) -> Tensor:
        """Project the per-token metadata, optionally outside autocast to keep the position resolvable."""
    
        if not self.meta_proj_fp32:
            return self.meta_proj(meta)
        with torch.autocast(device_type=meta.device.type, enabled=False):
            return self.meta_proj(meta.float()).to(meta.dtype)
    
    def forward(
        self,
        view: SignalView,
        pyramid_global: HaarPyramid,
        pyramid_local: HaarPyramid | None,
        *,
        recording_length: int,
        channel_indices: Sequence[int],
    ) -> ViewTokens:
        device = view.values.device
        channels = len(view.channel_ids)
        start, end = view.start, view.end
        length = end - start
        kind = view.kind
        quota = self.quota(kind)
        n_sal_bins = self.salience_bins(kind)
        if recording_length != view.recording_length:
            raise ValueError("recording_length disagrees with the view contract")
        channel_index = torch.as_tensor(tuple(channel_indices), dtype=torch.int64, device=device)
        if tuple(channel_index.tolist()) != tuple(view.channel_indices):
            raise ValueError("channel_indices must be the episode indices of the view channels")
        if channels and int(channel_index.max()) >= MAX_CHANNELS:
            raise ValueError("channel index outside the supported channel count")

        inputs = build_branch_inputs(view, pyramid_global, pyramid_local)
        is_local = inputs.is_local
        width = self.width
        fdtype = torch.float32

        # ---- deep features -------------------------------------------------
        a_feat = self.branches.forward_A(inputs.a_features, inputs.a_valid, is_local)
        d_feats = [
            self.branches.forward_D(f, v, level=j, is_local=is_local)
            for j, (f, v) in enumerate(zip(inputs.d_features, inputs.d_valid), start=1)
        ]
        raw_feat = self.branches.forward_raw(inputs.raw_features, inputs.raw_valid, is_local)

        # ---- uniform bin geometry -----------------------------------------
        edges = uniform_bin_edges(start, end, quota, device)
        bin_lo, bin_hi = edges[:-1], edges[1:]
        bin_len = bin_hi - bin_lo
        nonempty = bin_len > 0
        times = view.sample_times
        lo_rel = (bin_lo - start).clamp_(0, length - 1)
        hi_rel = (bin_hi - 1 - start).clamp_(0, length - 1)
        bin_centre_time = torch.where(
            nonempty, (times[lo_rel] + times[hi_rel]) * 0.5, torch.zeros_like(times[lo_rel])
        )
        bin_centre_pos = torch.where(
            nonempty,
            (bin_lo + bin_hi - 1).to(torch.float64) * 0.5,
            torch.zeros(quota, dtype=torch.float64, device=device),
        )
        sample_pos = torch.arange(start, end, dtype=torch.int64, device=device)
        sample_bin = _bin_of(edges, sample_pos)  # [L]
        sample_bins = sample_bin.unsqueeze(0).expand(channels, -1)
        routed = torch.where(view.valid, sample_bins, torch.full_like(sample_bins, quota))
        flat = (
            torch.arange(channels, dtype=torch.int64, device=device)[:, None] * (quota + 1) + routed
        ).reshape(-1)
        bin_valid_count = torch.bincount(flat, minlength=channels * (quota + 1)).reshape(
            channels, quota + 1
        )[:, :quota]
        bin_valid_fraction = bin_valid_count.to(fdtype) / bin_len.clamp_min(1).to(fdtype)

        # ---- raw uniform summaries (aux head input) ------------------------
        raw_mean, raw_max, raw_present = pool_uniform_bins(
            raw_feat, inputs.raw_valid, sample_bins, quota
        )
        raw_bin_features = mask_positions(
            self.raw_pool(torch.cat((raw_mean, raw_max), dim=-1)), raw_present
        )

        # ---- A uniform summaries ------------------------------------------
        a_band = pyramid_global.approximation
        a_pos = _support_centre(a_band)  # [N_A]
        a_bins = _bin_of(edges, a_pos).unsqueeze(0).expand(channels, -1)
        a_mean, a_max, a_present = pool_uniform_bins(a_feat, inputs.a_valid, a_bins, quota)
        a_bin_features = mask_positions(
            self.a_pool(torch.cat((a_mean, a_max), dim=-1)), a_present
        )

        # ---- D coverage summaries and salient candidates -------------------
        sal_edges = uniform_bin_edges(start, end, n_sal_bins, device)
        mean_sum = torch.zeros((channels, quota, width), dtype=a_feat.dtype, device=device)
        max_all = torch.full((channels, quota, width), -math.inf, dtype=a_feat.dtype, device=device)
        level_count = torch.zeros((channels, quota), dtype=torch.int64, device=device)
        cand_level: list[Tensor] = []
        cand_index: list[Tensor] = []
        cand_pos: list[Tensor] = []
        cand_sal_bin: list[Tensor] = []
        cand_uni_bin: list[Tensor] = []
        cand_start: list[Tensor] = []
        cand_end: list[Tensor] = []
        cand_score: list[Tensor] = []
        cand_usable: list[Tensor] = []
        cand_anchor: list[Tensor] = []
        for level_index, (band_feat, valid_j) in enumerate(zip(d_feats, inputs.d_valid)):
            level = level_index + 1
            band = pyramid_global.detail(level)
            pos = _support_centre(band)
            uni_bin = _bin_of(edges, pos)
            d_mean, d_max, d_present = pool_uniform_bins(
                band_feat, valid_j, uni_bin.unsqueeze(0).expand(channels, -1), quota
            )
            mean_sum = mean_sum + d_mean
            max_all = torch.where(d_present.unsqueeze(-1), torch.maximum(max_all, d_max), max_all)
            level_count = level_count + d_present.to(torch.int64)

            count = band.values.shape[1]
            cand_level.append(torch.full((count,), level, dtype=torch.int64, device=device))
            cand_index.append(torch.arange(count, dtype=torch.int64, device=device))
            cand_pos.append(pos)
            cand_sal_bin.append(_bin_of(sal_edges, pos))
            cand_uni_bin.append(uni_bin)
            cand_start.append(band.support_start)
            cand_end.append(band.support_end)
            scale = inputs.d_band_scale[level_index].to(torch.float32)
            cand_score.append(band.values.abs().to(torch.float32) / scale[:, None])
            cand_usable.append(band.usable)
            cand_anchor.append(band.anchor_time)
        d_cov_valid = level_count > 0
        d_cov_mean = mean_sum / level_count.clamp_min(1).to(mean_sum.dtype).unsqueeze(-1)
        d_cov_max = torch.where(d_cov_valid.unsqueeze(-1), max_all, torch.zeros_like(max_all))
        d_cov_features = mask_positions(
            self.d_pool(torch.cat((d_cov_mean, d_cov_max), dim=-1)), d_cov_valid
        )

        all_level = torch.cat(cand_level)
        all_index = torch.cat(cand_index)
        all_pos = torch.cat(cand_pos)
        all_sal_bin = torch.cat(cand_sal_bin)
        all_uni_bin = torch.cat(cand_uni_bin)
        all_start = torch.cat(cand_start)
        all_end = torch.cat(cand_end)
        all_score = torch.cat(cand_score, dim=1)  # [C, M]
        all_usable = torch.cat(cand_usable, dim=1)
        all_anchor = torch.cat(cand_anchor, dim=1)

        sal_feature_rows: list[Tensor] = []
        sal_valid = torch.zeros((channels, quota), dtype=torch.bool, device=device)
        sal_level = torch.zeros((channels, quota), dtype=torch.int64, device=device)
        sal_index = torch.zeros((channels, quota), dtype=torch.int64, device=device)
        sal_pos = torch.zeros((channels, quota), dtype=torch.float64, device=device)
        sal_uni_bin = torch.zeros((channels, quota), dtype=torch.int64, device=device)
        sal_start = torch.zeros((channels, quota), dtype=torch.int64, device=device)
        sal_end = torch.zeros((channels, quota), dtype=torch.int64, device=device)
        sal_anchor = torch.zeros((channels, quota), dtype=torch.float64, device=device)
        for channel in range(channels):
            usable_pos = all_usable[channel].nonzero().squeeze(1)
            chosen, _ = select_salient_coefficients(
                all_score[channel, usable_pos],
                all_level[usable_pos],
                all_index[usable_pos],
                all_sal_bin[usable_pos],
                quota,
            )
            picked = usable_pos[chosen]
            n_picked = int(picked.numel())
            slots = torch.arange(n_picked, dtype=torch.int64, device=device)
            rows = torch.zeros((quota, width), dtype=a_feat.dtype, device=device)
            if n_picked:
                # Gather the deep feature of every selected coefficient from its own
                # band: the loop is over levels (at most 12), never over samples, and
                # it avoids materialising a copy of the whole D feature stack.
                picked_level = all_level[picked]
                picked_index = all_index[picked]
                for level_index, band_feat in enumerate(d_feats):
                    take = (picked_level == level_index + 1).nonzero().squeeze(1)
                    if take.numel():
                        rows = rows.index_add(
                            0, slots[take], band_feat[channel][picked_index[take]]
                        )
                sal_valid[channel, slots] = True
                sal_level[channel, slots] = all_level[picked]
                sal_index[channel, slots] = all_index[picked]
                sal_pos[channel, slots] = all_pos[picked]
                sal_uni_bin[channel, slots] = all_uni_bin[picked]
                sal_start[channel, slots] = all_start[picked]
                sal_end[channel, slots] = all_end[picked]
                sal_anchor[channel, slots] = all_anchor[channel, picked]
            sal_feature_rows.append(rows)
        sal_features = torch.stack(sal_feature_rows, dim=0)  # [C, Q, 64]

        # ---- assemble the four families ------------------------------------
        slot_q = torch.arange(quota, dtype=torch.int64, device=device)
        uni_index = slot_q.unsqueeze(0).expand(channels, -1)
        uni_anchor = bin_centre_time.unsqueeze(0).expand(channels, -1)
        uni_pos = bin_centre_pos.unsqueeze(0).expand(channels, -1)
        uni_start = bin_lo.unsqueeze(0).expand(channels, -1)
        uni_end = bin_hi.unsqueeze(0).expand(channels, -1)
        approx_level = int(pyramid_global.approximation.level)

        own = torch.stack((a_bin_features, d_cov_features, sal_features, raw_bin_features), dim=1)
        valid = torch.stack((a_present, d_cov_valid, sal_valid, raw_present), dim=1)
        anchor_time = torch.stack((uni_anchor, uni_anchor, sal_anchor, uni_anchor), dim=1)
        anchor_pos = torch.stack((uni_pos, uni_pos, sal_pos, uni_pos), dim=1)
        support_start = torch.stack((uni_start, uni_start, sal_start, uni_start), dim=1)
        support_end = torch.stack((uni_end, uni_end, sal_end, uni_end), dim=1)
        coef_index = torch.stack((uni_index, uni_index, sal_index, uni_index), dim=1)
        bin_index = torch.stack((uni_index, uni_index, sal_uni_bin, uni_index), dim=1)
        level = torch.stack(
            (
                torch.full_like(uni_index, approx_level),
                torch.zeros_like(uni_index),
                sal_level,
                torch.zeros_like(uni_index),
            ),
            dim=1,
        )
        family = (
            torch.arange(N_FAMILIES, dtype=torch.int64, device=device)
            .view(1, N_FAMILIES, 1)
            .expand(channels, -1, quota)
        )
        valid_fraction = torch.stack(
            (bin_valid_fraction, bin_valid_fraction, torch.ones_like(bin_valid_fraction), bin_valid_fraction),
            dim=1,
        )

        n_per_channel = N_FAMILIES * quota
        # ---- anchored fusion inputs ----------------------------------------
        a_interp = torch.stack(
            [
                _interpolate_in_time(
                    a_pos, a_feat[channel], inputs.a_valid[channel], anchor_pos[channel].reshape(-1)
                )
                for channel in range(channels)
            ],
            dim=0,
        )  # [C, 4Q, 64]
        chan = torch.arange(channels, dtype=torch.int64, device=device)[:, None]
        flat_bins = bin_index.reshape(channels, -1)
        d_cov_ctx = d_cov_features[chan, flat_bins]
        raw_ctx = raw_bin_features[chan, flat_bins]
        fused_in = torch.cat(
            (own.reshape(channels, n_per_channel, width), a_interp, d_cov_ctx, raw_ctx), dim=-1
        ).reshape(-1, FUSION_INPUT_DIM)
        if self.position_fourier_freqs > 0:
            # va1 costruisce il token DA posizione e valore insieme (wavelet_encoder.py: in_dim = 1 + 2*n_pos_freqs
            # -> Linear -> GELU -> Linear).  Sommarle DOPO fuse_norm, come avevo fatto nel primo tentativo, mette
            # 32 dimensioni di ampiezza 1 sopra un vettore normalizzato: dominano, cambiano segno fra token
            # adiacenti, e il train del piolo e' crollato da 20/32 a 3/32 con tutto il resto identico.  Qui
            # entrano nell'INGRESSO, prima di ogni norma, e sono i pesi a decidere quanto contano.
            normalised = anchor_pos.reshape(-1).to(torch.float64) / float(recording_length)
            angle = normalised[:, None] * self.position_freqs[None, :]
            # l'argomento arriva a 2^15*pi radianti e in bf16 perderebbe ogni cifra: l'angolo si calcola in
            # float64, il seno sta in [-1, 1] e scende senza danno al dtype della fusione
            fourier = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1).to(fused_in.dtype)
            fused_in = torch.cat((fused_in, fourier), dim=-1)
        descriptors = self.fuse_norm(self.fuse_out(F.gelu(self.fuse_in(fused_in))))

        # ---- embeddings and numeric metadata ------------------------------
        n_tokens = channels * n_per_channel
        valid_flat = valid.reshape(-1)
        family_flat = family.reshape(-1)
        level_flat = level.reshape(-1)
        channel_flat = channel_index[:, None].expand(channels, n_per_channel).reshape(-1)
        kind_id = torch.full(
            (n_tokens,), 1 if is_local else 0, dtype=torch.int64, device=device
        )
        support_start_flat = support_start.reshape(-1)
        support_end_flat = support_end.reshape(-1)
        width_samples = (support_end_flat - support_start_flat).clamp_min(1).to(fdtype)
        log_global_scale = torch.log(view.global_stats.scale.to(fdtype))
        meta = torch.stack(
            (
                (anchor_pos.reshape(-1) / float(recording_length)).to(fdtype),
                torch.log2(width_samples),
                log_global_scale[:, None].expand(channels, n_per_channel).reshape(-1),
                inputs.log_scale_ratio[:, None].expand(channels, n_per_channel).reshape(-1),
                valid_fraction.reshape(-1),
            ),
            dim=-1,
        )
        descriptors = (
            descriptors
            + self.family_embedding(family_flat)
            + self.level_embedding(level_flat)
            + self.channel_embedding(channel_flat)
            + self.view_embedding(kind_id)
            + META_GAIN * self._project_meta(meta)
        )
        descriptors = mask_positions(descriptors, valid_flat)

        anchor_flat = torch.where(valid_flat, anchor_time.reshape(-1), torch.zeros_like(anchor_time.reshape(-1)))
        zero_i = torch.zeros_like(support_start_flat)
        support_start_flat = torch.where(valid_flat, support_start_flat, zero_i)
        support_end_flat = torch.where(valid_flat, support_end_flat, zero_i)
        coef_flat = coef_index.reshape(-1)
        sort_order = _lexsort(
            [
                (~valid_flat).to(torch.int64),
                anchor_flat,
                channel_flat,
                family_flat,
                level_flat,
                coef_flat,
            ]
        )

        return ViewTokens(
            descriptors=descriptors,
            valid=valid_flat,
            anchor_time=anchor_flat,
            support_start=support_start_flat,
            support_end=support_end_flat,
            channel_index=channel_flat,
            family=family_flat,
            level=level_flat,
            view_kind=kind,
            start=start,
            end=end,
            raw_bin_features=raw_bin_features,
            raw_bin_valid=raw_present,
            raw_bin_edges=edges,
            sort_order=sort_order,
            coefficient_index=coef_flat,
        )


class PerceptualFrontend(nn.Module):
    """Deep branches + tokenizer + auxiliary presence head (DESIGN 10.3)."""

    def __init__(self, spec: dict) -> None:
        super().__init__()
        self.branches = DeepBranches(spec)
        self.tokenizer = PerceptualTokenizer(self.branches, spec)
        self.aux_head = nn.Linear(self.branches.width, 1)

    def encode_view(self, view: SignalView, *, recording_length: int) -> ViewTokens:
        pyramid_global = build_haar_pyramid(view, trace="global")
        pyramid_local = (
            build_haar_pyramid(view, trace="local") if view.kind is ViewKind.LOCAL else None
        )
        return self.tokenizer(
            view,
            pyramid_global,
            pyramid_local,
            recording_length=recording_length,
            channel_indices=view.channel_indices,
        )

    def aux_logits(self, tokens: ViewTokens) -> Tensor:
        """Per raw uniform bin logits ``[C, Q]`` for the event-occupancy loss."""

        return self.aux_head(tokens.raw_bin_features).squeeze(-1)


__all__ = [
    "BranchInputs",
    "DESCRIPTOR_DIM",
    "FAMILY_A",
    "FAMILY_D_COVERAGE",
    "FAMILY_D_SALIENT",
    "FAMILY_NAMES",
    "FAMILY_RAW",
    "FUSION_INPUT_DIM",
    "META_DIM",
    "N_FAMILIES",
    "PerceptualFrontend",
    "PerceptualTokenizer",
    "SALIENT_PER_BIN",
    "ViewTokens",
    "build_branch_inputs",
    "pool_uniform_bins",
    "robust_band_scale",
    "select_salient_coefficients",
    "uniform_bin_edges",
]
