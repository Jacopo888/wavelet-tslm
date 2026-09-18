"""Programmatic *reading* of a view at the resolution the frontend actually represents (Astra 15/09, sentinella D4).

The reading names WHERE Haar detail energy concentrates (bin ranges in absolute sample coordinates at the view's bin
resolution), how channels compare by detail energy, and in which bin each leading channel deviates most from its own
median.  It deliberately never states extremes, medians or scales (copyable from the header), never gives
sample-exact positions (not observable at token resolution) and never reads gold.  Deterministic, pure torch,
CPU-testable.  Used as the supervised ``<think>`` segment before the final answer in P1-c (reading weight 0.5).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from .haar_selector import _normalised_energy_scores
from .signal_contract import SignalEpisode, compute_recording_normalization

MAD_TO_SIGMA = 1.4826
TOP_BUCKETS = 3
MAX_LISTED_CHANNELS = 4


def view_reading(
    values: Tensor,
    valid: Tensor,
    channel_ids: Sequence[str],
    *,
    start: int,
    n_bins: int,
    kind: str = "global",
    fmt: str = "verbose",
) -> str:
    """Reading of one view.

    ``values``: ``[C, L]`` normalised trace of the view (recording normalisation, as the global branch sees it);
    ``valid``: ``[C, L]`` bool; ``start``: absolute offset of the view; ``n_bins``: uniform bins per channel the
    tokenizer represents for this view kind (64 global, 128 local in the pinned spec).
    """

    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("values and valid must both have shape [C, L]")
    channels, length = map(int, values.shape)
    if channels != len(channel_ids):
        raise ValueError("channel_ids must match the number of channels")
    if length == 0:
        return "top none; peak none." if fmt == "compact" else "The view contains no samples."
    n_bins = max(1, min(int(n_bins), length))
    edges = [(index * length) // n_bins for index in range(n_bins + 1)]
    filled = torch.where(valid, values, torch.zeros_like(values)).to(torch.float32)
    with torch.no_grad():
        scores = _normalised_energy_scores(filled, n_bins)  # [C, n_bins] float64
    total = scores.sum(dim=1)
    order = sorted(range(channels), key=lambda c: (-float(total[c]), c))

    def span(bucket: int) -> str:
        return f"[{start + edges[bucket]},{start + edges[bucket + 1]})"

    flat = torch.argsort(scores.reshape(-1), descending=True, stable=True)
    top: list[tuple[int, int]] = []
    for flat_index in flat.tolist():
        channel, bucket = divmod(int(flat_index), n_bins)
        if float(scores[channel, bucket]) <= 0.0:
            break
        top.append((channel, bucket))
        if len(top) >= TOP_BUCKETS:
            break

    if fmt == "compact":
        # Astra 15/09 D2: bin INDICES on the grid declared in the prompt; channel identity, top bins and peak only
        # (no ranking).  ``top ch_00:62,61,60; peak ch_00:62.``
        grouped: dict[int, list[int]] = {}
        for c, b in top:
            grouped.setdefault(c, []).append(b)
        top_text = ", ".join(f"{channel_ids[c]}:" + ",".join(str(b) for b in bins) for c, bins in grouped.items()) or "none"
        peaks: list[str] = []
        for channel in order[: min(2, channels)]:
            mask = valid[channel]
            if int(mask.sum()) == 0:
                continue
            observed = filled[channel][mask]
            median = observed.median()
            mad = (observed - median).abs().median()
            scale = torch.maximum(mad * MAD_TO_SIGMA, torch.tensor(1e-6, dtype=observed.dtype))
            deviation = torch.where(mask, (filled[channel] - median).abs() / scale, torch.zeros_like(filled[channel]))
            peaks.append(f"{channel_ids[channel]}:{min(n_bins - 1, (int(deviation.argmax()) * n_bins) // length)}")
        return f"top {top_text}; peak {', '.join(peaks) or 'none'}."
    label = "Global view" if kind == "global" else "Local view"
    clauses = [f"{label} read at {math.ceil(length / n_bins)} samples per bin"]
    if top:
        clauses.append(
            "Haar detail energy concentrates in "
            + ", ".join(f"{channel_ids[c]} {span(b)}" for c, b in top)
        )
    else:
        clauses.append("no Haar detail energy stands out")
    if channels > 1:
        listed = order[:MAX_LISTED_CHANNELS]
        ranking = " > ".join(channel_ids[c] for c in listed)
        if channels > MAX_LISTED_CHANNELS:
            ranking += f" > ... > {channel_ids[order[-1]]}"
        clauses.append(f"channels ranked by detail energy: {ranking}")
    deviation_clauses: list[str] = []
    for channel in order[: min(2, channels)]:
        mask = valid[channel]
        if int(mask.sum()) == 0:
            continue
        observed = filled[channel][mask]
        median = observed.median()
        mad = (observed - median).abs().median()
        scale = torch.maximum(mad * MAD_TO_SIGMA, torch.tensor(1e-6, dtype=observed.dtype))
        deviation = torch.where(mask, (filled[channel] - median).abs() / scale, torch.zeros_like(filled[channel]))
        index = int(deviation.argmax())
        bucket = min(n_bins - 1, (index * n_bins) // length)
        deviation_clauses.append(f"{channel_ids[channel]} deviates most in {span(bucket)}")
    if deviation_clauses:
        clauses.append("; ".join(deviation_clauses))
    return ". ".join(clauses) + "."


def episode_reading(
    episode: SignalEpisode,
    *,
    channels: Sequence[str] | None = None,
    start: int = 0,
    end: int | None = None,
    n_bins: int,
    kind: str = "global",
    fmt: str = "verbose",
) -> str:
    """Reading of ``episode`` restricted to ``channels`` and ``[start, end)`` under the recording normalisation.

    A local view (teacher crop or executed zoom) passes its channels and window; the global view passes nothing.
    """

    stats = compute_recording_normalization(episode)
    ids = list(episode.channel_ids)
    selected = ids if channels is None else list(channels)
    positions = [ids.index(c) for c in selected]
    stop = episode.length if end is None else int(end)
    if not 0 <= int(start) < stop <= episode.length:
        raise ValueError("reading window out of range")
    index = torch.as_tensor(positions, dtype=torch.int64, device=episode.values.device)
    values = episode.values.index_select(0, index)[:, int(start):stop].to(torch.float32)
    valid = episode.valid.index_select(0, index)[:, int(start):stop]
    median = stats.median.to(values.device, torch.float32).index_select(0, index)[:, None]
    scale = stats.scale.to(values.device, torch.float32).index_select(0, index)[:, None]
    normalised = torch.where(valid, (values - median) / scale, torch.zeros_like(values))
    return view_reading(normalised.cpu(), valid.cpu(), selected, start=int(start), n_bins=n_bins, kind=kind, fmt=fmt)


__all__ = ["view_reading", "episode_reading"]
