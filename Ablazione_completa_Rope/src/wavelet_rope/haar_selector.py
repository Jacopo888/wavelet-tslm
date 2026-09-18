"""Blind Haar proposals for agentic negative and recovery trajectories.

The selector is intentionally independent of the learned frontend and of the
dataset adapter.  It uses the same useful part of the W1 selector as the L3
reference: squared Haar details are normalised by a robust per-level energy
scale, accumulated in uniform absolute-time buckets, and ranked jointly over
all channels.  Gold intervals are accepted only by the training-time filter;
they are never needed by :func:`blind_candidates`.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor


MAX_LEVELS = 12
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _check_values(values: Tensor) -> tuple[int, int]:
    if not isinstance(values, Tensor) or values.ndim != 2:
        raise TypeError("episode_values must be a torch.Tensor with shape [C, L]")
    channels, length = map(int, values.shape)
    if channels < 1 or length < 1:
        raise ValueError("episode_values must have positive channel and time dimensions")
    if not (values.is_floating_point() or values.is_complex()):
        raise TypeError("episode_values must have a floating-point dtype")
    if values.is_complex():
        raise TypeError("episode_values must be real-valued")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("episode_values must be finite")
    return channels, length


def _n_levels(length: int) -> int:
    if length < 2:
        return 0
    # This is the L3 depth rule, with the project-wide safety cap.  The second
    # bound is important for very short odd recordings: a nominal level whose
    # support is larger than the recording is not part of the reference
    # pyramid.
    requested = max(1, math.ceil(math.log2(max(2.0, length / 128.0))))
    return min(MAX_LEVELS, requested, max(1, int(math.log2(length))))


def _median(values: Tensor) -> Tensor:
    """NumPy-compatible median using only torch operations.

    ``torch.median`` selects the lower middle value for an even-sized vector,
    while the L3 reference uses ``numpy.median`` (the mean of the two middle
    values).  Keeping this tiny helper explicit makes the port's ranking
    arithmetic agree on ties and near-ties as well as on ordinary inputs.
    """

    ordered = values.sort().values
    size = int(ordered.numel())
    if size == 0:
        raise ValueError("median requires a non-empty tensor")
    middle = size // 2
    if size % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) * 0.5


def _normalised_energy_scores(values: Tensor, bucket_count: int) -> Tensor:
    """Return ``[C, bucket_count]`` scores using a torch Haar transform."""

    channels, length = map(int, values.shape)
    # The l3 implementation caps the requested number of buckets at L.  This
    # matters for short recordings where the minimum of eight would otherwise
    # create repeated integer edges.
    edges = (torch.arange(bucket_count + 1, dtype=torch.int64, device=values.device) * length) // bucket_count
    scores = torch.zeros((channels, bucket_count), dtype=torch.float64, device=values.device)
    approximation = values.to(torch.float64)
    levels = _n_levels(length)
    for level in range(1, levels + 1):
        if approximation.shape[-1] & 1:
            approximation = torch.cat((approximation, approximation[..., -1:]), dim=-1)
        even = approximation[..., 0::2]
        odd = approximation[..., 1::2]
        detail = (even - odd) * _INV_SQRT2
        approximation = (even + odd) * _INV_SQRT2

        energy = detail.square()
        # L3 applies its robust scale independently for every channel and
        # level.  Keeping that axis is important for the promised ranking
        # across channels: a high-amplitude channel must not set the scale for
        # a quiet channel.
        scales: list[Tensor] = []
        for channel in range(channels):
            positive = energy[channel][energy[channel] > 0]
            if positive.numel():
                scales.append(_median(positive).clamp_min(torch.finfo(torch.float64).eps))
            else:
                scales.append(torch.ones((), dtype=torch.float64, device=values.device))
        scale = torch.stack(scales)

        nominal_width = 2**level
        centers = torch.arange(detail.shape[-1], dtype=torch.int64, device=values.device) * nominal_width
        centers = centers + nominal_width // 2
        valid = centers < length
        if not bool(valid.any()):
            continue
        centers = centers[valid]
        energies = (energy[:, valid] / scale[:, None]).to(torch.float64)
        bucket_ids = torch.searchsorted(edges, centers, right=True) - 1
        keep = (bucket_ids >= 0) & (bucket_ids < bucket_count)
        if not bool(keep.any()):
            continue
        bucket_ids = bucket_ids[keep]
        energies = energies[:, keep]
        for channel in range(channels):
            scores[channel].index_add_(0, bucket_ids, energies[channel])
    return scores


def blind_candidates(
    episode_values: Tensor, width: int, count: int
) -> list[tuple[int, int]]:
    """Rank Haar-energy bucket centres jointly across channels.

    Returns ``(channel_index, centre)`` pairs.  Ties are resolved by the flat
    channel-major order, so the result is reproducible on CPU and does not
    depend on a Python or NumPy sort implementation.
    """

    _channels, length = _check_values(episode_values)
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("width must be a positive integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if count == 0:
        return []
    width = min(int(width), length)
    bucket_count = min(length, max(8, math.ceil(length / max(1, width))))
    scores = _normalised_energy_scores(episode_values, bucket_count)
    edges = (torch.arange(bucket_count + 1, dtype=torch.int64, device=episode_values.device) * length) // bucket_count
    centres = (edges[:-1] + edges[1:]) // 2

    # Stable descending argsort preserves channel-major/bucket-major order on
    # equal scores.  This is the explicit deterministic tie break.
    flat_order = torch.argsort(scores.reshape(-1), descending=True, stable=True)
    take = min(int(count), int(flat_order.numel()))
    out: list[tuple[int, int]] = []
    for flat in flat_order[:take].tolist():
        channel, bucket = divmod(int(flat), bucket_count)
        out.append((channel, int(centres[bucket])))
    return out


def window_containing_point(
    length: int, point: int, width: int, rng: random.Random
) -> tuple[int, int]:
    """Sample uniformly from in-bounds windows of ``width`` containing ``point``."""

    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("length must be a positive integer")
    if isinstance(point, bool) or not isinstance(point, int) or not 0 <= point < length:
        raise ValueError("point must be an integer inside the recording")
    if isinstance(width, bool) or not isinstance(width, int) or not 0 < width <= length:
        raise ValueError("width must satisfy 0 < width <= length")
    if not hasattr(rng, "randint"):
        raise TypeError("rng must provide randint")
    low = max(0, point - width + 1)
    high = min(point, length - width)
    if low > high:  # defensive fallback for unusual integer-like inputs
        low = high = max(0, min(point, length - width))
    start = int(rng.randint(low, high))
    return start, start + width


def _interval_parts(item: Any) -> tuple[str, int, int] | None:
    if hasattr(item, "channel") and hasattr(item, "start") and hasattr(item, "end"):
        return str(item.channel), int(item.start), int(item.end)
    if isinstance(item, dict) and {"channel", "start", "end"}.issubset(item):
        return str(item["channel"]), int(item["start"]), int(item["end"])
    return None


def _candidate_geometry(candidate: Any, width: int | None) -> tuple[Any, int, int | None]:
    """Decode common candidate forms while retaining the public 2-tuple form."""

    if hasattr(candidate, "start") and hasattr(candidate, "end"):
        if hasattr(candidate, "channel"):
            channel = candidate.channel
        elif hasattr(candidate, "channels") and candidate.channels:
            channel = candidate.channels[0]
        else:
            channel = None
        return channel, int(candidate.start), int(candidate.end)
    if isinstance(candidate, dict) and {"start", "end"}.issubset(candidate):
        channel = candidate.get("channel", candidate.get("channel_index"))
        return channel, int(candidate["start"]), int(candidate["end"])
    if hasattr(candidate, "channel_index") and (hasattr(candidate, "centre") or hasattr(candidate, "center")):
        channel = candidate.channel_index
        centre = int(candidate.centre if hasattr(candidate, "centre") else candidate.center)
        candidate_width = getattr(candidate, "width", width)
        if candidate_width is None:
            return channel, centre, None
        return channel, centre - int(candidate_width) // 2, centre + int(candidate_width) - int(candidate_width) // 2
    if not isinstance(candidate, (tuple, list)) or len(candidate) < 2:
        raise ValueError(f"invalid candidate {candidate!r}")
    channel = candidate[0]
    if len(candidate) == 2:
        if isinstance(candidate[1], (tuple, list)) and len(candidate[1]) == 2:
            start, end = int(candidate[1][0]), int(candidate[1][1])
            return channel, start, end
        centre = int(candidate[1])
        if width is None:
            return channel, centre, None
        width = int(width)
        return channel, centre - width // 2, centre - width // 2 + width
    # Also accept (channel, start, end), which is convenient for callers that
    # have already sampled a candidate window.
    start, end = int(candidate[1]), int(candidate[2])
    return channel, start, end


def gold_free_windows(
    candidates: Iterable[Any],
    gold_intervals: Sequence[Any],
    channels: Sequence[str],
    *,
    width: int | None = None,
) -> list[Any]:
    """Keep candidates whose window is disjoint from all requested gold spans.

    The required public candidate representation contains a centre only.  In
    that form the centre is the tested point (equivalent to a one-sample
    window); callers that want exact window overlap can pass ``width=`` or use
    ``(channel, start, end)`` candidates.  ``channels`` names the complete set
    of channels in the prospective tool call, so a candidate is rejected when
    *any* requested channel has an overlapping annotation.  Candidates are
    returned unchanged, making this a filter rather than a geometry rewrite.
    """

    requested = tuple(str(name) for name in channels)
    if not requested:
        raise ValueError("channels must not be empty")
    if width is not None and (isinstance(width, bool) or not isinstance(width, int) or width < 1):
        raise ValueError("width must be a positive integer when provided")
    spans = []
    requested_set = set(requested)
    for item in gold_intervals:
        parsed = _interval_parts(item)
        if parsed is None:
            raise TypeError(f"gold interval is not a channel/start/end record: {item!r}")
        channel, start, end = parsed
        if channel in requested_set and start < end:
            spans.append((start, end))

    kept: list[Any] = []
    for candidate in candidates:
        channel, start, end = _candidate_geometry(candidate, width)
        # The candidate channel is intentionally not remapped here:
        # ``channels`` governs which annotations protect the prospective
        # window, including the other channel of a prospective multichannel
        # call.
        if end is None:
            overlap = any(start >= lo and start < hi for lo, hi in spans)
        else:
            overlap = any(max(start, lo) < min(end, hi) for lo, hi in spans)
        if not overlap:
            kept.append(candidate)
    return kept


__all__ = [
    "blind_candidates",
    "gold_free_windows",
    "window_containing_point",
]
