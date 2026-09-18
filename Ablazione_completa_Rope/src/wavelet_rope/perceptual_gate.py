"""Perceptual-content gate for WR-Agent-4B-Wildcard-v1.

The gate deliberately lives outside the model and the trainer.  It freezes a
panel of absolute-coordinate crops, creates three complete signal episodes for
each crop, and compares the official precise scorer on the resulting answers.
The only annotation-dependent operations in this module are panel construction
and the deterministic choice of substitution windows; annotations are never
passed to :meth:`WRAgentModel.generate`.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from .data import GoldInterval, RowMeta, ZoomAction, stratified_subset
from .scoring import REQUIRED_KEYS, parse_structured_pred, score_precise
from .signal_contract import (
    SignalEpisode,
    apply_normalization,
    compute_recording_normalization,
)
from .templates import ViewHeader


CONDITIONS: tuple[str, ...] = ("TRUE", "SUBSTITUTED", "ZEROED")
TARGET_CATEGORIES: frozenset[str] = frozenset(
    {"localize", "classify_with_evidence", "localize_all_channels"}
)
DEFAULT_CROPS_PER_ROW = 3
DEFAULT_K = DEFAULT_CROPS_PER_ROW
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_COLLAPSE_TOLERANCE = 0.05
DEFAULT_INVALID_RATE_LIMIT = 0.01
ZERO_TOLERANCE = 1e-9  # Astra 15/09: a main category whose TRUE mean is zero vetoes the gate


def _stable_seed(*parts: object) -> int:
    """Derive a reproducible integer without depending on Python hash randomisation."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & ((1 << 63) - 1)


@dataclass(frozen=True, slots=True)
class SubstitutionSource:
    """One source window used for one channel of a panel crop."""

    start: int
    end: int
    overlap: int = 0
    gold_free: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": int(self.start),
            "end": int(self.end),
            "overlap": int(self.overlap),
            "gold_free": bool(self.gold_free),
        }

    @classmethod
    def from_value(cls, value: object) -> "SubstitutionSource":
        if isinstance(value, SubstitutionSource):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("a substitution source must be a mapping")
        start = int(value["start"])
        end = int(value["end"])
        overlap = int(value.get("overlap", 0))
        gold_free = bool(value.get("gold_free", overlap == 0))
        if not 0 <= start < end:
            raise ValueError("a substitution source must be a non-empty interval")
        return cls(start=start, end=end, overlap=overlap, gold_free=gold_free)


@dataclass(frozen=True, slots=True)
class PanelCrop:
    """A frozen crop and its per-channel substitution source windows."""

    row_id: str
    row_index: int
    category: str
    channels: tuple[str, ...]
    start: int
    end: int
    crop_kind: str
    substitution_sources: Mapping[str, SubstitutionSource] = field(default_factory=dict)

    @property
    def recording_id(self) -> str:
        return self.row_id

    @property
    def substitution_source_window(self) -> Mapping[str, SubstitutionSource]:
        return self.substitution_sources

    @property
    def substitution_source(self) -> Mapping[str, SubstitutionSource]:
        return self.substitution_sources

    @property
    def length(self) -> int:
        return self.end - self.start

    def action(self) -> ZoomAction:
        return ZoomAction(channels=self.channels, start=self.start, end=self.end)

    def to_dict(self) -> dict[str, Any]:
        sources = {
            channel: SubstitutionSource.from_value(self.substitution_sources[channel]).to_dict()
            for channel in self.channels
            if channel in self.substitution_sources
        }
        # ``substitution_source_window`` is the descriptive name used by the
        # protocol; ``substitution_source`` is retained as a compact alias for
        # callers that used the first draft of the panel format.
        return {
            "row_id": self.row_id,
            "recording_id": self.row_id,
            "row_index": int(self.row_index),
            "category": self.category,
            "channels": list(self.channels),
            "start": int(self.start),
            "end": int(self.end),
            "crop_kind": self.crop_kind,
            "substitution_source_window": sources,
            "substitution_source": sources,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PanelCrop":
        row_id = str(value.get("row_id", value.get("recording_id", "")))
        if not row_id:
            raise ValueError("panel crop is missing row_id")
        channels = tuple(str(item) for item in value["channels"])
        raw_sources = value.get("substitution_source_window", value.get("substitution_source", {}))
        if raw_sources is None:
            raw_sources = {}
        if not isinstance(raw_sources, Mapping):
            raise ValueError("substitution source windows must be a mapping by channel")
        sources = {
            str(channel): SubstitutionSource.from_value(source)
            for channel, source in raw_sources.items()
        }
        return cls(
            row_id=row_id,
            row_index=int(value.get("row_index", -1)),
            category=str(value["category"]),
            channels=channels,
            start=int(value["start"]),
            end=int(value["end"]),
            crop_kind=str(value["crop_kind"]),
            substitution_sources=sources,
        )


def panel_payload(
    crops: Sequence[PanelCrop], *, seed: int | None = None, crops_per_row: int | None = None
) -> dict[str, Any]:
    """Return the serialisable panel document."""

    payload: dict[str, Any] = {
        "version": 1,
        "crops": [crop.to_dict() for crop in crops],
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if crops_per_row is not None:
        payload["crops_per_row"] = int(crops_per_row)
    return payload


def save_panel(
    path: str | Path,
    crops: Sequence[PanelCrop],
    *,
    seed: int | None = None,
    crops_per_row: int | None = None,
) -> None:
    """Write a deterministic panel JSON document."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(panel_payload(crops, seed=seed, crops_per_row=crops_per_row), indent=2) + "\n",
        encoding="utf-8",
    )


def load_panel(path: str | Path) -> list[PanelCrop]:
    """Load and validate the crop list from a panel JSON document."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, Mapping):
        entries = raw.get("crops")
    else:
        entries = None
    if not isinstance(entries, list):
        raise ValueError("panel JSON must contain a crops list")
    return [PanelCrop.from_dict(item) for item in entries]


read_panel = load_panel


# --------------------------------------------------------------------------- panel construction


def _usable_gold(meta: RowMeta, channel: str | None = None) -> tuple[GoldInterval, ...]:
    names = set(meta.channel_names)
    return tuple(
        interval
        for interval in meta.gold_intervals
        if interval.channel in names
        and (channel is None or interval.channel == channel)
        and 0 <= interval.start < interval.end <= meta.length
    )


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted((int(a), int(b)) for a, b in intervals if a < b):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _overlap_length(start: int, end: int, spans: Sequence[tuple[int, int]]) -> int:
    return sum(max(0, min(end, high) - max(start, low)) for low, high in spans if low < end and high > start)


def _free_start_ranges(
    length: int, width: int, spans: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Inclusive start ranges whose window does not overlap any span."""

    if width < 1 or width > length:
        return []
    max_start = length - width
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for low, high in _merge_intervals(spans):
        if cursor + width <= low:
            ranges.append((cursor, min(max_start, low - width)))
        cursor = max(cursor, high)
    if cursor + width <= length:
        ranges.append((cursor, max_start))
    return [(low, high) for low, high in ranges if low <= high]


def _pick_from_ranges(ranges: Sequence[tuple[int, int]], rng: random.Random) -> int:
    total = sum(high - low + 1 for low, high in ranges)
    if total <= 0:
        raise ValueError("no selectable start in the supplied ranges")
    offset = rng.randrange(total)
    for low, high in ranges:
        count = high - low + 1
        if offset < count:
            return low + offset
        offset -= count
    raise AssertionError("range selection fell through")  # pragma: no cover


def _not_centred(start: int, width: int, interval: GoldInterval) -> bool:
    crop_center = (2 * start + width - 1) / 2.0
    event_center = (interval.start + interval.end - 1) / 2.0
    return not math.isclose(crop_center, event_center, rel_tol=0.0, abs_tol=1e-9)


def _pick_overlap_start(
    interval: GoldInterval,
    width: int,
    length: int,
    rng: random.Random,
    *,
    prefer_full: bool,
) -> int:
    max_start = length - width
    overlap = [(max(0, interval.start - width + 1), min(max_start, interval.end - 1))]
    full: list[tuple[int, int]] = []
    if interval.end - interval.start <= width:
        lo = max(0, interval.end - width)
        hi = min(max_start, interval.start)
        if lo <= hi:
            full.append((lo, hi))

    ranges = full if prefer_full and full else overlap
    candidates: list[tuple[int, int]] = []
    for low, high in ranges:
        if low <= high:
            candidates.append((low, high))
    if not candidates:
        candidates = overlap

    # Rejection sampling avoids materialising every possible start for long
    # recordings while making the centred position impossible when alternatives
    # exist.
    for _ in range(64):
        start = _pick_from_ranges(candidates, rng)
        if _not_centred(start, width, interval):
            return start
    for low, high in candidates:
        for start in (low, low + 1, high - 1, high):
            if low <= start <= high and _not_centred(start, width, interval):
                return start
    # A recording whose only admissible crop is exactly centred is a geometric
    # impossibility, not a randomisation failure.  Keep the crop valid.
    return _pick_from_ranges(candidates, rng)


def _row_channels(meta: RowMeta, interval: GoldInterval | None, rng: random.Random) -> tuple[str, ...]:
    if meta.n_channels <= 1:
        return (meta.channel_names[0],)
    if interval is None:
        chosen = rng.sample(list(meta.channel_names), 2)
        return tuple(name for name in meta.channel_names if name in chosen)
    # Every multichannel positive crop is simultaneous and 2-channel.  Prefer a
    # second annotated channel to expose misaligned cases; otherwise pair the
    # event channel with a clean channel.
    others = [name for name in meta.channel_names if name != interval.channel]
    annotated = [name for name in others if _usable_gold(meta, name)]
    second = rng.choice(annotated if annotated else others)
    return tuple(name for name in meta.channel_names if name in {interval.channel, second})


def _crop_kind(meta: RowMeta, channels: Sequence[str], start: int, end: int) -> str:
    overlap = False
    full = False
    for interval in _usable_gold(meta):
        if interval.channel not in channels:
            continue
        amount = max(0, min(end, interval.end) - max(start, interval.start))
        if amount:
            overlap = True
            if interval.start >= start and interval.end <= end:
                full = True
    if not overlap:
        return "negative"
    return "positive_full" if full else "positive_partial"


def _sample_width(length: int, rng: random.Random, *, low: int = 256, high: int = 8192) -> int:
    low = min(int(low), length)
    high = min(int(high), length)
    if low < 1 or high < low:
        raise ValueError("invalid crop width limits")
    if low == high:
        return high
    return max(low, min(high, int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))))


def _sample_negative(
    meta: RowMeta, channels: Sequence[str], width: int, rng: random.Random
) -> tuple[int, int] | None:
    # A panel negative is genuinely event-free in the recording.  This is a
    # little stricter than only checking the selected channels and prevents a
    # negative label from depending on which channel pair happened to be drawn.
    spans = [(interval.start, interval.end) for interval in _usable_gold(meta)]
    ranges = _free_start_ranges(meta.length, width, spans)
    if not ranges:
        return None
    start = _pick_from_ranges(ranges, rng)
    return start, start + width


def select_substitution_source(
    meta: RowMeta,
    channel: str,
    width: int,
    seed: int,
) -> SubstitutionSource:
    """Choose a same-channel window, preferring a gold-free window.

    Exact free windows are selected uniformly over their integer start
    positions.  If no such window exists, the candidates at all annotation
    boundaries are compared and a minimum-overlap candidate is selected
    deterministically.
    """

    if channel not in meta.channel_names:
        raise ValueError(f"unknown channel {channel!r} in row {meta.id!r}")
    if not 1 <= int(width) <= meta.length:
        raise ValueError("substitution width must fit inside the recording")
    width = int(width)
    rng = random.Random(_stable_seed("substitution", seed, meta.id, channel, width))
    spans = [(item.start, item.end) for item in _usable_gold(meta, channel)]
    free = _free_start_ranges(meta.length, width, spans)
    if free:
        start = _pick_from_ranges(free, rng)
        return SubstitutionSource(start, start + width, overlap=0, gold_free=True)

    merged = _merge_intervals(spans)
    max_start = meta.length - width
    candidates = {0, max_start}
    for low, high in merged:
        candidates.update(
            max(0, min(max_start, candidate))
            for candidate in (low - width, low, high - width, high)
        )
    scored = [
        (start, _overlap_length(start, start + width, merged))
        for start in sorted(candidates)
    ]
    best_overlap = min(overlap for _, overlap in scored)
    best = [start for start, overlap in scored if overlap == best_overlap]
    start = best[rng.randrange(len(best))]
    return SubstitutionSource(start, start + width, overlap=best_overlap, gold_free=best_overlap == 0)


def substitution_sources(meta: RowMeta, crop: PanelCrop, *, seed: int) -> dict[str, SubstitutionSource]:
    return {
        channel: select_substitution_source(meta, channel, crop.length, _stable_seed(seed, crop.row_id, crop.start, crop.end, channel))
        for channel in crop.channels
    }


select_substitution_window = select_substitution_source


def build_panel(
    metas: Sequence[RowMeta],
    *,
    seed: int,
    rows: int | None = None,
    crops_per_row: int = DEFAULT_CROPS_PER_ROW,
    negative_share: float = 1.0 / DEFAULT_CROPS_PER_ROW,
    k: int | None = None,
) -> list[PanelCrop]:
    """Build the deterministic validation panel from row metadata only.

    The eligible pool is the three specified precise categories, including
    their no-gold rows.  Positive rows receive event-overlapping crops and, if
    possible, at least one negative crop; no-gold rows are negative crops.
    """

    if k is not None:
        crops_per_row = k
    if isinstance(crops_per_row, bool) or crops_per_row < 1:
        raise ValueError("crops_per_row must be positive")
    if not 0.0 <= float(negative_share) <= 1.0:
        raise ValueError("negative_share must lie in [0, 1]")
    if hasattr(metas, "metas"):
        metas = metas.metas()  # type: ignore[assignment,union-attr]
    metas = list(metas)
    eligible = [meta for meta in metas if meta.category in TARGET_CATEGORIES]
    if rows is not None:
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ValueError("rows must be a non-negative integer or None")
        selected = set(stratified_subset(eligible, rows, seed)) if rows else set()
        if rows:
            eligible = [meta for meta in eligible if meta.row_index in selected]
    eligible.sort(key=lambda meta: meta.row_index)

    panel: list[PanelCrop] = []
    for meta in eligible:
        gold = _usable_gold(meta)
        if crops_per_row < 2 or negative_share <= 0.0:
            negative_crops = 0
        else:
            negative_crops = max(1, min(crops_per_row, int(round(crops_per_row * negative_share))))
        for crop_index in range(crops_per_row):
            rng = random.Random(_stable_seed("panel", seed, meta.id, meta.row_index, crop_index))
            width = _sample_width(meta.length, rng)
            if gold:
                # Cycling annotated intervals guarantees coverage of the two
                # sides of a genuinely misaligned multichannel row when K>=2;
                # the crop position and width remain seeded-random.
                interval = gold[crop_index % len(gold)]
            else:
                interval = None
            channels = _row_channels(meta, interval, rng)

            want_negative = not gold or crop_index >= crops_per_row - negative_crops
            negative = _sample_negative(meta, channels, width, rng) if want_negative else None
            if negative is not None:
                start, end = negative
            elif interval is not None:
                start = _pick_overlap_start(
                    interval,
                    width,
                    meta.length,
                    rng,
                    prefer_full=(rng.random() >= 0.2),
                )
                end = start + width
            else:
                # A no-gold row can only reach this branch when every possible
                # window was rejected by a malformed metadata length.
                start = rng.randint(0, meta.length - width)
                end = start + width

            kind = _crop_kind(meta, channels, start, end)
            if gold and kind == "negative" and negative is None:
                # A crop sampled as a negative on a positive row is allowed;
                # otherwise enforce that every positive-labelled crop actually
                # overlaps the selected event.
                if interval is not None:
                    start = _pick_overlap_start(interval, width, meta.length, rng, prefer_full=False)
                    end = start + width
                    kind = _crop_kind(meta, channels, start, end)
            provisional = PanelCrop(
                row_id=meta.id,
                row_index=meta.row_index,
                category=meta.category,
                channels=tuple(channels),
                start=int(start),
                end=int(end),
                crop_kind=kind,
                substitution_sources={},
            )
            panel.append(
                PanelCrop(
                    row_id=provisional.row_id,
                    row_index=provisional.row_index,
                    category=provisional.category,
                    channels=provisional.channels,
                    start=provisional.start,
                    end=provisional.end,
                    crop_kind=provisional.crop_kind,
                    substitution_sources=substitution_sources(meta, provisional, seed=seed),
                )
            )
    return panel


# --------------------------------------------------------------------------- perturbations


def _source_for_channel(
    source_map: Mapping[str, SubstitutionSource] | None,
    channel: str,
    *,
    meta: RowMeta,
    crop: PanelCrop,
    seed: int,
) -> SubstitutionSource:
    if source_map is not None and channel in source_map:
        return SubstitutionSource.from_value(source_map[channel])
    if channel in crop.substitution_sources:
        return SubstitutionSource.from_value(crop.substitution_sources[channel])
    return select_substitution_source(meta, channel, crop.length, _stable_seed(seed, channel))


def perturb_episode(
    episode: SignalEpisode,
    meta: RowMeta,
    crop: PanelCrop,
    condition: str,
    *,
    source_map: Mapping[str, SubstitutionSource] | None = None,
    seed: int = 0,
) -> SignalEpisode:
    """Return a new episode for TRUE, SUBSTITUTED, or ZEROED.

    Only selected channels inside the crop are changed; the valid mask, channel
    order, absolute time grid, and task are preserved.  ZEROED uses the true
    recording medians, so its globally-normalized values are exactly zero when
    evaluated with the true recording statistics.
    """

    condition = str(condition).upper()
    if condition not in CONDITIONS:
        raise ValueError(f"unknown perceptual-gate condition {condition!r}")
    if episode.length != meta.length or episode.channel_ids != meta.channel_names:
        raise ValueError("episode and row metadata do not describe the same recording")
    if not 0 <= crop.start < crop.end <= episode.length:
        raise ValueError("panel crop lies outside the recording")
    if tuple(dict.fromkeys(crop.channels)) != crop.channels:
        raise ValueError("panel crop channels must be unique")

    values = episode.values.clone()
    valid = episode.valid.clone()
    stats = compute_recording_normalization(episode)
    lookup = {channel: index for index, channel in enumerate(episode.channel_ids)}
    original = episode.values
    for channel in crop.channels:
        if channel not in lookup:
            raise ValueError(f"panel crop references unknown channel {channel!r}")
        index = lookup[channel]
        target = slice(crop.start, crop.end)
        if condition == "ZEROED":
            values[index, target] = stats.median[index]
        elif condition == "SUBSTITUTED":
            source = _source_for_channel(source_map, channel, meta=meta, crop=crop, seed=seed)
            if source.end - source.start != crop.length:
                raise ValueError("substitution source length does not match the crop")
            if not 0 <= source.start < source.end <= episode.length:
                raise ValueError("substitution source lies outside the recording")
            replacement = original[index, source.start : source.end]
            target_valid = episode.valid[index, target]
            source_finite = torch.isfinite(replacement)
            if bool(target_valid.any()) and not bool(source_finite[target_valid].all()):
                # Missing source samples have no sensory value.  Fill only the
                # positions that remain observed under the frozen target mask,
                # using the true global median prescribed by Astra's missing
                # data decision; never change ``valid`` or invent an event.
                replacement = torch.where(
                    target_valid & ~source_finite,
                    stats.median[index],
                    replacement,
                )
            values[index, target] = replacement

    dt = episode.dt if episode.dt is not None else 1.0
    return SignalEpisode.from_index_grid(
        values=values,
        valid=valid,
        channel_ids=episode.channel_ids,
        dt=dt,
        origin=float(episode.sample_times[0]),
        task=episode.task,
    )


build_perturbed_episode = perturb_episode
build_substituted_episode = lambda episode, meta, crop, **kwargs: perturb_episode(
    episode, meta, crop, "SUBSTITUTED", **kwargs
)
build_zeroed_episode = lambda episode, meta, crop, **kwargs: perturb_episode(
    episode, meta, crop, "ZEROED", **kwargs
)
make_perturbed_episode = perturb_episode


def globally_normalized(episode: SignalEpisode, reference_stats=None) -> torch.Tensor:
    """Normalize an episode, optionally with frozen true-recording statistics."""

    return apply_normalization(
        episode.values,
        episode.valid,
        compute_recording_normalization(episode) if reference_stats is None else reference_stats,
    )


# --------------------------------------------------------------------------- bootstrap and verdict


def _record_id(record: Mapping[str, Any]) -> str:
    for key in ("recording_id", "row_id", "id"):
        if key in record:
            return str(record[key])
    raise KeyError("result record has no recording id")


def _record_category(record: Mapping[str, Any]) -> str:
    return str(record.get("category", ""))


def _record_delta(record: Mapping[str, Any]) -> float:
    if "delta" in record:
        return float(record["delta"])
    primary = record.get("primary")
    if isinstance(primary, Mapping):
        true = float(primary["TRUE"])
        control = max(float(primary["SUBSTITUTED"]), float(primary["ZEROED"]))
        return true - control
    nested = {
        condition: record.get(condition)
        for condition in CONDITIONS
        if isinstance(record.get(condition), Mapping)
        and "primary" in record[condition]
    }
    if len(nested) == len(CONDITIONS):
        return float(nested["TRUE"]["primary"]) - max(
            float(nested["SUBSTITUTED"]["primary"]),
            float(nested["ZEROED"]["primary"]),
        )
    keys = {
        "TRUE": ("primary_true", "true_primary", "TRUE"),
        "SUBSTITUTED": ("primary_substituted", "substituted_primary", "SUBSTITUTED"),
        "ZEROED": ("primary_zeroed", "zeroed_primary", "ZEROED"),
    }
    values = {}
    for condition, names in keys.items():
        for name in names:
            if name in record:
                values[condition] = float(record[name])
                break
    if len(values) == len(CONDITIONS):
        return values["TRUE"] - max(values["SUBSTITUTED"], values["ZEROED"])
    raise KeyError("result record has no delta or three condition primary scores")


def aggregate_deltas_by_recording(
    records: Mapping[str, float] | Sequence[Mapping[str, Any]], *, category: str | None = None
) -> dict[str, float]:
    """Average crop deltas within each recording before any bootstrap."""

    if isinstance(records, Mapping):
        return {str(key): float(value) for key, value in records.items()}
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if category is not None and _record_category(record) != category:
            continue
        grouped[_record_id(record)].append(_record_delta(record))
    return {key: float(sum(values) / len(values)) for key, values in sorted(grouped.items()) if values}


def paired_bootstrap(
    values: Mapping[str, float] | Sequence[float],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap cluster means, where each input value is already an ID mean."""

    if isinstance(values, Mapping):
        ordered = [float(values[key]) for key in sorted(values)]
    else:
        ordered = [float(value) for value in values]
    if not ordered:
        return {
            "mean": None,
            "ci95": [None, None],
            "lower": None,
            "upper": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "n_ids": 0,
            "n_resamples": int(n_resamples),
        }
    if isinstance(n_resamples, bool) or int(n_resamples) < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    observed = float(np.mean(np.asarray(ordered, dtype=np.float64)))
    rng = np.random.default_rng(int(seed))
    samples = rng.integers(0, len(ordered), size=(int(n_resamples), len(ordered)))
    means = np.asarray(ordered, dtype=np.float64)[samples].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    lower, upper = (float(value) for value in np.quantile(means, [alpha, 1.0 - alpha]))
    # Keeping the point estimate inside the reported percentile interval makes
    # the diagnostic unambiguous for tiny panels (especially one recording).
    lower = min(lower, observed)
    upper = max(upper, observed)
    return {
        "mean": observed,
        "ci95": [lower, upper],
        "lower": lower,
        "upper": upper,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "n_ids": len(ordered),
        "n_resamples": int(n_resamples),
    }


def _condition_value(record: Mapping[str, Any], condition: str) -> float:
    primary = record.get("primary")
    if isinstance(primary, Mapping):
        return float(primary[condition])
    condition_payload = record.get(condition)
    if isinstance(condition_payload, Mapping) and "primary" in condition_payload:
        return float(condition_payload["primary"])
    aliases = {
        "TRUE": ("primary_true", "true_primary", "TRUE"),
        "SUBSTITUTED": ("primary_substituted", "substituted_primary", "SUBSTITUTED"),
        "ZEROED": ("primary_zeroed", "zeroed_primary", "ZEROED"),
    }
    for key in aliases[condition]:
        if key in record:
            return float(record[key])
    raise KeyError(f"result record has no primary for {condition}")


def _invalid_value(record: Mapping[str, Any], condition: str) -> bool:
    invalid = record.get("invalid_format")
    if isinstance(invalid, Mapping):
        return bool(invalid[condition])
    condition_payload = record.get(condition)
    if isinstance(condition_payload, Mapping) and "invalid_format" in condition_payload:
        return bool(condition_payload["invalid_format"])
    aliases = {
        "TRUE": ("invalid_true", "true_invalid"),
        "SUBSTITUTED": ("invalid_substituted", "substituted_invalid"),
        "ZEROED": ("invalid_zeroed", "zeroed_invalid"),
    }
    for key in aliases[condition]:
        if key in record:
            return bool(record[key])
    return False


def _condition_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        values = [_condition_value(record, condition) for record in records]
        invalid = sum(_invalid_value(record, condition) for record in records)
        result[condition] = {
            "mean": float(np.mean(values)) if values else None,
            "n_crops": len(values),
            "invalid": int(invalid),
            "invalid_rate": float(invalid / len(values)) if values else None,
        }
    return result


def _delta_summary(
    records: Sequence[Mapping[str, Any]], *, seed: int, n_resamples: int
) -> dict[str, Any]:
    by_id = aggregate_deltas_by_recording(records)
    bootstrap = paired_bootstrap(by_id, n_resamples=n_resamples, seed=seed)
    return {
        "n_crops": len(records),
        "n_recordings": len(by_id),
        "by_recording": by_id,
        "mean_delta": bootstrap["mean"],
        "ci95": bootstrap["ci95"],
        "bootstrap": bootstrap,
    }


def summarize_gate_results(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    collapse_tolerance: float = DEFAULT_COLLAPSE_TOLERANCE,
    invalid_rate_limit: float = DEFAULT_INVALID_RATE_LIMIT,
) -> dict[str, Any]:
    """Aggregate crop results and apply the DESIGN 12.2 gate criteria."""

    records = list(records)
    categories = sorted({_record_category(record) for record in records})
    overall = {
        "conditions": _condition_summary(records),
        "delta": _delta_summary(records, seed=seed, n_resamples=n_resamples),
    }
    by_category: dict[str, Any] = {}
    for category in categories:
        subset = [record for record in records if _record_category(record) == category]
        by_category[category] = {
            "conditions": _condition_summary(subset),
            "delta": _delta_summary(
                subset,
                seed=_stable_seed(seed, "category", category),
                n_resamples=n_resamples,
            ),
        }

    collapse: list[dict[str, Any]] = []
    for category, summary in by_category.items():
        true_mean = summary["conditions"]["TRUE"]["mean"]
        controls = [
            summary["conditions"][condition]["mean"]
            for condition in ("SUBSTITUTED", "ZEROED")
            if summary["conditions"][condition]["mean"] is not None
        ]
        best = max(controls) if controls else None
        gap = None if true_mean is None or best is None else float(best - true_mean)
        flagged = bool(gap is not None and gap > float(collapse_tolerance))
        item = {
            "category": category,
            "true_mean": true_mean,
            "best_control_mean": best,
            "true_below_best_control": gap,
            "tolerance": float(collapse_tolerance),
            "flagged": flagged,
        }
        if flagged:
            collapse.append(item)

    # Astra 15/09 (scala): TRUE = SUBSTITUTED = ZEROED = 0 is blindness, not absence of collapse; a main category
    # at zero cannot be authorised by the aggregate advantage of the others.
    at_zero = sorted(
        category for category, summary in by_category.items()
        if category in TARGET_CATEGORIES and (summary["conditions"]["TRUE"]["mean"] or 0.0) <= ZERO_TOLERANCE
    )
    invalid_rates = {
        condition: overall["conditions"][condition]["invalid_rate"] for condition in CONDITIONS
    }
    invalid_failures = [
        condition
        for condition, rate in invalid_rates.items()
        if rate is None or rate >= float(invalid_rate_limit)
    ]
    bootstrap = overall["delta"]["bootstrap"]
    ci_lower = bootstrap["lower"]
    criteria = {
        "overall_ci95_lower_gt_zero": bool(ci_lower is not None and ci_lower > 0.0),
        "no_category_collapse": not collapse,
        "no_main_category_at_zero": not at_zero,
        "invalid_rate_below_1pct": not invalid_failures,
        "collapse_tolerance": float(collapse_tolerance),
        "invalid_rate_limit": float(invalid_rate_limit),
    }
    passed = all(criteria[key] for key in (
        "overall_ci95_lower_gt_zero", "no_category_collapse", "no_main_category_at_zero", "invalid_rate_below_1pct"
    ))
    return {
        "n_crops": len(records),
        "n_recordings": len(aggregate_deltas_by_recording(records)),
        "overall": overall,
        "by_category": by_category,
        "invalid_format_rate": invalid_rates,
        "collapse": collapse,
        "main_categories_at_zero": at_zero,
        "criteria": criteria,
        "invalid_failures": invalid_failures,
        "verdict": "PASS" if passed else "FAIL",
        "pass": passed,
    }


def gate_verdict(
    results: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    seed: int = 0,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    collapse_tolerance: float = DEFAULT_COLLAPSE_TOLERANCE,
    invalid_rate_limit: float = DEFAULT_INVALID_RATE_LIMIT,
) -> dict[str, Any]:
    """Return verdict details for either raw crop rows or an existing report."""

    if isinstance(results, Mapping) and "verdict" in results and "criteria" in results:
        return dict(results)
    if isinstance(results, Mapping) and "records" in results:
        return summarize_gate_results(
            results["records"],
            seed=seed,
            n_resamples=n_resamples,
            collapse_tolerance=collapse_tolerance,
            invalid_rate_limit=invalid_rate_limit,
        )
    return summarize_gate_results(
        list(results),
        seed=seed,
        n_resamples=n_resamples,
        collapse_tolerance=collapse_tolerance,
        invalid_rate_limit=invalid_rate_limit,
    )


evaluate_verdict = gate_verdict
compute_verdict = gate_verdict
compute_gate_verdict = gate_verdict
aggregate_by_recording = aggregate_deltas_by_recording


# --------------------------------------------------------------------------- model runner


def _split_meta(split: Any, crop: PanelCrop) -> RowMeta:
    if hasattr(split, "meta"):
        meta = split.meta(crop.row_index)
    elif isinstance(split, Mapping):
        meta = split[crop.row_index]
    else:
        meta = next(meta for meta in split if meta.row_index == crop.row_index or meta.id == crop.row_id)
    if meta.id != crop.row_id:
        raise ValueError(f"panel row id mismatch: panel={crop.row_id!r}, split={meta.id!r}")
    if meta.category != crop.category:
        raise ValueError(f"panel category mismatch for {crop.row_id!r}")
    return meta


def _split_episode(split: Any, crop: PanelCrop) -> SignalEpisode:
    if hasattr(split, "episode"):
        return split.episode(crop.row_index)
    if isinstance(split, Mapping):
        item = split[crop.row_index]
        if isinstance(item, tuple):
            return item[1]
        return item
    raise TypeError("split must provide episode(row_index)")


def _move_episode(episode: SignalEpisode, device: torch.device | str | None) -> SignalEpisode:
    if device is None:
        return episode
    device = torch.device(device)
    if episode.values.device == device:
        return episode
    return SignalEpisode(
        values=episode.values.to(device),
        valid=episode.valid.to(device),
        channel_ids=episode.channel_ids,
        sample_times=episode.sample_times.to(device),
        task=episode.task,
    )


def _generation_text(result: Any) -> str:
    if isinstance(result, Mapping):
        return str(result.get("text", ""))
    return str(result)


def _true_headers(model: Any, episode: SignalEpisode, action: ZoomAction) -> dict[int, ViewHeader]:
    encoding = model.encode_episode(episode, [action])
    views = getattr(encoding, "views", None)
    if not isinstance(views, Sequence) or len(views) != 2:
        raise ValueError("encode_episode must return global and fixed-crop views")
    headers: dict[int, ViewHeader] = {}
    for index, view in enumerate(views):
        header = getattr(view, "header", None)
        if not isinstance(header, ViewHeader):
            raise TypeError("encode_episode returned a view without a ViewHeader")
        headers[index] = header
    return headers


def evaluate_gate(
    model: Any,
    split: Any,
    panel: Sequence[PanelCrop],
    *,
    device: torch.device | str | None = None,
    seed: int = 0,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    max_new_tokens: int = 768,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Run all three conditions for every frozen panel crop and summarise them."""

    records: list[dict[str, Any]] = []
    model_was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    inferred_device = device if device is not None else getattr(model, "device", "cpu")
    amp_device = torch.device(inferred_device).type
    use_amp = autocast_dtype is not None and amp_device == "cuda"
    for crop_number, crop in enumerate(panel):
        meta = _split_meta(split, crop)
        original = _move_episode(_split_episode(split, crop), device)
        action = crop.action()
        headers = _true_headers(model, original, action)
        primary: dict[str, float] = {}
        invalid: dict[str, bool] = {}
        predictions: dict[str, str] = {}
        for condition in CONDITIONS:
            episode = perturb_episode(
                original,
                meta,
                crop,
                condition,
                source_map=crop.substitution_sources,
                seed=_stable_seed(seed, crop.row_id, crop.start, crop.end),
            )
            context = (
                torch.autocast("cuda", dtype=autocast_dtype)
                if use_amp
                else nullcontext()
            )
            with context:
                generated = model.generate(
                    meta,
                    episode,
                    mode="fixed_crop",
                    actions=[action],
                    max_new_tokens=max_new_tokens,
                    header_overrides=headers,
                )
            text = _generation_text(generated)
            parsed = parse_structured_pred(text, REQUIRED_KEYS[meta.category])
            scored = score_precise(text, meta.answer)
            primary[condition] = float(scored["primary"])
            invalid[condition] = parsed is None
            predictions[condition] = text
        delta = primary["TRUE"] - max(primary["SUBSTITUTED"], primary["ZEROED"])
        records.append(
            {
                "crop_index": crop_number,
                "recording_id": crop.row_id,
                "row_id": crop.row_id,
                "row_index": crop.row_index,
                "category": crop.category,
                "channels": list(crop.channels),
                "start": crop.start,
                "end": crop.end,
                "crop_kind": crop.crop_kind,
                "substitution_source_window": {
                    channel: source.to_dict()
                    for channel, source in crop.substitution_sources.items()
                },
                "primary": primary,
                "invalid_format": invalid,
                "delta": float(delta),
                "predictions": predictions,
            }
        )
    report = summarize_gate_results(records, seed=seed, n_resamples=n_resamples)
    report["records"] = records
    report["protocol"] = {
        "conditions": list(CONDITIONS),
        "max_new_tokens": int(max_new_tokens),
        "seed": int(seed),
        "bootstrap_resamples": int(n_resamples),
        "validation_only": True,
        "header_source": "TRUE episode for every crop",
    }
    if model_was_training is True and hasattr(model, "train"):
        model.train()
    return report


def render_gate_summary(report: Mapping[str, Any]) -> str:
    """Render the short human-readable gate hand-off."""

    overall = report["overall"]
    bootstrap = overall["delta"]["bootstrap"]
    lines = [
        "# Perceptual gate",
        "",
        f"Verdict: **{report['verdict']}**",
        f"Crops: {report['n_crops']} across {report['n_recordings']} recordings.",
        (
            "Overall TRUE − best control: "
            f"{bootstrap['mean']:.6g} "
            f"(CI95 {bootstrap['lower']:.6g}, {bootstrap['upper']:.6g})."
            if bootstrap["mean"] is not None
            else "Overall TRUE − best control: no crops."
        ),
        "",
        "| Condition | primary mean | invalid-format rate |",
        "|---|---:|---:|",
    ]
    for condition in CONDITIONS:
        item = overall["conditions"][condition]
        mean = "n/a" if item["mean"] is None else f"{item['mean']:.6g}"
        rate = "n/a" if item["invalid_rate"] is None else f"{100 * item['invalid_rate']:.4g}%"
        lines.append(f"| {condition} | {mean} | {rate} |")
    lines.extend(["", "Category collapse checks:"])
    for item in report["collapse"]:
        lines.append(
            f"- {item['category']}: TRUE {item['true_mean']:.6g} vs best control "
            f"{item['best_control_mean']:.6g} (gap {item['true_below_best_control']:.6g})"
        )
    if not report["collapse"]:
        lines.append("- none")
    if report["invalid_failures"]:
        lines.extend(["", "Invalid-format criterion failed for: " + ", ".join(report["invalid_failures"])])
    return "\n".join(lines) + "\n"


def write_gate_outputs(out_dir: str | Path, report: Mapping[str, Any]) -> tuple[Path, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    report_path = target / "gate_report.json"
    summary_path = target / "gate_summary.md"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary_path.write_text(render_gate_summary(report), encoding="utf-8")
    return report_path, summary_path


__all__ = [
    "CONDITIONS",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_CROPS_PER_ROW",
    "DEFAULT_K",
    "PanelCrop",
    "SubstitutionSource",
    "TARGET_CATEGORIES",
    "aggregate_deltas_by_recording",
    "aggregate_by_recording",
    "build_panel",
    "build_perturbed_episode",
    "build_substituted_episode",
    "build_zeroed_episode",
    "evaluate_gate",
    "evaluate_verdict",
    "compute_verdict",
    "compute_gate_verdict",
    "gate_verdict",
    "globally_normalized",
    "load_panel",
    "panel_payload",
    "paired_bootstrap",
    "perturb_episode",
    "make_perturbed_episode",
    "read_panel",
    "render_gate_summary",
    "save_panel",
    "select_substitution_source",
    "select_substitution_window",
    "substitution_sources",
    "summarize_gate_results",
    "write_gate_outputs",
]
