"""AnomalyXL data side of WR-Agent-4B-Wildcard-v1.

This module owns everything between the pinned parquet files and the canonical
signal contract:

* :class:`AnomalyXLSplit` opens one split, verifies its row count against the
  manifest (DESIGN §11.3 and decision 3 of ``DECISIONI-ASTRA-20260914.md``:
  4488/520/800 must be checked on the files actually read), keeps only the cheap
  metadata columns resident and loads ``channel_values`` lazily, one row at a
  time, so the 2 GB train split never sits in RAM.
* :class:`RowMeta` / :class:`GoldInterval` carry the per-row metadata and the
  dense annotations (``injection_channels``/``starts``/``ends``/``kinds``).
* :class:`ZoomAction` and :func:`parse_zoom_action` implement the single tool
  contract of DESIGN §9.3 with absolute half-open coordinates.  The parser is
  strict on purpose: the runner never silently clamps or repairs an action.
* :class:`ModeAssigner` implements the per-pass stratified mode mix of DESIGN
  §10.1/§10.3 (every train id processed exactly once per pass).
* :class:`TeacherCropSampler` builds the fixed reading crops of P1/P2
  (DESIGN §10.3/§10.4): random internal event position, partial border cases,
  negative crops, synchronized and single-channel multichannel reads.
* :func:`occupancy_targets` produces the fractional targets of the auxiliary
  event-occupancy head (DESIGN §10.3).

Invariants respected here: non-finite samples become ``valid=False`` and are
**never** imputed at this boundary (imputation belongs to the frontend, decision
1 of 2026-09-14T19:46); annotations are read only to build TRAIN supervision and
never enter an observation; coordinates are absolute ``[start, end)`` indices of
the original recording.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow.parquet as pq
import torch

from .signal_contract import SignalEpisode

MIN_CROP_SAMPLES = 256
MAX_CROP_SAMPLES = 8192
MIN_CHANNELS_PER_CALL = 1
MAX_CHANNELS_PER_CALL = 2

MODES: tuple[str, ...] = ("direct", "fixed_crop", "agentic")
# ``prototype.spec.json`` spells the reading mode ``fixed_crop_reading``; DESIGN
# and the model call it ``fixed_crop``.  Accept both spellings, emit one.
_MODE_ALIASES = {"fixed_crop_reading": "fixed_crop"}

_METADATA_COLUMNS = (
    "id",
    "category",
    "length",
    "n_channels",
    "question",
    "answer",
    "channel_names",
    "injection_kinds",
    "injection_channels",
    "injection_starts",
    "injection_ends",
)


# --------------------------------------------------------------------------- rows


@dataclass(frozen=True)
class GoldInterval:
    """One dense annotation: half-open ``[start, end)`` on one channel."""

    channel: str
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class RowMeta:
    """Cheap per-row metadata.  Never carries signal values."""

    id: str
    category: str
    length: int
    n_channels: int
    channel_names: tuple[str, ...]
    question: str
    answer: str
    gold_intervals: tuple[GoldInterval, ...]
    split: str
    row_index: int


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field} must be an integer, got {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise ValueError(f"{field} must be an integer, got {type(value).__name__}")


def _as_str_tuple(value: object, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list of strings")
    out = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field} must contain strings, got {type(item).__name__}")
        out.append(item)
    return tuple(out)


def _gold_from_row(
    kinds: object, channels: object, starts: object, ends: object, *, row_id: str
) -> tuple[GoldInterval, ...]:
    """Zip the aligned ``injection_*`` lists into absolute half-open intervals."""

    channel_list = _as_str_tuple(channels, "injection_channels")
    start_list = [] if starts is None else list(starts)
    end_list = [] if ends is None else list(ends)
    kind_list = _as_str_tuple(kinds, "injection_kinds")
    if len(start_list) != len(channel_list) or len(end_list) != len(channel_list):
        raise ValueError(
            f"row {row_id!r}: injection_channels/starts/ends are not aligned "
            f"({len(channel_list)}/{len(start_list)}/{len(end_list)})"
        )
    intervals = []
    for index, channel in enumerate(channel_list):
        # ``injection_kinds`` is aligned when it has the same length; a single
        # kind is broadcast; anything else leaves the kind empty rather than
        # inventing a label.
        if len(kind_list) == len(channel_list):
            kind = kind_list[index]
        elif len(kind_list) == 1:
            kind = kind_list[0]
        else:
            kind = ""
        intervals.append(
            GoldInterval(
                channel=channel,
                start=_as_int(start_list[index], "injection_starts"),
                end=_as_int(end_list[index], "injection_ends"),
                kind=kind,
            )
        )
    return tuple(intervals)


# --------------------------------------------------------------------------- split


class AnomalyXLSplit:
    """One AnomalyXL parquet split, opened lazily.

    Only the metadata columns are materialised in memory; ``channel_values`` is
    read per row from its row group and released immediately, so opening the
    2 GB train split costs a few megabytes.  A row group whose uncompressed size
    exceeds ``max_resident_bytes`` (the pinned files may hold every row in one
    group) is streamed batch by batch instead of being materialised, which keeps
    the peak resident size bounded at the cost of decoding time.
    """

    _ITER_BATCH_ROWS = 1

    def __init__(
        self,
        parquet_path: str | Path,
        split_name: str,
        expected_rows: int | None = None,
        *,
        max_resident_bytes: int = 128 << 20,
    ) -> None:
        self.path = Path(parquet_path)
        self.split_name = str(split_name)
        if max_resident_bytes < 1:
            raise ValueError("max_resident_bytes must be positive")
        self.max_resident_bytes = int(max_resident_bytes)
        if not self.path.is_file():
            raise FileNotFoundError(f"missing {self.split_name} parquet at {self.path}")
        self._file = pq.ParquetFile(self.path)

        available = set(self._file.schema_arrow.names)
        missing = [name for name in (*_METADATA_COLUMNS, "channel_values") if name not in available]
        if missing:
            raise ValueError(f"{self.path} is missing required columns: {missing}")

        self.num_rows = int(self._file.metadata.num_rows)
        if expected_rows is not None and self.num_rows != int(expected_rows):
            raise ValueError(
                f"{self.split_name} split at {self.path} has {self.num_rows} rows, "
                f"expected {int(expected_rows)}"
            )

        # Row-group offsets for the lazy per-row reads (empty groups skipped).
        self._group_ids: list[int] = []
        self._group_starts: list[int] = []
        self._group_bytes: list[int] = []
        cursor = 0
        for group in range(self._file.metadata.num_row_groups):
            metadata = self._file.metadata.row_group(group)
            rows = int(metadata.num_rows)
            if rows <= 0:
                continue
            self._group_ids.append(group)
            self._group_starts.append(cursor)
            self._group_bytes.append(int(metadata.total_byte_size))
            cursor += rows
        if cursor != self.num_rows:
            raise ValueError(f"{self.path}: row-group sizes do not sum to {self.num_rows}")

        self._metas: tuple[RowMeta, ...] = self._read_metadata()

    # -- construction -------------------------------------------------------
    def _read_metadata(self) -> tuple[RowMeta, ...]:
        table = self._file.read(columns=list(_METADATA_COLUMNS))
        columns = {name: table.column(name).to_pylist() for name in _METADATA_COLUMNS}
        del table
        metas: list[RowMeta] = []
        for index in range(self.num_rows):
            row_id = str(columns["id"][index])
            names = _as_str_tuple(columns["channel_names"][index], "channel_names")
            n_channels = _as_int(columns["n_channels"][index], "n_channels")
            if len(names) != n_channels:
                raise ValueError(
                    f"row {row_id!r}: n_channels={n_channels} but {len(names)} channel names"
                )
            metas.append(
                RowMeta(
                    id=row_id,
                    category=str(columns["category"][index]),
                    length=_as_int(columns["length"][index], "length"),
                    n_channels=n_channels,
                    channel_names=names,
                    question=str(columns["question"][index]),
                    answer=str(columns["answer"][index]),
                    gold_intervals=_gold_from_row(
                        columns["injection_kinds"][index],
                        columns["injection_channels"][index],
                        columns["injection_starts"][index],
                        columns["injection_ends"][index],
                        row_id=row_id,
                    ),
                    split=self.split_name,
                    row_index=index,
                )
            )
        return tuple(metas)

    # -- access -------------------------------------------------------------
    def __len__(self) -> int:
        return self.num_rows

    def _check_index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("row index must be an integer")
        index = int(index)
        if not 0 <= index < self.num_rows:
            raise IndexError(f"row index {index} out of range for {self.num_rows} rows")
        return index

    def meta(self, i: int) -> RowMeta:
        return self._metas[self._check_index(i)]

    def metas(self) -> list[RowMeta]:
        return list(self._metas)

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for meta in self._metas:
            counts[meta.category] = counts.get(meta.category, 0) + 1
        return {name: counts[name] for name in sorted(counts)}

    def _locate(self, index: int) -> tuple[int, int, int]:
        slot = bisect.bisect_right(self._group_starts, index) - 1
        return self._group_ids[slot], index - self._group_starts[slot], self._group_bytes[slot]

    @staticmethod
    def _arrays_from_scalar(scalar, index: int, path: Path) -> list[np.ndarray]:
        if scalar is None or not scalar.is_valid:
            raise ValueError(f"row {index} of {path} has null channel_values")
        try:
            child = scalar.values
            return [
                np.asarray(child[c].values.to_numpy(zero_copy_only=False), dtype=np.float32)
                for c in range(len(child))
            ]
        except AttributeError:  # pragma: no cover - older pyarrow layouts
            return [np.asarray(channel, dtype=np.float32) for channel in scalar.as_py()]

    def channel_values(self, i: int) -> list[np.ndarray]:
        """Read one row's channels as float32 arrays, without caching them."""

        index = self._check_index(i)
        group, local, group_bytes = self._locate(index)
        if group_bytes <= self.max_resident_bytes:
            table = self._file.read_row_group(group, columns=["channel_values"])
            try:
                return self._arrays_from_scalar(table.column("channel_values")[local], index, self.path)
            finally:
                del table
        # Oversized row group: stream it so the peak resident size stays bounded.
        seen = 0
        for batch in self._file.iter_batches(
            batch_size=self._ITER_BATCH_ROWS, row_groups=[group], columns=["channel_values"]
        ):
            rows = batch.num_rows
            if local < seen + rows:
                return self._arrays_from_scalar(batch.column(0)[local - seen], index, self.path)
            seen += rows
        raise RuntimeError(  # pragma: no cover - row-group offsets are verified in __init__
            f"row {index} of {self.path} was not found in row group {group}"
        )

    def episode(self, i: int) -> SignalEpisode:
        """Build the canonical :class:`SignalEpisode` for one row.

        ``valid`` is exactly ``isfinite(values)``: a non-finite sample is an
        unobserved sample and is left non-finite here, never imputed.
        """

        meta = self.meta(i)
        arrays = self.channel_values(i)
        if len(arrays) != meta.n_channels:
            raise ValueError(
                f"row {meta.id!r}: {len(arrays)} channel arrays for n_channels={meta.n_channels}"
            )
        bad = [c for c, array in enumerate(arrays) if array.shape != (meta.length,)]
        if bad:
            raise ValueError(
                f"row {meta.id!r}: channels {bad} do not have the declared length {meta.length}"
            )
        values = torch.from_numpy(np.stack(arrays).astype(np.float32, copy=False)).contiguous()
        valid = torch.isfinite(values)
        return SignalEpisode.from_index_grid(
            values=values,
            valid=valid,
            channel_ids=meta.channel_names,
            dt=1.0,
            origin=0.0,
            task=meta,
        )


# --------------------------------------------------------------------------- tool


@dataclass(frozen=True)
class ZoomAction:
    """The single perceptual tool of DESIGN §9.3, in absolute coordinates."""

    channels: tuple[str, ...]
    start: int
    end: int

    def __post_init__(self) -> None:
        channels = tuple(self.channels)
        object.__setattr__(self, "channels", channels)
        if not MIN_CHANNELS_PER_CALL <= len(channels) <= MAX_CHANNELS_PER_CALL:
            raise ValueError(
                f"a zoom call selects {MIN_CHANNELS_PER_CALL}-{MAX_CHANNELS_PER_CALL} channels, "
                f"got {len(channels)}"
            )
        if any(not isinstance(name, str) or not name for name in channels):
            raise ValueError("every zoom channel must be a non-empty string")
        if len(set(channels)) != len(channels):
            raise ValueError("a zoom call must not repeat a channel")
        for name, value in (("start", self.start), ("end", self.end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"zoom {name} must be an integer")
        if not 0 <= self.start < self.end:
            raise ValueError("a zoom interval must satisfy 0 <= start < end")

    @property
    def width(self) -> int:
        return self.end - self.start

    def to_json(self) -> str:
        payload = {
            "tool": "zoom",
            "channels": list(self.channels),
            "start": int(self.start),
            "end": int(self.end),
        }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def parse_zoom_action(
    text: str,
    *,
    channel_names: Sequence[str],
    length: int,
    min_samples: int = MIN_CROP_SAMPLES,
    max_samples: int = MAX_CROP_SAMPLES,
) -> ZoomAction | None:
    """Parse one action payload, or return ``None``.

    The whole stripped text must be exactly one JSON object with the keys
    ``tool``/``channels``/``start``/``end``.  Nothing is clamped or corrected:
    a request outside the contract is invalid (DESIGN §9.3), and the runner
    charges it a call rather than repairing it.
    """

    if min_samples < 1 or max_samples < min_samples:
        raise ValueError("min_samples/max_samples must satisfy 1 <= min_samples <= max_samples")
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("length must be a positive integer")
    known = tuple(channel_names)

    stripped = (text or "").strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if set(payload) != {"tool", "channels", "start", "end"}:
        return None
    if payload["tool"] != "zoom":
        return None

    channels = payload["channels"]
    if not isinstance(channels, list):
        return None
    if not MIN_CHANNELS_PER_CALL <= len(channels) <= MAX_CHANNELS_PER_CALL:
        return None
    if any(not isinstance(name, str) for name in channels):
        return None
    if len(set(channels)) != len(channels):
        return None
    if any(name not in known for name in channels):
        return None

    start, end = payload["start"], payload["end"]
    for value in (start, end):
        if isinstance(value, bool) or not isinstance(value, int):
            return None
    if not 0 <= start < end <= length:
        return None

    width = end - start
    if length < min_samples:
        # Short recording: the only legal window is the whole series.
        if start != 0 or end != length:
            return None
    elif not min_samples <= width <= max_samples:
        return None

    return ZoomAction(channels=tuple(channels), start=int(start), end=int(end))


# --------------------------------------------------------------------------- mixes


def _derive_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Hamilton apportionment: integer counts summing exactly to ``total``."""

    keys = sorted(weights)
    exact = {key: total * weights[key] for key in keys}
    counts = {key: int(math.floor(exact[key])) for key in keys}
    remaining = total - sum(counts.values())
    order = sorted(keys, key=lambda key: (-(exact[key] - counts[key]), key))
    cursor = 0
    while remaining > 0:
        counts[order[cursor % len(order)]] += 1
        remaining -= 1
        cursor += 1
    return counts


def _normalize_mix(mix: dict[str, float]) -> dict[str, float]:
    if not mix:
        raise ValueError("the mode mix must not be empty")
    normalized: dict[str, float] = {}
    for raw_name, weight in mix.items():
        name = _MODE_ALIASES.get(str(raw_name), str(raw_name))
        if name not in MODES:
            raise ValueError(f"unknown training mode {raw_name!r}; expected one of {MODES}")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"mix weight for {name!r} must be a number")
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"mix weight for {name!r} must be finite and non-negative")
        if name in normalized:
            raise ValueError(f"mode {name!r} appears twice in the mix")
        normalized[name] = weight
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("the mode mix must have a positive total weight")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"the mode mix must sum to 1, got {total!r}")
    return {name: weight / total for name, weight in normalized.items()}


def _mix_key(normalized: dict[str, float]) -> str:
    return ",".join(f"{name}={normalized[name]!r}" for name in sorted(normalized))


def _bucket_by_category(metas: Sequence[RowMeta]) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = {}
    seen: set[int] = set()
    for meta in metas:
        if meta.row_index in seen:
            raise ValueError(f"row_index {meta.row_index} appears twice in the metas")
        seen.add(meta.row_index)
        buckets.setdefault(meta.category, []).append(meta.row_index)
    return buckets


class ModeAssigner:
    """Assign one training mode per row, stratified by category (DESIGN §10.1).

    Every pass processes each id exactly once; the proportions of ``mix`` hold
    inside every category (Hamilton apportionment on the category size), and the
    result is fully determined by ``(seed, pass_index, mix)``.
    """

    def assign(
        self,
        metas: list[RowMeta],
        mix: dict[str, float],
        *,
        seed: int,
        pass_index: int,
    ) -> list[tuple[int, str]]:
        normalized = _normalize_mix(mix)
        key = _mix_key(normalized)
        buckets = _bucket_by_category(metas)
        pairs: list[tuple[int, str]] = []
        for category in sorted(buckets):
            rows = sorted(buckets[category])
            rng = random.Random(_derive_seed("wr-mode-assign", seed, pass_index, key, category))
            rng.shuffle(rows)
            counts = _largest_remainder(len(rows), normalized)
            cursor = 0
            for mode in sorted(counts):
                take = counts[mode]
                pairs.extend((row, mode) for row in rows[cursor : cursor + take])
                cursor += take
            if cursor != len(rows):  # pragma: no cover - apportionment is exact
                raise RuntimeError("mode apportionment did not cover the category")
        order = random.Random(_derive_seed("wr-mode-order", seed, pass_index, key))
        order.shuffle(pairs)
        return pairs


def stratified_subset(metas: list[RowMeta], n: int, seed: int) -> list[int]:
    """Pick ``n`` row indices keeping the category proportions of ``metas``."""

    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")
    buckets = _bucket_by_category(metas)
    total = sum(len(rows) for rows in buckets.values())
    if n == 0 or total == 0:
        return []
    if n >= total:
        return sorted(index for rows in buckets.values() for index in rows)

    weights = {category: len(rows) / total for category, rows in buckets.items()}
    quota = _largest_remainder(n, weights)
    leftover = 0
    for category in sorted(quota):
        capacity = len(buckets[category])
        if quota[category] > capacity:
            leftover += quota[category] - capacity
            quota[category] = capacity
    while leftover > 0:
        progressed = False
        for category in sorted(quota, key=lambda c: (-(len(buckets[c]) - quota[c]), c)):
            if leftover <= 0:
                break
            if quota[category] < len(buckets[category]):
                quota[category] += 1
                leftover -= 1
                progressed = True
        if not progressed:  # pragma: no cover - n < total guarantees capacity
            break

    picked: list[int] = []
    for category in sorted(buckets):
        rows = sorted(buckets[category])
        rng = random.Random(_derive_seed("wr-stratified-subset", seed, n, category))
        rng.shuffle(rows)
        picked.extend(rows[: quota[category]])
    return sorted(picked)


# --------------------------------------------------------------------------- teacher


def _intersect(lo: int, hi: int, cap_lo: int, cap_hi: int) -> tuple[int, int] | None:
    lo, hi = max(lo, cap_lo), min(hi, cap_hi)
    return (lo, hi) if lo <= hi else None


class TeacherCropSampler:
    """Fixed reading crops for P1/P2 (DESIGN §10.3/§10.4).

    Annotations supervise the construction of these crops; that is allowed in
    TRAIN only, and the crops themselves never carry the answer.  Positive
    windows put the event at a random internal position — never systematically
    centred — and a share of them cut it at a border.  Rows without annotations
    always get a random window, so the mix never teaches "every crop contains an
    anomaly".
    """

    def __init__(
        self,
        *,
        min_samples: int = MIN_CROP_SAMPLES,
        max_samples: int = MAX_CROP_SAMPLES,
        positive_prob: float = 0.7,
        border_partial_prob: float = 0.2,
        pair_prob_two_channel: float = 0.5,
        pair_prob_multi_channel: float = 0.6,
    ) -> None:
        if min_samples < 1 or max_samples < min_samples:
            raise ValueError("min_samples/max_samples must satisfy 1 <= min_samples <= max_samples")
        for name, value in (
            ("positive_prob", positive_prob),
            ("border_partial_prob", border_partial_prob),
            ("pair_prob_two_channel", pair_prob_two_channel),
            ("pair_prob_multi_channel", pair_prob_multi_channel),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        self.min_samples = int(min_samples)
        self.max_samples = int(max_samples)
        self.positive_prob = float(positive_prob)
        self.border_partial_prob = float(border_partial_prob)
        self.pair_prob_two_channel = float(pair_prob_two_channel)
        self.pair_prob_multi_channel = float(pair_prob_multi_channel)

    # -- helpers ------------------------------------------------------------
    def usable_gold(self, meta: RowMeta) -> tuple[GoldInterval, ...]:
        """Annotations that are non-empty and inside the declared recording."""

        names = set(meta.channel_names)
        return tuple(
            item
            for item in meta.gold_intervals
            if item.channel in names and 0 <= item.start < item.end <= meta.length
        )

    def _width(self, length: int, rng: random.Random) -> int:
        low = min(self.min_samples, length)
        high = min(self.max_samples, length)
        if low >= high:
            return high
        width = int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))
        return max(low, min(high, width))

    def _positive_start(
        self, interval: GoldInterval, width: int, length: int, partial: bool, rng: random.Random
    ) -> int:
        a, b = interval.start, interval.end
        cap_lo, cap_hi = 0, length - width
        contain = _intersect(b - width, a, cap_lo, cap_hi)  # window ⊇ event
        overlap = _intersect(a - width + 1, b - 1, cap_lo, cap_hi)  # window ∩ event ≠ ∅
        if not partial and contain is not None:
            return rng.randint(*contain)
        # Strictly partial: the window ends inside the event, or starts inside it.
        ranges = [
            candidate
            for candidate in (
                _intersect(a - width + 1, b - width - 1, cap_lo, cap_hi),
                _intersect(a + 1, b - 1, cap_lo, cap_hi),
            )
            if candidate is not None
        ]
        if ranges:
            sizes = [hi - lo + 1 for lo, hi in ranges]
            pick = rng.randrange(sum(sizes))
            for (lo, hi), size in zip(ranges, sizes):
                if pick < size:
                    return lo + pick
                pick -= size
        if contain is not None:
            return rng.randint(*contain)
        if overlap is not None:  # pragma: no cover - overlap is non-empty when width <= length
            return rng.randint(*overlap)
        return rng.randint(cap_lo, cap_hi)

    def _channels(self, meta: RowMeta, base: str, rng: random.Random) -> tuple[str, ...]:
        names = meta.channel_names
        if len(names) <= 1:
            return (names[0],)
        pair_prob = (
            self.pair_prob_two_channel if len(names) == 2 else self.pair_prob_multi_channel
        )
        if rng.random() < pair_prob:
            others = [name for name in names if name != base]
            other = rng.choice(others)
            return tuple(sorted((base, other), key=names.index))
        return (base,)

    # -- public API ---------------------------------------------------------
    def sample(self, meta: RowMeta, rng: random.Random) -> ZoomAction:
        if meta.length < 1:
            raise ValueError(f"row {meta.id!r} has no samples")
        if not meta.channel_names:
            raise ValueError(f"row {meta.id!r} has no channels")
        gold = self.usable_gold(meta)
        positive = bool(gold) and rng.random() < self.positive_prob
        width = self._width(meta.length, rng)

        if positive:
            interval = rng.choice(gold)
            partial = rng.random() < self.border_partial_prob
            start = self._positive_start(interval, width, meta.length, partial, rng)
            base = interval.channel
        else:
            start = rng.randint(0, meta.length - width)
            base = rng.choice(meta.channel_names)

        channels = self._channels(meta, base, rng)
        return ZoomAction(channels=channels, start=int(start), end=int(start + width))


# --------------------------------------------------------------------------- aux head


def _merge_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def occupancy_targets(
    intervals: Sequence[GoldInterval], channel: str, bin_edges: torch.Tensor
) -> torch.Tensor:
    """Fraction of every bin covered by the union of one channel's annotations.

    ``bin_edges`` are ``Q+1`` absolute sample edges; the output is float32
    ``[Q]`` in ``[0, 1]`` (DESIGN §10.3 auxiliary event-occupancy head).
    """

    if not isinstance(bin_edges, torch.Tensor):
        raise TypeError("bin_edges must be a torch.Tensor")
    if bin_edges.ndim != 1 or bin_edges.numel() < 2:
        raise ValueError("bin_edges must be a one-dimensional tensor with at least two edges")
    if bin_edges.dtype != torch.int64:
        raise TypeError(f"bin_edges must be int64, got {bin_edges.dtype}")
    edges = [int(value) for value in bin_edges.tolist()]
    if any(edges[k + 1] < edges[k] for k in range(len(edges) - 1)):
        raise ValueError("bin_edges must be non-decreasing")

    spans = _merge_spans(
        [
            (item.start, item.end)
            for item in intervals
            if item.channel == channel and item.end > item.start
        ]
    )
    targets = torch.zeros(len(edges) - 1, dtype=torch.float32, device=bin_edges.device)
    if not spans:
        return targets
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        span = high - low
        if span <= 0:
            continue
        covered = 0
        for start, end in spans:
            if end <= low:
                continue
            if start >= high:
                break
            covered += min(high, end) - max(low, start)
        targets[index] = min(1.0, covered / span)
    return targets


__all__ = [
    "AnomalyXLSplit",
    "GoldInterval",
    "MAX_CHANNELS_PER_CALL",
    "MAX_CROP_SAMPLES",
    "MIN_CHANNELS_PER_CALL",
    "MIN_CROP_SAMPLES",
    "MODES",
    "ModeAssigner",
    "RowMeta",
    "TeacherCropSampler",
    "ZoomAction",
    "occupancy_targets",
    "parse_zoom_action",
    "stratified_subset",
]
