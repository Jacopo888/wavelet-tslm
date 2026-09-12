#!/usr/bin/env python3
"""Deterministic, CPU-testable trace planner for the Z1-bis zoom SFT pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

from l3.grpo import _atomic_json_dump, _runtime_imports, load_l3_config, sha256_file
from l3.zoom import ZoomAction, channel_names, gold_events, series_length


ZOOM_CATEGORIES = {
    "classify_with_evidence",
    "localize",
    "localize_all_channels",
}
DIRECT_CATEGORIES = {"measure_magnitude", "lead_lag_with_magnitude"}
ALL_CATEGORIES = ZOOM_CATEGORIES | DIRECT_CATEGORIES


@dataclass(frozen=True)
class Trace:
    index: int
    sample_id: str
    category: str
    source: str  # zoom | direct | replay
    action: ZoomAction | None = None
    event: dict[str, Any] | None = None
    position: float | None = None
    length_band: str | None = None
    centered: bool = False

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        if self.action is not None:
            row["action"] = asdict(self.action)
        return row


def _stable_rng(seed: int, *parts: Any) -> random.Random:
    identity = ":".join([str(seed), *(str(part) for part in parts)])
    value = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big")
    return random.Random(value)


def compact_trace_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only trace metadata; ``range`` preserves channel length without storing values."""
    length = series_length(sample)
    names = channel_names(sample)
    return {
        "id": str(sample.get("id", "")), "category": str(sample["category"]),
        "length": length, "channel_names": names,
        "time_series": [range(length) for _ in names],
        "answer": str(sample.get("answer", "")),
    }


def _weighted_counts(total: int, choices: Sequence[tuple[int, float]]) -> dict[int, int]:
    if total < 0 or not choices:
        raise ValueError("invalid weighted allocation")
    weights = [float(weight) for _, weight in choices]
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("window weights must be finite and positive")
    weight_sum = sum(weights)
    raw = [total * weight / weight_sum for weight in weights]
    counts = [math.floor(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(choices)), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return {choices[i][0]: counts[i] for i in range(len(choices))}


def _trace_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    sft = config.get("zoom_sft", {})
    if sft.get("trace_contract") != "z1bis_v1":
        raise ValueError("zoom_sft.trace_contract must be z1bis_v1")
    categories = sft.get("categories")
    if not isinstance(categories, Mapping) or set(categories) != ALL_CATEGORIES:
        raise ValueError(f"zoom_sft.categories must declare exactly {sorted(ALL_CATEGORIES)}")
    for category, rule in categories.items():
        if not isinstance(rule, Mapping):
            raise ValueError(f"invalid category rule for {category}")
        replay = float(rule.get("replay_fraction", math.nan))
        if not 0.0 <= replay <= 1.0:
            raise ValueError(f"invalid replay fraction for {category}")
        expected = "direct" if category in DIRECT_CATEGORIES else "zoom_if_positive"
        if rule.get("non_replay_route") != expected:
            raise ValueError(f"invalid non-replay route for {category}")
        quota_keys = (
            ("direct_fraction_of_non_replay",)
            if category in DIRECT_CATEGORIES else
            ("zoom_fraction_of_non_replay_positives",
             "direct_fraction_of_non_replay_negatives")
        )
        if any(float(rule.get(key, math.nan)) != 1.0 for key in quota_keys):
            raise ValueError(f"{category} conditional route quotas must be exactly 1.0")
    choices = sft.get("window_sizes")
    if not isinstance(choices, list) or not choices:
        raise ValueError("zoom_sft.window_sizes must be a non-empty list")
    samples = [int(item["samples"]) for item in choices]
    if samples != sorted(set(samples)) or samples[0] < 1:
        raise ValueError("window sizes must be unique, positive, and sorted")
    _weighted_counts(1, [(int(item["samples"]), float(item["weight"])) for item in choices])
    position = sft.get("position", {})
    if not (0 <= float(position.get("centered_radius", -1)) < 0.5):
        raise ValueError("position.centered_radius must be in [0, 0.5)")
    if not (0 <= float(position.get("centered_max_fraction", -1)) <= 0.15):
        raise ValueError("position.centered_max_fraction must be in [0, 0.15]")
    mean_range = position.get("mean_range")
    if not (isinstance(mean_range, list) and len(mean_range) == 2
            and 0 <= float(mean_range[0]) <= float(mean_range[1]) <= 1):
        raise ValueError("position.mean_range must be [low, high]")
    bins = position.get("histogram_bins")
    if not (isinstance(bins, list) and len(bins) >= 2 and bins == sorted(set(bins))
            and float(bins[0]) == 0.0 and float(bins[-1]) == 1.0):
        raise ValueError("position.histogram_bins must increase from 0 to 1")
    bands = sft.get("length_bands")
    if not isinstance(bands, list) or not bands:
        raise ValueError("zoom_sft.length_bands must be declared")
    maxima = [int(band["max_length"]) for band in bands]
    if maxima != sorted(set(maxima)):
        raise ValueError("length-band maxima must be unique and increasing")
    return sft


def length_band(length: int, sft: Mapping[str, Any]) -> str:
    for band in sft["length_bands"]:
        if length <= int(band["max_length"]):
            return str(band["name"])
    raise ValueError(f"series length {length} is outside declared length bands")


def sample_containing_action(
    sample: Mapping[str, Any], event: Mapping[str, Any], width: int, rng: random.Random,
) -> tuple[ZoomAction, float]:
    """Uniformly sample an integer crop start among all starts containing ``event``."""
    length = series_length(sample)
    start, end = int(event["start"]), int(event["end"])
    if not (0 <= start < end <= length) or width < end - start or width > length:
        raise ValueError("declared crop width cannot contain the complete gold event")
    low = max(0, end - width)
    high = min(start, length - width)
    if low > high:
        raise ValueError("no in-bounds crop can contain the complete gold event")
    crop_start = rng.randint(low, high)
    action = ZoomAction(str(event["channel"]), crop_start, crop_start + width, length)
    position = (((start + end) / 2.0) - crop_start) / width
    if not 0.0 <= position <= 1.0:
        raise RuntimeError("internal-position invariant failed")
    return action, position


def random_gold_action(
    sample: Mapping[str, Any], sft: Mapping[str, Any], rng: random.Random,
    *, width: int | None = None,
) -> tuple[ZoomAction, dict[str, Any], float]:
    events = gold_events(sample)
    if not events:
        raise ValueError("zoom traces require at least one gold interval")
    event = dict(rng.choice(sorted(
        events,
        key=lambda item: (str(item["channel"]), int(item["start"]), int(item["end"])),
    )))
    if width is None:
        choices = [(int(item["samples"]), float(item["weight"])) for item in sft["window_sizes"]]
        eligible = [(size, weight) for size, weight in choices
                    if size >= int(event["end"]) - int(event["start"])
                    and size <= series_length(sample)]
        if not eligible:
            raise ValueError("no declared window size can contain the selected gold event")
        width = rng.choices([item[0] for item in eligible], [item[1] for item in eligible], k=1)[0]
    action, position = sample_containing_action(sample, event, int(width), rng)
    return action, event, position


def _choose_replay_indices(
    samples: Sequence[Mapping[str, Any]], sft: Mapping[str, Any], seed: int,
) -> set[int]:
    by_category: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_category[str(sample["category"])].append(index)
    chosen: set[int] = set()
    for category in sorted(by_category):
        pool = by_category[category]
        _stable_rng(seed, category, "replay").shuffle(pool)
        count = round(len(pool) * float(sft["categories"][category]["replay_fraction"]))
        chosen.update(pool[:count])
    return chosen


def build_trace_plan(
    samples: Sequence[Mapping[str, Any]], config: Mapping[str, Any], seed: int,
) -> list[Trace]:
    """Build one deterministic route per train row; no heuristic/pseudo-recovery exists."""
    sft = _trace_config(config)
    if not samples:
        raise ValueError("cannot plan an empty dataset")
    replay = _choose_replay_indices(samples, sft, seed)
    provisional: dict[int, tuple[dict[str, Any], random.Random]] = {}
    for index, sample in enumerate(samples):
        category = str(sample.get("category", ""))
        if category not in ALL_CATEGORIES:
            raise ValueError(f"unexpected category at index {index}: {category}")
        if index in replay or category in DIRECT_CATEGORIES or not gold_events(sample):
            continue
        rng = _stable_rng(seed, sample.get("id", ""), index, "zoom")
        events = sorted(gold_events(sample), key=lambda e: (
            str(e["channel"]), int(e["start"]), int(e["end"]),
        ))
        provisional[index] = (dict(rng.choice(events)), rng)

    widths_by_index: dict[int, int] = {}
    choices = [(int(item["samples"]), float(item["weight"])) for item in sft["window_sizes"]]
    zoom_by_category: dict[str, list[int]] = defaultdict(list)
    for index in provisional:
        zoom_by_category[str(samples[index]["category"])].append(index)
    for category, indices in sorted(zoom_by_category.items()):
        counts = _weighted_counts(len(indices), choices)
        bag = [width for width, _ in choices for _ in range(counts[width])]
        _stable_rng(seed, category, "widths").shuffle(bag)
        ordered = list(indices)
        _stable_rng(seed, category, "rows").shuffle(ordered)
        for index, width in zip(ordered, bag):
            event, _ = provisional[index]
            if width < int(event["end"]) - int(event["start"]) or width > series_length(samples[index]):
                raise ValueError(
                    f"assigned window {width} cannot contain event at train index {index}; "
                    "no fallback is permitted"
                )
            widths_by_index[index] = width

    traces: list[Trace] = []
    radius = float(sft["position"]["centered_radius"])
    for index, sample in enumerate(samples):
        category = str(sample["category"])
        common = dict(index=index, sample_id=str(sample.get("id", "")), category=category)
        if index in replay:
            traces.append(Trace(**common, source="replay"))
        elif index not in provisional:
            if category in ZOOM_CATEGORIES and gold_events(sample):
                raise RuntimeError("positive zoom example was routed directly")
            traces.append(Trace(**common, source="direct"))
        else:
            event, rng = provisional[index]
            action, position = sample_containing_action(
                sample, event, widths_by_index[index], rng,
            )
            traces.append(Trace(
                **common, source="zoom", action=action, event=event,
                position=position, length_band=length_band(series_length(sample), sft),
                centered=abs(position - 0.5) <= radius,
            ))
    validate_trace_plan(samples, traces, config)
    return traces


def _histogram(values: Sequence[float], edges: Sequence[float]) -> list[int]:
    counts = [0] * (len(edges) - 1)
    for value in values:
        index = len(counts) - 1 if value == edges[-1] else next(
            i for i in range(len(counts)) if edges[i] <= value < edges[i + 1]
        )
        counts[index] += 1
    return counts


def summarize_trace_plan(traces: Sequence[Trace], config: Mapping[str, Any]) -> dict[str, Any]:
    sft = _trace_config(config)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    width_by_category: dict[str, Counter[int]] = defaultdict(Counter)
    position_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    centered: dict[tuple[str, str], int] = defaultdict(int)
    for trace in traces:
        by_category[trace.category][trace.source] += 1
        if trace.source == "zoom":
            assert trace.action is not None and trace.position is not None and trace.length_band
            width_by_category[trace.category][trace.action.window_len] += 1
            key = (trace.category, trace.length_band)
            position_values[key].append(trace.position)
            centered[key] += int(trace.centered)
    edges = [float(value) for value in sft["position"]["histogram_bins"]]
    distribution = {}
    for (category, band), values in sorted(position_values.items()):
        distribution[f"{category}/{band}"] = {
            "n": len(values), "mean": fmean(values), "std": pstdev(values),
            "centered": centered[(category, band)],
            "centered_fraction": centered[(category, band)] / len(values),
            "histogram_edges": edges, "histogram_counts": _histogram(values, edges),
        }
    return {
        "rows": len(traces),
        "source_counts": {category: dict(sorted(counts.items()))
                          for category, counts in sorted(by_category.items())},
        "window_counts": {category: {str(width): count for width, count in sorted(counts.items())}
                          for category, counts in sorted(width_by_category.items())},
        "position_distribution": distribution,
        "pseudo_recovery_count": 0,
    }


def validate_trace_plan(
    samples: Sequence[Mapping[str, Any]], traces: Sequence[Trace], config: Mapping[str, Any],
) -> dict[str, Any]:
    sft = _trace_config(config)
    if len(samples) != len(traces) or [trace.index for trace in traces] != list(range(len(samples))):
        raise ValueError("trace plan must cover every train row exactly once in dataset order")
    tolerance = float(sft.get("quota_tolerance", 0.02))
    for trace, sample in zip(traces, samples):
        if (trace.sample_id, trace.category) != (str(sample.get("id", "")), str(sample["category"])):
            raise ValueError(f"trace identity mismatch at index {trace.index}")
        events = gold_events(sample)
        if trace.source == "zoom":
            if trace.action is None or trace.event is None or not events:
                raise ValueError("zoom trace lacks a real gold event/action")
            event = trace.event
            if not (trace.action.channel == str(event["channel"])
                    and trace.action.start <= int(event["start"])
                    and trace.action.end >= int(event["end"])):
                raise ValueError("zoom action does not contain its complete gold event")
        elif trace.source == "direct":
            if trace.category not in DIRECT_CATEGORIES and events:
                raise ValueError("pseudo-recovery/direct positive trace is forbidden")
            if trace.action is not None:
                raise ValueError("direct trace unexpectedly has an action")
        elif trace.source == "replay":
            if trace.action is not None:
                raise ValueError("original-format replay unexpectedly has an action")
        else:
            raise ValueError(f"unknown trace source {trace.source}")

    summary = summarize_trace_plan(traces, config)
    observed_category_quotas: dict[str, dict[str, float]] = {}
    for category, rule in sft["categories"].items():
        counts = summary["source_counts"].get(category, {})
        total = sum(counts.values())
        observed = counts.get("replay", 0) / max(1, total)
        target = float(rule["replay_fraction"])
        if abs(observed - target) > tolerance:
            raise ValueError(
                f"{category} replay quota {observed:.4f} differs from {target:.4f} "
                f"by more than {tolerance:.4f}"
            )
        category_rows = [(sample, trace) for sample, trace in zip(samples, traces)
                         if trace.category == category]
        quota = {"replay_fraction": observed}
        if category in DIRECT_CATEGORIES:
            eligible = [trace for _, trace in category_rows if trace.source != "replay"]
            quota["direct_fraction_of_non_replay"] = (
                sum(trace.source == "direct" for trace in eligible) / len(eligible)
                if eligible else 1.0
            )
        else:
            positives = [trace for sample, trace in category_rows
                         if trace.source != "replay" and gold_events(sample)]
            negatives = [trace for sample, trace in category_rows
                         if trace.source != "replay" and not gold_events(sample)]
            quota["zoom_fraction_of_non_replay_positives"] = (
                sum(trace.source == "zoom" for trace in positives) / len(positives)
                if positives else 1.0
            )
            quota["direct_fraction_of_non_replay_negatives"] = (
                sum(trace.source == "direct" for trace in negatives) / len(negatives)
                if negatives else 1.0
            )
        for key, value in quota.items():
            target_value = float(rule[key])
            if abs(value - target_value) > tolerance:
                raise ValueError(
                    f"{category} {key} {value:.4f} differs from {target_value:.4f} "
                    f"by more than {tolerance:.4f}"
                )
        observed_category_quotas[category] = quota

    weights = {int(item["samples"]): float(item["weight"]) for item in sft["window_sizes"]}
    weight_sum = sum(weights.values())
    for category, counts in summary["window_counts"].items():
        total = sum(counts.values())
        for width, weight in weights.items():
            observed = counts.get(str(width), 0) / max(1, total)
            target = weight / weight_sum
            if abs(observed - target) > tolerance:
                raise ValueError(
                    f"{category} width-{width} quota {observed:.4f} differs from "
                    f"{target:.4f} by more than {tolerance:.4f}"
                )

    position = sft["position"]
    low, high = map(float, position["mean_range"])
    for stratum, stats in summary["position_distribution"].items():
        if not low <= stats["mean"] <= high:
            raise ValueError(f"{stratum} internal-position mean outside [{low}, {high}]")
        if stats["std"] < float(position["min_std"]):
            raise ValueError(f"{stratum} internal-position std below {position['min_std']}")
        if stats["centered_fraction"] > float(position["centered_max_fraction"]):
            raise ValueError(f"{stratum} centered fraction exceeds declared cap")
    expected_strata = {
        (trace.category, trace.length_band) for trace in traces if trace.source == "zoom"
    }
    if len(summary["position_distribution"]) != len(expected_strata):
        raise RuntimeError("position-distribution coverage is incomplete")
    summary["declared_category_quotas"] = {
        category: dict(rule) for category, rule in sft["categories"].items()
    }
    summary["observed_category_quotas"] = observed_category_quotas
    return summary


def write_trace_plan(
    out_dir: Path, samples: Sequence[Mapping[str, Any]], config: Mapping[str, Any], seed: int,
    *, train_sha256: str,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"trace output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    try:
        traces = build_trace_plan(samples, config, seed)
        rows_path = out_dir / "traces.jsonl"
        with rows_path.open("x", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(json.dumps(trace.to_json(), sort_keys=True) + "\n")
        validated = validate_trace_plan(samples, traces, config)
        summary = {
            "status": "DONE", "format": "l3_zoom_z1bis_trace_manifest_v1",
            "seed": seed, "train_sha256": train_sha256,
            "traces_sha256": sha256_file(rows_path),
            **validated,
        }
        _atomic_json_dump(out_dir / "summary.json", summary)
        _atomic_json_dump(out_dir / "DONE", summary)
        return summary
    except Exception as exc:
        _atomic_json_dump(out_dir / "FAILED", {
            "status": "FAILED", "error_type": type(exc).__name__, "reason": str(exc),
        })
        raise


def load_trace_plan(path: Path) -> list[Trace]:
    traces = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            action = row.get("action")
            row["action"] = ZoomAction(**action) if action is not None else None
            traces.append(Trace(**row))
    return traces


def _load_train_samples(config: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], str]:
    ExperimentConfig, _, _, _, DatasetClass = _runtime_imports(config)
    experiment = ExperimentConfig.from_dict({k: v for k, v in config.items() if not k.startswith("_")})
    kwargs = dict(experiment.dataset.extra_kwargs)
    kwargs["EOS_TOKEN"] = ""
    dataset = DatasetClass(split="train", **kwargs)
    expected = int(config["rl"]["expected_train_rows"])
    if len(dataset) != expected:
        raise RuntimeError(f"expected {expected} train rows, found {len(dataset)}")
    train_path = Path(dataset.train_dir) / "train.parquet"
    observed = sha256_file(train_path)
    if observed != config["rl"]["train_sha256"]:
        raise RuntimeError(f"train parquet SHA mismatch: {observed}")
    return [compact_trace_sample(dataset[index]) for index in range(len(dataset))], observed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    config = load_l3_config(args.config)
    sft = _trace_config(config)
    samples, train_sha = _load_train_samples(config)
    summary = write_trace_plan(
        args.out_dir.expanduser().resolve(), samples, config,
        int(sft.get("seed", config.get("seed", 42))), train_sha256=train_sha,
    )
    print(f"L3_Z1BIS_TRACES_OK rows={summary['rows']} out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
