#!/usr/bin/env python3
"""Deterministic paired bootstrap for v3 pilot generation and TF audits."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from typing import Callable


BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260909


def paired_bootstrap(values: list[float]) -> dict[str, object]:
    if not values:
        raise ValueError("paired bootstrap received no values")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("paired bootstrap received a non-finite value")
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(values)
    means = sorted(
        sum(rng.choices(values, k=count)) / count
        for _ in range(BOOTSTRAP_N)
    )
    low = means[int(0.025 * BOOTSTRAP_N)]
    high = means[int(0.975 * BOOTSTRAP_N)]
    return {
        "n": count,
        "mean_diff": sum(values) / count,
        "ci95": [low, high],
        "significant": low > 0.0 or high < 0.0,
    }


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"expected JSON object: {path}")
    return document


def _resolve_jsonl(summary_path: Path, summary: dict) -> Path:
    raw = summary.get("eval_jsonl")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"summary has no eval_jsonl: {summary_path}")
    candidate = Path(raw)
    choices = [candidate, Path.cwd() / candidate, summary_path.parent / candidate]
    for choice in choices:
        if choice.is_file():
            return choice.resolve()
    raise FileNotFoundError(f"cannot resolve eval_jsonl {raw!r} from {summary_path}")


def _load_generation(path: Path) -> tuple[dict, list[dict]]:
    summary = _load_json(path)
    jsonl_path = _resolve_jsonl(path, summary)
    with jsonl_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    expected = summary.get("samples_evaluated")
    if expected != len(rows):
        raise ValueError(f"coverage mismatch for {path}: summary={expected}, rows={len(rows)}")
    return summary, rows


def _identity(row: dict) -> tuple[object, object, object]:
    return row.get("id"), row.get("category"), row.get("ground_truth")


def _validate_same_rows(rows_by_condition: dict[str, list[dict]], expected: int) -> None:
    reference_name = next(iter(rows_by_condition))
    reference = rows_by_condition[reference_name]
    if len(reference) != expected:
        raise ValueError(f"expected {expected} rows, got {len(reference)} for {reference_name}")
    identities = [_identity(row) for row in reference]
    if any(identity[0] in (None, "") for identity in identities):
        raise ValueError(f"missing row id in {reference_name}")
    if len(set(identities)) != len(identities):
        raise ValueError(f"duplicate paired identity in {reference_name}")
    for name, rows in rows_by_condition.items():
        actual = [_identity(row) for row in rows]
        if actual != identities:
            raise ValueError(f"row identity/order mismatch: {reference_name} vs {name}")


def _comparison(
    rows_by_condition: dict[str, list[dict]],
    left: str,
    right: str,
    value: Callable[[dict], float],
) -> dict[str, object]:
    differences = [
        value(a) - value(b)
        for a, b in zip(rows_by_condition[left], rows_by_condition[right])
    ]
    by_category: dict[str, list[float]] = defaultdict(list)
    for row, difference in zip(rows_by_condition[left], differences):
        by_category[str(row.get("category"))].append(difference)
    return {
        "direction": f"{left} minus {right}",
        "primary": paired_bootstrap(differences),
        "per_category": {
            category: paired_bootstrap(values)
            for category, values in sorted(by_category.items())
        },
    }


def summarize_rows(
    rows_by_condition: dict[str, list[dict]],
    expected: int,
    value: Callable[[dict], float],
    metric: str,
) -> dict[str, object]:
    _validate_same_rows(rows_by_condition, expected)
    order = [pair for pair in (("on", "off"), ("on", "roll"), ("off", "roll"))
             if pair[0] in rows_by_condition and pair[1] in rows_by_condition]
    conditions = {}
    for name, rows in rows_by_condition.items():
        values = [value(row) for row in rows]
        by_category: dict[str, list[float]] = defaultdict(list)
        for row, item in zip(rows, values):
            by_category[str(row.get("category"))].append(item)
        conditions[name] = {
            "mean": sum(values) / len(values),
            "per_category": {
                category: sum(items) / len(items)
                for category, items in sorted(by_category.items())
            },
        }
    return {
        "metric": metric,
        "n": expected,
        "conditions": conditions,
        "comparisons": {
            f"{left}_minus_{right}": _comparison(
                rows_by_condition, left, right, value
            )
            for left, right in order
        },
    }


def generation_section(paths: dict[str, Path], expected: int) -> dict[str, object]:
    summaries = {}
    rows_by_condition = {}
    for condition, path in paths.items():
        summary, rows = _load_generation(path)
        summaries[condition] = summary
        rows_by_condition[condition] = rows
    checkpoints = {str(summary.get("checkpoint")) for summary in summaries.values()}
    if len(checkpoints) != 1 or None in {summary.get("checkpoint") for summary in summaries.values()}:
        raise ValueError(f"generation conditions do not share one checkpoint: {checkpoints}")
    if {summary.get("split") for summary in summaries.values()} != {"val"}:
        raise ValueError("paired generation inputs must all use the val split")
    section = summarize_rows(
        rows_by_condition, expected, lambda row: float(row.get("primary", 0.0)), "primary"
    )
    # Cross-check the official aggregate and per-category arithmetic.
    for condition, summary in summaries.items():
        observed = section["conditions"][condition]
        if not math.isclose(float(summary["primary"]), observed["mean"], abs_tol=1e-12):
            raise ValueError(f"official primary mismatch for {condition}: {paths[condition]}")
        official_categories = summary.get("category_primary")
        if not isinstance(official_categories, dict):
            raise ValueError(f"missing category_primary for {condition}: {paths[condition]}")
        if set(official_categories) != set(observed["per_category"]):
            raise ValueError(f"category set mismatch for {condition}: {paths[condition]}")
        if official_categories != observed["per_category"]:
            for category, value in official_categories.items():
                if not math.isclose(
                    float(value), observed["per_category"].get(category, math.nan), abs_tol=1e-12
                ):
                    raise ValueError(
                        f"official category_primary mismatch for {condition}/{category}"
                    )
    section["summaries"] = {name: str(path) for name, path in paths.items()}
    section["checkpoint"] = checkpoints.pop()
    return section


def teacher_forcing_section(path: Path, expected: int) -> dict[str, object]:
    document = _load_json(path)
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"teacher-forcing document has no rows: {path}")
    rows_by_condition: dict[str, list[dict]] = {}
    for condition, source in (("on", "on"), ("off", "off"), ("roll", "roll_tokens")):
        converted = []
        for row in rows:
            conditions = row.get("conditions", {})
            if source not in conditions:
                raise ValueError(f"teacher-forcing condition {source} absent: {path}")
            converted.append({
                "id": row.get("id"),
                "category": row.get("category"),
                "ground_truth": row.get("ground_truth"),
                "nll": conditions[source]["nll"],
            })
        rows_by_condition[condition] = converted
    section = summarize_rows(
        rows_by_condition, expected, lambda row: float(row["nll"]), "nll"
    )
    section["source"] = str(path)
    section["checkpoint"] = document.get("checkpoint")
    section["context_final_mean"] = document.get("context_final_mean", {})
    return section


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--generation-on", required=True, type=Path)
    parser.add_argument("--generation-off", required=True, type=Path)
    parser.add_argument("--generation-roll", required=True, type=Path)
    parser.add_argument("--teacher-forcing", required=True, type=Path)
    parser.add_argument("--full-on", required=True, type=Path)
    parser.add_argument("--full-off", required=True, type=Path)
    parser.add_argument("--full-roll", type=Path)
    parser.add_argument("--probe-n", type=int, default=80)
    parser.add_argument("--full-n", type=int, default=520)
    args = parser.parse_args()

    full_paths = {"on": args.full_on, "off": args.full_off}
    if args.full_roll is not None:
        full_paths["roll"] = args.full_roll
    generation_probe = generation_section({
        "on": args.generation_on,
        "off": args.generation_off,
        "roll": args.generation_roll,
    }, args.probe_n)
    teacher_forcing = teacher_forcing_section(args.teacher_forcing, args.probe_n)
    full_val = generation_section(full_paths, args.full_n)
    checkpoints = {
        generation_probe.get("checkpoint"),
        teacher_forcing.get("checkpoint"),
        full_val.get("checkpoint"),
    }
    if len(checkpoints) != 1 or None in checkpoints:
        raise ValueError(f"all paired sections must share one checkpoint: {checkpoints}")
    result = {
        "bootstrap": {"n_resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "ci": 0.95},
        "pairing": "exact ordered (id, category, ground_truth)",
        "checkpoint": checkpoints.pop(),
        "generation_probe80": generation_probe,
        "teacher_forcing_probe80": teacher_forcing,
        "full_val": full_val,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PAIRED_EVAL_OK {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
