#!/usr/bin/env python3
"""Operational Z1-bis gate on the complete 520-row validation split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from l3.grpo import _atomic_json_dump, load_l3_config
from l3.paired_eval_l3 import load_eval_jsonl, paired_report


ANOMALY_CATEGORIES = {
    "classify_with_evidence", "localize", "localize_all_channels",
}


def _payload(text: str) -> dict[str, Any] | None:
    start = str(text).find("{")
    if start < 0:
        return None
    depth = 0
    in_string = escaped = False
    for end, char in enumerate(str(text)[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(str(text)[start:end + 1])
                    return value if isinstance(value, dict) else None
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def _is_absent(category: str, text: str) -> bool | None:
    payload = _payload(text)
    if payload is None:
        return None
    if category == "localize":
        return payload.get("present") is False
    if category == "classify_with_evidence":
        # The precise validation currently contains only positive classify rows,
        # but keep the negative-label contract explicit for future pinned sets.
        return payload.get("kind") == "No anomaly"
    if category == "localize_all_channels":
        anomalies = payload.get("anomalies")
        return isinstance(anomalies, list) and len(anomalies) == 0
    return None


def behavioral_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("candidate evaluation is empty")
    total = len(rows)
    action = sum(row.get("zoom") is not None for row in rows)
    invalid = sum(bool(row.get("zoom_invalid")) for row in rows)
    empty = sum(bool(row.get("empty_final")) for row in rows)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    positives = negatives = false_negative = false_positive = 0
    positive_direct = negative_actions = 0
    unparsable_anomaly = 0
    for row in rows:
        category = str(row["category"])
        zoomed = row.get("zoom") is not None
        by_category[category]["rows"] += 1
        by_category[category]["action"] += int(zoomed)
        by_category[category]["direct"] += int(not zoomed)
        if category not in ANOMALY_CATEGORIES:
            continue
        gold_absent = _is_absent(category, str(row.get("ground_truth", "")))
        pred_absent = _is_absent(category, str(row.get("prediction", "")))
        if gold_absent is None:
            raise ValueError(f"cannot parse anomaly ground truth for {row.get('id')}")
        unparsable_anomaly += int(pred_absent is None)
        if gold_absent:
            negatives += 1
            false_positive += int(pred_absent is not True)
            negative_actions += int(zoomed)
        else:
            positives += 1
            false_negative += int(pred_absent is not False)
            positive_direct += int(not zoomed)
    if not positives or not negatives:
        raise ValueError("operational gate needs both positive and negative anomaly rows")
    return {
        "rows": total,
        "action_count": action, "action_rate": action / total,
        "direct_count": total - action, "direct_rate": (total - action) / total,
        "invalid_action_count": invalid, "invalid_action_rate": invalid / total,
        "empty_final_count": empty, "empty_final_rate": empty / total,
        "positive_rows": positives, "negative_rows": negatives,
        "false_negative_count_on_positives": false_negative,
        "false_negative_rate_on_positives": false_negative / positives,
        "false_positive_count_on_negatives": false_positive,
        "false_positive_rate_on_negatives": false_positive / negatives,
        "positive_direct_count": positive_direct,
        "positive_direct_rate": positive_direct / positives,
        "negative_action_count": negative_actions,
        "negative_action_rate": negative_actions / negatives,
        "unparsable_anomaly_predictions": unparsable_anomaly,
        "per_category_route": {
            category: {
                "rows": counts["rows"], "action_count": counts["action"],
                "action_rate": counts["action"] / counts["rows"],
                "direct_count": counts["direct"],
                "direct_rate": counts["direct"] / counts["rows"],
            }
            for category, counts in sorted(by_category.items())
        },
    }


def operational_report(
    candidate: Sequence[Mapping[str, Any]], va1: Sequence[Mapping[str, Any]],
    previous: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["operational_gate"]
    expected = int(gate["expected_rows"])
    if len(candidate) != expected:
        raise ValueError(f"candidate coverage {len(candidate)} != {expected}")
    seed = int(gate["seed"])
    resamples = int(gate["bootstrap_resamples"])
    vs_va1 = paired_report(
        va1, candidate, split="val", seed=seed, expected=expected, resamples=resamples,
    )
    vs_previous = paired_report(
        previous, candidate, split="val", seed=seed + 100, expected=expected,
        resamples=resamples,
    )
    behavior = behavioral_metrics(candidate)
    checks = {
        "coverage": len(candidate) == expected,
        "empty_rate": behavior["empty_final_rate"] <= float(gate["max_empty_rate"]),
        "invalid_action_rate": behavior["invalid_action_rate"] <= float(gate["max_invalid_action_rate"]),
        "min_action_rate": behavior["action_rate"] >= float(gate["min_action_rate"]),
        "max_action_rate": behavior["action_rate"] <= float(gate["max_action_rate"]),
        "false_negative_rate_on_positives": (
            behavior["false_negative_rate_on_positives"]
            <= float(gate["max_false_negative_rate_on_positives"])
        ),
        "false_positive_rate_on_negatives": (
            behavior["false_positive_rate_on_negatives"]
            <= float(gate["max_false_positive_rate_on_negatives"])
        ),
    }
    thresholds = gate["min_delta_vs_va1"]
    overall_delta = vs_va1["primary"]["candidate_minus_reference"]["mean"]
    checks["primary_vs_va1/overall"] = overall_delta >= float(thresholds["overall"])
    for category, minimum in thresholds.items():
        if category == "overall":
            continue
        if category not in vs_va1["per_category"]:
            raise ValueError(f"missing operational category {category}")
        delta = vs_va1["per_category"][category]["candidate_minus_reference"]["mean"]
        checks[f"primary_vs_va1/{category}"] = delta >= float(minimum)
    if not all(math.isfinite(float(row["primary"])) for row in candidate):
        raise ValueError("candidate contains non-finite primary values")
    return {
        "status": "DONE" if all(checks.values()) else "FAILED",
        "format": "l3_z1bis_operational_gate_v1",
        "gate_passed": all(checks.values()), "checks": checks,
        "thresholds": dict(gate),
        "behavior": behavior,
        "paired_vs_va1": vs_va1,
        "paired_vs_previous_vz1": vs_previous,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--va1-reference", required=True, type=Path)
    parser.add_argument("--previous-vz1-reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"operational-gate output already exists: {args.out}")
    config = load_l3_config(args.config)
    report = operational_report(
        load_eval_jsonl(args.candidate), load_eval_jsonl(args.va1_reference),
        load_eval_jsonl(args.previous_vz1_reference), config,
    )
    report["inputs"] = {
        "config": str(args.config.resolve()), "candidate": str(args.candidate.resolve()),
        "va1_reference": str(args.va1_reference.resolve()),
        "previous_vz1_reference": str(args.previous_vz1_reference.resolve()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_dump(args.out, report)
    if not report["gate_passed"]:
        raise RuntimeError("Z1-bis operational criterion failed")
    print(
        f"L3_Z1BIS_OPERATIONAL_OK primary_delta="
        f"{report['paired_vs_va1']['primary']['candidate_minus_reference']['mean']:+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
