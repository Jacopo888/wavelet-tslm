#!/usr/bin/env python3
"""Fail-closed paired bootstrap for the va1 reference and one L3 candidate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_SEED = 20260911
EXPECTED_COVERAGE = {"val": 520, "test": 800}


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("id", "")),
        str(row.get("category", "")),
        str(row.get("ground_truth", "")),
    )


def load_eval_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty eval JSONL: {path}")
    return rows


def bootstrap_mean(
    values: Sequence[float], *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> dict[str, Any]:
    if not values or resamples <= 0:
        raise ValueError("bootstrap requires values and a positive resample count")
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("bootstrap received a non-finite value")
    rng = random.Random(seed)
    count = len(numeric)
    means = sorted(
        sum(numeric[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )
    low = means[int(0.025 * resamples)]
    high = means[min(resamples - 1, int(0.975 * resamples))]
    return {"n": count, "mean": sum(numeric) / count, "ci95": [low, high]}


def paired_report(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    split: str,
    seed: int,
    expected: int | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    expected_rows = EXPECTED_COVERAGE[split] if expected is None else int(expected)
    if len(reference) != expected_rows or len(candidate) != expected_rows:
        raise ValueError(
            f"coverage mismatch for {split}: reference={len(reference)}, "
            f"candidate={len(candidate)}, expected={expected_rows}"
        )
    reference_ids = [_identity(row) for row in reference]
    candidate_ids = [_identity(row) for row in candidate]
    if any(not all(identity) for identity in reference_ids):
        raise ValueError("reference has an incomplete paired identity")
    if len(set(reference_ids)) != expected_rows:
        raise ValueError("reference has duplicate paired identities")
    if candidate_ids != reference_ids:
        raise ValueError("ordered (id, category, ground_truth) identity mismatch")

    ref_values: list[float] = []
    cand_values: list[float] = []
    categories: dict[str, list[int]] = defaultdict(list)
    for index, (ref_row, cand_row) in enumerate(zip(reference, candidate)):
        if "primary" not in ref_row or "primary" not in cand_row:
            raise ValueError(f"missing primary metric at row {index}")
        ref_value = float(ref_row["primary"])
        cand_value = float(cand_row["primary"])
        if not math.isfinite(ref_value) or not math.isfinite(cand_value):
            raise ValueError(f"non-finite primary metric at row {index}")
        ref_values.append(ref_value)
        cand_values.append(cand_value)
        categories[reference_ids[index][1]].append(index)

    def section(indices: Sequence[int], offset: int) -> dict[str, Any]:
        refs = [ref_values[index] for index in indices]
        cands = [cand_values[index] for index in indices]
        diffs = [candidate_value - reference_value
                 for reference_value, candidate_value in zip(refs, cands)]
        return {
            "n": len(indices),
            "reference": bootstrap_mean(refs, seed=seed + offset, resamples=resamples),
            "candidate": bootstrap_mean(cands, seed=seed + offset, resamples=resamples),
            "candidate_minus_reference": bootstrap_mean(
                diffs, seed=seed + offset, resamples=resamples
            ),
        }

    return {
        "format": "l3_paired_eval_v1",
        "split": split,
        "pairing": "exact ordered (id, category, ground_truth)",
        "coverage": {"reference": len(reference), "candidate": len(candidate)},
        "bootstrap": {"resamples": resamples, "seed": seed, "ci": 0.95},
        "primary": section(range(expected_rows), 0),
        "per_category": {
            category: section(indices, ordinal + 1)
            for ordinal, (category, indices) in enumerate(sorted(categories.items()))
        },
    }


def markdown_table(report: Mapping[str, Any]) -> str:
    rows = [("overall", report["primary"]), *report["per_category"].items()]
    lines = [
        f"# L3 paired evaluation — {report['split']}",
        "",
        "| Scope | n | Reference | Candidate | Delta | 95% CI delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, section in rows:
        delta = section["candidate_minus_reference"]
        lines.append(
            f"| {name} | {section['n']} | {section['reference']['mean']:.6f} | "
            f"{section['candidate']['mean']:.6f} | {delta['mean']:+.6f} | "
            f"[{delta['ci95'][0]:+.6f}, {delta['ci95'][1]:+.6f}] |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-markdown", required=True, type=Path)
    args = parser.parse_args()
    if args.out_json.exists() or args.out_markdown.exists():
        raise FileExistsError("paired-eval outputs must not already exist")
    report = paired_report(
        load_eval_jsonl(args.reference), load_eval_jsonl(args.candidate),
        split=args.split, seed=args.seed,
    )
    report["inputs"] = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.out_markdown.write_text(markdown_table(report), encoding="utf-8")
    print(f"L3_PAIRED_EVAL_OK {args.out_json} {args.out_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
