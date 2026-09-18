#!/usr/bin/env python3
"""Paired comparison of a va1r validation against va1, on exactly the same recordings.

Astra (16/09, punto 2): "Confrontare va1 e va1r sulle stesse righe di validazione: val104 contro test800 non dimostra
che la nuova architettura eguagli o superi va1."  This joins a va1r validation dump -- written by the trainer at
``runs/<id>/logs/val_<tag>_step<NNNNNN>.json`` with one record per recording -- against the archived va1 reference
rows, matching on recording id, and reports per-category means plus a paired bootstrap of the difference.

The va1 reference is `integrity/artifacts/l3_riferimenti/va1_val520_no_zoom_rows.jsonl`: va1's own predictions on the
520 validation recordings, no zoom, one JSON object per row with `id`, `category` and `primary`.

Usage:
  python tools/compare_with_va1.py --va1r runs/va1r/logs/val_quick_pass0_step004488.json \
      --va1 ../integrity/artifacts/l3_riferimenti/va1_val520_no_zoom_rows.jsonl [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000


def paired_bootstrap(values, seed: int = 5, n: int = BOOTSTRAP_RESAMPLES) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "ci95_lower": None, "ci95_upper": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n, arr.size))].mean(axis=1)
    return {"mean": float(arr.mean()), "ci95_lower": float(np.quantile(means, 0.025)),
            "ci95_upper": float(np.quantile(means, 0.975)), "n": int(arr.size)}


def has_anomaly(record) -> bool:
    """True when the gold answer of a trainer record describes a real anomaly."""

    try:
        gold = json.loads(record["gold"].split(":", 1)[-1])
    except (ValueError, TypeError, KeyError):
        return True
    if "present" in gold:
        return bool(gold["present"])
    if "anomalies" in gold:
        return bool(gold["anomalies"])
    return True


def decompose(records) -> dict:
    """Split each category into the easy half and the hard half (Astra 16/09, punto 1 e 5).

    The composite primary pays a yes/no on an empty row a full point and a decent window a fraction, so it can rise
    while perception does not move at all.  What must be reported instead: specificity (says empty when the row is
    empty), the false-empty rate on rows that do carry an anomaly, and the score on those rows -- IoU for localize,
    channel F1 for the multichannel question.
    """

    empty_marker = {"localize": '"present": false', "localize_all_channels": '"anomalies": []'}
    out = {}
    for category in sorted({r["category"] for r in records}):
        subset = [r for r in records if r["category"] == category]
        positives = [r for r in subset if has_anomaly(r)]
        negatives = [r for r in subset if not has_anomaly(r)]
        marker = empty_marker.get(category)
        said_empty = (lambda rows: sum(1 for r in rows if marker in r["pred"].replace("'", '"'))) if marker else None
        out[category] = {
            "n": len(subset),
            "primary": st.fmean(float(r["primary"]) for r in subset) if subset else None,
            "on_positives": st.fmean(float(r["primary"]) for r in positives) if positives else None,
            "on_negatives": st.fmean(float(r["primary"]) for r in negatives) if negatives else None,
            "n_positives": len(positives), "n_negatives": len(negatives),
            "specificity": (said_empty(negatives) / len(negatives)) if (said_empty and negatives) else None,
            "false_empty_rate": (said_empty(positives) / len(positives)) if (said_empty and positives) else None,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--va1r", type=Path, required=True, help="trainer validation dump of the new run")
    parser.add_argument("--va1", type=Path, required=True, help="va1 reference rows (jsonl)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    dump = json.loads(args.va1r.read_text(encoding="utf-8"))
    new_rows = {r["id"]: r for r in dump["records"]}
    reference = {}
    with args.va1.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            reference[row["id"]] = row

    shared = [rid for rid in new_rows if rid in reference]
    missing = len(new_rows) - len(shared)
    if not shared:
        raise SystemExit("no recording id is present in both files")

    per_category: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"va1": [], "va1r": [], "delta": []})
    for rid in shared:
        category = new_rows[rid]["category"]
        old, new = float(reference[rid]["primary"]), float(new_rows[rid]["primary"])
        bucket = per_category[category]
        bucket["va1"].append(old)
        bucket["va1r"].append(new)
        bucket["delta"].append(new - old)

    all_delta = [d for bucket in per_category.values() for d in bucket["delta"]]
    overall = {"rows": len(shared), "unmatched_rows": missing,
               "va1": st.fmean(v for bucket in per_category.values() for v in bucket["va1"]),
               "va1r": st.fmean(v for bucket in per_category.values() for v in bucket["va1r"]),
               "delta": paired_bootstrap(all_delta, args.seed)}
    categories = {name: {"n": len(bucket["va1"]), "va1": st.fmean(bucket["va1"]),
                         "va1r": st.fmean(bucket["va1r"]),
                         "delta": paired_bootstrap(bucket["delta"], args.seed)}
                  for name, bucket in sorted(per_category.items())}

    report = {"va1r_dump": str(args.va1r), "optimizer_step": dump["summary"].get("optimizer_step"),
              "tag": dump["summary"].get("tag"), "overall": overall, "by_category": categories,
              "decomposition": decompose(dump["records"])}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    step = report["optimizer_step"]
    print(f"VA1 COMPARISON step {step} on {len(shared)} shared recordings"
          + (f" ({missing} unmatched)" if missing else ""), flush=True)
    print(f"{'categoria':26s} {'n':>3s} {'va1':>8s} {'va1r':>8s} {'delta':>9s}  CI95")
    for name, stats in categories.items():
        d = stats["delta"]
        print(f"{name:26s} {stats['n']:3d} {stats['va1']:8.4f} {stats['va1r']:8.4f} "
              f"{d['mean']:+9.4f}  [{d['ci95_lower']:+.4f}; {d['ci95_upper']:+.4f}]")
    d = overall["delta"]
    print(f"{'COMPLESSIVO':26s} {overall['rows']:3d} {overall['va1']:8.4f} {overall['va1r']:8.4f} "
          f"{d['mean']:+9.4f}  [{d['ci95_lower']:+.4f}; {d['ci95_upper']:+.4f}]")

    print("")
    print("scomposizione facile/difficile (il composito non la mostra)")
    print(f"{'categoria':26s} {'tot':>7s} {'positivi':>9s} {'negativi':>9s} {'specif.':>8s} {'falsi vuoti':>12s}")
    for name, stats in report["decomposition"].items():
        def fmt(value, width=9):
            return f"{value:{width}.4f}" if value is not None else " " * (width - 1) + "-"
        print(f"{name:26s} {fmt(stats['primary'], 7)} {fmt(stats['on_positives'])} {fmt(stats['on_negatives'])} "
              f"{fmt(stats['specificity'], 8)} {fmt(stats['false_empty_rate'], 12)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
