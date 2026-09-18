#!/usr/bin/env python3
"""Does the salient selection land on the true anomaly?

Astra (16/09) accepted that the resolution hypothesis is plausible but not proven by a token count: the global view
carries 256 tokens per channel -- 64 approximation, 64 detail coverage, 64 salient detail, 64 raw -- chosen by robust
band magnitude, not 4096 uniform segments.  The open question is empirical: do the supports of the salient tokens
cover the gold window?

The selection is NOT learned (``tokenizer.learned_selection`` is false) and the Haar pyramid is a fixed transform, so
the supports are the same for any checkpoint and this runs on CPU with a freshly built frontend.

For every ``localize`` row whose gold answer carries an anomaly this reports, per token family:
  hit       fraction of rows where at least one valid token's support overlaps the gold window
  best IoU  how tightly the best single token isolates that window (1.0 = the support IS the window)
  cover     fraction of the gold window covered by the union of that family's supports

Usage:
  python tools/salience_coverage.py --config configs/va1r_mi300x.toml --rows 120 [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import AnomalyXLSplit  # noqa: E402
from wavelet_rope.signal_contract import ViewKind, compute_recording_normalization, make_view  # noqa: E402
from wavelet_rope.tokenizer import (  # noqa: E402
    FAMILY_A,
    FAMILY_D_COVERAGE,
    FAMILY_D_SALIENT,
    FAMILY_RAW,
    PerceptualFrontend,
)

FAMILIES = (("A", FAMILY_A), ("D_coverage", FAMILY_D_COVERAGE),
            ("D_salient", FAMILY_D_SALIENT), ("raw", FAMILY_RAW))


def gold_window(meta) -> tuple[int, int] | None:
    """Absolute sample bounds of the anomaly, or None when the row carries no window."""

    try:
        gold = json.loads(meta.answer.split(":", 1)[-1])
    except (ValueError, TypeError):
        return None
    if not gold.get("present", True):
        return None
    start, end = gold.get("start"), gold.get("end")
    if start is None or end is None or int(end) <= int(start):
        return None
    return int(start), int(end)


def union_covered(spans: list[tuple[int, int]], start: int, end: int) -> int:
    """Samples of ``[start, end)`` covered by the union of ``spans``."""

    clipped = sorted((max(a, start), min(b, end)) for a, b in spans if min(b, end) > max(a, start))
    covered, reach = 0, start
    for a, b in clipped:
        a = max(a, reach)
        if b > a:
            covered += b - a
            reach = b
    return covered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--category", default="localize")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_runtime_config(args.config, validate_paths=False)
    raw, spec = cfg.as_dict(), dict(cfg.spec)
    root = Path(raw["dataset"]["root"])
    split = AnomalyXLSplit(root / raw["dataset"]["validation_relative_path"], "validation",
                           int(raw["dataset"]["validation_rows"]))
    frontend = PerceptualFrontend(spec).eval()

    per_family: dict[str, dict[str, list[float]]] = {
        name: {"hit": [], "iou": [], "cover": []} for name, _ in FAMILIES}
    widths: list[int] = []
    records = []
    with torch.no_grad():
        for meta in split.metas():
            if meta.category != args.category:
                continue
            window = gold_window(meta)
            if window is None:
                continue
            start, end = window
            episode = split.episode(meta.row_index)
            tokens = frontend.encode_view(
                make_view(episode, compute_recording_normalization(episode), kind=ViewKind.GLOBAL),
                recording_length=int(episode.values.shape[-1]),
            )
            widths.append(end - start)
            row = {"id": meta.id, "window": [start, end], "width": end - start, "families": {}}
            for name, family in FAMILIES:
                keep = tokens.valid & (tokens.family == family)
                spans = [(int(a), int(b)) for a, b in
                         zip(tokens.support_start[keep].tolist(), tokens.support_end[keep].tolist()) if b > a]
                overlaps = [(a, b) for a, b in spans if b > start and a < end]
                best = 0.0
                for a, b in overlaps:
                    inter = min(b, end) - max(a, start)
                    union = max(b, end) - min(a, start)
                    best = max(best, inter / union if union > 0 else 0.0)
                cover = union_covered(spans, start, end) / (end - start)
                per_family[name]["hit"].append(1.0 if overlaps else 0.0)
                per_family[name]["iou"].append(best)
                per_family[name]["cover"].append(cover)
                row["families"][name] = {"tokens": len(spans), "overlapping": len(overlaps),
                                         "best_iou": round(best, 4), "cover": round(cover, 4)}
            records.append(row)
            if len(records) >= args.rows:
                break

    summary = {name: {"hit_rate": round(st.fmean(v["hit"]), 4),
                      "best_iou_median": round(st.median(v["iou"]), 4),
                      "best_iou_mean": round(st.fmean(v["iou"]), 4),
                      "cover_median": round(st.median(v["cover"]), 4)}
               for name, v in per_family.items() if v["hit"]}
    report = {"config": args.config, "category": args.category, "rows": len(records),
              "gold_width_median": st.median(widths) if widths else None, "summary": summary,
              "records": records}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SALIENCE_COVERAGE {args.category}: {len(records)} rows, gold width median "
          f"{report['gold_width_median']}", flush=True)
    for name, stats in summary.items():
        print(f"  {name:12s} hit {stats['hit_rate']:.3f}  best IoU median {stats['best_iou_median']:.4f} "
              f"(mean {stats['best_iou_mean']:.4f})  cover median {stats['cover_median']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
