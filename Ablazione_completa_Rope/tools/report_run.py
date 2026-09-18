#!/usr/bin/env python3
"""Summarize a training run directory (logs/train.jsonl + logs/val_*.json) as Markdown + JSON."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--last", type=int, default=100, help="window (examples) for the recent loss")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--train-parquet", type=Path, default=None,
                    help="recompute crop kinds (positive_full / positive_partial / negative) from the logged actions when the "
                         "run predates crop_kind logging (Astra 15/09 §4)")
    args = ap.parse_args()
    log = args.run_dir / "logs" / "train.jsonl"
    examples, steps, vals, failed = [], [], [], None
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = rec.get("event")
            if ev == "example":
                examples.append(rec)
            elif ev == "step":
                steps.append(rec)
            elif ev == "validation":
                vals.append(rec)
            elif ev == "failed":
                failed = rec
    by = defaultdict(list)
    for e in examples:
        by[(e["category"], e["mode"])].append(e["loss"])
    # effective teacher-crop kinds per category (Astra 15/09 §4)
    if args.train_parquet is not None and any(e.get("mode") == "fixed_crop" and e.get("crop_kind") is None for e in examples):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from wavelet_rope.data import AnomalyXLSplit, parse_zoom_action
        split = AnomalyXLSplit(args.train_parquet, "train")
        metas = {m.id: m for m in split.metas()}
        for e in examples:
            if e.get("mode") != "fixed_crop" or e.get("crop_kind") is not None or not e.get("actions"):
                continue
            meta = metas.get(e["id"])
            action = parse_zoom_action(e["actions"][0], channel_names=meta.channel_names, length=meta.length) if meta else None
            if meta is None or action is None:
                continue
            kinds = set()
            for g in meta.gold_intervals:
                if g.channel not in action.channels or g.end <= g.start:
                    continue
                if min(g.end, action.end) - max(g.start, action.start) <= 0:
                    continue
                kinds.add("positive_full" if (g.start >= action.start and g.end <= action.end) else "positive_partial")
            e["crop_kind"] = "positive_full" if "positive_full" in kinds else ("positive_partial" if kinds else "negative")
    crop_kinds = defaultdict(lambda: defaultdict(int))
    for e in examples:
        if e.get("mode") == "fixed_crop" and e.get("crop_kind"):
            crop_kinds[e["category"]][e["crop_kind"]] += 1
    recent = examples[-args.last:]
    out = {
        "run_dir": str(args.run_dir), "examples": len(examples), "optimizer_steps": steps[-1]["optimizer_step"] if steps else 0,
        "loss_first": round(statistics.fmean(e["loss"] for e in examples[: args.last]), 4) if examples else None,
        "loss_recent": round(statistics.fmean(e["loss"] for e in recent), 4) if recent else None,
        "seconds_per_example_recent": round(statistics.fmean(e["seconds"] for e in recent), 2) if recent else None,
        "tokens_mean_recent": round(statistics.fmean(e["tokens"] for e in recent), 0) if recent else None,
        "lambda": examples[-1]["lambda"] if examples else None,
        # Astra 15/09 (scala): drift of the soft-token RMS under a fixed projector gain (target = text-embedding RMS)
        "soft_rms": {"first": round(statistics.fmean(e["soft_rms"] for e in examples[: args.last] if "soft_rms" in e), 5),
                     "recent": round(statistics.fmean(e["soft_rms"] for e in recent if "soft_rms" in e), 5),
                     "min": round(min(e["soft_rms"] for e in examples if "soft_rms" in e), 5),
                     "max": round(max(e["soft_rms"] for e in examples if "soft_rms" in e), 5)}
        if any("soft_rms" in e for e in examples) else None,
        "by_category_mode": {f"{c}|{m}": {"n": len(v), "loss_mean": round(statistics.fmean(v), 4),
                                          "loss_recent": round(statistics.fmean(v[-25:]), 4)} for (c, m), v in sorted(by.items())},
        "crop_kinds_by_category": {c: dict(v) for c, v in sorted(crop_kinds.items())},
        "validations": [{k: r[k] for k in ("tag", "rows", "primary", "by_category", "invalid_format", "invalid_format_rate", "fixed_crop_loss", "optimizer_step") if k in r} for r in vals],
        "failed": failed,
        "done": (args.run_dir / "DONE").is_file(),
    }
    md = [f"# Run {args.run_dir.name}", "",
          f"esempi {out['examples']}, optimizer step {out['optimizer_steps']}, loss iniziale {out['loss_first']} → recente {out['loss_recent']}, "
          f"{out['seconds_per_example_recent']} s/esempio, lambda {out['lambda']}", "", "| categoria | modo | n | loss media | loss recente |", "|---|---|---:|---:|---:|"]
    for key, v in out["by_category_mode"].items():
        c, m = key.split("|")
        md.append(f"| {c} | {m} | {v['n']} | {v['loss_mean']} | {v['loss_recent']} |")
    if crop_kinds:
        md += ["", "| categoria | positive_full | positive_partial | negative |", "|---|---:|---:|---:|"]
        for c, v in sorted(crop_kinds.items()):
            md.append(f"| {c} | {v.get('positive_full', 0)} | {v.get('positive_partial', 0)} | {v.get('negative', 0)} |")
    if vals:
        md += ["", "| validazione | righe | primary | per categoria | formati invalidi | loss fixed-crop |", "|---|---:|---:|---|---:|---:|"]
        for r in out["validations"]:
            md.append(f"| {r['tag']} @ step {r['optimizer_step']} | {r['rows']} | {r['primary']} | {r['by_category']} | {r.get('invalid_format')} | {r.get('fixed_crop_loss', '')} |")
    if failed:
        md += ["", f"**FAILED** al pass {failed['pass']} posizione {failed['position']}: {failed['error']}"]
    text = "\n".join(md)
    print(text)
    if args.out:
        args.out.write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
