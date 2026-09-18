#!/usr/bin/env python3
"""Gate of stage 1 of P1-e (Astra 15/09, D2): are the soft tokens readable on recordings never trained on?

Loads one checkpoint, encodes the global view of N validation recordings (never used for fitting) and evaluates the two
linear supervision heads against the signal-derived targets.  Reports, in interpretable units:
  position MAE in bins   (threshold: <= 1 bin)
  top-3 recall           (fraction of the three highest-energy bins of the view that appear among the three highest
                          PREDICTED ones; threshold: >= 0.70)
plus the Pearson correlations of energy and peak.  Exit code 0 only when both thresholds are met.

Usage:
  python tools/token_supervision_eval.py --config configs/p1e1_mi300x.toml --checkpoint <dir> --out <json> [--rows 60]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safetensors.torch import load_file  # noqa: E402

from wavelet_rope.backbone import load_backbone  # noqa: E402
from wavelet_rope.checkpoint import inspect_foreign_checkpoint  # noqa: E402
from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import AnomalyXLSplit, stratified_subset  # noqa: E402
from wavelet_rope.model import WRAgentModel  # noqa: E402

POSITION_MAE_MAX_BINS = 1.0
TOP3_RECALL_MIN = 0.70


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--seed", type=int, default=29)
    args = ap.parse_args()
    t0 = time.time()

    cfg = load_runtime_config(args.config, validate_paths=False)
    raw, spec = cfg.as_dict(), dict(cfg.spec)
    device = torch.device("cuda:0")
    rt = raw["runtime"]
    bb = load_backbone(raw["backbone"]["snapshot"]["root"], backend=rt["attention_backend"],
                       pad_to=int(rt["attention_qk_pad_to"]), device=device, gradient_checkpointing=False)
    training = spec["training"]
    model = WRAgentModel(bb, spec, segment_weights=dict(training["segment_loss_weights"]), aux_weight=0.0)
    model.frontend.to(device); model.mixer.to(device); model.projector.to(device)
    if model.token_supervision is None:
        raise SystemExit("this config has no token supervision enabled")
    model.token_supervision.to(device)
    ref = inspect_foreign_checkpoint(args.checkpoint)
    model.load_trainable_state_dict(load_file(str(ref.path / "model.safetensors")))
    bb.ctx.lam = float(ref.trainer_state["wildcard_lambda"])
    model.eval(); bb.text.eval()

    root = Path(raw["dataset"]["root"])
    val = AnomalyXLSplit(root / raw["dataset"]["validation_relative_path"], "validation", 520)
    rows = stratified_subset(val.metas(), args.rows, args.seed)
    records = []
    with torch.no_grad():
        for row in rows:
            meta = val.meta(row)
            episode = val.episode(row)
            episode = type(episode)(episode.values.to(device), episode.valid.to(device), episode.channel_ids,
                                    episode.sample_times.to(device), episode.task)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                report = model.forward_loss(meta, episode, mode="direct", actions=[], answer=meta.answer,
                                            gold=meta.gold_intervals, supervise_action=False,
                                            reading=model.reading_text(episode, []))
            records.append({"id": meta.id, "category": meta.category,
                            "position_mae_bins": report.token_pos_mae_bins,
                            "top3_recall": report.token_top3_recall,
                            "energy_pearson": report.token_energy_pearson,
                            "peak_pearson": report.token_peak_pearson,
                            "position_loss": report.token_pos_loss, "energy_loss": report.token_energy_loss,
                            "peak_loss": report.token_peak_loss})
            if len(records) % 10 == 0:
                print(f"  {len(records)}/{len(rows)} recordings", flush=True)

    def mean(key: str) -> float:
        return st.fmean(r[key] for r in records)

    summary = {k: round(mean(k), 4) for k in ("position_mae_bins", "top3_recall", "energy_pearson", "peak_pearson",
                                              "position_loss", "energy_loss", "peak_loss")}
    criteria = {"position_mae_within_1_bin": summary["position_mae_bins"] <= POSITION_MAE_MAX_BINS,
                "top3_recall_at_least_070": summary["top3_recall"] >= TOP3_RECALL_MIN}
    passed = all(criteria.values())
    report = {"checkpoint": str(ref.path), "optimizer_step": ref.optimizer_step, "rows": len(records),
              "seed": args.seed, "summary": summary, "criteria": criteria,
              "thresholds": {"position_mae_bins": POSITION_MAE_MAX_BINS, "top3_recall": TOP3_RECALL_MIN},
              "verdict": "PASS" if passed else "FAIL", "seconds": round(time.time() - t0, 1), "records": records}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"TOKEN_SUPERVISION_EVAL step {ref.optimizer_step}: {summary} | criteria {criteria} | "
          f"VERDICT {report['verdict']} ({report['seconds']}s)", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
