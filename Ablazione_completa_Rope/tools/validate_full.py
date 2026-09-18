#!/usr/bin/env python3
"""Validate one checkpoint on the whole validation split, in the trainer's own dump format.

Astra (16/09, punto 3): the run must be closed with the full 520-row validation and with the checkpoint selection
the plan declared, not with the last epoch taken for granted.  The trainer only produces that dump at a pass end,
so this reproduces it for an arbitrary checkpoint: same generation path, same scorer, same record layout, so that
``tools/compare_with_va1.py`` reads it unchanged.

Selecting across checkpoints: run it on each candidate and compare ``summary.primary``; the plan declares selection
on the 520-row validation, so the last epoch wins only if it actually scores best.

Usage:
  python tools/validate_full.py --config configs/va1r_mi300x.toml --checkpoint <dir> --out <json> [--rows 520]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safetensors.torch import load_file  # noqa: E402

from wavelet_rope.backbone import load_backbone  # noqa: E402
from wavelet_rope.checkpoint import inspect_foreign_checkpoint  # noqa: E402
from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import AnomalyXLSplit  # noqa: E402
from wavelet_rope.model import WRAgentModel  # noqa: E402
from wavelet_rope.scoring import REQUIRED_KEYS, parse_structured_pred, score_precise  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=0, help="0 = the whole split")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-reasoning-tokens", type=int, default=384)
    args = parser.parse_args()
    started = time.time()

    cfg = load_runtime_config(args.config, validate_paths=False)
    raw, spec = cfg.as_dict(), dict(cfg.spec)
    device = torch.device("cuda:0")
    runtime = raw["runtime"]
    backbone = load_backbone(raw["backbone"]["snapshot"]["root"], backend=runtime["attention_backend"],
                             pad_to=int(runtime["attention_qk_pad_to"]), device=device,
                             gradient_checkpointing=False)
    training = spec["training"]
    model = WRAgentModel(backbone, spec, segment_weights=dict(training["segment_loss_weights"]), aux_weight=0.0)
    model.frontend.to(device)
    model.mixer.to(device)
    model.projector.to(device)
    ref = inspect_foreign_checkpoint(args.checkpoint)
    model.load_trainable_state_dict(load_file(str(ref.path / "model.safetensors")))
    backbone.ctx.lam = float(ref.trainer_state["wildcard_lambda"])
    model.eval()
    backbone.text.eval()
    thinking = bool(model.reading_segment)

    root = Path(raw["dataset"]["root"])
    split = AnomalyXLSplit(root / raw["dataset"]["validation_relative_path"], "validation",
                           int(raw["dataset"]["validation_rows"]))
    rows = list(range(len(split)))[: args.rows] if args.rows else list(range(len(split)))
    print(f"validating {len(rows)} rows of the validation split, thinking={thinking}", flush=True)

    per_category: dict[str, list[float]] = defaultdict(list)
    records, invalid = [], 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for number, row in enumerate(rows, start=1):
            meta = split.meta(row)
            episode = split.episode(row)
            episode = type(episode)(episode.values.to(device), episode.valid.to(device), episode.channel_ids,
                                    episode.sample_times.to(device), episode.task)
            generated = model.generate(meta, episode, mode="direct", max_new_tokens=args.max_new_tokens,
                                       max_reasoning_tokens=args.max_reasoning_tokens, thinking=thinking)
            prediction = generated.get("text") or ""
            scored = score_precise(prediction, meta.answer)
            if parse_structured_pred(prediction, REQUIRED_KEYS[meta.category]) is None:
                invalid += 1
            per_category[meta.category].append(float(scored["primary"]))
            records.append({"id": meta.id, "category": meta.category, "primary": scored["primary"],
                            "pred": prediction[:300], "gold": meta.answer,
                            "prompt_tokens": generated.get("prompt_length"),
                            "reading": (generated.get("reading") or "")[:300],
                            "reasoning_tokens": generated.get("reasoning_tokens"),
                            "forced_closure": generated.get("forced_closure")})
            if number % 50 == 0:
                done = sum(sum(v) for v in per_category.values()) / max(1, number)
                print(f"  {number}/{len(rows)} rows, running primary {done:.4f}", flush=True)

    summary = {k: round(sum(v) / len(v), 4) for k, v in per_category.items()}
    overall = round(sum(sum(v) for v in per_category.values()) / max(1, len(records)), 4)
    result = {"tag": f"full_{ref.optimizer_step}", "rows": len(records), "primary": overall,
              "by_category": summary, "invalid_format": invalid,
              "invalid_format_rate": round(invalid / max(1, len(records)), 4),
              "seconds": round(time.time() - started, 1), "optimizer_step": ref.optimizer_step,
              "checkpoint": str(ref.path)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": result, "records": records}, indent=2), encoding="utf-8")
    print(f"VALIDATE_FULL step {ref.optimizer_step}: primary {overall} {summary} "
          f"invalid {invalid}/{len(records)} ({result['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
