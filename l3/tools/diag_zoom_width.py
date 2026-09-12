"""Is it NARROWNESS or CENTERING that makes the zoom work?

Evaluates vz1 on val520 with the window CENTERED ON GOLD but widened to a target width
(default 6144, the width the policy actually picks). Compare with:
  oracle_text (gold +/-25%, ~1024 wide)  -> 0.5244
  policy windows (~6150-10100 wide, gold contained 70% of the time) -> 0.2869
Run from ROOT with PYTHONPATH=ROOT. Usage: diag_width.py <target_width> <out_json> <out_jsonl>
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from l3.zoom import (ZoomAction, _load_eval_runtime, clamp_zoom_window, gold_events,
                     heuristic_zoom, run_zoom_episode, series_length, channel_names,
                     zoom_iou_with_gold)

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 6144
OUT_JSON = Path(sys.argv[2] if len(sys.argv) > 2 else "/root/l3/zoom_20260911c/Z1/diag/val520_vz1_oracle_w.json")
OUT_JSONL = Path(sys.argv[3] if len(sys.argv) > 3 else "/root/l3/zoom_20260911c/Z1/diag/val520_vz1_oracle_w.jsonl")
CFG = "l3/configs/l3_zoom_z1.yaml"
CKPT = "/root/l3/zoom_20260911c/Z1/train/vz1_epoch_1_end.pt"

model, dataset, _ = _load_eval_runtime(CFG, CKPT, "val")
model.eval()
names_cache = {}


def wide_gold_action(sample):
    events = gold_events(sample)
    if not events:
        return heuristic_zoom(sample)
    length = series_length(sample)
    names = channel_names(sample)
    event = min(events, key=lambda e: (-(e["end"] - e["start"]),
                                       names.index(str(e["channel"])), e["start"]))
    mid = (int(event["start"]) + int(event["end"])) // 2
    window = clamp_zoom_window(mid - TARGET // 2, mid + (TARGET - TARGET // 2), length)
    if window is None:
        return heuristic_zoom(sample)
    return ZoomAction(str(event["channel"]), *window, length)


sums, counts, widths, ious = {}, {}, [], []
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
with OUT_JSONL.open("w", encoding="utf-8") as handle:
    for index in range(len(dataset)):
        sample = dataset[index]
        action = wide_gold_action(sample)
        episode = run_zoom_episode(model, sample, sample_generation=False,
                                   max_action_tokens=500, max_answer_tokens=500,
                                   forced_action=action)
        result = dataset.evaluate_answer(episode.final_text, sample)
        category = str(sample["category"])
        primary = float(result[dataset.primary_metric])
        counts[category] = counts.get(category, 0) + 1
        sums[category] = sums.get(category, 0.0) + primary
        widths.append(action.window_len)
        ious.append(zoom_iou_with_gold(sample, action))
        handle.write(json.dumps({
            "index": index, "id": str(sample.get("id", "")), "category": category,
            "ground_truth": dataset.get_ground_truth(sample), "prediction": episode.final_text,
            "primary": primary, "zoom": {"channel": action.channel, "start": action.start,
                                          "end": action.end, "window_len": action.window_len},
        }, sort_keys=True) + "\n")
        handle.flush()
        if (index + 1) % 100 == 0:
            print(f"DIAG_W step {index+1}/{len(dataset)} running={sum(sums.values())/(index+1):.4f}",
                  flush=True)

total = sum(sums.values()) / max(1, sum(counts.values()))
summary = {
    "status": "DONE", "mode": f"oracle_width_{TARGET}", "split": "val",
    "samples_evaluated": sum(counts.values()), "primary": total,
    "per_category": {c: {"count": counts[c], "primary": sums[c] / counts[c]} for c in sorted(counts)},
    "zoom_window_len": sum(widths) / len(widths),
    "zoom_iou_with_gold": sum(ious) / len(ious),
    "checkpoint": CKPT,
}
OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("DIAG_W_DONE", json.dumps({k: summary[k] for k in ("primary", "samples_evaluated",
                                                          "zoom_window_len")}))
print(json.dumps(summary["per_category"], indent=1))
