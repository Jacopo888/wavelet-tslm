"""Gate a pesi fissi (Codex, 12/09 punto 7.1): il guadagno dell'oracle viene dal CONTENUTO
della finestra o dalle sue COORDINATE?

Stesso checkpoint vz1, stesso formato a due turni, stesso header, stessa azione, stesso numero
di token percettivi; cambia SOLO il tensore della finestra iniettato nell'encoder.

Condizioni:
  true       finestra gold stretta, contenuto vero              (riferimento)
  swap       header IDENTICO a `true`, contenuto preso da un'altra finestra dello stesso
             canale, stessa lunghezza, priva di eventi gold     (test del contenuto)
  zero       header IDENTICO a `true`, contenuto azzerato       (controllo fuori distribuzione)
  offcenter  header VERO di una finestra che contiene il gold in posizione interna casuale
             (10-90%), stessa lunghezza                         (il modello legge la posizione?)

Sottoinsieme stratificato fissato PRIMA dell'esperimento (seed 20260912), niente prime-80.
Uso: diag_crop_content.py <out_dir> [n_per_stratum]
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from l3.zoom import (ZoomAction, _generate, _load_eval_runtime, build_turn2_sample,
                     channel_names, clamp_zoom_window, gold_events, series_length)
from l3.grpo import _autocast_for

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/l3/zoom_20260911c/Z1/gate")
N_PER = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SEED = 20260912
CFG = "l3/configs/l3_zoom_z1.yaml"
CKPT = "/root/l3/zoom_20260911c/Z1/train/vz1_epoch_1_end.pt"
MARGIN = 0.25

model, dataset, _ = _load_eval_runtime(CFG, CKPT, "val")
model.eval()
OUT.mkdir(parents=True, exist_ok=True)


def payload(text):
    text = text or ""
    i = text.find("{")
    if i < 0:
        return {}
    depth, instr, esc = 0, False, False
    for j, ch in enumerate(text[i:], i):
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
        elif ch == '"':
            instr = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i:j + 1])
                except Exception:
                    return {}
    return {}


def stratify():
    """Fixed stratified subset: category x (positive/negative) x length band."""
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        gold = gold_events(sample)
        category = str(sample["category"])
        length = int(sample.get("length", 0))
        band = "short" if length <= 32768 else "long"
        rows.append((index, category, bool(gold), band))
    rng = random.Random(SEED)
    strata = {}
    for index, category, positive, band in rows:
        if category == "lead_lag_with_magnitude":
            continue  # escluso dal ramo zoom (decisione operativa, vedi review §6)
        strata.setdefault((category, positive, band), []).append(index)
    chosen = []
    for key in sorted(strata):
        pool = sorted(strata[key])
        rng.shuffle(pool)
        take = N_PER if key[1] else max(6, N_PER // 3)
        chosen.extend(pool[:min(take, len(pool))])
    return sorted(set(chosen))


def oracle_narrow(sample):
    events = gold_events(sample)
    if not events:
        return None
    names = channel_names(sample)
    event = min(events, key=lambda e: (-(e["end"] - e["start"]),
                                       names.index(str(e["channel"])), e["start"]))
    width = max(1, int(event["end"]) - int(event["start"]))
    extra = int(np.ceil(width * MARGIN))
    window = clamp_zoom_window(int(event["start"]) - extra, int(event["end"]) + extra,
                              series_length(sample))
    if window is None:
        return None
    return ZoomAction(str(event["channel"]), *window, series_length(sample)), event


def offcenter_action(sample, event, width, rng):
    """Truthful header: window of the same width containing the gold at 10-90% inside."""
    length = series_length(sample)
    gold_lo, gold_hi = int(event["start"]), int(event["end"])
    gold_w = max(1, gold_hi - gold_lo)
    frac = rng.uniform(0.10, 0.90)
    start = int(round(gold_lo - frac * (width - gold_w)))
    window = clamp_zoom_window(start, start + width, length)
    if window is None:
        return None
    return ZoomAction(str(event["channel"]), *window, length)


def swap_values(sample, action, rng):
    """Values from another window of the same channel, same length, gold-free."""
    names = channel_names(sample)
    channel = np.asarray(sample["time_series"][names.index(action.channel)], dtype=np.float32)
    length, width = channel.size, action.window_len
    bad = [(int(e["start"]), int(e["end"])) for e in gold_events(sample)
           if str(e["channel"]) == action.channel]
    candidates = []
    for start in range(0, max(1, length - width), max(1, width // 2)):
        end = start + width
        if end > length:
            break
        if all(min(end, b_hi) - max(start, b_lo) <= 0 for b_lo, b_hi in bad):
            centre = (start + end) / 2
            candidates.append((abs(centre - (action.start + action.end) / 2), start))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    start = candidates[0][1]
    return channel[start:start + width].copy(), start


def run(sample, action, values):
    """Identical text/tokens; only the injected zoom features differ."""
    turn2 = build_turn2_sample(sample, action)
    prepared = model.prepare_batch([turn2], training=False)
    prepared.pop("labels", None)
    original = model.prepare_batch([sample], training=False)
    series = original["time_series"]
    tensor = torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=series.dtype,
                             device=series.device).unsqueeze(0)
    with torch.no_grad(), _autocast_for(model):
        features = torch.cat((model._encode_and_project(series),
                              model._encode_and_project(tensor)), dim=0).detach()
    prepared["_l3_ts_features"] = features
    ids = _generate(model, prepared, sample=False, max_new_tokens=500)
    return model.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def pred_interval(text):
    p = payload(text)
    if isinstance(p.get("start"), int) and isinstance(p.get("end"), int) and p["start"] < p["end"]:
        return int(p["start"]), int(p["end"])
    return None


indices = stratify()
print(f"GATE subset n={len(indices)} seed={SEED}", flush=True)
conditions = ("true", "swap", "zero", "offcenter")
acc = {c: {} for c in conditions}
rng = random.Random(SEED)
with (OUT / "rows.jsonl").open("w", encoding="utf-8") as handle:
    for count, index in enumerate(indices, 1):
        sample = dataset[index]
        category = str(sample["category"])
        built = oracle_narrow(sample)
        gold = gold_events(sample)
        if built is None:
            continue
        action, event = built
        variants = {"true": (action, None), "zero": (action, np.zeros(action.window_len, np.float32))}
        swapped = swap_values(sample, action, rng)
        if swapped is not None:
            variants["swap"] = (action, swapped[0])
        off = offcenter_action(sample, event, action.window_len, rng)
        if off is not None:
            variants["offcenter"] = (off, None)
        for name, (act, values) in variants.items():
            if values is None:
                index_channel = channel_names(sample).index(act.channel)
                channel = np.asarray(sample["time_series"][index_channel], dtype=np.float32)
                values = channel[act.start:act.end].copy()
            text = run(sample, act, values)
            result = dataset.evaluate_answer(text, sample)
            primary = float(result[dataset.primary_metric])
            bucket = acc[name].setdefault(category, {"n": 0, "primary": 0.0, "iou": 0.0,
                                                     "iou_n": 0, "empty": 0})
            bucket["n"] += 1
            bucket["primary"] += primary
            bucket["empty"] += int(not text)
            pi = pred_interval(text)
            if pi and gold:
                bucket["iou"] += max(iou(pi, (int(g["start"]), int(g["end"])))
                                     for g in gold if str(g["channel"]) == act.channel) \
                    if any(str(g["channel"]) == act.channel for g in gold) else 0.0
                bucket["iou_n"] += 1
            handle.write(json.dumps({
                "index": index, "id": str(sample.get("id", "")), "category": category,
                "condition": name, "zoom": {"channel": act.channel, "start": act.start,
                                            "end": act.end, "width": act.window_len},
                "prediction": text, "primary": primary,
                "ground_truth": dataset.get_ground_truth(sample),
            }, sort_keys=True) + "\n")
            handle.flush()
        if count % 20 == 0:
            print(f"GATE {count}/{len(indices)} " + " ".join(
                f"{c}={sum(v['primary'] for v in acc[c].values())/max(1,sum(v['n'] for v in acc[c].values())):.3f}"
                for c in conditions), flush=True)

summary = {"subset_size": len(indices), "seed": SEED, "checkpoint": CKPT,
           "margin": MARGIN, "conditions": {}}
for name in conditions:
    per_category = {}
    for category, v in sorted(acc[name].items()):
        per_category[category] = {
            "n": v["n"], "primary": v["primary"] / max(1, v["n"]),
            "iou_mean": v["iou"] / max(1, v["iou_n"]), "iou_n": v["iou_n"],
            "empty": v["empty"],
        }
    total_n = sum(v["n"] for v in acc[name].values())
    summary["conditions"][name] = {
        "n": total_n,
        "primary": sum(v["primary"] for v in acc[name].values()) / max(1, total_n),
        "per_category": per_category,
    }
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
print("GATE_DONE")
for name in conditions:
    c = summary["conditions"][name]
    print(name, "n", c["n"], "primary", round(c["primary"], 4),
          {k: (round(v["primary"], 3), round(v["iou_mean"], 3)) for k, v in c["per_category"].items()})
