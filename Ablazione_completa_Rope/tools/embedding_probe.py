#!/usr/bin/env python3
"""Does the anomaly position survive THROUGH the backbone, to the place where the answer is written?

Astra (16/09) put this first in the order of suspects, and it is the one link never measured.  We know the position
is present in the soft tokens: the two linear heads of P1-e decoded it to 1.174 bins on unseen recordings.  We know
the model never writes it.  What is missing is whether the information is still there at the end of the prompt, in
the hidden state from which the first answer token is produced.

Two outcomes, two different diagnoses:
  decodable   the information arrives and is not used -- the problem is the objective or the output format;
  not         the information is lost inside the backbone -- the problem is the interface or the geometry.

Method: for each recording the prompt is built exactly as at inference (generation prompt, no answer), a forward is
run, and the hidden state of the LAST prompt position is taken at several depths.  A ridge regression maps it to the
normalised anomaly centre, fitted on a disjoint set of recordings and reported only on held-out ones.  A shuffled
target runs alongside as the null: a probe that also "succeeds" on shuffled labels is fitting noise.

Usage:
  python tools/embedding_probe.py --config <toml> --checkpoint <dir> --out <json> [--rows 120]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safetensors.torch import load_file  # noqa: E402

from wavelet_rope.backbone import load_backbone  # noqa: E402
from wavelet_rope.checkpoint import inspect_foreign_checkpoint  # noqa: E402
from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import AnomalyXLSplit  # noqa: E402
from wavelet_rope.model import WRAgentModel  # noqa: E402


def ridge_fit_eval(x_train, y_train, x_val, y_val, alpha: float) -> dict:
    """Ridge with an unpenalised intercept, scored only on held-out rows."""

    mu, sigma = x_train.mean(0, keepdims=True), x_train.std(0, keepdims=True) + 1e-6
    a = (x_train - mu) / sigma
    b = (x_val - mu) / sigma
    centre = y_train.mean()
    gram = a.T @ a + alpha * np.eye(a.shape[1])
    weights = np.linalg.solve(gram, a.T @ (y_train - centre))
    pred = b @ weights + centre
    resid = float(np.sum((y_val - pred) ** 2))
    total = float(np.sum((y_val - y_val.mean()) ** 2))
    if pred.std() > 1e-12 and y_val.std() > 1e-12:
        pearson = float(np.corrcoef(pred, y_val)[0, 1])
    else:
        pearson = 0.0
    # Fisher interval on the held-out correlation.  Without it a point estimate invites the mistake this tool
    # already caused once: +0.28 at one layer and -0.27 at another, on 31 rows where the standard error is 0.19.
    n = len(y_val)
    if n > 4 and abs(pearson) < 0.999:
        z = np.arctanh(pearson)
        half = 1.96 / np.sqrt(n - 3)
        lo, hi = float(np.tanh(z - half)), float(np.tanh(z + half))
    else:
        lo = hi = float("nan")
    return {"pearson": pearson, "pearson_ci95": [lo, hi], "n_val": int(n),
            "r2": 1.0 - resid / total if total > 0 else 0.0,
            "median_abs_error": float(np.median(np.abs(pred - y_val)))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--layers", default="0,8,16,24,31,final")
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=11)
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

    root = Path(raw["dataset"]["root"])
    split = AnomalyXLSplit(root / raw["dataset"]["validation_relative_path"], "validation",
                           int(raw["dataset"]["validation_rows"]))
    # Every category that carries a window is usable: localize, classify_with_evidence and the multichannel one.
    # With only the 102 positive localize rows the held-out set is 31 recordings, where the standard error of a
    # correlation is about 0.19 -- the first run produced +0.28 at one layer and -0.27 at another, which is what
    # noise looks like.  Pooling the categories roughly triples the sample.
    rows = []
    for meta in split.metas():
        try:
            gold = json.loads(meta.answer.split(":", 1)[1])
        except (ValueError, TypeError):
            continue
        start, end = gold.get("start"), gold.get("end")
        if start is None or end is None:
            anomalies = gold.get("anomalies") or []
            if not anomalies:
                continue
            start = min(float(a["start"]) for a in anomalies)
            end = max(float(a["end"]) for a in anomalies)
        if gold.get("present") is False:
            continue
        rows.append((meta, (float(start) + float(end)) / 2.0 / float(meta.length)))
        if len(rows) >= args.rows:
            break
    if len(rows) < 60:
        raise SystemExit("too few positive localize rows for a probe")
    print(f"{len(rows)} recordings with a window, probing layers {args.layers}", flush=True)

    names = [n.strip() for n in args.layers.split(",") if n.strip()]
    features: dict[str, list] = {n: [] for n in names}
    targets = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for number, (meta, centre) in enumerate(rows, start=1):
            episode = split.episode(meta.row_index)
            episode = type(episode)(episode.values.to(device), episode.valid.to(device), episode.channel_ids,
                                    episode.sample_times.to(device), episode.task)
            encoding = model.encode_episode(episode, [], gold=meta.gold_intervals)
            sequence = model.build_sequence(meta, encoding, mode="direct", actions=[], answer=None,
                                            supervise_action=False, generation=True, thinking=False)
            embeds, is_signal, tau = model._inputs(sequence, encoding)
            backbone.ctx.set_tokens(is_signal, tau)
            out = backbone.text(inputs_embeds=embeds, use_cache=False, output_hidden_states=True)
            states = out.hidden_states
            for name in names:
                index = len(states) - 1 if name == "final" else min(int(name), len(states) - 1)
                features[name].append(states[index][0, -1].float().cpu().numpy())
            targets.append(centre)
            if number % 25 == 0:
                print(f"  {number}/{len(rows)}", flush=True)

    y = np.asarray(targets, dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    train_idx, val_idx = order[:cut], order[cut:]
    shuffled = y.copy()
    rng.shuffle(shuffled)

    results = {}
    for name in names:
        x = np.stack(features[name])
        real = ridge_fit_eval(x[train_idx], y[train_idx], x[val_idx], y[val_idx], args.alpha)
        null = ridge_fit_eval(x[train_idx], shuffled[train_idx], x[val_idx], shuffled[val_idx], args.alpha)
        results[name] = {"real": real, "shuffled_null": null,
                         "median_abs_error_samples": real["median_abs_error"] * float(rows[0][0].length)}
        lo, hi = real["pearson_ci95"]
        print(f"  layer {name:6s} pearson {real['pearson']:+.3f} CI95 [{lo:+.3f}; {hi:+.3f}] "
              f"(null {null['pearson']:+.3f})  errore mediano {real['median_abs_error']:.4f} di L", flush=True)

    report = {"checkpoint": str(ref.path), "optimizer_step": ref.optimizer_step, "rows": len(rows),
              "train_rows": len(train_idx), "val_rows": len(val_idx), "alpha": args.alpha,
              "layers": results, "seconds": round(time.time() - started, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    best = max(results.items(), key=lambda kv: kv[1]["real"]["pearson"])
    print(f"EMBED_PROBE step {ref.optimizer_step}: miglior layer {best[0]} pearson "
          f"{best[1]['real']['pearson']:+.3f} contro null {best[1]['shuffled_null']['pearson']:+.3f} "
          f"({report['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
