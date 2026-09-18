#!/usr/bin/env python3
"""Dove si perde, fra l'ingresso di Qwen e la posizione che scrive la cifra?

Sappiamo due cose che non stanno insieme.  Sui soft token -- gli stessi vettori scritti nei segnaposto di
``inputs_embeds`` -- una mappa lineare trova la finestra con AUC 0,99 e un errore dello 0,18% di L su
registrazioni mai adattate, e lo fa anche a pesi NON addestrati: l'informazione e' li' per costruzione.  Eppure
azzerare o sostituire quegli stessi token non cambia una virgola della risposta.

Fra i due fatti ci sono 36 layer, e nessuno li ha mai guardati.  Il probe precedente sugli stati nascosti aveva
sondato l'ultima posizione del PROMPT, ma -- obiezione di Astra, corretta -- quella posizione precede il campo
delle coordinate: il modello scrive ``localize:{"_L": 16384, "end": ...`` e le cifre arrivano molti token dopo,
quindi poteva calcolare la posizione nel momento in cui la scrive.  Un nullo li' non stabiliva niente.

Due misure, entrambe solo forward, sulle stesse registrazioni:

  A  profondita'   lo stesso probe per token (questo token sta sull'anomalia?) sugli stati nascosti a ogni
                   profondita', non su ``soft``.  Se l'AUC regge fino in fondo, l'informazione sopravvive allo
                   stack; se crolla a un layer preciso, li' si perde.
  B  scrittura     con la risposta vera in teacher forcing, lo stato nascosto della posizione che EMETTE la
                   prima cifra di ``"start"``.  Ridge sul centro normalizzato, split disgiunto, null mescolato.

La combinazione discrimina due diagnosi che finora non erano separate:
  A alta e B nulla  ->  l'informazione resta nei token di segnale ma non viene instradata alle posizioni di
                        risposta: guasto di attenzione, coerente col 2-4% di massa misurato;
  A che crolla      ->  lo stack la distrugge prima, e il punto di rottura ha un indirizzo.

Usage:
  python tools/depth_probe.py --config <toml> --checkpoint <dir> --out <json> [--rows 120] [--layers 0,8,16,24,31,final]
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
from wavelet_rope.perceptual_rope import tau_from_anchor  # noqa: E402
from wavelet_rope.templates import SEG_FINAL  # noqa: E402


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = labels > 0.5
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit(x: np.ndarray, y: np.ndarray, alpha: float):
    mu, sigma = x.mean(0, keepdims=True), x.std(0, keepdims=True) + 1e-6
    a = (x - mu) / sigma
    centre = float(y.mean())
    weights = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]), a.T @ (y - centre))
    return weights, mu, sigma, centre


def apply(model, x: np.ndarray) -> np.ndarray:
    weights, mu, sigma, centre = model
    return ((x - mu) / sigma) @ weights + centre


def pearson_with_ci(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    if pred.std() < 1e-12 or truth.std() < 1e-12:
        return 0.0, float("nan"), float("nan")
    r = float(np.corrcoef(pred, truth)[0, 1])
    n = len(truth)
    if n <= 4 or abs(r) >= 0.999:
        return r, float("nan"), float("nan")
    half = 1.96 / np.sqrt(n - 3)
    z = np.arctanh(r)
    return r, float(np.tanh(z - half)), float(np.tanh(z + half))


def digit_position(model, seq, field: str) -> int | None:
    """Indice della posizione il cui stato nascosto produce la PRIMA cifra di ``"start"``.

    Lo stato in posizione i predice il token i+1, quindi si cerca l'ultimo token del prefisso che termina con
    ``"start":`` e si restituisce il suo indice.  Il campo si chiama sempre cosi' nelle risposte localize e
    classify; se non compare, la registrazione viene saltata invece di sondare un posto a caso.
    """

    final = (seq.segment == SEG_FINAL).nonzero().squeeze(1).tolist()
    if not final:
        return None
    ids = seq.input_ids.tolist()
    for cut in range(1, len(final) + 1):
        text = model.bb.tokenizer.decode([ids[i] for i in final[:cut]], skip_special_tokens=False)
        if field in text:
            return final[cut - 1]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--layers", default="0,8,16,24,31,final")
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    started = time.time()

    cfg = load_runtime_config(args.config, validate_paths=False)
    raw, spec = cfg.as_dict(), dict(cfg.spec)
    device = torch.device("cuda:0")
    runtime = raw["runtime"]
    backbone = load_backbone(raw["backbone"]["snapshot"]["root"], backend=runtime["attention_backend"],
                             pad_to=int(runtime["attention_qk_pad_to"]), device=device,
                             gradient_checkpointing=False)
    model = WRAgentModel(backbone, spec, segment_weights=dict(spec["training"]["segment_loss_weights"]),
                         aux_weight=0.0)
    model.frontend.to(device)
    model.mixer.to(device)
    model.projector.to(device)
    if args.checkpoint:
        ref = inspect_foreign_checkpoint(args.checkpoint)
        model.load_trainable_state_dict(load_file(str(ref.path / "model.safetensors")))
        backbone.ctx.lam = float(ref.trainer_state["wildcard_lambda"])
        name, step = ref.path.name, ref.optimizer_step
    else:
        backbone.ctx.lam = 1.0
        name, step = "NON ADDESTRATO", 0
    model.eval()
    backbone.text.eval()
    names = [n.strip() for n in args.layers.split(",") if n.strip()]
    print(f"checkpoint {name}, passo {step}, profondita' {names}", flush=True)

    root = Path(raw["dataset"]["root"])
    split = AnomalyXLSplit(root / raw["dataset"]["validation_relative_path"], "validation",
                           int(raw["dataset"]["validation_rows"]))

    token_feats: dict[str, list] = {n: [] for n in names}   # per riga: [n_token, dim]
    token_labels: list = []
    write_feats: dict[str, dict] = {n: {} for n in names}   # profondita' -> campo -> per riga [dim]
    centres: list = []
    used = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for meta in split.metas():
            gold = list(meta.gold_intervals or ())
            if len(gold) != 1:
                continue
            interval = gold[0]
            episode = split.episode(meta.row_index)
            episode = type(episode)(episode.values.to(device), episode.valid.to(device), episode.channel_ids,
                                    episode.sample_times.to(device), episode.task)
            encoding = model.encode_episode(episode, [], gold=meta.gold_intervals)
            view = encoding.views[0]
            if view.channel_slot is None or view.soft.shape[0] == 0:
                continue
            try:
                slot = view.channel_ids.index(interval.channel)
            except ValueError:
                continue
            # teacher forcing: la risposta vera e' nella sequenza, quindi la posizione della cifra e' nota
            seq = model.build_sequence(meta, encoding, mode="direct", actions=[], answer=meta.answer,
                                       supervise_action=False, generation=False, thinking=False)
            # Le chiavi del JSON sono in ordine alfabetico: _L, end, present, start.  Quindi "end" e' la PRIMA
            # quantita' posizionale, e la posizione che la emette non ha nulla da cui copiare; quella di "start"
            # ha gia' davanti il valore di "end" scritto in chiaro, e un probe li' puo' limitarsi a rileggerlo.
            # Si misurano entrambe proprio per separare le due cose: "end" e' la misura, "start" e' il controllo.
            where = {f: digit_position(model, seq, f) for f in ('"end":', '"start":')}
            if any(v is None for v in where.values()):
                continue
            embeds, is_signal, tau = model._inputs(seq, encoding)
            backbone.ctx.set_tokens(is_signal, tau)
            out = backbone.text(inputs_embeds=embeds, use_cache=False, output_hidden_states=True)
            states = out.hidden_states

            # posizioni dei segnaposto di QUESTO canale, nell'ordine della sequenza
            positions = None
            for view_index, block_slot, pos in seq.view_slots:
                if view_index == 0 and block_slot == slot:
                    positions = pos.to(device)
            if positions is None or positions.numel() < 8:
                continue
            anchors = tau[0, positions].float().cpu().numpy()
            dt = episode.dt if episode.dt is not None else 1.0
            origin = float(episode.sample_times[0])
            edges = tau_from_anchor(torch.tensor([float(interval.start), float(interval.end)],
                                                 dtype=torch.float64), origin, dt).numpy()
            unique = np.unique(anchors)
            spacing = float(np.median(np.diff(unique))) if unique.size > 1 else 0.0
            labels = ((anchors >= edges[0] - spacing / 2.0) & (anchors < edges[1] + spacing / 2.0)).astype(np.float64)
            if labels.sum() == 0 or labels.sum() == len(labels):
                continue
            span = float(tau_from_anchor(torch.tensor([float(meta.length)], dtype=torch.float64),
                                         origin, dt).item())
            token_labels.append(labels)
            centres.append(float(edges[0] + edges[1]) / 2.0 / span)
            for n in names:
                index = len(states) - 1 if n == "final" else min(int(n), len(states) - 1)
                token_feats[n].append(states[index][0, positions].float().cpu().numpy())
                for field, pos in where.items():
                    write_feats[n].setdefault(field, []).append(states[index][0, pos].float().cpu().numpy())
            used += 1
            if used % 25 == 0:
                print(f"  {used} registrazioni", flush=True)
            if used >= args.rows:
                break

    if used < 60:
        raise SystemExit(f"troppe poche registrazioni utilizzabili: {used}")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(used)
    cut = int(0.7 * used)
    train_idx, val_idx = order[:cut], order[cut:]
    y = np.asarray(centres, dtype=np.float64)
    shuffled = y.copy()
    rng.shuffle(shuffled)
    print(f"{used} registrazioni, {cut} per l'adattamento, {used - cut} tenute fuori", flush=True)

    results = {}
    for n in names:
        # A: per token, a questa profondita'
        x_train = np.concatenate([token_feats[n][i] for i in train_idx])
        y_train = np.concatenate([token_labels[i] for i in train_idx])
        probe = fit(x_train, y_train, args.alpha)
        scores = np.concatenate([apply(probe, token_feats[n][i]) for i in val_idx])
        truth = np.concatenate([token_labels[i] for i in val_idx])
        shuffled_tokens = y_train.copy()
        rng.shuffle(shuffled_tokens)
        null_probe = fit(x_train, shuffled_tokens, args.alpha)
        null_scores = np.concatenate([apply(null_probe, token_feats[n][i]) for i in val_idx])

        # B: le due posizioni di scrittura.  "end" e' la misura (niente da copiare prima), "start" e' il
        # controllo (ha gia' davanti il valore di "end"): se solo "start" e' alto, si stava leggendo il testo.
        write = {}
        for field, rows_of in write_feats[n].items():
            w_train = np.stack([rows_of[i] for i in train_idx])
            w_val = np.stack([rows_of[i] for i in val_idx])
            real = apply(fit(w_train, y[train_idx], args.alpha), w_val)
            null = apply(fit(w_train, shuffled[train_idx], args.alpha), w_val)
            r, lo, hi = pearson_with_ci(real, y[val_idx])
            r_null, _, _ = pearson_with_ci(null, shuffled[val_idx])
            write[field] = {"pearson": r, "ci95": [lo, hi], "pearson_null": r_null,
                            "median_abs_error_of_L": float(np.median(np.abs(real - y[val_idx])))}

        results[n] = {"token_auc": auc(scores, truth), "token_auc_null": auc(null_scores, truth),
                      "write": write}
        end, start = write.get('"end":', {}), write.get('"start":', {})
        print(f"  layer {n:6s} | token AUC {results[n]['token_auc']:.4f} "
              f"(null {results[n]['token_auc_null']:.4f}) | end {end.get('pearson', float('nan')):+.3f} "
              f"CI95 [{end.get('ci95', [float('nan')] * 2)[0]:+.3f}; {end.get('ci95', [float('nan')] * 2)[1]:+.3f}] "
              f"| start {start.get('pearson', float('nan')):+.3f} (controllo)", flush=True)

    report = {"checkpoint": args.checkpoint or "untrained", "optimizer_step": step, "rows": used,
              "val_rows": used - cut, "layers": results, "alpha": args.alpha,
              "seconds": round(time.time() - started, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    best_token = max(results.items(), key=lambda kv: kv[1]["token_auc"])
    key_end, key_start = '"end":', '"start":'
    def best_of(field):
        return max(results.items(), key=lambda kv: abs(kv[1]["write"].get(field, {}).get("pearson", 0.0)))
    name_end, row_end = best_of(key_end)
    name_start, row_start = best_of(key_start)
    print("DEPTH_PROBE {}: token AUC massima {:.4f} al layer {}; end massimo {:+.3f} al layer {}; "
          "start massimo {:+.3f} al layer {} ({}s)".format(
              name, best_token[1]["token_auc"], best_token[0],
              row_end["write"][key_end]["pearson"], name_end,
              row_start["write"][key_start]["pearson"], name_start, report["seconds"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
