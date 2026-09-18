#!/usr/bin/env python3
"""I soft token di QUESTO braccio dicono quale di loro sta sull'anomalia?

Jacopo, 17/09: il probe di P1-e mostrava la posizione decodificabile a 1,174 bin su registrazioni nuove, e la
domanda naturale e' se questo renda superflua la scala di generalizzazione.  Non lo rende, per due motivi gia'
annotati in RISULTATI-VA1R: quel probe girava su **un altro checkpoint**, addestrato CON la supervisione per token
(``position_weight`` attivo), e il suo bersaglio era la posizione del token sulla griglia di lettura, non quella
dell'anomalia.  Nella famiglia va1r/va1rz ``token_supervision.enabled`` e' false e ``position_weight`` e' 0: niente
ha mai spinto questi token a codificare alcunche'.  Se l'informazione c'e' comunque, e' perche' la CE finale l'ha
prodotta come effetto collaterale -- ed e' esattamente quello che nessuno ha mai misurato qui.

Probe POST-HOC: nessun addestramento del modello, solo forward.  Per ogni registrazione si prendono i soft token
della vista globale, cioe' **gli stessi vettori che finiscono nella sequenza del LM**, e si chiede a una mappa
lineare, per ogni token separatamente, se quel token sta dentro la finestra annotata.  L'etichetta e' l'ancora
temporale del token, che l'``EncodedView`` porta gia' in ``tau``.

Due cose lo rendono informativo dove il probe sugli stati nascosti non lo era:

  il posto    questi sono i token in ingresso, non lo stato nascosto dell'ultima posizione del prompt -- che,
              come ha osservato Astra, precede il campo delle coordinate e quindi era il posto sbagliato;
  la forma    non si chiede al probe di fare aritmetica.  Se un token sa dire "l'anomalia e' qui", la posizione
              segue dall'ancora, che il modello ha gia' in tau.  E' la via piu' facile possibile: se e' chiusa,
              sono chiuse anche quelle piu' difficili.

Riportati su registrazioni MAI adattate:
  auc         quanto il punteggio lineare separa i token sull'anomalia dagli altri
  errore      distanza fra l'ancora del token col punteggio piu' alto e il centro vero, in frazione di L
  contro      lo stesso errore predicendo il centro della registrazione, e scegliendo un token a caso
  null        le stesse quantita' con le etichette mescolate nel training del probe

Usage:
  python tools/token_position_probe.py --config <toml> --checkpoint <dir> --out <json> [--rows 300]
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


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC, by ranks: P(score of a positive > score of a negative)."""

    positives = labels > 0.5
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # media dei ranghi per i pareggi, altrimenti un probe degenere sembra migliore del caso
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu, sigma = x.mean(0, keepdims=True), x.std(0, keepdims=True) + 1e-6
    a = (x - mu) / sigma
    centre = y.mean()
    gram = a.T @ a + alpha * np.eye(a.shape[1])
    weights = np.linalg.solve(gram, a.T @ (y - centre))
    return weights, mu, sigma, float(centre)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="senza, si sonda il modello NON addestrato: e' il controllo "
                                            "che distingue 'il training l'ha prodotta' da 'il frontend "
                                            "wavelet ce l'ha per costruzione'")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=300)
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
    print(f"checkpoint {name}, passo {step}, lambda {backbone.ctx.lam}", flush=True)

    root = Path(raw["dataset"]["root"])
    split = AnomalyXLSplit(root / raw["dataset"]["validation_relative_path"], "validation",
                           int(raw["dataset"]["validation_rows"]))

    per_row = []          # (feature, etichette, ancore, centro, larghezza, durata) -- tutto in unita' tau
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for meta in split.metas():
            gold = [g for g in (meta.gold_intervals or ())]
            if len(gold) != 1:
                continue                      # una sola finestra: niente ambiguita' su quale sia il bersaglio
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
                continue                      # il canale annotato non e' in questa vista
            keep = (view.channel_slot == slot).nonzero().squeeze(1)
            if keep.numel() < 8:
                continue
            anchors = view.tau[keep].float().cpu().numpy()
            # tau NON e' l'indice di campione: vale (ancora - origine) / (TIME_UNIT_SAMPLES * dt), cioe' 1/64 del
            # campione con dt=1.  Confrontarlo con start/end in campioni dava zero positivi su 350 registrazioni.
            # Si convertono i bordi con la stessa funzione del modello, invece di indovinare il fattore.
            dt = episode.dt if episode.dt is not None else 1.0
            origin = float(episode.sample_times[0])
            edges = tau_from_anchor(torch.tensor([float(interval.start), float(interval.end)],
                                                 dtype=torch.float64), origin, dt).numpy()
            # tolleranza di mezza spaziatura: un token conta come positivo se il suo bin COPRE la finestra,
            # altrimenti una finestra piu' stretta della griglia non contiene nessuna ancora per costruzione
            unique = np.unique(anchors)
            spacing = float(np.median(np.diff(unique))) if unique.size > 1 else 0.0
            labels = ((anchors >= edges[0] - spacing / 2.0)
                      & (anchors < edges[1] + spacing / 2.0)).astype(np.float64)
            if labels.sum() == 0 or labels.sum() == len(labels):
                continue                      # senza entrambe le classi non c'e' niente da separare
            # il centro vero e l'errore si misurano in tau, la stessa scala delle ancore
            per_row.append((view.soft[keep].float().cpu().numpy(), labels, anchors,
                            float(edges[0] + edges[1]) / 2.0, float(edges[1] - edges[0]),
                            float(tau_from_anchor(torch.tensor([float(meta.length)], dtype=torch.float64),
                                                  origin, dt).item())))
            if len(per_row) >= args.rows:
                break
            if len(per_row) % 50 == 0:
                print(f"  {len(per_row)} registrazioni", flush=True)

    if len(per_row) < 60:
        raise SystemExit(f"troppe poche registrazioni utilizzabili: {len(per_row)}")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(per_row))
    cut = int(0.7 * len(per_row))
    train_rows = [per_row[i] for i in order[:cut]]
    val_rows = [per_row[i] for i in order[cut:]]
    positives = sum(int(r[1].sum()) for r in per_row)
    tokens = sum(len(r[1]) for r in per_row)
    print(f"{len(per_row)} registrazioni, {tokens} token, {positives} sull'anomalia "
          f"({100.0 * positives / tokens:.2f}%); {len(train_rows)} per l'adattamento, {len(val_rows)} tenute fuori",
          flush=True)

    x_train = np.concatenate([r[0] for r in train_rows])
    y_train = np.concatenate([r[1] for r in train_rows])

    def evaluate(weights, mu, sigma, centre) -> dict:
        scores_all, labels_all, errors, random_errors = [], [], [], []
        for features, labels, anchors, gold_centre, _width, length in val_rows:
            s = ((features - mu) / sigma) @ weights + centre
            scores_all.append(s)
            labels_all.append(labels)
            errors.append(abs(float(anchors[int(np.argmax(s))]) - gold_centre) / length)
            random_errors.append(abs(float(anchors[int(rng.integers(len(anchors)))]) - gold_centre) / length)
        return {"auc": auc(np.concatenate(scores_all), np.concatenate(labels_all)),
                "median_error_of_L": float(np.median(errors)),
                "median_error_random_token": float(np.median(random_errors)),
                "hit_rate_top_token": float(np.mean([e < 0.01 for e in errors]))}

    real = evaluate(*ridge(x_train, y_train, args.alpha))
    shuffled = y_train.copy()
    rng.shuffle(shuffled)
    null = evaluate(*ridge(x_train, shuffled, args.alpha))
    # predire sempre il centro della registrazione: il riferimento che non guarda nulla
    centre_error = float(np.median([abs(0.5 * r[5] - r[3]) / r[5] for r in val_rows]))

    report = {"checkpoint": args.checkpoint or "untrained", "optimizer_step": step, "rows": len(per_row),
              "val_rows": len(val_rows), "tokens": tokens, "positive_tokens": positives, "alpha": args.alpha,
              "real": real, "shuffled_null": null, "median_error_recording_centre": centre_error,
              "seconds": round(time.time() - started, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  auc {real['auc']:.4f} (null {null['auc']:.4f})", flush=True)
    print(f"  errore mediano del token migliore {real['median_error_of_L']:.4f} di L "
          f"| token a caso {real['median_error_random_token']:.4f} | centro fisso {centre_error:.4f}", flush=True)
    print(f"  finestra centrata entro l'1% di L: {100 * real['hit_rate_top_token']:.1f}% delle registrazioni",
          flush=True)
    verdict = ("DECODIFICABILE" if real["auc"] > 0.65 and real["median_error_of_L"] < 0.5 * centre_error
               else "NON DECODIFICABILE")
    print(f"TOKEN_POSITION_PROBE {name} passo {step}: {verdict} ({report['seconds']}s)", flush=True)
    return 0 if verdict == "DECODIFICABILE" else 1


if __name__ == "__main__":
    sys.exit(main())
