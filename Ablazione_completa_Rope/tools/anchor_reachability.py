#!/usr/bin/env python3
"""Quante finestre gold contengono l'ancora di un token abbastanza FINE da localizzarle?

Fable, 18/09: il 93 % di ``salience_coverage.py`` e' **sovrapposizione** -- un token il cui supporto tocca la
finestra gold -- e un supporto largo che tocca una finestra di 137 campioni non dice dov'e' l'anomalia.  Se la
strategia realistica del LM e' "scegli il token saliente e ricopia la sua ancora", il tetto di quella strategia
non e' la sovrapposizione: e' la frazione di registrazioni in cui esiste un token la cui ancora **cade dentro**
la finestra e il cui supporto e' abbastanza stretto da puntare, non solo da coprire.

Questo strumento misura quel tetto in funzione della larghezza massima del supporto, e non serve nessun
checkpoint: guarda solo il frontend percettivo, che e' deterministico e identico a pesi non addestrati (il probe
lineare sui soft token dava AUC 0,997 senza addestramento).

Per ogni soglia W riporta la frazione di registrazioni con almeno un token che soddisfa:
  ancora dentro [start, end)  E  supporto <= W campioni
piu', come riferimento, la sola sovrapposizione del supporto -- cioe' la quantita' gia' misurata al 93 %.

Usage:
  python tools/anchor_reachability.py --config <toml> --out <json> [--split train] [--rows 909]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import AnomalyXLSplit  # noqa: E402
from wavelet_rope.signal_contract import ViewKind, compute_recording_normalization, make_view  # noqa: E402
from wavelet_rope.tokenizer import FAMILY_NAMES, PerceptualFrontend  # noqa: E402

WIDTHS = (8, 16, 32, 64, 128, 256, 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--rows", type=int, default=909)
    args = parser.parse_args()
    started = time.time()

    cfg = load_runtime_config(args.config, validate_paths=False)
    raw, spec = cfg.as_dict(), dict(cfg.spec)
    device = torch.device("cuda:0")
    frontend = PerceptualFrontend(spec).to(device)
    frontend.eval()

    root = Path(raw["dataset"]["root"])
    relative = raw["dataset"]["train_relative_path" if args.split == "train" else "validation_relative_path"]
    count = int(raw["dataset"]["train_rows" if args.split == "train" else "validation_rows"])
    split = AnomalyXLSplit(root / relative, args.split, count)

    hits = {w: 0 for w in WIDTHS}
    hits_salient = {w: 0 for w in WIDTHS}
    overlap = 0
    used = 0
    widths_of_hit = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for meta in split.metas():
            gold = list(meta.gold_intervals or ())
            if len(gold) != 1 or meta.category != "localize":
                continue
            interval = gold[0]
            episode = split.episode(meta.row_index)
            episode = type(episode)(episode.values.to(device), episode.valid.to(device), episode.channel_ids,
                                    episode.sample_times.to(device), episode.task)
            stats = compute_recording_normalization(episode)
            view = make_view(episode, stats, kind=ViewKind.GLOBAL)
            tokens = frontend.encode_view(view, recording_length=episode.length)
            valid = tokens.valid
            if not bool(valid.any()):
                continue
            start = tokens.support_start[valid].to(torch.float64)
            end = tokens.support_end[valid].to(torch.float64)
            family = tokens.family[valid]
            anchor = tokens.anchor_time[valid].to(torch.float64)
            support = (end - start).clamp_min(1)

            inside = (anchor >= float(interval.start)) & (anchor < float(interval.end))
            touches = (start < float(interval.end)) & (end > float(interval.start))
            salient = family == FAMILY_NAMES.index("D_salient")
            if bool(touches.any()):
                overlap += 1
            for w in WIDTHS:
                narrow = support <= float(w)
                if bool((inside & narrow).any()):
                    hits[w] += 1
                if bool((inside & narrow & salient).any()):
                    hits_salient[w] += 1
            picked = support[inside]
            if picked.numel():
                widths_of_hit.append(float(picked.min()))
            used += 1
            if used % 100 == 0:
                print(f"  {used} registrazioni", flush=True)
            if used >= args.rows:
                break

    if used == 0:
        raise SystemExit("nessuna riga localize positiva trovata")
    report = {"config": args.config, "split": args.split, "rows": used,
              "overlap_any_support": overlap / used,
              "anchor_inside_by_max_support": {str(w): hits[w] / used for w in WIDTHS},
              "anchor_inside_salient_only": {str(w): hits_salient[w] / used for w in WIDTHS},
              "median_narrowest_support_of_a_hit": (
                  float(torch.tensor(widths_of_hit).median()) if widths_of_hit else None),
              "seconds": round(time.time() - started, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{used} registrazioni localize positive ({args.split})", flush=True)
    print(f"  sovrapposizione con qualunque supporto: {100 * report['overlap_any_support']:.1f}%  "
          f"(e' la quantita' gia' nota al 93%)", flush=True)
    for w in WIDTHS:
        print(f"  ancora dentro, supporto <= {w:5d}: {100 * hits[w] / used:5.1f}%   "
              f"(solo D_salient: {100 * hits_salient[w] / used:5.1f}%)", flush=True)
    print(f"ANCHOR_REACHABILITY: tetto della strategia 'copia l'ancora' a supporto <= 8 campioni = "
          f"{100 * hits[8] / used:.1f}% ({report['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
