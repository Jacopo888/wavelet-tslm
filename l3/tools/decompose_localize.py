#!/usr/bin/env python3
"""Scompone il punteggio di localize (e del multicanale) fra rilevare l'assenza e puntare.

Il `primary` di localize e' `iou_or_correct_empty`: sulle righe senza anomalia vale 1
se il modello risponde «assente», sulle righe con anomalia vale l'IoU. Il `primary` di
localize_all_channels e' un F1 su coppie (canale, intervallo): sulle righe con lista
vuota vale 1 se il modello risponde con la lista vuota. In entrambi i casi due
capacita' diverse finiscono sommate in un numero, e vanno riportate separate: il
totale puo' salire per merito della sola rilevazione dell'assenza.

Include i riferimenti che una scorciatoia statistica potrebbe raggiungere: rispondere
sempre «assente» (o lista vuota) e, per localize, il miglior intervallo costante in
coordinate normalizzate (ottimizzato sui dati stessi, quindi un limite superiore
ottimistico).

Con `--paired` confronta ogni file successivo col primo, appaiato per id, con
bootstrap sulle sole righe con anomalia e su tutte le righe.

Solo CPU. Uso:
  decompose_localize.py [--category localize|localize_all_channels] [--paired] r1.jsonl [r2.jsonl ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
from typing import Any, Mapping, Sequence

Row = tuple[dict[str, Any], dict[str, Any]]


def is_present(category: str, gt: Mapping[str, Any]) -> bool:
    if category == "localize":
        return bool(gt.get("present"))
    return bool(gt.get("anomalies"))


def load(path: Path, category: str) -> list[Row]:
    out: list[Row] = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("category")) != category:
            continue
        out.append((row, json.loads(str(row["ground_truth"]).partition(":")[2])))
    if not out:
        raise ValueError(f"nessuna riga {category} in {path}")
    return out


def iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def best_constant_interval(present: Sequence[Row], steps: int = 200) -> tuple[float, float, float]:
    """Miglior intervallo fisso, cercato sugli stessi dati: limite superiore ottimistico."""
    best = (0.0, 0.0, 0.0)
    targets = [(g["start"] / g["_L"], g["end"] / g["_L"]) for _, g in present]
    for ci in range(1, steps):
        centre = ci / steps
        for wi in range(1, steps // 3):
            width = wi / steps
            score = statistics.fmean(
                iou(centre - width / 2, centre + width / 2, b0, b1) for b0, b1 in targets)
            if score > best[0]:
                best = (score, centre, width)
    return best


def report(path: Path, category: str) -> dict[str, Any]:
    rows = load(path, category)
    present = [(r, g) for r, g in rows if is_present(category, g)]
    absent = [(r, g) for r, g in rows if not is_present(category, g)]
    n = len(rows)
    mean = lambda items, key="primary": statistics.fmean(float(r[key]) for r, _ in items)
    out = {
        "file": path.name, "category": category, "rows": n,
        "present": len(present), "absent": len(absent),
        "primary_total": mean(rows),
        "absent_contribution": mean(absent) * len(absent) / n if absent else 0.0,
        "present_contribution": mean(present) * len(present) / n if present else 0.0,
        "absent_hit_rate": mean(absent) if absent else None,
        "score_on_present": mean(present) if present else None,
    }
    if category == "localize":
        out["acc_at_iou_0p3"] = mean(present, "acc_at_iou_0p3") if present else None
        out["acc_at_iou_0p5"] = mean(present, "acc_at_iou_0p5") if present else None
    else:
        out["tp_total"] = sum(float(r["tp"]) for r, _ in present)
        out["rows_with_tp"] = sum(1 for r, _ in present if float(r["tp"]) > 0)
    return out


def paired(base: Sequence[Row], other: Sequence[Row], category: str,
           seed: int = 0, draws: int = 10000) -> dict[str, Any]:
    """Differenza other - base, appaiata per id; fallisce se id o verita' differiscono."""
    b = {r["id"]: (r, g) for r, g in base}
    o = {r["id"]: (r, g) for r, g in other}
    if set(b) != set(o):
        raise ValueError("id diversi fra i due file")
    for k in b:
        if b[k][1] != o[k][1]:
            raise ValueError(f"verita' diversa per id {k}")
    rng = random.Random(seed)

    def boot(keys: list[str]) -> tuple[float, float, float]:
        d = [float(o[k][0]["primary"]) - float(b[k][0]["primary"]) for k in keys]
        bs = sorted(statistics.fmean(rng.choice(d) for _ in d) for _ in range(draws))
        return statistics.fmean(d), bs[int(0.025 * draws)], bs[int(0.975 * draws)]

    keys_present = [k for k in b if is_present(category, b[k][1])]
    return {"present_only": boot(keys_present), "all_rows": boot(sorted(b))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--category", default="localize", choices=["localize", "localize_all_channels"])
    ap.add_argument("--paired", action="store_true", help="confronta ogni file col primo, appaiato per id")
    args = ap.parse_args()
    cat = args.category
    reports = [report(p, cat) for p in args.files]
    if cat == "localize":
        print("{:34} {:>7} {:>9} {:>9} {:>8} {:>8} {:>8}".format(
            "file", "totale", "assenti", "presenti", "IoU pres", "acc0.3", "acc0.5"))
        for d in reports:
            print("{:34} {:7.4f} {:9.4f} {:9.4f} {:8.4f} {:8.3f} {:8.3f}".format(
                d["file"][:34], d["primary_total"], d["absent_contribution"],
                d["present_contribution"], d["score_on_present"], d["acc_at_iou_0p3"],
                d["acc_at_iou_0p5"]))
    else:
        print("{:34} {:>7} {:>9} {:>9} {:>9} {:>8} {:>5} {:>9}".format(
            "file", "totale", "vuote", "presenti", "hit vuote", "F1 pres", "TP", "righe TP"))
        for d in reports:
            print("{:34} {:7.4f} {:9.4f} {:9.4f} {:9.3f} {:8.4f} {:5.0f} {:9d}".format(
                d["file"][:34], d["primary_total"], d["absent_contribution"],
                d["present_contribution"], d["absent_hit_rate"], d["score_on_present"],
                d["tp_total"], d["rows_with_tp"]))
    rows = load(args.files[0], cat)
    present = [(r, g) for r, g in rows if is_present(cat, g)]
    absent = [(r, g) for r, g in rows if not is_present(cat, g)]
    print("\nriferimenti raggiungibili senza leggere il segnale ({} righe, {} con anomalia):".format(
        len(rows), len(present)))
    print("  sempre 'assente' / lista vuota:  {:.4f} sul totale".format(len(absent) / len(rows)))
    if cat == "localize":
        score, centre, width = best_constant_interval(present)
        print("  miglior intervallo costante:     IoU {:.4f} sulle presenti "
              "(centro {:.3f}, ampiezza {:.3f})".format(score, centre, width))
        print("  presenza perfetta + intervallo costante: {:.4f} sul totale".format(
            (score * len(present) + len(absent)) / len(rows)))
    if args.paired and len(args.files) > 1:
        print("\ndifferenze appaiate rispetto a {} (bootstrap 10k, seme 0):".format(args.files[0].name))
        for p in args.files[1:]:
            d = paired(rows, load(p, cat), cat)
            m, lo, hi = d["present_only"]
            m2, lo2, hi2 = d["all_rows"]
            print("  {:30} solo con anomalia {:+.4f} [{:+.4f}; {:+.4f}]   tutte {:+.4f} [{:+.4f}; {:+.4f}]".format(
                p.name[:30], m, lo, hi, m2, lo2, hi2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
