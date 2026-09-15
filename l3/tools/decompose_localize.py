#!/usr/bin/env python3
"""Scompone il punteggio di localize fra rilevare l'assenza e puntare l'anomalia.

Il `primary` di localize e' `iou_or_correct_empty`: sulle righe senza anomalia vale 1
se il modello risponde «assente», sulle righe con anomalia vale l'IoU. Le due cose
misurano capacita' diverse e vanno riportate separate: il totale aggregato puo'
salire per merito della sola rilevazione dell'assenza.

Include i riferimenti che una scorciatoia statistica potrebbe raggiungere: rispondere
sempre «assente», e il miglior intervallo costante in coordinate normalizzate
(ottimizzato sui dati stessi, quindi un limite superiore ottimistico).

Solo CPU. Uso: decompose_localize.py righe1.jsonl [righe2.jsonl ...]
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


def load(path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("category")) != "localize":
            continue
        out.append((row, json.loads(str(row["ground_truth"]).partition(":")[2])))
    if not out:
        raise ValueError(f"nessuna riga localize in {path}")
    return out


def iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def best_constant_interval(present: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
                           steps: int = 200) -> tuple[float, float, float]:
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


def report(path: Path) -> dict[str, Any]:
    rows = load(path)
    present = [(r, g) for r, g in rows if g.get("present")]
    absent = [(r, g) for r, g in rows if not g.get("present")]
    n = len(rows)
    mean = lambda items, key="primary": statistics.fmean(float(r[key]) for r, _ in items)
    return {
        "file": path.name, "rows": n, "present": len(present), "absent": len(absent),
        "primary_total": mean(rows),
        "absent_contribution": mean(absent) * len(absent) / n if absent else 0.0,
        "present_contribution": mean(present) * len(present) / n if present else 0.0,
        "absent_hit_rate": mean(absent) if absent else None,
        "iou_on_present": mean(present) if present else None,
        "acc_at_iou_0p3": mean(present, "acc_at_iou_0p3") if present else None,
        "acc_at_iou_0p5": mean(present, "acc_at_iou_0p5") if present else None,
    }


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    paths = [Path(p) for p in sys.argv[1:]]
    reports = [report(p) for p in paths]
    head = "{:34} {:>7} {:>9} {:>9} {:>8} {:>8} {:>8}".format(
        "file", "totale", "assenti", "presenti", "IoU pres", "acc0.3", "acc0.5")
    print(head)
    for d in reports:
        print("{:34} {:7.4f} {:9.4f} {:9.4f} {:8.4f} {:8.3f} {:8.3f}".format(
            d["file"][:34], d["primary_total"], d["absent_contribution"],
            d["present_contribution"], d["iou_on_present"], d["acc_at_iou_0p3"],
            d["acc_at_iou_0p5"]))
    rows = load(paths[0])
    present = [(r, g) for r, g in rows if g.get("present")]
    absent = [(r, g) for r, g in rows if not g.get("present")]
    score, centre, width = best_constant_interval(present)
    print("\nriferimenti raggiungibili senza leggere il segnale:")
    print("  sempre 'assente':            {:.4f} sul totale".format(len(absent) / len(rows)))
    print("  miglior intervallo costante: IoU {:.4f} sulle presenti "
          "(centro {:.3f}, ampiezza {:.3f})".format(score, centre, width))
    print("  presenza perfetta + intervallo costante: {:.4f} sul totale".format(
        (score * len(present) + len(absent)) / len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
