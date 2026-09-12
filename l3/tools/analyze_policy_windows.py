"""Geometria delle finestre scelte dalla policy (Codex punto 1, parte CPU).

L'IoU fra finestra (~2.000 campioni) e anomalia (decine di campioni) e' bassa per
costruzione: non misura la qualita' del puntamento. Le metriche corrette sono il
*contenimento* dell'evento nella finestra (sul canale giusto) e l'errore di centro
normalizzato sull'ampiezza.

Solo CPU: legge il JSONL di una valutazione che registra il campo `zoom`.
Uso: analyze_policy_windows.py rows.jsonl [rows.jsonl ...]
"""
import json
import math
import statistics
import sys
from pathlib import Path


def parse_gold(raw):
    """Ritorna la lista di eventi gold [(channel|None, start, end)] della riga."""
    head, _, payload = raw.partition(":")
    data = json.loads(payload)
    if head == "localize":
        if data.get("present") and data.get("start") is not None:
            return [(None, int(data["start"]), int(data["end"]))]
        return []
    if head == "classify_with_evidence":
        if data.get("start") is not None and data.get("kind") not in (None, "None"):
            return [(None, int(data["start"]), int(data["end"]))]
        return []
    if head == "localize_all_channels":
        out = []
        for a in data.get("anomalies") or []:
            if a.get("start") is None:
                continue
            out.append((str(a.get("channel")), int(a["start"]), int(a["end"])))
        return out
    return []


def describe(name, values, fmt="{:.4f}"):
    if not values:
        return f"  {name:34} n=0"
    m = statistics.fmean(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    med = statistics.median(values)
    return (f"  {name:34} n={len(values):4} media=" + fmt.format(m)
            + " sd=" + fmt.format(sd) + " mediana=" + fmt.format(med))


def analyze(path):
    rows = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
    acted = [r for r in rows if r.get("zoom")]
    print(f"\n=== {path}")
    print(f"righe={len(rows)} con azione={len(acted)}")

    buckets = {}
    for r in acted:
        z = r["zoom"]
        ws, we = int(z["start"]), int(z["end"])
        width = we - ws
        gold = parse_gold(r["ground_truth"])
        cat = r["category"]
        b = buckets.setdefault(cat, {
            "rows": 0, "no_gold": 0, "contained": 0, "overlap": 0, "miss": 0,
            "pos": [], "absdev": [], "width": [], "primary_contained": [],
            "primary_miss": [], "chan_hit": 0, "chan_rows": 0, "iou": [],
        })
        b["rows"] += 1
        b["width"].append(width)
        b["iou"].append(float(r.get("iou") or 0.0))
        if not gold:
            b["no_gold"] += 1
            continue
        # canale: rilevante solo dove il gold lo porta (multicanale)
        chan = str(z.get("channel"))
        if any(g[0] is not None for g in gold):
            b["chan_rows"] += 1
            same = [g for g in gold if g[0] == chan]
            if same:
                b["chan_hit"] += 1
            gold_here = same
        else:
            gold_here = gold
        if not gold_here:
            b["miss"] += 1
            b["primary_miss"].append(float(r["primary"]))
            continue
        best = None
        for _, gs, ge in gold_here:
            inside = gs >= ws and ge <= we
            ov = max(0, min(ge, we) - max(gs, ws))
            score = (2 if inside else (1 if ov > 0 else 0), ov)
            if best is None or score > best[0]:
                best = (score, gs, ge, inside, ov)
        _, gs, ge, inside, ov = best
        if inside:
            b["contained"] += 1
            b["primary_contained"].append(float(r["primary"]))
        elif ov > 0:
            b["overlap"] += 1
            b["primary_miss"].append(float(r["primary"]))
        else:
            b["miss"] += 1
            b["primary_miss"].append(float(r["primary"]))
        centre = (gs + ge) / 2.0
        pos = (centre - ws) / width
        b["pos"].append(pos)
        b["absdev"].append(abs(pos - 0.5))

    tot = {k: 0 for k in ("rows", "no_gold", "contained", "overlap", "miss", "chan_hit", "chan_rows")}
    agg = {k: [] for k in ("pos", "absdev", "width", "primary_contained", "primary_miss", "iou")}
    print(f"\n{'categoria':24} {'n':>4} {'gold':>5} {'dentro':>7} {'parz':>6} {'fuori':>6} "
          f"{'canale':>8} {'ampiezza':>9} {'IoU':>7}")
    for cat in sorted(buckets):
        b = buckets[cat]
        for k in tot:
            tot[k] += b[k]
        for k in agg:
            agg[k].extend(b[k])
        with_gold = b["rows"] - b["no_gold"]
        chan = (f"{b['chan_hit']}/{b['chan_rows']}" if b["chan_rows"] else "-")
        pct = (lambda x: f"{x}" if with_gold == 0 else f"{x} ({100.0*x/with_gold:.0f}%)")
        print(f"{cat:24} {b['rows']:4} {with_gold:5} {pct(b['contained']):>7} "
              f"{b['overlap']:6} {b['miss']:6} {chan:>8} "
              f"{statistics.fmean(b['width']):9.0f} {statistics.fmean(b['iou']):7.4f}")

    with_gold = tot["rows"] - tot["no_gold"]
    print(f"\nTOTALE con gold={with_gold}: contenuto {tot['contained']} "
          f"({100.0*tot['contained']/max(1,with_gold):.1f} %), parziale {tot['overlap']}, "
          f"fuori {tot['miss']}")
    if tot["chan_rows"]:
        print(f"canale corretto (solo multicanale): {tot['chan_hit']}/{tot['chan_rows']} "
              f"({100.0*tot['chan_hit']/tot['chan_rows']:.1f} %)")
    print(describe("posizione interna del gold", agg["pos"]))
    print(describe("|posizione - 0,5|", agg["absdev"]))
    print(describe("ampiezza finestra", agg["width"], "{:.0f}"))
    print(describe("IoU finestra-gold", agg["iou"]))
    print(describe("primary | evento contenuto", agg["primary_contained"]))
    print(describe("primary | evento non contenuto", agg["primary_miss"]))
    return buckets


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        analyze(p)
