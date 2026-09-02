import json, os

BASE = os.path.dirname(os.path.abspath(__file__))

def iou(a1, a2, b1, b2):
    inter = max(0, min(a2, b2) - max(a1, b1) + 1)
    union = (a2 - a1 + 1) + (b2 - b1 + 1) - inter
    return inter / union if union > 0 else 0.0

def main():
    with open(os.path.join(BASE, "gt", "gt.json")) as f:
        gt = json.load(f)
    with open(os.path.join(BASE, "results.json")) as f:
        res = json.load(f)

    rows = []
    for cond in ("code", "wave"):
        for ep in range(1, 11):
            key = f"{ep}_{cond}"
            g = gt[str(ep)]
            r = res.get(key)
            if r is None:
                rows.append((ep, cond, g["type"], None, None, None, None))
                continue
            i = iou(g["start"], g["end"], int(r["start"]), int(r["end"]))
            gset, pset = set(g["channels"]), set(r["channels"])
            chan_ok = gset == pset
            chan_j = len(gset & pset) / len(gset | pset)
            i_eff = i if chan_j > 0 else 0.0
            rows.append((ep, cond, g["type"], i_eff, chan_ok, chan_j, r.get("n_commands")))

    print(f"{'ep':>2} {'cond':<5} {'tipo':<8} {'IoU':>5} {'ch_ok':>5} {'ch_J':>4} {'cmd':>4}")
    for ep, cond, typ, i, ok, cj, nc in rows:
        if i is None:
            print(f"{ep:>2} {cond:<5} {typ:<8} MANCANTE")
        else:
            print(f"{ep:>2} {cond:<5} {typ:<8} {i:5.2f} {str(ok):>5} {cj:4.2f} {nc:>4}")

    for cond in ("code", "wave"):
        sel = [r for r in rows if r[1] == cond and r[3] is not None]
        n = len(sel)
        miou = sum(r[3] for r in sel) / n
        acc = sum(1 for r in sel if r[4]) / n
        mcmd = sum(r[6] for r in sel if r[6] is not None) / n
        hit = sum(1 for r in sel if r[3] >= 0.5) / n
        print(f"\n{cond}: n={n} IoU_media={miou:.3f} IoU>=0.5={hit:.0%} "
              f"canali_esatti={acc:.0%} comandi_medi={mcmd:.1f}")

if __name__ == "__main__":
    main()
