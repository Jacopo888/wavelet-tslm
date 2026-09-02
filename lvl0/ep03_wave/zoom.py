import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    ap = argparse.ArgumentParser(description="Zoom nella piramide multirisoluzione.")
    ap.add_argument("--channel", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--mode", required=True, choices=["detail", "approx", "energy"])
    ap.add_argument("--level", type=int, required=True, choices=range(1, 8))
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    with open(os.path.join(HERE, "calls.log"), "a") as f:
        f.write(" ".join(sys.argv[1:]) + "\n")

    with open(os.path.join(HERE, "pyramid.json")) as f:
        pyr = json.load(f)
    ch = pyr["channels"][args.channel - 1]
    block = 2 ** args.level
    s, e = max(0, args.start), min(pyr["n"] - 1, args.end)

    if args.mode == "detail":
        d = np.array(ch["details"][str(args.level)])
        centers = np.arange(len(d)) * block + block // 2
        mask = (centers >= s) & (centers <= e)
        dv, dc = d[mask], centers[mask]
        sigma = 1.4826 * np.median(np.abs(np.array(ch["details"][str(args.level)])))
        nbig = int(np.sum(np.abs(dv) > 3 * sigma))
        order = np.argsort(-np.abs(dv))[: args.top]
        order = order[np.argsort(dc[order])]
        print(f"level={args.level} range=[{s},{e}] coeffs_nel_range={len(dv)} "
              f"sigma_rumore_livello={sigma:.3f} sopra_3sigma={nbig}")
        print(f"top {min(args.top, len(dv))} per |valore| (t = centro del supporto, ampiezza {block}):")
        for i in order:
            print(f"t={int(dc[i])} v={dv[i]:+.3f}")
    elif args.mode == "approx":
        a = np.array(ch["approx"][str(args.level)])
        starts = np.arange(len(a)) * block
        mask = (starts + block - 1 >= s) & (starts <= e)
        av, ast = a[mask], starts[mask]
        if len(av) > 200:
            print(f"ERRORE: {len(av)} blocchi nel range al livello {args.level} (max 200). "
                  f"Usa un livello piu' alto o restringi il range.")
            return
        print(f"level={args.level} range=[{s},{e}] media_a_blocchi (blocco={block} campioni):")
        for st, v in zip(ast, av):
            print(f"t={int(st)} m={v:+.3f}")
    else:
        d = np.array(ch["details"][str(args.level)])
        centers = np.arange(len(d)) * block + block // 2
        mask = (centers >= s) & (centers <= e)
        dv, dc = d[mask], centers[mask]
        if len(dv) < 16:
            print("ERRORE: meno di 16 coefficienti nel range a questo livello; usa un livello piu' fine.")
            return
        e2 = dv * dv
        buckets = np.array_split(np.arange(len(dv)), 16)
        tot = e2.sum()
        print(f"level={args.level} range=[{s},{e}] energia % in 16 finestre (uniforme ~ 6%):")
        for b in buckets:
            t0, t1 = int(dc[b[0]]), int(dc[b[-1]])
            print(f"[{t0},{t1}] {100.0 * e2[b].sum() / tot:.0f}%")

if __name__ == "__main__":
    main()
