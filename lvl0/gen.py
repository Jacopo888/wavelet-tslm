import json, os, random, shutil
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
N = 16384
LMAX = 7

def make_base(rng, n):
    t = np.arange(n)
    f1 = rng.uniform(0.0008, 0.002)
    f2 = rng.uniform(0.006, 0.012)
    a1 = rng.uniform(1.5, 2.5)
    a2 = rng.uniform(0.6, 1.2)
    p1 = rng.uniform(0, 6.28)
    p2 = rng.uniform(0, 6.28)
    sig = a1 * np.sin(2 * np.pi * f1 * t + p1) + a2 * np.sin(2 * np.pi * f2 * t + p2)
    e = np.empty(n)
    prev = 0.0
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    innov = nrng.normal(0, 0.3, n)
    for i in range(n):
        prev = 0.9 * prev + innov[i]
        e[i] = prev
    return sig, e

def inject(kind, x, noise, rng, s, e):
    if kind == "spike":
        k = rng.randrange(6, 11)
        pos = sorted(rng.sample(range(s, e + 1), k))
        for p in pos:
            amp = rng.uniform(4.0, 6.5) * rng.choice([-1, 1])
            w = rng.randrange(1, 4)
            x[p:p + w] += amp
        return pos[0], pos[-1]
    if kind == "shift":
        amp = rng.uniform(1.6, 2.4)
        x[s:e + 1] += amp
        return s, e
    if kind == "freq":
        amp = rng.uniform(1.5, 2.5)
        tt = np.arange(s, e + 1)
        x[s:e + 1] += amp * np.sin(2 * np.pi * 0.12 * tt)
        return s, e
    if kind == "varup":
        nrng = np.random.default_rng(rng.randrange(1 << 30))
        x[s:e + 1] += nrng.normal(0, 2.0, e - s + 1)
        return s, e
    if kind == "vardown":
        x[s:e + 1] -= noise[s:e + 1] * 0.9
        return s, e
    raise ValueError(kind)

def haar(x):
    details = {}
    approx_means = {}
    a = x.astype(float)
    for l in range(1, LMAX + 1):
        ev = a[0::2]
        od = a[1::2]
        details[l] = (ev - od) / np.sqrt(2)
        a = (ev + od) / np.sqrt(2)
        block = 2 ** l
        approx_means[l] = a / (np.sqrt(2) ** l)
        _ = block
    return details, approx_means

EPISODES = [
    (1, "spike",   [2], 300),
    (2, "shift",   [1], 800),
    (3, "freq",    [3], 500),
    (4, "varup",   [1], 600),
    (5, "spike",   [1], 200),
    (6, "shift",   [3], 1200),
    (7, "freq",    [2], 400),
    (8, "vardown", [2], 700),
    (9, "spike",   [1, 3], 300),
    (10, "shift",  [1, 2], 900),
]

def overview_text(chans_data):
    lines = []
    lines.append("VISTA COARSE (sempre nel contesto)")
    lines.append(f"Serie: {N} campioni, 3 canali.")
    lines.append("")
    lines.append("A7 = media a blocchi di 128 campioni (128 valori per canale).")
    lines.append("Il blocco k copre t in [k*128, (k+1)*128).")
    for ci, (details, am, x) in enumerate(chans_data):
        vals = " ".join(f"{v:.2f}" for v in am[7])
        lines.append(f"ch{ci+1} std={np.std(x):.2f}: {vals}")
    lines.append("")
    lines.append("Mappa di energia dei dettagli: per canale e livello L (1=fine ... 7=coarse),")
    lines.append("percentuale dell'energia dei coefficienti in 16 finestre temporali uguali")
    lines.append(f"(ogni finestra copre {N // 16} campioni; uniforme ~ 6% ovunque).")
    for ci, (details, am, x) in enumerate(chans_data):
        for l in range(1, LMAX + 1):
            d = details[l]
            e2 = d * d
            buckets = np.array_split(e2, 16)
            tot = e2.sum()
            pct = [100.0 * b.sum() / tot for b in buckets]
            lines.append(f"ch{ci+1} L{l}: " + " ".join(f"{p:.0f}" for p in pct))
    return "\n".join(lines)

def main():
    rng = random.Random(20260825)
    gt = {}
    for ep, kind, chans, length in EPISODES:
        erng = random.Random(1000 + ep)
        s = erng.randrange(2000, 14000 - length)
        e = s + length - 1
        chans_data = []
        raw = []
        gs, ge = s, e
        for ci in range(3):
            sig, noise = make_base(erng, N)
            x = sig + noise
            if (ci + 1) in chans:
                rs, re = inject(kind, x, noise, erng, s, e)
                gs, ge = rs, re
            raw.append(x)
        for ci in range(3):
            details, am = haar(raw[ci])
            chans_data.append((details, am, raw[ci]))
        gt[str(ep)] = {"type": kind, "channels": chans, "start": int(gs), "end": int(ge)}

        cdir = os.path.join(BASE, f"ep{ep:02d}_code")
        wdir = os.path.join(BASE, f"ep{ep:02d}_wave")
        os.makedirs(cdir, exist_ok=True)
        os.makedirs(wdir, exist_ok=True)

        with open(os.path.join(cdir, "raw.csv"), "w") as f:
            f.write("ch1,ch2,ch3\n")
            for i in range(N):
                f.write(f"{raw[0][i]:.4f},{raw[1][i]:.4f},{raw[2][i]:.4f}\n")

        pyr = {"n": N, "levels": LMAX, "channels": []}
        for ci in range(3):
            details, am, _ = chans_data[ci]
            pyr["channels"].append({
                "details": {str(l): [round(float(v), 4) for v in details[l]] for l in range(1, LMAX + 1)},
                "approx": {str(l): [round(float(v), 4) for v in am[l]] for l in range(1, LMAX + 1)},
            })
        with open(os.path.join(wdir, "pyramid.json"), "w") as f:
            json.dump(pyr, f)
        with open(os.path.join(wdir, "overview.txt"), "w", encoding="utf-8") as f:
            f.write(overview_text(chans_data))
        shutil.copy(os.path.join(BASE, "zoom.py"), os.path.join(wdir, "zoom.py"))

    os.makedirs(os.path.join(BASE, "gt"), exist_ok=True)
    with open(os.path.join(BASE, "gt", "gt.json"), "w") as f:
        json.dump(gt, f, indent=1)
    print("done")
    for k, v in gt.items():
        print(k, v)

if __name__ == "__main__":
    main()
