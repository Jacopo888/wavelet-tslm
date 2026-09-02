import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1, dtype=np.float64)
n = data.shape[0]

for c in range(3):
    col = data[:, c]
    d = np.diff(col)
    gstd = d.std()
    nblocks = 128
    bs = len(d) // nblocks
    scores = np.zeros(nblocks)
    for b in range(nblocks):
        seg = d[b*bs:(b+1)*bs]
        scores[b] = seg.std() / (gstd + 1e-12)
    order = np.argsort(scores)[::-1][:6]
    print(f"ch{c+1}: diff_gstd={gstd:.4f} median_ratio={np.median(scores):.3f}")
    print(f"  top blocks(diff bs={bs})={list(order)} ratios={[round(float(scores[i]),3) for i in order]}")
    print(f"  -> row ranges: {[(int(i*bs), int((i+1)*bs)) for i in order]}")
