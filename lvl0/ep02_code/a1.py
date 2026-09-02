import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
n = data.shape[0]
print("n rows", n, "cols", data.shape[1])

B = 128  # block size
nb = n // B
for ch in range(3):
    x = data[:nb*B, ch].reshape(nb, B)
    bmean = x.mean(axis=1)
    bstd = x.std(axis=1)
    med_m = np.median(bmean)
    mad_m = np.median(np.abs(bmean - med_m)) + 1e-12
    med_s = np.median(bstd)
    mad_s = np.median(np.abs(bstd - med_s)) + 1e-12
    zm = np.abs(bmean - med_m) / (1.4826*mad_m)
    zs = np.abs(bstd - med_s) / (1.4826*mad_s)
    score = np.maximum(zm, zs)
    top = np.argsort(-score)[:8]
    top = np.sort(top)
    print(f"--- ch{ch+1} med_mean={med_m:.4f} med_std={med_s:.4f}")
    for b in top:
        print(f"block {b} range[{b*B},{b*B+B-1}] mean={bmean[b]:.4f} std={bstd[b]:.4f} zm={zm[b]:.2f} zs={zs[b]:.2f}")
