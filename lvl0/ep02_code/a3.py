import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
n = data.shape[0]
B = 16
nb = n // B

for ch in range(3):
    x = data[:nb*B, ch].reshape(nb, B)
    bmean = x.mean(axis=1)  # 1024 values
    W = 32  # window in blocks -> 512 samples
    nW = nb - W + 1
    # rolling std of bmean over window W, stride 8 blocks
    stride = 8
    idxs = np.arange(0, nW, stride)
    wstd = np.array([bmean[i:i+W].std() for i in idxs])
    med = np.median(wstd)
    mad = np.median(np.abs(wstd-med)) + 1e-12
    z = (wstd-med)/(1.4826*mad)
    top = np.argsort(-z)[:6]
    top = np.sort(top)
    print(f"--- ch{ch+1} median_wstd={med:.3f}")
    for t in top:
        b = idxs[t]
        s = b*B
        e = s + W*B - 1
        print(f"win start_idx={s} end_idx={e} wstd={wstd[t]:.3f} z={z[t]:.2f}")
