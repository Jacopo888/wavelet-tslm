import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
ch1 = data[:,0]
n = len(ch1)
B = 16
nb = n // B
x = ch1[:nb*B].reshape(nb, B)
bmean = x.mean(axis=1)

W = 32
nW = nb - W + 1
wstd = np.array([bmean[i:i+W].std() for i in range(nW)])
med = np.median(wstd)
mad = np.median(np.abs(wstd-med)) + 1e-12
z = (wstd-med)/(1.4826*mad)

for thr in [1.0, 1.5]:
    mask = z > thr
    idx_true = np.where(mask)[0]
    breaks = np.where(np.diff(idx_true)>1)[0] if len(idx_true)>0 else []
    runs = np.split(idx_true, breaks+1) if len(idx_true)>0 else []
    print(f"thr={thr} n_runs={len(runs)}")
    for r in runs:
        b0, b1 = r[0], r[-1]
        s0 = b0*B
        e0 = (b1+W-1)*B + B - 1
        if e0 - s0 > 100:  # only show substantial runs
            print(f"  run blocks[{b0},{b1}] samples[{s0},{e0}] len={e0-s0+1}")
