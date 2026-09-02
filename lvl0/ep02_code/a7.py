import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
ch1 = data[:,0]
n = len(ch1)
B = 16
nb = n // B
x = ch1[:nb*B].reshape(nb, B)
bmean = x.mean(axis=1)  # 1024 block means, block b covers [b*16,(b+1)*16-1]

W = 32  # window in blocks -> 512 samples, matches a3
# global robust baseline using ALL windows at stride 1 (fine), but exclude nothing (robust to ~15% contamination)
nW = nb - W + 1
wstd_all = np.array([bmean[i:i+W].std() for i in range(nW)])
med = np.median(wstd_all)
mad = np.median(np.abs(wstd_all-med)) + 1e-12
z = (wstd_all-med)/(1.4826*mad)
print("global median", med, "mad", mad, "max z", z.max(), "at block", np.argmax(z))

thr = 3.0
mask = z > thr
idx_true = np.where(mask)[0]
breaks = np.where(np.diff(idx_true)>1)[0] if len(idx_true)>0 else []
runs = np.split(idx_true, breaks+1) if len(idx_true)>0 else []
print("n_runs at thr", thr, ":", len(runs))
for r in runs:
    b0, b1 = r[0], r[-1]
    s0 = b0*B
    e0 = (b1+W-1)*B + B - 1
    print(f"  run window-blocks [{b0},{b1}] -> sample range approx [{s0},{e0}] len={e0-s0+1}")
