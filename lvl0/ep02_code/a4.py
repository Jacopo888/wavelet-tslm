import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
ch1 = data[:,0]
n = len(ch1)

W = 32
S = 8
starts = np.arange(0, n-W+1, S)
rmean = np.array([ch1[s:s+W].mean() for s in starts])

med = np.median(rmean)
mad = np.median(np.abs(rmean-med)) + 1e-12
z = np.abs(rmean-med)/(1.4826*mad)

thr = 4.0
mask = z > thr
anom_starts = starts[mask]
print("median rmean", med, "mad", mad)
print("num anomalous windows:", mask.sum(), "out of", len(mask))
if mask.sum()>0:
    print("first anomalous window start:", anom_starts[0], "last:", anom_starts[-1])

# find contiguous runs of anomalous windows
idx_true = np.where(mask)[0]
if len(idx_true)>0:
    breaks = np.where(np.diff(idx_true)>1)[0]
    run_boundaries = np.split(idx_true, breaks+1)
    print("num runs:", len(run_boundaries))
    for r in run_boundaries:
        s0 = starts[r[0]]
        e0 = starts[r[-1]] + W - 1
        print(f"run: start={s0} end={e0} len={e0-s0+1} n_windows={len(r)}")
