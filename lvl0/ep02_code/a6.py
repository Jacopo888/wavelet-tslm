import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
ch1 = data[:,0]
n = len(ch1)

# moving average baseline via cumsum, window 800, centered
W = 800
csum = np.cumsum(np.insert(ch1, 0, 0))
def mavg(i):
    lo = max(0, i-W//2)
    hi = min(n, i+W//2)
    return (csum[hi]-csum[lo])/(hi-lo)

# compute residual at stride 8 for speed
stride = 8
idxs = np.arange(0, n, stride)
resid = np.array([ch1[i]-mavg(i) for i in idxs])
absres = np.abs(resid)

med = np.median(absres)
mad = np.median(np.abs(absres-med)) + 1e-12
z = (absres-med)/(1.4826*mad)
print("median absres", med, "mad", mad, "max z", z.max(), "at idx", idxs[np.argmax(z)])

for thr in [3.0]:
    mask = z > thr
    idx_true = np.where(mask)[0]
    breaks = np.where(np.diff(idx_true)>1)[0] if len(idx_true)>0 else []
    run_boundaries = np.split(idx_true, breaks+1) if len(idx_true)>0 else []
    print(f"thr={thr}: n_runs={len(run_boundaries)}")
    for r in run_boundaries:
        s0 = idxs[r[0]]
        e0 = idxs[r[-1]]
        print(f"  run: start={s0} end={e0} len={e0-s0+1} n_pts={len(r)}")
