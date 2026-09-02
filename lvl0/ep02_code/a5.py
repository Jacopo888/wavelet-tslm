import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
ch1 = data[:,0]
n = len(ch1)

W = 64
S = 16
starts = np.arange(0, n-W+1, S)
rstd = np.array([ch1[s:s+W].std() for s in starts])

med = np.median(rstd)
mad = np.median(np.abs(rstd-med)) + 1e-12
z = (rstd-med)/(1.4826*mad)

print("median rstd", med, "mad", mad, "max z", z.max())

for thr in [3.0, 2.0, 1.5]:
    mask = z > thr
    idx_true = np.where(mask)[0]
    if len(idx_true)==0:
        print(f"thr={thr}: none")
        continue
    breaks = np.where(np.diff(idx_true)>1)[0]
    run_boundaries = np.split(idx_true, breaks+1)
    print(f"thr={thr}: n_runs={len(run_boundaries)}")
    for r in run_boundaries:
        s0 = starts[r[0]]
        e0 = starts[r[-1]] + W - 1
        print(f"  run: start={s0} end={e0} len={e0-s0+1}")
