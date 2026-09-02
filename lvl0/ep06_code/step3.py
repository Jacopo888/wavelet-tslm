import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1, dtype=np.float64)
n = data.shape[0]

for c in range(3):
    col = data[:, c]
    gmean = col.mean()
    gstd = col.std()
    z = (col - gmean) / gstd
    idx = np.where(np.abs(z) > 3.0)[0]
    print(f"ch{c+1}: n_outliers(|z|>3) = {len(idx)}", end="")
    if len(idx) > 0:
        print(f", min_idx={idx.min()}, max_idx={idx.max()}")
    else:
        print()

# focus window based on earlier finding: ch3 blocks 62,66,67 (bs=128) -> rows ~7936-8703
# let's do finer block scan for ch3 in window 7000-9500
col = data[:, 2]
gmean = col.mean()
gstd = col.std()
lo, hi = 7000, 9500
bs = 50
nb = (hi - lo) // bs
means = []
stds = []
for b in range(nb):
    seg = col[lo+b*bs: lo+(b+1)*bs]
    means.append(round(float(seg.mean()), 3))
    stds.append(round(float(seg.std()), 3))
print("ch3 window means (bs=50, start=7000):")
print(means)
print("ch3 window stds:")
print(stds)
