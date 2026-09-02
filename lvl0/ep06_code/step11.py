import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1, dtype=np.float64)
n = data.shape[0]
col3 = data[:, 2]

w = 101
kernel = np.ones(w) / w
sm = np.convolve(col3, kernel, mode='same')
# ignore edges affected by convolution boundary
valid = slice(w, n - w)

med = np.median(sm[valid])
mad = np.median(np.abs(sm[valid] - med))
thresh = med + 5 * mad * 1.4826
print(f"ch3 smoothed(w={w}): median={med:.3f} mad={mad:.3f} thresh={thresh:.3f}")

above = sm > thresh
idx = np.where(above)[0]
idx = idx[(idx >= w) & (idx < n - w)]
if len(idx) > 0:
    # find contiguous runs
    runs = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i - prev > 5:  # allow small gaps
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    print("runs (start,end):", runs)
else:
    print("no points above threshold")

# check ch1,ch2 in candidate window vs elsewhere
for c in range(3):
    col = data[:, c]
    seg = col[7680:8800]
    rest_mean = np.concatenate([col[:7680], col[8800:]]).mean()
    rest_std = np.concatenate([col[:7680], col[8800:]]).std()
    print(f"ch{c+1}: seg(7680:8800) mean={seg.mean():.3f} std={seg.std():.3f} min={seg.min():.3f} max={seg.max():.3f} | rest mean={rest_mean:.3f} std={rest_std:.3f}")
