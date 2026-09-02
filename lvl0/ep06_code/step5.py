import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1, dtype=np.float64)
col = data[:, 2]  # ch3

def scan(lo, hi, w=20):
    nb = (hi - lo) // w
    means = []
    ptp = []
    for b in range(nb):
        seg = col[lo+b*w: lo+(b+1)*w]
        means.append(round(float(seg.mean()), 2))
        ptp.append(round(float(seg.max() - seg.min()), 2))
    return means, ptp

for lo, hi in [(0, 1500), (14000, 15500)]:
    means, ptp = scan(lo, hi)
    print(f"range {lo}-{hi}: mean_of_means={np.mean(means):.2f} std_of_means={np.std(means):.2f} max_abs_mean={max(abs(m) for m in means):.2f} max_ptp={max(ptp):.2f}")

# also global stats excluding suspect window 7900-8700 vs including
mask = np.ones(len(col), dtype=bool)
mask[7900:8700] = False
print("ch3 std excluding 7900-8700:", col[mask].std(), "mean:", col[mask].mean())
print("ch3 std of 7900:8700 segment:", col[7900:8700].std(), "mean:", col[7900:8700].mean())
