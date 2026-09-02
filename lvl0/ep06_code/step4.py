import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1, dtype=np.float64)
col = data[:, 2]  # ch3

lo, hi = 7500, 9000
w = 20
nb = (hi - lo) // w
means = []
ptp = []
for b in range(nb):
    seg = col[lo+b*w: lo+(b+1)*w]
    means.append(round(float(seg.mean()), 2))
    ptp.append(round(float(seg.max() - seg.min()), 2))

print(f"ch3 window {lo}-{hi} w={w} (index = {lo} + i*{w}):")
print("means:", means)
print("ptp  :", ptp)
