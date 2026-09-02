import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1, dtype=np.float64)
col = data[:, 2]  # ch3

lo, hi, w = 7550, 8850, 25
nb = (hi - lo) // w
means = []
mins = []
for b in range(nb):
    seg = col[lo+b*w: lo+(b+1)*w]
    means.append(round(float(seg.mean()), 2))
    mins.append(round(float(seg.min()), 2))

print(f"lo={lo} w={w} nb={nb} (row = {lo} + i*{w})")
print("means:", means)
print("mins :", mins)
