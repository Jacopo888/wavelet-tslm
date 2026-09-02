import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1, dtype=np.float64)
col = data[:, 2]  # ch3
n = len(col)

w = 512
nb = n // w
mins = []
maxs = []
means = []
for b in range(nb):
    seg = col[b*w:(b+1)*w]
    mins.append(round(float(seg.min()), 2))
    maxs.append(round(float(seg.max()), 2))
    means.append(round(float(seg.mean()), 2))

print(f"ch3 windows w={w}, n windows={nb}, index=b*{w}")
print("mins :", mins)
print("maxs :", maxs)
print("means:", means)
