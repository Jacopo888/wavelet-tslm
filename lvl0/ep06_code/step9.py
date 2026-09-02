import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1, dtype=np.float64)
col = data[:, 2]  # ch3

def dump(lo, hi):
    seg = col[lo:hi]
    print(f"rows {lo}-{hi-1} (len={len(seg)}):")
    print([round(float(v), 2) for v in seg])

dump(7700, 7780)
dump(8760, 8840)
