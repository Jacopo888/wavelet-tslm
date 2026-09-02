import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
ch1 = data[:,0]
ch2 = data[:,1]
ch3 = data[:,2]

lo, hi = 10240, 11648  # zoom region
B = 16
nb = (hi-lo)//B
for name, x_full in [("ch1", ch1), ("ch2", ch2), ("ch3", ch3)]:
    x = x_full[lo:lo+nb*B].reshape(nb, B)
    bmean = x.mean(axis=1)
    bstd = x.std(axis=1)
    print(f"--- {name} blocks of {B} from {lo}")
    line=""
    for i in range(nb):
        start = lo + i*B
        line += f"[{start}]m={bmean[i]:.2f}s={bstd[i]:.2f} "
        if (i+1)%4==0:
            print(line); line=""
    if line: print(line)
