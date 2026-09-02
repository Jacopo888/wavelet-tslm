import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
chans = {'ch1': data[:,0], 'ch2': data[:,1], 'ch3': data[:,2]}
n = data.shape[0]
block = 128
nb = (n - 1) // block

for name, ch in chans.items():
    d = np.diff(ch)  # length n-1
    d_use = d[:nb*block]  # approx align
    d_b = d_use.reshape(nb, block)
    dstd = d_b.std(axis=1)
    med = np.median(dstd)
    mad = np.median(np.abs(dstd-med))*1.4826 + 1e-12
    z = (dstd-med)/mad
    top = np.argsort(np.abs(z))[-6:][::-1]
    print(name, 'median_dstd=%.4f'%med)
    for i in top:
        print(' block=%d start=%d dstd=%.4f z=%.2f'%(i, i*block, dstd[i], z[i]))
