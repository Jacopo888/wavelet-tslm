import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
chans = {'ch1': data[:,0], 'ch2': data[:,1], 'ch3': data[:,2]}
n = data.shape[0]
block = 128
nb = n // block

for name, ch in chans.items():
    ch_b = ch[:nb*block].reshape(nb, block)
    means = ch_b.mean(axis=1)
    stds = ch_b.std(axis=1)
    gm, gs = ch.mean(), ch.std()
    zmean = (means - gm) / (gs + 1e-12)
    # z-score of local std relative to distribution of block stds
    mstd, sstd = stds.mean(), stds.std()
    zstd = (stds - mstd) / (sstd + 1e-12)
    im = int(np.argmax(np.abs(zmean)))
    istd = int(np.argmax(np.abs(zstd)))
    print(name, 'n=',n, 'gm=%.4f'%gm, 'gs=%.4f'%gs,
          'maxmean_block=%d'%im, 'zmean=%.2f'%zmean[im],
          'maxstd_block=%d'%istd, 'zstd=%.2f'%zstd[istd])
