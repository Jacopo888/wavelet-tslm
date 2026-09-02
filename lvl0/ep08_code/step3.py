import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
chans = {'ch1': data[:,0], 'ch2': data[:,1], 'ch3': data[:,2]}
n = data.shape[0]

def rolling_mean_std(ch, w):
    csum = np.cumsum(np.insert(ch, 0, 0))
    csum2 = np.cumsum(np.insert(ch**2, 0, 0))
    rsum = csum[w:] - csum[:-w]
    rsum2 = csum2[w:] - csum2[:-w]
    rmean = rsum / w
    rvar = rsum2/w - rmean**2
    rvar = np.clip(rvar, 1e-12, None)
    rstd = np.sqrt(rvar)
    return rmean, rstd  # index i => window starting at i, [i, i+w)

for name, ch in chans.items():
    med = np.median(ch)
    mad = np.median(np.abs(ch - med)) * 1.4826 + 1e-12
    for w in [16, 64, 256]:
        rmean, rstd = rolling_mean_std(ch, w)
        zmean = np.abs(rmean - med) / (mad)
        # robust baseline for rstd via its own median/mad
        smed = np.median(rstd)
        smad = np.median(np.abs(rstd - smed)) * 1.4826 + 1e-12
        zstd = np.abs(rstd - smed) / smad
        im = int(np.argmax(zmean))
        istd = int(np.argmax(zstd))
        print(name, 'w=%d'%w, 'best_mean_start=%d'%im, 'zmean=%.2f'%zmean[im],
              'best_std_start=%d'%istd, 'zstd=%.2f'%zstd[istd])
