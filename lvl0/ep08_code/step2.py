import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
chans = {'ch1': data[:,0], 'ch2': data[:,1], 'ch3': data[:,2]}
n = data.shape[0]

for name, ch in chans.items():
    gm, gs = ch.mean(), ch.std()
    z = np.abs((ch - gm) / (gs + 1e-12))
    top_idx = np.argsort(z)[-5:][::-1]
    print(name, 'top_abs_z_idx', top_idx.tolist(), 'vals=', [round(float(z[i]),2) for i in top_idx])

    # diff series (captures oscillation/high-freq anomalies)
    d = np.diff(ch)
    dm, ds = d.mean(), d.std()
    zd = np.abs((d - dm) / (ds + 1e-12))
    top_idx_d = np.argsort(zd)[-5:][::-1]
    print(name, 'top_abs_zdiff_idx', top_idx_d.tolist(), 'vals=', [round(float(zd[i]),2) for i in top_idx_d])

    # block std with block=32
    block = 32
    nb = n // block
    ch_b = ch[:nb*block].reshape(nb, block)
    stds = ch_b.std(axis=1)
    mstd, sstd = stds.mean(), stds.std()
    zstd = (stds - mstd) / (sstd + 1e-12)
    top_b = np.argsort(np.abs(zstd))[-5:][::-1]
    print(name, 'top_std_blocks(32)', top_b.tolist(), 'z=', [round(float(zstd[i]),2) for i in top_b])
