import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
chans = {'ch1': data[:,0], 'ch2': data[:,1], 'ch3': data[:,2]}
n = data.shape[0]

def rolling_std(ch, w):
    csum = np.cumsum(np.insert(ch, 0, 0))
    csum2 = np.cumsum(np.insert(ch**2, 0, 0))
    rsum = csum[w:] - csum[:-w]
    rsum2 = csum2[w:] - csum2[:-w]
    rmean = rsum / w
    rvar = np.clip(rsum2/w - rmean**2, 1e-12, None)
    return np.sqrt(rvar)

for name, ch in chans.items():
    for w in [50, 100, 200, 400]:
        rstd = rolling_std(ch, w)
        med = np.median(rstd)
        top3 = np.argsort(rstd)[-3:][::-1]
        print(name, 'w=%d'%w, 'median=%.3f'%med,
              'top3_start=', top3.tolist(),
              'top3_val=', [round(float(rstd[i]),3) for i in top3],
              'ratio=', [round(float(rstd[i]/med),2) for i in top3])
