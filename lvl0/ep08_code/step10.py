import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
chans = {'ch1': data[:,0], 'ch2': data[:,1], 'ch3': data[:,2]}

lo, hi = 15000, 15500
block = 25
nb = (hi-lo)//block

for name, ch in chans.items():
    seg = ch[lo:hi]
    print(name)
    for b in range(nb):
        s = seg[b*block:(b+1)*block]
        sd = s - s.mean()
        denom = np.sum(sd**2)
        ac1 = np.sum(sd[:-1]*sd[1:])/denom if denom>1e-12 else 0.0
        std = s.std()
        start = lo + b*block
        print(' start=%d ac1=%.3f std=%.3f'%(start, ac1, std))
