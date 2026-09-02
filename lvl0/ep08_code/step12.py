import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
chans = {'ch1': data[:,0], 'ch2': data[:,1], 'ch3': data[:,2]}

lo, hi = 10300, 11450
block = 32
nb = (hi-lo)//block

for name, ch in chans.items():
    seg = ch[lo:hi]
    d = np.diff(seg)
    print(name)
    for b in range(nb):
        db = d[b*block:(b+1)*block-0] if b*block+block-1 <= len(d) else d[b*block:]
        if len(db)==0: continue
        start = lo + b*block
        print(' start=%d dstd=%.4f'%(start, db.std()))
