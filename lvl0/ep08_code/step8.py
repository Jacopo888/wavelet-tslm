import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
chans = {'ch1': data[:,0], 'ch2': data[:,1], 'ch3': data[:,2]}
n = data.shape[0]

block = 128
nb = n // block

for name, ch in chans.items():
    # global lag-1 autocorr
    x = ch - ch.mean()
    global_ac1 = np.sum(x[:-1]*x[1:]) / np.sum(x**2)

    ac1 = np.zeros(nb)
    for b in range(nb):
        seg = ch[b*block:(b+1)*block]
        s = seg - seg.mean()
        denom = np.sum(s**2)
        ac1[b] = np.sum(s[:-1]*s[1:]) / denom if denom > 1e-12 else 0.0

    med = np.median(ac1)
    mad = np.median(np.abs(ac1-med))*1.4826 + 1e-12
    z = np.abs(ac1-med)/mad
    top = np.argsort(z)[-5:][::-1]
    print(name, 'global_ac1=%.3f'%global_ac1, 'median_block_ac1=%.3f'%med)
    print(name, 'top_blocks(128)=', top.tolist(), 'ac1=', [round(float(ac1[i]),3) for i in top], 'z=', [round(float(z[i]),2) for i in top])
