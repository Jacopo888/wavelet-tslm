import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
ch2 = data[:,1]
d = np.diff(ch2)  # length n-1; d[i] = ch2[i+1]-ch2[i]

w = 10
csum = np.cumsum(np.insert(d, 0, 0))
csum2 = np.cumsum(np.insert(d**2, 0, 0))

def rstd_at(i):
    # window covering d[i:i+w]
    s = csum[i+w]-csum[i]
    s2 = csum2[i+w]-csum2[i]
    m = s/w
    v = max(s2/w - m*m, 0)
    return v**0.5

print('--- start boundary (rolling diff-std, w=10) ---')
for i in range(10480, 10540):
    print(i, round(rstd_at(i),4))

print('--- end boundary (rolling diff-std, w=10) ---')
for i in range(11180, 11235):
    print(i, round(rstd_at(i),4))
