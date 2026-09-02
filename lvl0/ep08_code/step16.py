import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
ch2 = data[:,1]
d = np.diff(ch2)

for i in range(11208, 11224):
    print(i, round(float(d[i]),4))
