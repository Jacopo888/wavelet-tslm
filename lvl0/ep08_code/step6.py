import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
ch1,ch2,ch3 = data[:,0], data[:,1], data[:,2]

lo, hi = 13650, 13850
print('idx  ch1      ch2      ch3')
for i in range(lo, hi):
    print(i, round(float(ch1[i]),3), round(float(ch2[i]),3), round(float(ch3[i]),3))
