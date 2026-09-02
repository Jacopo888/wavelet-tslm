import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
ch2 = data[:,1]

print('--- boundary start region (fine, every sample) ---')
for i in range(10495, 10535):
    print(i, round(float(ch2[i]),4))

print('--- shape overview (every 8th sample) ---')
for i in range(10500, 11220, 8):
    print(i, round(float(ch2[i]),4))

print('--- boundary end region (fine, every sample) ---')
for i in range(11185, 11235):
    print(i, round(float(ch2[i]),4))
