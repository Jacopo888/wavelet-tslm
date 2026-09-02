import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
ch2 = data[:,1]
d = np.diff(ch2)

print('--- abs diffs near start ---')
for i in range(10505, 10522):
    print(i, round(float(d[i]),4))

print('--- abs diffs near end ---')
for i in range(11198, 11212):
    print(i, round(float(d[i]),4))
