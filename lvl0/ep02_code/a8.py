import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1)
ch1 = data[:,0]

def show(lo, hi, label):
    print(f"--- {label} [{lo},{hi})")
    line = ""
    for i in range(lo, hi):
        line += f"{i}:{ch1[i]:.2f} "
        if (i-lo+1) % 10 == 0:
            print(line); line = ""
    if line: print(line)

show(10080, 10180, "start boundary region")
show(10870, 10970, "gap region")

