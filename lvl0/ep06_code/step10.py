import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1, dtype=np.float64)
col = data[:, 2]  # ch3
n = len(col)

# smooth with moving average to remove fast noise, keep slow oscillation
w = 21
kernel = np.ones(w) / w
sm = np.convolve(col, kernel, mode='same')

# find local minima and maxima of smoothed signal
d = np.diff(sm)
sign = np.sign(d)
# local min: sign goes from - to +
minima = []
maxima = []
for i in range(1, len(sign)):
    if sign[i-1] < 0 and sign[i] > 0:
        minima.append(i)
    if sign[i-1] > 0 and sign[i] < 0:
        maxima.append(i)

print(f"n minima={len(minima)}, n maxima={len(maxima)}")
print("minima idx:", minima)
print("minima val(sm):", [round(float(sm[i]),2) for i in minima])
print("maxima idx:", maxima)
print("maxima val(sm):", [round(float(sm[i]),2) for i in maxima])
