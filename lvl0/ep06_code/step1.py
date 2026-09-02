import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1, dtype=np.float64)
n = data.shape[0]
print("n rows:", n, "shape:", data.shape)

nblocks = 128
bs = n // nblocks  # block size
scores = np.zeros((nblocks, 3))

for c in range(3):
    col = data[:, c]
    gmean = col.mean()
    gstd = col.std()
    for b in range(nblocks):
        seg = col[b*bs:(b+1)*bs]
        # combine mean shift and std shift signals
        mscore = abs(seg.mean() - gmean) / (gstd + 1e-12)
        sscore = abs(seg.std() - gstd) / (gstd + 1e-12)
        scores[b, c] = max(mscore, sscore)

maxscore = scores.max(axis=1)
# print as ints scaled to keep short
print("block scores (max over channels, x100 rounded int):")
print(list((maxscore*100).astype(int)))

top = np.argsort(maxscore)[::-1][:5]
print("top blocks:", top, "scores:", maxscore[top])
