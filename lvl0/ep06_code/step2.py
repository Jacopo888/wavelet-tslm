import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1, dtype=np.float64)
n = data.shape[0]
nblocks = 128
bs = n // nblocks

for c in range(3):
    col = data[:, c]
    gmean = col.mean()
    gstd = col.std()
    mscores = np.zeros(nblocks)
    sscores = np.zeros(nblocks)
    for b in range(nblocks):
        seg = col[b*bs:(b+1)*bs]
        mscores[b] = abs(seg.mean() - gmean) / (gstd + 1e-12)
        sscores[b] = abs(seg.std() - gstd) / (gstd + 1e-12)
    combined = np.maximum(mscores, sscores)
    order = np.argsort(combined)[::-1][:8]
    print(f"ch{c+1} gmean={gmean:.4f} gstd={gstd:.4f} median_score={np.median(combined):.3f}")
    print(f"  top blocks idx={list(order)}")
    print(f"  top scores  ={[round(float(combined[i]),3) for i in order]}")
    print(f"  top mscore  ={[round(float(mscores[i]),3) for i in order]}")
    print(f"  top sscore  ={[round(float(sscores[i]),3) for i in order]}")
