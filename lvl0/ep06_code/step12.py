import numpy as np

data = np.genfromtxt('raw.csv', delimiter=',', skip_header=1, dtype=np.float64)
col = data[:, 2]

lo, hi = 6800, 9500
seg = col[lo:hi]
w = 60
sw = np.lib.stride_tricks.sliding_window_view(seg, w)
roll_min = sw.min(axis=1)  # roll_min[k] corresponds to window [lo+k, lo+k+w-1], center ~ lo+k+w//2

thresh = -2.6
below = roll_min <= thresh  # "normal deep trough" present in this window
# find contiguous region where below is False (i.e., anomalous, no deep trough) of significant length
n = len(below)
runs = []
i = 0
while i < n:
    if not below[i]:
        j = i
        while j < n and not below[j]:
            j += 1
        runs.append((i, j - 1, j - i))
        i = j
    else:
        i += 1

runs.sort(key=lambda r: -r[2])
print("top 'no-deep-trough' runs (win_start_idx, win_end_idx, length) relative to seg:")
for r in runs[:5]:
    center_start = lo + r[0] + w // 2
    center_end = lo + r[1] + w // 2
    print(f"  win[{r[0]},{r[1]}] len={r[2]} -> approx row range [{center_start},{center_end}]")

# for the biggest run, find precise boundary: last raw point <= thresh before run, first raw point <= thresh after run
best = runs[0]
win_lo = lo + best[0]
win_hi = lo + best[1] + w  # end of last window
print(f"searching precise raw crossing near row {win_lo-70} to {win_hi+70}")
search_seg = col[win_lo-70: win_hi+70]
below_raw = search_seg <= thresh
idx_below = np.where(below_raw)[0] + (win_lo - 70)
print("raw points <= thresh in extended search zone:", list(idx_below))
