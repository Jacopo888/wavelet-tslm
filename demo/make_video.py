# -*- coding: utf-8 -*-
"""Demo video: how a long time series becomes LLM tokens in wavelet-TSLM.

Storyboard (~72 s):
  1 signal        the raw series, and the anomaly that is 0.9% of it
  2 cost          what uniform patching charges for that length
  3 analysis      Haar pyramid -> detail layers D1..DJ + approximation A
  4 selection     which layer/coefficients get a token (now: |d|/MAD top-k;
                  next: the LLM asks for a level with a zoom tool call)
  5 projection    the head slides along the chosen layer, each selected
                  coefficient -> (value, band, position) -> MLP -> embedding
  6 budget        256 tokens whatever L is, vs L/P for uniform patching
  7 answer        embeddings interleaved with text tokens -> the LLM

Numbers and mechanics mirror sft/tslm/src/models/wavelet_encoder.py and
configs/wavelet_precise_full.yaml (num_tokens=256, coarse_blocks=128,
n_pos_freqs=16, output_dim=512).

Usage:
  python make_video.py            # render wavelet_tokenization.mp4
  python make_video.py --stills   # dump 2 PNG per scene for checking
"""

import argparse
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, FancyArrowPatch

FPS = 30
CW, CH = 160.0, 90.0          # canvas units (16:9)

BG, PANEL, PANEL2 = "#0b0f16", "#131a25", "#1b2432"
FG, MUT, DIM = "#e8eef6", "#8d9aab", "#3d4757"
BLUE, GREEN, ORANGE = "#5aa9ff", "#3fd07a", "#ffa657"
RED, VIO = "#ff6b5e", "#b78cff"
LVL = ["#ff7bd5", "#f778ba", "#d98cf0", "#b78cff", "#8aa2ff", "#5aa9ff", "#4fc3f7"]
COARSE = GREEN

MONO = {"family": "DejaVu Sans Mono"}
STEPS = ["signal", "cost", "analysis", "selection", "projection", "budget",
         "answer"]

# ----------------------------------------------------------------- data ----
L = 16384
NUM_TOKENS, COARSE_BLOCKS, N_POS = 256, 128, 16
LEVELS = max(1, min(10, math.ceil(math.log2(max(2, L / COARSE_BLOCKS)))))   # 7
A0, A1 = 9184, 9328                                                        # anomaly


def make_signal():
    rng = np.random.default_rng(11)
    t = np.arange(L) / L
    x = (1.00 * np.sin(2 * np.pi * 2.3 * t)
         + 0.42 * np.sin(2 * np.pi * 9.0 * t + 0.9)
         + 0.16 * np.sin(2 * np.pi * 41.0 * t + 2.1))
    walk = np.cumsum(rng.standard_normal(L)) / np.sqrt(L)
    x = x + 0.55 * walk + 0.045 * rng.standard_normal(L)
    n = A1 - A0
    # deliberately modest: at full-view scale this is a blip, at the right
    # wavelet scale it is unmistakable. that gap is the whole point.
    x[A0:A1] += 0.34 * np.sin(2 * np.pi * np.arange(n) / 9.0) * np.hanning(n)
    x[A0:] += 0.18                       # small level shift left behind
    return x


def haar_pyramid(x, levels):
    """Same recursion as WaveletTSEncoder.haar_pyramid, in numpy."""
    details, a = [], x.astype(float).copy()
    r2 = math.sqrt(2.0)
    for _ in range(levels):
        if len(a) % 2:
            a = np.append(a, a[-1])
        ev, od = a[0::2], a[1::2]
        details.append((ev - od) / r2)
        a = (ev + od) / r2
    return details, a / (r2 ** levels)


SIG = make_signal()
MED = np.median(SIG)
SCALE = 1.4826 * np.median(np.abs(SIG - MED)) + 1e-8
XN = (SIG - MED) / SCALE
DETAILS, COARSE_BAND = haar_pyramid(XN, LEVELS)

# top-k selection over the pooled detail bands, exactly as the encoder does
_v, _b, _p, _s = [], [], [], []
for li, d in enumerate(DETAILS, start=1):
    sigma = 1.4826 * np.median(np.abs(d)) + 1e-8
    blk = 2 ** li
    _v.append(d)
    _b.append(np.full(len(d), li))
    _p.append((np.arange(len(d)) * blk + blk / 2) / L)
    _s.append(np.abs(d) / sigma)
DV = np.concatenate(_v)
DB = np.concatenate(_b)
DP = np.concatenate(_p)
DS = np.concatenate(_s)
N_DETAIL = NUM_TOKENS - len(COARSE_BAND)                     # 128
TOP = np.sort(np.argsort(DS)[::-1][:N_DETAIL])
SEL_BAND, SEL_POS, SEL_VAL, SEL_SCORE = DB[TOP], DP[TOP], DV[TOP], DS[TOP]
# the level the model zooms into: the one holding most of the selected budget
# inside the anomaly span
_in = (SEL_POS > A0 / L - 0.004) & (SEL_POS < A1 / L + 0.004)
ZOOM_L = int(np.bincount(SEL_BAND[_in], minlength=LEVELS + 1).argmax()) if _in.any() else 3
ZOOM_L = max(1, ZOOM_L)
ZOOM_D = DETAILS[ZOOM_L - 1]
ZOOM_SEL = np.where(SEL_BAND == ZOOM_L)[0]

# illustrative (random-init) token embeddings, built from the real features
_rng = np.random.default_rng(3)
WPROJ = _rng.standard_normal((1 + 2 * N_POS, 24)) * 0.55
BEMB = _rng.standard_normal((LEVELS + 1, 24)) * 0.35


def embed(val, band, pos):
    ang = pos * (2.0 ** np.arange(N_POS)) * np.pi
    feat = np.concatenate([[np.clip(val, -20, 20)], np.sin(ang), np.cos(ang)])
    return np.tanh(feat @ WPROJ + BEMB[band])


TOKENS = np.stack([embed(v, b, p) for v, b, p in zip(SEL_VAL, SEL_BAND, SEL_POS)])
CTOK = np.stack([embed(v, 0, (i * 2 ** LEVELS + 2 ** (LEVELS - 1)) / L)
                 for i, v in enumerate(COARSE_BAND)])

# ------------------------------------------------------------- plumbing ----
FIG = plt.figure(figsize=(16, 9), dpi=110, facecolor=BG)
AX = FIG.add_axes([0, 0, 1, 1])


def reset():
    AX.clear()
    AX.set_xlim(0, CW)
    AX.set_ylim(0, CH)
    AX.set_facecolor(BG)
    AX.set_axis_off()


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def seg(u, a, b):
    """Progress of a sub-animation running from a to b inside u in [0,1]."""
    return ease(0.0 if b <= a else (u - a) / (b - a))


def box(x, y, w, h, fc=PANEL, ec=None, lw=1.0, alpha=1.0, r=1.0, z=1, ls="-"):
    p = FancyBboxPatch((x + r, y + r), max(w - 2 * r, .1), max(h - 2 * r, .1),
                       boxstyle="round,pad=%.3f" % r, fc=fc, ec=ec or fc,
                       lw=lw, alpha=alpha, zorder=z, linestyle=ls)
    AX.add_patch(p)
    return p


def txt(x, y, s, size=11, color=FG, ha="left", va="center", weight="normal",
        alpha=1.0, mono=False, z=6, style="normal"):
    return AX.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                   fontweight=weight, alpha=max(0.0, min(1.0, alpha)),
                   zorder=z, style=style, **(MONO if mono else {}))


def envelope(y, nb):
    """min/max per bin, padding rather than truncating so bin i really is at
    i/nb of the series (otherwise every highlight drifts left)."""
    y = np.asarray(y, float)
    if len(y) <= nb:
        return y, y
    per = int(math.ceil(len(y) / nb))
    if per * nb > len(y):
        y = np.concatenate([y, np.full(per * nb - len(y), y[-1])])
    r = y.reshape(nb, per)
    return r.min(1), r.max(1)


def wave(x0, y0, w, h, y, color, vlim=None, frac=1.0, nb=760, lw=0.9,
         alpha=1.0, z=4, fill=True):
    lo, hi = envelope(y, nb)
    n = len(lo)
    k = max(2, int(n * min(max(frac, 0.0), 1.0)))
    vmin, vmax = vlim if vlim else (float(lo.min()), float(hi.max()))
    rngv = (vmax - vmin) or 1.0
    xs = x0 + np.linspace(0, w, n)[:k]
    ylo = y0 + (lo[:k] - vmin) / rngv * h
    yhi = y0 + (hi[:k] - vmin) / rngv * h
    if fill:
        AX.fill_between(xs, ylo, yhi, color=color, alpha=0.45 * alpha, lw=0,
                        zorder=z)
    AX.plot(xs, (ylo + yhi) / 2, color=color, lw=lw, alpha=alpha, zorder=z + .1)


def strip(x0, y0, w, h, arr, z=5, alpha=1.0):
    """arr: (n_tokens, dims) -> heat strip, tokens along x."""
    if len(arr) == 0 or w <= 0:
        return
    AX.imshow(arr.T, extent=[x0, x0 + w, y0, y0 + h], aspect="auto",
              cmap="magma", vmin=-1, vmax=1, zorder=z, alpha=alpha,
              interpolation="nearest")


def chrome(step, u_all):
    box(0, CH - 8.4, CW, 8.4, fc="#080c12", r=0.01, z=0)
    txt(6, CH - 4.2, "wavelet-TSLM", size=15, weight="bold")
    txt(30.5, CH - 4.1, "how a long time series becomes tokens the LLM can read",
        size=11, color=MUT)
    x = 96
    for i, s in enumerate(STEPS):
        on = i == step
        txt(x, CH - 4.1, s, size=9, mono=True,
            color=FG if on else DIM, weight="bold" if on else "normal")
        x += len(s) * 0.62 + 3.2
    AX.plot([0, CW * u_all], [0.35, 0.35], color=BLUE, lw=2.2, zorder=9)


SCENES = []


def scene(step, seconds):
    def deco(fn):
        SCENES.append((step, int(seconds * FPS), fn))
        return fn
    return deco


# --------------------------------------------------------------- scenes ----
@scene(0, 7.5)
def s_signal(u):
    box(8, 30, 144, 44, fc=PANEL, ec=PANEL2, lw=1.2)
    txt(12, 69.5, "raw time series", size=13, weight="bold")
    txt(12, 65.6, "AnomalyXL episode   L = 16,384 samples   1 of 8 channels",
        size=10, color=MUT)
    wave(12, 34, 136, 27, XN, BLUE, vlim=(XN.min(), XN.max()),
         frac=ease(u / .55))
    if u > .58:
        a = seg(u, .58, .74)
        cx = 12 + 136 * (A0 + A1) / 2 / L
        AX.add_patch(Rectangle((cx - 1.2, 34), 2.4, 27, fc=RED, alpha=.18 * a,
                               ec=RED, lw=1.2 * a, zorder=5))
        txt(cx + 3, 62.5, "anomaly: 144 samples = 0.9% of the series",
            size=10, color=RED, alpha=a)
    if u > .74:                                   # magnifier
        a = seg(u, .74, .92)
        box(96, 9, 56, 18, fc=PANEL, ec=RED, lw=1.2, alpha=a)
        piece = XN[A0 - 60:A1 + 60]
        wave(99, 12, 50, 12, piece, RED, frac=1.0, nb=len(piece), alpha=a,
             lw=1.1)
        txt(99, 24.6, "zoom x114: a fast burst, then a small level shift",
            size=9.5, color=MUT, alpha=a)
    if u > .8:
        a = seg(u, .8, .95)
        txt(8, 21, "the evidence lives at a scale the full view cannot show,",
            size=12.5, color=FG, alpha=a)
        txt(8, 16, "and the answer has to name it to the sample.",
            size=12.5, color=MUT, alpha=a)


@scene(1, 6.5)
def s_cost(u):
    box(8, 58, 144, 20, fc=PANEL, ec=PANEL2, lw=1.2)
    txt(12, 74, "uniform patching, the usual TSLM front-end", size=12.5,
        weight="bold")
    wave(12, 60.5, 136, 10, XN, DIM, vlim=(XN.min(), XN.max()), fill=False,
         lw=.7)
    n_show, k = 96, ease(u / .35)
    for i in range(int(n_show * k)):
        x = 12 + 136 * i / n_show
        AX.add_patch(Rectangle((x, 60.5), 136 / n_show * .82, 10, fc=ORANGE,
                               alpha=.30, ec=ORANGE, lw=.4, zorder=5))
    if u > .3:
        txt(12, 53, "L / P  =  16,384 / 16  =  1,024 tokens per channel",
            size=13, mono=True, color=ORANGE, alpha=seg(u, .3, .5))
    if u > .45:
        txt(12, 47.5, "x 8 channels  =  8,192 perceptual tokens for one question",
            size=13, mono=True, color=ORANGE, alpha=seg(u, .45, .62))
    if u > .6:
        a = seg(u, .6, .78)
        box(12, 24, 136, 17, fc="#2a1113", ec=RED, lw=1.4, alpha=a)
        txt(17, 35, "measured, not argued", size=11, color=RED, weight="bold",
            alpha=a)
        txt(17, 30.4, "the ChatTS front-end, same recipe and same GPUs as ours: "
            "1,500 of 1,500 training batches hit CUDA OOM.",
            size=11.5, color=FG, alpha=a)
        txt(17, 26.6, "zero optimisation steps. the token count is a function "
            "of L, and L wins.", size=11.5, color=MUT, alpha=a)
    if u > .8:
        txt(12, 15, "so stop spending tokens uniformly.", size=13.5, color=FG,
            alpha=seg(u, .8, .95))


@scene(2, 14.0)
def s_analysis(u):
    txt(8, 79, "multiresolution analysis, Haar", size=13, weight="bold")
    txt(56, 79.2, "one pass, no parameters, invertible", size=10, color=MUT)
    box(8, 68, 144, 8.5, fc=PANEL, ec=PANEL2)
    wave(22, 69.4, 116, 5.7, XN, BLUE, vlim=(XN.min(), XN.max()), lw=.7)
    txt(11, 71, "x", size=11, mono=True, color=BLUE, weight="bold")
    txt(15.4, 71, "%6d" % L, size=9, mono=True, color=MUT)

    if u > .06:
        a = seg(u, .06, .18)
        txt(46, 65.2, "pairwise:   d = (even - odd) / sqrt2      "
            "a = (even + odd) / sqrt2", size=11, mono=True, color=MUT, alpha=a)
        txt(46, 61.9, "keep d as a detail layer, then recurse on a",
            size=11, color=MUT, alpha=a)

    rows = LEVELS + 1                       # D1..D7 + A
    top, hgt, gap = 53.4, 4.5, 1.25
    for i in range(rows):
        st = .20 + i * .085
        a = seg(u, st, st + .075)
        if a <= 0.01:
            continue
        y = top - i * (hgt + gap)
        det = i < LEVELS
        y_data = DETAILS[i] if det else COARSE_BAND
        col = LVL[i] if det else COARSE
        name = ("D%d" % (i + 1)) if det else "A"
        box(8, y, 144, hgt, fc=PANEL, ec=PANEL2, alpha=a)
        vl = (min(y_data.min(), -.5), max(y_data.max(), .5))
        wave(22, y + .6, 116, hgt - 1.2, y_data, col, vlim=vl, frac=a, lw=.7,
             alpha=a, nb=520)
        txt(11, y + hgt / 2, name, size=11, mono=True, color=col,
            weight="bold", alpha=a)
        txt(15.4, y + hgt / 2, "%6d" % len(y_data), size=9, mono=True,
            color=MUT, alpha=a)
        txt(149.5, y + hgt / 2,
            ("width %d" % 2 ** (i + 1)) if det else "block means",
            size=8.5, mono=True, color=DIM, alpha=a, ha="right")

    if u > .82:
        a = seg(u, .82, .95)
        y = top - (ZOOM_L - 1) * (hgt + gap)
        AX.add_patch(Rectangle((8, y - .5), 144, hgt + 1, fc="none", ec=RED,
                               lw=1.6, alpha=a, zorder=8))
        cx = 22 + 116 * (A0 + A1) / 2 / L
        AX.add_patch(Rectangle((cx - .8, y - .5), 1.6, hgt + 1, fc=RED,
                               alpha=.25 * a, zorder=8))
        txt(8, 6.5, "what was 0.9%% of x is a handful of large coefficients "
            "in D%d." % ZOOM_L, size=12.5, color=FG, alpha=a)


@scene(3, 8.5)
def s_selection(u):
    txt(8, 79, "which coefficients deserve a token", size=13, weight="bold")

    box(8, 44, 70, 30, fc=PANEL, ec=GREEN, lw=1.3)
    txt(12, 70, "implemented now", size=10.5, color=GREEN, weight="bold")
    txt(12, 65.4, "score = |d| / (1.4826 * MAD)", size=12, mono=True)
    txt(12, 61, "per band, then top-k over all bands", size=11, color=MUT)
    txt(12, 55, "128 coarse tokens  +  128 detail tokens", size=11.5, mono=True,
        alpha=seg(u, .15, .35))
    txt(12, 50, "resolution follows the evidence,", size=11, color=MUT,
        alpha=seg(u, .25, .45))
    txt(12, 46.6, "the budget never moves", size=11, color=MUT,
        alpha=seg(u, .25, .45))

    box(82, 44, 70, 30, fc=PANEL, ec=VIO, lw=1.3, ls=(0, (4, 3)))
    txt(86, 70, "next: the agentic loop", size=10.5, color=VIO, weight="bold")
    call = 'zoom(level="D%d", span=[9.0k, 9.5k])' % ZOOM_L
    k = int(len(call) * seg(u, .25, .62))
    txt(86, 65.4, call[:k] + ("|" if k < len(call) else ""), size=11.5,
        mono=True, color=VIO)
    if u > .5:
        txt(86, 61, "the level becomes a decision the model makes",
            size=11, color=MUT, alpha=seg(u, .5, .68))
        txt(86, 57.6, "instead of a hyper-parameter we fix", size=11, color=MUT,
            alpha=seg(u, .5, .68))
        txt(86, 50, "same tokenizer, one more turn in the loop", size=11,
            color=MUT, alpha=seg(u, .62, .8), style="italic")

    if u > .55:
        a = seg(u, .55, .72)
        txt(8, 36, "where the 256 tokens actually went", size=11, color=MUT,
            alpha=a)
        counts = [len(COARSE_BAND)] + [int((SEL_BAND == l).sum())
                                       for l in range(1, LEVELS + 1)]
        x = 8.0
        fill = seg(u, .6, .95)
        for i, c in enumerate(counts):
            w = 144 * c / NUM_TOKENS
            col = COARSE if i == 0 else LVL[i - 1]
            ww = w * fill
            if ww > .05:
                box(x, 26, ww, 6, fc=col, alpha=.85 * a, r=.35)
            if w > 6:
                txt(x + w / 2, 29, "%d" % c, size=9.5, mono=True,
                    color="#0b0f16", ha="center", weight="bold",
                    alpha=a * fill)
                txt(x + w / 2, 22.5, "A" if i == 0 else "D%d" % i, size=9,
                    mono=True, color=col, ha="center", alpha=a)
            x += w
        txt(8, 14, "128 slots pinned to the shape of the whole series, "
            "128 chased to wherever the surprise is.",
            size=12.5, color=FG, alpha=seg(u, .8, .95))


@scene(4, 19.0)
def s_projection(u):
    txt(8, 79, "the head slides along the chosen layer", size=13, weight="bold")
    txt(72, 79.2, "D%d   %d coefficients   %d of them are in the budget"
        % (ZOOM_L, len(ZOOM_D), len(ZOOM_SEL)), size=10, color=MUT)

    ly, lh = 60.0, 13.0
    box(8, ly, 144, lh, fc=PANEL, ec=PANEL2)
    vl = (min(ZOOM_D.min(), -.5), max(ZOOM_D.max(), .5))
    wave(11, ly + 1, 138, lh - 2, ZOOM_D, LVL[ZOOM_L - 1], vlim=vl, lw=.8,
         nb=690)

    prog = ease(min(u / .82, 1.0))
    hx = 11 + 138 * prog
    n_sel = len(ZOOM_SEL)
    done = int(n_sel * prog)

    for j in range(done):                     # coefficients already collected
        x = 11 + 138 * SEL_POS[ZOOM_SEL[j]]
        AX.plot([x, x], [ly + 1, ly + lh - 1], color=RED, lw=1.0, alpha=.55,
                zorder=6)
    AX.add_patch(Rectangle((hx - 1.0, ly), 2.0, lh, fc=BLUE, alpha=.30,
                           ec=BLUE, lw=1.3, zorder=7))

    idx = ZOOM_SEL[min(done, n_sel - 1)]
    box(8, 34, 44, 20, fc=PANEL, ec=PANEL2)
    txt(11, 50.5, "current coefficient", size=10, color=MUT)
    txt(11, 46.6, "value    %+.2f" % SEL_VAL[idx], size=11.5, mono=True)
    txt(11, 43.1, "band     D%d" % SEL_BAND[idx], size=11.5, mono=True,
        color=LVL[SEL_BAND[idx] - 1])
    txt(11, 39.6, "position %.4f" % SEL_POS[idx], size=11.5, mono=True)
    txt(11, 36.1, "score    %.1f x MAD" % SEL_SCORE[idx], size=11.5, mono=True,
        color=RED)

    AX.add_patch(Polygon([[56, 52], [70, 47.5], [70, 40.5], [56, 36]],
                         closed=True, fc=PANEL2, ec=BLUE, lw=1.2, zorder=3))
    txt(63, 46.5, "MLP", size=11.5, ha="center", weight="bold")
    txt(63, 42.5, "33 -> 512", size=10, mono=True, ha="center", color=MUT)
    txt(56, 32.6, "[ value | sin/cos Fourier(position), 16 freqs ]  +  "
        "band embedding", size=10, mono=True, color=MUT)
    for y0, y1 in ((47.5, 46.0), (44.0, 44.0), (40.5, 42.0)):
        AX.add_patch(FancyArrowPatch((52, y0), (56, y1), color=DIM, lw=1.1,
                                     arrowstyle="-|>", mutation_scale=9,
                                     zorder=3))
    AX.add_patch(FancyArrowPatch((70, 44), (76, 44), color=BLUE, lw=1.4,
                                 arrowstyle="-|>", mutation_scale=12, zorder=3))

    box(76, 34, 76, 20, fc=PANEL, ec=PANEL2)
    txt(79, 50.5, "token sequence", size=10, color=MUT)
    txt(149, 50.5, "%3d / %d" % (len(COARSE_BAND) + done, NUM_TOKENS),
        size=11, mono=True, color=BLUE, ha="right")
    allt = np.concatenate([CTOK, TOKENS[ZOOM_SEL[:max(done, 1)]]])
    full_w = 70.0
    strip(79, 36, full_w * len(allt) / NUM_TOKENS, 12, allt)
    AX.add_patch(Rectangle((79, 36), full_w, 12, fc="none", ec=DIM, lw=.8,
                           zorder=6))
    xsplit = 79 + full_w * len(CTOK) / NUM_TOKENS
    AX.plot([xsplit, xsplit], [36, 48], color=BG, lw=1.4, zorder=7)
    txt((79 + xsplit) / 2, 34.6, "coarse band", size=8.5, mono=True,
        color=COARSE, ha="center")
    txt(79 + full_w * .74, 34.6, "details, in position order", size=8.5,
        mono=True, color=LVL[ZOOM_L - 1], ha="center")

    if u > .30:
        a = seg(u, .30, .45)
        txt(8, 25, "every token carries what the value is, how wide it is, "
            "and where it sits,", size=12.5, color=FG, alpha=a)
        txt(8, 20.5, "so the model can answer in sample coordinates instead of "
            "somewhere in the middle.", size=12.5, color=MUT, alpha=a)
    if u > .58:
        txt(8, 13, "a projector then lifts 512 -> d_LLM and the tokens take "
            "their place in the prompt.", size=11.5, color=MUT,
            alpha=seg(u, .58, .72))
    if u > .80:
        txt(8, 7.5, "L never enters the count: the same 256 slots describe "
            "16k samples or 131k.", size=11.5, color=BLUE,
            alpha=seg(u, .80, .93))


@scene(5, 9.5)
def s_budget(u):
    txt(8, 79, "what this buys", size=13, weight="bold")
    x0, y0, w, h = 14, 22, 84, 44
    box(8, 16, 96, 58, fc=PANEL, ec=PANEL2)
    Ls = np.array([2 ** k for k in range(10, 18)])
    ylo, yhi = math.log10(64), math.log10(20000)

    def ypix(v):
        return y0 + (math.log10(v) - ylo) / (yhi - ylo) * h

    for v in (100, 1000, 10000):
        yy = ypix(v)
        AX.plot([x0, x0 + w], [yy, yy], color=DIM, lw=.6, alpha=.5, zorder=2)
        txt(x0 - 1.5, yy, "{:,}".format(v), size=9, mono=True, color=DIM,
            ha="right")
    for i, v in enumerate(Ls):
        if i % 2 == 0:
            txt(x0 + w * i / (len(Ls) - 1), 19.5, "%dk" % (v // 1024),
                size=9, mono=True, color=DIM, ha="center")
    txt(x0 + w / 2, 17.4, "series length L", size=10, color=MUT, ha="center")
    txt(x0 - 1.5, y0 + h + 3.5, "perceptual tokens per channel", size=10,
        color=MUT)

    xs = x0 + w * np.arange(len(Ls)) / (len(Ls) - 1)
    k = max(2, int(len(Ls) * ease(u / .45)))
    AX.plot(xs[:k], [ypix(v / 16) for v in Ls[:k]], color=ORANGE, lw=2.4,
            zorder=5, marker="o", ms=4)
    AX.plot(xs[:k], [ypix(NUM_TOKENS)] * k, color=BLUE, lw=2.4, zorder=5)
    txt(x0 + w * .40, ypix(Ls[4] / 16) + 3.4, "uniform patching, L/16",
        size=11, mono=True, color=ORANGE, alpha=seg(u, .3, .45))
    txt(x0 + w * .26, ypix(NUM_TOKENS) - 4, "wavelet budget, 256, flat",
        size=11, mono=True, color=BLUE, alpha=seg(u, .35, .5))
    if u > .5:
        a = seg(u, .5, .65)
        AX.plot([xs[-1]], [ypix(Ls[-1] / 16)], marker="x", ms=11, color=RED,
                mew=2.4, zorder=6, alpha=a)
        txt(xs[-1] - 2, ypix(Ls[-1] / 16) + 4, "L = 131k, 8,192 tokens: OOM",
            size=10.5, mono=True, color=RED, ha="right", alpha=a)

    box(108, 16, 44, 58, fc=PANEL, ec=PANEL2)
    txt(112, 69, "the trade", size=11, color=MUT)
    items = [(BLUE, "constant memory in L"),
             (BLUE, "no length ceiling"),
             (GREEN, "coarse shape always kept"),
             (GREEN, "fine detail where it matters"),
             (ORANGE, "lossy by construction"),
             (ORANGE, "selection is a hypothesis,"),
             (ORANGE, "which is why level 2 lets"),
             (ORANGE, "the model revise it")]
    for i, (c, s) in enumerate(items):
        a = seg(u, .35 + i * .06, .45 + i * .06)
        txt(112, 62 - i * 5.4, "-", size=12, color=c, alpha=a)
        txt(115, 62 - i * 5.4, s, size=11, color=FG if c != ORANGE else MUT,
            alpha=a)


@scene(6, 10.0)
def s_answer(u):
    txt(8, 79, "into the language model", size=13, weight="bold")
    box(8, 55, 144, 18, fc=PANEL, ec=PANEL2)
    txt(12, 69.5, "prompt", size=10, color=MUT)
    pieces = [("Which channel", "t"), ("carries the anomaly in", "t"),
              ("<ts>", "s"), ("? Give start and end", "t")]
    x = 12.0
    for i, (s, kind) in enumerate(pieces):
        a = seg(u, .05 + i * .04, .14 + i * .04)
        if kind == "t":
            wbox = len(s) * 1.05 + 2
            box(x, 58, wbox, 6.4, fc=PANEL2, alpha=a, r=.5)
            txt(x + wbox / 2, 61.2, s, size=10.5, mono=True, ha="center",
                alpha=a)
        else:
            wbox = 34
            strip(x, 58, wbox, 6.4, np.concatenate([CTOK, TOKENS])[::2],
                  alpha=a)
            AX.add_patch(Rectangle((x, 58), wbox, 6.4, fc="none", ec=BLUE,
                                   lw=1.1, alpha=a, zorder=6))
            txt(x + wbox / 2, 56.2, "256 wavelet tokens", size=8.5, mono=True,
                color=BLUE, ha="center", alpha=a)
        x += wbox + 2

    if u > .38:
        a = seg(u, .38, .5)
        box(50, 40, 60, 10, fc=PANEL2, ec=BLUE, lw=1.3, alpha=a)
        txt(80, 45, "Qwen3.5-4B  +  LoRA r16", size=12.5, mono=True,
            ha="center", alpha=a)
        AX.add_patch(FancyArrowPatch((80, 55), (80, 50), color=DIM, lw=1.4,
                                     arrowstyle="-|>", mutation_scale=12,
                                     alpha=a))
        AX.add_patch(FancyArrowPatch((80, 40), (80, 34), color=DIM, lw=1.4,
                                     arrowstyle="-|>", mutation_scale=12,
                                     alpha=a))

    if u > .52:
        ans = '{"channel": 3, "start": 9184, "end": 9328, "magnitude": 2.41}'
        k = int(len(ans) * seg(u, .52, .82))
        box(28, 22, 104, 10, fc=PANEL, ec=GREEN, lw=1.3)
        txt(32, 27, ans[:k] + ("|" if k < len(ans) else ""), size=12,
            mono=True, color=GREEN)

    if u > .84:
        a = seg(u, .84, 1.0)
        txt(80, 12, "one perception budget, many resolutions, "
            "sample-precise answers.", size=13, color=FG, ha="center", alpha=a)
        txt(80, 6.5, "wavelet-TSLM   multiresolution tokenization for long "
            "time series", size=10.5, color=MUT, ha="center", alpha=a)


# ---------------------------------------------------------------- render ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="wavelet_tokenization.mp4")
    ap.add_argument("--stills", action="store_true")
    ap.add_argument("--fps", type=int, default=FPS)
    args = ap.parse_args()

    total = sum(n for _, n, _ in SCENES)
    here = os.path.dirname(os.path.abspath(__file__))

    if args.stills:
        d = os.path.join(here, "stills")
        os.makedirs(d, exist_ok=True)
        gi = 0
        for si, (step, n, fn) in enumerate(SCENES):
            for frac in (0.45, 0.99):
                reset()
                fn(frac)
                chrome(step, (gi + n * frac) / total)
                FIG.savefig(os.path.join(d, "s%d_%02d.png"
                                         % (si, int(frac * 100))),
                            facecolor=BG)
            gi += n
        print("stills ->", d)
        return

    out = os.path.join(here, args.out)
    writer = FFMpegWriter(fps=args.fps, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "18",
                                      "-preset", "medium"])
    gi = 0
    with writer.saving(FIG, out, dpi=110):
        for step, n, fn in SCENES:
            for f in range(n):
                reset()
                fn(f / max(n - 1, 1))
                chrome(step, gi / total)
                writer.grab_frame(facecolor=BG)
                gi += 1
                if gi % 150 == 0:
                    print("  %d/%d frames" % (gi, total), flush=True)
    print("wrote", out, "(%.1f s, %d frames)" % (total / args.fps, total))


if __name__ == "__main__":
    main()
