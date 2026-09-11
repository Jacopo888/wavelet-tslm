"""Multiresolution (Haar) pyramid over a 1-D series, with compact text queries.

Core perception structure for the wavelet_zoom tool/skill: the series is
decomposed once into detail bands D1..DL plus block-mean approximations A1..AL,
and the agent navigates it through three query modes (detail / approx / energy)
that return short, token-budgeted text instead of raw samples.

Numpy-only on purpose: importable both from the timeseries_qa environment
(v0 tool) and from an rlm skill (v1) without extra dependencies.
"""

from __future__ import annotations

import math

import numpy as np

_SQRT2 = math.sqrt(2.0)


class WaveletPyramid:
    """Orthonormal Haar decomposition of one channel.

    Detail coefficient k at level L has time support [k*2^L, (k+1)*2^L).
    Approximations are stored rescaled to plain block means so their values
    live on the same scale as the raw series. Odd lengths at any level are
    handled by edge-padding one sample; coefficients whose support crosses
    the original end of the series are marked padded and excluded from the
    per-level noise estimate.
    """

    def __init__(self, x: np.ndarray, levels: int | None = None):
        x = np.asarray(x, dtype=float).ravel()
        if x.size < 2:
            raise ValueError("series too short")
        self.n = int(x.size)
        if levels is None:
            levels = max(1, math.ceil(math.log2(max(2, self.n / 128))))
        self.levels = int(min(levels, max(1, int(math.log2(self.n)))))

        self.details: dict[int, np.ndarray] = {}
        self.approx: dict[int, np.ndarray] = {}
        self._sigma: dict[int, float] = {}

        a = x
        for lev in range(1, self.levels + 1):
            if a.size % 2 == 1:
                a = np.append(a, a[-1])
            ev, od = a[0::2], a[1::2]
            d = (ev - od) / _SQRT2
            a = (ev + od) / _SQRT2
            self.details[lev] = d
            self.approx[lev] = a / (_SQRT2 ** lev)
            valid = d[: self._n_valid(lev, d.size)]
            med = float(np.median(np.abs(valid))) if valid.size else 0.0
            self._sigma[lev] = 1.4826 * med

    def _n_valid(self, level: int, size: int) -> int:
        """Coefficients whose support lies fully inside the original series."""
        return min(size, self.n // (2 ** level))

    def sigma(self, level: int) -> float:
        """Robust (MAD) noise scale of the detail band at `level`."""
        return self._sigma[level]

    def _clip(self, start: int, end: int) -> tuple[int, int]:
        return max(0, int(start)), min(self.n - 1, int(end))

    def detail(self, level: int, start: int, end: int, top: int = 40) -> str:
        """Largest |detail| coefficients whose support center falls in [start, end]."""
        level = self._check_level(level)
        s, e = self._clip(start, end)
        block = 2 ** level
        d = self.details[level]
        centers = np.arange(d.size) * block + block // 2
        mask = (centers >= s) & (centers <= e)
        dv, dc = d[mask], centers[mask]
        if dv.size == 0:
            return f"level={level} range=[{s},{e}] no coefficients at this level"
        sig = self._sigma[level]
        nbig = int(np.sum(np.abs(dv) > 3 * sig)) if sig > 0 else int(dv.size)
        order = np.argsort(-np.abs(dv))[: max(1, int(top))]
        order = order[np.argsort(dc[order])]
        lines = [
            f"level={level} range=[{s},{e}] coeffs_in_range={dv.size} "
            f"noise_sigma={sig:.3f} above_3sigma={nbig}",
            f"top {order.size} by |value| (t = support center, width {block}):",
        ]
        lines += [f"t={int(dc[i])} v={dv[i]:+.3f}" for i in order]
        return "\n".join(lines)

    def approx_view(self, level: int, start: int, end: int,
                    max_blocks: int = 200) -> str:
        """Block means (block = 2^level samples) covering [start, end]."""
        level = self._check_level(level)
        s, e = self._clip(start, end)
        block = 2 ** level
        a = self.approx[level]
        starts = np.arange(a.size) * block
        mask = (starts + block - 1 >= s) & (starts <= e)
        av, ast = a[mask], starts[mask]
        if av.size > max_blocks:
            return (f"ERROR: {av.size} blocks in range at level {level} "
                    f"(max {max_blocks}); use a higher level or narrow the range")
        lines = [f"level={level} range=[{s},{e}] block_means (block={block}):"]
        lines += [f"t={int(t0)} m={v:+.3f}" for t0, v in zip(ast, av)]
        return "\n".join(lines)

    def energy(self, level: int, start: int, end: int, buckets: int = 16) -> str:
        """Percent of the level's detail energy in `buckets` sub-windows of the range."""
        level = self._check_level(level)
        s, e = self._clip(start, end)
        block = 2 ** level
        d = self.details[level]
        centers = np.arange(d.size) * block + block // 2
        mask = (centers >= s) & (centers <= e)
        dv, dc = d[mask], centers[mask]
        if dv.size < buckets:
            return (f"ERROR: only {dv.size} coefficients in range at level {level}; "
                    f"use a finer level")
        e2 = dv * dv
        tot = float(e2.sum()) or 1.0
        lines = [f"level={level} range=[{s},{e}] energy % in {buckets} windows "
                 f"(uniform ~ {100 // buckets}%):"]
        for idx in np.array_split(np.arange(dv.size), buckets):
            t0, t1 = int(dc[idx[0]]), int(dc[idx[-1]])
            lines.append(f"[{t0},{t1}] {100.0 * float(e2[idx].sum()) / tot:.0f}%")
        return "\n".join(lines)

    def _check_level(self, level: int) -> int:
        level = int(level)
        if not 1 <= level <= self.levels:
            raise ValueError(f"level must be in [1,{self.levels}]")
        return level

    def reconstruct(self) -> np.ndarray:
        """Inverse transform (exactness check; padding truncated at the end)."""
        a = self.approx[self.levels] * (_SQRT2 ** self.levels)
        for lev in range(self.levels, 0, -1):
            d = self.details[lev]
            a = a[: d.size]
            out = np.empty(2 * d.size)
            out[0::2] = (a + d) / _SQRT2
            out[1::2] = (a - d) / _SQRT2
            a = out
        return a[: self.n]


def overview_text(pyramids: dict[str, "WaveletPyramid"], windows: int = 16,
                  coarse_blocks: int = 160) -> str:
    """Always-in-context coarse view: top-level block means per channel plus a
    per-channel, per-level map of where each detail band's energy sits."""
    names = list(pyramids)
    any_p = pyramids[names[0]]
    lines = [f"COARSE VIEW - {any_p.n} samples, {len(names)} channel(s): "
             f"{', '.join(names)}."]
    for name, p in pyramids.items():
        a = p.approx[p.levels]
        stride = max(1, math.ceil(a.size / coarse_blocks))
        block = 2 ** p.levels * stride
        shown = a[: a.size // stride * stride].reshape(-1, stride).mean(axis=1) \
            if stride > 1 else a
        vals = " ".join(f"{v:.2f}" for v in shown)
        lines.append(f"{name} block_means (block={block}): {vals}")
    lines.append(f"Detail energy map: % of each level's energy in {windows} equal "
                 f"time windows (uniform ~ {100 // windows}%). L1=finest.")
    for name, p in pyramids.items():
        for lev in range(1, p.levels + 1):
            d = p.details[lev]
            e2 = d * d
            tot = float(e2.sum()) or 1.0
            pct = " ".join(f"{100.0 * float(b.sum()) / tot:.0f}"
                           for b in np.array_split(e2, windows))
            lines.append(f"{name} L{lev}: {pct}")
    return "\n".join(lines)
