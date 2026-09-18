"""Compact multiresolution anomaly navigator for long time series.

The skill is pre-imported by the RLM harness. The intended first call is simply::

    print(await wavelet_zoom())

It reads ``context.json`` itself and returns a short, ranked set of candidate intervals.
The implementation uses dyadic windows and Haar-like contrasts (local mean, scale/energy,
and within-window slope) rather than exposing thousands of coefficients to the policy.
"""

from __future__ import annotations

import json
import os

import numpy as np


_CACHE: dict[tuple[str, float], dict[str, np.ndarray]] = {}
_KIND_LABELS = {
    "mean": "Level Shift",
    "peak": "Transient Spike",
    "variance": "Change in Variance",
    "seasonality": "Change in Seasonality",
    "trend": "Change in Trend",
}


def _load_series(context_path: str = "context.json") -> dict[str, np.ndarray]:
    path = os.path.abspath(context_path)
    key = (path, os.path.getmtime(path))
    if key in _CACHE:
        return _CACHE[key]
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    series = payload.get("series", payload)
    if not isinstance(series, dict) or not series:
        raise ValueError(f"no series found in {context_path}")
    arrays: dict[str, np.ndarray] = {}
    for name, values in series.items():
        if isinstance(values, dict):
            values = values.get("values", values.get("data"))
        if isinstance(values, (list, tuple)) and values:
            arr = np.asarray(values, dtype=float).ravel()
            if arr.size >= 16 and np.all(np.isfinite(arr)):
                arrays[str(name)] = arr
    if not arrays:
        raise ValueError(f"no finite numeric channels found in {context_path}")
    lengths = {arr.size for arr in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"channels have different lengths: {sorted(lengths)}")
    _CACHE.clear()
    _CACHE[key] = arrays
    return arrays


def _prefix_moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate(([0.0], np.cumsum(x, dtype=float))),
        np.concatenate(([0.0], np.cumsum(x * x, dtype=float))),
    )


def _moments(prefix: tuple[np.ndarray, np.ndarray], starts: np.ndarray, ends: np.ndarray):
    """Vectorized population mean/variance on half-open intervals."""
    c1, c2 = prefix
    count = np.maximum(1, ends - starts)
    total = c1[ends] - c1[starts]
    mean = total / count
    var = np.maximum(0.0, (c2[ends] - c2[starts]) / count - mean * mean)
    return mean, var, total, count


def _positive_robust_z(values: np.ndarray) -> np.ndarray:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = max(1.4826 * mad, 0.1 * float(np.std(values)), 1e-3)
    return np.maximum(0.0, (values - med) / scale)


def _spectral_kind(segment: np.ndarray) -> str:
    """Separate a coherent oscillation from an unstructured variance burst."""
    if segment.size < 16:
        return "variance"
    t = np.linspace(-1.0, 1.0, segment.size)
    coef = np.polyfit(t, segment, 1)
    residual = segment - (coef[0] * t + coef[1])
    power = np.abs(np.fft.rfft(residual))[1:] ** 2
    if power.size == 0 or float(power.sum()) <= 0:
        return "variance"
    take = min(3, power.size)
    concentration = float(np.partition(power, -take)[-take:].sum() / power.sum())
    return "seasonality" if concentration >= 0.55 else "variance"


def _scan_channel(name: str, x: np.ndarray, query_start: int, query_end: int) -> list[dict]:
    span = query_end - query_start
    widths = [w for w in (8, 16, 32, 64, 128, 256, 512, 1024) if w <= span // 2]
    if not widths:
        widths = [max(8, span // 2)]
    candidates: list[dict] = []
    prefix = _prefix_moments(x)

    for width in widths:
        stride = max(2, width // 4)
        starts = np.arange(query_start, query_end - width + 1, stride, dtype=int)
        if starts.size == 0:
            continue
        ends = starts + width
        mean, var, _, _ = _moments(prefix, starts, ends)

        left_starts = np.maximum(query_start, starts - width)
        right_ends = np.minimum(query_end, ends + width)
        left_mean, left_var, left_sum, _ = _moments(prefix, left_starts, starts)
        right_mean, right_var, right_sum, _ = _moments(prefix, ends, right_ends)
        # At an outer boundary one flank can be empty. Give all weight to the other flank.
        left_count = starts - left_starts
        right_count = right_ends - ends
        base_count = np.maximum(1, left_count + right_count)
        base_mean = (left_sum + right_sum) / base_count
        base_second = (
            left_count * (left_var + left_mean * left_mean)
            + right_count * (right_var + right_mean * right_mean)
        ) / base_count
        base_var = np.maximum(1e-6, base_second - base_mean * base_mean)
        base_std = np.sqrt(base_var)

        half = max(1, width // 2)
        first_mean, _, _, _ = _moments(prefix, starts, starts + half)
        second_mean, _, _, _ = _moments(prefix, ends - half, ends)
        mean_effect = np.abs(mean - base_mean) / base_std
        variance_effect = np.abs(np.log((np.sqrt(var) + 1e-6) / (base_std + 1e-6)))
        trend_effect = np.abs(second_mean - first_mean) / base_std

        z_mean = _positive_robust_z(mean_effect)
        z_variance = _positive_robust_z(variance_effect)
        z_trend = _positive_robust_z(trend_effect)
        score = np.maximum.reduce((z_mean, z_variance, z_trend))

        # Only the two strongest windows at each scale are needed before cross-scale NMS.
        for idx in np.argsort(-score)[:2]:
            s, e = int(starts[idx]), int(ends[idx])
            feature_z = {
                "mean": float(z_mean[idx]),
                "variance": float(z_variance[idx]),
                "trend": float(z_trend[idx]),
            }
            dominant = max(feature_z, key=feature_z.get)
            segment = x[s:e]
            peak_sigma = float(np.quantile(np.abs(segment - base_mean[idx]), 0.95) / base_std[idx])
            mean_sigma = float(mean_effect[idx])
            # A Gaussian baseline contributes about 1.64 sigma at its 95th percentile.
            # Removing that expected noise floor makes the estimate useful even when the
            # selected dyadic window covers only part of a level shift.
            magnitude_sigma = max(mean_sigma, peak_sigma - 1.64)
            if dominant == "variance":
                kind = _spectral_kind(segment)
            elif dominant == "trend":
                kind = "trend"
            else:
                kind = "peak" if peak_sigma > max(2.5, 2.2 * mean_sigma) else "mean"
            candidates.append({
                "channel": name,
                "start": s,
                "end": e,
                "width": width,
                "score": float(score[idx]),
                "kind": _KIND_LABELS[kind],
                "magnitude_sigma": magnitude_sigma,
                "peak_sigma": peak_sigma,
            })
    return candidates


def _overlap_fraction(a: dict, b: dict) -> float:
    if a["channel"] != b["channel"]:
        return 0.0
    overlap = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    return overlap / max(1, min(a["width"], b["width"]))


def _rank_candidates(arrays: dict[str, np.ndarray], channel: str = "", start: int = 0,
                     end: int | None = None, top: int = 6) -> tuple[int, list[dict]]:
    names = list(arrays)
    if channel:
        if channel not in arrays:
            raise ValueError(f"unknown channel {channel!r}; available: {names}")
        names = [channel]
    n = arrays[names[0]].size
    query_start = max(0, int(start))
    query_end = n if end is None else min(n, int(end))
    if query_end - query_start < 16:
        raise ValueError("query range must contain at least 16 samples")

    raw = [candidate for name in names for candidate in _scan_channel(
        name, arrays[name], query_start, query_end
    )]
    raw.sort(key=lambda item: item["score"], reverse=True)
    kept: list[dict] = []
    for candidate in raw:
        if all(_overlap_fraction(candidate, previous) < 0.65 for previous in kept):
            kept.append(candidate)
        if len(kept) >= max(1, min(int(top), 12)):
            break
    return n, kept


def _render(n: int, names: list[str], candidates: list[dict], mode: str) -> str:
    lines = [
        f"WAVELET_{mode.upper()} n={n} channels={','.join(names)}; intervals are [start,end).",
        "Ranked anomaly candidates (score is relative within this series):",
    ]
    for rank, item in enumerate(candidates, 1):
        lines.append(
            f"{rank}. channel={item['channel']} start={item['start']} end={item['end']} "
            f"width={item['width']} score={item['score']:.1f} kind_hint={item['kind']!r} "
            f"magnitude_sigma={item['magnitude_sigma']:.2f} peak_sigma={item['peak_sigma']:.2f}"
        )
    if not candidates:
        lines.append("No candidate could be computed.")
    lines += [
        "Use the highest plausible candidate to produce the question's exact JSON now.",
        "Do not import wavelet_zoom and do not print context.json. If one refinement is needed, call "
        "print(await wavelet_zoom(mode='zoom', channel='<name>', start=<start>, end=<end>)).",
    ]
    return "\n".join(lines)


async def run(mode: str = "overview", channel: str = "", start: int = 0,
              end: int | None = None, top: int = 6) -> str:
    """Return compact ranked anomaly intervals from ``context.json``.

    Call ``print(await wavelet_zoom())`` first. Do not import the skill and do not load or
    print the raw JSON. ``mode='overview'`` scans all channels. ``mode='zoom'`` repeats the
    scan inside one half-open interval, normally around a candidate from the overview.

    Args:
        mode: ``overview`` or ``zoom``.
        channel: Exact channel name; required for ``zoom`` and optional for ``overview``.
        start: Inclusive query start.
        end: Exclusive query end; defaults to the series end.
        top: Number of ranked candidates, from 1 to 12.
    """
    try:
        if mode not in {"overview", "zoom"}:
            return "ERROR: mode must be 'overview' or 'zoom'. Start with print(await wavelet_zoom())."
        if mode == "zoom" and not channel:
            return "ERROR: zoom requires channel. Use the exact name shown by the overview."
        arrays = _load_series()
        n, candidates = _rank_candidates(arrays, channel=channel, start=start, end=end, top=top)
        shown_names = [channel] if channel else list(arrays)
        return _render(n, shown_names, candidates, mode)
    except Exception as exc:
        return f"ERROR: {exc}. Correct first call: print(await wavelet_zoom())"
