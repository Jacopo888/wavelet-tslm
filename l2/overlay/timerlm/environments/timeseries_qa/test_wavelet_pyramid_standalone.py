"""Standalone checks for wavelet_pyramid (numpy only): python test_wavelet_pyramid_standalone.py"""

import numpy as np

from wavelet_pyramid import WaveletPyramid, overview_text


def test_reconstruction_pow2():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 4096)
    p = WaveletPyramid(x)
    assert np.allclose(p.reconstruct(), x, atol=1e-10)


def test_reconstruction_odd_length():
    rng = np.random.default_rng(1)
    for n in (4097, 5000, 12345):
        x = rng.normal(0, 1, n)
        p = WaveletPyramid(x)
        assert np.allclose(p.reconstruct(), x, atol=1e-10), n


def test_spike_localization():
    rng = np.random.default_rng(2)
    x = np.sin(2 * np.pi * 0.002 * np.arange(8192)) + rng.normal(0, 0.3, 8192)
    x[5000] += 8.0
    p = WaveletPyramid(x)
    out = p.detail(1, 0, 8191, top=1)
    t = int(out.splitlines()[-1].split()[0].split("=")[1])
    assert abs(t - 5000) <= 2, out


def test_energy_hotspot():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.3, 16384)
    x[9000:9500] += 2.5 * np.sin(2 * np.pi * 0.12 * np.arange(500))
    p = WaveletPyramid(x)
    out = p.energy(2, 0, 16383, buckets=16)
    rows = [r for r in out.splitlines()[1:]]
    shares = [float(r.split()[-1].rstrip("%")) for r in rows]
    hot = max(range(16), key=lambda i: shares[i])
    lo = int(rows[hot].split(",")[0].lstrip("["))
    hi = int(rows[hot].split(",")[1].split("]")[0])
    assert lo <= 9250 <= hi and shares[hot] > 25, out


def test_approx_matches_block_means():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 2048)
    p = WaveletPyramid(x, levels=4)
    out = p.approx_view(4, 0, 2047, max_blocks=200)
    vals = [float(r.split("m=")[1]) for r in out.splitlines()[1:]]
    ref = x.reshape(-1, 16).mean(axis=1)
    assert np.allclose(vals, ref, atol=5e-3)


def test_overview_runs():
    rng = np.random.default_rng(5)
    ps = {f"ch{i}": WaveletPyramid(rng.normal(0, 1, 10000)) for i in (1, 2)}
    txt = overview_text(ps)
    assert "ch1" in txt and "ch2" in txt and "L1:" in txt
    assert len(txt) < 20000, len(txt)


if __name__ == "__main__":
    for fn in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"{fn.__name__}: OK")
    print("all tests passed")
