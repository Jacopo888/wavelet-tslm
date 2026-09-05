"""CPU tests for RecurrentWaveletTSEncoder (v3). No GPU, no Qwen, no data.

Run:  python v3-recurrent/patch/test_encoder_v3.py      (or pytest -q on it)
On the node the same file runs from ~/patch after the overlay.

The v1 file (kaggle/patch/wavelet_encoder.py) is imported through a temporary
`src.models` package so the test works locally and on the node alike.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
V1_CANDIDATES = [
    HERE / "wavelet_encoder.py",                                   # node: ~/patch
    HERE.parent.parent / "kaggle" / "patch" / "wavelet_encoder.py",  # repo
]


def _install_pkg() -> None:
    if "src.models.wavelet_encoder_v3" in sys.modules:
        return
    v1 = next((p for p in V1_CANDIDATES if p.exists()), None)
    assert v1 is not None, f"wavelet_encoder.py (v1) not found in {V1_CANDIDATES}"
    tmp = Path(tempfile.mkdtemp(prefix="v3pkg_"))
    (tmp / "src" / "models").mkdir(parents=True)
    (tmp / "src" / "__init__.py").write_text("")
    (tmp / "src" / "models" / "__init__.py").write_text("")
    shutil.copy(v1, tmp / "src" / "models" / "wavelet_encoder.py")
    shutil.copy(HERE / "wavelet_encoder_v3.py", tmp / "src" / "models" / "wavelet_encoder_v3.py")
    sys.path.insert(0, str(tmp))


_install_pkg()
from src.models.wavelet_encoder_v3 import RecurrentWaveletTSEncoder  # noqa: E402

D, K = 512, 256


def make(**kw) -> RecurrentWaveletTSEncoder:
    torch.manual_seed(0)
    return RecurrentWaveletTSEncoder(output_dim=D, num_tokens=K, **kw).eval()


def expected_steps(L: int, coarse_blocks: int = 128, max_levels: int = 10) -> int:
    return 1 + max(1, min(max_levels, math.ceil(math.log2(max(2, L / coarse_blocks)))))


def test_shapes_and_steps():
    enc = make()
    for L in (64, 1000, 4096, 131072):
        x = torch.randn(2, L)
        with torch.no_grad():
            y = enc(x)
        assert y.shape == (2, K, D), (L, y.shape)
        assert torch.isfinite(y).all()
        assert enc.last_steps == expected_steps(L), (L, enc.last_steps)
        assert enc.last_depth == enc.last_steps
        assert enc.get_num_output_patches(L) == K


def test_multichannel_and_multivariable():
    enc = make()
    with torch.no_grad():
        y = enc(torch.randn(16, 4096))          # all channels of one example, as ChatTS does
    assert y.shape == (16, K, D)
    enc_mv = make(multi_variable=True)
    with torch.no_grad():
        y = enc_mv(torch.randn(2, 2048, 3))
    assert y.shape == (2, 3, K, D)


def test_depth_override_env():
    enc = make()
    x = torch.randn(2, 8192)
    with torch.no_grad():
        full = enc(x)
        os.environ["TSLM_V3_DEPTH"] = "1"
        try:
            shallow = enc(x)
            assert enc.last_depth == 1
        finally:
            del os.environ["TSLM_V3_DEPTH"]
        again = enc(x)
    assert enc.last_depth == enc.last_steps
    assert not torch.allclose(full, shallow)
    assert torch.allclose(full, again)


def test_stochastic_depth_only_in_training():
    enc = make(depth_mode="sdepth", sdepth_min=1)
    x = torch.randn(1, 8192)
    enc.train()
    torch.manual_seed(1)
    depths = set()
    with torch.no_grad():
        for _ in range(12):
            enc(x)
            depths.add(enc.last_depth)
    assert len(depths) > 1, depths
    assert max(depths) <= enc.last_steps
    enc.eval()
    with torch.no_grad():
        enc(x)
    assert enc.last_depth == enc.last_steps


def test_noise_gate_skips_noise_levels_but_reads_anomalies():
    enc = make(depth_mode="noise_gate", noise_tau=1.0)
    torch.manual_seed(2)
    noise = torch.randn(4, 8192)
    spike = noise.clone()
    # one-sample spike + an unaligned 5-sample step: Haar sees both at every
    # level (a plateau aligned to Haar blocks would vanish at the fine levels)
    spike[:, 4001] += 60.0
    spike[:, 6003:6008] += 40.0
    with torch.no_grad():
        enc(noise)
        act_noise = enc.last_active
        enc(spike)
        act_spike = enc.last_active
    assert act_noise[0] == 1.0 and act_spike[0] == 1.0          # coarse always read
    assert sum(act_spike) > sum(act_noise), (act_noise, act_spike)
    assert min(act_noise[1:]) < 1.0, act_noise                  # some noise levels skipped
    # the finest levels are read for the spike (the anomaly is above the noise floor there)
    assert act_spike[-1] == 1.0 and act_spike[-2] == 1.0, act_spike


def test_full_depth_never_skips():
    enc = make(depth_mode="full")
    with torch.no_grad():
        enc(torch.randn(3, 8192))
    assert all(a == 1.0 for a in enc.last_active)
    assert len(enc.last_gates) == enc.last_steps
    assert all(0.0 < g < 1.0 for g in enc.last_gates)


def test_level_budget_bounds_tokens_read():
    enc = make(level_topm=64)
    seen = []
    orig = enc._tokens

    def spy(v, pos, band):
        seen.append(v.shape[-1])
        return orig(v, pos, band)

    enc._tokens = spy  # type: ignore[assignment]
    with torch.no_grad():
        enc(torch.randn(1, 65536))
    assert max(seen[1:]) <= 64, seen                         # every detail level <= level_topm
    assert seen[0] == 128, seen                              # coarse band read whole (128 blocks)


def test_tied_vs_untied_params():
    tied, untied = make(tied=True), make(tied=False)
    n_t = sum(p.numel() for p in tied.parameters())
    n_u = sum(p.numel() for p in untied.parameters())
    block = sum(p.numel() for p in tied.blocks[0].parameters())
    assert n_u - n_t == block * (tied.max_levels + 1 - 1), (n_t, n_u, block)
    assert n_t < 6e6, n_t                                     # small next to LoRA r16 on 4B


def test_gradients_flow_everywhere():
    enc = make().train()
    x = torch.randn(2, 4096)
    enc(x).float().pow(2).mean().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        if "gate_head" not in name and "step_embed" not in name and "band_embed" not in name:
            assert p.grad.abs().sum() > 0, name


def test_bf16_forward_cpu():
    enc = make().to(torch.bfloat16)
    with torch.no_grad():
        y = enc(torch.randn(2, 4096).to(torch.bfloat16))
    assert y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()


def test_deterministic_eval():
    enc = make()
    x = torch.randn(2, 4096)
    with torch.no_grad():
        assert torch.equal(enc(x), enc(x))


def test_config_kwargs_like_yaml():
    cfg = dict(type="wavelet_v3", output_dim=512, num_tokens=256, coarse_blocks=128,
               n_pos_freqs=16, freeze=False, level_topm=1024, n_heads=8, mlp_ratio=4,
               dropout=0.0, level_gate=True, noise_tau=1.0, pos_dtype="input",
               max_levels=10, tied=True, depth_mode="full")
    cfg.pop("type")
    enc = RecurrentWaveletTSEncoder(**cfg)
    assert enc.tied and enc.depth_mode == "full" and enc.level_topm == 1024


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
