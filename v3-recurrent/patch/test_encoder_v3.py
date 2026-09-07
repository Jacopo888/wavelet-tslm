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
from src.models.wavelet_encoder import WaveletTSEncoder, haar_pyramid  # noqa: E402
from src.models.wavelet_encoder_v3 import (  # noqa: E402
    RecurrentWaveletTSEncoder,
    position_decodability,
    token_diversity,
)

D, K = 512, 256


def make(**kw) -> RecurrentWaveletTSEncoder:
    torch.manual_seed(0)
    return RecurrentWaveletTSEncoder(output_dim=D, num_tokens=K, **kw).eval()


def expected_steps(L: int, coarse_blocks: int = 128, max_levels: int = 10) -> int:
    return 1 + max(1, min(max_levels, math.ceil(math.log2(max(2, L / coarse_blocks)))))


def make_small(**kw) -> RecurrentWaveletTSEncoder:
    torch.manual_seed(0)
    defaults = dict(
        output_dim=32,
        num_tokens=16,
        max_levels=3,
        coarse_blocks=8,
        level_topm=16,
        n_heads=4,
        mlp_ratio=2,
    )
    defaults.update(kw)
    return RecurrentWaveletTSEncoder(**defaults).eval()


def make_diversity_case():
    """Deterministic 256-slot case: small D/L keeps the CPU gate quick."""
    torch.manual_seed(11)
    enc = RecurrentWaveletTSEncoder(
        output_dim=64,
        num_tokens=256,
        max_levels=2,
        coarse_blocks=32,
        level_topm=64,
        n_pos_freqs=16,
        n_heads=4,
        mlp_ratio=2,
    )
    torch.manual_seed(12)
    base = torch.randn(1, 512)
    different = torch.sin(torch.linspace(0, 80, 512)).unsqueeze(0)
    different[:, 311:327] += 20.0
    early = base.clone()
    late = base.clone()
    early[:, 96:120] += 30.0
    late[:, 392:416] += 30.0
    return enc, base, different, early, late


def assert_diverse(metrics: dict[str, float]) -> None:
    assert metrics["mean_token_cosine"] < 0.5, metrics
    assert metrics["between_total_std_ratio"] >= 0.3, metrics


def test_token_diversity_and_input_dependence_at_init():
    enc, base, different, early, late = make_diversity_case()
    different_metrics = token_diversity(enc, base, different)
    location_metrics = token_diversity(enc, early, late)
    default_metrics = token_diversity(enc, early)

    assert_diverse(different_metrics)
    assert different_metrics["input_relative_difference"] > 0.1, different_metrics
    assert location_metrics["input_relative_difference"] > 0.1, location_metrics
    assert default_metrics["input_relative_difference"] > 0.1, default_metrics
    assert abs(enc.latents.float().std().item() - 1.0) < 0.05
    assert enc.step_embed.weight.float().std().item() < 0.03

    # The native-scale local read must not depend on the randomly initialised
    # attention out_proj having a favourable gain.
    with torch.no_grad():
        for block in enc.blocks:
            block.cross.out_proj.weight.zero_()
            block.cross.out_proj.bias.zero_()
    bypass_metrics = token_diversity(enc, early, late)
    assert bypass_metrics["input_relative_difference"] > 0.1, bypass_metrics


def test_position_presence_and_magnitude_decodability_at_init():
    # D=256 is large enough for the same random-feature regime as production
    # while keeping this deterministic CPU regression comfortably lightweight.
    torch.manual_seed(0)
    v1 = WaveletTSEncoder(
        output_dim=256, num_tokens=256, max_levels=10,
        coarse_blocks=128, n_pos_freqs=16,
    ).eval()
    torch.manual_seed(0)
    v3 = RecurrentWaveletTSEncoder(
        output_dim=256, num_tokens=256, max_levels=10,
        coarse_blocks=128, n_pos_freqs=16, level_topm=256,
        n_heads=8, mlp_ratio=2,
    ).eval()
    v1_probe = position_decodability(v1, n=288)
    v3_probe = position_decodability(v3, n=288)
    # This is deliberately a little stricter than the requested 1.5x gate:
    # cfd5256 scores ~1.45x here, while the content-highway design is ~1.36x.
    assert v3_probe["position_mae"] <= 1.4 * v1_probe["position_mae"], (
        v1_probe, v3_probe
    )
    assert v3_probe["presence_accuracy"] >= 0.60, v3_probe
    assert v3_probe["magnitude_mae"] <= 0.20, v3_probe


def test_diversity_and_latent_gradient_survive_synthetic_optimization():
    enc, base, different, early, late = make_diversity_case()
    enc.train()
    x = torch.cat([early, late], dim=0)
    positions = (torch.arange(256, dtype=torch.float32) + 0.5) / 256
    centers = torch.tensor([0.2, 0.8])
    target = torch.exp(
        -((positions.unsqueeze(0) - centers.unsqueeze(1)) / 0.05).square()
    )
    probe = torch.nn.Linear(64, 1, bias=False)
    optimizer = torch.optim.Adam(
        list(enc.parameters()) + list(probe.parameters()), lr=1e-3
    )
    gradient_ratios = []
    for _ in range(4):
        optimizer.zero_grad()
        logits = probe(enc(x)).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        value_grads = torch.cat([
            parameter.grad.float().reshape(-1)
            for parameter in enc.value_proj.parameters()
            if parameter.grad is not None
        ])
        latent_grad = enc.latents.grad
        assert latent_grad is not None
        gradient_ratios.append(
            (latent_grad.float().norm() / value_grads.norm()).item()
        )
        optimizer.step()

    # The explicit value highway intentionally strengthens value_proj's
    # denominator; latents must still receive a material, stable gradient.
    assert min(gradient_ratios) > 5e-3, gradient_ratios
    different_metrics = token_diversity(enc, base, different)
    location_metrics = token_diversity(enc, early, late)
    assert_diverse(different_metrics)
    assert different_metrics["input_relative_difference"] > 0.1, different_metrics
    assert location_metrics["input_relative_difference"] > 0.1, location_metrics

    torch.manual_seed(11)
    v1 = WaveletTSEncoder(
        output_dim=64, num_tokens=256, max_levels=2,
        coarse_blocks=32, n_pos_freqs=16,
    ).eval()
    v1_probe = position_decodability(v1, n=72)
    trained_probe = position_decodability(enc, n=72)
    assert trained_probe["position_mae"] <= 1.5 * v1_probe["position_mae"], (
        v1_probe, trained_probe
    )


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

    enc_mv = make_small(multi_variable=True, depth_mode="sdepth").train()
    calls = 0
    requested_depth = enc_mv._requested_depth

    def counted(n_steps):
        nonlocal calls
        calls += 1
        return requested_depth(n_steps)

    enc_mv._requested_depth = counted  # type: ignore[method-assign]
    with torch.no_grad():
        enc_mv(torch.randn(1, 64, 3))
    assert calls == 1


def test_depth_override_env():
    enc = make()
    x = torch.randn(2, 8192)
    with torch.no_grad():
        full = enc(x)
        full_steps = enc.last_steps
        os.environ["TSLM_V3_DEPTH"] = "1"
        try:
            shallow = enc(x)
            assert enc.last_depth == 1
        finally:
            del os.environ["TSLM_V3_DEPTH"]
        os.environ["TSLM_V3_DEPTH"] = str(full_steps)
        try:
            explicit_full = enc(x)
        finally:
            del os.environ["TSLM_V3_DEPTH"]
        again = enc(x)
    assert enc.last_depth == enc.last_steps
    assert not torch.allclose(full, shallow)
    assert torch.equal(full, explicit_full)
    assert torch.equal(full, again)


def test_invalid_depth_override_fails_fast():
    enc = make_small()
    for bad in ("0", "-1", "abc"):
        os.environ["TSLM_V3_DEPTH"] = bad
        try:
            try:
                enc(torch.randn(1, 64))
            except ValueError as exc:
                assert "TSLM_V3_DEPTH" in str(exc)
            else:
                raise AssertionError(f"invalid depth accepted: {bad}")
        finally:
            del os.environ["TSLM_V3_DEPTH"]


def test_stochastic_depth_only_in_training():
    enc = make(depth_mode="sdepth", sdepth_min=1)
    x = torch.randn(1, 8192)
    enc.train()

    def sequence(seed):
        torch.manual_seed(seed)
        out = []
        with torch.no_grad():
            for _ in range(12):
                enc(x)
                out.append(enc.last_depth)
        return out

    depths_a = sequence(1)
    depths_b = sequence(1)
    assert depths_a == depths_b
    assert len(set(depths_a)) > 1, depths_a
    assert max(depths_a) <= enc.last_steps
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
    expected_gate = torch.sigmoid(torch.tensor(2.0)).item()
    assert all(abs(g - expected_gate) < 1e-6 for g in enc.last_gates)


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


def test_constant_and_short_series_are_finite():
    enc = make_small(depth_mode="noise_gate").to(torch.bfloat16)
    x = torch.full((2, 64), 7.0, dtype=torch.bfloat16)
    with torch.no_grad():
        y = enc(x)
    assert y.shape == (2, 16, 32)
    assert torch.isfinite(y.float()).all()
    assert enc.last_steps == 1 + 3
    assert enc.last_active[0] == 1.0
    assert all(a == 0.0 for a in enc.last_active[1:]), enc.last_active


def test_nonfinite_input_fails_fast():
    enc = make_small()
    x = torch.zeros(1, 64)
    x[0, 3] = float("nan")
    try:
        enc(x)
    except ValueError as exc:
        assert "NaN/inf" in str(exc)
    else:
        raise AssertionError("non-finite input was accepted")


def test_deterministic_eval():
    enc = make()
    x = torch.randn(2, 4096)
    with torch.no_grad():
        assert torch.equal(enc(x), enc(x))


def test_v1_level_featurization_parity_input_dtype():
    # L=64 and one Haar level expose all 32 coarse + 32 detail values in v1,
    # so the two contiguous slices can be compared with v3._tokens directly.
    torch.manual_seed(4)
    v1 = WaveletTSEncoder(
        output_dim=32, num_tokens=64, max_levels=1, coarse_blocks=128,
        n_pos_freqs=4,
    ).to(torch.bfloat16).eval()
    v3 = RecurrentWaveletTSEncoder(
        output_dim=32, num_tokens=64, max_levels=1, coarse_blocks=128,
        n_pos_freqs=4, n_heads=4, mlp_ratio=2, pos_dtype="input",
    ).to(torch.bfloat16).eval()
    v3.value_proj.load_state_dict(v1.value_proj.state_dict())
    v3.band_embed.load_state_dict(v1.band_embed.state_dict())

    x = torch.randn(2, 64, dtype=torch.bfloat16)
    med = x.median(dim=-1, keepdim=True).values
    mad = (x - med).abs().median(dim=-1, keepdim=True).values
    xn = (x - med) / (1.4826 * mad + 1e-8)
    details, coarse = haar_pyramid(xn, 1)
    cpos = ((torch.arange(32, dtype=x.dtype) * 2 + 1) / 64).expand(2, 32)
    dpos = cpos
    with torch.no_grad():
        v1_tokens = v1(x)
        coarse_v3 = v3._tokens(coarse, cpos, 0)
        detail_v3 = v3._tokens(details[0], dpos, 1)
    assert torch.equal(v1_tokens[:, :32], coarse_v3)
    assert torch.equal(v1_tokens[:, 32:64], detail_v3)


def test_state_dict_roundtrip_and_slim_key_coverage():
    enc = make_small(level_gate=True)
    state = enc.state_dict()
    parameter_names = {name for name, _ in enc.named_parameters()}
    assert set(state) == parameter_names
    assert "pos_freqs" not in state
    required = {
        "latents", "step_embed.weight", "gate_head.weight", "gate_head.bias",
        "ln_out.weight", "ln_out.bias", "content_scale", "context_scale",
    }
    assert required <= set(state)

    restored = make_small(level_gate=True)
    missing, unexpected = restored.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    x = torch.randn(2, 64)
    with torch.no_grad():
        assert torch.equal(enc(x), restored(x))

    # Mirror train.py's trainable-only filter on a minimal parent model.  The
    # encoder keys must all retain their ts_encoder.* prefix, while frozen
    # tensors are the only keys reported missing by strict=False at eval.
    parent = torch.nn.Module()
    parent.ts_encoder = enc
    parent.frozen = torch.nn.Linear(3, 3)
    parent.frozen.requires_grad_(False)
    trainable = {name for name, p in parent.named_parameters() if p.requires_grad}
    slim = {key: value for key, value in parent.state_dict().items() if key in trainable}
    assert all(key.startswith("ts_encoder.") for key in slim)
    assert {f"ts_encoder.{name}" for name in parameter_names} == set(slim)
    assert not any("pos_freqs" in key for key in slim)

    loaded = torch.nn.Module()
    loaded.ts_encoder = make_small(level_gate=True)
    loaded.frozen = torch.nn.Linear(3, 3)
    missing, unexpected = loaded.load_state_dict(slim, strict=False)
    assert not unexpected
    assert set(missing) == {"frozen.weight", "frozen.bias"}


def test_loop_activation_checkpoint_matches_plain_backward():
    plain = make_small(dropout=0.0).train()
    checked = make_small(dropout=0.0).train()
    checked.load_state_dict(plain.state_dict())
    x = torch.randn(2, 64)
    target = torch.randn(2, 16, 32)

    os.environ.pop("TSLM_GRAD_CKPT", None)
    (plain(x) * target).sum().backward()
    os.environ["TSLM_GRAD_CKPT"] = "1"
    try:
        (checked(x) * target).sum().backward()
    finally:
        del os.environ["TSLM_GRAD_CKPT"]

    for (name_a, p_a), (name_b, p_b) in zip(
        plain.named_parameters(), checked.named_parameters(), strict=True
    ):
        assert name_a == name_b
        assert p_a.grad is not None and p_b.grad is not None
        assert torch.allclose(p_a.grad, p_b.grad, rtol=1e-4, atol=1e-5), name_a


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
