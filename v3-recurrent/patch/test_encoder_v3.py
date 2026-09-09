"""CPU tests for RecurrentWaveletTSEncoder (v3). No GPU, no Qwen, no data.

Run:  python v3-recurrent/patch/test_encoder_v3.py      (or pytest -q on it)
On the node the same file runs from ~/patch after the overlay.

The v1 file (kaggle/patch/wavelet_encoder.py) is imported through a temporary
`src.models` package so the test works locally and on the node alike.
"""

from __future__ import annotations

import math
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import types
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
    context_input_dependence,
    position_decodability,
    token_diversity,
)

D, K = 512, 256
RUN_CONTEXT_SCALE_INIT = 0.1
RUN_CONTEXT_LR = 1.0e-3


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


def make_exact_pair(dtype=torch.float32, **kw):
    context_scale_init = kw.pop("context_scale_init", 0.0)
    level_topm = kw.pop("level_topm", 64)
    defaults = dict(
        output_dim=32,
        num_tokens=256,
        max_levels=10,
        coarse_blocks=128,
        n_pos_freqs=4,
    )
    defaults.update(kw)
    torch.manual_seed(7)
    v1 = WaveletTSEncoder(**defaults).to(dtype).eval()
    torch.manual_seed(7)
    v3 = RecurrentWaveletTSEncoder(
        **defaults, n_heads=4, mlp_ratio=2, level_topm=level_topm,
        context_scale_init=context_scale_init,
    ).to(dtype).eval()
    return v1, v3


def test_init_matches_v1_exact_fp32_bf16_short_and_16ch_long():
    for dtype in (torch.float32, torch.bfloat16):
        v1, v3 = make_exact_pair(dtype, context_scale_init=0.0)
        for shape in ((2, 7), (2, 4096), (16, 131072)):
            torch.manual_seed(sum(shape))
            x = torch.randn(*shape).to(dtype)
            with torch.no_grad():
                expected = v1(x)
                actual = v3(x)
            assert torch.equal(actual, expected), (
                dtype, shape, (actual.float() - expected.float()).abs().max().item()
            )
            assert v3.last_base_positions is not None
            assert v3.last_base_bands is not None
            assert v3.last_base_valid is not None
            assert v3.last_base_positions.shape == (shape[0], 256)
            assert v3.last_base_bands.shape == (shape[0], 256)
            assert v3.last_base_valid.shape == (shape[0], 256)
        assert (v3.last_base_bands[v3.last_base_valid] >= 0).all()

        # Exercise the actual residual arithmetic as well as the zero-scale
        # no-grad fast path used by the large equality cases.
        x = torch.randn(1, 64).to(dtype)
        expected = v1(x)
        actual = v3(x)
        assert torch.equal(actual, expected), dtype


def test_reader_modes_and_unit_scales_preserve_disabled_residual_identity():
    x = torch.randn(2, 96)
    for reader_mode in ("cross_only", "full"):
        for context_mode in ("full", "local", "global"):
            v1, v3 = make_exact_pair(
                context_scale_init=0.0,
                reader_mode=reader_mode,
                context_mode=context_mode,
                self_attn_scale_init=1.0,
                mlp_scale_init=1.0,
            )
            with torch.no_grad():
                assert torch.equal(v3(x), v1(x)), (reader_mode, context_mode)


def test_context_modes_are_an_exact_valid_token_decomposition_by_base_band():
    enc = make_small(reader_mode="cross_only")
    torch.manual_seed(20260909)
    z = torch.randn(2, 8, 32)
    bands = torch.tensor([
        [0, 0, 1, 1, 2, 2, -1, -1],
        [0, 0, 1, 1, 3, 3, -1, -1],
    ])
    valid = bands >= 0
    historical_full = enc._local_context(z)
    enc.context_mode = "full"
    assert torch.equal(
        enc._context_for_mode(z, bands, valid), historical_full
    )
    enc.context_mode = "local"
    local = enc._context_for_mode(z, bands, valid)
    enc.context_mode = "global"
    global_context = enc._context_for_mode(z, bands, valid)
    assert torch.allclose(local + global_context, historical_full, atol=1e-7)
    for batch in range(z.shape[0]):
        for band in torch.unique(bands[batch, valid[batch]]):
            mask = valid[batch] & (bands[batch] == band)
            assert torch.allclose(
                local[batch, mask].mean(dim=0),
                torch.zeros(z.shape[-1]),
                atol=1e-7,
            )
            values = global_context[batch, mask]
            assert torch.equal(values, values[:1].expand_as(values))


def test_context_mode_witness_separates_position_from_global_presence_magnitude():
    enc = make_small(reader_mode="cross_only")
    bands = torch.tensor([[0] * 4 + [1] * 4] * 4)
    valid = torch.ones_like(bands, dtype=torch.bool)
    z = torch.zeros(4, 8, 2)
    magnitudes = torch.tensor([0.0, 1.5, 2.5, 4.0])
    positions = torch.tensor([0, 1, 2, 3])
    for sample, position in enumerate(positions):
        # A zero-sum detail pattern carries only within-band position.
        z[sample, 4:8, 0] = -0.25
        z[sample, 4 + position, 0] = 0.75
        # A band-shared channel carries only presence/magnitude.
        z[sample, 4:8, 1] = magnitudes[sample]
    local, global_context = enc._context_components(z, bands, valid)
    local_positions = local[:, 4:8, 0].argmax(dim=1)
    global_positions = global_context[:, 4:8, 0].argmax(dim=1)
    assert torch.equal(local_positions, positions)
    assert not torch.equal(global_positions, positions)
    assert torch.allclose(global_context[:, 4:8, 1].mean(dim=1), magnitudes)
    assert torch.equal(
        global_context[:, 4:8, 1].mean(dim=1) > 0,
        magnitudes > 0,
    )
    assert torch.allclose(local[:, 4:8, 1], torch.zeros(4, 4))


def test_context_snapshot_covers_all_256_slots_and_all_512_features():
    enc = make().eval()
    enc.last_base_bands = torch.tensor([[0] * 128 + [1] * 128] * 2)
    enc.last_base_valid = torch.ones(2, 256, dtype=torch.bool)
    z = torch.zeros(2, 256, 512)
    z[1, 0, 0] = 8.0
    z[0, 255, 511] = 32.0
    z[1, 255, 511] = -32.0
    stats = enc._context_snapshot(z)
    expected = z.square().mean().sqrt().item()
    assert math.isclose(stats["z_rms_total"], expected, rel_tol=1e-6)
    assert stats["z_rms_coarse"] > 0.0
    assert stats["z_rms_detail"] > 0.0
    for group in ("total", "coarse", "detail"):
        assert {
            f"token_variable_fraction_{group}",
            f"mean_token_cosine_{group}",
            f"input_series_relative_{group}",
        } <= set(stats)
        assert stats[f"input_series_relative_{group}"] > 0.0

    assert set(stats["by_base_band"]) == {"0", "1"}
    for band_stats in stats["by_base_band"].values():
        assert band_stats["n_valid_tokens"] == 256
        assert {
            "token_variable_fraction", "mean_token_cosine",
            "group_mean_rms_over_h_rms", "input_series_relative",
        } <= set(band_stats)
        assert band_stats["group_mean_rms_over_h_rms"] >= 0.0


def test_context_eval_roll_is_valid_only_norm_preserving_and_base_invariant():
    enc = make_small(context_scale_init=0.1).eval()
    base = torch.randn(2, 16, 32)
    context = torch.randn_like(base)
    valid = torch.tensor([
        [True] * 11 + [False] * 5,
        [True] * 7 + [False] * 9,
    ])
    enc.last_base_valid = valid
    rolled = enc._roll_valid_context(context, valid)
    assert torch.equal(rolled[~valid], context[~valid])
    for row in range(context.shape[0]):
        before = context[row, valid[row]].norm(dim=-1).sort().values
        after = rolled[row, valid[row]].norm(dim=-1).sort().values
        assert torch.equal(before, after)
        assert not torch.equal(rolled[row, valid[row]], context[row, valid[row]])

    original_base = base.clone()
    try:
        os.environ["TSLM_V3_CONTEXT_EVAL"] = "roll_tokens"
        output, gate = enc._merge_context(base, context)
    finally:
        os.environ.pop("TSLM_V3_CONTEXT_EVAL", None)
    assert torch.equal(base, original_base)
    assert torch.equal(gate, torch.ones_like(gate))
    assert torch.equal(output, base + enc.context_scale.detach() * rolled)


def test_context_eval_off_legacy_equivalence_and_zero_scale_conditions():
    enc = make_small(context_scale_init=0.1).eval()
    base = torch.randn(2, 16, 32)
    context = torch.randn_like(base)
    enc.last_base_valid = torch.ones(2, 16, dtype=torch.bool)
    try:
        os.environ["TSLM_V3_CONTEXT_EVAL"] = "off"
        new_off, _ = enc._merge_context(base, context)
        os.environ.pop("TSLM_V3_CONTEXT_EVAL")
        os.environ["TSLM_V3_CONTEXT_OFF"] = "1"
        legacy_off, _ = enc._merge_context(base, context)
        assert torch.equal(new_off, legacy_off)
        assert torch.equal(new_off, base)

        os.environ.pop("TSLM_V3_CONTEXT_OFF")
        with torch.no_grad():
            enc.context_scale.zero_()
        outputs = []
        for mode in ("on", "off", "roll_tokens"):
            os.environ["TSLM_V3_CONTEXT_EVAL"] = mode
            outputs.append(enc._merge_context(base, context)[0])
        assert all(torch.equal(output, base) for output in outputs)

        enc.train()
        os.environ["TSLM_V3_CONTEXT_EVAL"] = "roll_tokens"
        try:
            enc._merge_context(base, context)
        except RuntimeError as error:
            assert "evaluation-only" in str(error)
        else:
            raise AssertionError("training accepted a context intervention")
    finally:
        os.environ.pop("TSLM_V3_CONTEXT_OFF", None)
        os.environ.pop("TSLM_V3_CONTEXT_EVAL", None)


def test_one_pyramid_per_channel_is_shared_with_context():
    import src.models.wavelet_encoder as v1_module

    _, v3 = make_exact_pair(context_scale_init=0.1)
    original = v1_module.haar_pyramid
    calls = 0

    def counted(x, levels):
        nonlocal calls
        calls += 1
        return original(x, levels)

    v1_module.haar_pyramid = counted
    try:
        with torch.no_grad():
            v3(torch.randn(2, 256))
    finally:
        v1_module.haar_pyramid = original
    assert calls == 1, calls


def test_token_diversity_and_input_dependence_at_init():
    enc, base, different, early, late = make_diversity_case()
    different_metrics = token_diversity(enc, base, different)
    location_metrics = token_diversity(enc, early, late)
    default_metrics = token_diversity(enc, early)

    assert_diverse(different_metrics)
    assert different_metrics["input_relative_difference"] > 0.1, different_metrics
    assert location_metrics["input_relative_difference"] > 0.1, location_metrics
    assert default_metrics["input_relative_difference"] > 0.1, default_metrics
    assert enc.step_embed.weight.float().std().item() < 0.03


def _reader_context_stats(reader_mode: str, scale: float) -> dict[str, object]:
    torch.manual_seed(314)
    enc = RecurrentWaveletTSEncoder(
        output_dim=64,
        num_tokens=256,
        max_levels=2,
        coarse_blocks=32,
        level_topm=64,
        n_pos_freqs=16,
        n_heads=4,
        mlp_ratio=2,
        reader_mode=reader_mode,
        self_attn_scale_init=scale,
        mlp_scale_init=scale,
        context_scale_init=0.1,
    ).eval()
    torch.manual_seed(315)
    base = torch.randn(1, 512)
    different = torch.sin(torch.linspace(0, 90, 512)).unsqueeze(0)
    translated = torch.roll(base, 173, dims=-1)
    batch = torch.cat([base, different, translated], dim=0)
    with torch.no_grad():
        enc(batch)
    dependence = context_input_dependence(enc, base, different)
    stats = enc.last_context_stats
    assert len(stats["iterations"]) == enc.last_depth
    for row in stats["iterations"]:
        assert set(("z_rms", "h_rms", "token_variable_fraction",
                    "mean_token_cosine", "cross_rms", "self_attn_rms",
                    "mlp_rms", "different_series_relative",
                    "translated_input_relative")) <= set(row)
    stats["measured_input_dependence"] = dependence
    return stats


def test_context_only_diagnostics_preserve_structure_at_init():
    cross = _reader_context_stats("cross_only", 1.0)
    full = _reader_context_stats("full", 1.0)
    small = _reader_context_stats("full", 0.1)
    cross_final = cross["final"]
    full_final = full["final"]
    small_final = small["final"]

    assert cross_final["token_variable_fraction"] >= 0.3, cross
    assert cross["measured_input_dependence"]["different_series_relative"] > 0.1
    assert cross["measured_input_dependence"]["translated_input_relative"] > 0.1
    assert all(row["self_attn_rms"] == 0.0 for row in cross["iterations"])
    assert all(row["mlp_rms"] == 0.0 for row in cross["iterations"])
    assert any(row["self_attn_rms"] > 0.0 for row in full["iterations"])
    assert any(row["mlp_rms"] > 0.0 for row in full["iterations"])
    assert abs(
        small_final["token_variable_fraction"]
        - cross_final["token_variable_fraction"]
    ) < 0.25
    assert abs(
        small_final["token_variable_fraction"]
        - cross_final["token_variable_fraction"]
    ) <= abs(
        full_final["token_variable_fraction"]
        - cross_final["token_variable_fraction"]
    ), (cross, full, small)
    print(
        "CONTEXT_INIT_STATS",
        {
            "cross_only": cross_final,
            "full": full_final,
            "full_small": small_final,
        },
    )


def test_token_gate_identity_and_exact_zero_slot():
    enc = make_small(token_gate=True, context_scale_init=0.1)
    assert enc.context_token_gate is not None
    base = torch.randn(2, 16, 32)
    context = torch.randn_like(base)
    ungated = base + enc.context_scale.detach() * context
    actual, gates = enc._merge_context(base, context)
    assert torch.equal(gates, torch.ones_like(gates))
    assert torch.equal(actual, ungated)

    override = torch.ones(2, 16, 1)
    override[:, 7] = 0.0
    slotted, used = enc._merge_context(base, context, override)
    assert torch.equal(used, override)
    assert torch.equal(slotted[:, 7], base[:, 7])
    assert torch.equal(slotted[:, :7], ungated[:, :7])

    with torch.no_grad():
        enc(torch.randn(2, 64))
    gate_stats = enc.last_context_stats["final"]
    assert gate_stats["gate_min"] == gate_stats["gate_max"] == 1.0
    assert gate_stats["gate_saturation_low"] == 0.0
    assert gate_stats["gate_saturation_high"] == 0.0


def test_position_presence_and_magnitude_decodability_at_init():
    v1, v3 = make_exact_pair(
        output_dim=64,
        n_pos_freqs=16,
        level_topm=256,
        context_scale_init=RUN_CONTEXT_SCALE_INIT,
    )
    v1_probe = position_decodability(v1, n=72)
    v3_probe = position_decodability(v3, n=72)
    assert v3_probe["position_mae"] <= 1.1 * v1_probe["position_mae"], (
        v1_probe, v3_probe
    )
    assert v3_probe["magnitude_mae"] <= 1.1 * v1_probe["magnitude_mae"], (
        v1_probe, v3_probe
    )
    assert v3_probe["presence_accuracy"] >= v1_probe["presence_accuracy"] / 1.1, (
        v1_probe, v3_probe
    )


def _context_parameters(enc: RecurrentWaveletTSEncoder) -> list[torch.nn.Parameter]:
    params = [
        *enc.blocks.parameters(),
        *enc.step_embed.parameters(),
        *enc.context_norm.parameters(),
        enc.context_scale,
    ]
    if enc.gate_head is not None:
        params.extend(enc.gate_head.parameters())
    return params


def _gradient_norm(parameters) -> float:
    terms = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return torch.stack(terms).sum().sqrt().item() if terms else 0.0


def _run_context_deadlock_witness(context_scale_init: float) -> dict[str, float]:
    """Train the context group on a location absent from the v1 tokens.

    Eight coarse coefficients consume v1's full token budget. The paired
    impulses cancel from that band, so their location is available only to the
    per-level reader. A one-coordinate initial context makes the desired
    second-coordinate correction exactly orthogonal. At scale zero both
    dL/dscale and dL/dblocks are therefore exactly zero; this deterministic
    witness avoids relying on random finite-sample cancellation.
    """
    torch.manual_seed(123)
    enc = RecurrentWaveletTSEncoder(
        output_dim=8,
        num_tokens=8,
        max_levels=3,
        coarse_blocks=8,
        level_topm=8,
        n_pos_freqs=2,
        n_heads=2,
        mlp_ratio=1,
        dropout=0.0,
        context_scale_init=context_scale_init,
    ).train()
    with torch.no_grad():
        enc.context_norm.weight.zero_()
        enc.context_norm.weight[0] = 1.0
        enc.context_norm.bias.zero_()

    x = torch.zeros(2, 64)
    x[0, 10], x[0, 11] = 5.0, -5.0
    x[1, 50], x[1, 51] = 5.0, -5.0
    centers = torch.tensor([10.5, 50.5]) / 64
    positions = (torch.arange(8, dtype=torch.float32) + 0.5) / 8
    location = torch.exp(
        -((positions.unsqueeze(0) - centers.unsqueeze(1)) / 0.12).square()
    )

    with torch.no_grad():
        initial_scale = enc.context_scale.item()
        enc.context_scale.zero_()
        base = enc(x).detach()
        assert torch.equal(base[0], base[1])
        assert bool((enc.last_base_bands == 0).all())  # v1 has no detail slots
        enc.context_scale.fill_(initial_scale)
    target = base.clone()
    target[:, :, 1] += location

    optimizer = torch.optim.AdamW(
        _context_parameters(enc),
        lr=RUN_CONTEXT_LR,
        weight_decay=0.01,
    )
    first_block_norm = first_value_norm = 0.0
    max_block_norm = max_scale_grad = 0.0
    for step in range(16):
        optimizer.zero_grad(set_to_none=True)
        enc.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(enc(x), target)
        loss.backward()
        block_norm = _gradient_norm(enc.blocks.parameters())
        value_norm = _gradient_norm(enc.value_proj.parameters())
        scale_grad = abs(enc.context_scale.grad.detach().item())
        if step == 0:
            first_block_norm, first_value_norm = block_norm, value_norm
        max_block_norm = max(max_block_norm, block_norm)
        max_scale_grad = max(max_scale_grad, scale_grad)
        optimizer.step()

    return {
        "scale_move": abs(enc.context_scale.detach().item() - initial_scale),
        "first_block_relative_to_value": first_block_norm / max(first_value_norm, 1e-30),
        "max_block_norm": max_block_norm,
        "max_scale_grad": max_scale_grad,
    }


def test_nonzero_run_init_breaks_zero_scale_context_deadlock():
    deadlocked = _run_context_deadlock_witness(0.0)
    active = _run_context_deadlock_witness(RUN_CONTEXT_SCALE_INIT)

    assert deadlocked["scale_move"] == 0.0, deadlocked
    assert deadlocked["max_block_norm"] == 0.0, deadlocked
    assert deadlocked["max_scale_grad"] == 0.0, deadlocked
    assert active["first_block_relative_to_value"] > 1.0e-3, active
    assert active["max_block_norm"] > 0.0, active
    assert active["scale_move"] > 1.0e-2, active


def test_cross_only_heldout_witness_adds_missing_information_and_is_optional():
    """Intermediate-detail location is absent from v1's eight coarse slots."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(901)
        enc = RecurrentWaveletTSEncoder(
            output_dim=8, num_tokens=8, max_levels=3, coarse_blocks=8,
            level_topm=8, n_pos_freqs=2, n_heads=2, mlp_ratio=1,
            reader_mode="cross_only", context_scale_init=1.0,
            level_gate=False,
        ).train()
        base_encoder = WaveletTSEncoder(
            output_dim=8, num_tokens=8, max_levels=3, coarse_blocks=8,
            n_pos_freqs=2,
        ).eval()
        base_encoder.load_state_dict({
            key: value
            for key, value in enc.state_dict().items()
            if key in base_encoder.state_dict()
        })

        def detail_cases(positions):
            rows, labels = [], []
            for position in positions:
                row = torch.zeros(64)
                row[position], row[position + 1] = 5.0, -5.0
                rows.append(row)
                labels.append(float(position >= 32))
            return torch.stack(rows), torch.tensor(labels)

        train_x, train_y = detail_cases((4, 8, 16, 24, 36, 40, 48, 56))
        test_x, test_y = detail_cases((12, 20, 44, 52))
        with torch.no_grad():
            train_base = base_encoder(train_x).flatten(1)
            test_base = base_encoder(test_x).flatten(1)
        # The complete v1 budget is coarse; paired impulses cancel there.
        assert bool((enc._base_metadata(
            *enc._v1_tokens_and_pyramid(train_x)[1:], 64, train_x.dtype
        )[1] == 0).all())
        assert (test_base - test_base[:1]).abs().max().item() < 1.0e-6

        torch.manual_seed(902)
        base_probe = torch.nn.Linear(64, 1)
        torch.manual_seed(902)
        context_probe = torch.nn.Linear(64, 1)
        base_opt = torch.optim.AdamW(base_probe.parameters(), lr=0.01)
        context_opt = torch.optim.AdamW(
            [*_context_parameters(enc), *context_probe.parameters()], lr=0.01
        )
        for _ in range(32):
            base_opt.zero_grad(set_to_none=True)
            base_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                base_probe(train_base).squeeze(-1), train_y
            )
            base_loss.backward()
            base_opt.step()

            context_opt.zero_grad(set_to_none=True)
            context_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                context_probe(enc(train_x).flatten(1)).squeeze(-1), train_y
            )
            context_loss.backward()
            context_opt.step()

        enc.eval()
        with torch.no_grad():
            base_logits = base_probe(test_base).squeeze(-1)
            on_logits = context_probe(enc(test_x).flatten(1)).squeeze(-1)
            os.environ["TSLM_V3_CONTEXT_OFF"] = "1"
            off_logits = context_probe(enc(test_x).flatten(1)).squeeze(-1)
            os.environ.pop("TSLM_V3_CONTEXT_OFF")
        base_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            base_logits, test_y
        ).item()
        on_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            on_logits, test_y
        ).item()
        off_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            off_logits, test_y
        ).item()
        on_accuracy = ((on_logits > 0) == test_y.bool()).float().mean().item()
        assert on_accuracy == 1.0 and on_bce < 0.25 * base_bce
        assert on_bce < 0.25 * off_bce

        # Negative control: a low-frequency orientation is already present in
        # v1.  A fresh identity-scale context is not needed; both conditions
        # must solve the held-out rows.
        torch.manual_seed(903)
        optional = RecurrentWaveletTSEncoder(
            output_dim=8, num_tokens=8, max_levels=3, coarse_blocks=8,
            level_topm=8, n_pos_freqs=2, n_heads=2, mlp_ratio=1,
            reader_mode="cross_only", context_scale_init=0.1,
            level_gate=False,
        ).eval()

        def coarse_cases(n, seed):
            generator = torch.Generator().manual_seed(seed)
            axis = torch.linspace(-1.0, 1.0, 64)
            rows, labels = [], []
            for index in range(n):
                label = index % 2
                row = (1.0 if label else -1.0) * axis
                row = row + 0.05 * torch.randn(64, generator=generator)
                rows.append(row)
                labels.append(float(label))
            return torch.stack(rows), torch.tensor(labels)

        coarse_train_x, coarse_train_y = coarse_cases(16, 1)
        coarse_test_x, coarse_test_y = coarse_cases(8, 2)
        torch.manual_seed(904)
        optional_probe = torch.nn.Linear(64, 1)
        optional_opt = torch.optim.AdamW(optional_probe.parameters(), lr=0.05)
        with torch.no_grad():
            optional_train = optional(coarse_train_x).flatten(1)
        for _ in range(30):
            optional_opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                optional_probe(optional_train).squeeze(-1), coarse_train_y
            )
            loss.backward()
            optional_opt.step()
        with torch.no_grad():
            optional_on = optional_probe(optional(coarse_test_x).flatten(1)).squeeze(-1)
            os.environ["TSLM_V3_CONTEXT_OFF"] = "1"
            optional_off = optional_probe(optional(coarse_test_x).flatten(1)).squeeze(-1)
            os.environ.pop("TSLM_V3_CONTEXT_OFF")
        on_optional_accuracy = (
            (optional_on > 0) == coarse_test_y.bool()
        ).float().mean().item()
        off_optional_accuracy = (
            (optional_off > 0) == coarse_test_y.bool()
        ).float().mean().item()
        assert on_optional_accuracy == off_optional_accuracy == 1.0
        print("HELDOUT_WITNESS", {
            "needed": {"base_bce": base_bce, "off_bce": off_bce,
                       "on_bce": on_bce, "on_accuracy": on_accuracy},
            "not_needed": {"on_accuracy": on_optional_accuracy,
                           "off_accuracy": off_optional_accuracy},
        })
    finally:
        os.environ.pop("TSLM_V3_CONTEXT_OFF", None)
        torch.set_num_threads(previous_threads)


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
    # At the mandated zero initialization every depth is exactly the v1 base.
    assert torch.equal(full, shallow)
    assert torch.equal(full, explicit_full)
    assert torch.equal(full, again)

    with torch.no_grad():
        enc.context_scale.fill_(0.1)
        full_context = enc(x)
        seen_bands = []
        original_tokens = enc._tokens

        def record_band(values, positions, band):
            seen_bands.append(band)
            return original_tokens(values, positions, band)

        enc._tokens = record_band  # type: ignore[method-assign]
        os.environ["TSLM_V3_DEPTH"] = "1"
        try:
            coarse_context = enc(x)
            assert enc.last_depth == 1
            assert len(enc.last_active) == 1
        finally:
            del os.environ["TSLM_V3_DEPTH"]
            enc._tokens = original_tokens  # type: ignore[method-assign]
    assert seen_bands == [0], seen_bands
    assert not torch.allclose(full_context, coarse_context)

    v1 = WaveletTSEncoder(
        output_dim=D, num_tokens=K, max_levels=enc.max_levels,
        coarse_blocks=enc.coarse_blocks, n_pos_freqs=enc.n_pos_freqs,
    ).eval()
    v1.load_state_dict({
        key: value for key, value in enc.state_dict().items()
        if key in v1.state_dict()
    }, strict=True)
    with torch.no_grad():
        base = v1(x)
    assert not torch.equal(coarse_context, base)


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
    with torch.no_grad():
        enc.context_scale.fill_(0.1)
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
        enc.context_scale.fill_(0.1)
        enc(torch.randn(3, 8192))
    assert all(a == 1.0 for a in enc.last_active)
    assert len(enc.last_gates) == enc.last_steps
    expected_gate = torch.sigmoid(torch.tensor(2.0)).item()
    assert all(abs(g - expected_gate) < 1e-6 for g in enc.last_gates)


def test_level_budget_bounds_tokens_read():
    enc = make(level_topm=64)
    with torch.no_grad():
        enc.context_scale.fill_(0.1)
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
    with torch.no_grad():
        enc.context_scale.fill_(0.1)
    x = torch.randn(2, 4096)
    enc(x).float().pow(2).mean().backward()
    for name, p in enc.named_parameters():
        if name == "pad_token":
            assert p.grad is None  # no padding at this input length
            continue
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        if (
            name != "pad_token"
            and "gate_head" not in name
            and "step_embed" not in name
            and "band_embed" not in name
        ):
            assert p.grad.abs().sum() > 0, name


def test_bf16_forward_cpu():
    enc = make().to(torch.bfloat16)
    with torch.no_grad():
        y = enc(torch.randn(2, 4096).to(torch.bfloat16))
    assert y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()


def test_constant_and_short_series_are_finite():
    enc = make_small(depth_mode="noise_gate").to(torch.bfloat16)
    with torch.no_grad():
        enc.context_scale.fill_(0.1)
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
        "pad_token", "value_proj.0.weight", "band_embed.weight",
        "step_embed.weight", "gate_head.weight", "gate_head.bias",
        "context_norm.weight", "context_norm.bias", "context_scale",
        "blocks.0.self_attn_scale", "blocks.0.mlp_scale",
    }
    assert required <= set(state)

    restored = make_small(level_gate=True)
    missing, unexpected = restored.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    x = torch.randn(2, 64)
    with torch.no_grad():
        assert torch.equal(enc(x), restored(x))

    pre_scale_state = {
        key: value for key, value in state.items()
        if not key.endswith(("self_attn_scale", "mlp_scale"))
    }
    migrated = make_small(level_gate=True, self_attn_scale_init=0.1,
                          mlp_scale_init=0.1)
    missing, unexpected = migrated.load_state_dict(pre_scale_state, strict=True)
    assert not missing and not unexpected
    assert all(block.self_attn_scale.item() == 1.0 for block in migrated.blocks)
    assert all(block.mlp_scale.item() == 1.0 for block in migrated.blocks)

    gated_migration = make_small(level_gate=True, token_gate=True)
    missing, unexpected = gated_migration.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    assert gated_migration.context_token_gate.weight.count_nonzero().item() == 0
    assert gated_migration.context_token_gate.bias.count_nonzero().item() == 0

    bootstrap = make_small(self_attn_scale_init=0.1, mlp_scale_init=0.1)
    v1_bootstrap = WaveletTSEncoder(
        output_dim=32, num_tokens=16, max_levels=3, coarse_blocks=8,
    )
    missing, unexpected = bootstrap.load_state_dict(
        v1_bootstrap.state_dict(), strict=False
    )
    assert not unexpected
    assert all(math.isclose(block.self_attn_scale.item(), 0.1, rel_tol=1e-6)
               for block in bootstrap.blocks)
    assert all(math.isclose(block.mlp_scale.item(), 0.1, rel_tol=1e-6)
               for block in bootstrap.blocks)

    # The run init changes only the constructor value, never the state schema.
    # A residual-on-v1 checkpoint made with scale zero must strictly load and
    # overwrite the new 0.1 constructor value.
    run_init = make_small(level_gate=True, context_scale_init=RUN_CONTEXT_SCALE_INIT)
    assert set(run_init.state_dict()) == set(state)
    missing, unexpected = run_init.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    assert run_init.context_scale.item() == enc.context_scale.item()

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

    # cfd5256/0d2484e used learned latents and a different output boundary.
    legacy = dict(state)
    legacy["latents"] = torch.zeros(16, 32)
    try:
        make_small().load_state_dict(legacy, strict=False)
    except RuntimeError as exc:
        assert "legacy latent-only v3 checkpoint" in str(exc)
    else:
        raise AssertionError("legacy v3 checkpoint was accepted")


def test_model_splits_context_optimizer_group_without_v1_parameters():
    """Exercise wavelet_v3's inherited-group split without loading Qwen."""
    dummy_wavelet = types.ModuleType("src.models.wavelet")

    class DummyWaveletTSModel(torch.nn.Module):
        def get_trainable_parameters(self):
            return {
                "encoder": list(self.ts_encoder.parameters()),
                "projector": list(self.ts_projector.parameters()),
            }

    dummy_wavelet.WaveletTSModel = DummyWaveletTSModel
    previous = sys.modules.get("src.models.wavelet")
    sys.modules["src.models.wavelet"] = dummy_wavelet
    module_name = "src.models.wavelet_v3_group_test"
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, HERE / "wavelet_v3.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        model = module.WaveletV3TSModel()
        model.ts_encoder = make_small(level_gate=True)
        model.ts_projector = torch.nn.Linear(32, 7)
        groups = model.get_trainable_parameters()

        expected_scale = {id(model.ts_encoder.context_scale)}
        expected_context = {
            id(parameter) for parameter in _context_parameters(model.ts_encoder)
        } - expected_scale
        actual_context = {id(parameter) for parameter in groups["context"]}
        actual_scale = {id(parameter) for parameter in groups["context_scale"]}
        encoder_ids = {id(parameter) for parameter in groups["encoder"]}
        projector_ids = {id(parameter) for parameter in groups["projector"]}
        assert actual_context == expected_context
        assert actual_scale == expected_scale
        assert not actual_context & encoder_ids
        assert encoder_ids == {
            id(parameter) for parameter in model.ts_encoder.parameters()
        } - expected_context - expected_scale
        assert projector_ids == {id(parameter) for parameter in model.ts_projector.parameters()}
        flat_ids = [id(parameter) for params in groups.values() for parameter in params]
        assert len(flat_ids) == len(set(flat_ids))

        gated = module.WaveletV3TSModel()
        gated.ts_encoder = make_small(level_gate=True, token_gate=True)
        gated.ts_projector = torch.nn.Linear(32, 7)
        gated_groups = gated.get_trainable_parameters()
        expected_gate = {
            id(parameter) for parameter in gated.ts_encoder.context_token_gate.parameters()
        }
        assert {
            id(parameter) for parameter in gated_groups["context_token_gate"]
        } == expected_gate
        gated_flat = [
            id(parameter)
            for params in gated_groups.values()
            for parameter in params
        ]
        assert len(gated_flat) == len(set(gated_flat))
    finally:
        sys.modules.pop(module_name, None)
        if previous is None:
            del sys.modules["src.models.wavelet"]
        else:
            sys.modules["src.models.wavelet"] = previous


def test_loop_activation_checkpoint_matches_plain_backward():
    plain = make_small(dropout=0.0).train()
    checked = make_small(dropout=0.0).train()
    checked.load_state_dict(plain.state_dict())
    with torch.no_grad():
        plain.context_scale.fill_(0.1)
        checked.context_scale.fill_(0.1)
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
        if name_a == "pad_token":
            assert p_a.grad is None and p_b.grad is None
            continue
        assert p_a.grad is not None and p_b.grad is not None
        assert torch.allclose(p_a.grad, p_b.grad, rtol=1e-4, atol=1e-5), name_a


def test_config_kwargs_like_yaml():
    cfg = dict(type="wavelet_v3", output_dim=512, num_tokens=256, coarse_blocks=128,
               n_pos_freqs=16, freeze=False, level_topm=1024, n_heads=8, mlp_ratio=4,
               dropout=0.0, level_gate=True, noise_tau=1.0, pos_dtype="input",
               locality_sigma=0.0078125, context_scale_init=RUN_CONTEXT_SCALE_INIT,
               max_levels=10, tied=True, depth_mode="full",
               reader_mode="cross_only", context_mode="local", self_attn_scale_init=0.1,
               mlp_scale_init=0.1, token_gate=True)
    cfg.pop("type")
    enc = RecurrentWaveletTSEncoder(**cfg)
    assert enc.tied and enc.depth_mode == "full" and enc.level_topm == 1024
    assert enc.reader_mode == "cross_only" and enc.token_gate_enabled
    assert enc.context_mode == "local"
    assert math.isclose(enc.context_scale.item(), RUN_CONTEXT_SCALE_INIT, rel_tol=1e-6)


def test_all_run_and_smoke_configs_enable_context_coadaptation():
    names = (
        "wavelet_precise_v3tied.yaml",
        "wavelet_precise_v3untied.yaml",
        "wavelet_precise_v3sdepth.yaml",
        "wavelet_precise_v3smoke_tied.yaml",
        "wavelet_precise_v3smoke_untied.yaml",
    )
    for name in names:
        candidates = (HERE / name, HERE.parent / "configs" / name)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        assert path is not None, (name, candidates)
        text = path.read_text(encoding="utf-8")
        assert re.search(r"(?m)^\s+context_scale_init:\s+0\.1\s*$", text), path
        assert re.search(r"(?m)^\s+default:\s+1\.0e-4\s*$", text), path
        assert re.search(r"(?m)^\s+context:\s+1\.0e-3\s*$", text), path

    pilot_expectations = {
        "wavelet_precise_v3pilot_cross.yaml": ("cross_only", "full", "false"),
        "wavelet_precise_v3pilot_fullsmall.yaml": ("full", "full", "false"),
        "wavelet_precise_v3pilot_gate.yaml": ("cross_only", "full", "true"),
        "wavelet_precise_v3pilot_crosslocal.yaml": ("cross_only", "local", "false"),
        "wavelet_precise_v3pilot_crossglobal.yaml": ("cross_only", "global", "false"),
    }
    for name, (reader_mode, context_mode, gate) in pilot_expectations.items():
        candidates = (HERE / name, HERE.parent / "configs" / name)
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        assert path is not None, (name, candidates)
        text = path.read_text(encoding="utf-8")
        assert re.search(rf"(?m)^\s+reader_mode:\s+{reader_mode}\s*$", text), path
        configured_mode = re.search(r"(?m)^\s+context_mode:\s+(\w+)\s*$", text)
        assert (configured_mode.group(1) if configured_mode else "full") == context_mode, path
        assert re.search(r"(?m)^\s+self_attn_scale_init:\s+0\.1\s*$", text), path
        assert re.search(r"(?m)^\s+mlp_scale_init:\s+0\.1\s*$", text), path
        assert re.search(rf"(?m)^\s+token_gate:\s+{gate}\s*$", text), path
        assert re.search(r"(?m)^\s+context_scale:\s+1\.0e-5\s*$", text), path
        assert re.search(r"(?m)^\s+context_token_gate:\s+1\.0e-4\s*$", text), path


def test_trainer_and_pilot_hooks_are_env_gated_and_val_only():
    root = HERE.parent.parent
    train_path = root / "kaggle" / "patch" / "train.py"
    pilot_path = HERE.parent / "node" / "node_pilot_v3.sh"
    posthoc_path = HERE.parent / "node" / "node_pilot_posthoc.sh"
    paired_path = HERE.parent / "audit" / "paired_eval.py"
    if not all(path.exists() for path in (train_path, pilot_path, posthoc_path, paired_path)):
        # Repo-layout test: on the node the patch files live flat in ~/patch,
        # so the layout under test is absent and the check does not apply.
        print("SKIP (repo layout absent)", end=" ")
        return
    train_text = train_path.read_text(encoding="utf-8")
    for name in (
        "TSLM_INIT_STATE_PATH", "TSLM_DATA_ORDER_SEED",
        "TSLM_V3_CTX_LOG_EVERY", "TSLM_CHECKPOINT_STEPS",
        "TSLM_V3_CONTEXT_OFF", "TSLM_V3_COLLECT_CONTEXT_STATS",
    ):
        assert name in train_text or name == "TSLM_V3_CONTEXT_OFF", name
    assert 'lr_config.get("context", default_lr)' in train_text
    assert "optimizer checkpoint topology is incompatible" in train_text
    pilot_text = pilot_path.read_text(encoding="utf-8")
    assert "TSLM_SCHED_TOTAL_STEPS=22440" in pilot_text
    assert "TSLM_SUBSET_SEED=42" in pilot_text
    assert "trained 4488, failed 0, expected 4488" in pilot_text
    assert "--split test" not in pilot_text
    assert "node_train_v3.sh" not in pilot_text
    assert "node_eval_worker_v3.sh" not in pilot_text
    assert "TSLM_V3_CONTEXT_EVAL=off" in pilot_text
    assert "TSLM_V3_CONTEXT_EVAL=roll_tokens" in pilot_text
    assert "full_val_off.json" in pilot_text
    assert "epoch${epoch}_paired.json" in pilot_text
    posthoc_text = posthoc_path.read_text(encoding="utf-8")
    assert "epoch_${EPOCH}_end.pt" in posthoc_text
    assert "KEEP $output" in posthoc_text
    assert "--split test" not in posthoc_text
    paired_text = paired_path.read_text(encoding="utf-8")
    assert "BOOTSTRAP_N = 10_000" in paired_text
    assert "BOOTSTRAP_SEED = 20260909" in paired_text


def test_paired_bootstrap_is_seeded_category_aware_and_order_strict():
    path = HERE.parent / "audit" / "paired_eval.py"
    if not path.exists():
        print("SKIP (repo layout absent)", end=" ")
        return
    spec = importlib.util.spec_from_file_location("v3_paired_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    identities = [
        {"id": "a", "category": "cat1", "ground_truth": "g1"},
        {"id": "b", "category": "cat2", "ground_truth": "g2"},
        {"id": "c", "category": "cat1", "ground_truth": "g3"},
    ]
    values = {
        "on": [0.9, 0.2, 0.7],
        "off": [0.4, 0.3, 0.1],
        "roll": [0.5, 0.1, 0.2],
    }
    rows = {
        condition: [dict(identity, primary=value) for identity, value in zip(identities, scores)]
        for condition, scores in values.items()
    }
    first = module.summarize_rows(rows, 3, lambda row: row["primary"], "primary")
    second = module.summarize_rows(rows, 3, lambda row: row["primary"], "primary")
    assert first == second
    assert set(first["comparisons"]) == {
        "on_minus_off", "on_minus_roll", "off_minus_roll"
    }
    assert set(first["comparisons"]["on_minus_off"]["per_category"]) == {
        "cat1", "cat2"
    }
    mismatched = {name: list(condition_rows) for name, condition_rows in rows.items()}
    mismatched["roll"] = list(reversed(mismatched["roll"]))
    try:
        module.summarize_rows(
            mismatched, 3, lambda row: row["primary"], "primary"
        )
    except ValueError as error:
        assert "identity/order mismatch" in str(error)
    else:
        raise AssertionError("paired summarizer accepted reordered rows")


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
