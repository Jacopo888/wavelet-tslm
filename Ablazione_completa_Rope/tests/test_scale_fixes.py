"""Astra 15/09 (scala): hard clip of the linear stem columns, per-token normalisation of the fused content, projector
gain calibrated to a target RMS, and the gate veto for main categories at zero."""

from __future__ import annotations

import torch

from wavelet_rope.deep_branches import BranchStem
from wavelet_rope.perceptual_gate import summarize_gate_results
from wavelet_rope.perceptual_rope import MIXER_DIM, Projector
from wavelet_rope.tokenizer import BAND_SCALE_FLOOR, LINEAR_CLIP, robust_band_scale


def test_stem_output_is_layer_normalised_and_masked() -> None:
    stem = BranchStem(9, 64)
    feats = torch.randn(2, 7, 9) * 1000.0  # wildly scaled inputs
    valid = torch.ones(2, 7, dtype=torch.bool)
    valid[1, 3:] = False
    out = stem(feats, valid)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms[valid], torch.full_like(norms[valid], 8.0), atol=0.05)
    assert (out[~valid] == 0).all()


def test_band_scale_floor_and_clip_constants() -> None:
    values = torch.tensor([[1.0, 1.0, 1.0, 1.0 + 1e-7]])  # nearly constant band: MAD 0
    scale = robust_band_scale(values, torch.ones_like(values, dtype=torch.bool))
    assert abs(float(scale[0]) - BAND_SCALE_FLOOR) < 1e-9
    assert LINEAR_CLIP == 16.0


def test_projector_normalises_every_token_to_the_target_rms() -> None:
    proj = Projector(2560)
    assert proj.target_rms is None and (proj.proj_out.bias == 0).all()
    fixed = Projector(2560, target_rms=0.0137)
    z = torch.randn(300, MIXER_DIM) * 3.0
    with torch.no_grad():
        y = fixed(z)
    per_token = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(per_token, torch.full_like(per_token, 0.0137), rtol=1e-3)
    assert "out_scale" not in fixed.state_dict()  # no learned or calibrated gain (Astra 15/09, sentinella D2)


def _records(true: float, substituted: float, zeroed: float, *, category: str, rid: str) -> dict:
    return {
        "recording_id": rid, "category": category,
        "primary": {"TRUE": true, "SUBSTITUTED": substituted, "ZEROED": zeroed},
        "invalid_format": {"TRUE": False, "SUBSTITUTED": False, "ZEROED": False},
    }


def test_gate_vetoes_main_category_at_zero_even_with_aggregate_advantage() -> None:
    records = []
    for i in range(12):
        records.append(_records(0.9, 0.1, 0.1, category="localize", rid=f"a{i}"))
        records.append(_records(0.0, 0.0, 0.0, category="classify_with_evidence", rid=f"b{i}"))
    summary = summarize_gate_results(records, seed=1, n_resamples=500)
    assert summary["criteria"]["overall_ci95_lower_gt_zero"]
    assert summary["criteria"]["no_category_collapse"]  # flat zeros are not a collapse in the old sense
    assert summary["main_categories_at_zero"] == ["classify_with_evidence"]
    assert not summary["criteria"]["no_main_category_at_zero"]
    assert summary["verdict"] == "FAIL"
    healthy = [_records(0.9, 0.1, 0.1, category=c, rid=f"{c}{i}") for i in range(12)
               for c in ("localize", "classify_with_evidence")]
    assert summarize_gate_results(healthy, seed=1, n_resamples=500)["verdict"] == "PASS"
