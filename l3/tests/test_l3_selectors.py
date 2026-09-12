"""Test CPU per il confronto dei selettori e per il gate del contenuto autonomo."""

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l3.tools.compare_selectors import (
    band_of,
    geometry,
    oracle_at_width,
    recenter,
    summarize,
    wavelet_zoom_at_width,
    wavelet_zoom_in_channel,
)
from l3.zoom import MIN_ZOOM_SAMPLES, ZoomAction, heuristic_zoom


def sample(category="localize", length=8192, channels=1, start=5800, end=5900):
    names = [f"ch{i}" for i in range(channels)]
    values = [np.zeros(length, dtype=np.float32) for _ in names]
    values[-1][start:end] = 9.0
    if category == "localize_all_channels":
        payload = {"_L": length, "anomalies": [
            {"channel": names[-1], "start": start, "end": end},
        ]}
    else:
        payload = {"_L": length, "present": True, "start": start, "end": end}
    return {
        "id": "s", "category": category, "length": length,
        "n_channels": channels, "channel_names": names,
        "time_series": values,
        "time_series_text": [f"The following is channel {name}:" for name in names],
        "pre_prompt": "system", "post_prompt": "Question: locate",
        "answer": f"{category}:" + json.dumps(payload, separators=(",", ":")) + "</s>",
    }


def test_wavelet_selector_mirrors_heuristic_at_native_settings():
    """Il selettore parametrizzato deve coincidere con l3.zoom.heuristic_zoom."""
    for channels in (1, 3):
        s = sample(channels=channels)
        length = int(s["length"])
        native = heuristic_zoom(s)
        mirrored = wavelet_zoom_at_width(
            s, max(MIN_ZOOM_SAMPLES, math.ceil(length / 8)), buckets=64)
        assert (native.channel, native.start, native.end) == (
            mirrored.channel, mirrored.start, mirrored.end)


def test_wavelet_selector_respects_the_requested_width_and_finds_the_event():
    s = sample(length=16384, start=11000, end=11100)
    action = wavelet_zoom_at_width(s, 2048)
    assert action.window_len == 2048
    assert geometry(s, action)["state"] == "contained"


def test_narrow_window_is_a_miss_and_widening_recovers_it():
    s = sample(length=16384, start=11000, end=11100)
    far = ZoomAction("ch0", 8000, 10048, 16384)
    narrow = geometry(s, far)
    assert narrow["state"] == "outside"
    assert narrow["distance_widths"] == pytest.approx((11000 - 10048) / 2048)
    wide = geometry(s, recenter(far, 2048 * 4, 16384))
    assert wide["state"] == "contained"
    assert recenter(far, 2048 * 4, 16384).window_len == 8192


def test_geometry_reports_the_wrong_channel_without_scoring_position():
    s = sample(category="localize_all_channels", channels=4)
    gold_channel = s["channel_names"][-1]
    right = geometry(s, ZoomAction(gold_channel, 5000, 7048, int(s["length"])))
    wrong = geometry(s, ZoomAction(s["channel_names"][0], 5000, 7048, int(s["length"])))
    assert right["state"] == "contained" and right["channel_hit"] is True
    assert wrong["state"] == "wrong_channel" and wrong["channel_hit"] is False
    assert "position" not in wrong


def test_oracle_at_width_contains_the_event_by_construction():
    for length, start in ((16384, 200), (32768, 31000), (65536, 30000)):
        s = sample(length=length, start=start, end=start + 64)
        action = oracle_at_width(s, 1024)
        assert action is not None
        assert geometry(s, action)["state"] == "contained"


def test_negative_rows_are_marked_without_gold():
    s = sample()
    s["answer"] = "localize:" + json.dumps({"_L": 8192, "present": False}) + "</s>"
    assert geometry(s, ZoomAction("ch0", 0, 2048, 8192))["state"] == "no_gold"


def test_summarize_excludes_unscored_rows_from_the_rates():
    records = [
        {"conditions": {"k": {"state": "contained", "position": 0.5,
                              "distance_widths": 0.0, "width": 2048,
                              "channel_hit": True}}},
        {"conditions": {"k": {"state": "outside", "position": 2.0,
                              "distance_widths": 1.5, "width": 2048,
                              "channel_hit": True}}},
        {"conditions": {"k": {"state": "no_gold"}}},
    ]
    out = summarize(records, "k")
    assert out["n"] == 3 and out["scored"] == 2
    assert out["contained_rate"] == pytest.approx(0.5)
    assert out["miss_distance_median_widths"] == pytest.approx(1.5)
    assert out["mean_width"] == pytest.approx(2048)


def test_band_of_maps_the_declared_lengths():
    assert [band_of(n) for n in (16384, 20000, 32768, 70000, 131072)] == [
        16384, 32768, 32768, 131072, 131072]


def test_selfwindow_gate_paires_conditions_and_registers_the_subhypothesis():
    from l3.tools.gate_selfwindow_content import paired_section, report

    rows = []
    for index in range(12):
        contained = index < 8
        for condition, value in (("self_true", 0.4 if contained else 0.0),
                                 ("self_swap", 0.1), ("self_zero", 0.05)):
            rows.append({
                "id": f"row{index}", "condition": condition, "primary": value,
                "category": "localize", "band": 16384,
                "geometry": {"state": "contained" if contained else "outside"},
            })
    out = report(rows, seed=1, resamples=200, min_delta=0.03,
                 config_path=Path("c.yaml"), checkpoint=Path("k.pt"),
                 policy_rows=Path("p.jsonl"))
    assert out["overall"]["n"] == 12
    assert out["overall"]["best_control"] == "self_swap"
    assert out["gate_passed"] is True
    sub = out["registered_subhypothesis"]
    assert sub["contained_delta"] == pytest.approx(0.3)
    assert sub["not_contained_delta"] == pytest.approx(-0.1)
    assert sub["contained_exceeds_not_contained"] is True
    assert paired_section(rows, 1, 200, ["row0"])["n"] == 1


def test_selfwindow_gate_fails_closed_when_the_effect_is_absent():
    from l3.tools.gate_selfwindow_content import report

    rows = []
    for index in range(10):
        for condition in ("self_true", "self_swap", "self_zero"):
            rows.append({
                "id": f"row{index}", "condition": condition, "primary": 0.2,
                "category": "localize", "band": 16384,
                "geometry": {"state": "contained"},
            })
    out = report(rows, seed=1, resamples=200, min_delta=0.03,
                 config_path=Path("c.yaml"), checkpoint=Path("k.pt"),
                 policy_rows=Path("p.jsonl"))
    assert out["gate_passed"] is False
    assert out["checks"] == {"effect_at_least_threshold": False,
                             "ci95_excludes_zero": False}


def test_width_ladder_is_paired_and_stratified_on_the_base_misses():
    from l3.tools.gate_selfwindow_content import report

    rows = []
    for index in range(10):
        contained = index < 6
        for condition, value in (("self_true", 0.3 if contained else 0.0),
                                 ("self_swap", 0.05), ("self_zero", 0.05),
                                 ("self_wide_x2", 0.3 if contained else 0.2),
                                 ("self_wide_x4", 0.1)):
            rows.append({
                "id": f"row{index}", "condition": condition, "primary": value,
                "category": "localize", "band": 16384,
                "geometry": {"state": "contained" if contained
                             else ("contained" if condition == "self_wide_x2" else "outside")},
            })
    out = report(rows, seed=1, resamples=200, min_delta=0.03,
                 config_path=Path("c.yaml"), checkpoint=Path("k.pt"),
                 policy_rows=Path("p.jsonl"))
    ladder = out["width_ladder"]
    assert set(ladder) == {"self_wide_x2", "self_wide_x4"}
    assert ladder["self_wide_x2"]["n"] == 10
    assert ladder["self_wide_x2"]["contained_rate"] == pytest.approx(1.0)
    assert ladder["self_wide_x2"]["delta_vs_self_true"]["mean"] == pytest.approx(0.08)
    misses = ladder["self_wide_x2"]["on_base_misses"]
    assert misses["n"] == 4 and misses["base_mean"] == pytest.approx(0.0)
    assert misses["delta"]["mean"] == pytest.approx(0.2)
    assert ladder["self_wide_x4"]["delta_vs_self_true"]["mean"] == pytest.approx(-0.08)
    assert out["gate_passed"] is True


def test_channel_restricted_selector_stays_in_the_requested_channel():
    s = sample(category="localize_all_channels", channels=4, start=6000, end=6100)
    gold_channel = s["channel_names"][-1]
    right = wavelet_zoom_in_channel(s, gold_channel, 2048)
    assert right.channel == gold_channel and right.window_len == 2048
    assert geometry(s, right)["state"] == "contained"
    other = wavelet_zoom_in_channel(s, s["channel_names"][0], 2048)
    assert other.channel == s["channel_names"][0]
    assert geometry(s, other)["state"] == "wrong_channel"
    with pytest.raises(ValueError):
        wavelet_zoom_in_channel(s, "ch_inesistente", 2048)


def test_selector_scores_pair_against_the_policy_arm(tmp_path):
    from l3.tools.score_selectors import baseline_scores, build_action, report

    s = sample(length=16384, start=11000, end=11100)
    policy = ZoomAction("ch0", 8000, 10048, 16384)
    from l3.tools.compare_selectors import channel_details
    details = channel_details(s)
    for name in ("wavelet_matched", "wavelet_in_policy_channel", "oracle_matched"):
        action = build_action(name, s, policy, details)
        assert action is not None and action.window_len == policy.window_len
    with pytest.raises(ValueError):
        build_action("inesistente", s, policy, details)

    path = tmp_path / "rows.jsonl"
    lines = [json.dumps({"id": f"row{i}", "condition": c, "primary": v})
             for i in range(4)
             for c, v in (("self_true", 0.1), ("self_swap", 0.0))]
    path.write_text(chr(10).join(lines), encoding="utf-8")
    base = baseline_scores(path, "self_true")
    assert base == {f"row{i}": 0.1 for i in range(4)}

    rows = [{"id": f"row{i}", "selector": "oracle_matched", "primary": 0.5,
             "category": "localize", "band": 16384,
             "geometry": {"state": "contained"}} for i in range(4)]
    out = report(rows, base, selectors=["oracle_matched"], seed=1, resamples=200,
                 config_path=Path("c.yaml"), checkpoint=Path("k.pt"),
                 policy_rows=Path("p.jsonl"), baseline_rows=path)
    arm = out["arms"]["oracle_matched"]
    assert arm["contained_rate"] == pytest.approx(1.0)
    assert arm["paired"]["delta_vs_policy"]["mean"] == pytest.approx(0.4)
    assert arm["mean_by_state"] == {"contained": pytest.approx(0.5)}
