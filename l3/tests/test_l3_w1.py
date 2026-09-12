"""Test CPU del pianificatore W1 (imitazione del selettore)."""

import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l3.tools.compare_selectors import channel_details, geometry
from l3.zoom_traces_w1 import (
    _w1_config,
    build_w1_routes,
    plan_positive_row,
    plan_zoom_row,
    sample_containing_point,
    trace_width,
    wavelet_candidate_centers,
)


def base_config(**overrides):
    w1 = {"candidate_buckets": 4, "draws_per_candidate": 3, "min_wavelet_rate": 0.45}
    w1.update(overrides.pop("w1", {}))
    categories = {
        name: {"replay_fraction": 0.0, "non_replay_route": "zoom_if_positive",
               "zoom_fraction_of_non_replay_positives": 1.0,
               "direct_fraction_of_non_replay_negatives": 1.0}
        for name in ("classify_with_evidence", "localize", "localize_all_channels")
    }
    categories.update({
        name: {"replay_fraction": 0.0, "non_replay_route": "direct",
               "direct_fraction_of_non_replay": 1.0}
        for name in ("measure_magnitude", "lead_lag_with_magnitude")
    })
    config = {
        "zoom_sft": {
            "trace_contract": "z1bis_v1",
            "w1": w1,
            "window_sizes": [{"samples": 1024, "weight": 0.5},
                             {"samples": 2048, "weight": 0.5}],
            "length_bands": [{"name": "L16k", "max_length": 16384},
                             {"name": "L32k", "max_length": 32768}],
            "position": {"centered_radius": 0.01, "centered_max_fraction": 0.15,
                         "mean_range": [0.35, 0.65], "min_std": 0.15},
            "categories": categories,
        },
    }
    config["zoom_sft"].update(overrides)
    return config


def sample(category="localize", length=16384, channels=1, start=11000, end=11100,
           decoy=None):
    names = [f"ch{i}" for i in range(channels)]
    values = [np.zeros(length, dtype=np.float32) for _ in names]
    values[-1][start:end] = 9.0
    if decoy is not None:
        values[-1][decoy:decoy + (end - start)] = 40.0
    if category == "localize_all_channels":
        payload = {"_L": length, "anomalies": [
            {"channel": names[-1], "start": start, "end": end}]}
    elif category == "measure_magnitude":
        payload = {"_L": length, "magnitude_sigma": 12.5}
    else:
        payload = {"_L": length, "present": True, "start": start, "end": end}
    return {
        "id": f"s{length}{start}", "category": category, "length": length,
        "n_channels": channels, "channel_names": names, "time_series": values,
        "time_series_text": [f"The following is channel {name}:" for name in names],
        "pre_prompt": "system", "post_prompt": "Question: locate",
        "answer": f"{category}:" + json.dumps(payload, separators=(",", ":")) + "</s>",
    }


def test_config_validation_is_fail_closed():
    with pytest.raises(ValueError):
        _w1_config({"zoom_sft": {"trace_contract": "legacy_z1", "w1": {}}})
    with pytest.raises(ValueError):
        _w1_config({"zoom_sft": {"trace_contract": "z1bis_v1"}})
    with pytest.raises(ValueError):
        _w1_config(base_config(w1={"candidate_buckets": 0}))
    with pytest.raises(ValueError):
        _w1_config(base_config(w1={"draws_per_candidate": 0}))
    assert _w1_config(base_config())["candidate_buckets"] == 4


def test_containing_point_sampler_stays_in_bounds_and_contains_the_point():
    rng = random.Random(0)
    for point in (0, 1, 5000, 16383):
        for width in (1024, 2048, 16384):
            start, end = sample_containing_point(16384, point, width, rng)
            assert 0 <= start < end <= 16384 and end - start == width
            if width < 16384:
                assert start <= point < end
    with pytest.raises(ValueError):
        sample_containing_point(16384, 16384, 1024, rng)
    with pytest.raises(ValueError):
        sample_containing_point(16384, 10, 0, rng)


def test_candidate_centers_are_ordered_by_energy():
    s = sample(start=11000, end=11100, decoy=3000)
    details = channel_details(s)
    centers = wavelet_candidate_centers(s, 0, 2048, 4, details)
    assert len(centers) == 4 and len(set(centers)) == 4
    # Il decoy e' piu' energetico: deve venire prima dell'evento vero.
    assert abs(centers[0] - 3000) < 2048
    assert any(abs(center - 11050) < 2048 for center in centers)


def test_positive_row_teaches_a_wavelet_window_when_the_energy_is_on_the_event():
    s = sample(start=11000, end=11100)
    config = base_config()
    action, event, position, origin = plan_positive_row(
        s, config["zoom_sft"], _w1_config(config), random.Random(1), channel_details(s))
    assert origin == "wavelet_rank_1"
    assert action.start <= int(event["start"]) and int(event["end"]) <= action.end
    assert 0.0 <= position <= 1.0
    assert geometry(s, action)["state"] == "contained"


def test_decoy_energy_is_recovered_by_a_lower_ranked_candidate():
    s = sample(start=11000, end=11100, decoy=3000)
    config = base_config()
    _, event, _, origin = plan_positive_row(
        s, config["zoom_sft"], _w1_config(config), random.Random(2), channel_details(s))
    assert origin.startswith("wavelet_rank_")
    assert int(origin.rsplit("_", 1)[1]) >= 2


def test_single_candidate_on_a_decoy_falls_back_to_the_gold_window():
    s = sample(start=11000, end=11100, decoy=3000)
    config = base_config(w1={"candidate_buckets": 1, "draws_per_candidate": 2,
                             "min_wavelet_rate": 0.45})
    action, event, _, origin = plan_positive_row(
        s, config["zoom_sft"], _w1_config(config), random.Random(3), channel_details(s))
    assert origin == "gold_fallback"
    # Anche la ricaduta deve contenere l'evento: il piano resta valido.
    assert action.start <= int(event["start"]) and int(event["end"]) <= action.end


def test_routes_split_replay_direct_and_zoom():
    samples = [
        sample(category="localize"),
        sample(category="measure_magnitude"),
        {**sample(category="localize"),
         "answer": "localize:" + json.dumps({"_L": 16384, "present": False}) + "</s>"},
    ]
    config = base_config()
    config["zoom_sft"]["categories"]["localize"]["replay_fraction"] = 0.0
    fixed, zoom_indices, replay = build_w1_routes(samples, config, seed=7)
    assert zoom_indices == {0}
    assert fixed[1].source == "direct" and fixed[2].source == "direct"
    assert replay == set()


def test_replay_quota_is_honoured_by_the_routes():
    samples = [sample(category="localize", start=1000 + 100 * i, end=1100 + 100 * i)
               for i in range(10)]
    for index, item in enumerate(samples):
        item["id"] = f"row{index}"
    config = base_config()
    config["zoom_sft"]["categories"]["localize"]["replay_fraction"] = 0.30
    fixed, zoom_indices, replay = build_w1_routes(samples, config, seed=11)
    assert len(replay) == 3
    assert len(zoom_indices) == 7
    assert all(fixed[index].source == "replay" for index in replay)


def test_plan_zoom_row_is_deterministic_for_a_given_seed():
    s = sample(start=9000, end=9100)
    config = base_config()
    details = channel_details(s)
    first, origin_a = plan_zoom_row(s, details, config, seed=5, index=3)
    second, origin_b = plan_zoom_row(s, details, config, seed=5, index=3)
    assert (first.action.channel, first.action.start, first.action.end) == (
        second.action.channel, second.action.start, second.action.end)
    assert origin_a == origin_b
    other, _ = plan_zoom_row(s, details, config, seed=6, index=3)
    assert first.source == other.source == "zoom"
    assert first.length_band == "L16k"


def test_trace_width_only_uses_declared_sizes_that_fit_the_event():
    s = sample(start=1000, end=2500)
    config = base_config()
    rng = random.Random(0)
    widths = {trace_width(s, config["zoom_sft"],
                          {"channel": "ch0", "start": 1000, "end": 2500}, rng)
              for _ in range(20)}
    assert widths == {2048}
    with pytest.raises(ValueError):
        trace_width(s, config["zoom_sft"],
                    {"channel": "ch0", "start": 0, "end": 9000}, rng)
