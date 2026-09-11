import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l3.grpo import load_l3_config, zoom_pool_for_round
from l3.zoom import (
    ACTION_INSTRUCTION,
    MIN_ZOOM_SAMPLES,
    ZoomAction,
    build_turn2_sample,
    build_zoom_block,
    choose_supervised_action,
    clamp_zoom_window,
    heuristic_zoom,
    make_zoom_prompt,
    oracle_zoom,
    parse_zoom_decision,
    prepare_zoom_batch,
    supervised_segments,
    two_segment_token_mask,
    zoom_iou_with_gold,
)


def sample(category="localize", length=4096, channels=1):
    names = [f"ch{i}" for i in range(channels)]
    values = [np.zeros(length, dtype=np.float32) for _ in names]
    values[-1][1800:1900] = 8.0
    if category == "localize_all_channels":
        payload = {"_L": length, "anomalies": [
            {"channel": names[0], "start": 100, "end": 200},
            {"channel": names[-1], "start": 1800, "end": 1950},
        ]}
    else:
        payload = {"_L": length, "present": True, "start": 1800, "end": 1900}
    return {
        "id": "s", "category": category, "length": length,
        "n_channels": channels, "channel_names": names,
        "time_series": values,
        "time_series_text": [f"The following is channel {name}:" for name in names],
        "pre_prompt": "system", "post_prompt": "Question: locate",
        "answer": f"{category}:" + json.dumps(payload, separators=(",", ":")) + "</s>",
    }


def test_window_clamps_and_expands_to_minimum():
    assert clamp_zoom_window(-50, 100, 4096) == (0, MIN_ZOOM_SAMPLES)
    assert clamp_zoom_window(4000, 5000, 4096) == (4096 - MIN_ZOOM_SAMPLES, 4096)
    assert clamp_zoom_window(-2000, -1, 4096) is None
    assert clamp_zoom_window(20, 20, 4096) is None
    assert clamp_zoom_window(0, 10, 100) == (0, 100)


def test_strict_action_parser_valid_invalid_and_out_of_range():
    valid = parse_zoom_decision(
        'zoom:{"channel":"ch0","start":-2,"end":100}', ["ch0"], 4096
    )
    assert valid.kind == "action" and valid.action == ZoomAction("ch0", 0, 1024, 4096)
    assert parse_zoom_decision("not zoom", ["ch0"], 4096).kind == "final"
    assert parse_zoom_decision(
        'zoom:{"channel":"ch0","start":1,"end":2,"extra":3}', ["ch0"], 4096
    ).kind == "invalid"
    assert parse_zoom_decision(
        'zoom:{"channel":"missing","start":1,"end":2000}', ["ch0"], 4096
    ).kind == "invalid"
    assert parse_zoom_decision(
        'zoom:{"channel":"ch0","start":5000,"end":6000}', ["ch0"], 4096
    ).kind == "invalid"
    assert parse_zoom_decision(
        'zoom:{"channel":"ch0","start":true,"end":2000}', ["ch0"], 4096
    ).kind == "invalid"


def test_zoom_block_keeps_absolute_header_and_exact_raw_slice():
    s = sample()
    s["time_series"][0] = np.arange(4096, dtype=np.float32)
    action = ZoomAction("ch0", 1000, 2024, 4096)
    header, values = build_zoom_block(s, action)
    prompt = make_zoom_prompt(s, action)
    assert header == "Zoom on channel ch0, samples [1000, 2024) of 4096:"
    assert np.array_equal(values, np.arange(1000, 2024, dtype=np.float32))
    assert prompt.get_text() == header and np.array_equal(prompt.get_time_series(), values)
    second = build_turn2_sample(s, action)
    assert second["time_series_text"][-1] == header
    assert second["time_series_text"][:-1] == s["time_series_text"]
    assert ACTION_INSTRUCTION in second["post_prompt"]
    assert "Assistant action: zoom:" in second["post_prompt"]
    assert second["post_prompt"].endswith("Now output the final answer in absolute coordinates.")


def test_prepare_zoom_batch_encodes_window_at_its_own_length():
    class FakeLlm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class FakeModel:
        def __init__(self):
            self.llm_model = FakeLlm()
            self.encoded_shapes = []

        def parameters(self):
            return self.llm_model.parameters()

        def prepare_batch(self, batch, training=False):
            channels = batch[0]["time_series"]
            width = max(map(len, channels))
            tensor = torch.zeros((len(channels), width))
            return {"time_series": tensor, "input_ids": torch.ones((1, 8), dtype=torch.long),
                    "attention_mask": torch.ones((1, 8), dtype=torch.long), "labels": None}

        def _encode_and_project(self, values):
            self.encoded_shapes.append(tuple(values.shape))
            return torch.zeros((values.shape[0] * 256, 2))

    model = FakeModel()
    s = sample()
    action = ZoomAction("ch0", 1000, 2024, 4096)
    prepared = prepare_zoom_batch(model, s, action, training=False)
    assert model.encoded_shapes == [(1, 4096), (1, 1024)]
    assert prepared["_l3_ts_features"].shape == (512, 2)


def test_supervised_segments_target_only_action_then_gold():
    s = sample()
    action = ZoomAction("ch0", 1000, 2024, 4096)
    first, second = supervised_segments(s, action, "</s>")
    assert first["answer"] == action.text() + "</s>"
    assert second["answer"] == s["answer"]
    assert len(first["time_series"]) == 1 and len(second["time_series"]) == 2


def test_two_segment_mask_never_marks_environment_tokens():
    mask = two_segment_token_mask(5, 2, 9, 3)
    assert mask.tolist() == [False] * 5 + [True] * 2 + [False] * 9 + [True] * 3


def test_oracle_multichannel_choice_jitter_and_iou():
    s = sample("localize_all_channels", channels=2)
    deterministic = oracle_zoom(s)
    assert deterministic.channel == "ch1"  # wider event wins in Z0
    assert deterministic.start <= 1800 < 1950 <= deterministic.end
    assert zoom_iou_with_gold(s, deterministic) > 0
    seen = {oracle_zoom(s, margin=0.25, rng=random.Random(seed),
                        random_multichannel=True).channel for seed in range(20)}
    assert seen == {"ch0", "ch1"}


def test_heuristic_energy_scan_selects_signal_channel_and_valid_window():
    s = sample("localize_all_channels", channels=2)
    action = heuristic_zoom(s)
    assert action.channel == "ch1"
    assert 0 <= action.start < action.end <= 4096
    assert action.window_len >= MIN_ZOOM_SAMPLES
    assert action.start < 1900 and action.end > 1800


def test_z1_action_mix_always_returns_one_valid_action():
    s = sample()
    action, source = choose_supervised_action(s, random.Random(7), 0.30)
    assert source in {"gold_jitter", "heuristic"}
    assert action.window_len >= MIN_ZOOM_SAMPLES


def test_zoom_pool_is_50x4_resampled_and_excludes_lead_lag():
    categories = [
        "classify_with_evidence", "measure_magnitude", "localize",
        "localize_all_channels", "lead_lag_with_magnitude",
    ]
    metadata = []
    for cat_index, category in enumerate(categories):
        for offset in range(80):
            metadata.append({
                "index": cat_index * 1000 + offset, "category": category,
                "length": [16384, 32768, 65536, 131072][offset % 4],
            })
    first = zoom_pool_for_round(metadata, 20260911, 0)
    again = zoom_pool_for_round(metadata, 20260911, 0)
    second = zoom_pool_for_round(metadata, 20260911, 1)
    assert first == again and first != second and len(first) == len(set(first)) == 200
    rows = {row["index"]: row for row in metadata}
    assert all(rows[index]["category"] != "lead_lag_with_magnitude" for index in first)
    assert {category: sum(rows[index]["category"] == category for index in first)
            for category in categories[:-1]} == {category: 50 for category in categories[:-1]}


@pytest.mark.parametrize("name", ["z0", "z1", "z2"])
def test_zoom_configs_freeze_declared_contract(name):
    config = load_l3_config(ROOT / "l3" / "configs" / f"l3_zoom_{name}.yaml")
    rl = config["rl"]
    assert rl["contract"] == "zoom" and rl["steps"] == 200
    assert rl["excluded_categories"] == ["lead_lag_with_magnitude"]
    assert (rl["pool_size"], rl["pool_per_category"], rl["resample_every"]) == (200, 50, 100)
