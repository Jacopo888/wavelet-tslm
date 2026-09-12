from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l3.grpo import load_l3_config
from l3.tools.diag_crop_content import crop_header_fingerprints
from l3.tools.gate_z1bis_operational import behavioral_metrics
from l3.zoom import ZoomAction
from l3.zoom_traces import (
    Trace,
    _load_train_samples,
    build_trace_plan,
    sample_containing_action,
    validate_trace_plan,
)


CONFIG_PATH = ROOT / "l3" / "configs" / "l3_zoom_z1bis.yaml"


def _parquet_row(index, category, sentinel):
    if category == "localize":
        payload = {"_L": 16, "present": False}
    elif category == "classify_with_evidence":
        payload = {"_L": 16, "kind": "No anomaly"}
    elif category == "localize_all_channels":
        payload = {"_L": 16, "anomalies": []}
    elif category == "measure_magnitude":
        payload = {"_L": 16, "magnitude_sigma": 2.0}
    else:
        payload = {"_L": 16, "direction": "independent", "lag_samples": 0}
    return {
        "id": f"row-{index}", "category": category, "length": 16,
        "n_channels": 1, "channel_names": ["ch0"],
        "channel_values": [[float(sentinel)] * 16],
        "answer": f"{category}:" + json.dumps(payload),
    }


def _planner_test_config(train_dir, train_sha, rows):
    config = deepcopy(load_l3_config(CONFIG_PATH))
    config["dataset"]["extra_kwargs"]["train_dir"] = str(train_dir)
    config["rl"]["expected_train_rows"] = rows
    config["rl"]["train_sha256"] = train_sha
    config["zoom_sft"]["quota_tolerance"] = 0.0
    for rule in config["zoom_sft"]["categories"].values():
        rule["replay_fraction"] = 0.0
    return config


def _sample(index, category, *, positive=True, length=16384):
    start = 3000 + (index * 37) % 8000
    end = start + 80
    names = ["ch0", "ch1"] if category == "localize_all_channels" else ["ch0"]
    if category == "localize":
        payload = ({"_L": length, "present": True, "start": start, "end": end}
                   if positive else {"_L": length, "present": False})
    elif category == "classify_with_evidence":
        payload = ({"_L": length, "kind": "Transient Spike", "start": start, "end": end}
                   if positive else {"_L": length, "kind": "No anomaly"})
    elif category == "localize_all_channels":
        payload = {"_L": length, "anomalies": ([{
            "channel": "ch1", "start": start, "end": end,
        }] if positive else [])}
    elif category == "measure_magnitude":
        payload = {"_L": length, "magnitude_sigma": 2.0}
    else:
        payload = {"_L": length, "direction": "independent", "lag_samples": 0}
    return {
        "id": f"{category}-{index}", "category": category, "length": length,
        "channel_names": names, "time_series": [range(length) for _ in names],
        "answer": f"{category}:" + json.dumps(payload),
    }


def test_train_loader_projects_metadata_and_ignores_channel_values(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from l3.grpo import sha256_file
    from l3.zoom_traces import TRAIN_TRACE_COLUMNS

    categories = sorted(load_l3_config(CONFIG_PATH)["zoom_sft"]["categories"])
    plans = []
    real_read_table = pq.read_table
    projected_columns = []

    def checked_read_table(*args, **kwargs):
        projected_columns.append(kwargs.get("columns"))
        return real_read_table(*args, **kwargs)

    monkeypatch.setattr(pq, "read_table", checked_read_table)
    for variant, sentinel in enumerate((123.0, -987.0)):
        train_dir = tmp_path / f"variant-{variant}"
        train_dir.mkdir()
        train_path = train_dir / "train.parquet"
        rows = [_parquet_row(index, category, sentinel)
                for index, category in enumerate(categories)]
        pq.write_table(pa.Table.from_pylist(rows), train_path)
        config = _planner_test_config(train_dir, sha256_file(train_path), len(rows))
        samples, observed_sha = _load_train_samples(config)
        assert observed_sha == config["rl"]["train_sha256"]
        assert all(
            isinstance(channel, range) and len(channel) == sample["length"]
            for sample in samples for channel in sample["time_series"]
        )
        plans.append(build_trace_plan(samples, config, 20260912))

    assert projected_columns == [list(TRAIN_TRACE_COLUMNS), list(TRAIN_TRACE_COLUMNS)]
    assert "channel_values" not in TRAIN_TRACE_COLUMNS
    assert plans[0] == plans[1]


def test_train_loader_fails_closed_on_missing_required_column(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from l3.grpo import sha256_file

    train_path = tmp_path / "train.parquet"
    row = _parquet_row(0, "localize", 1.0)
    del row["answer"]
    pq.write_table(pa.Table.from_pylist([row]), train_path)
    config = _planner_test_config(tmp_path, sha256_file(train_path), 1)
    with pytest.raises(ValueError, match=r"missing required column: answer"):
        _load_train_samples(config)


def test_uniform_containing_start_has_variable_internal_position():
    sample = _sample(0, "localize")
    event = {"channel": "ch0", "start": 6000, "end": 6080}
    positions = [
        sample_containing_action(sample, event, 1024, random.Random(seed))[1]
        for seed in range(1000)
    ]
    mean = sum(positions) / len(positions)
    std = (sum((value - mean) ** 2 for value in positions) / len(positions)) ** 0.5
    assert 0.47 < mean < 0.53
    assert std > 0.20
    assert len({round(value, 3) for value in positions}) > 500


def test_trace_plan_enforces_category_quotas_and_has_no_pseudo_recovery():
    config = load_l3_config(CONFIG_PATH)
    categories = sorted(config["zoom_sft"]["categories"])
    samples = []
    for category in categories:
        for index in range(240):
            positive = not (category in {"localize", "classify_with_evidence", "localize_all_channels"}
                            and index % 4 == 0)
            samples.append(_sample(index, category, positive=positive))
    traces = build_trace_plan(samples, config, 20260912)
    summary = validate_trace_plan(samples, traces, config)
    assert summary["pseudo_recovery_count"] == 0
    for category in categories:
        counts = summary["source_counts"][category]
        assert counts["replay"] == 36
        assert all(value in {0.15, 1.0}
                   for value in summary["observed_category_quotas"][category].values())
    assert all(trace.action is not None for trace in traces if trace.source == "zoom")
    assert all(trace.action is None for trace in traces if trace.source != "zoom")
    assert all(
        trace.source != "direct" or not trace.event
        for trace in traces
    )


def test_positive_direct_trace_is_rejected_as_pseudo_recovery():
    config = load_l3_config(CONFIG_PATH)
    sample = _sample(0, "localize", positive=True)
    trace = Trace(index=0, sample_id=sample["id"], category="localize", source="direct")
    with pytest.raises(ValueError, match="pseudo-recovery"):
        validate_trace_plan([sample], [trace], config)


def test_perceptual_crop_conditions_have_identical_headers():
    action = ZoomAction("ch0", 1234, 3282, 16384)
    fingerprints = crop_header_fingerprints(action)
    assert set(fingerprints) == {"offcenter_true", "offcenter_swap", "offcenter_zero"}
    assert len(set(fingerprints.values())) == 1


def test_operational_metrics_report_both_error_directions_and_direct_use():
    rows = [
        {"id": "p", "category": "localize",
         "ground_truth": 'localize:{"present":true,"start":1,"end":2}',
         "prediction": 'localize:{"present":false}', "zoom": None,
         "zoom_invalid": False, "empty_final": False, "primary": 0.0},
        {"id": "n", "category": "localize",
         "ground_truth": 'localize:{"present":false}',
         "prediction": 'localize:{"present":true,"start":1,"end":2}',
         "zoom": {"start": 0}, "zoom_invalid": False, "empty_final": False,
         "primary": 0.0},
    ]
    metrics = behavioral_metrics(rows)
    assert metrics["false_negative_rate_on_positives"] == 1.0
    assert metrics["false_positive_rate_on_negatives"] == 1.0
    assert metrics["positive_direct_rate"] == 1.0
    assert metrics["negative_action_rate"] == 1.0


def test_config_and_launcher_freeze_the_predesign_and_review_interlock():
    config = load_l3_config(CONFIG_PATH)
    assert config["model"]["encoder"]["freeze"] is True
    assert config["zoom_sft"]["frozen_parameter_prefixes"] == ["ts_encoder.", "ts_projector."]
    assert config["perceptual_gate"]["min_true_minus_best_control"] == 0.05
    assert config["perceptual_gate"]["bootstrap_resamples"] == 10000
    driver = (ROOT / "l3" / "node_l3_z1bis.sh").read_text(encoding="utf-8")
    assert "L3_Z1BIS_DIFF_REVIEWED=CLAUDE_APPROVED" in driver
    assert driver.index("start_phase traces") < driver.index("start_phase sft")
    assert driver.index("start_phase sft") < driver.index("start_phase perceptual_gate")
    assert driver.index("start_phase perceptual_gate") < driver.index("start_phase operational_gate")
    forbidden_test_split = "--split" + " test"
    assert "--mode policy" in driver and forbidden_test_split not in driver
    assert 'TRACE_TIMEOUT_SECONDS="${L3_Z1BIS_TRACE_TIMEOUT_SECONDS:-900}"' in driver
    assert 'TRACE_PEAK_LIMIT_MIB="${L3_Z1BIS_TRACE_PEAK_LIMIT_MIB:-4096}"' in driver
    assert "run_trace_planner_guarded" in driver
    assert 'CUDA_VISIBLE_DEVICES=""' in driver
