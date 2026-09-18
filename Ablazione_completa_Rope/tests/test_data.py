"""Tests for the AnomalyXL data adapter (``wavelet_rope.data``).

Everything runs on a tiny synthetic parquet written with pyarrow in a tmp dir:
same schema and column types as ``Localize/{train,val,test}.parquet``, three
row groups so the lazy per-row reads have to resolve a row-group offset.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from wavelet_rope.data import (
    AnomalyXLSplit,
    GoldInterval,
    ModeAssigner,
    RowMeta,
    TeacherCropSampler,
    ZoomAction,
    occupancy_targets,
    parse_zoom_action,
    stratified_subset,
)
from wavelet_rope.scoring import PRECISE_SYSTEM_PROMPT, score_precise
from wavelet_rope.signal_contract import (
    SignalEpisode,
    ViewKind,
    compute_recording_normalization,
    make_view,
)

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("tier", pa.int8()),
        ("category", pa.string()),
        ("length", pa.int32()),
        ("n_channels", pa.int32()),
        ("n_channels_a", pa.int32()),
        ("n_channels_b", pa.int32()),
        ("question", pa.string()),
        ("options", pa.list_(pa.string())),
        ("answer", pa.string()),
        ("semantic_class", pa.string()),
        ("coupling_kind", pa.string()),
        ("coupling_lag", pa.int32()),
        ("injection_kinds", pa.list_(pa.string())),
        ("injection_channels", pa.list_(pa.string())),
        ("injection_starts", pa.list_(pa.int32())),
        ("injection_ends", pa.list_(pa.int32())),
        ("channel_names", pa.list_(pa.string())),
        ("channel_values", pa.list_(pa.list_(pa.float32()))),
        ("seed", pa.int64()),
    ]
)


def _channel_values(row_index: int, n_channels: int, length: int) -> list[list[float]]:
    rng = np.random.default_rng(1000 + row_index)
    out = []
    for channel in range(n_channels):
        values = (rng.standard_normal(length) + channel).astype(np.float32)
        out.append([float(v) for v in values])
    return out


def _row(
    row_index: int,
    row_id: str,
    category: str,
    length: int,
    n_channels: int,
    injections: list[tuple[str, int, int, str]],
    answer: str,
) -> dict:
    names = [f"ch_{index:02d}" for index in range(n_channels)]
    return {
        "id": row_id,
        "tier": 1,
        "category": category,
        "length": length,
        "n_channels": n_channels,
        "n_channels_a": 1 if n_channels > 1 else 0,
        "n_channels_b": 1 if n_channels > 1 else 0,
        "question": f"Where is the anomaly? (Here L = {length}.)",
        "options": [],
        "answer": answer,
        "semantic_class": "synthetic",
        "coupling_kind": "none",
        "coupling_lag": 0,
        "injection_kinds": [item[3] for item in injections],
        "injection_channels": [item[0] for item in injections],
        "injection_starts": [item[1] for item in injections],
        "injection_ends": [item[2] for item in injections],
        "channel_names": names,
        "channel_values": _channel_values(row_index, n_channels, length),
        "seed": 42 + row_index,
    }


def _rows() -> list[dict]:
    specs = [
        ("loc-000", "localize", 1024, 1, [("ch_00", 300, 340, "spike")],
         'localize:{"_L": 1024, "end": 340, "present": true, "start": 300}'),
        ("loc-001", "localize", 1024, 1, [],
         'localize:{"_L": 1024, "present": false}'),
        ("loc-002", "localize", 128, 1, [("ch_00", 10, 30, "spike")],
         'localize:{"_L": 128, "end": 30, "present": true, "start": 10}'),
        ("cls-000", "classify_with_evidence", 1024, 1, [("ch_00", 700, 760, "level_shift")],
         'classify_with_evidence:{"_L": 1024, "end": 760, "kind": "level_shift", "start": 700}'),
        ("cls-001", "classify_with_evidence", 1024, 1, [("ch_00", 120, 180, "burst")],
         'classify_with_evidence:{"_L": 1024, "end": 180, "kind": "burst", "start": 120}'),
        ("mag-000", "measure_magnitude", 1024, 1, [("ch_00", 100, 164, "spike")],
         'measure_magnitude:{"_L": 1024, "magnitude_sigma": 4.5}'),
        ("mag-001", "measure_magnitude", 1024, 1, [("ch_00", 500, 520, "spike")],
         'measure_magnitude:{"_L": 1024, "magnitude_sigma": 2.0}'),
        ("all-000", "localize_all_channels", 1024, 16,
         [("ch_03", 200, 260, "spike"), ("ch_09", 640, 700, "drift")],
         'localize_all_channels:{"_L": 1024, "anomalies": [{"channel": "ch_03", "start": 200, '
         '"end": 260}, {"channel": "ch_09", "start": 640, "end": 700}]}'),
        ("all-001", "localize_all_channels", 512, 16, [("ch_00", 50, 90, "spike")],
         'localize_all_channels:{"_L": 512, "anomalies": [{"channel": "ch_00", "start": 50, '
         '"end": 90}]}'),
        ("lag-000", "lead_lag_with_magnitude", 1024, 2,
         [("ch_00", 400, 440, "burst"), ("ch_01", 420, 460, "burst")],
         'lead_lag_with_magnitude:{"_L": 1024, "direction": "a_leads_b", "lag_samples": 20}'),
        ("lag-001", "lead_lag_with_magnitude", 1024, 2, [("ch_00", 100, 140, "burst")],
         'lead_lag_with_magnitude:{"_L": 1024, "direction": "independent", "lag_samples": 0}'),
    ]
    rows = [
        _row(index, row_id, category, length, n_channels, injections, answer)
        for index, (row_id, category, length, n_channels, injections, answer) in enumerate(specs)
    ]
    # Non-finite samples in one channel of one row: they must become invalid and
    # must not be imputed by the adapter.
    values = rows[4]["channel_values"][0]
    values[5] = float("nan")
    values[6] = float("inf")
    values[7] = float("-inf")
    return rows


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[dict]]:
    rows = _rows()
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    path = tmp_path_factory.mktemp("anomalyxl") / "train.parquet"
    pq.write_table(table, path, row_group_size=3)
    return path, rows


@pytest.fixture(scope="module")
def split(dataset: tuple[Path, list[dict]]) -> AnomalyXLSplit:
    path, rows = dataset
    return AnomalyXLSplit(path, "train", expected_rows=len(rows))


def _meta_by_id(split: AnomalyXLSplit, row_id: str) -> RowMeta:
    for meta in split.metas():
        if meta.id == row_id:
            return meta
    raise AssertionError(f"missing fixture row {row_id!r}")


def _synthetic_metas(per_category: dict[str, int]) -> list[RowMeta]:
    metas: list[RowMeta] = []
    index = 0
    for category in sorted(per_category):
        for _ in range(per_category[category]):
            metas.append(
                RowMeta(
                    id=f"{category}-{index:04d}",
                    category=category,
                    length=1024,
                    n_channels=1,
                    channel_names=("ch_00",),
                    question="q",
                    answer="a",
                    gold_intervals=(),
                    split="train",
                    row_index=index,
                )
            )
            index += 1
    return metas


# --------------------------------------------------------------------- split


def test_row_count_is_verified(dataset: tuple[Path, list[dict]]) -> None:
    path, rows = dataset
    opened = AnomalyXLSplit(path, "train", expected_rows=len(rows))
    assert len(opened) == len(rows) == 11
    with pytest.raises(ValueError, match="expected 4488"):
        AnomalyXLSplit(path, "train", expected_rows=4488)
    with pytest.raises(FileNotFoundError):
        AnomalyXLSplit(path.parent / "nope.parquet", "val", expected_rows=None)


def test_category_counts(split: AnomalyXLSplit) -> None:
    assert split.category_counts() == {
        "classify_with_evidence": 2,
        "lead_lag_with_magnitude": 2,
        "localize": 3,
        "localize_all_channels": 2,
        "measure_magnitude": 2,
    }


def test_meta_fields_and_gold_alignment(split: AnomalyXLSplit, dataset) -> None:
    _, rows = dataset
    for index, row in enumerate(rows):
        meta = split.meta(index)
        assert (meta.id, meta.category, meta.split, meta.row_index) == (
            row["id"],
            row["category"],
            "train",
            index,
        )
        assert meta.length == row["length"]
        assert meta.n_channels == row["n_channels"]
        assert meta.channel_names == tuple(row["channel_names"])
        assert meta.question == row["question"] and meta.answer == row["answer"]
        assert [item.channel for item in meta.gold_intervals] == row["injection_channels"]
        assert [item.start for item in meta.gold_intervals] == row["injection_starts"]
        assert [item.end for item in meta.gold_intervals] == row["injection_ends"]
        assert [item.kind for item in meta.gold_intervals] == row["injection_kinds"]

    multi = _meta_by_id(split, "all-000")
    assert multi.gold_intervals == (
        GoldInterval("ch_03", 200, 260, "spike"),
        GoldInterval("ch_09", 640, 700, "drift"),
    )
    assert _meta_by_id(split, "loc-001").gold_intervals == ()
    assert len(split.metas()) == len(split)
    with pytest.raises(IndexError):
        split.meta(len(split))


def test_episode_values_match_parquet_and_stay_lazy(split: AnomalyXLSplit, dataset) -> None:
    _, rows = dataset
    # Nothing bulky is resident: no arrow table, numpy array or tensor is kept.
    assert not any(
        isinstance(value, (pa.Table, np.ndarray, torch.Tensor)) for value in vars(split).values()
    )

    for index in (0, 7, 8, 10):  # spans several row groups, 1/2/16 channels
        row = rows[index]
        episode = split.episode(index)
        assert isinstance(episode, SignalEpisode)
        assert episode.values.dtype == torch.float32
        assert episode.values.shape == (row["n_channels"], row["length"])
        assert episode.channel_ids == tuple(row["channel_names"])
        assert episode.dt == 1.0
        assert float(episode.sample_times[0]) == 0.0
        assert episode.task is split.meta(index)
        expected = torch.tensor(np.asarray(row["channel_values"], dtype=np.float32))
        assert torch.equal(episode.values, expected)

    # Reading the same row twice re-reads the row group and is stable.
    assert torch.equal(split.episode(3).values, split.episode(3).values)


def test_oversized_row_groups_are_streamed(dataset: tuple[Path, list[dict]]) -> None:
    path, rows = dataset
    # max_resident_bytes below every row group forces the batch-streaming path.
    streamed = AnomalyXLSplit(path, "train", expected_rows=len(rows), max_resident_bytes=1)
    direct = AnomalyXLSplit(path, "train", expected_rows=len(rows))
    for index in range(len(rows)):
        left = streamed.channel_values(index)
        right = direct.channel_values(index)
        assert len(left) == len(right)
        assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(left, right))
    assert torch.equal(streamed.episode(9).values, direct.episode(9).values)
    with pytest.raises(ValueError):
        AnomalyXLSplit(path, "train", max_resident_bytes=0)


def test_valid_mask_comes_from_finiteness_only(split: AnomalyXLSplit, dataset) -> None:
    _, rows = dataset
    episode = split.episode(4)
    assert rows[4]["id"] == "cls-001"
    invalid = (~episode.valid[0]).nonzero().squeeze(1).tolist()
    assert invalid == [5, 6, 7]
    assert bool(episode.valid[0, :5].all()) and bool(episode.valid[0, 8:].all())
    # The non-finite samples are NOT imputed at this boundary.
    assert math.isnan(float(episode.values[0, 5]))
    assert float(episode.values[0, 6]) == float("inf")
    assert float(episode.values[0, 7]) == float("-inf")
    # Every other row is fully observed.
    assert bool(split.episode(0).valid.all())


# ----------------------------------------------------------------- zoom tool


def test_zoom_action_to_json_is_exact() -> None:
    action = ZoomAction(channels=("ch_02", "ch_05"), start=12288, end=14336)
    assert action.to_json() == '{"tool":"zoom","channels":["ch_02","ch_05"],"start":12288,"end":14336}'
    assert ZoomAction(("ch_00",), 0, 256).to_json() == (
        '{"tool":"zoom","channels":["ch_00"],"start":0,"end":256}'
    )
    assert action.width == 2048
    with pytest.raises(ValueError):
        ZoomAction(channels=("ch_00", "ch_00"), start=0, end=256)
    with pytest.raises(ValueError):
        ZoomAction(channels=("a", "b", "c"), start=0, end=256)
    with pytest.raises(ValueError):
        ZoomAction(channels=(), start=0, end=256)
    with pytest.raises(ValueError):
        ZoomAction(channels=("ch_00",), start=256, end=256)


def test_parse_zoom_action_round_trips() -> None:
    names = ("ch_00", "ch_01", "ch_02")
    action = ZoomAction(("ch_00", "ch_02"), 1024, 3072)
    parsed = parse_zoom_action(action.to_json(), channel_names=names, length=8192)
    assert parsed == action
    # whitespace around the object is stripped
    assert parse_zoom_action(
        "  " + action.to_json() + "\n", channel_names=names, length=8192
    ) == action
    # boundary widths and end == L are legal
    assert parse_zoom_action(
        ZoomAction(("ch_01",), 0, 256).to_json(), channel_names=names, length=8192
    ) is not None
    assert parse_zoom_action(
        ZoomAction(("ch_01",), 0, 8192).to_json(), channel_names=names, length=8192
    ) is not None
    assert parse_zoom_action(
        ZoomAction(("ch_01",), 7936, 8192).to_json(), channel_names=names, length=8192
    ) is not None


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "   ",
        "not json",
        "[1, 2]",
        '"zoom"',
        'null',
        '{"tool":"zoom","channels":["ch_00"],"start":0,"end":512} trailing',
        'here it is {"tool":"zoom","channels":["ch_00"],"start":0,"end":512}',
        '{"tool":"zoom","channels":["ch_00"],"start":0,"end":512,"why":"extra"}',  # extra key
        '{"tool":"zoom","channels":["ch_00"],"start":0}',  # missing key
        '{"tool":"crop","channels":["ch_00"],"start":0,"end":512}',  # wrong tool
        '{"tool":"zoom","channels":["ch_99"],"start":0,"end":512}',  # unknown channel
        '{"tool":"zoom","channels":["ch_00","ch_00"],"start":0,"end":512}',  # duplicate
        '{"tool":"zoom","channels":["ch_00","ch_01","ch_02"],"start":0,"end":512}',  # 3 channels
        '{"tool":"zoom","channels":[],"start":0,"end":512}',  # 0 channels
        '{"tool":"zoom","channels":"ch_00","start":0,"end":512}',  # not a list
        '{"tool":"zoom","channels":[0],"start":0,"end":512}',  # not a string
        '{"tool":"zoom","channels":["ch_00"],"start":true,"end":512}',  # bool start
        '{"tool":"zoom","channels":["ch_00"],"start":0.0,"end":512}',  # float start
        '{"tool":"zoom","channels":["ch_00"],"start":"0","end":512}',  # string start
        '{"tool":"zoom","channels":["ch_00"],"start":-1,"end":512}',  # negative start
        '{"tool":"zoom","channels":["ch_00"],"start":512,"end":512}',  # empty window
        '{"tool":"zoom","channels":["ch_00"],"start":600,"end":500}',  # reversed
        '{"tool":"zoom","channels":["ch_00"],"start":0,"end":2049}',  # end > L
        '{"tool":"zoom","channels":["ch_00"],"start":0,"end":255}',  # narrower than 256
        '{"tool":"zoom","channels":["ch_00"],"start":0,"end":2048}',  # wider than max_samples
    ],
)
def test_parse_zoom_action_rejects_every_violation(payload: str) -> None:
    assert (
        parse_zoom_action(
            payload,
            channel_names=("ch_00", "ch_01", "ch_02"),
            length=2048,
            min_samples=256,
            max_samples=1024,
        )
        is None
    )


def test_parse_zoom_action_short_series_exception() -> None:
    names = ("ch_00",)
    full = '{"tool":"zoom","channels":["ch_00"],"start":0,"end":128}'
    parsed = parse_zoom_action(full, channel_names=names, length=128)
    assert parsed == ZoomAction(("ch_00",), 0, 128)
    # below 256 samples only the whole series may be requested
    assert parse_zoom_action(
        '{"tool":"zoom","channels":["ch_00"],"start":0,"end":100}', channel_names=names, length=128
    ) is None
    assert parse_zoom_action(
        '{"tool":"zoom","channels":["ch_00"],"start":10,"end":128}', channel_names=names, length=128
    ) is None
    # a 256-sample recording is no longer short: the full series is also a legal width
    assert parse_zoom_action(
        '{"tool":"zoom","channels":["ch_00"],"start":0,"end":256}', channel_names=names, length=256
    ) == ZoomAction(("ch_00",), 0, 256)


def test_parse_zoom_action_argument_guards() -> None:
    with pytest.raises(ValueError):
        parse_zoom_action("{}", channel_names=("ch_00",), length=0)
    with pytest.raises(ValueError):
        parse_zoom_action("{}", channel_names=("ch_00",), length=1024, min_samples=0)
    with pytest.raises(ValueError):
        parse_zoom_action("{}", channel_names=("ch_00",), length=1024, min_samples=512, max_samples=256)


# ------------------------------------------------------------- mode assigner


def test_mode_assigner_proportions_per_category() -> None:
    metas = _synthetic_metas(
        {
            "localize": 100,
            "classify_with_evidence": 100,
            "measure_magnitude": 40,
            "localize_all_channels": 7,
            "lead_lag_with_magnitude": 13,
        }
    )
    mix = {"direct": 0.25, "fixed_crop": 0.25, "agentic": 0.5}
    pairs = ModeAssigner().assign(metas, mix, seed=7, pass_index=0)

    assert len(pairs) == len(metas)
    assert sorted(row for row, _ in pairs) == [meta.row_index for meta in metas]
    assert {mode for _, mode in pairs} <= {"direct", "fixed_crop", "agentic"}

    by_row = dict(pairs)
    for category in {meta.category for meta in metas}:
        rows = [meta.row_index for meta in metas if meta.category == category]
        counts = {mode: 0 for mode in mix}
        for row in rows:
            counts[by_row[row]] += 1
        assert sum(counts.values()) == len(rows)
        for mode, share in mix.items():
            assert abs(counts[mode] - share * len(rows)) <= 1
    # divisible categories land exactly on the mix
    exact = [meta.row_index for meta in metas if meta.category == "localize"]
    assert sum(1 for row in exact if by_row[row] == "agentic") == 50


def test_mode_assigner_p1_mix_and_spec_alias() -> None:
    metas = _synthetic_metas({"localize": 10, "measure_magnitude": 10})
    pairs = ModeAssigner().assign(metas, {"direct": 0.5, "fixed_crop": 0.5}, seed=1, pass_index=0)
    modes = [mode for _, mode in pairs]
    assert modes.count("direct") == modes.count("fixed_crop") == 10
    # prototype.spec.json spells the reading mode "fixed_crop_reading"
    aliased = ModeAssigner().assign(
        metas, {"direct": 0.5, "fixed_crop_reading": 0.5}, seed=1, pass_index=0
    )
    assert aliased == pairs


def test_mode_assigner_p2_mix_and_reading_alias() -> None:
    metas = _synthetic_metas({"localize": 100})
    canonical = ModeAssigner().assign(
        metas, {"direct": 0.25, "fixed_crop": 0.25, "agentic": 0.5}, seed=12, pass_index=0
    )
    aliased = ModeAssigner().assign(
        metas, {"direct": 0.25, "fixed_crop_reading": 0.25, "agentic": 0.5}, seed=12, pass_index=0
    )
    assert aliased == canonical
    counts = {mode: sum(value == mode for _row, value in canonical) for mode in ("direct", "fixed_crop", "agentic")}
    assert counts == {"direct": 25, "fixed_crop": 25, "agentic": 50}


def test_mode_assigner_is_deterministic_and_shuffled() -> None:
    metas = _synthetic_metas({"localize": 60, "measure_magnitude": 60})
    mix = {"direct": 0.5, "fixed_crop": 0.5}
    first = ModeAssigner().assign(metas, mix, seed=3, pass_index=1)
    assert first == ModeAssigner().assign(metas, mix, seed=3, pass_index=1)
    assert first != ModeAssigner().assign(metas, mix, seed=3, pass_index=2)
    assert first != ModeAssigner().assign(metas, mix, seed=4, pass_index=1)
    assert first != ModeAssigner().assign(
        metas, {"direct": 0.25, "fixed_crop": 0.75}, seed=3, pass_index=1
    )
    # the returned order is shuffled, not the natural row order
    assert [row for row, _ in first] != sorted(row for row, _ in first)


def test_mode_assigner_rejects_bad_mixes() -> None:
    metas = _synthetic_metas({"localize": 4})
    for bad in ({}, {"direct": 0.4, "fixed_crop": 0.4}, {"wander": 1.0}, {"direct": -1.0}):
        with pytest.raises(ValueError):
            ModeAssigner().assign(metas, bad, seed=0, pass_index=0)


# ---------------------------------------------------------- teacher crops


def test_teacher_actions_always_parse(split: AnomalyXLSplit) -> None:
    sampler = TeacherCropSampler()
    for meta in split.metas():
        for seed in range(40):
            action = sampler.sample(meta, random.Random(seed))
            assert parse_zoom_action(
                action.to_json(), channel_names=meta.channel_names, length=meta.length
            ) == action


def test_teacher_positive_crop_contains_the_event(split: AnomalyXLSplit) -> None:
    meta = _meta_by_id(split, "loc-000")
    gold = meta.gold_intervals[0]
    sampler = TeacherCropSampler(positive_prob=1.0, border_partial_prob=0.0)
    for seed in range(200):
        action = sampler.sample(meta, random.Random(seed))
        assert action.start <= gold.start and action.end >= gold.end
        assert action.channels == ("ch_00",)


def test_teacher_positive_event_is_not_centred(split: AnomalyXLSplit) -> None:
    meta = _meta_by_id(split, "loc-000")
    gold = meta.gold_intervals[0]
    sampler = TeacherCropSampler(positive_prob=1.0, border_partial_prob=0.0)
    offsets = []
    for seed in range(300):
        action = sampler.sample(meta, random.Random(seed))
        offsets.append((gold.start + gold.end) / 2 - (action.start + action.end) / 2)
    assert len(set(offsets)) > 50
    centred = sum(1 for offset in offsets if abs(offset) <= 1.0)
    assert centred < 0.1 * len(offsets)
    assert min(offsets) < 0 < max(offsets)


def test_teacher_border_crops_cut_the_event(split: AnomalyXLSplit) -> None:
    meta = _meta_by_id(split, "loc-000")
    gold = meta.gold_intervals[0]
    sampler = TeacherCropSampler(
        min_samples=256, max_samples=512, positive_prob=1.0, border_partial_prob=1.0
    )
    cut_head, cut_tail = 0, 0
    for seed in range(200):
        action = sampler.sample(meta, random.Random(seed))
        assert parse_zoom_action(
            action.to_json(),
            channel_names=meta.channel_names,
            length=meta.length,
            min_samples=256,
            max_samples=512,
        ) == action
        overlap = min(action.end, gold.end) - max(action.start, gold.start)
        assert overlap > 0, "a border crop still has to see the event"
        assert action.start > gold.start or action.end < gold.end
        cut_head += action.start > gold.start
        cut_tail += action.end < gold.end
    assert cut_head > 0 and cut_tail > 0


def test_teacher_channel_selection_rules(split: AnomalyXLSplit) -> None:
    sampler = TeacherCropSampler(positive_prob=1.0)

    single = _meta_by_id(split, "loc-000")
    assert {
        sampler.sample(single, random.Random(seed)).channels for seed in range(50)
    } == {("ch_00",)}

    pair = _meta_by_id(split, "lag-000")
    observed = {sampler.sample(pair, random.Random(seed)).channels for seed in range(80)}
    assert ("ch_00", "ch_01") in observed  # synchronized read of both channels
    assert observed & {("ch_00",), ("ch_01",)}  # and single-channel reads
    assert all(set(item) <= {"ch_00", "ch_01"} for item in observed)

    many = _meta_by_id(split, "all-000")
    annotated = {"ch_03", "ch_09"}
    widths, partners = set(), set()
    for seed in range(200):
        action = sampler.sample(many, random.Random(seed))
        assert annotated & set(action.channels), "a positive crop keeps its annotated channel"
        widths.add(len(action.channels))
        if len(action.channels) == 2:
            partners |= set(action.channels) - annotated
    assert widths == {1, 2}
    assert len(partners) > 1  # the second channel is drawn at random, not fixed


def test_teacher_rows_without_gold_get_random_crops(split: AnomalyXLSplit) -> None:
    meta = _meta_by_id(split, "loc-001")
    assert meta.gold_intervals == ()
    sampler = TeacherCropSampler(positive_prob=1.0)  # no annotation to be positive about
    starts = set()
    for seed in range(100):
        action = sampler.sample(meta, random.Random(seed))
        assert parse_zoom_action(
            action.to_json(), channel_names=meta.channel_names, length=meta.length
        ) == action
        starts.add(action.start)
    assert len(starts) > 20


def test_teacher_short_series_returns_the_full_recording(split: AnomalyXLSplit) -> None:
    meta = _meta_by_id(split, "loc-002")
    sampler = TeacherCropSampler()
    for seed in range(20):
        action = sampler.sample(meta, random.Random(seed))
        assert (action.start, action.end) == (0, meta.length)


def test_teacher_is_deterministic_given_the_rng(split: AnomalyXLSplit) -> None:
    sampler = TeacherCropSampler()
    for meta in split.metas():
        assert sampler.sample(meta, random.Random(11)) == sampler.sample(meta, random.Random(11))
    meta = _meta_by_id(split, "all-000")
    drawn = [sampler.sample(meta, random.Random(seed)) for seed in range(30)]
    assert len({(item.channels, item.start, item.end) for item in drawn}) > 1


def test_teacher_mixes_positive_and_negative_crops(split: AnomalyXLSplit) -> None:
    meta = _meta_by_id(split, "loc-000")
    gold = meta.gold_intervals[0]
    sampler = TeacherCropSampler()  # default 0.7 positive
    touching = 0
    for seed in range(400):
        action = sampler.sample(meta, random.Random(seed))
        touching += min(action.end, gold.end) > max(action.start, gold.start)
    assert 0 < touching < 400, "the mix must contain crops without an event"
    assert touching / 400 > 0.7


def test_teacher_action_drives_make_view(split: AnomalyXLSplit) -> None:
    """The sampled coordinates must be accepted verbatim by the signal contract."""

    for row_id in ("loc-000", "lag-000", "all-000"):
        meta = _meta_by_id(split, row_id)
        episode = split.episode(meta.row_index)
        stats = compute_recording_normalization(episode)
        action = TeacherCropSampler().sample(meta, random.Random(2))
        view = make_view(
            episode,
            stats,
            kind=ViewKind.LOCAL,
            start=action.start,
            end=action.end,
            channels=action.channels,
        )
        assert view.channel_ids == action.channels
        assert (view.start, view.end) == (action.start, action.end)
        assert view.values.shape == (len(action.channels), action.width)


# ------------------------------------------------------------ occupancy head


def test_occupancy_targets_partial_bins_and_union() -> None:
    edges = torch.tensor([0, 100, 200, 300, 400], dtype=torch.int64)
    intervals = [
        GoldInterval("ch_00", 50, 150, "spike"),   # 50 in bin 0, 50 in bin 1
        GoldInterval("ch_00", 120, 260, "drift"),  # overlaps the first one
        GoldInterval("ch_01", 0, 400, "other"),    # different channel: ignored
    ]
    targets = occupancy_targets(intervals, "ch_00", edges)
    assert targets.dtype == torch.float32 and targets.shape == (4,)
    # union on ch_00 is [50, 260)
    assert targets.tolist() == pytest.approx([0.5, 1.0, 0.6, 0.0])

    other = occupancy_targets(intervals, "ch_01", edges)
    assert other.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert occupancy_targets(intervals, "ch_07", edges).tolist() == [0.0, 0.0, 0.0, 0.0]
    assert occupancy_targets([], "ch_00", edges).tolist() == [0.0, 0.0, 0.0, 0.0]


def test_occupancy_targets_exact_small_cases() -> None:
    edges = torch.tensor([0, 4, 8, 12], dtype=torch.int64)
    # a single sample inside the second bin
    assert occupancy_targets(
        [GoldInterval("c", 5, 6, "k")], "c", edges
    ).tolist() == pytest.approx([0.0, 0.25, 0.0])
    # two adjacent intervals merge into one contiguous cover
    assert occupancy_targets(
        [GoldInterval("c", 0, 6, "k"), GoldInterval("c", 6, 12, "k")], "c", edges
    ).tolist() == pytest.approx([1.0, 1.0, 1.0])
    # nested intervals are counted once
    assert occupancy_targets(
        [GoldInterval("c", 0, 8, "k"), GoldInterval("c", 2, 4, "k")], "c", edges
    ).tolist() == pytest.approx([1.0, 1.0, 0.0])
    # overlapping intervals are the union, not the sum (0.75, never 1.0)
    assert occupancy_targets(
        [GoldInterval("c", 0, 2, "k"), GoldInterval("c", 1, 3, "k")], "c", edges
    ).tolist() == pytest.approx([0.75, 0.0, 0.0])
    # ... and the union is insensitive to the order the annotations arrive in
    assert occupancy_targets(
        [GoldInterval("c", 1, 3, "k"), GoldInterval("c", 0, 2, "k")], "c", edges
    ).tolist() == pytest.approx([0.75, 0.0, 0.0])
    # an interval reaching past the last edge only counts inside the bins
    assert occupancy_targets(
        [GoldInterval("c", 10, 40, "k")], "c", edges
    ).tolist() == pytest.approx([0.0, 0.0, 0.5])
    # a degenerate annotation contributes nothing
    assert occupancy_targets([GoldInterval("c", 4, 4, "k")], "c", edges).tolist() == [0.0, 0.0, 0.0]


def test_occupancy_targets_matches_a_real_row(split: AnomalyXLSplit) -> None:
    meta = _meta_by_id(split, "all-000")
    edges = torch.arange(0, meta.length + 1, meta.length // 8, dtype=torch.int64)
    targets = occupancy_targets(meta.gold_intervals, "ch_03", edges)
    # gold [200, 260) inside bins of width 128: bin 1 = [128, 256), bin 2 = [256, 384)
    assert targets.tolist() == pytest.approx([0.0, 56 / 128, 4 / 128, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert occupancy_targets(meta.gold_intervals, "ch_00", edges).sum().item() == 0.0


def test_occupancy_targets_rejects_bad_edges() -> None:
    intervals = [GoldInterval("c", 0, 4, "k")]
    with pytest.raises(TypeError):
        occupancy_targets(intervals, "c", [0, 4, 8])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        occupancy_targets(intervals, "c", torch.tensor([0.0, 4.0]))
    with pytest.raises(ValueError):
        occupancy_targets(intervals, "c", torch.tensor([4], dtype=torch.int64))
    with pytest.raises(ValueError):
        occupancy_targets(intervals, "c", torch.tensor([[0, 4]], dtype=torch.int64))
    with pytest.raises(ValueError):
        occupancy_targets(intervals, "c", torch.tensor([0, 8, 4], dtype=torch.int64))


# --------------------------------------------------------- stratified subset


def test_stratified_subset_keeps_category_shares() -> None:
    metas = _synthetic_metas({"a": 100, "b": 50, "c": 50})
    picked = stratified_subset(metas, 40, seed=5)
    assert len(picked) == 40 and len(set(picked)) == 40
    assert picked == sorted(picked)
    by_row = {meta.row_index: meta.category for meta in metas}
    counts = {"a": 0, "b": 0, "c": 0}
    for row in picked:
        counts[by_row[row]] += 1
    assert counts == {"a": 20, "b": 10, "c": 10}
    assert stratified_subset(metas, 40, seed=5) == picked
    assert stratified_subset(metas, 40, seed=6) != picked


def test_stratified_subset_edge_cases(split: AnomalyXLSplit) -> None:
    metas = split.metas()
    assert stratified_subset(metas, 0, seed=1) == []
    assert stratified_subset(metas, len(metas), seed=1) == list(range(len(metas)))
    assert stratified_subset(metas, 1000, seed=1) == list(range(len(metas)))
    picked = stratified_subset(metas, 5, seed=1)
    assert len(picked) == 5
    categories = {split.meta(row).category for row in picked}
    assert len(categories) >= 4  # spread across categories, not one bucket
    with pytest.raises(ValueError):
        stratified_subset(metas, -1, seed=1)


# ------------------------------------------------------------------- scorer


def test_score_precise_smoke(split: AnomalyXLSplit) -> None:
    assert PRECISE_SYSTEM_PROMPT.startswith("You analyze long-context time series")
    meta = _meta_by_id(split, "loc-000")
    correct = json.dumps({"present": True, "start": 300, "end": 340})
    wrong = json.dumps({"present": True, "start": 800, "end": 900})
    good = score_precise(correct, meta.answer)
    bad = score_precise(wrong, meta.answer)
    assert good["category"] == "localize"
    assert good["primary"] == pytest.approx(1.0)
    assert bad["primary"] == 0.0
    assert bad["presence_correct"] == 1.0

    empty = _meta_by_id(split, "loc-001")
    assert score_precise(json.dumps({"present": False}), empty.answer)["primary"] == 1.0
    assert score_precise("no json here", empty.answer)["primary"] == 0.0
