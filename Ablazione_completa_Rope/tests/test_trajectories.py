"""Backbone-free P2 selector, kind assignment, and trajectory teacher checks."""

from __future__ import annotations

import random

import pytest
import torch

from wavelet_rope.data import GoldInterval, RowMeta, parse_zoom_action
from wavelet_rope.haar_selector import blind_candidates, gold_free_windows, window_containing_point
from wavelet_rope.signal_contract import SignalEpisode
from wavelet_rope.templates import ViewRef
from wavelet_rope.trajectories import (
    TRAJECTORY_KINDS,
    TrajectoryKindAssigner,
    TrajectoryTeacher,
    compatible_kinds,
)


def _fixture_meta(n_channels: int = 16, row_index: int = 0) -> RowMeta:
    names = tuple(f"ch_{index:02d}" for index in range(n_channels))
    gold = (
        GoldInterval("ch_03", 300, 340, "spike"),
        GoldInterval("ch_09", 700, 750, "drift"),
    )
    answer = (
        'localize_all_channels:{"_L": 1024, "anomalies": '
        '[{"channel":"ch_03","start":300,"end":340},'
        '{"channel":"ch_09","start":700,"end":750}]}'
    )
    return RowMeta(
        id=f"synthetic-{row_index}", category="localize_all_channels", length=1024,
        n_channels=n_channels, channel_names=names, question="Locate the events.",
        answer=answer, gold_intervals=gold, split="train", row_index=row_index,
    )


def _fixture_episode(meta: RowMeta) -> SignalEpisode:
    values = torch.zeros((meta.n_channels, meta.length), dtype=torch.float32)
    values += torch.arange(meta.length, dtype=torch.float32)[None] * 0.001
    values[3, 320] = 25.0
    values[9, 720:735] += 8.0
    return SignalEpisode.from_index_grid(
        values=values, valid=torch.ones_like(values, dtype=torch.bool),
        channel_ids=meta.channel_names, dt=1.0,
    )


def test_blind_candidates_rank_across_channels_and_are_deterministic() -> None:
    values = torch.zeros((2, 512), dtype=torch.float32)
    values[1, 257] = 100.0
    first = blind_candidates(values, width=64, count=8)
    assert first == blind_candidates(values, width=64, count=8)
    assert first and first[0][0] == 1
    assert all(0 <= channel < 2 and 0 <= centre < 512 for channel, centre in first)


def test_gold_free_filter_uses_half_open_overlap_exactly() -> None:
    candidates = [(0, 10), (0, 100), (1, 200)]
    gold = [GoldInterval("ch_00", 90, 110, "x"), GoldInterval("ch_01", 190, 210, "y")]
    assert gold_free_windows(candidates, gold, ("ch_00", "ch_01")) == [(0, 10)]
    assert gold_free_windows(candidates, gold, ("ch_00", "ch_01"), width=20) == [(0, 10)]
    # [110,130) only touches [90,110), so it is gold-free under [start,end).
    assert gold_free_windows([(0, 120)], gold, ("ch_00",), width=20) == [(0, 120)]


def test_containing_window_sampler_is_uniformly_in_bounds() -> None:
    rng = random.Random(7)
    windows = [window_containing_point(100, 37, 20, rng) for _ in range(100)]
    assert all(0 <= start < end <= 100 and start <= 37 < end for start, end in windows)
    assert len(set(windows)) > 1
    with pytest.raises(ValueError):
        window_containing_point(100, 100, 20, random.Random(0))


def test_teacher_realises_each_kind_and_keeps_injected_error_unsupervised() -> None:
    meta = _fixture_meta()
    episode = _fixture_episode(meta)
    for kind in TRAJECTORY_KINDS:
        teacher = TrajectoryTeacher(kind=kind, max_reasoning_tokens=384)
        trajectory = teacher.build(meta, episode, random.Random(11))
        assert trajectory.kind == kind
        assert trajectory.turns[0].role == "system"
        assert any(isinstance(piece, ViewRef) for turn in trajectory.turns for piece in turn.pieces)
        for action in trajectory.actions:
            assert parse_zoom_action(
                action.to_json(), channel_names=meta.channel_names, length=meta.length
            ) == action
        for turn in trajectory.turns:
            if turn.role.startswith("assistant_reasoning"):
                assert len(str(turn.pieces[0]).split()) * 1.5 <= 384

    wrong = TrajectoryTeacher(kind="wrong_then_recover").build(meta, episode, random.Random(5))
    assert wrong.injected_error and wrong.injected_error_action == wrong.actions[0]
    assert not wrong.turns[2].supervise
    wrong_action = wrong.actions[0]
    assert all(
        not (item.channel in wrong_action.channels and max(item.start, wrong_action.start) < min(item.end, wrong_action.end))
        for item in meta.gold_intervals
    )
    assert wrong.actions[-1] != wrong.actions[0]


def test_kind_assigner_reports_realised_stratified_distribution() -> None:
    metas = [_fixture_meta(row_index=index) for index in range(70)]
    assigner = TrajectoryKindAssigner()
    assignments = assigner.assign(metas, seed=123, pass_index=1)
    report = assigner.report(assignments)
    assert len(assignments) == len(metas)
    assert report["total"] == len(metas)
    assert set(report["counts"]) == set(TRAJECTORY_KINDS)
    assert sum(report["counts"].values()) == len(metas)
    assert report == assigner.last_report
    assert set(compatible_kinds(metas[0])) >= {"final_no_tool", "two_channels", "multi_event"}


def test_coarse_to_fine_width_scales_with_the_recording() -> None:
    """Astra 16/09: w0 = clip(L/16, 1024, 8192), then a quarter per refinement, floor 256.

    The log-uniform rule the Z1-bis traces used ignores L, which is why containment fell from 80% at L=16.384 to
    32% at L=131.072.  This pins the replacement and leaves the old rule as the default.
    """

    import random

    from wavelet_rope.trajectories import TrajectoryTeacher

    teacher = TrajectoryTeacher(width_rule="coarse_to_fine")
    rng = random.Random(0)

    assert teacher._width(131072, rng) == 8192          # L/16 = 8192, at the cap
    assert teacher._width(32768, rng) == 2048           # L/16
    assert teacher._width(16384, rng) == 1024           # L/16 = 1024, at the floor
    assert teacher._width(8192, rng) == 1024            # L/16 = 512 -> raised to the floor

    # refinements divide by four and stop at 256
    assert teacher._width(131072, rng, level=1) == 2048
    assert teacher._width(131072, rng, level=2) == 512
    assert teacher._width(131072, rng, level=3) == 256
    assert teacher._width(131072, rng, level=9) == 256

    # the rule is deterministic: the generator is never consulted
    before = rng.random()
    rng2 = random.Random(0)
    rng2.random()
    assert teacher._width(65536, rng) == teacher._width(65536, rng2)
    assert before == random.Random(0).random()

    # default behaviour is unchanged and does depend on the draw
    legacy = TrajectoryTeacher()
    widths = {legacy._width(131072, random.Random(seed)) for seed in range(12)}
    assert len(widths) > 1
    assert all(256 <= w <= 8192 for w in widths)
