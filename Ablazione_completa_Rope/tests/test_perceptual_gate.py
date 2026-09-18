"""CPU tests for the frozen-panel perceptual-content gate."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from wavelet_rope.data import GoldInterval, RowMeta, ZoomAction
from wavelet_rope.model import WRAgentModel
from wavelet_rope.perceptual_gate import (
    PanelCrop,
    aggregate_deltas_by_recording,
    build_panel,
    paired_bootstrap,
    perturb_episode,
    select_substitution_source,
    summarize_gate_results,
)
from wavelet_rope.perceptual_rope import PerceptualMixer, Projector
from wavelet_rope.signal_contract import SignalEpisode, apply_normalization, compute_recording_normalization
from wavelet_rope.templates import ViewHeader
from wavelet_rope.tokenizer import ViewTokens


def _meta(
    row_index: int,
    row_id: str,
    category: str,
    *,
    channels: tuple[str, ...] = ("ch0",),
    gold: tuple[GoldInterval, ...] = (),
) -> RowMeta:
    return RowMeta(
        id=row_id,
        category=category,
        length=4096,
        n_channels=len(channels),
        channel_names=channels,
        question="where?",
        answer='localize:{"present": true, "start": 1000, "end": 1100}',
        gold_intervals=gold,
        split="validation",
        row_index=row_index,
    )


def _episode(meta: RowMeta, *, length: int | None = None) -> SignalEpisode:
    length = meta.length if length is None else length
    values = torch.stack(
        [torch.arange(length, dtype=torch.float32) + float(channel) for channel in range(meta.n_channels)]
    )
    valid = torch.ones_like(values, dtype=torch.bool)
    return SignalEpisode.from_index_grid(
        values=values,
        valid=valid,
        channel_ids=meta.channel_names,
        dt=0.5,
        origin=10.0,
        task=meta,
    )


def _records(*, true: float, substituted: float, zeroed: float, row_id: str = "r0", category: str = "localize", invalid=None):
    return {
        "recording_id": row_id,
        "category": category,
        "primary": {"TRUE": true, "SUBSTITUTED": substituted, "ZEROED": zeroed},
        "invalid_format": invalid or {"TRUE": False, "SUBSTITUTED": False, "ZEROED": False},
    }


def test_panel_is_deterministic_and_respects_event_and_channel_constraints() -> None:
    metas = [
        _meta(0, "positive-single", "localize", gold=(GoldInterval("ch0", 1000, 1100, "spike"),)),
        _meta(
            1,
            "positive-multi",
            "localize_all_channels",
            channels=("ch0", "ch1"),
            gold=(
                GoldInterval("ch0", 700, 800, "spike"),
                GoldInterval("ch1", 1200, 1300, "drift"),
            ),
        ),
        _meta(2, "negative", "localize"),
    ]
    first = build_panel(metas, seed=17, crops_per_row=3)
    second = build_panel(metas, seed=17, crops_per_row=3)
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len(first) == 9
    for crop in first:
        meta = next(item for item in metas if item.id == crop.row_id)
        assert 0 <= crop.start < crop.end <= meta.length
        assert set(crop.channels) <= set(meta.channel_names)
        if meta.n_channels == 2:
            assert len(crop.channels) == 2
        overlaps = [
            interval
            for interval in meta.gold_intervals
            if interval.channel in crop.channels
            and min(crop.end, interval.end) > max(crop.start, interval.start)
        ]
        if crop.crop_kind == "negative":
            assert not overlaps
        else:
            assert overlaps
            # The event is not deliberately centred.  The degenerate case in
            # which geometry makes that impossible is covered by the sampler's
            # explicit fallback, not by this ordinary panel fixture.
            event = overlaps[0]
            assert (2 * crop.start + crop.length - 1) != (event.start + event.end - 1)


def test_substitution_sources_are_same_length_and_gold_free_when_available() -> None:
    meta = _meta(
        0,
        "source",
        "localize",
        gold=(GoldInterval("ch0", 1800, 1900, "spike"),),
    )
    source = select_substitution_source(meta, "ch0", 256, seed=9)
    assert source.end - source.start == 256
    assert 0 <= source.start < source.end <= meta.length
    assert source.gold_free
    assert source.overlap == 0
    assert source.start + 256 <= 1800 or source.start >= 1900

    covered = _meta(
        1,
        "covered",
        "localize",
        gold=(GoldInterval("ch0", 0, 4096, "drift"),),
    )
    fallback = select_substitution_source(covered, "ch0", 512, seed=9)
    assert fallback.end - fallback.start == 512
    assert not fallback.gold_free and fallback.overlap == 512


def test_substitution_and_zeroed_episodes_change_only_the_declared_window() -> None:
    meta = _meta(
        0,
        "perturb",
        "localize",
        channels=("ch0", "ch1"),
        gold=(GoldInterval("ch0", 1500, 1600, "spike"),),
    )
    episode = _episode(meta)
    crop = PanelCrop(
        row_id=meta.id,
        row_index=meta.row_index,
        category=meta.category,
        channels=("ch0",),
        start=1200,
        end=1456,
        crop_kind="negative",
        substitution_sources={"ch0": select_substitution_source(meta, "ch0", 256, seed=2)},
    )
    original_stats = compute_recording_normalization(episode)
    zeroed = perturb_episode(episode, meta, crop, "ZEROED")
    assert torch.equal(zeroed.valid, episode.valid)
    assert torch.equal(zeroed.values[1], episode.values[1])
    assert torch.equal(zeroed.values[0, : crop.start], episode.values[0, : crop.start])
    assert torch.equal(zeroed.values[0, crop.end :], episode.values[0, crop.end :])
    assert torch.all(zeroed.values[0, crop.start : crop.end] == original_stats.median[0])
    normalized_with_true_stats = apply_normalization(
        zeroed.values, zeroed.valid, original_stats
    )
    expected_zero = torch.zeros(crop.length)
    assert torch.allclose(normalized_with_true_stats[0, crop.start : crop.end], expected_zero)

    source = crop.substitution_sources["ch0"]
    substituted = perturb_episode(episode, meta, crop, "SUBSTITUTED")
    assert torch.equal(
        substituted.values[0, crop.start : crop.end], episode.values[0, source.start : source.end]
    )
    assert torch.equal(substituted.values[:, : crop.start], episode.values[:, : crop.start])
    assert torch.equal(substituted.values[:, crop.end :], episode.values[:, crop.end :])


class _FakeFrontend(nn.Module):
    def encode_view(self, view, *, recording_length: int) -> ViewTokens:
        device = view.values.device
        descriptors = torch.zeros((2, 512), dtype=torch.float32, device=device)
        valid = torch.tensor([True, False], dtype=torch.bool, device=device)
        anchor = torch.tensor([float(view.start), 0.0], dtype=torch.float64, device=device)
        zeros_i = torch.zeros(2, dtype=torch.int64, device=device)
        channel = torch.zeros(2, dtype=torch.int64, device=device)
        return ViewTokens(
            descriptors=descriptors,
            valid=valid,
            anchor_time=anchor,
            support_start=zeros_i,
            support_end=zeros_i,
            channel_index=channel,
            family=zeros_i,
            view_kind=view.kind,
            start=view.start,
            end=view.end,
            raw_bin_features=torch.zeros((1, 1, 64), device=device),
            raw_bin_valid=torch.ones((1, 1), dtype=torch.bool, device=device),
            raw_bin_edges=torch.tensor([view.start, view.end], dtype=torch.int64, device=device),
            sort_order=torch.tensor([0, 1], dtype=torch.int64, device=device),
            coefficient_index=zeros_i,
            level=zeros_i,
        )

    def aux_logits(self, tokens: ViewTokens) -> torch.Tensor:
        return torch.zeros_like(tokens.raw_bin_features[..., 0])


def _tiny_model_for_header_test() -> WRAgentModel:
    model = WRAgentModel.__new__(WRAgentModel)
    nn.Module.__init__(model)
    model.max_calls = 6
    model.max_views = 8
    model.frontend = _FakeFrontend()
    model.mixer = PerceptualMixer()
    model.projector = Projector(2560)
    return model


def test_true_headers_are_overridable_but_token_count_mismatch_raises() -> None:
    meta = _meta(0, "header", "localize", gold=())
    episode = _episode(meta)
    action = ZoomAction(("ch0",), 1024, 1280)
    model = _tiny_model_for_header_test()
    true = model.encode_episode(episode, [action])
    assert true.views[0].header.tokens_total == true.views[0].soft.shape[0]
    frozen = replace(true.views[0].header, median=(999.0,))
    overridden = model.encode_episode(episode, [action], header_overrides={0: frozen})
    assert overridden.views[0].header is frozen
    bad = replace(frozen, tokens_total=frozen.tokens_total + 1)
    with pytest.raises(AssertionError):
        model.encode_episode(episode, [action], header_overrides={0: bad})


def test_paired_bootstrap_aggregates_within_id_before_resampling() -> None:
    records = [
        _records(true=1.0, substituted=0.0, zeroed=0.0, row_id="a"),
        _records(true=3.0, substituted=0.0, zeroed=0.0, row_id="a"),
        _records(true=5.0, substituted=0.0, zeroed=0.0, row_id="b"),
    ]
    grouped = aggregate_deltas_by_recording(records)
    assert grouped == {"a": 2.0, "b": 5.0}
    boot = paired_bootstrap(grouped, n_resamples=5000, seed=4)
    assert boot["mean"] == pytest.approx(3.5)
    assert boot["lower"] <= 3.5 <= boot["upper"]


def test_verdict_passes_and_fails_each_gate_criterion() -> None:
    passed = summarize_gate_results(
        [_records(true=1.0, substituted=0.0, zeroed=0.0)], seed=1, n_resamples=1000
    )
    assert passed["verdict"] == "PASS"

    ci_fail = summarize_gate_results(
        [_records(true=0.5, substituted=0.5, zeroed=0.5)], seed=1, n_resamples=1000
    )
    assert ci_fail["verdict"] == "FAIL"
    assert not ci_fail["criteria"]["overall_ci95_lower_gt_zero"]

    collapse_fail = summarize_gate_results(
        [
            _records(true=1.0, substituted=0.0, zeroed=0.0, category="localize"),
            _records(true=0.0, substituted=0.5, zeroed=0.2, category="classify_with_evidence"),
        ],
        seed=1,
        n_resamples=1000,
    )
    assert collapse_fail["verdict"] == "FAIL"
    assert {item["category"] for item in collapse_fail["collapse"]} == {"classify_with_evidence"}

    invalid_fail = summarize_gate_results(
        [_records(true=1.0, substituted=0.0, zeroed=0.0, invalid={"TRUE": True, "SUBSTITUTED": False, "ZEROED": False})],
        seed=1,
        n_resamples=1000,
    )
    assert invalid_fail["verdict"] == "FAIL"
    assert invalid_fail["invalid_failures"] == ["TRUE"]
