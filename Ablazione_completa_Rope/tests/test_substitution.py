"""CPU tests for the train-time O2 signal substitution contract."""

from __future__ import annotations

import random
from types import SimpleNamespace

import torch

from wavelet_rope.data import GoldInterval, RowMeta
from wavelet_rope.signal_contract import SignalEpisode
from wavelet_rope.substitution import (
    NO_COMPATIBLE_TRAIN_DONOR,
    make_substituted_episode,
    mask_gold_intervals,
    prepare_substitution,
    select_substitution_donor,
)
from wavelet_rope.templates import ConversationBuilder, SEG_FINAL, Turn


def _meta(
    row_index: int,
    row_id: str,
    *,
    length: int = 8,
    channels: tuple[str, ...] = ("ch0", "ch1"),
    split: str = "train",
) -> RowMeta:
    return RowMeta(
        id=row_id,
        category="synthetic",
        length=length,
        n_channels=len(channels),
        channel_names=channels,
        question="q",
        answer="receiver answer",
        gold_intervals=(GoldInterval(channels[0], 1, 4, "event"),),
        split=split,
        row_index=row_index,
    )


def _episode(
    meta: RowMeta,
    *,
    value: float,
    dt: float = 0.5,
    valid: torch.Tensor | None = None,
) -> SignalEpisode:
    values = torch.arange(
        meta.n_channels * meta.length, dtype=torch.float32
    ).reshape(meta.n_channels, meta.length) + value
    if valid is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    values = values.masked_fill(~valid, float("nan"))
    return SignalEpisode.from_index_grid(
        values=values,
        valid=valid,
        channel_ids=meta.channel_names,
        dt=dt,
        origin=10.0,
        task=meta,
    )


def test_donor_selection_is_deterministic_and_enforces_all_constraints() -> None:
    receiver_meta = _meta(0, "receiver")
    receiver_valid = torch.ones((2, 8), dtype=torch.bool)
    receiver_valid[0, 2] = False
    receiver = _episode(receiver_meta, value=0.0, valid=receiver_valid)

    wrong_length = _meta(1, "wrong-length", length=9)
    wrong_dt = _meta(2, "wrong-dt")
    wrong_names = _meta(3, "wrong-names", channels=("ch1", "ch0"))
    missing_valid = _meta(4, "missing-valid")
    missing_valid_mask = torch.ones((2, 8), dtype=torch.bool)
    missing_valid_mask[1, 3] = False
    compatible = _meta(5, "compatible")
    validation_row = _meta(6, "validation-row", split="validation")
    metas = [receiver_meta, wrong_length, wrong_dt, wrong_names, missing_valid, compatible, validation_row]
    episodes = {
        0: receiver,
        1: _episode(wrong_length, value=1.0),
        2: _episode(wrong_dt, value=2.0, dt=1.0),
        3: _episode(wrong_names, value=3.0),
        4: _episode(missing_valid, value=4.0, valid=missing_valid_mask),
        5: _episode(compatible, value=5.0),
        6: _episode(validation_row, value=6.0),
    }

    first = select_substitution_donor(
        metas,
        receiver_meta,
        receiver,
        episodes.__getitem__,
        rng=random.Random("42:0:0"),
    )
    second = select_substitution_donor(
        metas,
        receiver_meta,
        receiver,
        episodes.__getitem__,
        rng=random.Random("42:0:0"),
    )

    assert first is not None and second is not None
    assert first.meta.id == second.meta.id == "compatible"
    assert first.meta.row_index != receiver_meta.row_index
    assert first.meta.split == "train"
    assert first.meta.length == receiver_meta.length
    assert first.meta.channel_names == receiver_meta.channel_names
    assert first.episode.dt == receiver.dt
    assert bool((~receiver.valid | first.episode.valid).all())


def test_substituted_episode_uses_donor_values_and_receiver_geometry() -> None:
    receiver_meta = _meta(0, "receiver")
    donor_meta = _meta(1, "donor")
    receiver_valid = torch.ones((2, 8), dtype=torch.bool)
    receiver_valid[0, 1] = False
    receiver_valid[1, 6:] = False
    receiver = _episode(receiver_meta, value=0.0, valid=receiver_valid)
    donor = _episode(donor_meta, value=100.0, dt=0.5)

    substituted = make_substituted_episode(receiver, donor)

    assert torch.equal(substituted.values, donor.values)
    assert torch.equal(substituted.valid, receiver.valid)
    assert torch.equal(substituted.sample_times, receiver.sample_times)
    assert substituted.channel_ids == receiver.channel_ids
    assert substituted.task is receiver.task


def test_zero_share_does_not_load_or_choose_a_donor() -> None:
    receiver_meta = _meta(0, "receiver")
    receiver = _episode(receiver_meta, value=0.0)

    def fail_if_loaded(_index: int) -> SignalEpisode:
        raise AssertionError("substitution_share=0 must not load a donor")

    decision = prepare_substitution(
        0.0,
        metas=[receiver_meta],
        receiver_meta=receiver_meta,
        receiver=receiver,
        episode_getter=fail_if_loaded,
        rng=random.Random("42:0:0"),
    )

    assert not decision.requested
    assert not decision.substituted
    assert decision.donor_id is None
    assert decision.reason is None


def test_requested_substitution_logs_when_no_compatible_train_donor_exists() -> None:
    receiver_meta = _meta(0, "receiver")
    receiver = _episode(receiver_meta, value=0.0)
    decision = prepare_substitution(
        1.0,
        metas=[receiver_meta],
        receiver_meta=receiver_meta,
        receiver=receiver,
        episode_getter=lambda index: receiver,
        rng=random.Random("42:0:0"),
    )

    assert decision.requested
    assert not decision.substituted
    assert decision.donor_id is None
    assert decision.reason == NO_COMPATIBLE_TRAIN_DONOR


def test_donor_gold_is_masked_before_auxiliary_targets_are_built() -> None:
    meta = _meta(0, "receiver", channels=("ch0",))
    valid = torch.ones((1, 8), dtype=torch.bool)
    valid[0, 2:4] = False
    receiver = _episode(meta, value=0.0, valid=valid)

    masked = mask_gold_intervals((GoldInterval("ch0", 1, 7, "event"),), receiver)

    assert masked == (
        GoldInterval("ch0", 1, 2, "event"),
        GoldInterval("ch0", 4, 7, "event"),
    )


def test_realised_fraction_is_close_to_the_declared_quarter() -> None:
    count = 400
    metas = [_meta(index, f"row-{index}", channels=("ch0",)) for index in range(count)]
    episodes = {
        index: _episode(metas[index], value=float(index), valid=torch.ones((1, 8), dtype=torch.bool))
        for index in range(count)
    }
    realised = 0
    for index, meta in enumerate(metas):
        decision = prepare_substitution(
            0.25,
            metas=metas,
            receiver_meta=meta,
            receiver=episodes[index],
            episode_getter=episodes.__getitem__,
            rng=random.Random(f"42:0:{index}"),
        )
        realised += int(decision.substituted)

    fraction = realised / count
    assert 0.18 <= fraction <= 0.32


class _FakeTokenizer:
    def __init__(self) -> None:
        self._ids = {"<|image_pad|>": 999, "<|im_end|>": 1000}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids.get(token, 1001)

    def __call__(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=[2000 + index for index, _ in enumerate(text)])

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        rendered = "".join(
            f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>\n" for item in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return rendered


def test_substituted_example_has_reading_targets_but_no_final_targets() -> None:
    builder = ConversationBuilder(_FakeTokenizer())
    sequence = builder.build([
        Turn(
            "assistant_reasoning_final",
            ["donor reading", "receiver final answer"],
            SEG_FINAL,
            supervise_final=False,
        )
    ])

    assert bool((sequence.labels != -100).any())
    final_positions = sequence.segment == SEG_FINAL
    assert not bool((sequence.labels[final_positions] != -100).any())
