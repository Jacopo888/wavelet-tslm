"""CPU tests for the oracle-text probe helpers and prompt insertion."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tools.oracle_text_probe import (
    derangement_indices,
    derangement_offset,
    final_field_logit_positions,
    final_field_positions,
)
from wavelet_rope.data import RowMeta
from wavelet_rope.model import EncodedView, EpisodeEncoding, WRAgentModel
from wavelet_rope.perceptual_rope import PerceptualMemory
from wavelet_rope.templates import SEG_FINAL, SEG_REASONING, ConversationBuilder, PreTokenized, ViewHeader


class _FakeTokenizer:
    def __init__(self) -> None:
        self._ids = {"<|image_pad|>": 999, "<|im_end|>": 1000}
        self.eos_token_id = 1000
        self.decoded = {}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids.get(token, 1001)

    def __call__(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=[2000 + index for index, _ in enumerate(text)])

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self.decoded.get(int(token_id), "") for token_id in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        rendered = "".join(
            f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>\n" for item in messages
        )
        if add_generation_prompt:
            rendered += (
                "<|im_start|>assistant\n<think>\n"
                if enable_thinking
                else "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            )
        return rendered


def _meta() -> RowMeta:
    return RowMeta(
        id="oracle-row",
        category="localize",
        length=64,
        n_channels=1,
        channel_names=("ch_00",),
        question="Where?",
        answer='localize:{"_L":64,"end":34,"present":true,"start":12}',
        gold_intervals=(),
        split="validation",
        row_index=0,
    )


def _fake_model_and_encoding() -> tuple[WRAgentModel, RowMeta, EpisodeEncoding]:
    model = WRAgentModel.__new__(WRAgentModel)
    torch.nn.Module.__init__(model)
    model.builder = ConversationBuilder(_FakeTokenizer())
    model.max_calls = 3
    model.max_views = 3
    header = ViewHeader(
        kind="global",
        channel_ids=("ch_00",),
        start=0,
        end=64,
        recording_length=64,
        dt=1.0,
        tokens_total=2,
        tokens_per_channel=(2,),
        median=(0.0,),
        scale=(1.0,),
        minimum=(-1.0,),
        maximum=(1.0,),
        valid_count=(64,),
        reading_bins=4,
    )
    view = EncodedView(
        kind="global",
        channel_ids=("ch_00",),
        start=0,
        end=64,
        soft=torch.zeros((2, 4)),
        tau=torch.zeros(2),
        header=header,
        aux_logits=torch.zeros((1, 1)),
        aux_valid=torch.ones((1, 1), dtype=torch.bool),
        aux_edges=torch.tensor([0, 64]),
        n_tokens_per_channel=(2,),
    )
    memory = PerceptualMemory(t_origin=0.0, dt_ref=1.0)
    return model, _meta(), EpisodeEncoding(views=[view], memory=memory, n_signal_tokens=2)


def test_derangement_is_fixed_shift_without_fixed_points() -> None:
    mapping = derangement_indices(17, seed=31)
    assert mapping == derangement_indices(17, seed=31)
    assert all(index != target for index, target in enumerate(mapping))
    assert sorted(mapping) == list(range(17))
    k = derangement_offset(17, seed=31)
    assert 1 <= k <= 16
    assert mapping == [(index + k) % 17 for index in range(17)]


def test_build_sequence_inserts_oracle_once_before_signal_and_none_is_unchanged() -> None:
    model, meta, encoding = _fake_model_and_encoding()
    oracle = "top ch_00:3,2,1; peak ch_00:3."
    with_oracle = model.build_sequence(
        meta,
        encoding,
        mode="direct",
        answer=meta.answer,
        supervise_action=False,
        oracle_text=oracle,
    )
    without_oracle = model.build_sequence(
        meta,
        encoding,
        mode="direct",
        answer=meta.answer,
        supervise_action=False,
    )
    without_oracle_again = model.build_sequence(
        meta,
        encoding,
        mode="direct",
        answer=meta.answer,
        supervise_action=False,
        oracle_text=None,
    )

    oracle_line = f"Reading: {oracle}\n"
    grid = "Reading grid: 4 bins of 16 samples from 0; bin b covers [0+16*b, 0+16*(b+1)).\n"
    signal_marker = "<view 0.-1:2>"   # -1 = the whole view in one block (a channel slot would be >= 0)
    assert with_oracle.text.count(oracle_line) == 1
    assert with_oracle.text.index(grid) < with_oracle.text.index(oracle_line)
    assert with_oracle.text.index(oracle_line) < with_oracle.text.index(signal_marker)
    assert without_oracle.text == without_oracle_again.text
    assert without_oracle.input_ids.tolist() == without_oracle_again.input_ids.tolist()
    assert without_oracle.labels.tolist() == without_oracle_again.labels.tolist()
    assert without_oracle.segment.tolist() == without_oracle_again.segment.tolist()
    assert without_oracle.is_signal.tolist() == without_oracle_again.is_signal.tolist()


def test_final_field_selection_only_keeps_digit_tokens_in_supervised_final_segment() -> None:
    tokenizer = _FakeTokenizer()
    tokenizer.decoded.update({10: "prefix", 11: "7", 12: "suffix", 13: "12", 14: "9", 15: "8"})
    sequence = SimpleNamespace(
        input_ids=torch.tensor([99, 10, 11, 12, 13, 14, 15]),
        labels=torch.tensor([-100, 10, 11, 12, 13, -100, 15]),
        segment=torch.tensor([0, SEG_FINAL, SEG_FINAL, SEG_FINAL, SEG_FINAL, SEG_FINAL, 0]),
    )

    assert final_field_positions(sequence, tokenizer).tolist() == [2, 4]
    assert final_field_logit_positions(sequence, tokenizer).tolist() == [1, 3]


def test_derangement_rejects_single_row() -> None:
    with pytest.raises(ValueError, match="at least two"):
        derangement_indices(1, seed=1)


def _sequence_with_replayed_reading(reasoning_ids):
    model, meta, encoding = _fake_model_and_encoding()
    return model.build_sequence(
        meta,
        encoding,
        mode="direct",
        answer=meta.answer,
        supervise_action=False,
        reading=PreTokenized(reasoning_ids),
    )


def test_replayed_reading_is_spliced_verbatim_without_a_second_closure() -> None:
    replayed = [501, 502, 503, 504]
    sequence = _sequence_with_replayed_reading(replayed)

    ids = sequence.input_ids.tolist()
    segments = sequence.segment.tolist()
    labels = sequence.labels.tolist()
    reasoning_positions = [index for index, seg in enumerate(segments) if seg == SEG_REASONING]

    # exactly the generated ids, in order, and nothing appended after them
    assert [ids[index] for index in reasoning_positions] == replayed
    assert reasoning_positions == list(range(reasoning_positions[0], reasoning_positions[0] + len(replayed)))
    assert [labels[index] for index in reasoning_positions] == replayed


def test_replayed_reading_never_carries_a_teacher_reading_into_the_assistant_turn() -> None:
    teacher = "top ch_00:3,2,1; peak ch_00:3."
    sequence = _sequence_with_replayed_reading([601, 602])
    assert teacher not in sequence.text
    # the opener is the builder's, the content is only what was generated
    assert sequence.text.count("<think>") == 1


def test_digits_of_the_replayed_reading_are_not_cross_entropy_targets() -> None:
    model, meta, encoding = _fake_model_and_encoding()
    tokenizer = model.builder.tok
    replayed = [701, 702]
    tokenizer.decoded.update({701: "62", 702: "61"})  # digits, but they live in the reasoning segment
    sequence = model.build_sequence(
        meta, encoding, mode="direct", answer=meta.answer,
        supervise_action=False, reading=PreTokenized(replayed),
    )

    assert final_field_positions(sequence, tokenizer).tolist() == []

    segments = sequence.segment.tolist()
    final_position = next(index for index, seg in enumerate(segments) if seg == SEG_FINAL)
    tokenizer.decoded[int(sequence.input_ids[final_position])] = "34"
    selected = final_field_positions(sequence, tokenizer).tolist()
    assert final_position in selected
    assert all(segments[index] == SEG_FINAL for index in selected)
