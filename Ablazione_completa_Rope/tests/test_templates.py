"""Conversation assembly checks with the pinned Qwen3.5 tokenizer (skipped when the tokenizer is not cached)."""

from __future__ import annotations

import pytest
import torch

from wavelet_rope.templates import (
    IGNORE_INDEX,
    SEG_ACTION,
    SEG_FINAL,
    SEG_NONE,
    SEG_REASONING,
    ConversationBuilder,
    Turn,
    ViewHeader,
)

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        path = snapshot_download("Qwen/Qwen3.5-4B", revision=REVISION, local_files_only=True,
                                 allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "*config.json", "chat_template*"])
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"pinned tokenizer not available locally: {exc}")


def _header(kind: str, n_tokens: int, *, channels=("ch_00", "ch_01"), start=0, end=4096, obs=0) -> ViewHeader:
    c = len(channels)
    return ViewHeader(kind=kind, channel_ids=tuple(channels), start=start, end=end, recording_length=4096, dt=1.0,
                      tokens_total=n_tokens, tokens_per_channel=tuple([n_tokens // c] * c), median=(0.1,) * c,
                      scale=(1.5,) * c, minimum=(-3.0,) * c, maximum=(4.0,) * c, valid_count=(end - start,) * c,
                      local_median=None if kind == "global" else (0.2,) * c,
                      local_scale=None if kind == "global" else (0.7,) * c,
                      local_available=None if kind == "global" else (True,) * c,
                      observation_index=obs, remaining_calls=5, remaining_channel_views=6)


def test_direct_sequence_supervises_only_the_final_answer(tokenizer) -> None:
    b = ConversationBuilder(tokenizer)
    answer = 'localize:{"_L": 4096, "end": 700, "present": true, "start": 650}'
    turns = [Turn("system", [b.system_prompt("direct", max_calls=6, max_views=8)]),
             Turn("user", b.global_pieces(_header("global", 40), "Where is the anomaly?", 0)),
             Turn("assistant_final", [answer], SEG_FINAL)]
    seq = b.build(turns)
    assert seq.n_signal_tokens == 40
    (view_index, slot, positions), = seq.view_slots
    # slot -1 = the whole view in one time-ordered block; a non-negative slot is one channel (ETH layout)
    assert view_index == 0 and slot == -1 and positions.numel() == 40
    assert bool(seq.is_signal[positions].all()) and int(seq.is_signal.sum()) == 40
    assert bool((seq.input_ids[positions] == b.placeholder_id).all())
    supervised = seq.labels != IGNORE_INDEX
    assert bool((seq.segment[supervised] == SEG_FINAL).all())
    assert bool((seq.segment[~supervised] == SEG_NONE).all())
    # the supervised tokens decode to the answer followed by <|im_end|>
    decoded = tokenizer.decode(seq.labels[supervised].tolist())
    assert decoded == answer + "<|im_end|>"
    # nothing before the assistant turn is supervised, and no placeholder is
    assert not bool(supervised[positions].any())
    assert seq.text.endswith(answer + "<|im_end|>\n")


def test_fixed_crop_sequence_has_action_and_observation(tokenizer) -> None:
    b = ConversationBuilder(tokenizer)
    action = '{"tool":"zoom","channels":["ch_01"],"start":600,"end":1112}'
    turns = [Turn("system", [b.system_prompt("tool", max_calls=6, max_views=8)]),
             Turn("user", b.global_pieces(_header("global", 32), "Q?", 0)),
             Turn("assistant_action", [action], SEG_ACTION),
             Turn("user", b.observation_pieces(_header("local", 24, channels=("ch_01",), start=600, end=1112, obs=1), 1)),
             Turn("assistant_final", ['localize:{"present": false}'], SEG_FINAL)]
    seq_unsup = b.build(turns, supervise_action=False)
    seq_sup = b.build(turns, supervise_action=True)
    assert [v[0] for v in seq_unsup.view_slots] == [0, 1]
    assert seq_unsup.view_slots[1][2].numel() == 24 and seq_unsup.n_signal_tokens == 56
    assert int((seq_unsup.segment == SEG_ACTION).sum()) == 0
    n_action = int((seq_sup.segment == SEG_ACTION).sum())
    assert n_action > 0
    assert tokenizer.decode(seq_sup.labels[seq_sup.segment == SEG_ACTION].tolist()) == action + "<|im_end|>"
    assert "Observation 1: zoom on ch_01 [600, 1112)" in seq_sup.text
    # generation prompt ends with the empty think block and has no labels
    gen = b.build(turns[:-1], generation_prompt=True)
    assert gen.text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert int((gen.labels != IGNORE_INDEX).sum()) == 0


def test_budgets_are_enforced(tokenizer) -> None:
    b = ConversationBuilder(tokenizer, max_signal=100)
    turns = [Turn("system", ["s"]), Turn("user", b.global_pieces(_header("global", 101), "Q?", 0)),
             Turn("assistant_final", ["a"], SEG_FINAL)]
    with pytest.raises(ValueError, match="signal token budget"):
        b.build(turns)
    partial = _header("local", 0, channels=("ch_00",), start=0, end=256, obs=2)
    partial.local_available = (False,)
    pieces = b.observation_pieces(partial, 1)
    assert not any(isinstance(p, tuple) for p in pieces)  # empty channel: no soft-token block
    assert "no valid sample in this window" in "".join(p for p in pieces if isinstance(p, str))


def test_reasoning_action_masks_are_per_turn_and_supervise_the_closure(tokenizer) -> None:
    b = ConversationBuilder(tokenizer)
    turns = [
        Turn("system", ["s"]),
        Turn("user", ["question"]),
        Turn("assistant_reasoning_action", ["look at the evidence", '{"tool":"zoom"}'], supervise=False),
        Turn("user", ["observation without a payload"]),
        Turn("assistant_reasoning_final", ["the evidence is enough", '{"present":false}']),
    ]
    seq = b.build(turns)
    reasoning = seq.segment == SEG_REASONING
    final = seq.segment == SEG_FINAL
    assert int(reasoning.sum()) > 0
    assert int(final.sum()) > 0
    # The unsupervised action turn, including its rationale and closure, is in
    # context but contributes no labels.
    first_assistant = seq.text.index("look at the evidence")
    first_final = seq.text.index('{"present":false}')
    assert first_assistant < first_final
    supervised = seq.labels != IGNORE_INDEX
    assert not bool((seq.segment[supervised] == SEG_ACTION).any())
    assert bool((seq.segment[supervised] == SEG_REASONING).any())
    assert bool((seq.segment[supervised] == SEG_FINAL).any())
    # The supervised reasoning labels contain the literal closing block, while
    # the thinking opener is template text and is never a target.
    decoded_reasoning = tokenizer.decode(seq.labels[reasoning].tolist())
    assert decoded_reasoning.endswith("\n</think>\n\n")
    assert "<think>" not in decoded_reasoning
    unsup_slice = seq.text[first_assistant:first_final]
    assert "</think>" in unsup_slice


def test_thinking_on_generation_prompt_is_open_think_only(tokenizer) -> None:
    b = ConversationBuilder(tokenizer)
    on = b.build([Turn("system", ["s"])], generation_prompt=True, thinking=True)
    off = b.build([Turn("system", ["s"])], generation_prompt=True, thinking=False)
    assert on.text.endswith("<|im_start|>assistant\n<think>\n")
    assert off.text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert int((on.labels != IGNORE_INDEX).sum()) == 0


def test_error_observation_has_no_soft_tokens_and_duplicate_reference(tokenizer) -> None:
    b = ConversationBuilder(tokenizer)
    pieces = b.error_observation_pieces(
        "duplicate zoom action", remaining_calls=3, remaining_channel_views=5, duplicate_of=2
    )
    assert all(isinstance(piece, str) for piece in pieces)
    assert "existing observation 2" in "".join(pieces)
    seq = b.build([Turn("user", pieces)])
    assert seq.n_signal_tokens == 0
    assert not any(isinstance(piece, tuple) for piece in pieces)


def test_reduced_header_drops_min_and_max_but_keeps_median_and_scale() -> None:
    """Astra 16/09, opzione C: min/max answer measure_magnitude on their own, median/scale do not."""

    from wavelet_rope.templates import ConversationBuilder, ViewHeader

    header = ViewHeader(
        kind="global", channel_ids=("ch_00",), start=0, end=64, recording_length=64, dt=1.0,
        tokens_total=2, tokens_per_channel=(2,), median=(0.5,), scale=(2.0,), minimum=(-7.0,),
        maximum=(9.0,), valid_count=(64,),
    )
    full = "".join(p for p in ConversationBuilder.global_pieces(header, "Where?", 0) if isinstance(p, str))
    reduced = "".join(p for p in ConversationBuilder.global_pieces(header, "Where?", 0, reduced_stats=True)
                      if isinstance(p, str))

    assert "min " in full and "max " in full
    assert "min " not in reduced and "max " not in reduced
    for kept in ("median 0.5", "scale 2", "valid 64/64", "2 tokens", "Question: Where?"):
        assert kept in reduced, kept
    # only the extent is gone: nothing else about the channel line changed
    assert reduced == full.replace("min -7, max 9, ", "").replace("min -7.0, max 9.0, ", "")


def test_eth_layout_emits_one_labelled_block_per_channel() -> None:
    """The ETH/OpenTSLM prompt verbatim: label, then that channel's tokens, then the next channel."""

    header = _header("global", 12, channels=("ch_00", "ch_01", "ch_02"))
    pieces = ConversationBuilder.global_pieces(header, "Where?", 0, eth_layout=True)

    text = "".join(p for p in pieces if isinstance(p, str))
    blocks = [p for p in pieces if not isinstance(p, str)]

    # one block per channel, each carrying that channel's own slot and count
    assert blocks == [("view", 0, 0, 4), ("view", 0, 1, 4), ("view", 0, 2, 4)]
    assert text == ("The following is channel ch_00: \n"
                    "The following is channel ch_01: \n"
                    "The following is channel ch_02: \n"
                    "Question: Where?")
    # nothing computed from the signal reaches the text
    for banned in ("median", "scale", "min ", "max ", "valid ", "tokens", "Recording:", "Global view"):
        assert banned not in text, banned


def test_eth_layout_system_prompt_is_the_benchmark_one_verbatim() -> None:
    from wavelet_rope.scoring import PRECISE_SYSTEM_PROMPT

    assert ConversationBuilder.system_prompt(
        "direct", max_calls=6, max_views=8, eth_layout=True) == PRECISE_SYSTEM_PROMPT
    # the agentic prompt still needs its tool instructions
    assert ConversationBuilder.system_prompt(
        "agentic", max_calls=6, max_views=8, eth_layout=True) != PRECISE_SYSTEM_PROMPT


def test_eth_layout_slots_partition_the_view(tokenizer) -> None:
    """Every soft token of the view lands in exactly one block, and blocks keep the channel order."""

    b = ConversationBuilder(tokenizer)
    header = _header("global", 12, channels=("ch_00", "ch_01", "ch_02"))
    seq = b.build([Turn("system", [b.system_prompt("direct", max_calls=6, max_views=8, eth_layout=True)]),
                   Turn("user", b.global_pieces(header, "Where?", 0, eth_layout=True))],
                  generation_prompt=True)

    assert [(v, s, int(p.numel())) for v, s, p in seq.view_slots] == [(0, 0, 4), (0, 1, 4), (0, 2, 4)]
    positions = torch.cat([p for _, _, p in seq.view_slots])
    assert int(seq.is_signal.sum()) == 12
    assert positions.tolist() == sorted(positions.tolist())          # blocks are in channel order
    assert seq.is_signal.nonzero().squeeze(1).tolist() == positions.tolist()


def test_legacy_layout_marks_the_whole_view_with_slot_minus_one() -> None:
    header = _header("global", 12, channels=("ch_00", "ch_01", "ch_02"))
    blocks = [p for p in ConversationBuilder.global_pieces(header, "Where?", 0) if not isinstance(p, str)]
    assert blocks == [("view", 0, -1, 12)]


def test_flat_layout_has_no_chat_template_token(tokenizer) -> None:
    """The benchmark tokenizes one flat string: the only special token is the EOS closing the answer."""

    b = ConversationBuilder(tokenizer)
    header = _header("global", 8, channels=("ch_00", "ch_01"))
    answer = 'localize:{"_L": 4096, "present": false}'
    turns = [Turn("system", [b.system_prompt("direct", max_calls=6, max_views=8, eth_layout=True)]),
             Turn("user", b.global_pieces(header, "Where?", 0, eth_layout=True)),
             Turn("assistant_final", [answer], SEG_FINAL)]
    seq = b.build(turns, flat=True)

    for marker in ("<|im_start|>", "<think>", "</think>"):
        assert marker not in seq.text, marker
    assert seq.text.count("<|im_end|>") == 1              # only the EOS of the answer
    assert seq.text.endswith(answer + "<|im_end|>")
    assert " " + answer in seq.text                        # glued to the question by a single space
    assert seq.text.startswith("You analyze long-context")

    # supervision covers the answer and its EOS, and nothing of the prompt
    supervised = seq.labels != IGNORE_INDEX
    assert tokenizer.decode(seq.input_ids[supervised].tolist()) == " " + answer + "<|im_end|>"


def test_flat_generation_prompt_stops_at_the_question(tokenizer) -> None:
    b = ConversationBuilder(tokenizer)
    header = _header("global", 8, channels=("ch_00", "ch_01"))
    seq = b.build([Turn("system", [b.system_prompt("direct", max_calls=6, max_views=8, eth_layout=True)]),
                   Turn("user", b.global_pieces(header, "Where?", 0, eth_layout=True))],
                  generation_prompt=True, flat=True)
    assert seq.text.endswith("Question: Where?")
    assert "<|im_start|>" not in seq.text and "<|im_end|>" not in seq.text
    assert int((seq.labels != IGNORE_INDEX).sum()) == 0


def test_flat_layout_refuses_the_agentic_turns(tokenizer) -> None:
    b = ConversationBuilder(tokenizer)
    turns = [Turn("system", ["S"]), Turn("assistant_action", ['{"zoom": 1}'], SEG_ACTION)]
    with pytest.raises(ValueError, match="flat layout has no turns"):
        b.build(turns, flat=True)
