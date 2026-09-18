"""Conversation assembly for the direct and tool modes (DESIGN §9) with token-level segments.

The sequence is built piece by piece from the literal Qwen3.5 chat template verified on the node::

    <|im_start|>system\\n{system}<|im_end|>\\n
    <|im_start|>user\\n{user}<|im_end|>\\n
    <|im_start|>assistant\\n{action}<|im_end|>\\n            (legacy fixed-crop history turn)
    <|im_start|>assistant\\n<think>\\n{reasoning}\\n</think>\\n\\n{action-or-final}<|im_end|>\\n
    <|im_start|>user\\n{observation}<|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n{final}<|im_end|>\\n   (thinking disabled: empty think block)

Signal soft tokens occupy ``<|image_pad|>`` placeholder positions (one existing special token id, never a
new vocabulary entry); the model replaces those embeddings and marks them ``m=1`` with their ``tau``.
Every other token is text (``m=0``).  Labels cover only assistant content (plus its ``<|im_end|>``); the
segment id tells the trainer which weight applies (reasoning 0.2, action 1.0, final 1.0 in P2; the fixed
teacher action of P1 is not supervised).  Observations and prompts are never predicted.

DESIGN 6.3: the valid tokens of an observation are ordered by (time, channel, family, level, index), so the
channels of a view are interleaved in time inside ONE placeholder block; channel identity lives in the
token embedding and the per-channel numeric metadata are stated in the header text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from .scoring import PRECISE_SYSTEM_PROMPT

SEG_NONE, SEG_REASONING, SEG_ACTION, SEG_FINAL = 0, 1, 2, 3
PLACEHOLDER_TOKEN = "<|image_pad|>"
IGNORE_INDEX = -100

TOOL_INSTRUCTIONS = (
    "You may inspect the signal at higher resolution before answering. To do so, reply with exactly one JSON "
    'object on a single line: {"tool":"zoom","channels":["<channel>"],"start":<int>,"end":<int>} '
    "(one or two listed channels, absolute sample coordinates [start,end), 256 to 8192 samples). "
    "You will receive a local view of that window. You may zoom several times; when the evidence is sufficient, "
    "reply with the final answer in the requested JSON schema. Budget: {calls} zoom calls and {views} channel views."
)
DIRECT_INSTRUCTIONS = "No tool is available: answer directly from the global view."


def fmt(x: float) -> str:
    return f"{float(x):.4g}"


def reading_grid_line(header: "ViewHeader") -> str | None:
    """Grid and offset of the reading (Astra 15/09 D2): bin b covers [start + w*b, start + w*(b+1))."""

    if header.reading_bins is None:
        return None
    n = int(header.reading_bins)
    width = max(1, -(-(header.end - header.start) // n))
    return (f"Reading grid: {n} bins of {width} samples from {header.start}; bin b covers "
            f"[{header.start}+{width}*b, {header.start}+{width}*(b+1)).\n")


@dataclass
class ViewHeader:
    """Everything the text header may say about a view; computed from the signal only, never from gold."""

    kind: str  # "global" | "local"
    channel_ids: tuple[str, ...]
    start: int
    end: int
    recording_length: int
    dt: float
    tokens_total: int  # valid soft tokens of the view, inserted as ONE block ordered by (time, channel, ...)
    tokens_per_channel: tuple[int, ...]  # informative per-channel counts (stated in the header)
    median: tuple[float, ...]
    scale: tuple[float, ...]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    valid_count: tuple[int, ...]
    local_median: tuple[float, ...] | None = None
    local_scale: tuple[float, ...] | None = None
    local_available: tuple[bool, ...] | None = None
    observation_index: int = 0
    remaining_calls: int = 0
    remaining_channel_views: int = 0
    reading_bins: int | None = None  # Astra 15/09 D2: bins of the reading grid declared in the prompt (None = no line)


@dataclass(frozen=True)
class PreTokenized:
    """Reasoning content supplied as token ids instead of text.

    The oracle probe (Astra 16/09) replays the reading the model generated
    itself, so the CE sequence must carry exactly those ids -- including the
    closing think delimiter, natural or forced -- and not a re-tokenisation
    of their decoded form.
    """

    ids: tuple = ()

    def __init__(self, ids) -> None:
        object.__setattr__(self, "ids", tuple(int(i) for i in ids))


@dataclass(frozen=True)
class ViewRef:
    """A view placeholder used by the programmatic trajectory teacher.

    The teacher does not own the learned frontend and therefore cannot know how
    many of a view's fixed slots survive validity/salience selection.  A
    :class:`ViewRef` is resolved by ``WRAgentModel.build_from_turns`` against
    the corresponding :class:`EncodedView`; ordinary callers can continue to
    use the concrete ``("view", index, slot, n_tokens)`` tuples directly.
    """

    view_index: int
    kind: str = "observation"  # global | observation
    question: str | None = None


@dataclass
class Turn:
    role: str  # system | user | assistant_action | assistant_final | assistant_reasoning_action | assistant_reasoning_final
    pieces: list[Any]  # str, ViewRef, or ("view", view_index, channel_slot, n_tokens)
    segment: int = SEG_NONE
    supervise: bool = True
    # Reasoning and final content normally share ``supervise``.  Substitution
    # keeps the reading target but must remove the receiver's final answer
    # from the target tensor, so the two parts can be controlled separately.
    supervise_reasoning: bool | None = None
    supervise_final: bool | None = None


@dataclass
class BuiltSequence:
    input_ids: Tensor  # [S] int64
    labels: Tensor  # [S] int64, IGNORE_INDEX outside supervised content
    segment: Tensor  # [S] int64
    is_signal: Tensor  # [S] bool
    view_slots: list[tuple[int, int, Tensor]]  # (view_index, channel_slot, positions[int64]) in insertion order
    n_text_tokens: int
    n_signal_tokens: int
    n_assistant_tokens: int
    text: str

    @property
    def length(self) -> int:
        return int(self.input_ids.shape[0])


class ConversationBuilder:
    def __init__(self, tokenizer, *, max_context: int = 16384, max_text: int = 4096, max_signal: int = 8192,
                 max_assistant: int = 4096) -> None:
        self.tok = tokenizer
        self.placeholder_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)
        if self.placeholder_id is None or self.placeholder_id < 0:
            raise ValueError("tokenizer has no placeholder token")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.max_context, self.max_text, self.max_signal, self.max_assistant = (
            max_context, max_text, max_signal, max_assistant)
        # verify the literal template pieces against the tokenizer's own rendering once
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
             {"role": "assistant", "content": "A"}, {"role": "user", "content": "O"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        expected = ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\nA<|im_end|>\n"
                    "<|im_start|>user\nO<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        if rendered != expected:
            raise RuntimeError(f"unexpected chat template rendering: {rendered!r}")

    # ----------------------------------------------------------------- text pieces
    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False).input_ids

    @staticmethod
    def system_prompt(mode: str, *, max_calls: int, max_views: int, eth_layout: bool = False) -> str:
        if eth_layout and mode == "direct":
            # The benchmark's own system prompt and nothing else: no sentence about views or tools,
            # because in the ETH layout there is no "global view" to refer to.
            return PRECISE_SYSTEM_PROMPT
        if mode == "direct":
            return PRECISE_SYSTEM_PROMPT + " " + DIRECT_INSTRUCTIONS
        return (PRECISE_SYSTEM_PROMPT + " "
                + TOOL_INSTRUCTIONS.replace("{calls}", str(max_calls)).replace("{views}", str(max_views)))

    @staticmethod
    def global_pieces(
        header: ViewHeader,
        question: str,
        view_index: int,
        *,
        oracle_text: str | None = None,
        reduced_stats: bool = False,
        raw_stats: bool = False,
        eth_layout: bool = False,
    ) -> list[Any]:
        # ``reduced_stats`` drops min and max (Astra 16/09, opzione C): on 120 validation rows the peak
        # deviation from the median over the robust scale reproduces measure_magnitude's magnitude_sigma with a
        # median ratio of 1.16, so those two numbers answer one category out of five without reading a signal
        # token.  Median and scale stay: they are the normalisation the soft tokens are expressed in.
        if eth_layout:
            # AnomalyXL/OpenTSLM verbatim (verified 17/09 against qa_dataset.py:130-139 and chatts.py:324-366):
            # one labelled block per channel, its tokens right after the label, and no statistic whatsoever.
            # The only numbers the benchmark gives are L and C, and both already travel inside the question.
            eth: list[Any] = []
            for slot, ch in enumerate(header.channel_ids):
                n = int(header.tokens_per_channel[slot])
                if n <= 0:
                    continue
                eth.append(f"The following is channel {ch}: ")
                eth.append(("view", view_index, slot, n))
                eth.append(chr(10))
            if oracle_text is not None:
                eth.append(f"Reading: {oracle_text}" + chr(10))
            eth.append(f"Question: {question}")
            return eth
        pieces: list[Any] = [
            f"Recording: L={header.recording_length} samples (dt={fmt(header.dt)}), "
            f"channels: {', '.join(header.channel_ids)}.\n"
        ]
        for slot, ch in enumerate(header.channel_ids):
            extent = ("" if reduced_stats else
                      f"min {fmt(header.minimum[slot])}, max {fmt(header.maximum[slot])}, ")
            if raw_stats:
                # Astra/Jacopo 17/09: no number computed from the signal reaches the text.  The five per-token meta
                # features already carry position, log scale and valid fraction, so the header was redundancy -- and
                # for Qwen a digit in the prompt is far cheaper than a scalar inside an embedding, which is why any
                # written statistic becomes the preferred route.  This restores va1's input condition.
                pieces.append(f"Channel {ch}: {header.tokens_per_channel[slot]} tokens.\n")
            else:
                pieces.append(
                f"Channel {ch}: median {fmt(header.median[slot])}, scale {fmt(header.scale[slot])}, "
                f"{extent}"
                f"valid {header.valid_count[slot]}/{header.recording_length}, "
                f"{header.tokens_per_channel[slot]} tokens.\n"
                )
        grid = reading_grid_line(header)
        if grid is not None and not raw_stats:
            pieces.append(grid)
        if oracle_text is not None:
            pieces.append(f"Reading: {oracle_text}\n")
        pieces.append(f"Global view, {header.tokens_total} signal tokens ordered by time:\n")
        pieces.append(("view", view_index, -1, header.tokens_total))
        pieces.append(f"\nQuestion: {question}")
        return pieces

    @staticmethod
    def observation_pieces(header: ViewHeader, view_index: int) -> list[Any]:
        pieces: list[Any] = [
            f"Observation {header.observation_index}: zoom on {', '.join(header.channel_ids)} "
            f"[{header.start}, {header.end}) of {header.recording_length} (dt={fmt(header.dt)}), "
            f"{header.end - header.start} samples; remaining calls {header.remaining_calls}, "
            f"remaining channel views {header.remaining_channel_views}.\n"
        ]
        grid = reading_grid_line(header)  # observation grid (Astra 15/09 D2)
        if grid is not None:
            pieces.append(grid)
        for slot, ch in enumerate(header.channel_ids):
            valid = header.valid_count[slot]
            if header.local_available is not None and not header.local_available[slot]:
                pieces.append(
                    f"Channel {ch}: valid {valid}/{header.end - header.start}; no valid sample in this window, "
                    f"no local view (global median {fmt(header.median[slot])}, scale {fmt(header.scale[slot])}).\n")
                continue
            pieces.append(
                f"Channel {ch}: valid {valid}/{header.end - header.start}; normalization global "
                f"(median {fmt(header.median[slot])}, scale {fmt(header.scale[slot])}), local "
                f"(median {fmt(header.local_median[slot])}, scale {fmt(header.local_scale[slot])}); "
                f"{header.tokens_per_channel[slot]} tokens.\n"
            )
        if header.tokens_total > 0:
            pieces.append(f"Local view, {header.tokens_total} signal tokens ordered by time:\n")
            pieces.append(("view", view_index, -1, header.tokens_total))
            pieces.append("\n")
        return pieces

    @staticmethod
    def error_observation_pieces(
        reason: str,
        *,
        remaining_calls: int,
        remaining_channel_views: int,
        duplicate_of: int | None = None,
    ) -> list[str]:
        """Render a structured tool error without a perceptual payload.

        Errors are deliberately represented as text-only user pieces.  In
        particular, a duplicate refers to the already inserted observation by
        index and never causes a second soft-token block to be allocated.
        """

        if duplicate_of is not None:
            reason = f"{reason}; existing observation {int(duplicate_of)}"
        return [
            f"Tool error: {reason}; remaining calls {int(remaining_calls)}, "
            f"remaining channel views {int(remaining_channel_views)}"
        ]

    # These two spellings make the helper convenient for callers that describe
    # the result as an observation rather than an error.
    observation_error_pieces = error_observation_pieces
    tool_error_pieces = error_observation_pieces

    @classmethod
    def error_observation_turn(
        cls,
        reason: str,
        *,
        remaining_calls: int,
        remaining_channel_views: int,
        duplicate_of: int | None = None,
    ) -> Turn:
        return Turn("user", cls.error_observation_pieces(
            reason,
            remaining_calls=remaining_calls,
            remaining_channel_views=remaining_channel_views,
            duplicate_of=duplicate_of,
        ))

    # ----------------------------------------------------------------- assembly
    def build(
        self,
        turns: Sequence[Turn],
        *,
        supervise_action: bool = True,
        generation_prompt: bool = False,
        thinking: bool = False,
        flat: bool = False,
    ) -> BuiltSequence:
        ids: list[int] = []
        labels: list[int] = []
        segs: list[int] = []
        signal: list[bool] = []
        slots: list[tuple[int, int, Tensor]] = []
        text_parts: list[str] = []

        def add_text(text: str, *, label: bool = False, seg: int = SEG_NONE) -> None:
            piece_ids = self._ids(text)
            ids.extend(piece_ids)
            labels.extend(piece_ids if label else [IGNORE_INDEX] * len(piece_ids))
            segs.extend([seg if label else SEG_NONE] * len(piece_ids))
            signal.extend([False] * len(piece_ids))
            text_parts.append(text)

        def add_ids(piece_ids, *, label: bool = False, seg: int = SEG_NONE) -> None:
            piece_ids = [int(i) for i in piece_ids]
            ids.extend(piece_ids)
            labels.extend(piece_ids if label else [IGNORE_INDEX] * len(piece_ids))
            segs.extend([seg if label else SEG_NONE] * len(piece_ids))
            signal.extend([False] * len(piece_ids))
            text_parts.append(self.tok.decode(piece_ids, skip_special_tokens=False) if piece_ids else "")

        def add_view(view_index: int, slot: int, n: int | None) -> None:
            if n is None:
                n = 0
            if n <= 0:
                return
            start = len(ids)
            ids.extend([self.placeholder_id] * n)
            labels.extend([IGNORE_INDEX] * n)
            segs.extend([SEG_NONE] * n)
            signal.extend([True] * n)
            slots.append((view_index, slot, torch.arange(start, start + n, dtype=torch.int64)))
            text_parts.append(f"<view {view_index}.{slot}:{n}>")

        if flat:
            # AnomalyXL/OpenTSLM assemble ONE flat string and tokenize it as is (chatts.py:324-366): no chat
            # template, no roles, no turn delimiters -- the only special token in the whole sequence is the EOS
            # that closes the answer.  Qwen3.5 adds nothing of its own (no BOS, add_special_tokens is a no-op),
            # so this reproduces their token stream exactly.  Only the direct shape fits: the agentic modes are
            # conversations and have no flat form, so say so instead of silently falling back.
            allowed = {"system", "user", "assistant_final"}
            unexpected = sorted({t.role for t in turns} - allowed)
            if unexpected:
                raise ValueError(f"the flat layout has no turns: {unexpected}")

        for turn in turns:
            if turn.role == "system":
                if flat:
                    # the benchmark strips the pre-prompt and joins the blocks with a newline
                    add_text("".join(turn.pieces).strip() + chr(10))
                else:
                    add_text("<|im_start|>system\n" + "".join(turn.pieces) + "<|im_end|>\n")
            elif turn.role == "user":
                if not flat:
                    add_text("<|im_start|>user" + chr(10))
                for piece in turn.pieces:
                    if isinstance(piece, str):
                        add_text(piece)
                    elif isinstance(piece, ViewRef):
                        # A standalone builder has no frontend encoding from
                        # which to recover the valid-slot count.  Keep the
                        # teacher reference in textual context; the model's
                        # ``build_from_turns`` resolves it to exact soft slots
                        # before any real forward.
                        add_text(f"<view-ref {piece.view_index}>")
                    else:
                        _, view_index, slot, n = piece
                        add_view(view_index, slot, n)
                if not flat:
                    add_text("<|im_end|>" + chr(10))
            elif turn.role == "assistant_action":
                add_text("<|im_start|>assistant\n")
                supervise = bool(supervise_action and turn.supervise)
                add_text("".join(turn.pieces), label=supervise, seg=SEG_ACTION)
                add_text("<|im_end|>", label=supervise, seg=SEG_ACTION)
                add_text("\n")
            elif turn.role == "assistant_final":
                if not flat:
                    add_text("<|im_start|>assistant" + chr(10) + "<think>" + chr(10)*2 + "</think>" + chr(10)*2)
                supervise = bool(
                    turn.supervise
                    and (turn.supervise_final if turn.supervise_final is not None else True)
                )
                # flat: the benchmark glues the answer to the question with a single space, and that space
                # belongs to the first supervised token (their prompt_lengths stop before it).
                add_text((" " if flat else "") + "".join(turn.pieces), label=supervise, seg=SEG_FINAL)
                add_text("<|im_end|>", label=supervise, seg=SEG_FINAL)
                if not flat:
                    add_text(chr(10))
            elif turn.role == "assistant_reasoning_action":
                if len(turn.pieces) != 2:
                    raise ValueError("assistant_reasoning_action requires [reasoning, action_json]")
                reasoning, action = turn.pieces
                supervise = bool(
                    turn.supervise
                    and (turn.supervise_reasoning if turn.supervise_reasoning is not None else True)
                )
                add_text("<|im_start|>assistant\n<think>\n")
                add_text(str(reasoning), label=supervise, seg=SEG_REASONING)
                # The closing think delimiter is part of the rationale target.
                add_text("\n</think>\n\n", label=supervise, seg=SEG_REASONING)
                action_supervise = bool(turn.supervise)
                add_text(str(action), label=action_supervise, seg=SEG_ACTION)
                add_text("<|im_end|>", label=action_supervise, seg=SEG_ACTION)
                add_text("\n")
            elif turn.role == "assistant_reasoning_final":
                if len(turn.pieces) != 2:
                    raise ValueError("assistant_reasoning_final requires [reasoning, final]")
                reasoning, final = turn.pieces
                add_text("<|im_start|>assistant\n<think>\n")
                reasoning_supervise = bool(
                    turn.supervise
                    and (turn.supervise_reasoning if turn.supervise_reasoning is not None else True)
                )
                final_supervise = bool(
                    turn.supervise
                    and (turn.supervise_final if turn.supervise_final is not None else True)
                )
                if isinstance(reasoning, PreTokenized):
                    # Replayed generation: those ids already carry their own closure.
                    add_ids(reasoning.ids, label=reasoning_supervise, seg=SEG_REASONING)
                else:
                    add_text(str(reasoning), label=reasoning_supervise, seg=SEG_REASONING)
                    # Astra explicitly supervises the closure, but never the opener.
                    add_text("\n</think>\n\n", label=reasoning_supervise, seg=SEG_REASONING)
                add_text(str(final), label=final_supervise, seg=SEG_FINAL)
                add_text("<|im_end|>", label=final_supervise, seg=SEG_FINAL)
                add_text("\n")
            else:
                raise ValueError(f"unknown turn role {turn.role!r}")
        if generation_prompt:
            # flat: the benchmark prompt ends at the question and the answer follows after a single space,
            # so there is no opener to add -- the model writes that space itself as its first token.
            if not flat:
                prompt = ("<|im_start|>assistant" + chr(10) + "<think>" + chr(10)
                          if thinking else
                          "<|im_start|>assistant" + chr(10) + "<think>" + chr(10)*2 + "</think>" + chr(10)*2)
                add_text(prompt)
            # a generation prompt never carries supervision, whatever the history turns were
            labels = [IGNORE_INDEX] * len(ids)
            segs = [SEG_NONE] * len(ids)

        input_ids = torch.tensor(ids, dtype=torch.int64)
        is_signal = torch.tensor(signal, dtype=torch.bool)
        n_signal = int(is_signal.sum())
        n_text = int(input_ids.shape[0]) - n_signal
        n_assist = int(sum(1 for s in segs if s != SEG_NONE))
        if n_text > self.max_text:
            raise ValueError(f"text budget exceeded: {n_text} > {self.max_text}")
        if n_signal > self.max_signal:
            raise ValueError(f"signal token budget exceeded: {n_signal} > {self.max_signal}")
        if n_assist > self.max_assistant:
            raise ValueError(f"assistant budget exceeded: {n_assist} > {self.max_assistant}")
        if input_ids.shape[0] > self.max_context:
            raise ValueError(f"context budget exceeded: {input_ids.shape[0]} > {self.max_context}")
        return BuiltSequence(
            input_ids=input_ids,
            labels=torch.tensor(labels, dtype=torch.int64),
            segment=torch.tensor(segs, dtype=torch.int64),
            is_signal=is_signal,
            view_slots=slots,
            n_text_tokens=n_text,
            n_signal_tokens=n_signal,
            n_assistant_tokens=n_assist,
            text="".join(text_parts),
        )


__all__ = [
    "BuiltSequence",
    "ConversationBuilder",
    "DIRECT_INSTRUCTIONS",
    "IGNORE_INDEX",
    "PLACEHOLDER_TOKEN",
    "SEG_ACTION",
    "SEG_FINAL",
    "SEG_NONE",
    "SEG_REASONING",
    "TOOL_INSTRUCTIONS",
    "Turn",
    "ViewRef",
    "ViewHeader",
    "fmt",
]
