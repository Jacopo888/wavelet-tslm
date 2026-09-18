"""CPU control-flow tests for the agentic runner using an injected decoder."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType, SimpleNamespace

import torch

from wavelet_rope.data import RowMeta, ZoomAction
from wavelet_rope.model import EncodedView, EpisodeEncoding, WRAgentModel
from wavelet_rope.perceptual_rope import PerceptualMemory
from wavelet_rope.signal_contract import SignalEpisode
from wavelet_rope.templates import ConversationBuilder, ViewHeader


class _FakeTokenizer:
    def __init__(self) -> None:
        self._ids = {"<|image_pad|>": 999, "<|im_end|>": 1000}
        self.eos_token_id = 1000

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids.get(token, 1001)

    def __call__(self, text: str, add_special_tokens: bool = False):
        return SimpleNamespace(input_ids=[2000 + index for index, _ in enumerate(text)])

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        rendered = "".join(
            f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>\n" for item in messages
        )
        if add_generation_prompt:
            rendered += (
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                if not enable_thinking else "<|im_start|>assistant\n<think>\n"
            )
        return rendered


def _meta() -> RowMeta:
    return RowMeta(
        id="runner", category="localize", length=512, n_channels=1,
        channel_names=("ch_00",), question="Q?", answer='{"present":false}',
        gold_intervals=(), split="validation", row_index=0,
    )


def _episode(meta: RowMeta) -> SignalEpisode:
    values = torch.zeros((1, meta.length), dtype=torch.float32)
    return SignalEpisode.from_index_grid(
        values=values, valid=torch.ones_like(values, dtype=torch.bool),
        channel_ids=meta.channel_names, dt=1.0,
    )


def _fake_model() -> WRAgentModel:
    model = WRAgentModel.__new__(WRAgentModel)
    torch.nn.Module.__init__(model)
    model.builder = ConversationBuilder(_FakeTokenizer())
    model.max_calls = 3
    model.max_views = 3

    def encode(self, episode, actions, **kwargs):
        memory = PerceptualMemory(t_origin=0.0, dt_ref=1.0)
        views = []
        specs = [("global", episode.channel_ids, 0, episode.length)]
        specs.extend(("local", action.channels, action.start, action.end) for action in actions)
        for index, (kind, channels, start, end) in enumerate(specs):
            n = 2 if kind == "global" else len(channels) * 3
            per_channel = tuple([n // len(channels)] * len(channels))
            header = ViewHeader(
                kind=kind, channel_ids=tuple(channels), start=start, end=end,
                recording_length=episode.length, dt=1.0, tokens_total=n,
                tokens_per_channel=per_channel, median=(0.0,) * len(channels),
                scale=(1.0,) * len(channels), minimum=(0.0,) * len(channels),
                maximum=(0.0,) * len(channels), valid_count=(end - start,) * len(channels),
                local_median=None if kind == "global" else (0.0,) * len(channels),
                local_scale=None if kind == "global" else (1.0,) * len(channels),
                local_available=None if kind == "global" else (True,) * len(channels),
                observation_index=index, remaining_calls=0, remaining_channel_views=0,
            )
            views.append(EncodedView(
                kind=kind, channel_ids=tuple(channels), start=start, end=end,
                soft=torch.zeros((n, 2560)), tau=torch.zeros(n), header=header,
                aux_logits=torch.zeros((len(channels), 1)),
                aux_valid=torch.ones((len(channels), 1), dtype=torch.bool),
                aux_edges=torch.tensor([0, episode.length], dtype=torch.int64),
                n_tokens_per_channel=per_channel,
            ))
        return EpisodeEncoding(views=views, memory=memory, n_signal_tokens=sum(v.soft.shape[0] for v in views))

    model.encode_episode = MethodType(encode, model)
    return model


def test_runner_executes_valid_action_and_penalises_duplicate_and_invalid() -> None:
    model = _fake_model()
    action = ZoomAction(("ch_00",), 0, 256).to_json()
    script = iter([
        f"global evidence</think>\n\n{action}",
        f"try again</think>\n\n{action}",
        'bad</think>\n\n{"tool":"zoom","channels":["ch_00"],"start":0}',
        'enough</think>\n\n{"present":false}',
    ])

    def fake_decode(*, turns, tool_available, max_tokens):
        assert max_tokens >= 0
        if not tool_available:
            assert any("Tool no longer available" in str(piece) for turn in turns for piece in turn.pieces if isinstance(piece, str))
        return {"text": next(script), "finished": True}

    result = model.generate_agentic(
        _meta(), _episode(_meta()), max_calls=3, max_views=3,
        max_assistant_tokens=200, final_reserve=20, max_reasoning_tokens=10,
        decode=fake_decode,
    )
    assert result["finished"]
    assert len(result["executed_actions"]) == 1
    assert result["calls"] == 3
    assert len(result["duplicate_actions"]) == 1
    assert len(result["invalid_actions"]) == 1
    assert result["token_counts"]["soft"] == 5
    error_turns = [turn for turn in result["turns"] if turn.role == "user" and any("Tool error:" in str(piece) for piece in turn.pieces)]
    assert len(error_turns) == 2
    assert all(not any(isinstance(piece, tuple) for piece in turn.pieces) for turn in error_turns)
    assert "existing observation 1" in "".join(str(piece) for piece in error_turns[0].pieces + error_turns[1].pieces)


def test_runner_blocks_tool_when_final_reserve_would_be_breached() -> None:
    model = _fake_model()
    action = ZoomAction(("ch_00",), 0, 256).to_json()
    script = iter([
        f"r</think>\n\n{action}",
        'final</think>\n\n{"present":false}',
    ])

    def fake_decode(*, tool_available, **kwargs):
        return {"text": next(script), "finished": True}

    result = model.generate_agentic(
        _meta(), _episode(_meta()), max_calls=3, max_views=3,
        max_assistant_tokens=30, final_reserve=25, max_reasoning_tokens=10,
        decode= fake_decode,
    )
    assert result["finished"]
    assert not result["executed_actions"]
    assert result["calls"] == 0
    assert any(
        turn.role == "user" and any("Tool no longer available" in str(piece) for piece in turn.pieces)
        for turn in result["turns"]
    )
    assert result["assistant_tokens"] <= 30
