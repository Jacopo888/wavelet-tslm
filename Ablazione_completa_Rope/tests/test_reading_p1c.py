"""Astra 15/09 (sentinella D2/D4): programmatic reading at token resolution, decorrelation penalty, reading turn."""

from __future__ import annotations

import re

import pytest
import torch

from wavelet_rope.perceptual_rope import soft_token_decorrelation
from wavelet_rope.reading import view_reading
from wavelet_rope.templates import IGNORE_INDEX, SEG_FINAL, SEG_REASONING, ConversationBuilder, Turn

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _signal() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(3)
    x = torch.randn(3, 16384, generator=g) * 0.1
    x[1, 5000:5300] += 4.0
    x[2, 12000:12050] -= 3.0
    valid = torch.ones_like(x, dtype=torch.bool)
    valid[0, :100] = False
    return x, valid


def test_reading_is_deterministic_bin_resolved_and_header_free() -> None:
    x, valid = _signal()
    ids = ["ch_00", "ch_01", "ch_02"]
    text = view_reading(x, valid, ids, start=0, n_bins=64)
    assert text == view_reading(x.clone(), valid.clone(), ids, start=0, n_bins=64)
    assert "256 samples per bin" in text
    # the injected step on ch_01 is the strongest energy and the strongest deviation, named at bin resolution
    assert "ch_01 [5120,5376)" in text or "ch_01 [4864,5120)" in text
    assert "channels ranked by detail energy: ch_01 > ch_02 > ch_00" in text
    spans = re.findall(r"\[(\d+),(\d+)\)", text)
    assert spans and all((int(b) - int(a)) == 256 for a, b in spans)
    # nothing copyable from the header and no sample-exact positions or braces
    for word in ("median", "scale", "min ", "max ", "{", "}"):
        assert word not in text
    assert all(int(a) % 256 == 0 for a, _ in spans)


def test_reading_local_view_uses_absolute_coordinates_and_local_bins() -> None:
    x, valid = _signal()
    text = view_reading(x[1:2, 4096:8192], valid[1:2, 4096:8192], ["ch_01"], start=4096, n_bins=128, kind="local")
    assert text.startswith("Local view read at 32 samples per bin")
    spans = re.findall(r"\[(\d+),(\d+)\)", text)
    assert spans and all(4096 <= int(a) < int(b) <= 8192 and int(b) - int(a) == 32 for a, b in spans)
    assert "ranked" not in text  # single channel: no ranking clause


def test_reading_handles_degenerate_views() -> None:
    flat = torch.zeros(2, 512)
    text = view_reading(flat, torch.ones_like(flat, dtype=torch.bool), ["a", "b"], start=0, n_bins=64)
    assert "no Haar detail energy stands out" in text
    assert view_reading(torch.zeros(1, 0), torch.zeros(1, 0, dtype=torch.bool), ["a"], start=0, n_bins=64)


def test_decorrelation_penalty_values_and_gradient() -> None:
    same = torch.ones(50, 16)
    assert float(soft_token_decorrelation([same])) == pytest.approx(1.0)
    ortho = torch.eye(16)
    assert float(soft_token_decorrelation([ortho])) == pytest.approx(0.0, abs=1e-7)
    mixed = torch.randn(400, 64, requires_grad=True)
    value = soft_token_decorrelation([mixed, ortho])
    assert 0.0 < float(value.detach()) < 0.5
    value.backward()
    assert mixed.grad is not None and bool((mixed.grad != 0).any())
    assert float(soft_token_decorrelation([torch.randn(1, 8)])) == 0.0  # a single token has no pair
    big = torch.randn(5000, 8)
    assert torch.isfinite(soft_token_decorrelation([big], max_tokens=512))


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


def test_reading_turn_supervises_reading_and_final_separately(tokenizer) -> None:
    b = ConversationBuilder(tokenizer)
    reading = "Global view read at 256 samples per bin. Haar detail energy concentrates in ch_01 [5120,5376)."
    answer = 'localize:{"_L": 16384, "end": 5300, "present": true, "start": 5000}'
    turns = [Turn("system", ["s"]), Turn("user", ["q"]), Turn("assistant_reasoning_final", [reading, answer], SEG_FINAL)]
    seq = b.build(turns)
    supervised = seq.labels != IGNORE_INDEX
    reasoning = seq.segment == SEG_REASONING
    final = seq.segment == SEG_FINAL
    assert int(reasoning.sum()) > 0 and int(final.sum()) > 0
    assert tokenizer.decode(seq.labels[reasoning].tolist()).startswith(reading)
    assert tokenizer.decode(seq.labels[final].tolist()) == answer + "<|im_end|>"
    assert bool((seq.segment[supervised] != 0).all())
    gen = b.build(turns[:-1], generation_prompt=True, thinking=True)
    assert gen.text.endswith("<|im_start|>assistant\n<think>\n")

def test_compact_reading_uses_bin_indices_without_ranking() -> None:
    from wavelet_rope.templates import ViewHeader, reading_grid_line

    x, valid = _signal()
    text = view_reading(x, valid, ["ch_00", "ch_01", "ch_02"], start=0, n_bins=64, fmt="compact")
    assert text.startswith("top ") and "; peak " in text and text.endswith(".")
    assert "ranked" not in text and "[" not in text
    # the injected step on ch_01 (samples 5000-5300) is bin 19 or 20 on the 64-bin grid
    assert "ch_01:20" in text or "ch_01:19" in text
    assert len(text) < 80
    header = ViewHeader(kind="global", channel_ids=("ch_00",), start=0, end=16384, recording_length=16384, dt=1.0,
                        tokens_total=256, tokens_per_channel=(256,), median=(0.0,), scale=(1.0,), minimum=(-1.0,),
                        maximum=(1.0,), valid_count=(16384,), reading_bins=64)
    expected = "Reading grid: 64 bins of 256 samples from 0; bin b covers [0+256*b, 0+256*(b+1))." + chr(10)
    assert reading_grid_line(header) == expected
    assert reading_grid_line(ViewHeader(**{**header.__dict__, "reading_bins": None})) is None
