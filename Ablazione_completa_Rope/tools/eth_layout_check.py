#!/usr/bin/env python3
"""Do the soft tokens of each channel land in that channel's own labelled block?

Jacopo, 17/09: use the AnomalyXL/OpenTSLM prompt verbatim for training -- one ``The following is channel X:``
block per channel, its 256 tokens right after the label, and no statistic anywhere.  Our layout put every
channel in ONE time-ordered block and left channel identity to ``channel_embedding``, so the reference gave the
model a name-to-token correspondence we never gave it.

This checks the rewiring on real recordings, not on a synthetic header:

  partition   every soft token of the view appears in exactly one block, none twice, none missing;
  identity    the tokens inside block ``slot`` are exactly the tokens whose ``channel_slot`` is ``slot``,
              in the same order, compared on the embeddings actually written into the sequence;
  tau         the physical-time axis follows the same selection (a mismatch here would silently de-anchor RoPE);
  text        no statistic reaches the prompt, and the labels are the benchmark's own string.

Usage:
  python tools/eth_layout_check.py --config <toml> [--rows 8] [--show 1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wavelet_rope.backbone import load_backbone  # noqa: E402
from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import AnomalyXLSplit, stratified_subset  # noqa: E402
from wavelet_rope.model import WRAgentModel  # noqa: E402
from wavelet_rope.scoring import PRECISE_SYSTEM_PROMPT  # noqa: E402

def tokenizer_decode(model, ids) -> str:
    return model.bb.tokenizer.decode([int(i) for i in ids], skip_special_tokens=False)


BANNED = ("median ", "scale ", "min ", "max ", "valid ", " tokens", "Recording:", "Global view", "Reading grid")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--show", type=int, default=1, help="how many prompts to print in full")
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    cfg = load_runtime_config(args.config, validate_paths=False)
    raw, spec = cfg.as_dict(), dict(cfg.spec)
    device = torch.device("cuda:0")
    runtime = raw["runtime"]
    backbone = load_backbone(raw["backbone"]["snapshot"]["root"], backend=runtime["attention_backend"],
                             pad_to=int(runtime["attention_qk_pad_to"]), device=device,
                             gradient_checkpointing=False)
    model = WRAgentModel(backbone, spec, segment_weights=dict(spec["training"]["segment_loss_weights"]),
                         aux_weight=0.0)
    model.frontend.to(device)
    model.mixer.to(device)
    model.projector.to(device)
    model.eval()
    mode = getattr(model, "header_stats", "full")
    print(f"header_stats = {mode!r}", flush=True)
    if mode != "eth":
        raise SystemExit("this check is for the ETH layout: set training.header_stats = \"eth\" in the spec")

    root = Path(raw["dataset"]["root"])
    split = AnomalyXLSplit(root / raw["dataset"]["validation_relative_path"], "validation",
                           int(raw["dataset"]["validation_rows"]))
    rows = stratified_subset(split.metas(), args.rows, args.seed)
    by_row = {meta.row_index: meta for meta in split.metas()}

    shown = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for meta in (by_row[r] for r in rows):
            episode = split.episode(meta.row_index)
            episode = type(episode)(episode.values.to(device), episode.valid.to(device), episode.channel_ids,
                                    episode.sample_times.to(device), episode.task)
            encoding = model.encode_episode(episode, [], gold=meta.gold_intervals)
            sequence = model.build_sequence(meta, encoding, mode="direct", actions=[], answer=None,
                                            supervise_action=False, generation=True, thinking=False)
            embeds, is_signal, tau = model._inputs(sequence, encoding)
            view = encoding.views[0]
            channels = len(view.channel_ids)

            blocks = [(v, s, p) for v, s, p in sequence.view_slots]
            if [s for _, s, _ in blocks] != list(range(channels)):
                raise SystemExit(f"row {meta.row_index}: expected one block per channel, got "
                                 f"{[s for _, s, _ in blocks]}")

            # partition: the blocks together are exactly the signal positions, each once
            covered = torch.cat([p for _, _, p in blocks]).tolist()
            signal_positions = is_signal[0].nonzero().squeeze(1).cpu().tolist()
            if sorted(covered) != signal_positions or len(set(covered)) != len(covered):
                raise SystemExit(f"row {meta.row_index}: the blocks do not partition the signal positions")
            if len(covered) != int(view.soft.shape[0]):
                raise SystemExit(f"row {meta.row_index}: {len(covered)} placeholders for "
                                 f"{int(view.soft.shape[0])} soft tokens")

            # identity: what got written at those positions is that channel's own tokens, in order
            for _, slot, positions in blocks:
                selected = (view.channel_slot == slot).nonzero().squeeze(1)
                expected = view.soft[selected].to(embeds.dtype)
                written = embeds[0, positions.to(device)]
                if not torch.equal(written, expected):
                    raise SystemExit(f"row {meta.row_index}, channel {slot}: embeddings do not match")
                if not torch.equal(tau[0, positions.to(device)], view.tau[selected].to(device)):
                    raise SystemExit(f"row {meta.row_index}, channel {slot}: tau does not match")
                if int(positions.numel()) != int(view.n_tokens_per_channel[slot]):
                    raise SystemExit(f"row {meta.row_index}, channel {slot}: count mismatch")

            text = sequence.text
            # the flat string of the benchmark: not one chat-template token in the whole prompt
            for marker in ("<|im_start|>", "<|im_end|>", "<think>", "</think>"):
                if marker in text:
                    raise SystemExit(f"row {meta.row_index}: {marker} reached the flat prompt")
            if not text.startswith(PRECISE_SYSTEM_PROMPT):
                raise SystemExit(f"row {meta.row_index}: the prompt does not open with the benchmark system prompt")
            for banned in BANNED:
                if banned in text:
                    raise SystemExit(f"row {meta.row_index}: {banned!r} reached the prompt")
            for slot, ch in enumerate(view.channel_ids):
                if f"The following is channel {ch}: " not in text:
                    raise SystemExit(f"row {meta.row_index}: missing the label of channel {ch}")
            if PRECISE_SYSTEM_PROMPT not in text:
                raise SystemExit(f"row {meta.row_index}: the benchmark system prompt is not verbatim")

            # what is trained must be what is asked: removing the assistant opener is only safe if the
            # generation prompt is still a byte-exact prefix of the supervised sequence, ids included.
            trained = model.build_sequence(meta, encoding, mode="direct", actions=[], answer=meta.answer,
                                           supervise_action=False, generation=False, thinking=False)
            n = int(sequence.input_ids.numel())
            if trained.input_ids[:n].tolist() != sequence.input_ids.tolist():
                raise SystemExit(f"row {meta.row_index}: the generation prompt is not a prefix of the training "
                                 f"sequence")
            tail = tokenizer_decode(model, trained.input_ids[n:])
            if tail != " " + meta.answer.strip() + "<|im_end|>":
                raise SystemExit(f"row {meta.row_index}: the supervised tail is {tail!r}")
            supervised = trained.labels != -100
            if tokenizer_decode(model, trained.input_ids[supervised]) != tail:
                raise SystemExit(f"row {meta.row_index}: supervision does not cover exactly the answer")

            per_channel = sorted({int(n) for n in view.n_tokens_per_channel})
            print(f"riga {meta.row_index} ({meta.category}): {channels} canali, "
                  f"{int(view.soft.shape[0])} token, per canale {per_channel}  OK", flush=True)
            if shown < args.show:
                print("-" * 78)
                print(text)
                print("-" * 78, flush=True)
                shown += 1

    print(f"ETH_LAYOUT: {len(rows)} righe, partizione e identita' per canale verificate", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
