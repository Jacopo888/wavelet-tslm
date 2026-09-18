#!/usr/bin/env python3
"""How much attention does the answer pay to the signal tokens, and does it fall across epochs?

Jacopo's hypothesis (17/09): the failure is not that the model cannot read the soft tokens, but that in the first
epochs it learns to lean on the header digits, attention drifts away from the signal, and by epoch 3-5 the embeddings
no longer carry enough weight to change the answer.  If that is right, the attention mass on signal tokens must
DECLINE from epoch 1 to epoch 5; if it is flat, the mechanism is something else.

The three attention backends of this codebase (FLEX_PAD512, EFFICIENT_ATTENTION, MATH) all call fused kernels that
do not return weights, so the weights are recomputed here from the same Q*/K* the model uses, by patching
``wildcard_attention``.  That patch touches only the eight wrapped full-attention layers -- which are exactly the
layers where the signal tokens are seen through the wildcard geometry -- and it changes nothing the model computes:
the original function is still called and its output returned unchanged.

Reported per layer and per checkpoint, at the LAST prompt position (where the answer starts):
  signal   share of attention mass on soft-token positions
  header   share on the text of the header, from "Recording:" to just before the signal block
  other    system prompt, question, chat scaffolding

Usage:
  python tools/attention_mass.py --config <toml> --out <json> --checkpoints <dir>[,<dir>...] [--rows 24]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safetensors.torch import load_file  # noqa: E402

from wavelet_rope import qwen_temporal_rope as qtr  # noqa: E402
from wavelet_rope.backbone import load_backbone  # noqa: E402
from wavelet_rope.checkpoint import inspect_foreign_checkpoint  # noqa: E402
from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import AnomalyXLSplit, stratified_subset  # noqa: E402
from wavelet_rope.model import WRAgentModel  # noqa: E402

ATTENTION_SCALE = qtr.ATTENTION_SCALE


def install_probe(record: list) -> callable:
    """Patch ``wildcard_attention`` so it also reports the attention row of the last query position."""

    original = qtr.wildcard_attention

    def patched(q_star, k_star, v, *, ctx, past_len, dtype):
        out = original(q_star, k_star, v, ctx=ctx, past_len=past_len, dtype=dtype)
        try:
            heads, kv_heads = q_star.shape[1], k_star.shape[1]
            rep = heads // kv_heads
            k16 = k_star.repeat_interleave(rep, dim=1) if rep > 1 else k_star
            # only the final query position matters: it is where the first answer token is produced
            q_last = q_star[:, :, -1:, :].float()
            logits = torch.matmul(q_last, k16.float().transpose(-1, -2)) * ATTENTION_SCALE
            weights = torch.softmax(logits, dim=-1)          # [B, H, 1, KV]
            record.append(weights[0, :, 0, :].mean(0).detach().to("cpu"))
        except Exception as error:                            # never let the probe change the forward
            record.append(error)
        return out

    qtr.wildcard_attention = patched
    return original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", required=True, help="comma-separated checkpoint directories, in order")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    started = time.time()

    cfg = load_runtime_config(args.config, validate_paths=False)
    raw, spec = cfg.as_dict(), dict(cfg.spec)
    device = torch.device("cuda:0")
    runtime = raw["runtime"]
    backbone = load_backbone(raw["backbone"]["snapshot"]["root"], backend=runtime["attention_backend"],
                             pad_to=int(runtime["attention_qk_pad_to"]), device=device,
                             gradient_checkpointing=False)
    training = spec["training"]
    model = WRAgentModel(backbone, spec, segment_weights=dict(training["segment_loss_weights"]), aux_weight=0.0)
    model.frontend.to(device)
    model.mixer.to(device)
    model.projector.to(device)

    root = Path(raw["dataset"]["root"])
    split = AnomalyXLSplit(root / raw["dataset"]["validation_relative_path"], "validation",
                           int(raw["dataset"]["validation_rows"]))
    rows = stratified_subset(split.metas(), args.rows, args.seed)
    by_row = {meta.row_index: meta for meta in split.metas()}

    captured: list = []
    install_probe(captured)

    report = {"config": args.config, "rows": len(rows), "checkpoints": {}}
    for path in [p.strip() for p in args.checkpoints.split(",") if p.strip()]:
        ref = inspect_foreign_checkpoint(path)
        model.load_trainable_state_dict(load_file(str(ref.path / "model.safetensors")))
        backbone.ctx.lam = float(ref.trainer_state["wildcard_lambda"])
        model.eval()
        backbone.text.eval()
        per_layer: dict[int, list[tuple[float, float, float]]] = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for meta in (by_row[r] for r in rows):
                episode = split.episode(meta.row_index)
                episode = type(episode)(episode.values.to(device), episode.valid.to(device), episode.channel_ids,
                                        episode.sample_times.to(device), episode.task)
                encoding = model.encode_episode(episode, [], gold=meta.gold_intervals)
                sequence = model.build_sequence(meta, encoding, mode="direct", actions=[], answer=None,
                                                supervise_action=False, generation=True, thinking=False)
                embeds, is_signal, tau = model._inputs(sequence, encoding)
                signal_mask = is_signal[0].to("cpu")
                # the header runs from the first signal-free text after the system prompt to the signal block
                first_signal = int(torch.nonzero(signal_mask)[0]) if signal_mask.any() else len(signal_mask)
                text = sequence.text
                header_chars = text.find("Recording:")
                header_ratio = 0.0 if header_chars < 0 else header_chars / max(1, len(text))
                header_start = int(header_ratio * first_signal)
                captured.clear()
                backbone.ctx.set_tokens(is_signal, tau)
                backbone.text(inputs_embeds=embeds, use_cache=False)
                for index, weights in enumerate(captured):
                    if isinstance(weights, Exception):
                        continue
                    w = weights[: signal_mask.numel()]
                    signal = float(w[signal_mask].sum())
                    header = float(w[header_start:first_signal].sum())
                    other = float(w.sum()) - signal - header
                    per_layer.setdefault(index, []).append((signal, header, other))
        summary = {str(k): {"signal": round(st.fmean(s for s, _, _ in v), 5),
                            "header": round(st.fmean(h for _, h, _ in v), 5),
                            "other": round(st.fmean(o for _, _, o in v), 5),
                            "n": len(v)}
                   for k, v in sorted(per_layer.items())}
        overall = {k: round(st.fmean(entry[k] for entry in summary.values()), 5)
                   for k in ("signal", "header", "other")} if summary else {}
        report["checkpoints"][str(ref.optimizer_step)] = {"path": str(ref.path), "per_layer": summary,
                                                          "mean": overall}
        print(f"passo {ref.optimizer_step}: segnale {overall.get('signal')} | header {overall.get('header')} | "
              f"altro {overall.get('other')}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    steps = sorted(report["checkpoints"], key=int)
    if len(steps) > 1:
        first = report["checkpoints"][steps[0]]["mean"].get("signal")
        last = report["checkpoints"][steps[-1]]["mean"].get("signal")
        print(f"ATTENTION_MASS: segnale {first} al passo {steps[0]} -> {last} al passo {steps[-1]} "
              f"({'CALA' if last is not None and first is not None and last < first else 'NON CALA'}) "
              f"({round(time.time() - started, 1)}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
