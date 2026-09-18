#!/usr/bin/env python3
"""Causal LoRA diagnostic on a frozen validation subset.

The comparison is teacher-forced and keeps the prompt, masks, lambda and headers
fixed.  It reports the effect of enabling LoRA and of replacing the soft tokens
with zeros at the informative reading fields.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safetensors.torch import load_file  # noqa: E402

from wavelet_rope.backbone import load_backbone  # noqa: E402
from wavelet_rope.checkpoint import inspect_foreign_checkpoint  # noqa: E402
from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import AnomalyXLSplit, stratified_subset  # noqa: E402
from wavelet_rope.model import WRAgentModel  # noqa: E402
from wavelet_rope.templates import IGNORE_INDEX, SEG_FINAL, SEG_REASONING  # noqa: E402


CONDITIONS = ("TRUE", "ZEROED")
FIELD = re.compile(r"\d")


def _positions(seq, tokenizer, segment: int | None = None) -> torch.Tensor:
    ids = seq.input_ids.tolist()
    labels = seq.labels.tolist()
    segments = seq.segment.tolist()
    positions = []
    for index in range(1, len(ids)):
        if labels[index] == IGNORE_INDEX or (segment is not None and segments[index] != segment):
            continue
        if segment == SEG_REASONING and not FIELD.search(tokenizer.decode([ids[index]])):
            continue
        positions.append(index - 1)
    return torch.tensor(positions, dtype=torch.long)


def _move_episode(episode, device: torch.device):
    return type(episode)(episode.values.to(device), episode.valid.to(device), episode.channel_ids,
                         episode.sample_times.to(device), episode.task)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="complete checkpoint directory")
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--out", "--report", dest="out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_runtime_config(args.config, validate_paths=False)
    raw, spec = cfg.as_dict(), dict(cfg.spec)
    device = torch.device("cuda:0")
    runtime = raw["runtime"]
    backbone = load_backbone(
        raw["backbone"]["snapshot"]["root"],
        backend=runtime["attention_backend"],
        pad_to=int(runtime["attention_qk_pad_to"]),
        device=device,
        gradient_checkpointing=False,
    )
    training = spec["training"]
    model = WRAgentModel(
        backbone,
        spec,
        segment_weights=dict(training["segment_loss_weights"]),
        aux_weight=0.0,
    )
    model.frontend.to(device)
    model.mixer.to(device)
    model.projector.to(device)

    reference = inspect_foreign_checkpoint(args.checkpoint)
    model.load_trainable_state_dict(load_file(str(reference.path / "model.safetensors")))
    backbone.ctx.lam = float(reference.trainer_state["wildcard_lambda"])
    model.eval()
    backbone.text.eval()

    dataset = raw["dataset"]
    validation = AnomalyXLSplit(
        Path(dataset["root"]) / dataset["validation_relative_path"],
        "validation",
        int(dataset["validation_rows"]),
    )
    metas = validation.metas()
    by_row = {meta.row_index: meta for meta in metas}
    rows = stratified_subset(metas, args.rows, args.seed)

    prepared = []
    for row in rows:
        meta = by_row[row]
        episode = _move_episode(validation.episode(row), device)
        reading = model.reading_text(episode, [])
        encoding = model.encode_episode(episode, [], gold=meta.gold_intervals)
        sequence = model.build_sequence(
            meta,
            encoding,
            mode="direct",
            actions=[],
            answer=meta.answer,
            supervise_action=False,
            reading=reading,
        )
        bin_positions = _positions(sequence, backbone.tokenizer, SEG_REASONING)
        final_positions = _positions(sequence, backbone.tokenizer, SEG_FINAL)
        if bin_positions.numel() == 0 or final_positions.numel() == 0:
            continue
        embeds, is_signal, tau = model._inputs(sequence, encoding)
        embeds = embeds.detach()
        prepared.append({
            "meta": meta,
            "embeds": embeds,
            "zero_embeds": embeds.masked_fill(is_signal.unsqueeze(-1), 0.0),
            "is_signal": is_signal,
            "tau": tau,
            "labels": sequence.labels.to(device),
            "bin_positions": bin_positions.to(device),
            "final_positions": final_positions.to(device),
        })
    if not prepared:
        raise RuntimeError("no validation rows contain informative reading and final targets")

    lora = [(name, parameter) for name, parameter in backbone.text.named_parameters() if "lora_B" in name]
    if not lora:
        raise RuntimeError("no LoRA B parameters found")
    saved_lora = {name: parameter.detach().clone() for name, parameter in lora}

    captures: dict[int, torch.Tensor] = {}
    hooks = []
    for layer_index in backbone.wrapped_layers:
        attention = backbone.text.layers[layer_index].self_attn

        def capture_input(_module, inputs, *, index=layer_index):
            if inputs:
                captures[index] = inputs[0].detach()

        hooks.append(attention.o_proj.register_forward_pre_hook(capture_input))

    def set_lora(enabled: bool) -> None:
        with torch.no_grad():
            for name, parameter in lora:
                if enabled:
                    parameter.copy_(saved_lora[name])
                else:
                    parameter.zero_()

    def forward(embeds: torch.Tensor, is_signal: torch.Tensor, tau: torch.Tensor,
                bin_positions: torch.Tensor, final_positions: torch.Tensor) -> dict:
        captures.clear()
        backbone.ctx.set_tokens(is_signal, tau)
        # LoRA weights are fp32: every forward gets a fresh autocast cache after a weight mutation.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = backbone.text(inputs_embeds=embeds, use_cache=False).last_hidden_state[0]
            bin_logits = backbone.lm_head(hidden[bin_positions]).float()
            final_logits = backbone.lm_head(hidden[final_positions]).float()
        return {"bin_logits": bin_logits, "final_logits": final_logits, "captures": dict(captures)}

    results = {}
    try:
        for enabled, label in ((True, "lora_on"), (False, "lora_off")):
            set_lora(enabled)
            bin_true, bin_zero = [], []
            final_true, final_zero = [], []
            logit_delta = []
            layer_contributions = {index: [] for index in backbone.wrapped_layers}
            rows_better = 0
            for item in prepared:
                true = forward(item["embeds"], item["is_signal"], item["tau"], item["bin_positions"],
                               item["final_positions"])
                zero = forward(item["zero_embeds"], item["is_signal"], item["tau"], item["bin_positions"],
                               item["final_positions"])
                labels = item["labels"]
                bin_targets = labels[item["bin_positions"] + 1]
                final_targets = labels[item["final_positions"] + 1]
                true_bin_ce = float(F.cross_entropy(true["bin_logits"], bin_targets))
                zero_bin_ce = float(F.cross_entropy(zero["bin_logits"], bin_targets))
                bin_true.append(true_bin_ce)
                bin_zero.append(zero_bin_ce)
                final_true.append(float(F.cross_entropy(true["final_logits"], final_targets)))
                final_zero.append(float(F.cross_entropy(zero["final_logits"], final_targets)))
                rows_better += int(true_bin_ce < zero_bin_ce)
                logit_delta.append(float((true["bin_logits"] - zero["bin_logits"]).abs().mean()))
                positions = item["bin_positions"]
                for index in backbone.wrapped_layers:
                    true_input = true["captures"][index][0, positions].float()
                    zero_input = zero["captures"][index][0, positions].float()
                    numerator = float((true_input - zero_input).abs().mean())
                    denominator = max(float(true_input.abs().mean()), 1e-12)
                    layer_contributions[index].append(numerator / denominator)
            mean_bin_true = st.fmean(bin_true)
            mean_bin_zero = st.fmean(bin_zero)
            mean_final_true = st.fmean(final_true)
            mean_final_zero = st.fmean(final_zero)
            contribution = {
                index: round(st.fmean(layer_contributions[index]), 4)
                for index in backbone.wrapped_layers
            }
            result = {
                "n": len(prepared),
                "ce_bins_true": mean_bin_true,
                "ce_bins_zero": mean_bin_zero,
                "advantage_true": mean_bin_zero - mean_bin_true,
                "rows_better": rows_better,
                "ce_final_true": mean_final_true,
                "ce_final_zero": mean_final_zero,
                "mean_abs_dlogit_bins": st.fmean(logit_delta),
                "signal_contribution": contribution,
            }
            results[label] = result
            print(
                f"{label}: n={len(prepared)} CE bins true {mean_bin_true:.4f} zero {mean_bin_zero:.4f} "
                f"(advantage of true {mean_bin_zero - mean_bin_true:.4f}, rows better {rows_better}/{len(prepared)}) "
                f"| CE final true {mean_final_true:.4f} zero {mean_final_zero:.4f} "
                f"| mean |dlogit| at bin positions {st.fmean(logit_delta):.4f}",
                flush=True,
            )
            print(f"   signal contribution at bin positions (|d o_proj input| / |o_proj input|) per wrapped layer: "
                  f"{contribution}", flush=True)
    finally:
        set_lora(True)
        for hook in hooks:
            hook.remove()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"checkpoint": str(reference.path), "results": results}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
