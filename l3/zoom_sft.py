#!/usr/bin/env python3
"""Z1 supervised warm-up for the two-turn zoom policy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch

from l3.grpo import (
    _atomic_json_dump,
    _autocast_for,
    _load_policy_and_train,
    _save_composed_checkpoint,
    _write_jsonl,
    load_l3_config,
    sha256_file,
)
from l3.zoom import (
    choose_supervised_action,
    prepare_zoom_batch,
    supervised_segments,
    with_action_instruction,
    zoom_training_loss,
)
from l3.zoom_traces import compact_trace_sample, load_trace_plan, validate_trace_plan


def _sample_rng(seed: int, sample: dict[str, Any]) -> random.Random:
    identity = f"{seed}:{sample.get('id', '')}:{sample.get('_l3_train_index', '')}"
    value = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big")
    return random.Random(value)


def _linear_warmup_decay(optimizer: torch.optim.Optimizer, total: int, warmup: int):
    def factor(step: int) -> float:
        if warmup and step < warmup:
            return float(step) / float(warmup)
        return max(0.0, float(total - step) / max(1.0, float(total - warmup)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _gold_answer(sample: dict[str, Any], eos_token: str) -> str:
    answer = str(sample["answer"])
    if eos_token and not answer.endswith(eos_token):
        answer += eos_token
    return answer


def train_zoom_sft(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(f"Z1 run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    try:
        resolved = load_l3_config(args.config)
        sft = resolved.get("zoom_sft", {})
        if int(sft.get("epochs", 0)) != 1 or int(sft.get("batch_size", 0)) != 1:
            raise ValueError("Z1 requires exactly one epoch and batch size one")
        if not math.isclose(float(sft.get("lr", math.nan)), 1e-4):
            raise ValueError("Z1 learning rate must be 1e-4")
        if not math.isclose(float(sft.get("weight_decay", math.nan)), 0.01):
            raise ValueError("Z1 weight decay must be 0.01")
        z1bis = sft.get("trace_contract") == "z1bis_v1"
        if z1bis and args.traces is None:
            raise ValueError("Z1-bis requires a reviewed --traces manifest")
        if not z1bis and not math.isclose(float(sft.get("heuristic_fraction", math.nan)), 0.30):
            raise ValueError("legacy Z1 heuristic fraction must be 0.30")
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        model, dataset, optimized, composed, train_sha = _load_policy_and_train(
            resolved, checkpoint
        )
        if len(dataset) != 4488:
            raise RuntimeError(f"Z1 expected 4488 rows, found {len(dataset)}")
        if any(parameter.requires_grad for name, parameter in model.named_parameters()
               if name not in optimized):
            raise RuntimeError("Z1 must train LoRA tensors only")
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if z1bis:
            frozen = tuple(str(value) for value in sft.get("frozen_parameter_prefixes", ()))
            if frozen != ("ts_encoder.", "ts_projector."):
                raise ValueError("Z1-bis must declare frozen ts_encoder/ts_projector prefixes")
            if any(name.startswith(frozen) for name in optimized):
                raise RuntimeError("Z1-bis encoder/projector must remain frozen")
            traces_path = Path(args.traces).expanduser().resolve()
            traces = load_trace_plan(traces_path)
            trace_summary = validate_trace_plan(
                [compact_trace_sample(dataset[index]) for index in range(len(dataset))],
                traces, resolved,
            )
            trace_manifest = json.loads(
                (traces_path.parent / "summary.json").read_text(encoding="utf-8")
            )
            if (trace_manifest.get("status") != "DONE"
                    or trace_manifest.get("format") != "l3_zoom_z1bis_trace_manifest_v1"
                    or trace_manifest.get("train_sha256") != train_sha
                    or int(trace_manifest.get("seed", -1)) != int(sft["seed"])):
                raise RuntimeError("Z1-bis trace-manifest metadata mismatch")
            if sha256_file(traces_path) != trace_manifest["traces_sha256"]:
                raise RuntimeError("Z1-bis trace-manifest SHA mismatch")
        else:
            traces_path = None
            traces = []
            trace_summary = None
        optimizer = torch.optim.AdamW(
            trainable, lr=float(sft["lr"]), weight_decay=float(sft["weight_decay"])
        )
        total_steps = len(dataset)
        scheduler = _linear_warmup_decay(
            optimizer, total_steps, int(float(sft.get("warmup_ratio", 0.03)) * total_steps)
        )
        seed = int(sft.get("seed", resolved.get("seed", 42)))
        order = list(range(total_steps))
        random.Random(seed).shuffle(order)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
        manifest = {
            "format": "l3_zoom_sft_manifest_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "initial_checkpoint": str(checkpoint),
            "initial_checkpoint_sha256": sha256_file(checkpoint),
            "train_sha256": train_sha, "resolved_config": resolved,
            "seed": seed, "order_sha256": hashlib.sha256(
                json.dumps(order, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "optimized_names": optimized, "composed_slim_names": composed,
            "trace_contract": sft.get("trace_contract", "legacy_z1"),
            "traces": str(traces_path) if traces_path else None,
            "traces_sha256": sha256_file(traces_path) if traces_path else None,
            "trace_summary": trace_summary,
        }
        _atomic_json_dump(run_dir / "RUN_MANIFEST.json", manifest)
        coverage: dict[str, int] = {}
        sources: dict[str, int] = (
            {"zoom": 0, "direct": 0, "replay": 0}
            if z1bis else {"gold_jitter": 0, "heuristic": 0}
        )
        sources_by_category: dict[str, dict[str, int]] = {}
        loss_sum = 0.0
        token_sum = 0
        segment_sum = 0
        started = time.perf_counter()
        model.train()
        with (run_dir / "steps.jsonl").open("x", encoding="utf-8") as log:
            for step, index in enumerate(order, start=1):
                sample = dataset[index]
                sample["_l3_train_index"] = index
                rng = _sample_rng(seed, sample)
                if z1bis:
                    trace = traces[index]
                    source, action = trace.source, trace.action
                else:
                    action, source = choose_supervised_action(
                        sample, rng, float(sft["heuristic_fraction"])
                    )
                optimizer.zero_grad(set_to_none=True)
                if action is not None:
                    first, second = supervised_segments(sample, action, model.get_eos_token())
                    first_inputs = model.prepare_batch([first], training=True)
                    second_inputs = prepare_zoom_batch(
                        model, sample, action, training=True, answer=second["answer"]
                    )
                    first_tokens = int((first_inputs["labels"] != -100).sum().item())
                    second_tokens = int((second_inputs["labels"] != -100).sum().item())
                    supervised_tokens = first_tokens + second_tokens
                    if first_tokens < 1 or second_tokens < 1:
                        raise RuntimeError(f"empty supervised segment at train index {index}")
                    with _autocast_for(model):
                        first_loss = model(**first_inputs)["loss"]
                    weighted_first = first_loss * (first_tokens / supervised_tokens)
                    first_value = float(weighted_first.detach().item())
                    weighted_first.backward()
                    del first_inputs, first_loss, weighted_first
                    with _autocast_for(model):
                        second_loss = zoom_training_loss(model, second_inputs)
                    weighted_second = second_loss * (second_tokens / supervised_tokens)
                    weighted_second.backward()
                    loss_value = first_value + float(weighted_second.detach().item())
                    del second_inputs, second_loss, weighted_second
                    segments = 2
                else:
                    direct = dict(sample) if source == "replay" else with_action_instruction(sample)
                    direct["answer"] = _gold_answer(sample, model.get_eos_token())
                    direct_inputs = model.prepare_batch([direct], training=True)
                    first_tokens = 0
                    second_tokens = int((direct_inputs["labels"] != -100).sum().item())
                    supervised_tokens = second_tokens
                    if second_tokens < 1:
                        raise RuntimeError(f"empty direct segment at train index {index}")
                    with _autocast_for(model):
                        direct_loss = model(**direct_inputs)["loss"]
                    loss_value = float(direct_loss.detach().item())
                    direct_loss.backward()
                    del direct_inputs, direct_loss
                    segments = 1
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable, float(sft.get("max_grad_norm", 1.0))
                )
                if not torch.isfinite(grad_norm) or not math.isfinite(loss_value):
                    raise RuntimeError(f"non-finite Z1 value at train index {index}")
                optimizer.step()
                scheduler.step()
                category = str(sample.get("category", "unknown"))
                coverage[category] = coverage.get(category, 0) + 1
                sources[source] += 1
                category_sources = sources_by_category.setdefault(category, {})
                category_sources[source] = category_sources.get(source, 0) + 1
                loss_sum += loss_value
                token_sum += supervised_tokens
                segment_sum += segments
                _write_jsonl(log, {
                    "step": step, "train_index": index,
                    "id": str(sample.get("id", "")), "category": category,
                    "action_source": source,
                    "zoom_channel": action.channel if action else None,
                    "zoom_start": action.start if action else None,
                    "zoom_end": action.end if action else None,
                    "zoom_window_len": action.window_len if action else None,
                    "action_tokens": first_tokens, "answer_tokens": second_tokens,
                    "loss": loss_value,
                    "grad_norm": float(grad_norm.item()),
                    "lr": float(scheduler.get_last_lr()[0]),
                })
                if step % int(sft.get("log_every", 25)) == 0:
                    print(
                        f"L3_Z1_STEP step={step}/{total_steps} category={category} "
                        f"source={source} grad={float(grad_norm.item()):.6g}", flush=True,
                    )
        if sum(coverage.values()) != total_steps or set(coverage) != {
            "classify_with_evidence", "measure_magnitude", "localize",
            "localize_all_channels", "lead_lag_with_magnitude",
        }:
            raise RuntimeError(f"Z1 category coverage is incomplete: {coverage}")
        if z1bis and sources != {
            source: sum(values.get(source, 0) for values in sources_by_category.values())
            for source in ("zoom", "direct", "replay")
        }:
            raise RuntimeError("Z1-bis source/category accounting mismatch")
        checkpoint_out = run_dir / ("vz1bis_epoch_1_end.pt" if z1bis else "vz1_epoch_1_end.pt")
        _save_composed_checkpoint(
            checkpoint_out, model, composed, optimized, resolved, total_steps, cursor=total_steps
        )
        summary = {
            "status": "DONE", "steps": total_steps, "episodes": total_steps,
            "segments": segment_sum, "supervised_tokens": token_sum,
            "loss_mean": loss_sum / total_steps,
            "coverage": coverage, "action_sources": sources,
            "action_sources_by_category": sources_by_category,
            "trace_summary": trace_summary,
            "heuristic_rate": (sources["heuristic"] / total_steps if not z1bis else None),
            "zoom_rate": (sources["zoom"] / total_steps if z1bis else 1.0),
            "direct_rate": (sources["direct"] / total_steps if z1bis else 0.0),
            "replay_rate": (sources["replay"] / total_steps if z1bis else 0.0),
            "peak_mem_mib": (float(torch.cuda.max_memory_allocated() / 1024**2)
                             if torch.cuda.is_available() else 0.0),
            "seconds": time.perf_counter() - started,
            "checkpoint": str(checkpoint_out),
            "checkpoint_sha256": sha256_file(checkpoint_out),
            "optimized_tensor_count": len(optimized),
            "composed_tensor_count": len(composed),
        }
        _atomic_json_dump(run_dir / "summary.json", summary)
        _atomic_json_dump(run_dir / "DONE", summary)
    except Exception as exc:
        _atomic_json_dump(run_dir / "FAILED", {
            "status": "FAILED", "error_type": type(exc).__name__, "reason": str(exc),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--traces")
    return parser


def main() -> int:
    train_zoom_sft(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
