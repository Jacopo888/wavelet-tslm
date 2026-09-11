#!/usr/bin/env python3
"""Minimal single-turn GRPO for the va1 multimodal WaveletTSModel.

The pure helpers at the top are intentionally CPU-testable. The CLI imports the
TSLM checkout lazily, so local unit tests do not load Qwen or require a GPU.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CATEGORIES = {
    "classify_with_evidence",
    "measure_magnitude",
    "localize",
    "localize_all_channels",
    "lead_lag_with_magnitude",
}
ZOOM_RL_CATEGORIES = REQUIRED_CATEGORIES - {"lead_lag_with_magnitude"}
MAX_ABS_LOG_RATIO = 1.0e-3
KL_NEGATIVE_TOLERANCE = -1.0e-7


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _resolve_base(config_path: Path, value: str) -> Path:
    raw = Path(value).expanduser()
    candidates = [
        raw if raw.is_absolute() else config_path.parent / raw,
        REPO_ROOT / "kaggle" / "patch" / raw.name,
        Path.cwd() / "configs" / raw.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve base_config {value!r}: {candidates}")


def load_l3_config(path: str | Path) -> dict[str, Any]:
    """Resolve base YAML + L3 overlay and reject unsafe smoke settings."""
    config_path = Path(path).resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("base_config"), str):
        raise ValueError("L3 config requires a string base_config")
    base_path = _resolve_base(config_path, document["base_config"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError(f"base config is not a mapping: {base_path}")
    overlay = {key: value for key, value in document.items() if key != "base_config"}
    resolved = _deep_merge(base, overlay)
    resolved["_l3"] = {
        "config_path": str(config_path),
        "base_config_path": str(base_path),
    }
    validate_l3_config(resolved)
    return resolved


def validate_l3_config(config: Mapping[str, Any]) -> None:
    rl = config.get("rl")
    if not isinstance(rl, Mapping):
        raise ValueError("missing rl config")
    contract = str(rl.get("contract", "smoke"))
    common = {
        "prompt_split": "train",
        "expected_train_rows": 4488,
        "do_sample": True,
        "trainable": "lora",
    }
    for key, expected in common.items():
        if rl.get(key) != expected:
            raise ValueError(f"rl.{key} must be {expected!r}, got {rl.get(key)!r}")
    numeric = {
        "temperature": 0.7,
        "top_p": 0.95,
        "lr": 1e-5,
        "kl_coef": 0.01,
        "format_penalty": 0.0,
        "max_grad_norm": 1.0,
    }
    for key, expected in numeric.items():
        if not math.isclose(float(rl.get(key, math.nan)), expected, abs_tol=1e-15):
            raise ValueError(f"rl.{key} must be {expected}")
    if config.get("model", {}).get("architecture") != "wavelet":
        raise ValueError("L3 smoke requires model.architecture=wavelet")
    if config.get("backbone", {}).get("model_id") != "Qwen/Qwen3.5-4B":
        raise ValueError("L3 smoke requires Qwen/Qwen3.5-4B")
    if config.get("training", {}).get("backbone_training") != "lora":
        raise ValueError("base config must inject LoRA")
    if contract == "smoke":
        fixed = {"G": 4, "steps": 20, "prompts_per_step": 1}
    elif contract == "longrun":
        fixed = {"G": 8, "prompts_per_step": 2, "weight_decay": 0.0}
        steps = rl.get("steps")
        if type(steps) is not int or not 1 <= steps <= 400:
            raise ValueError("longrun rl.steps must be an integer in [1, 400]")
        if rl.get("checkpoint_every") != 25:
            raise ValueError("longrun rl.checkpoint_every must be 25")
        if rl.get("pool_seed") != 20260911 or rl.get("pool_max_rollouts") != 1600:
            raise ValueError("longrun pool seed/cap must be 20260911/1600")
        memory = rl.get("memory_gate_indices")
        if memory != [3804, 3930]:
            raise ValueError("longrun memory gate must use train indices [3804, 3930]")
        if rl.get("min_pool_per_category") != 8:
            raise ValueError("longrun requires at least 8 pool prompts per category")
    elif contract == "zoom":
        fixed = {
            "G": 8, "prompts_per_step": 2, "steps": 200,
            "weight_decay": 0.0, "checkpoint_every": 25,
            "pool_size": 200, "resample_every": 100,
        }
        if set(rl.get("excluded_categories", ())) != {"lead_lag_with_magnitude"}:
            raise ValueError("zoom RL must exclude lead_lag_with_magnitude")
        if int(rl.get("pool_per_category", 0)) != 50:
            raise ValueError("zoom RL requires 50 prompts per allowed category")
        if int(rl.get("pool_seed", -1)) != 20260911:
            raise ValueError("zoom RL pool seed must be 20260911")
        if rl.get("memory_gate_indices") != [3804, 3930]:
            raise ValueError("zoom memory gate must use train indices [3804, 3930]")
    else:
        raise ValueError(f"unknown rl.contract {contract!r}")
    for key, expected in fixed.items():
        if rl.get(key) != expected:
            raise ValueError(f"{contract} rl.{key} must be {expected!r}")

    if contract in {"longrun", "zoom"}:
        expected = rl.get("train_prompt_expectations")
        if not isinstance(expected, Mapping):
            raise ValueError("missing longrun train_prompt_expectations")
        for index in rl["memory_gate_indices"]:
            row = expected.get(str(index))
            if not isinstance(row, Mapping):
                raise ValueError(f"missing memory-gate metadata for train index {index}")
            if (row.get("category"), row.get("n_channels"), row.get("length")) != (
                "localize_all_channels", 16, 131072
            ):
                raise ValueError(f"invalid longrun memory-gate metadata at {index}")
        if contract == "longrun":
            return

    if contract == "zoom":
        init_indices = rl.get("init_prompt_indices")
        if not isinstance(init_indices, list) or not init_indices:
            raise ValueError("zoom init_prompt_indices must be non-empty")
        return

    indices = rl.get("train_prompt_indices")
    init_indices = rl.get("init_prompt_indices")
    if not isinstance(indices, list) or len(indices) != 20 or len(set(indices)) != 20:
        raise ValueError("train_prompt_indices must contain 20 distinct integers")
    if not isinstance(init_indices, list) or len(init_indices) != 5:
        raise ValueError("init_prompt_indices must contain five indices")
    upper = int(rl["expected_train_rows"])
    all_indices = list(indices) + list(init_indices) + [rl.get("memory_gate_index")]
    if any(type(index) is not int or not 0 <= index < upper for index in all_indices):
        raise ValueError("all prompt indices must be integer offsets in the train split")
    expected = rl.get("train_prompt_expectations")
    if not isinstance(expected, Mapping):
        raise ValueError("missing train_prompt_expectations")
    missing = {str(index) for index in all_indices} - set(expected)
    if missing:
        raise ValueError(f"missing metadata expectations for train indices: {sorted(missing)}")


def normalize_advantages(
    rewards: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, bool]:
    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("rewards must be a non-empty vector")
    if not torch.isfinite(rewards).all():
        raise ValueError("non-finite reward")
    centered = rewards - rewards.mean()
    variance = centered.square().mean()
    if variance.item() == 0.0:
        return torch.zeros_like(rewards), True
    return centered / (variance.sqrt() + eps), False


def k3_kl(log_policy: torch.Tensor, log_reference: torch.Tensor) -> torch.Tensor:
    """Per-token Schulman k3 estimator for KL(policy || reference)."""
    if log_policy.shape != log_reference.shape:
        raise ValueError("policy/reference log-prob shapes differ")
    delta = log_reference.float() - log_policy.float()
    value = torch.expm1(delta) - delta
    if not torch.isfinite(value).all():
        raise ValueError("k3 KL produced a non-finite value")
    minimum = float(value.min().item()) if value.numel() else 0.0
    if minimum < KL_NEGATIVE_TOLERANCE:
        raise ValueError(f"k3 KL below numerical tolerance: {minimum}")
    # Tiny negatives within tolerance can only be floating-point noise.  Clamp
    # them so logged KL values retain the estimator's mathematical invariant.
    return value.clamp_min(0.0)


def ppo_ratio(log_policy: torch.Tensor, log_old: torch.Tensor) -> torch.Tensor:
    if log_policy.shape != log_old.shape:
        raise ValueError("policy/old log-prob shapes differ")
    return torch.exp(log_policy - log_old)


def max_abs_log_ratio(log_policy: torch.Tensor, log_old: torch.Tensor) -> torch.Tensor:
    """Largest token-level |log pi - log pi_old| for the pre-update gate."""
    if log_policy.shape != log_old.shape or log_policy.numel() == 0:
        raise ValueError("policy/old log-probs must have the same non-empty shape")
    return (log_policy - log_old).abs().max()


def response_token_mask(prompt_tokens: int, answer_tokens: int) -> torch.Tensor:
    if prompt_tokens < 1 or answer_tokens < 1:
        raise ValueError("prompt_tokens and answer_tokens must be positive")
    mask = torch.zeros(prompt_tokens + answer_tokens, dtype=torch.bool)
    mask[prompt_tokens:] = True
    return mask


@dataclass(frozen=True)
class RewardResult:
    primary: float
    optimization: float
    format_adjustment: float
    parsable: bool
    extracted: str


def reward_completion(
    dataset: Any,
    prediction: str,
    sample: Mapping[str, Any],
    format_penalty: float = 0.0,
) -> RewardResult:
    """Use the official metric; parsing only controls zero/format accounting."""
    if format_penalty < 0:
        raise ValueError("format_penalty is a non-negative magnitude")
    scoring = _official_scoring()
    REQUIRED_KEYS = scoring.REQUIRED_KEYS
    parse_structured_pred = scoring.parse_structured_pred

    category = str(sample.get(dataset.category_key, ""))
    if category not in REQUIRED_KEYS:
        raise ValueError(f"unknown precise category {category!r}")
    extracted = dataset.extract_answer(prediction, sample)
    prefix, separator, _ = extracted.partition(":")
    category_ok = bool(separator) and prefix.strip() == category
    parsed = parse_structured_pred(extracted, REQUIRED_KEYS[category])
    parsable = category_ok and parsed is not None
    primary = 0.0
    if parsable:
        evaluated = dataset.evaluate_answer(extracted, sample)
        primary = float(evaluated.get(dataset.primary_metric, 0.0))
        if not math.isfinite(primary):
            raise ValueError("official scorer returned a non-finite primary")
    adjustment = 0.0 if parsable else -float(format_penalty)
    return RewardResult(primary, primary + adjustment, adjustment, parsable, extracted)


def _official_scoring() -> Any:
    """Import the official scorer, with a direct-file CPU-test fallback.

    The package's ``__init__`` imports Hugging Face datasets even though
    ``scoring.py`` itself does not. The Windows test interpreter intentionally
    need not have that GPU/node dependency installed.
    """
    try:
        from src.datasets.anomaly_xl import scoring
        return scoring
    except ModuleNotFoundError as exc:
        if exc.name != "datasets":
            raise
    candidates = (
        REPO_ROOT / "timerlm" / "sft" / "tslm" / "src" / "datasets" / "anomaly_xl" / "scoring.py",
        Path.cwd() / "src" / "datasets" / "anomaly_xl" / "scoring.py",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError("official AnomalyXL scoring.py not found")
    spec = importlib.util.spec_from_file_location("l3_official_anomalyxl_scoring", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official scorer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_score_any(prediction: str, gold: str) -> dict[str, Any]:
    """CPU-testable entry point to the unmodified official scorer."""
    return _official_scoring().score_any(prediction, gold)


class RolloutAccounting:
    """Positive accounting: finalize fails unless every rollout was recorded."""

    def __init__(self, expected: int) -> None:
        if expected <= 0:
            raise ValueError("expected rollouts must be positive")
        self.expected = expected
        self.rows: list[tuple[RewardResult, int]] = []

    def record(self, reward: RewardResult, answer_tokens: int) -> None:
        if answer_tokens <= 0:
            raise ValueError("a rollout must contain at least one answer token")
        self.rows.append((reward, answer_tokens))

    def finalize(self) -> dict[str, Any]:
        if len(self.rows) != self.expected:
            raise RuntimeError(
                f"lost rollout: recorded {len(self.rows)}, expected {self.expected}"
            )
        primary = [row.primary for row, _ in self.rows]
        optimized = [row.optimization for row, _ in self.rows]
        adjustments = [row.format_adjustment for row, _ in self.rows]
        parsable = sum(row.parsable for row, _ in self.rows)
        return {
            "rollouts": len(self.rows),
            "parsable": parsable,
            "unparsable": len(self.rows) - parsable,
            "reward_mean": sum(primary) / len(primary),
            "reward_std": _population_std(primary),
            "reward_sum": sum(primary),
            "reward_square_sum": sum(value * value for value in primary),
            "optimization_reward_mean": sum(optimized) / len(optimized),
            "optimization_reward_std": _population_std(optimized),
            "format_adjustment_mean": sum(adjustments) / len(adjustments),
            "format_penalty": max(0.0, -sum(adjustments) / len(adjustments)),
            "tokens_answer": sum(tokens for _, tokens in self.rows),
            "tokens_answer_mean": sum(tokens for _, tokens in self.rows) / len(self.rows),
        }


def _population_std(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


class LoraReference:
    """CPU snapshot of va1 LoRA tensors, applied through an exact swap."""

    def __init__(self, model: torch.nn.Module) -> None:
        named = {name: parameter for name, parameter in model.named_parameters() if "lora_" in name}
        if not named:
            raise ValueError("model has no LoRA tensors")
        self.names = tuple(sorted(named))
        self.state_cpu = {
            name: named[name].detach().cpu().clone() for name in self.names
        }
        self._active = False

    @contextmanager
    def applied(self, model: torch.nn.Module) -> Iterator[None]:
        if self._active:
            raise RuntimeError("nested reference swap is not supported")
        named = dict(model.named_parameters())
        if set(self.names) - set(named):
            raise ValueError("reference/model LoRA names differ")
        backup = {name: named[name].detach().clone() for name in self.names}
        self._active = True
        try:
            with torch.no_grad():
                for name in self.names:
                    source, target = self.state_cpu[name], named[name]
                    if source.shape != target.shape or source.dtype != target.dtype:
                        raise ValueError(f"reference mismatch for {name}")
                    target.copy_(source.to(target.device))
            yield
        finally:
            with torch.no_grad():
                for name in self.names:
                    named[name].copy_(backup[name])
            self._active = False


def configure_lora_only(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        trainable = "lora_" in name
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append((name, parameter))
    if not selected:
        raise ValueError("no LoRA parameters selected")
    return selected


def _promote_lora_masters(
    named_lora: Sequence[tuple[str, torch.nn.Parameter]]
) -> None:
    if os.environ.get("TSLM_AMP") != "1":
        return
    for _, parameter in named_lora:
        if parameter.dtype in (torch.float16, torch.bfloat16):
            parameter.data = parameter.data.float()


def completion_log_probs(
    model: Any,
    prompt_inputs: Mapping[str, torch.Tensor],
    completion_ids: torch.Tensor,
    ts_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Teacher-forced log-probs for response tokens only, batch size one."""
    input_ids = prompt_inputs["input_ids"]
    attention = prompt_inputs["attention_mask"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("GRPO requires one prompt per step")
    if completion_ids.ndim != 2 or completion_ids.shape[0] != 1 or completion_ids.shape[1] < 1:
        raise ValueError("completion_ids must have shape (1, answer_tokens>=1)")
    prompt_len = input_ids.shape[1]
    full_ids = torch.cat((input_ids, completion_ids.to(input_ids.device)), dim=1)
    answer_attention = torch.ones_like(completion_ids, device=attention.device, dtype=attention.dtype)
    full_attention = torch.cat((attention, answer_attention), dim=1)
    if ts_features is None:
        ts_features = model._encode_and_project(prompt_inputs["time_series"])
    inputs_embeds = model._merge_features(ts_features, full_ids)
    keep = torch.arange(
        prompt_len - 1,
        prompt_len + completion_ids.shape[1] - 1,
        device=full_ids.device,
    )
    outputs = model.llm_model(
        inputs_embeds=inputs_embeds,
        attention_mask=full_attention,
        use_cache=False,
        logits_to_keep=keep,
    )
    logits = outputs.logits
    if logits.shape[1] != completion_ids.shape[1]:
        logits = logits.index_select(1, keep)
    targets = completion_ids.to(logits.device)
    return F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]


def _trim_completion(ids: torch.Tensor, eos_id: int | None, pad_id: int | None) -> torch.Tensor:
    row = ids[0].tolist()
    kept: list[int] = []
    for token in row:
        if pad_id is not None and token == pad_id and kept:
            break
        kept.append(token)
        if eos_id is not None and token == eos_id:
            break
    if not kept:
        raise RuntimeError("generation returned no response token")
    return torch.tensor([kept], dtype=ids.dtype, device=ids.device)


def generate_completion(
    model: Any,
    prompt_inputs: Mapping[str, torch.Tensor],
    rl: Mapping[str, Any],
    *,
    sample: bool,
) -> torch.Tensor:
    """L3 rollout wrapper; generation remains WaveletTSModel.generate."""
    kwargs: dict[str, Any] = {
        "do_sample": sample,
        "max_new_tokens": int(rl["max_new_tokens"]),
    }
    if sample:
        kwargs.update(
            temperature=float(rl["temperature"]),
            top_p=float(rl["top_p"]),
        )
    with _autocast_for(model):
        generated = model.generate(**prompt_inputs, **kwargs)
    return _trim_completion(
        generated, model.tokenizer.eos_token_id, model.tokenizer.pad_token_id
    )


def _gradient_checkpoint_train_mode(model: torch.nn.Module) -> None:
    """Enable checkpointing's training branch while removing dropout noise."""
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


@contextmanager
def _autocast_for(model: Any) -> Iterator[None]:
    device_type = next(model.parameters()).device.type
    enabled = os.environ.get("TSLM_AMP") == "1" and device_type == "cuda"
    dtype = getattr(torch, os.environ.get("TSLM_AMP_DTYPE", "bfloat16"))
    with torch.autocast(device_type, dtype=dtype, enabled=enabled):
        yield


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _gradient_norm(gradients: Sequence[torch.Tensor | None]) -> float:
    squares = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().item())


def _add_gradients(
    accumulator: list[torch.Tensor | None], gradients: Sequence[torch.Tensor | None]
) -> None:
    for index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        detached = gradient.detach()
        if accumulator[index] is None:
            accumulator[index] = detached.clone()
        else:
            accumulator[index].add_(detached)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _longrun_prompt_group(
    model: Any,
    reference: LoraReference,
    dataset: Any,
    sample: Mapping[str, Any],
    rl: Mapping[str, Any],
    step_index: int,
    prompt_ordinal: int,
    trainable: Sequence[torch.nn.Parameter],
    policy_accumulator: list[torch.Tensor | None],
    kl_accumulator: list[torch.Tensor | None],
    rollout_log: Any,
) -> dict[str, Any]:
    """Process one G8 group and release all prompt-sized state before returning."""
    group_size = int(rl["G"])
    accounting = RolloutAccounting(group_size)
    prompt_inputs = model.prepare_batch([sample], training=False)
    prompt_inputs = {key: value for key, value in prompt_inputs.items() if key != "labels"}
    prompt_tokens = int(prompt_inputs["attention_mask"].sum().item())
    model.eval()
    with torch.no_grad(), _autocast_for(model):
        shared_features = model._encode_and_project(prompt_inputs["time_series"])

    completions: list[torch.Tensor] = []
    old_log_probs: list[torch.Tensor] = []
    rewards: list[RewardResult] = []
    for rollout_ordinal in range(1, group_size + 1):
        model.eval()
        completion = generate_completion(model, prompt_inputs, rl, sample=True)
        text = model.tokenizer.batch_decode(completion, skip_special_tokens=True)[0]
        reward = reward_completion(dataset, text, sample, float(rl["format_penalty"]))
        _gradient_checkpoint_train_mode(model)
        with torch.no_grad(), _autocast_for(model):
            old_lp = completion_log_probs(model, prompt_inputs, completion, shared_features)
        if not torch.isfinite(old_lp).all():
            raise RuntimeError("old policy produced non-finite log-prob")
        completions.append(completion)
        old_log_probs.append(old_lp.detach().clone())
        rewards.append(reward)
        accounting.record(reward, completion.shape[1])
        _write_jsonl(rollout_log, {
            "step": step_index,
            "prompt": prompt_ordinal,
            "rollout": rollout_ordinal,
            "train_index": int(sample["_l3_train_index"]),
            "id": str(sample.get("id", "")),
            "category": str(sample.get(dataset.category_key, "unknown")),
            "reward_primary": reward.primary,
            "reward_optimization": reward.optimization,
            "parsable": reward.parsable,
            "response_tokens": int(completion.shape[1]),
            "text_sha256": _sha256_text(text),
        })

    reward_tensor = torch.tensor(
        [reward.optimization for reward in rewards],
        device=old_log_probs[0].device,
        dtype=torch.float32,
    )
    advantages, zero_variance = normalize_advantages(
        reward_tensor, float(rl["advantage_eps"])
    )
    ratio_values: list[torch.Tensor] = []
    max_log_ratios: list[torch.Tensor] = []
    kl_values: list[torch.Tensor] = []
    policy_losses: list[float] = []
    kl_losses: list[float] = []
    scale = 1.0 / (int(rl["prompts_per_step"]) * group_size)
    for completion, old_lp, advantage in zip(completions, old_log_probs, advantages):
        with reference.applied(model), torch.no_grad(), _autocast_for(model):
            ref_lp = completion_log_probs(model, prompt_inputs, completion, shared_features)
        with _autocast_for(model):
            current_lp = completion_log_probs(model, prompt_inputs, completion, shared_features)
            ratio = ppo_ratio(current_lp, old_lp)
            clipped = ratio.clamp(1.0 - float(rl["clip_eps"]), 1.0 + float(rl["clip_eps"]))
            policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
            kl_loss = k3_kl(current_lp, ref_lp).mean()
        log_ratio = max_abs_log_ratio(current_lp.detach(), old_lp)
        if not torch.isfinite(log_ratio) or log_ratio.item() > MAX_ABS_LOG_RATIO:
            raise RuntimeError(
                f"pre-update max |log ratio| {log_ratio.item():.6g} exceeds "
                f"{MAX_ABS_LOG_RATIO:.6g}"
            )
        if not torch.isfinite(policy_loss) or not torch.isfinite(kl_loss):
            raise RuntimeError("non-finite longrun loss")
        # Use ordinary backward, not autograd.grad: Hugging Face may enable
        # re-entrant gradient checkpointing, which explicitly rejects grad().
        # The two immediate backward passes share the forward graph and give
        # exact, separately reportable policy/KL norms.
        for parameter in trainable:
            parameter.grad = None
        (policy_loss * scale).backward(retain_graph=True)
        _add_gradients(policy_accumulator, [parameter.grad for parameter in trainable])
        for parameter in trainable:
            parameter.grad = None
        (kl_loss * scale).backward()
        _add_gradients(kl_accumulator, [parameter.grad for parameter in trainable])
        for parameter in trainable:
            parameter.grad = None
        ratio_values.append(ratio.detach())
        max_log_ratios.append(log_ratio)
        kl_values.append(kl_loss.detach())
        policy_losses.append(float(policy_loss.detach().item()))
        kl_losses.append(float(kl_loss.detach().item()))

    base = accounting.finalize()
    flat_ratios = torch.cat(ratio_values)
    result = {
        "prompt": prompt_ordinal,
        "train_index": int(sample["_l3_train_index"]),
        "id": str(sample.get("id", "")),
        "category": str(sample.get(dataset.category_key, "unknown")),
        "length": int(sample.get("length", 0)),
        "n_channels": int(sample.get("n_channels", 0)),
        **base,
        "informative": not zero_variance,
        "kl_mean": float(torch.stack(kl_values).mean().item()),
        "ratio_mean": float(flat_ratios.mean().item()),
        "ratio_all_exact_one": bool(torch.equal(flat_ratios, torch.ones_like(flat_ratios))),
        "max_abs_log_ratio": float(torch.stack(max_log_ratios).max().item()),
        "policy_loss_mean": sum(policy_losses) / len(policy_losses),
        "kl_loss_mean": sum(kl_losses) / len(kl_losses),
        "tokens_prompt": prompt_tokens,
    }
    # Explicitly destroy feature/completion tensors here: the caller does not
    # start prompt 2 until this function has returned.
    del completions, old_log_probs, shared_features, prompt_inputs
    gc.collect()
    return result


def zoom_episode_log_probs(model: Any, episode: Any) -> torch.Tensor:
    """Concatenate action/answer log-probs; environment-inserted text is masked."""
    segments: list[tuple[Mapping[str, torch.Tensor], torch.Tensor]] = []
    if episode.action_ids is not None:
        if episode.action_prompt is None:
            raise ValueError("zoom episode has action ids without its prompt")
        segments.append((episode.action_prompt, episode.action_ids))
    segments.append((episode.final_prompt, episode.final_ids))
    values = [completion_log_probs(
        model, prompt, ids, prompt.get("_l3_ts_features")
    ) for prompt, ids in segments]
    if any(value.numel() == 0 for value in values):
        raise RuntimeError("zoom episode contains an empty emitted segment")
    return torch.cat(values)


def _zoom_prompt_group(
    model: Any,
    reference: LoraReference,
    dataset: Any,
    sample: Mapping[str, Any],
    rl: Mapping[str, Any],
    step_index: int,
    prompt_ordinal: int,
    trainable: Sequence[torch.nn.Parameter],
    policy_accumulator: list[torch.Tensor | None],
    kl_accumulator: list[torch.Tensor | None],
    rollout_log: Any,
) -> dict[str, Any]:
    from l3.zoom import run_zoom_episode, zoom_iou_with_gold

    group_size = int(rl["G"])
    accounting = RolloutAccounting(group_size)
    episodes: list[Any] = []
    old_log_probs: list[torch.Tensor] = []
    rewards: list[RewardResult] = []
    prompt_tokens = 0
    zoom_count = 0
    zoom_invalid = 0
    zoom_lengths: list[int] = []
    zoom_ious: list[float] = []
    for rollout_ordinal in range(1, group_size + 1):
        model.eval()
        episode = run_zoom_episode(
            model, sample, sample_generation=True,
            max_action_tokens=int(rl.get("action_max_new_tokens", rl["max_new_tokens"])),
            max_answer_tokens=int(rl["max_new_tokens"]),
            temperature=float(rl["temperature"]), top_p=float(rl["top_p"]),
        )
        reward = reward_completion(
            dataset, episode.final_text, sample, float(rl["format_penalty"])
        )
        _gradient_checkpoint_train_mode(model)
        with torch.no_grad(), _autocast_for(model):
            old_lp = zoom_episode_log_probs(model, episode)
        if not torch.isfinite(old_lp).all():
            raise RuntimeError("old zoom policy produced non-finite log-prob")
        episodes.append(episode)
        old_log_probs.append(old_lp.detach().clone())
        rewards.append(reward)
        emitted_tokens = int(old_lp.numel())
        accounting.record(reward, emitted_tokens)
        prompt_tokens += int(episode.final_prompt["attention_mask"].sum().item())
        if episode.action_prompt is not None:
            prompt_tokens += int(episode.action_prompt["attention_mask"].sum().item())
        zoom_count += int(episode.zoomed)
        zoom_invalid += int(episode.zoom_invalid)
        if episode.zoomed:
            zoom_lengths.append(int(episode.action.window_len))
            zoom_ious.append(float(zoom_iou_with_gold(sample, episode.action)))
        _write_jsonl(rollout_log, {
            "step": step_index, "prompt": prompt_ordinal, "rollout": rollout_ordinal,
            "train_index": int(sample["_l3_train_index"]),
            "id": str(sample.get("id", "")),
            "category": str(sample.get(dataset.category_key, "unknown")),
            "reward_primary": reward.primary, "reward_optimization": reward.optimization,
            "parsable": reward.parsable, "response_tokens": int(episode.final_ids.shape[1]),
            "action_tokens": 0 if episode.action_ids is None else int(episode.action_ids.shape[1]),
            "zoom_valid": episode.zoomed, "zoom_invalid": episode.zoom_invalid,
            "zoom_window_len": int(episode.action.window_len) if episode.zoomed else 0,
            "zoom_iou_with_gold": zoom_ious[-1] if episode.zoomed else 0.0,
            "text_sha256": _sha256_text(episode.final_text),
            "action_text_sha256": (_sha256_text(episode.action_text)
                                   if episode.action_text is not None else None),
        })

    reward_tensor = torch.tensor(
        [reward.optimization for reward in rewards], device=old_log_probs[0].device,
        dtype=torch.float32,
    )
    advantages, zero_variance = normalize_advantages(
        reward_tensor, float(rl["advantage_eps"])
    )
    ratio_values: list[torch.Tensor] = []
    max_log_ratios: list[torch.Tensor] = []
    kl_values: list[torch.Tensor] = []
    policy_losses: list[float] = []
    kl_losses: list[float] = []
    scale = 1.0 / (int(rl["prompts_per_step"]) * group_size)
    for episode, old_lp, advantage in zip(episodes, old_log_probs, advantages):
        with reference.applied(model), torch.no_grad(), _autocast_for(model):
            ref_lp = zoom_episode_log_probs(model, episode)
        with _autocast_for(model):
            current_lp = zoom_episode_log_probs(model, episode)
            ratio = ppo_ratio(current_lp, old_lp)
            clipped = ratio.clamp(1.0 - float(rl["clip_eps"]),
                                  1.0 + float(rl["clip_eps"]))
            policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
            kl_loss = k3_kl(current_lp, ref_lp).mean()
        log_ratio = max_abs_log_ratio(current_lp.detach(), old_lp)
        if not torch.isfinite(log_ratio) or log_ratio.item() > MAX_ABS_LOG_RATIO:
            raise RuntimeError(
                f"pre-update zoom max |log ratio| {log_ratio.item():.6g} exceeds "
                f"{MAX_ABS_LOG_RATIO:.6g}"
            )
        if not torch.isfinite(policy_loss) or not torch.isfinite(kl_loss):
            raise RuntimeError("non-finite zoom GRPO loss")
        for parameter in trainable:
            parameter.grad = None
        (policy_loss * scale).backward(retain_graph=True)
        _add_gradients(policy_accumulator, [p.grad for p in trainable])
        for parameter in trainable:
            parameter.grad = None
        (kl_loss * scale).backward()
        _add_gradients(kl_accumulator, [p.grad for p in trainable])
        for parameter in trainable:
            parameter.grad = None
        ratio_values.append(ratio.detach())
        max_log_ratios.append(log_ratio)
        kl_values.append(kl_loss.detach())
        policy_losses.append(float(policy_loss.detach().item()))
        kl_losses.append(float(kl_loss.detach().item()))

    base = accounting.finalize()
    ratios = torch.cat(ratio_values)
    return {
        "prompt": prompt_ordinal,
        "train_index": int(sample["_l3_train_index"]),
        "id": str(sample.get("id", "")),
        "category": str(sample.get(dataset.category_key, "unknown")),
        "length": int(sample.get("length", 0)), "n_channels": int(sample.get("n_channels", 0)),
        **base, "informative": not zero_variance,
        "kl_mean": float(torch.stack(kl_values).mean().item()),
        "ratio_mean": float(ratios.mean().item()),
        "ratio_all_exact_one": bool(torch.equal(ratios, torch.ones_like(ratios))),
        "max_abs_log_ratio": float(torch.stack(max_log_ratios).max().item()),
        "policy_loss_mean": sum(policy_losses) / len(policy_losses),
        "kl_loss_mean": sum(kl_losses) / len(kl_losses),
        "tokens_prompt": prompt_tokens, "zoom_count": zoom_count,
        "zoom_invalid": zoom_invalid,
        "zoom_window_len_sum": sum(zoom_lengths),
        "zoom_iou_sum": sum(zoom_ious),
    }


def grpo_zoom_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    reference: LoraReference,
    dataset: Any,
    samples: Sequence[Mapping[str, Any]],
    rl: Mapping[str, Any],
    step_index: int,
    rollout_log: Any,
) -> dict[str, Any]:
    """P2/G8 two-turn GRPO update with separate action and answer forwards."""
    started = time.perf_counter()
    if len(samples) != int(rl["prompts_per_step"]):
        raise ValueError("zoom step received the wrong prompt count")
    if any(str(sample.get(dataset.category_key)) not in ZOOM_RL_CATEGORIES for sample in samples):
        raise ValueError("excluded category reached zoom RL")
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    policy_accumulator: list[torch.Tensor | None] = [None] * len(trainable)
    kl_accumulator: list[torch.Tensor | None] = [None] * len(trainable)
    groups = [
        _zoom_prompt_group(model, reference, dataset, sample, rl, step_index, ordinal,
                           trainable, policy_accumulator, kl_accumulator, rollout_log)
        for ordinal, sample in enumerate(samples, start=1)
    ]
    policy_grad_norm = _gradient_norm(policy_accumulator)
    kl_grad_norm = _gradient_norm(kl_accumulator)
    with torch.no_grad():
        for parameter, policy_gradient, kl_gradient in zip(
            trainable, policy_accumulator, kl_accumulator
        ):
            combined = torch.zeros_like(parameter)
            if policy_gradient is not None:
                combined.add_(policy_gradient)
            if kl_gradient is not None:
                combined.add_(kl_gradient, alpha=float(rl["kl_coef"]))
            parameter.grad = combined
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(rl["max_grad_norm"]))
    if not torch.isfinite(grad_norm):
        raise RuntimeError("non-finite zoom gradient norm")
    optimizer.step()
    rollouts = sum(int(group["rollouts"]) for group in groups)
    reward_sum = sum(float(group["reward_sum"]) for group in groups)
    reward_square_sum = sum(float(group["reward_square_sum"]) for group in groups)
    zoom_count = sum(int(group["zoom_count"]) for group in groups)
    by_category: dict[str, dict[str, float]] = {}
    for group in groups:
        bucket = by_category.setdefault(group["category"], {
            "episodes": 0, "zoom_count": 0, "zoom_invalid": 0,
            "zoom_window_len_sum": 0.0, "zoom_iou_sum": 0.0,
        })
        bucket["episodes"] += group["rollouts"]
        bucket["zoom_count"] += group["zoom_count"]
        bucket["zoom_invalid"] += group["zoom_invalid"]
        bucket["zoom_window_len_sum"] += group["zoom_window_len_sum"]
        bucket["zoom_iou_sum"] += group["zoom_iou_sum"]
    for bucket in by_category.values():
        z = max(1, int(bucket["zoom_count"]))
        bucket["zoom_rate"] = bucket["zoom_count"] / bucket["episodes"]
        bucket["zoom_window_len"] = bucket["zoom_window_len_sum"] / z
        bucket["zoom_iou_with_gold"] = bucket["zoom_iou_sum"] / z
    informative_groups = sum(bool(group["informative"]) for group in groups)
    row = {
        "step": step_index,
        "prompts": [{key: group[key] for key in (
            "prompt", "train_index", "id", "category", "length", "n_channels",
            "reward_mean", "reward_std", "parsable", "unparsable", "informative",
        )} for group in groups],
        "categories": [group["category"] for group in groups],
        "rollouts": rollouts,
        "parsable": sum(int(group["parsable"]) for group in groups),
        "unparsable": sum(int(group["unparsable"]) for group in groups),
        "reward_sum": reward_sum, "reward_square_sum": reward_square_sum,
        "reward_mean": reward_sum / rollouts,
        "reward_std": math.sqrt(max(0.0, reward_square_sum / rollouts -
                                    (reward_sum / rollouts) ** 2)),
        "optimization_reward_mean": sum(
            float(group["optimization_reward_mean"]) * group["rollouts"] for group in groups
        ) / rollouts,
        "informative_groups": informative_groups,
        "policy_informative_steps": int(informative_groups > 0),
        "zero_variance_groups": len(groups) - informative_groups,
        "kl_mean": sum(group["kl_mean"] for group in groups) / len(groups),
        "ratio_mean": sum(group["ratio_mean"] for group in groups) / len(groups),
        "ratio_all_exact_one": all(group["ratio_all_exact_one"] for group in groups),
        "max_abs_log_ratio": max(group["max_abs_log_ratio"] for group in groups),
        "policy_grad_norm": policy_grad_norm, "kl_grad_norm": kl_grad_norm,
        "grad_norm": float(grad_norm.item()),
        "tokens_prompt": sum(group["tokens_prompt"] for group in groups),
        "tokens_answer": sum(group["tokens_answer"] for group in groups),
        "zoom_rate": zoom_count / rollouts,
        "zoom_invalid": sum(group["zoom_invalid"] for group in groups),
        "zoom_window_len": sum(group["zoom_window_len_sum"] for group in groups) /
                           max(1, zoom_count),
        "zoom_iou_with_gold": sum(group["zoom_iou_sum"] for group in groups) /
                              max(1, zoom_count),
        "zoom_by_category": by_category,
        "peak_mem_mib": (float(torch.cuda.max_memory_allocated() / 1024**2)
                         if torch.cuda.is_available() else 0.0),
        "s_per_step": time.perf_counter() - started,
    }
    for key in ("reward_mean", "reward_std", "kl_mean", "ratio_mean",
                "max_abs_log_ratio", "policy_grad_norm", "kl_grad_norm", "grad_norm",
                "zoom_rate", "zoom_window_len", "zoom_iou_with_gold", "peak_mem_mib",
                "s_per_step"):
        if not _finite_number(row[key]):
            raise RuntimeError(f"non-finite zoom step field {key}")
    return row


def grpo_longrun_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    reference: LoraReference,
    dataset: Any,
    samples: Sequence[Mapping[str, Any]],
    rl: Mapping[str, Any],
    step_index: int,
    rollout_log: Any,
) -> dict[str, Any]:
    """Two sequential prompt groups, one averaged-gradient optimizer update."""
    started = time.perf_counter()
    if len(samples) != int(rl["prompts_per_step"]):
        raise ValueError("longrun step received the wrong prompt count")
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    policy_accumulator: list[torch.Tensor | None] = [None] * len(trainable)
    kl_accumulator: list[torch.Tensor | None] = [None] * len(trainable)
    groups = []
    for prompt_ordinal, sample in enumerate(samples, start=1):
        groups.append(_longrun_prompt_group(
            model, reference, dataset, sample, rl, step_index, prompt_ordinal,
            trainable, policy_accumulator, kl_accumulator, rollout_log,
        ))

    policy_grad_norm = _gradient_norm(policy_accumulator)
    kl_grad_norm = _gradient_norm(kl_accumulator)
    beta = float(rl["kl_coef"])
    with torch.no_grad():
        for parameter, policy_gradient, kl_gradient in zip(
            trainable, policy_accumulator, kl_accumulator
        ):
            combined = torch.zeros_like(parameter)
            if policy_gradient is not None:
                combined.add_(policy_gradient)
            if kl_gradient is not None:
                combined.add_(kl_gradient, alpha=beta)
            parameter.grad = combined
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(rl["max_grad_norm"]))
    if not torch.isfinite(grad_norm):
        raise RuntimeError("non-finite gradient norm")
    optimizer.step()

    rollouts = sum(int(group["rollouts"]) for group in groups)
    reward_sum = sum(float(group["reward_sum"]) for group in groups)
    reward_square_sum = sum(float(group["reward_square_sum"]) for group in groups)
    informative_groups = sum(bool(group["informative"]) for group in groups)
    row = {
        "step": step_index,
        "prompts": [
            {key: group[key] for key in (
                "prompt", "train_index", "id", "category", "length", "n_channels",
                "reward_mean", "reward_std", "parsable", "unparsable", "informative",
            )}
            for group in groups
        ],
        "categories": [str(group["category"]) for group in groups],
        "rollouts": rollouts,
        "parsable": sum(int(group["parsable"]) for group in groups),
        "unparsable": sum(int(group["unparsable"]) for group in groups),
        "reward_sum": reward_sum,
        "reward_square_sum": reward_square_sum,
        "reward_mean": reward_sum / rollouts,
        "reward_std": math.sqrt(max(0.0, reward_square_sum / rollouts - (reward_sum / rollouts) ** 2)),
        "optimization_reward_mean": sum(
            float(group["optimization_reward_mean"]) * int(group["rollouts"]) for group in groups
        ) / rollouts,
        "informative_groups": informative_groups,
        "policy_informative_steps": int(informative_groups > 0),
        "zero_variance_groups": len(groups) - informative_groups,
        "kl_mean": sum(float(group["kl_mean"]) for group in groups) / len(groups),
        "ratio_mean": sum(float(group["ratio_mean"]) for group in groups) / len(groups),
        "ratio_all_exact_one": all(bool(group["ratio_all_exact_one"]) for group in groups),
        "max_abs_log_ratio": max(float(group["max_abs_log_ratio"]) for group in groups),
        "policy_grad_norm": policy_grad_norm,
        "kl_grad_norm": kl_grad_norm,
        "grad_norm": float(grad_norm.item()),
        "tokens_prompt": sum(int(group["tokens_prompt"]) for group in groups),
        "tokens_answer": sum(int(group["tokens_answer"]) for group in groups),
        "peak_mem_mib": (
            float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
        ),
        "s_per_step": time.perf_counter() - started,
    }
    for key in (
        "reward_mean", "reward_std", "kl_mean", "ratio_mean", "max_abs_log_ratio",
        "policy_grad_norm", "kl_grad_norm", "grad_norm", "peak_mem_mib", "s_per_step",
    ):
        if not _finite_number(row[key]):
            raise RuntimeError(f"non-finite longrun step field {key}")
    return row


def grpo_step(
    model: Any,
    optimizer: torch.optim.Optimizer,
    reference: LoraReference,
    dataset: Any,
    sample: Mapping[str, Any],
    rl: Mapping[str, Any],
    step_index: int,
) -> dict[str, Any]:
    """Run G sequential rollouts and exactly one optimizer update."""
    started = time.perf_counter()
    group_size = int(rl["G"])
    accounting = RolloutAccounting(group_size)
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prompt_inputs = model.prepare_batch([sample], training=False)
    prompt_inputs = {key: value for key, value in prompt_inputs.items() if key != "labels"}
    prompt_tokens = int(prompt_inputs["attention_mask"].sum().item())
    model.eval()
    with torch.no_grad(), _autocast_for(model):
        shared_features = model._encode_and_project(prompt_inputs["time_series"])

    completions: list[torch.Tensor] = []
    old_log_probs: list[torch.Tensor] = []
    rewards: list[RewardResult] = []
    tokenizer = model.tokenizer
    for _ in range(group_size):
        model.eval()
        completion = generate_completion(model, prompt_inputs, rl, sample=True)
        text = tokenizer.batch_decode(completion, skip_special_tokens=True)[0]
        reward = reward_completion(dataset, text, sample, float(rl["format_penalty"]))
        _gradient_checkpoint_train_mode(model)
        with torch.no_grad(), _autocast_for(model):
            old_lp = completion_log_probs(model, prompt_inputs, completion, shared_features)
        if not torch.isfinite(old_lp).all():
            raise RuntimeError("old policy produced non-finite log-prob")
        completions.append(completion)
        old_log_probs.append(old_lp.detach().clone())
        rewards.append(reward)
        accounting.record(reward, completion.shape[1])

    reward_tensor = torch.tensor(
        [reward.optimization for reward in rewards],
        device=old_log_probs[0].device,
        dtype=torch.float32,
    )
    advantages, zero_variance = normalize_advantages(
        reward_tensor, float(rl["advantage_eps"])
    )

    ratio_values: list[torch.Tensor] = []
    max_abs_log_ratios: list[torch.Tensor] = []
    kl_values: list[torch.Tensor] = []
    losses: list[float] = []
    for completion, old_lp, advantage in zip(completions, old_log_probs, advantages):
        with reference.applied(model), torch.no_grad(), _autocast_for(model):
            ref_lp = completion_log_probs(model, prompt_inputs, completion, shared_features)
        with _autocast_for(model):
            current_lp = completion_log_probs(model, prompt_inputs, completion, shared_features)
            ratio = ppo_ratio(current_lp, old_lp)
            clipped = ratio.clamp(1.0 - float(rl["clip_eps"]), 1.0 + float(rl["clip_eps"]))
            policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
            kl = k3_kl(current_lp, ref_lp).mean()
            loss = policy_loss + float(rl["kl_coef"]) * kl
        log_ratio_max = max_abs_log_ratio(current_lp.detach(), old_lp)
        if not torch.isfinite(log_ratio_max) or log_ratio_max.item() > MAX_ABS_LOG_RATIO:
            raise RuntimeError(
                "pre-update max |log ratio| "
                f"{log_ratio_max.item():.6g} exceeds {MAX_ABS_LOG_RATIO:.6g}"
            )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite GRPO loss")
        (loss / group_size).backward()
        ratio_values.append(ratio.detach())
        max_abs_log_ratios.append(log_ratio_max)
        kl_values.append(kl.detach())
        losses.append(float(loss.detach().item()))

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float(rl["max_grad_norm"]))
    if not torch.isfinite(grad_norm):
        raise RuntimeError("non-finite gradient norm")
    optimizer.step()

    base = accounting.finalize()
    flat_ratios = torch.cat(ratio_values)
    category = str(sample.get(dataset.category_key, "unknown"))
    result = {
        "step": step_index,
        "train_index": int(sample["_l3_train_index"]),
        "id": str(sample.get("id", "")),
        "category": category,
        "length": int(sample.get("length", 0)),
        "n_channels": int(sample.get("n_channels", 0)),
        **base,
        "by_category": {
            category: {
                "rollouts": base["rollouts"],
                "parsable": base["parsable"],
                "unparsable": base["unparsable"],
                "reward_mean": base["reward_mean"],
                "reward_std": base["reward_std"],
            }
        },
        "zero_variance_groups": int(zero_variance),
        "kl_mean": float(torch.stack(kl_values).mean().item()),
        "ratio_mean": float(flat_ratios.mean().item()),
        "ratio_all_exact_one": bool(torch.equal(flat_ratios, torch.ones_like(flat_ratios))),
        "max_abs_log_ratio": float(torch.stack(max_abs_log_ratios).max().item()),
        "grad_norm": float(grad_norm.item()),
        "loss_mean": sum(losses) / len(losses),
        "tokens_prompt": prompt_tokens,
        "peak_mem_mib": (
            float(torch.cuda.max_memory_allocated() / 1024**2)
            if torch.cuda.is_available() else 0.0
        ),
        "s_per_step": time.perf_counter() - started,
    }
    if not all(_finite_number(result[key]) for key in (
        "reward_mean", "reward_std", "kl_mean", "ratio_mean", "max_abs_log_ratio", "grad_norm",
        "peak_mem_mib", "s_per_step",
    )):
        raise RuntimeError("step counters contain a non-finite value")
    return result


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _atomic_json_dump(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    _json_dump(temporary, value)
    os.replace(temporary, path)


def _write_jsonl(handle: Any, value: Any) -> None:
    handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_prompt_order(indices: Sequence[int], seed: int) -> list[int]:
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("pool indices must be non-empty and unique")
    order = [int(index) for index in indices]
    random.Random(seed).shuffle(order)
    return order


def prompt_indices_from_cursor(
    order: Sequence[int], cursor: int, count: int
) -> tuple[list[int], int]:
    if not order or cursor < 0 or count < 0:
        raise ValueError("invalid prompt cursor request")
    selected = [int(order[(cursor + offset) % len(order)]) for offset in range(count)]
    return selected, cursor + count


def stratified_candidate_indices(
    metadata: Sequence[Mapping[str, Any]], seed: int, limit: int
) -> list[int]:
    """Round-robin shuffled (category, length) strata without replacement."""
    strata: dict[tuple[str, int], list[int]] = {}
    for row in metadata:
        category = str(row["category"])
        length = int(row["length"])
        index = int(row["index"])
        if category in REQUIRED_CATEGORIES:
            strata.setdefault((category, length), []).append(index)
    rng = random.Random(seed)
    for indices in strata.values():
        rng.shuffle(indices)
    keys = sorted(strata)
    rng.shuffle(keys)
    result: list[int] = []
    while len(result) < limit:
        added = False
        for key in keys:
            if strata[key] and len(result) < limit:
                result.append(strata[key].pop())
                added = True
        if not added:
            break
    return result


def select_balanced_pool(
    accepted: Sequence[Mapping[str, Any]], per_category: int
) -> list[dict[str, Any]]:
    """Choose equal category counts while cycling available length strata."""
    output: list[dict[str, Any]] = []
    for category in sorted(REQUIRED_CATEGORIES):
        rows = [dict(row) for row in accepted if row.get("category") == category]
        by_length: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_length.setdefault(int(row["length"]), []).append(row)
        chosen: list[dict[str, Any]] = []
        lengths = sorted(by_length)
        while len(chosen) < per_category:
            added = False
            for length in lengths:
                if by_length[length] and len(chosen) < per_category:
                    chosen.append(by_length[length].pop(0))
                    added = True
            if not added:
                break
        if len(chosen) < per_category:
            raise RuntimeError(
                f"pool cap did not yield {per_category} informative prompts for {category}"
            )
        output.extend(chosen)
    return output


def resampled_zoom_pool(
    metadata: Sequence[Mapping[str, Any]], seed: int, *,
    pool_size: int = 200,
    excluded_categories: Sequence[str] = ("lead_lag_with_magnitude",),
) -> list[int]:
    """Sample a fresh balanced pool, cycling lengths within each category."""
    allowed = sorted(REQUIRED_CATEGORIES - set(excluded_categories))
    if not allowed or pool_size % len(allowed):
        raise ValueError("zoom pool size must divide evenly across allowed categories")
    quota = pool_size // len(allowed)
    rng = random.Random(int(seed))
    chunks: list[list[int]] = []
    for category in allowed:
        by_length: dict[int, list[int]] = {}
        for row in metadata:
            if str(row.get("category")) == category:
                by_length.setdefault(int(row.get("length", 0)), []).append(int(row["index"]))
        if not by_length:
            raise ValueError(f"no train prompts for zoom category {category}")
        lengths = sorted(by_length)
        for length in lengths:
            rng.shuffle(by_length[length])
        cursor = {length: 0 for length in lengths}
        selected: list[int] = []
        while len(selected) < quota:
            progressed = False
            for length in lengths:
                position = cursor[length]
                if position < len(by_length[length]):
                    selected.append(by_length[length][position])
                    cursor[length] += 1
                    progressed = True
                    if len(selected) == quota:
                        break
            if not progressed:
                raise ValueError(f"insufficient unique prompts for {category}: need {quota}")
        chunks.append(selected)
    return [chunks[cat][offset] for offset in range(quota) for cat in range(len(allowed))]


def zoom_pool_for_round(
    metadata: Sequence[Mapping[str, Any]], base_seed: int, round_index: int,
    *, pool_size: int = 200,
    excluded_categories: Sequence[str] = ("lead_lag_with_magnitude",),
) -> list[int]:
    if round_index < 0:
        raise ValueError("round_index must be non-negative")
    return resampled_zoom_pool(
        metadata, int(base_seed) + int(round_index), pool_size=pool_size,
        excluded_categories=excluded_categories,
    )


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.get("torch_cuda", []))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def truncate_jsonl_after_step(path: Path, step: int) -> None:
    rows = [row for row in _read_jsonl(path) if int(row["step"]) <= step]
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            _write_jsonl(handle, row)
    os.replace(temporary, path)


def _runtime_imports(resolved: Mapping[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    # On the node grpo.py is run from /root/timerlm/sft/tslm. Locally the ignored
    # upstream checkout lives under timerlm/sft/tslm.
    local_checkout = REPO_ROOT / "timerlm" / "sft" / "tslm"
    if (local_checkout / "train.py").is_file():
        sys.path.insert(0, str(local_checkout))
    sys.path.insert(0, str(Path.cwd()))
    from train import ExperimentConfig, configure_training_mode, load_checkpoint, load_model
    from src.datasets.anomaly_xl.qa_dataset import (
        AnomalyXLDataset,
        _load_parquet_as_hf,
    )

    class TrainOnlyAnomalyXLDataset(AnomalyXLDataset):
        def _load_splits(self):
            train = _load_parquet_as_hf(Path(self.train_dir) / "train.parquet")
            empty = train.select([])
            return train, empty, empty

    return ExperimentConfig, configure_training_mode, load_checkpoint, load_model, TrainOnlyAnomalyXLDataset


def _load_policy_and_train(
    resolved: dict[str, Any], checkpoint_path: Path, *, promote_lora: bool = True
) -> tuple[Any, Any, list[str], list[str], str]:
    expected_sha = resolved["rl"].get("initial_checkpoint_sha256")
    observed_sha = sha256_file(checkpoint_path)
    if expected_sha and observed_sha != expected_sha:
        raise RuntimeError(f"initial checkpoint SHA mismatch: {observed_sha}")
    ExperimentConfig, configure_training_mode, load_checkpoint, load_model, DatasetClass = _runtime_imports(resolved)
    experiment = ExperimentConfig.from_dict({k: v for k, v in resolved.items() if not k.startswith("_")})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Preserve the va1/smoke loader seed.  New long-run entry points reseed
    # explicitly after loading, before their first stochastic operation.
    torch.manual_seed(int(resolved["seed"]))
    random.seed(int(resolved["seed"]))
    model = load_model(experiment, device)
    configure_training_mode(model, experiment)
    lora = configure_lora_only(model)
    if promote_lora:
        _promote_lora_masters(lora)
    os.environ["TSLM_SAVE_TRAINABLE_ONLY"] = "1"
    checkpoint = load_checkpoint(str(checkpoint_path), model)
    composed_names = sorted(checkpoint["model_state"])
    optimized_names = sorted(name for name, _ in lora)
    if len(optimized_names) != 32:
        raise RuntimeError(f"expected 32 LoRA tensors, found {len(optimized_names)}")
    if len(composed_names) != 42 or not set(optimized_names).issubset(composed_names):
        raise RuntimeError("initial checkpoint is not the expected 42-tensor va1 slim state")

    kwargs = dict(experiment.dataset.extra_kwargs)
    kwargs["EOS_TOKEN"] = model.get_eos_token()
    dataset = DatasetClass(split="train", **kwargs)
    if len(dataset) != int(resolved["rl"]["expected_train_rows"]):
        raise RuntimeError(f"train row count drift: {len(dataset)}")
    train_path = Path(dataset.train_dir) / "train.parquet"
    observed_train_sha = sha256_file(train_path)
    if observed_train_sha != resolved["rl"]["train_sha256"]:
        raise RuntimeError(f"train parquet SHA mismatch: {observed_train_sha}")
    return model, dataset, optimized_names, composed_names, observed_train_sha


def _checked_sample(dataset: Any, index: int, rl: Mapping[str, Any]) -> dict[str, Any]:
    sample = dataset[index]
    actual = {
        "id": str(sample.get("id", "")),
        "category": str(sample.get("category", "")),
        "length": int(sample.get("length", 0)),
        "n_channels": int(sample.get("n_channels", 0)),
    }
    expected = rl.get("train_prompt_expectations", {}).get(str(index))
    if expected is not None:
        normalized_expected = {
            "id": str(expected["id"]),
            "category": str(expected["category"]),
            "length": int(expected["length"]),
            "n_channels": int(expected["n_channels"]),
        }
        if actual != normalized_expected:
            raise RuntimeError(f"train metadata drift at {index}: {actual} != {normalized_expected}")
    if actual["category"] not in REQUIRED_CATEGORIES:
        raise RuntimeError(f"unexpected train category at {index}: {actual['category']}")
    sample["_l3_train_index"] = index
    return sample


def _resolved_rl(resolved: Mapping[str, Any]) -> dict[str, Any]:
    rl = dict(resolved["rl"])
    rl["max_new_tokens"] = int(resolved["training"].get("eval_max_new_tokens", 500))
    return rl


def _init_check(args: argparse.Namespace) -> None:
    output = Path(args.out).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved = load_l3_config(args.config)
    checkpoint_path = Path(args.checkpoint).expanduser()
    model, dataset, optimized, _, train_sha = _load_policy_and_train(
        resolved, checkpoint_path, promote_lora=False
    )
    rl = _resolved_rl(resolved)
    model.eval()
    baselines: dict[int, torch.Tensor] = {}
    for index in rl["init_prompt_indices"]:
        sample = _checked_sample(dataset, index, rl)
        prompt = model.prepare_batch([sample], training=False)
        prompt = {key: value for key, value in prompt.items() if key != "labels"}
        with _autocast_for(model):
            raw = model.generate(**prompt, do_sample=False, max_new_tokens=rl["max_new_tokens"])
        baselines[index] = _trim_completion(
            raw, model.tokenizer.eos_token_id, model.tokenizer.pad_token_id
        ).cpu()

    named_lora = [(name, parameter) for name, parameter in model.named_parameters() if "lora_" in name]
    _promote_lora_masters(named_lora)
    _, _, load_checkpoint, _, _ = _runtime_imports(resolved)
    load_checkpoint(str(checkpoint_path), model)
    reference = LoraReference(model)
    model.eval()
    rows = []
    for index in rl["init_prompt_indices"]:
        sample = _checked_sample(dataset, index, rl)
        prompt = model.prepare_batch([sample], training=False)
        prompt = {key: value for key, value in prompt.items() if key != "labels"}
        baseline = baselines[index]
        wrapped = generate_completion(model, prompt, rl, sample=False)
        wrapped_cpu = wrapped.cpu()
        token_equal = torch.equal(baseline, wrapped_cpu)
        completion = wrapped
        with torch.no_grad(), _autocast_for(model):
            features = model._encode_and_project(prompt["time_series"])
            old_lp = completion_log_probs(model, prompt, completion, features)
        with reference.applied(model), torch.no_grad(), _autocast_for(model):
            ref_lp = completion_log_probs(model, prompt, completion, features)
        equal_lp = torch.equal(old_lp, ref_lp)
        ratio = ppo_ratio(old_lp, old_lp)
        rows.append({
            "train_index": index,
            "id": sample["id"],
            "category": sample["category"],
            "tokens_equal": token_equal,
            "text_equal": model.tokenizer.decode(baseline[0], skip_special_tokens=True)
                == model.tokenizer.decode(wrapped_cpu[0], skip_special_tokens=True),
            "baseline_text": model.tokenizer.decode(baseline[0], skip_special_tokens=True),
            "wrapper_text": model.tokenizer.decode(wrapped_cpu[0], skip_special_tokens=True),
            "logprob_equal_reference": equal_lp,
            "max_logprob_diff": float((old_lp - ref_lp).abs().max().item()),
            "ratio_mean": float(ratio.mean().item()),
            "ratio_all_exact_one": bool(torch.equal(ratio, torch.ones_like(ratio))),
        })
    ok = all(
        row["tokens_equal"] and row["text_equal"]
        and row["max_logprob_diff"] <= MAX_ABS_LOG_RATIO
        for row in rows
    )
    report = {
        "stage": ("z2_init_equals_vz1" if rl.get("contract") == "zoom"
                  else "b_init_equals_va1"),
        "ok": ok,
        "checkpoint_sha256": sha256_file(Path(args.checkpoint).expanduser()),
        "train_sha256": train_sha,
        "optimized_tensor_count": len(optimized),
        "rows": rows,
    }
    _json_dump(output, report)
    if not ok:
        raise RuntimeError("init identity gate failed")


def _save_composed_checkpoint(
    path: Path,
    model: Any,
    composed_names: Sequence[str],
    optimized_names: Sequence[str],
    resolved: Mapping[str, Any],
    steps: int,
    *,
    cursor: int | None = None,
) -> None:
    state = model.state_dict()
    missing = set(composed_names) - set(state)
    if missing:
        raise RuntimeError(f"cannot compose checkpoint; missing keys {sorted(missing)}")
    payload = {
        "model_state": {name: state[name].detach().cpu().clone() for name in composed_names},
        "optimizer_state": None,
        "scheduler_state": None,
        "epoch": 0,
        "batch_idx": steps,
        "val_loss": 0.0,
        "best_val_metric": None,
        "config": {k: v for k, v in resolved.items() if not k.startswith("_")},
        "l3": {
            "format": "l3_grpo_composed_slim_v1",
            "steps": steps,
            "optimized_names": list(optimized_names),
            "composed_slim_names": list(composed_names),
            "pool_cursor": cursor,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _save_lora_checkpoint(
    path: Path,
    model: Any,
    optimized_names: Sequence[str],
    resolved: Mapping[str, Any],
    steps: int,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    cursor: int | None = None,
    rng_state: Mapping[str, Any] | None = None,
    cumulative: Mapping[str, Any] | None = None,
) -> None:
    state = model.state_dict()
    if set(optimized_names) - set(state):
        raise RuntimeError("cannot save LoRA-only checkpoint: parameter missing")
    payload = {
        "model_state": {
            name: state[name].detach().cpu().clone() for name in optimized_names
        },
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": None,
        "epoch": 0,
        "batch_idx": steps,
        "val_loss": 0.0,
        "best_val_metric": None,
        "config": {k: v for k, v in resolved.items() if not k.startswith("_")},
        "l3": {
            "format": "l3_grpo_lora_slim_v1",
            "steps": steps,
            "optimized_names": list(optimized_names),
            "pool_cursor": cursor,
            "rng_state": dict(rng_state) if rng_state is not None else None,
            "cumulative": dict(cumulative) if cumulative is not None else None,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def latest_valid_checkpoint(
    run_dir: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    records = manifest.get("checkpoints")
    if not isinstance(records, list):
        raise RuntimeError("run manifest has no checkpoint records")
    for record in reversed(records):
        try:
            lora_path = run_dir / str(record["lora_file"])
            composed_path = run_dir / str(record["composed_file"])
            if sha256_file(lora_path) != record["lora_sha256"]:
                continue
            if sha256_file(composed_path) != record["composed_sha256"]:
                continue
            payload = torch.load(lora_path, map_location="cpu", weights_only=False)
            l3_state = payload.get("l3", {})
            if int(l3_state.get("steps", -1)) != int(record["step"]):
                continue
            if int(l3_state.get("pool_cursor", -1)) != int(record["pool_cursor"]):
                continue
            if payload.get("optimizer_state") is None or l3_state.get("rng_state") is None:
                continue
            return dict(record), payload
        except Exception:
            continue
    raise RuntimeError("no valid resumable checkpoint found")


def _checkpoint_longrun(
    run_dir: Path,
    manifest: dict[str, Any],
    model: Any,
    optimizer: torch.optim.Optimizer,
    optimized: Sequence[str],
    composed: Sequence[str],
    resolved: Mapping[str, Any],
    step: int,
    cursor: int,
    cumulative: Mapping[str, Any],
) -> dict[str, Any]:
    current_trainable = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if current_trainable != sorted(optimized):
        raise RuntimeError("LoRA-only trainable-name invariant failed before checkpoint")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    stem = f"step_{step:06d}"
    lora_path = checkpoint_dir / f"{stem}_lora.pt"
    composed_path = checkpoint_dir / f"{stem}_composed.pt"
    if lora_path.exists() or composed_path.exists():
        raise FileExistsError(f"checkpoint step already exists: {step}")
    rng_state = capture_rng_state()
    _save_lora_checkpoint(
        lora_path, model, optimized, resolved, step, optimizer=optimizer,
        cursor=cursor, rng_state=rng_state, cumulative=cumulative,
    )
    _save_composed_checkpoint(
        composed_path, model, composed, optimized, resolved, step, cursor=cursor
    )
    record = {
        "step": step,
        "pool_cursor": cursor,
        "lora_file": str(lora_path.relative_to(run_dir)),
        "lora_sha256": sha256_file(lora_path),
        "lora_tensor_count": len(optimized),
        "composed_file": str(composed_path.relative_to(run_dir)),
        "composed_sha256": sha256_file(composed_path),
        "composed_tensor_count": len(composed),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest.setdefault("checkpoints", []).append(record)
    _atomic_json_dump(run_dir / "RUN_MANIFEST.json", manifest)
    return record


def _raw_train_metadata(dataset: Any) -> list[dict[str, Any]]:
    raw = getattr(dataset, "_raw_dataset", None)
    if raw is None:
        return [
            {
                "index": index,
                "id": str(dataset[index].get("id", "")),
                "category": str(dataset[index].get("category", "")),
                "length": int(dataset[index].get("length", 0)),
                "n_channels": int(dataset[index].get("n_channels", 0)),
            }
            for index in range(len(dataset))
        ]
    slim = raw.select_columns(["id", "category", "length", "n_channels"])
    return [
        {
            "index": index,
            "id": str(row.get("id", "")),
            "category": str(row.get("category", "")),
            "length": int(row.get("length", 0)),
            "n_channels": int(row.get("n_channels", 0)),
        }
        for index, row in enumerate(slim)
    ]


def _pool(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(f"pool directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    try:
        resolved = load_l3_config(args.config)
        rl = _resolved_rl(resolved)
        if rl.get("contract") != "longrun":
            raise ValueError("pool filtering requires the longrun contract")
        model, dataset, optimized, composed, train_sha = _load_policy_and_train(
            resolved, Path(args.checkpoint).expanduser()
        )
        pool_seed = int(rl["pool_seed"])
        torch.manual_seed(pool_seed)
        random.seed(pool_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(pool_seed)
        metadata = _raw_train_metadata(dataset)
        max_candidates = int(rl["pool_max_rollouts"]) // int(rl["G"])
        candidates = stratified_candidate_indices(metadata, int(rl["pool_seed"]), max_candidates)
        accepted: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        rollout_count = 0
        with (run_dir / "rollouts.jsonl").open("x", encoding="utf-8") as rollout_log, (
            run_dir / "candidates.jsonl"
        ).open("x", encoding="utf-8") as candidate_log:
            for candidate_number, index in enumerate(candidates, start=1):
                sample = _checked_sample(dataset, index, rl)
                accounting = RolloutAccounting(int(rl["G"]))
                prompt = model.prepare_batch([sample], training=False)
                prompt = {key: value for key, value in prompt.items() if key != "labels"}
                for rollout in range(1, int(rl["G"]) + 1):
                    model.eval()
                    completion = generate_completion(model, prompt, rl, sample=True)
                    text = model.tokenizer.batch_decode(completion, skip_special_tokens=True)[0]
                    reward = reward_completion(dataset, text, sample, float(rl["format_penalty"]))
                    accounting.record(reward, int(completion.shape[1]))
                    _write_jsonl(rollout_log, {
                        "candidate": candidate_number, "train_index": index,
                        "category": sample["category"], "rollout": rollout,
                        "reward_primary": reward.primary,
                        "reward_optimization": reward.optimization,
                        "parsable": reward.parsable,
                        "response_tokens": int(completion.shape[1]),
                        "text_sha256": _sha256_text(text),
                    })
                    del completion
                counts = accounting.finalize()
                rollout_count += int(counts["rollouts"])
                row = {
                    "candidate": candidate_number,
                    "index": index,
                    "id": str(sample["id"]),
                    "category": str(sample["category"]),
                    "length": int(sample["length"]),
                    "n_channels": int(sample["n_channels"]),
                    "reward_mean": counts["reward_mean"],
                    "reward_std": counts["reward_std"],
                    "parsable": counts["parsable"],
                    "informative": counts["reward_std"] > 0.0,
                }
                _write_jsonl(candidate_log, row)
                candidate_rows.append(row)
                if row["informative"]:
                    accepted.append(row)
                del prompt, sample
                available = {category: sum(r["category"] == category for r in accepted)
                             for category in REQUIRED_CATEGORIES}
                if all(count >= int(rl["min_pool_per_category"]) for count in available.values()):
                    break
        selected = select_balanced_pool(accepted, int(rl["min_pool_per_category"]))
        pool_core = {
            "seed": int(rl["pool_seed"]),
            "temperature": float(rl["temperature"]),
            "G": int(rl["G"]),
            "indices": [int(row["index"]) for row in selected],
            "rows": selected,
        }
        pool_document = {
            "format": "l3_filtered_pool_v1",
            **pool_core,
            "pool_content_sha256": canonical_sha256(pool_core),
            "candidates_evaluated": len(candidate_rows),
            "rollouts": rollout_count,
            "train_sha256": train_sha,
        }
        _json_dump(run_dir / "pool.json", pool_document)
        summary = {
            "status": "DONE", "candidates_evaluated": len(candidate_rows),
            "rollouts": rollout_count, "pool_size": len(selected),
            "per_category": {category: sum(row["category"] == category for row in selected)
                             for category in sorted(REQUIRED_CATEGORIES)},
            "pool": str(run_dir / "pool.json"),
            "pool_file_sha256": sha256_file(run_dir / "pool.json"),
            "pool_content_sha256": pool_document["pool_content_sha256"],
            "optimized_tensor_count": len(optimized),
            "composed_tensor_count": len(composed),
        }
        _json_dump(run_dir / "summary.json", summary)
    except Exception as exc:
        _json_dump(run_dir / "FAILED", {
            "status": "FAILED", "error_type": type(exc).__name__, "reason": str(exc),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise


def _load_pool(path: Path, train_sha: str, rl: Mapping[str, Any]) -> tuple[dict[str, Any], list[int]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != "l3_filtered_pool_v1" or document.get("train_sha256") != train_sha:
        raise RuntimeError("pool format/train SHA mismatch")
    core = {key: document[key] for key in ("seed", "temperature", "G", "indices", "rows")}
    if canonical_sha256(core) != document.get("pool_content_sha256"):
        raise RuntimeError("pool content SHA mismatch")
    if document.get("seed") != rl["pool_seed"] or document.get("G") != rl["G"]:
        raise RuntimeError("pool sampling contract mismatch")
    indices = [int(value) for value in document["indices"]]
    counts = {category: sum(row["category"] == category for row in document["rows"])
              for category in REQUIRED_CATEGORIES}
    if any(count < int(rl["min_pool_per_category"]) for count in counts.values()):
        raise RuntimeError(f"pool category coverage is insufficient: {counts}")
    if len(indices) != len(set(indices)):
        raise RuntimeError("pool contains duplicate train indices")
    return document, indices


def _cumulative_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {
        "steps": len(rows),
        "rollouts": sum(int(row["rollouts"]) for row in rows),
        "parsable": sum(int(row["parsable"]) for row in rows),
        "unparsable": sum(int(row["unparsable"]) for row in rows),
        "informative_groups": sum(int(row["informative_groups"]) for row in rows),
        "policy_informative_steps": sum(int(row["policy_informative_steps"]) for row in rows),
        "zero_variance_groups": sum(int(row.get("zero_variance_groups", 0)) for row in rows),
        "reward_sum": sum(float(row.get("reward_sum", 0.0)) for row in rows),
        "reward_square_sum": sum(float(row.get("reward_square_sum", 0.0)) for row in rows),
        "tokens_prompt": sum(int(row.get("tokens_prompt", 0)) for row in rows),
        "tokens_answer": sum(int(row.get("tokens_answer", 0)) for row in rows),
        "seconds": sum(float(row.get("s_per_step", 0.0)) for row in rows),
    }
    if any("zoom_rate" in row for row in rows):
        result.update({
            "zoom_count": sum(round(float(row.get("zoom_rate", 0.0)) * int(row["rollouts"]))
                              for row in rows),
            "zoom_invalid": sum(int(row.get("zoom_invalid", 0)) for row in rows),
            "zoom_window_len_weighted_sum": sum(
                float(row.get("zoom_window_len", 0.0)) *
                round(float(row.get("zoom_rate", 0.0)) * int(row["rollouts"]))
                for row in rows
            ),
            "zoom_iou_weighted_sum": sum(
                float(row.get("zoom_iou_with_gold", 0.0)) *
                round(float(row.get("zoom_rate", 0.0)) * int(row["rollouts"]))
                for row in rows
            ),
        })
    return result


def _check_longrun_windows(rows: Sequence[Mapping[str, Any]]) -> None:
    # 50 groups = 25 P2 steps.  Only complete, non-overlapping windows count.
    if len(rows) < 50 or len(rows) % 25:
        return
    windows = [rows[-50:-25], rows[-25:]]
    parsability = [
        sum(int(row["parsable"]) for row in window) /
        sum(int(row["rollouts"]) for row in window)
        for window in windows
    ]
    informative = [
        sum(int(row["informative_groups"]) for row in window) / (2 * len(window))
        for window in windows
    ]
    if all(value < 0.95 for value in parsability):
        raise RuntimeError(f"parsability below 95% for two consecutive 50-group windows: {parsability}")
    if all(value < 0.50 for value in informative):
        raise RuntimeError(f"informative groups below 50% for two consecutive windows: {informative}")


def _longrun_summary(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], status: str
) -> dict[str, Any]:
    cumulative = _cumulative_from_rows(rows)
    total = cumulative["rollouts"]
    reward_sum = sum(float(row["reward_sum"]) for row in rows)
    square_sum = sum(float(row["reward_square_sum"]) for row in rows)
    latest = manifest["checkpoints"][-1]
    return {
        "status": status,
        **cumulative,
        "pool_cursor": int(latest["pool_cursor"]),
        "reward_mean": reward_sum / total,
        "reward_std": math.sqrt(max(0.0, square_sum / total - (reward_sum / total) ** 2)),
        "kl_mean": sum(float(row["kl_mean"]) for row in rows) / len(rows),
        "max_abs_log_ratio": max(float(row["max_abs_log_ratio"]) for row in rows),
        "policy_grad_norm_max": max(float(row["policy_grad_norm"]) for row in rows),
        "kl_grad_norm_max": max(float(row["kl_grad_norm"]) for row in rows),
        "grad_norm_max": max(float(row["grad_norm"]) for row in rows),
        "peak_mem_mib": max(float(row["peak_mem_mib"]) for row in rows),
        "seconds": sum(float(row["s_per_step"]) for row in rows),
        "latest_checkpoint": latest,
    }


def _train_longrun(args: argparse.Namespace) -> None:
    resume_dir = Path(args.resume).expanduser().resolve() if args.resume else None
    run_dir = resume_dir or Path(args.run_dir).expanduser().resolve()
    if resume_dir is None:
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
    elif not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    resolved = load_l3_config(args.config)
    rl = _resolved_rl(resolved)
    if rl.get("contract") != "longrun":
        raise ValueError("longrun mode requires rl.contract=longrun")
    memory_mode = args.mode == "longrun-memory"
    seed = int(args.seed if args.seed is not None else rl["seed"])
    model, dataset, optimized, composed, train_sha = _load_policy_and_train(
        resolved, Path(args.checkpoint).expanduser()
    )
    reference = LoraReference(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(rl["lr"]), weight_decay=float(rl["weight_decay"])
    )
    target = 1 if memory_mode else int(args.target_step or rl["steps"])
    if not 1 <= target <= int(rl["steps"]):
        raise ValueError("target step is outside the declared longrun budget")

    steps_path = run_dir / "steps.jsonl"
    rollouts_path = run_dir / "rollouts.jsonl"
    if resume_dir is not None:
        if (run_dir / "DONE").exists() or (run_dir / "FAILED").exists():
            raise RuntimeError("refusing to resume a terminal run")
        manifest = json.loads((run_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        if manifest.get("seed") != seed or manifest.get("resolved_config") != resolved:
            raise RuntimeError("resume seed/config mismatch")
        record, payload = latest_valid_checkpoint(run_dir, manifest)
        chosen_index = next(
            index for index, item in enumerate(manifest["checkpoints"])
            if item == record
        )
        invalid_newer = manifest["checkpoints"][chosen_index + 1:]
        start_step = int(record["step"])
        if invalid_newer:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            for invalid in invalid_newer:
                for key in ("lora_file", "composed_file"):
                    path = run_dir / str(invalid[key])
                    if path.exists():
                        os.replace(path, path.with_name(path.name + f".invalid.{stamp}"))
            manifest.setdefault("invalid_checkpoints", []).extend(invalid_newer)
            manifest["checkpoints"] = manifest["checkpoints"][:chosen_index + 1]
            _atomic_json_dump(run_dir / "RUN_MANIFEST.json", manifest)
        # A crash between the two atomic file writes and the atomic manifest
        # update can leave unreferenced future files.  Preserve, but quarantine,
        # them so replay from the last valid cursor cannot collide with them.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for orphan in (run_dir / "checkpoints").glob("step_*_*.pt"):
            try:
                orphan_step = int(orphan.name.split("_", 2)[1])
            except (IndexError, ValueError):
                continue
            if orphan_step > start_step:
                os.replace(orphan, orphan.with_name(orphan.name + f".invalid.{stamp}"))
        state = payload["model_state"]
        named = dict(model.named_parameters())
        if set(state) != set(optimized):
            raise RuntimeError("resume LoRA tensor-name invariant failed")
        with torch.no_grad():
            for name in optimized:
                named[name].copy_(state[name].to(named[name].device, dtype=named[name].dtype))
        optimizer.load_state_dict(payload["optimizer_state"])
        restore_rng_state(payload["l3"]["rng_state"])
        cursor = int(record["pool_cursor"])
        truncate_jsonl_after_step(steps_path, start_step)
        truncate_jsonl_after_step(rollouts_path, start_step)
        rows = _read_jsonl(steps_path)
        if _cumulative_from_rows(rows) != payload["l3"]["cumulative"]:
            raise RuntimeError("resume cumulative counters disagree with step log")
    else:
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if memory_mode:
            pool_document = None
            pool_indices = [int(index) for index in rl["memory_gate_indices"]]
            order = pool_indices
        else:
            if args.pool is None:
                raise ValueError("longrun training requires --pool")
            pool_path = Path(args.pool).expanduser().resolve()
            pool_document, pool_indices = _load_pool(pool_path, train_sha, rl)
            order = deterministic_prompt_order(pool_indices, seed)
        manifest = {
            "format": "l3_grpo_longrun_manifest_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "seed": seed,
            "initial_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "initial_checkpoint_sha256": sha256_file(Path(args.checkpoint).expanduser()),
            "train_sha256": train_sha,
            "resolved_config": resolved,
            "pool": None if memory_mode else str(Path(args.pool).expanduser().resolve()),
            "pool_file_sha256": None if memory_mode else sha256_file(Path(args.pool).expanduser()),
            "pool_content_sha256": None if memory_mode else pool_document["pool_content_sha256"],
            "prompt_order": order,
            "optimized_names": optimized,
            "composed_slim_names": composed,
            "checkpoints": [],
        }
        _atomic_json_dump(run_dir / "RUN_MANIFEST.json", manifest)
        steps_path.touch(exist_ok=False)
        rollouts_path.touch(exist_ok=False)
        rows = []
        start_step = 0
        cursor = 0
        _checkpoint_longrun(
            run_dir, manifest, model, optimizer, optimized, composed, resolved,
            0, 0, _cumulative_from_rows(rows),
        )

    order = [int(index) for index in manifest["prompt_order"]]
    if target <= start_step:
        raise ValueError(f"target step {target} is not beyond checkpoint step {start_step}")
    if (run_dir / "PAUSED").exists():
        (run_dir / "PAUSED").unlink()
    try:
        with steps_path.open("a", encoding="utf-8") as step_log, rollouts_path.open(
            "a", encoding="utf-8"
        ) as rollout_log:
            for step in range(start_step + 1, target + 1):
                indices, next_cursor = prompt_indices_from_cursor(
                    order, cursor, int(rl["prompts_per_step"])
                )
                samples = [_checked_sample(dataset, index, rl) for index in indices]
                row = grpo_longrun_step(
                    model, optimizer, reference, dataset, samples, rl, step, rollout_log
                )
                if row["rollouts"] != int(rl["G"]) * int(rl["prompts_per_step"]):
                    raise RuntimeError("longrun rollout accounting mismatch")
                if row["peak_mem_mib"] >= float(rl["peak_limit_mib"]):
                    raise RuntimeError(f"memory peak {row['peak_mem_mib']:.1f} MiB reached limit")
                _write_jsonl(step_log, row)
                rows.append(row)
                cursor = next_cursor
                print(
                    "L3_STEP "
                    f"step={step} category={'+'.join(row['categories'])} "
                    f"reward={row['reward_mean']:.6f} informative={row['informative_groups']}/2 "
                    f"kl={row['kl_mean']:.6g} peak_mib={row['peak_mem_mib']:.1f} "
                    f"s_per_step={row['s_per_step']:.1f}",
                    flush=True,
                )
                _check_longrun_windows(rows)
                if step % int(rl["checkpoint_every"]) == 0 or step == target:
                    _checkpoint_longrun(
                        run_dir, manifest, model, optimizer, optimized, composed, resolved,
                        step, cursor, _cumulative_from_rows(rows),
                    )
        status = "DONE" if target == int(rl["steps"]) or memory_mode else "PAUSED"
        summary = _longrun_summary(rows, manifest, status)
        _atomic_json_dump(run_dir / "summary.json", summary)
        _json_dump(run_dir / status, summary)
    except Exception as exc:
        _json_dump(run_dir / "FAILED", {
            "status": "FAILED", "error_type": type(exc).__name__, "reason": str(exc),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise


def _train_zoom(args: argparse.Namespace) -> None:
    """Resumable 200-step Z2 run with deterministic pool refresh at step 100."""
    resume_dir = Path(args.resume).expanduser().resolve() if args.resume else None
    run_dir = resume_dir or Path(args.run_dir).expanduser().resolve()
    if resume_dir is None:
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
    elif not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    resolved = load_l3_config(args.config)
    rl = _resolved_rl(resolved)
    if rl.get("contract") != "zoom":
        raise ValueError("zoom mode requires rl.contract=zoom")
    memory_mode = args.mode == "zoom-memory"
    seed = int(args.seed if args.seed is not None else rl["seed"])
    model, dataset, optimized, composed, train_sha = _load_policy_and_train(
        resolved, Path(args.checkpoint).expanduser()
    )
    reference = LoraReference(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(rl["lr"]), weight_decay=float(rl["weight_decay"])
    )
    target = 1 if memory_mode else int(args.target_step or rl["steps"])
    if not 1 <= target <= int(rl["steps"]):
        raise ValueError("target step is outside the declared zoom budget")
    steps_path, rollouts_path = run_dir / "steps.jsonl", run_dir / "rollouts.jsonl"
    metadata = _raw_train_metadata(dataset)

    if resume_dir is not None:
        if (run_dir / "DONE").exists() or (run_dir / "FAILED").exists():
            raise RuntimeError("refusing to resume a terminal zoom run")
        manifest = json.loads((run_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
        if manifest.get("format") != "l3_zoom_grpo_manifest_v1":
            raise RuntimeError("resume manifest is not a zoom run")
        if manifest.get("seed") != seed or manifest.get("resolved_config") != resolved:
            raise RuntimeError("zoom resume seed/config mismatch")
        record, payload = latest_valid_checkpoint(run_dir, manifest)
        position = manifest["checkpoints"].index(record)
        if position + 1 < len(manifest["checkpoints"]):
            raise RuntimeError("zoom resume found corrupt newer checkpoint; manual audit required")
        start_step = int(record["step"])
        named = dict(model.named_parameters())
        if set(payload["model_state"]) != set(optimized):
            raise RuntimeError("zoom resume LoRA tensor-name invariant failed")
        with torch.no_grad():
            for name in optimized:
                named[name].copy_(payload["model_state"][name].to(
                    named[name].device, dtype=named[name].dtype
                ))
        optimizer.load_state_dict(payload["optimizer_state"])
        restore_rng_state(payload["l3"]["rng_state"])
        cursor = int(record["pool_cursor"])
        truncate_jsonl_after_step(steps_path, start_step)
        truncate_jsonl_after_step(rollouts_path, start_step)
        rows = _read_jsonl(steps_path)
        if _cumulative_from_rows(rows) != payload["l3"]["cumulative"]:
            raise RuntimeError("zoom resume cumulative counters disagree with step log")
    else:
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        pool_rounds = []
        if memory_mode:
            pool_rounds.append({
                "round": 0, "seed": int(rl["pool_seed"]),
                "indices": [int(i) for i in rl["memory_gate_indices"]],
            })
        else:
            rounds = math.ceil(int(rl["steps"]) / int(rl["resample_every"]))
            for round_index in range(rounds):
                indices = zoom_pool_for_round(
                    metadata, int(rl["pool_seed"]), round_index,
                    pool_size=int(rl["pool_size"]),
                    excluded_categories=rl["excluded_categories"],
                )
                core = {"round": round_index, "seed": int(rl["pool_seed"]) + round_index,
                        "indices": indices}
                pool_rounds.append({**core, "content_sha256": canonical_sha256(core)})
        manifest = {
            "format": "l3_zoom_grpo_manifest_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode, "seed": seed,
            "initial_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "initial_checkpoint_sha256": sha256_file(Path(args.checkpoint).expanduser()),
            "train_sha256": train_sha, "resolved_config": resolved,
            "excluded_categories": list(rl["excluded_categories"]),
            "pool_rounds": pool_rounds, "optimized_names": optimized,
            "composed_slim_names": composed, "checkpoints": [],
        }
        _atomic_json_dump(run_dir / "RUN_MANIFEST.json", manifest)
        steps_path.touch(exist_ok=False)
        rollouts_path.touch(exist_ok=False)
        rows, start_step, cursor = [], 0, 0
        _checkpoint_longrun(run_dir, manifest, model, optimizer, optimized, composed,
                            resolved, 0, 0, _cumulative_from_rows(rows))

    if target <= start_step:
        raise ValueError(f"target step {target} is not beyond checkpoint step {start_step}")
    if (run_dir / "PAUSED").exists():
        (run_dir / "PAUSED").unlink()
    try:
        with steps_path.open("a", encoding="utf-8") as step_log, rollouts_path.open(
            "a", encoding="utf-8"
        ) as rollout_log:
            for step in range(start_step + 1, target + 1):
                if memory_mode:
                    indices = [int(i) for i in rl["memory_gate_indices"]]
                else:
                    round_index = (step - 1) // int(rl["resample_every"])
                    offset = ((step - 1) % int(rl["resample_every"])) * int(rl["prompts_per_step"])
                    pool = manifest["pool_rounds"][round_index]["indices"]
                    indices = [int(i) for i in pool[offset:offset + int(rl["prompts_per_step"])]]
                if len(indices) != int(rl["prompts_per_step"]):
                    raise RuntimeError("zoom pool exhausted before its resample boundary")
                samples = [_checked_sample(dataset, index, rl) for index in indices]
                row = grpo_zoom_step(
                    model, optimizer, reference, dataset, samples, rl, step, rollout_log
                )
                expected = int(rl["G"]) * int(rl["prompts_per_step"])
                if row["rollouts"] != expected:
                    raise RuntimeError("zoom rollout accounting mismatch")
                if row["peak_mem_mib"] >= float(rl["peak_limit_mib"]):
                    raise RuntimeError(f"memory peak {row['peak_mem_mib']:.1f} MiB reached limit")
                _write_jsonl(step_log, row)
                rows.append(row)
                cursor += int(rl["prompts_per_step"])
                print(
                    "L3_ZOOM_STEP "
                    f"step={step} category={'+'.join(row['categories'])} "
                    f"reward={row['reward_mean']:.6f} informative={row['informative_groups']}/2 "
                    f"zoom_rate={row['zoom_rate']:.3f} zoom_invalid={row['zoom_invalid']} "
                    f"kl={row['kl_mean']:.6g} peak_mib={row['peak_mem_mib']:.1f} "
                    f"s_per_step={row['s_per_step']:.1f}", flush=True,
                )
                _check_longrun_windows(rows)
                if step % int(rl["checkpoint_every"]) == 0 or step == target:
                    _checkpoint_longrun(
                        run_dir, manifest, model, optimizer, optimized, composed,
                        resolved, step, cursor, _cumulative_from_rows(rows),
                    )
        status = "DONE" if target == int(rl["steps"]) or memory_mode else "PAUSED"
        summary = _longrun_summary(rows, manifest, status)
        cumulative = _cumulative_from_rows(rows)
        zoom_count = int(cumulative.get("zoom_count", 0))
        summary.update({
            "zoom_rate": zoom_count / max(1, int(cumulative["rollouts"])),
            "zoom_invalid": int(cumulative.get("zoom_invalid", 0)),
            "zoom_window_len": float(cumulative.get("zoom_window_len_weighted_sum", 0.0)) /
                               max(1, zoom_count),
            "zoom_iou_with_gold": float(cumulative.get("zoom_iou_weighted_sum", 0.0)) /
                                  max(1, zoom_count),
        })
        _atomic_json_dump(run_dir / "summary.json", summary)
        _json_dump(run_dir / status, summary)
    except Exception as exc:
        _json_dump(run_dir / "FAILED", {
            "status": "FAILED", "error_type": type(exc).__name__, "reason": str(exc),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise


def seal_eval_result(
    checkpoint: Path,
    manifest_path: Path,
    eval_json: Path,
    output: Path,
    before_sha256: str,
    split: str,
    expected_rows: int,
) -> dict[str, Any]:
    after = sha256_file(checkpoint)
    if after != before_sha256:
        raise RuntimeError("eval checkpoint changed while evaluation was running")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [
        record for record in manifest.get("checkpoints", [])
        if record.get("composed_sha256") == after and
        (manifest_path.parent / str(record.get("composed_file"))).resolve() == checkpoint.resolve()
    ]
    if len(matches) != 1:
        raise RuntimeError("eval checkpoint SHA/path does not match run manifest")
    result = json.loads(eval_json.read_text(encoding="utf-8"))
    if result.get("split") != split or int(result.get("samples_evaluated", -1)) != expected_rows:
        raise RuntimeError("eval split/coverage mismatch")
    if not _finite_number(result.get("primary")) or int(result.get("eval_errors", 0)) != 0:
        raise RuntimeError("eval primary is non-finite or eval_errors is non-zero")
    raw_jsonl = result.get("eval_jsonl")
    if not isinstance(raw_jsonl, str) or not raw_jsonl:
        raise RuntimeError("eval result does not identify its per-row JSONL")
    jsonl_candidates = (
        Path(raw_jsonl).expanduser(),
        Path.cwd() / raw_jsonl,
        eval_json.parent / raw_jsonl,
    )
    jsonl_path = next((path.resolve() for path in jsonl_candidates if path.is_file()), None)
    if jsonl_path is None:
        raise RuntimeError(f"cannot resolve eval JSONL {raw_jsonl!r}")
    eval_rows = _read_jsonl(jsonl_path)
    if len(eval_rows) != expected_rows:
        raise RuntimeError(f"eval JSONL coverage mismatch: {len(eval_rows)} != {expected_rows}")
    if any(not _finite_number(row.get("primary")) for row in eval_rows):
        raise RuntimeError("eval JSONL contains a missing/non-finite primary")
    result["eval_jsonl"] = str(jsonl_path)
    result.update({
        "checkpoint_sha256_before": before_sha256,
        "checkpoint_sha256_after": after,
        "manifest_checkpoint_match": True,
        "run_manifest": str(manifest_path.resolve()),
        "run_manifest_sha256": sha256_file(manifest_path),
    })
    _json_dump(output, result)
    return result


def _seal_eval(args: argparse.Namespace) -> None:
    output = Path(args.out).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    seal_eval_result(
        Path(args.checkpoint).expanduser().resolve(),
        Path(args.manifest).expanduser().resolve(),
        Path(args.eval_json).expanduser().resolve(), output,
        args.before_sha256, args.split, args.expected_rows,
    )


def _train(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    try:
        resolved = load_l3_config(args.config)
        rl = _resolved_rl(resolved)
        model, dataset, optimized, composed, train_sha = _load_policy_and_train(
            resolved, Path(args.checkpoint).expanduser()
        )
        reference = LoraReference(model)
        if args.mode == "memory-gate":
            indices = [int(rl["memory_gate_index"])]
        else:
            indices = list(rl["train_prompt_indices"])
        samples = [_checked_sample(dataset, index, rl) for index in indices]
        categories = {sample["category"] for sample in samples}
        if args.mode == "smoke" and categories != REQUIRED_CATEGORIES:
            raise RuntimeError(f"fixed prompt category coverage is incomplete: {categories}")
        if args.mode == "memory-gate":
            sample = samples[0]
            if (sample["n_channels"], sample["length"]) != (16, 131072):
                raise RuntimeError("memory gate sample is not 16ch x 131072")
        steps = 1 if args.mode == "memory-gate" else int(rl["steps"])
        if len(samples) != steps:
            raise RuntimeError(f"step/index mismatch: {steps} vs {len(samples)}")
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        # RL never inherits SFT decay: va1 is a fixed point when policy and KL
        # gradients are both zero.
        weight_decay = float(rl.get("weight_decay", 0.0))
        optimizer = torch.optim.AdamW(trainable, lr=float(rl["lr"]), weight_decay=weight_decay)
        manifest = {
            "format": "l3_grpo_manifest_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "initial_checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "initial_checkpoint_sha256": sha256_file(Path(args.checkpoint).expanduser()),
            "train_sha256": train_sha,
            "resolved_config": resolved,
            "optimized_names": optimized,
            "composed_slim_names": composed,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        }
        _json_dump(run_dir / "RUN_MANIFEST.json", manifest)
        step_rows = []
        with (run_dir / "steps.jsonl").open("x", encoding="utf-8") as log:
            for step_index, sample in enumerate(samples, start=1):
                row = grpo_step(model, optimizer, reference, dataset, sample, rl, step_index)
                _write_jsonl(log, row)
                step_rows.append(row)
        expected_rollouts = steps * int(rl["G"])
        total_rollouts = sum(row["rollouts"] for row in step_rows)
        total_parsable = sum(row["parsable"] for row in step_rows)
        total_unparsable = sum(row["unparsable"] for row in step_rows)
        if total_rollouts != expected_rollouts or total_parsable + total_unparsable != total_rollouts:
            raise RuntimeError("aggregate rollout accounting mismatch")
        if not all(row["max_abs_log_ratio"] <= MAX_ABS_LOG_RATIO for row in step_rows):
            raise RuntimeError("a step exceeded the pre-update max |log ratio| gate")
        peak = max(row["peak_mem_mib"] for row in step_rows)
        if args.mode == "memory-gate" and peak >= float(rl["peak_limit_mib"]):
            raise RuntimeError(f"memory peak {peak:.1f} MiB reached limit")
        checkpoint = run_dir / f"lora_checkpoint_after_{steps}_steps.pt"
        _save_lora_checkpoint(checkpoint, model, optimized, resolved, steps)
        eval_checkpoint = run_dir / f"eval_composed_after_{steps}_steps.pt"
        _save_composed_checkpoint(
            eval_checkpoint, model, composed, optimized, resolved, steps
        )
        by_category: dict[str, dict[str, float | int]] = {}
        for row in step_rows:
            counter = by_category.setdefault(row["category"], {
                "steps": 0, "rollouts": 0, "parsable": 0, "unparsable": 0,
                "reward_sum": 0.0, "reward_square_sum": 0.0, "kl_sum": 0.0,
            })
            counter["steps"] += 1
            counter["rollouts"] += row["rollouts"]
            counter["parsable"] += row["parsable"]
            counter["unparsable"] += row["unparsable"]
            counter["reward_sum"] += row["reward_sum"]
            counter["reward_square_sum"] += row["reward_square_sum"]
            counter["kl_sum"] += row["kl_mean"] * row["rollouts"]
        category_summary = {}
        for category, counter in sorted(by_category.items()):
            count = int(counter["rollouts"])
            mean = float(counter["reward_sum"]) / count
            variance = max(0.0, float(counter["reward_square_sum"]) / count - mean * mean)
            category_summary[category] = {
                **counter,
                "reward_mean": mean,
                "reward_std": math.sqrt(variance),
                "kl_mean": float(counter["kl_sum"]) / count,
            }
        reward_sum = sum(row["reward_sum"] for row in step_rows)
        reward_square_sum = sum(row["reward_square_sum"] for row in step_rows)
        reward_mean = reward_sum / total_rollouts
        reward_variance = max(0.0, reward_square_sum / total_rollouts - reward_mean**2)
        summary = {
            "status": "DONE",
            "mode": args.mode,
            "steps": steps,
            "rollouts": total_rollouts,
            "parsable": total_parsable,
            "unparsable": total_unparsable,
            "zero_variance_groups": sum(row["zero_variance_groups"] for row in step_rows),
            "ratio_all_exact_one": all(row["ratio_all_exact_one"] for row in step_rows),
            "max_abs_log_ratio": max(row["max_abs_log_ratio"] for row in step_rows),
            "reward_mean": reward_mean,
            "reward_std": math.sqrt(reward_variance),
            "format_penalty": sum(row["format_penalty"] * row["rollouts"] for row in step_rows) / total_rollouts,
            "kl_mean": sum(row["kl_mean"] for row in step_rows) / steps,
            "ratio_mean": sum(row["ratio_mean"] for row in step_rows) / steps,
            "grad_norm": max(row["grad_norm"] for row in step_rows),
            "max_grad_norm": max(row["grad_norm"] for row in step_rows),
            "positive_grad_steps": sum(row["grad_norm"] > 0.0 for row in step_rows),
            "tokens_prompt": sum(row["tokens_prompt"] for row in step_rows),
            "tokens_answer": sum(row["tokens_answer"] for row in step_rows),
            "peak_mem_mib": peak,
            "seconds": sum(row["s_per_step"] for row in step_rows),
            "s_per_step": sum(row["s_per_step"] for row in step_rows) / steps,
            "by_category": category_summary,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_tensor_count": len(optimized),
            "eval_composed_checkpoint": str(eval_checkpoint),
            "eval_composed_checkpoint_sha256": sha256_file(eval_checkpoint),
            "eval_composed_tensor_count": len(composed),
        }
        if not _finite_number(summary["reward_mean"]) or not _finite_number(summary["kl_mean"]):
            raise RuntimeError("aggregate reward/KL is non-finite")
        _json_dump(run_dir / "summary.json", summary)
        _json_dump(run_dir / "DONE", summary)
    except Exception as exc:
        _json_dump(run_dir / "FAILED", {
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "reason": str(exc),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)
    common.add_argument("--checkpoint", required=True)
    init = subparsers.add_parser("init-check", parents=[common])
    init.add_argument("--out", required=True)
    pool = subparsers.add_parser("pool", parents=[common])
    pool.add_argument("--run-dir", required=True)
    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--run-dir")
    train.add_argument("--resume")
    train.add_argument(
        "--mode", choices=(
            "memory-gate", "smoke", "longrun-memory", "longrun",
            "zoom-memory", "zoom",
        ),
        default="smoke",
    )
    train.add_argument("--pool")
    train.add_argument("--seed", type=int)
    train.add_argument("--target-step", type=int)
    seal = subparsers.add_parser("seal-eval")
    seal.add_argument("--checkpoint", required=True)
    seal.add_argument("--manifest", required=True)
    seal.add_argument("--eval-json", required=True)
    seal.add_argument("--out", required=True)
    seal.add_argument("--before-sha256", required=True)
    seal.add_argument("--split", choices=("val", "test"), required=True)
    seal.add_argument("--expected-rows", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "init-check":
        _init_check(args)
    elif args.command == "pool":
        _pool(args)
    elif args.command == "seal-eval":
        _seal_eval(args)
    elif args.mode in {"zoom", "zoom-memory"}:
        if not args.run_dir and not args.resume:
            raise SystemExit("zoom train requires --run-dir or --resume")
        _train_zoom(args)
    elif args.mode in {"longrun", "longrun-memory"}:
        if not args.run_dir and not args.resume:
            raise SystemExit("longrun train requires --run-dir or --resume")
        _train_longrun(args)
    else:
        if not args.run_dir or args.resume:
            raise SystemExit("smoke train requires --run-dir and does not accept --resume")
        _train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
