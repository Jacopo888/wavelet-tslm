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
MAX_ABS_LOG_RATIO = 1.0e-3


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
    fixed = {
        "prompt_split": "train",
        "expected_train_rows": 4488,
        "G": 4,
        "do_sample": True,
        "steps": 20,
        "prompts_per_step": 1,
        "trainable": "lora",
    }
    for key, expected in fixed.items():
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
    delta = log_reference - log_policy
    return torch.exp(delta) - delta - 1.0


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


def _write_jsonl(handle: Any, value: Any) -> None:
    handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


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
    expected_sha = resolved["rl"]["initial_checkpoint_sha256"]
    observed_sha = sha256_file(checkpoint_path)
    if observed_sha != expected_sha:
        raise RuntimeError(f"initial checkpoint SHA mismatch: {observed_sha}")
    ExperimentConfig, configure_training_mode, load_checkpoint, load_model, DatasetClass = _runtime_imports(resolved)
    experiment = ExperimentConfig.from_dict({k: v for k, v in resolved.items() if not k.startswith("_")})
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    expected = rl["train_prompt_expectations"][str(index)]
    sample = dataset[index]
    actual = {
        "id": str(sample.get("id", "")),
        "category": str(sample.get("category", "")),
        "length": int(sample.get("length", 0)),
        "n_channels": int(sample.get("n_channels", 0)),
    }
    normalized_expected = {
        "id": str(expected["id"]),
        "category": str(expected["category"]),
        "length": int(expected["length"]),
        "n_channels": int(expected["n_channels"]),
    }
    if actual != normalized_expected:
        raise RuntimeError(f"train metadata drift at {index}: {actual} != {normalized_expected}")
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
        "stage": "b_init_equals_va1",
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
) -> None:
    state = model.state_dict()
    if set(optimized_names) - set(state):
        raise RuntimeError("cannot save LoRA-only checkpoint: parameter missing")
    payload = {
        "model_state": {
            name: state[name].detach().cpu().clone() for name in optimized_names
        },
        "optimizer_state": None,
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
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


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
        weight_decay = float(resolved["training"].get("weight_decay", 0.01))
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
    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--run-dir", required=True)
    train.add_argument("--mode", choices=("memory-gate", "smoke"), default="smoke")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "init-check":
        _init_check(args)
    else:
        _train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
