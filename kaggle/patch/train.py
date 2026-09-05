#!/usr/bin/env python3
"""
Train a TSLM baseline (chatts / flamingo / itformer) on AnomalyXL.

All configuration is driven by a single YAML file (see configs/).

Usage:
    # Standard training run
    python train.py --config configs/chatts_precise.yaml

    # Disable W&B logging (overrides config)
    python train.py --config configs/chatts_precise.yaml --no-wandb
"""

import argparse
import gc
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from src.models.base import BaseModel
from src.models.chatts import ChatTSModel
from src.models.flamingo import OpenTSLMFlamingo
from src.models.itformer import ITFormerModel
from src.models.wavelet import WaveletTSModel
from src.datasets.anomaly_xl.qa_dataset import AnomalyXLDataset
from src.training.logging import ExperimentLogger
from src.utils.config import ExperimentConfig

# Integrity helpers (deployed next to train.py by every launcher overlay).
# Hard import on purpose: a launcher that forgets the file must fail loudly,
# not fall back to the pre-integrity behavior.
from eval_integrity import (
    EvalJsonlWriter,
    budget_active,
    check_eval_coverage,
    check_resume_checkpoint,
    coverage_abort,
    forbid_budget_under_coverage,
    parse_status_for,
    restore_grad_ckpt,
)

# The 3 TSLM architectures these 6 configs use — a plain dict beats registry
# indirection now that there's no need to support pluggable architectures
# beyond these.
MODEL_CLASSES: dict[str, type[BaseModel]] = {
    "chatts": ChatTSModel,
    "flamingo": OpenTSLMFlamingo,
    "itformer": ITFormerModel,
    "wavelet": WaveletTSModel,
}

# v3 (variante ricorrente, cartella v3-recurrent/): registrata solo se il file
# e' presente nell'overlay del nodo; la v1 (`wavelet`) resta identica.
try:
    from src.models.wavelet_v3 import WaveletV3TSModel
    MODEL_CLASSES["wavelet_v3"] = WaveletV3TSModel
except ModuleNotFoundError as exc:
    # The base va1 overlay legitimately has no v3 module.  Do not hide a
    # broken dependency *inside* wavelet_v3: that must abort during import.
    if exc.name != "src.models.wavelet_v3":
        raise


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


# ==============================================================================
# Model Setup
# ==============================================================================

def load_model(config: ExperimentConfig, device: str) -> BaseModel:
    """
    Load model by architecture name.

    Each model class implements ``from_config()`` to extract the parameters
    it needs, so this function stays architecture-agnostic.
    """
    print(f"\nInitializing {config.model.architecture} model with LLM: {config.backbone.model_id}")

    model_cls = MODEL_CLASSES[config.model.architecture]
    model = model_cls.from_config(config, device)
    if not os.environ.get("TSLM_DEVICE_MAP"):
        # With TSLM_DEVICE_MAP the LLM is dispatched across GPUs by accelerate
        # and a blanket .to() would break its hooks.
        model = model.to(device)

    # Optionally load pretrained weights from HuggingFace
    if config.backbone.hf_checkpoint_repo:
        try:
            from huggingface_hub import hf_hub_download
            print(f"  Downloading checkpoint from: {config.backbone.hf_checkpoint_repo}")
            checkpoint_path = hf_hub_download(
                repo_id=config.backbone.hf_checkpoint_repo,
                filename=config.backbone.hf_checkpoint_file,
            )
            print(f"  Loading checkpoint: {checkpoint_path}")
            model.load_from_file(checkpoint_path)
        except Exception as e:
            print(f"  Warning: Could not load from HuggingFace: {e}")
            print("  Proceeding with randomly initialized weights")

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Memory: {get_gpu_memory_mb():.1f} MB")

    return model


def configure_training_mode(
    model: BaseModel,
    config: ExperimentConfig,
) -> dict[str, list]:
    """Configure which parameters are trainable based on ``config.training.backbone_training``.

    Modes:
        ``freeze`` — backbone LLM frozen, model components trained
            (per ``get_trainable_parameters()``).
        ``lora`` — same as freeze + LoRA adapters injected into the
            backbone LLM.
        ``full`` — all parameters trainable.

    Returns:
        Dictionary mapping group names to parameter lists, ready for
        the optimizer.
    """
    mode = config.training.backbone_training

    if mode == "freeze":
        model.requires_grad_(False)
        param_groups = model.get_trainable_parameters()
        for params in param_groups.values():
            for p in params:
                p.requires_grad = True
        print(f"\nTraining mode: freeze (backbone frozen)")

    elif mode == "lora":
        model.requires_grad_(False)

        # Inject LoRA into backbone
        backbone_module = model.get_backbone_module()
        if backbone_module is None:
            raise ValueError(
                f"{type(model).__name__} does not expose a backbone module. "
                "Implement get_backbone_module() to use lora mode."
            )
        model.backbone.apply_peft(backbone_module, config.training.lora)

        # Re-unfreeze trainable groups — inject_adapter_in_model re-freezes
        # all non-LoRA params inside the backbone, which clobbers cross-attn
        # and embeddings that live inside lang_encoder.
        param_groups = model.get_trainable_parameters()
        for params in param_groups.values():
            for p in params:
                p.requires_grad = True

        lora_params = [
            p for n, p in model.named_parameters()
            if "lora_" in n and p.requires_grad
        ]
        param_groups["backbone_lora"] = lora_params

        lora_cfg = config.training.lora
        print(f"\nTraining mode: lora (r={lora_cfg.get('r', 16)}, "
              f"alpha={lora_cfg.get('alpha', 32)})")

    elif mode == "full":
        model.requires_grad_(True)
        param_groups = model.get_trainable_parameters()
        # Add remaining backbone params as their own group
        grouped_ids = {id(p) for ps in param_groups.values() for p in ps}
        remaining = [p for p in model.parameters() if id(p) not in grouped_ids]
        if remaining:
            param_groups["backbone"] = remaining
        print(f"\nTraining mode: full (all params trainable)")

    else:
        raise ValueError(
            f"Unknown training.backbone_training: {mode!r}. "
            "Use 'freeze', 'lora', or 'full'."
        )

    # Print summary
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    for group_name, params in param_groups.items():
        group_count = sum(p.numel() for p in params if p.requires_grad)
        print(f"  - {group_name}: {group_count:,}")

    return param_groups


# ==============================================================================
# Data Loading
# ==============================================================================

def create_dataloaders(
    config: ExperimentConfig,
    eos_token: str,
) -> tuple:
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Experiment configuration
        eos_token: The model's EOS token, injected into datasets so they
            stay model-agnostic.

    Returns:
        Tuple of (train_loader, val_loader, test_loader,
                  train_dataset, val_dataset, test_dataset)
    """
    # dataset_name ("anomaly_xl_precise" / "anomaly_xl_coarse") both map to
    # AnomalyXLDataset — the "precise"/"coarse" distinction is carried by
    # the `variant` kwarg below, not by a dataset-class registry.
    dataset_name = config.dataset.name
    dataset_kwargs = dict(config.dataset.extra_kwargs)
    dataset_kwargs["EOS_TOKEN"] = eos_token  # Bridge model → dataset
    print(f"\nLoading dataset '{dataset_name}' ({AnomalyXLDataset.__name__})...")
    if dataset_kwargs:
        print(f"  Dataset kwargs: {dataset_kwargs}")

    train_dataset = AnomalyXLDataset(split="train", **dataset_kwargs)
    val_dataset = AnomalyXLDataset(split="validation", **dataset_kwargs)
    test_dataset = AnomalyXLDataset(split="test", **dataset_kwargs)

    # Keep original QADataset instances for evaluation methods
    # (Subset wrapping below hides them)
    raw_train_dataset = train_dataset
    raw_val_dataset = val_dataset
    raw_test_dataset = test_dataset

    # Apply max_samples if specified
    # max_samples controls train split; val/test get eval_samples_ratio (default 10%) each
    max_samples = config.runtime.max_samples
    if max_samples is not None:
        eval_samples = max(1, int(max_samples * config.runtime.eval_samples_ratio))
        print(f"  Limiting to {max_samples} train / {eval_samples} val / {eval_samples} test samples")
        from torch.utils.data import Subset
        import random

        # Session-chained training: TSLM_SUBSET_SEED varies the train subset
        # per session (coverage of the full train set across sessions), while
        # val/test always come from a fixed RNG so the eval rows are identical
        # in every session and metrics stay comparable session-over-session.
        subset_seed = int(os.environ.get("TSLM_SUBSET_SEED", config.seed))
        train_rng = random.Random(subset_seed)
        eval_rng = random.Random(config.seed)
        print(f"  Train subset seed: {subset_seed} (val/test fixed at seed {config.seed})")

        if len(train_dataset) > max_samples:
            train_indices = train_rng.sample(range(len(train_dataset)), max_samples)
            train_dataset = Subset(train_dataset, train_indices)
        if len(val_dataset) > eval_samples:
            val_indices = eval_rng.sample(range(len(val_dataset)), eval_samples)
            val_dataset = Subset(val_dataset, val_indices)
        if len(test_dataset) > eval_samples:
            test_indices = eval_rng.sample(range(len(test_dataset)), eval_samples)
            test_dataset = Subset(test_dataset, test_indices)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Identity collate — model.prepare_batch() handles patch-alignment
    def collate_fn(batch):
        return batch

    dl_cfg = config.training.dataloader
    num_workers = dl_cfg.get("num_workers", 8 if torch.cuda.is_available() else 0)
    prefetch_factor = dl_cfg.get("prefetch_factor", 4 if num_workers > 0 else None)
    pin_memory = dl_cfg.get("pin_memory", torch.cuda.is_available())

    if num_workers > 0:
        print(f"  DataLoader workers: {num_workers} (prefetch_factor={prefetch_factor})")

    loader_kwargs = dict(
        batch_size=config.training.batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return (train_loader, val_loader, test_loader,
            raw_train_dataset, raw_val_dataset, raw_test_dataset)


# ==============================================================================
# Checkpointing
# ==============================================================================

def save_checkpoint(
    model,
    optimizer: AdamW,
    scheduler,
    epoch: int,
    val_loss: float,
    output_dir: Path,
    config: ExperimentConfig,
    is_best: bool = False,
    batch_idx: Optional[int] = None,
    best_val_metric: Optional[float] = None,
) -> Path:
    """Save training checkpoint."""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # DEVIATION (Kaggle repro): full model+optimizer checkpoints are ~13GB and
    # overflow the 20GB working disk; TSLM_SAVE_TRAINABLE_ONLY=1 keeps only the
    # trainable tensors (LoRA + encoder + projector + embeddings, ~1.3GB) and
    # drops optimizer/scheduler state (eval-only reload, no resume).
    slim = os.environ.get("TSLM_SAVE_TRAINABLE_ONLY") == "1"
    if slim:
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        model_state = {k: v for k, v in model.state_dict().items() if k in trainable}
    else:
        model_state = model.state_dict()
    # TSLM_SAVE_OPT_STATE=1: keep optimizer/scheduler even in slim mode (a few
    # tens of MB for LoRA-scale trainables) so a follow-up session can resume
    # the official LR schedule mid-recipe instead of weights-only restarting.
    keep_opt = (not slim) or os.environ.get("TSLM_SAVE_OPT_STATE") == "1"
    checkpoint = {
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict() if keep_opt else None,
        "scheduler_state": scheduler.state_dict() if keep_opt else None,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "val_loss": val_loss,
        "best_val_metric": best_val_metric,
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    latest_path = checkpoint_dir / "latest_checkpoint.pt"

    if is_best:
        path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, path)
    elif batch_idx is not None:
        path = checkpoint_dir / f"checkpoint_epoch_{epoch}_batch_{batch_idx}.pt"
        torch.save(checkpoint, path)
    else:
        # End-of-epoch refresh with no val-metric improvement: pin
        # latest_checkpoint.pt to the most recent completed epoch and
        # skip the per-epoch named file.
        path = latest_path

    # Always update "latest" for easy resumption.
    torch.save(checkpoint, latest_path)

    if is_best:
        print(f"  Saved best checkpoint to {path}")
    elif batch_idx is not None:
        print(f"  [Checkpoint] Saved at epoch {epoch}, batch {batch_idx}")

    return path


def find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    """Find the latest checkpoint in a run directory."""
    checkpoint_dir = output_dir / "checkpoints"
    latest_path = checkpoint_dir / "latest_checkpoint.pt"

    if latest_path.exists():
        return str(latest_path)
    return None


def load_checkpoint(
    path: str,
    model,
    optimizer: Optional[AdamW] = None,
    scheduler=None,
) -> Dict[str, Any]:
    """Load training checkpoint."""
    print(f"\nLoading checkpoint from: {path}")

    checkpoint = torch.load(path, map_location="cpu")

    if os.environ.get("TSLM_SAVE_TRAINABLE_ONLY") == "1":
        missing, unexpected = model.load_state_dict(
            checkpoint["model_state"], strict=False
        )
        print(f"  Partial (trainable-only) load: {len(checkpoint['model_state'])} "
              f"tensors loaded, {len(missing)} frozen tensors kept from init")
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:5]}")
    else:
        model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    print(f"  Resumed from epoch {checkpoint['epoch']}")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")

    return checkpoint


# ==============================================================================
# Training Loop
# ==============================================================================

def train_epoch(
    model,
    train_loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    config: ExperimentConfig,
    epoch: int,
    output_dir: Path,
    logger: Optional[ExperimentLogger] = None,
    global_step: int = 0,
    deadline: Optional[float] = None,
) -> tuple:
    """
    Train for one epoch.

    Returns:
        Tuple of (average training loss, updated global step)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    # Per-category coverage accounting: every sample ends up in exactly one
    # of these. Born from the m30 failure (all 2770 16-channel cells silently
    # skipped while monitoring showed "zero errors"): coverage is now counted,
    # printed, and — with TSLM_REQUIRE_FULL_COVERAGE=1 — enforced.
    cat_ok: dict = {}
    cat_fail: dict = {}

    def _count(batch_, target):
        for s in batch_:
            c = s.get("category", "?") if isinstance(s, dict) else "?"
            target[c] = target.get(c, 0) + 1

    # TSLM_AMP=1: half-precision weights + autocast forward + fp32 master
    # copies of the trainables (cast in train()). TSLM_AMP_DTYPE picks the
    # half format: float16 needs the GradScaler (gradient underflow) but
    # Qwen's forward activations OVERFLOW fp16's range — verified: every batch
    # of the Modal m1 run produced inf/nan loss. bfloat16 (Ampere+/Ada) has
    # fp32's exponent range: no overflow, no scaler needed. Kaggle 2xT4 has
    # no bf16 and keeps fp32 (TSLM_AMP unset).
    use_amp = os.environ.get("TSLM_AMP") == "1"
    amp_dtype = getattr(torch, os.environ.get("TSLM_AMP_DTYPE", "bfloat16"))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    # Opt-in smoke diagnostic.  Synchronization is deliberately restricted to
    # this env gate so production runs keep their normal asynchronous timing.
    log_step_timing = os.environ.get("TSLM_STEP_TIMING") == "1"

    def _log_step_timing(step_started_: float) -> None:
        if not log_step_timing:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[StepTiming] step={global_step} seconds="
              f"{time.perf_counter() - step_started_:.6f}", flush=True)

    for batch_idx, batch in enumerate(pbar):
        step_started = time.perf_counter() if log_step_timing else 0.0

        optimizer.zero_grad()

        try:
            batch_inputs = model.prepare_batch(batch)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                output = model(**batch_inputs)
                loss = output["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  Warning: Invalid loss at batch {batch_idx}, skipping")
                _count(batch, cat_fail)
                coverage_abort(f"invalid loss at batch {batch_idx}")
                continue

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
            scheduler.step()

            _count(batch, cat_ok)
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            avg_loss = total_loss / num_batches
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.2e}",
            })

            # Log step metrics
            if logger is not None:
                logger.log_metrics({
                    "train/loss": loss.item(),
                    "train/lr": current_lr,
                    "train/avg_loss": avg_loss,
                }, step=global_step)
            global_step += 1
            _log_step_timing(step_started)

            # Save intra-epoch checkpoint every N batches
            ckpt_interval = config.checkpoint_interval
            if ckpt_interval > 0 and (batch_idx + 1) % ckpt_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_loss=avg_loss,
                    output_dir=output_dir,
                    config=config,
                    batch_idx=batch_idx + 1,
                )

            # Optimizer-step budget for schedule-faithful top-up runs: stop
            # after exactly N successful steps this run (e.g. to complete the
            # official 22440-step recipe after a crash truncated an epoch).
            step_cap = int(os.environ.get("TSLM_MAX_TRAIN_STEPS", "0"))
            if step_cap and global_step >= step_cap:
                print(f"\n  [StepBudget] {global_step} optimizer steps this "
                      f"run; saving checkpoint and stopping training",
                      flush=True)
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_loss=total_loss / max(num_batches, 1),
                    output_dir=output_dir,
                    config=config,
                    batch_idx=batch_idx + 1,
                )
                break

            # Wall-clock budget for Kaggle sessions: stop training cleanly
            # (checkpoint saved) and let val + test eval still run before the
            # 12h session wall / weekly quota kills the VM.
            if deadline is not None and time.time() > deadline:
                print(f"\n  [TimeBudget] reached at batch {batch_idx + 1}/"
                      f"{len(train_loader)}; saving checkpoint and stopping "
                      f"training early", flush=True)
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    val_loss=total_loss / max(num_batches, 1),
                    output_dir=output_dir,
                    config=config,
                    batch_idx=batch_idx + 1,
                )
                break

        except Exception as e:
            # TSLM_CKPT_FALLBACK=1: a CUDA OOM (the ~12% longest cells exceed
            # the no-checkpointing footprint) retries the SAME batch with
            # gradient checkpointing enabled (proven to fit in ~22GB), then
            # switches it back off. Guarantees every sample is trained on —
            # "all episodes, all epochs" — at recompute cost only on the
            # cells that need it.
            is_oom = "out of memory" in str(e).lower()
            if is_oom and os.environ.get("TSLM_CKPT_FALLBACK") == "1":
                try:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    lm = model.get_backbone_module()
                    if not getattr(lm, "_ckpt_fallback_ready", False):
                        lm.enable_input_require_grads()
                        lm._ckpt_fallback_ready = True
                    lm.gradient_checkpointing_enable()
                    batch_inputs = model.prepare_batch(batch)
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        output = model(**batch_inputs)
                        loss = output["loss"]
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\n  [CkptFallback] invalid loss at batch {batch_idx}, skipping")
                        _count(batch, cat_fail)
                        coverage_abort(f"CkptFallback invalid loss at batch {batch_idx}")
                    else:
                        if use_amp:
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                            optimizer.step()
                        scheduler.step()
                        _count(batch, cat_ok)
                        total_loss += loss.item()
                        num_batches += 1
                        global_step += 1
                        _log_step_timing(step_started)
                        print(f"\n  [CkptFallback] batch {batch_idx} recovered "
                              f"with gradient checkpointing", flush=True)
                except Exception as e2:
                    print(f"\n  [CkptFallback] batch {batch_idx} FAILED anche "
                          f"col fallback: {e2}")
                    _count(batch, cat_fail)
                    coverage_abort(f"CkptFallback FAILED at batch {batch_idx}: {e2}")
                finally:
                    # Restore the GLOBAL checkpointing setting (the old
                    # unconditional disable stripped ckpt from every batch
                    # after the first OOM on GRAD_CKPT=1 runs).
                    restore_grad_ckpt(model.get_backbone_module())
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                continue
            print(f"\n  Error at batch {batch_idx}: {e}")
            _count(batch, cat_fail)
            coverage_abort(f"unrecoverable error at batch {batch_idx}: {e}")
            continue

    return total_loss / max(num_batches, 1), global_step, cat_ok, cat_fail


def validate(
    model,
    val_loader: DataLoader,
    epoch: int,
    dataset=None,
    output_dir: Optional[Path] = None,
    split_name: str = "val",
    log_outputs: bool = False,
    max_new_tokens: int = 500,
) -> Dict[str, Any]:
    """
    Validate the model with optional output logging.

    Returns:
        Dictionary with validation loss and optional metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # For output logging
    logged_outputs = []
    samples_generated = 0
    category_correct = {}
    category_total = {}
    category_metric_sums: dict[str, float] = {}
    category_key = dataset.category_key if dataset else None
    primary_metric = dataset.primary_metric if dataset else "accuracy"

    # Positive coverage accounting: every example leaves exactly one JSONL
    # row (id, category, gold, prediction, parse status, metrics) and the
    # row count is checked against the dataset size at the end. Errors are
    # recorded as 0-score rows, never swallowed.
    expected_examples = None
    try:
        expected_examples = len(dataset) if dataset is not None else len(val_loader.dataset)
    except (TypeError, AttributeError):
        pass
    jsonl_writer = None
    _pred_parser, _pred_required = None, None
    if log_outputs:
        jsonl_path = None
        if output_dir:
            logs_dir = output_dir / "output_logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = logs_dir / f"{split_name}_epoch_{epoch}.jsonl"
        jsonl_writer = EvalJsonlWriter(str(jsonl_path) if jsonl_path else None)
        try:
            # Official parser, injected only to LABEL parse status in the
            # JSONL; the official scorer itself is untouched.
            from src.datasets.anomaly_xl.scoring import (
                parse_structured_pred as _pred_parser,
                REQUIRED_KEYS as _pred_required,
            )
        except ImportError:
            pass

    def _record_error_rows(batch_, status, err=""):
        # A sample whose loss or generation failed scores 0 and leaves an
        # explicit row in aggregates and JSONL — silence is never success.
        nonlocal samples_generated
        for sample in batch_:
            category = sample.get(category_key, "unknown") if category_key else "all"
            if category not in category_correct:
                category_correct[category] = 0
                category_total[category] = 0
                category_metric_sums[category] = 0.0
            category_total[category] += 1
            entry = {
                "sample_idx": samples_generated,
                "id": sample.get("id", "") if isinstance(sample, dict) else "",
                "question": ((sample.get("question", "") if isinstance(sample, dict) else "") or "")[:200],
                "ground_truth": dataset.get_ground_truth(sample) if dataset else "",
                "predicted_answer": "",
                "prediction_raw": "",
                "correct": False,
                "parse_status": status,
                "error": str(err)[:300],
            }
            if category_key:
                entry["category"] = category
            if primary_metric != "accuracy":
                entry[primary_metric] = 0.0
            logged_outputs.append(entry)
            if jsonl_writer is not None:
                jsonl_writer.write(entry)
            samples_generated += 1

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [{split_name.upper()}]", leave=False)

    # Under TSLM_AMP the trainables are fp32 masters inside a half-precision
    # model, so eval forward and generation must also run under autocast or
    # the mixed dtypes crash (generation errors are swallowed silently below).
    use_amp = os.environ.get("TSLM_AMP") == "1"
    amp_dtype = getattr(torch, os.environ.get("TSLM_AMP_DTYPE", "bfloat16"))

    with torch.no_grad():
        for batch in pbar:
            try:
                # Compute loss
                batch_inputs = model.prepare_batch(batch)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    output = model(**batch_inputs)
                loss = output["loss"]

                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    num_batches += 1

                avg_loss = total_loss / max(num_batches, 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            except Exception as loss_err:
                if log_outputs:
                    _record_error_rows(batch, "loss_error", loss_err)
                continue

            # Generate outputs for logging (own try: a generation failure
            # must not be confused with a loss failure, and both leave rows)
            if log_outputs:
                    try:
                        eval_inputs = model.prepare_batch(batch, training=False)
                        gen_inputs = {k: v for k, v in eval_inputs.items() if k != "labels"}
                        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                            token_ids = model.generate(**gen_inputs, max_new_tokens=max_new_tokens)
                        predictions = model.tokenizer.batch_decode(
                            token_ids, skip_special_tokens=True
                        )

                        for sample, pred in zip(batch, predictions):
                            category = sample.get(category_key, "unknown") if category_key else "all"
                            question = sample.get("question", "")
                            ground_truth = dataset.get_ground_truth(sample) if dataset else ""

                            pred_answer = dataset.extract_answer(pred, sample) if dataset else pred.strip()
                            eval_result = dataset.evaluate_answer(pred_answer, sample) if dataset else {"correct": False}
                            is_correct = eval_result["correct"]

                            if category not in category_correct:
                                category_correct[category] = 0
                                category_total[category] = 0
                                category_metric_sums[category] = 0.0
                            category_total[category] += 1
                            if is_correct:
                                category_correct[category] += 1
                            if primary_metric != "accuracy" and primary_metric in eval_result:
                                category_metric_sums[category] += eval_result[primary_metric]

                            entry = {
                                "sample_idx": samples_generated,
                                "id": sample.get("id", "") if isinstance(sample, dict) else "",
                                "question": question[:200],
                                "ground_truth": ground_truth,
                                "predicted_answer": pred_answer,
                                "prediction_raw": pred,
                                "correct": is_correct,
                                "parse_status": parse_status_for(
                                    category, pred_answer,
                                    _pred_parser, _pred_required),
                            }
                            if isinstance(sample, dict):
                                for meta in ("length", "n_channels"):
                                    if meta in sample:
                                        entry[meta] = sample[meta]
                            if category_key:
                                entry["category"] = category
                            # Include any extra eval metrics (e.g. iou)
                            for k, v in eval_result.items():
                                if k != "correct" and v is not None:
                                    entry[k] = v

                            logged_outputs.append(entry)
                            if jsonl_writer is not None:
                                jsonl_writer.write(entry)
                            samples_generated += 1

                    except Exception as gen_err:
                        _record_error_rows(batch, "generation_error", gen_err)

    avg_loss = total_loss / max(num_batches, 1)

    result = {
        "loss": avg_loss,
        "num_batches": num_batches,
    }

    if jsonl_writer is not None:
        jsonl_writer.close()
        if jsonl_writer.error_count:
            print(f"  [EvalCoverage] {split_name}: {jsonl_writer.error_count} "
                  f"examples recorded as ERROR rows (scored 0)", flush=True)
        check_eval_coverage(jsonl_writer.count, expected_examples, split_name)
        result["eval_errors"] = jsonl_writer.error_count
        if jsonl_writer.path:
            result["eval_jsonl"] = jsonl_writer.path

    # Add evaluation metrics if we logged outputs
    if logged_outputs:
        result["samples_evaluated"] = len(logged_outputs)

        if primary_metric == "accuracy":
            total_correct = sum(1 for o in logged_outputs if o["correct"])
            result["accuracy"] = total_correct / len(logged_outputs)
            if category_key:
                result["category_accuracy"] = {
                    cat: category_correct[cat] / category_total[cat] if category_total[cat] > 0 else 0.0
                    for cat in category_correct
                }
        else:
            metric_values = [o[primary_metric] for o in logged_outputs if primary_metric in o]
            result[primary_metric] = sum(metric_values) / len(metric_values) if metric_values else 0.0
            if category_key:
                result[f"category_{primary_metric}"] = {
                    cat: category_metric_sums[cat] / category_total[cat] if category_total[cat] > 0 else 0.0
                    for cat in category_metric_sums
                }

        # Save logged outputs to file
        if output_dir:
            logs_dir = output_dir / "output_logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / f"{split_name}_epoch_{epoch}.json"
            log_data: dict[str, Any] = {
                "epoch": epoch,
                "split": split_name,
                "loss": avg_loss,
                "outputs": logged_outputs,
            }
            if primary_metric == "accuracy":
                log_data["accuracy"] = result.get("accuracy", 0.0)
                log_data["category_accuracy"] = result.get("category_accuracy", {})
            else:
                log_data[primary_metric] = result.get(primary_metric, 0.0)
                cat_key = f"category_{primary_metric}"
                if cat_key in result:
                    log_data[cat_key] = result[cat_key]
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=2)

    return result


def train(
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Main training function.
    """
    # Seed everything for reproducibility
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device()
    dataset_name = config.dataset.name
    num_epochs = config.training.num_epochs
    default_lr = config.training.learning_rate.get("default", 2e-4)

    # Build output directory: <output_dir>/<dataset>/<run_folder>
    base_dir = config.runtime.output_dir / dataset_name
    resume_from = config.runtime.resume_from
    # Session chaining on Kaggle: the previous session's checkpoint lives at a
    # mount path only known at runtime, so allow an env override.
    env_resume = os.environ.get("TSLM_RESUME_FROM")
    if env_resume:
        resume_from = env_resume
    # A full-coverage run refuses partial budgets outright (a truncated
    # epoch would save an epoch_N_end.pt that falsely looks complete).
    forbid_budget_under_coverage()

    if resume_from is None:
        if config.runtime.run_name:
            run_folder = config.runtime.run_name
            potential_output_dir = base_dir / run_folder

            if potential_output_dir.exists():
                latest_ckpt = find_latest_checkpoint(potential_output_dir)
                if latest_ckpt:
                    print(f"\nFound existing run '{run_folder}', auto-resuming from latest checkpoint...")
                    resume_from = latest_ckpt
                    output_dir = potential_output_dir
                else:
                    output_dir = potential_output_dir
            else:
                output_dir = potential_output_dir
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_folder = f"run_{timestamp}"
            output_dir = base_dir / run_folder
    else:
        if config.runtime.run_name:
            output_dir = base_dir / config.runtime.run_name
        else:
            output_dir = config.runtime.output_dir

    if resume_from is not None and output_dir == Path("results"):
        checkpoint_path = Path(resume_from)
        if checkpoint_path.parent.name == "checkpoints":
            output_dir = checkpoint_path.parent.parent
        else:
            output_dir = checkpoint_path.parent

    # Coverage mode refuses non-epoch-boundary resume points. Checked on the
    # RESOLVED path (2nd review round: the auto-resume discovery above picks
    # latest_checkpoint.pt AFTER any earlier check would have run).
    check_resume_checkpoint(resume_from)

    print("=" * 70)
    print(f"Training [{dataset_name}]")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Architecture: {config.model.architecture}")
    print(f"  Encoder: {config.model.encoder.get('type', 'cnn_tokenizer')}")
    print(f"  Device: {device}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {default_lr}")
    print(f"  Output dir: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config (full structured dump)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    from src.utils.config import save_yaml
    save_yaml(config.to_dict(), output_dir / "config.yaml")

    # Initialize experiment logger
    if not config.use_wandb:
        print("\n  NOTE: W&B logging is disabled. Using local-only logging.")
        print("  To enable W&B, set training.logging.wandb.enabled: true in config.")

    exp_logger = ExperimentLogger(
        experiment_name=config.runtime.run_name or output_dir.name,
        config=config.to_dict(),
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        wandb_group=config.wandb_group,
        log_dir=str(output_dir),
    )

    # Load model
    model = load_model(config, device)

    # Configure which parameters are trainable (freeze / lora / full)
    param_groups = configure_training_mode(model, config)

    # TSLM_AMP=1: promote the trainables to fp32 master weights in-place
    # (same tensor objects, so the optimizer groups built from param_groups
    # see them). Done BEFORE the resume load below, so fp32 checkpoint values
    # land in fp32 masters with no precision loss. Frozen weights stay fp16.
    if os.environ.get("TSLM_AMP") == "1":
        promoted = 0
        for p in model.parameters():
            if p.requires_grad and p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.float()
                promoted += 1
        print(f"  [AMP] promoted {promoted} trainable tensors to fp32 masters")

    # Create dataloaders (+ raw dataset instances for evaluation methods)
    eos_token = model.get_eos_token()
    (train_loader, val_loader, test_loader,
     _, val_dataset, test_dataset) = create_dataloaders(config, eos_token=eos_token)

    # Setup optimizer with per-group learning rates
    lr_config = config.training.learning_rate
    optimizer_groups = []
    for group_name, params in param_groups.items():
        group_lr = lr_config.get(group_name, default_lr)
        optimizer_groups.append({"params": params, "lr": group_lr})

    optimizer = AdamW(
        optimizer_groups,
        lr=default_lr,
        weight_decay=config.training.weight_decay,
    )

    # Setup scheduler
    total_steps = num_epochs * len(train_loader)
    # Top-up runs bump num_epochs to re-enter the epoch loop after a resume,
    # but the LR schedule must keep the ORIGINAL recipe length; the resumed
    # scheduler state restores the step counter, this restores the lambda.
    env_total = os.environ.get("TSLM_SCHED_TOTAL_STEPS")
    if env_total:
        total_steps = int(env_total)
        print(f"  Scheduler total steps overridden: {total_steps}")
    warmup_steps = int(config.training.warmup_ratio * total_steps)

    scheduler_type = getattr(config.training, "scheduler", None)
    scheduler_type = scheduler_type.get("type", "linear_with_warmup") if isinstance(scheduler_type, dict) else "linear_with_warmup"

    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    print(f"\nTraining setup:")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_metric = None

    # Determine the primary metric and whether higher is better
    val_primary_metric = val_dataset.primary_metric if val_dataset else "accuracy"
    # For loss-like metrics lower is better; for accuracy/f1/etc higher is better
    metric_higher_is_better = val_primary_metric != "loss"

    if resume_from:
        if config.runtime.resume_weights_only or os.environ.get("TSLM_RESUME_WEIGHTS_ONLY") == "1":
            checkpoint = load_checkpoint(resume_from, model)
            print(f"  Loaded model weights from: {resume_from}")
        else:
            checkpoint = load_checkpoint(
                resume_from, model, optimizer, scheduler
            )
            start_epoch = checkpoint["epoch"] + 1
            best_val_metric = checkpoint.get("best_val_metric", checkpoint.get("val_loss"))
            print(f"  Resuming training from epoch {start_epoch}")

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": None,
        "best_val_metric": None,
        "best_val_metric_name": val_primary_metric,
    }

    epochs_no_improve = 0
    global_step = 0
    training_start_time = time.time()

    # Optional wall-clock training budget (seconds) for Kaggle sessions.
    budget_s = float(os.environ.get("TSLM_MAX_TRAIN_SECONDS", "0"))
    train_deadline = (training_start_time + budget_s) if budget_s > 0 else None
    if train_deadline is not None:
        print(f"  Training time budget: {format_time(budget_s)}")

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()

        # Train
        train_loss, global_step, cat_ok, cat_fail = train_epoch(
            model, train_loader, optimizer, scheduler, config, epoch,
            output_dir=output_dir,
            logger=exp_logger, global_step=global_step,
            deadline=train_deadline,
        )

        # Coverage report — counters, not adjectives. With
        # TSLM_REQUIRE_FULL_COVERAGE=1 any failed sample aborts the run
        # loudly instead of silently producing a partially-trained model.
        n_ok = sum(cat_ok.values())
        n_fail = sum(cat_fail.values())
        try:
            n_expected = len(train_loader.dataset)
        except (TypeError, AttributeError):
            n_expected = None
        print(f"  [Coverage] epoch {epoch}: trained {n_ok}, failed {n_fail}"
              + (f", expected {n_expected}" if n_expected is not None else ""))
        for c in sorted(set(cat_ok) | set(cat_fail)):
            print(f"    {c}: ok={cat_ok.get(c, 0)} failed={cat_fail.get(c, 0)}")
        if exp_logger is not None:
            exp_logger.log_metrics(
                {f"coverage/failed_{c}": v for c, v in cat_fail.items()},
                step=global_step)
        if os.environ.get("TSLM_REQUIRE_FULL_COVERAGE") == "1":
            if n_fail > 0:
                raise SystemExit(
                    f"[Coverage] ABORT: {n_fail} training samples FAILED in epoch "
                    f"{epoch} (by category: {cat_fail}). Full coverage is required "
                    f"(TSLM_REQUIRE_FULL_COVERAGE=1) — fix the config (e.g. "
                    f"TSLM_GRAD_CKPT=1) instead of training a partial model.")
            # Belt: the counters must also account for EVERY dataset row.
            # A dataloader that silently drops samples leaves both counters
            # clean while shrinking the epoch — catch that too (skipped when
            # a time/step budget legitimately truncates the epoch; "0"/"" do
            # NOT count as budgets — 2nd review round).
            if (n_expected is not None and n_ok + n_fail != n_expected
                    and not budget_active()):
                raise SystemExit(
                    f"[Coverage] ABORT: epoch {epoch} accounted for "
                    f"{n_ok + n_fail}/{n_expected} dataset rows — samples were "
                    f"dropped before reaching the training loop.")

        # Validate with output logging. TSLM_VAL_GEN=0 skips the per-sample
        # generations on val (~35-40s each): val loss alone drives best-model
        # bookkeeping and the presentability check runs on the test split.
        val_result = validate(
            model, val_loader, epoch,
            dataset=val_dataset,
            output_dir=output_dir,
            split_name="val",
            log_outputs=os.environ.get("TSLM_VAL_GEN", "1") != "0",
            max_new_tokens=config.training.eval_max_new_tokens,
        )
        val_loss = val_result["loss"]

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        val_primary_metric = val_dataset.primary_metric if val_dataset else "accuracy"
        val_metric_value = val_result.get(val_primary_metric)
        if val_metric_value is not None:
            hist_key = f"val_{val_primary_metric}"
            if hist_key not in history:
                history[hist_key] = []
            history[hist_key].append(val_metric_value)

        epoch_time = time.time() - epoch_start_time

        # Print epoch summary
        summary_str = (f"Epoch {epoch}/{num_epochs}: "
                      f"train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}")
        if val_metric_value is not None:
            if val_primary_metric == "accuracy":
                summary_str += f", val_acc={val_metric_value*100:.1f}%"
            else:
                summary_str += f", val_{val_primary_metric}={val_metric_value:.4f}"
        summary_str += f", time={format_time(epoch_time)}"
        print(summary_str)

        cat_breakdown_key = f"category_{val_primary_metric}" if val_primary_metric != "accuracy" else "category_accuracy"
        if cat_breakdown_key in val_result:
            for cat, val in sorted(val_result[cat_breakdown_key].items()):
                if val_primary_metric == "accuracy":
                    print(f"    {cat}: {val*100:.1f}%")
                else:
                    print(f"    {cat}: {val:.4f}")

        # Log epoch metrics
        epoch_metrics = {
            "train/epoch_loss": train_loss,
            "val/loss": val_loss,
        }
        if val_metric_value is not None:
            epoch_metrics[f"val/{val_primary_metric}"] = val_metric_value
        if cat_breakdown_key in val_result:
            for cat, val in val_result[cat_breakdown_key].items():
                epoch_metrics[f"val/{val_primary_metric}_{cat}"] = val
        exp_logger.log_epoch_metrics(epoch_metrics, epoch=epoch, step=global_step)

        # Check for improvement based on primary metric
        current_metric = val_metric_value if val_metric_value is not None else -val_loss
        is_improvement = (
            best_val_metric is None
            or (metric_higher_is_better and current_metric > best_val_metric)
            or (not metric_higher_is_better and current_metric < best_val_metric)
        )
        if is_improvement:
            best_val_metric = current_metric
            history["best_epoch"] = epoch
            history["best_val_metric"] = best_val_metric
            epochs_no_improve = 0

            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                output_dir=output_dir, config=config, is_best=True,
                best_val_metric=best_val_metric,
            )
            exp_logger.log_artifact(checkpoint_path, artifact_type="model", name="best_model")
            print(f"  New best model! ({val_primary_metric}={current_metric:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epochs")
            # Refresh latest_checkpoint.pt to the most recent completed
            # epoch even when val didn't improve. save_checkpoint's
            # end-of-epoch branch (no is_best, no batch_idx) writes only
            # latest, so no per-epoch file is created.
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                output_dir=output_dir, config=config,
                best_val_metric=best_val_metric,
            )

        # Save history
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Keep an immutable end-of-epoch checkpoint (latest_checkpoint.pt was
        # just refreshed by both branches above) so parallel eval workers can
        # score every epoch for upstream-style best-by-primary selection.
        # MUST run before the early-stop break, or the final epoch is lost
        # (bit the m30 run: epoch_5_end.pt was never written).
        if os.environ.get("TSLM_KEEP_EPOCH_CKPTS") == "1":
            import shutil as _shutil
            _latest = output_dir / "checkpoints" / "latest_checkpoint.pt"
            _shutil.copy(_latest, output_dir / "checkpoints" / f"epoch_{epoch}_end.pt")
            print(f"  [EpochCkpt] kept epoch_{epoch}_end.pt")

        # Early stopping — honor the config's enabled flag (upstream's
        # early_stop_patience property ignores it and fired on m30 despite
        # enabled: false).
        es_enabled = config.training.early_stopping.get("enabled", True)
        if es_enabled and epochs_no_improve >= config.early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - training_start_time

    # Final summary
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Best epoch: {history['best_epoch']}")
    # A resumed run that never improves on the loaded best leaves these None;
    # crashing here (as on 2026-08-29, killing the final eval) helps nobody.
    if history["best_val_metric"] is not None:
        print(f"  Best val {val_primary_metric}: {history['best_val_metric']:.4f}")
    else:
        print(f"  Best val {val_primary_metric}: n/a (no improvement this run)")
    print(f"  Checkpoint: {output_dir / 'checkpoints' / 'best_model.pt'}")

    # TSLM_SKIP_FINAL_TEST=1: skip the built-in test eval on best-by-val-LOSS.
    # The single-GPU pipeline selects best-by-val-PRIMARY afterwards and runs
    # the test on THAT winner — the built-in eval would burn ~800 generations
    # on a model the selection may discard (review finding, 02/09).
    if os.environ.get("TSLM_SKIP_FINAL_TEST") == "1":
        print("\n[SkipFinalTest] built-in test eval skipped: selection "
              "best-by-primary + test run externally (eval worker)")
        if exp_logger is not None:
            exp_logger.finish()
        return

    # Load best model for final evaluation
    if history["best_epoch"] is not None:
        load_checkpoint(
            str(output_dir / "checkpoints" / "best_model.pt"),
            model
        )

    # Final test evaluation
    print("\nFinal test evaluation...")
    test_result = validate(
        model, test_loader, epoch=0,
        dataset=test_dataset,
        output_dir=output_dir,
        split_name="test",
        log_outputs=True,
        max_new_tokens=config.training.eval_max_new_tokens,
    )
    test_primary_metric = test_dataset.primary_metric if test_dataset else "accuracy"
    test_metric_value = test_result.get(test_primary_metric)
    print(f"  Test loss: {test_result['loss']:.4f}")
    if test_metric_value is not None:
        if test_primary_metric == "accuracy":
            print(f"  Test accuracy: {test_metric_value*100:.1f}%")
        else:
            print(f"  Test {test_primary_metric}: {test_metric_value:.4f}")
        test_cat_key = f"category_{test_primary_metric}" if test_primary_metric != "accuracy" else "category_accuracy"
        if test_cat_key in test_result:
            print(f"  Per-category {test_primary_metric}:")
            for cat, val in sorted(test_result[test_cat_key].items()):
                if test_primary_metric == "accuracy":
                    print(f"    {cat}: {val*100:.1f}%")
                else:
                    print(f"    {cat}: {val:.4f}")

    history["test_loss"] = test_result["loss"]
    if test_metric_value is not None:
        history[f"test_{test_primary_metric}"] = test_metric_value
        test_cat_key = f"category_{test_primary_metric}" if test_primary_metric != "accuracy" else "category_accuracy"
        if test_cat_key in test_result:
            history[f"test_{test_cat_key}"] = test_result[test_cat_key]

    # Log test metrics so they appear as chart data in W&B
    test_metrics = {"test/loss": test_result["loss"]}
    if test_metric_value is not None:
        test_metrics[f"test/{test_primary_metric}"] = test_metric_value
    test_cat_key = f"category_{test_primary_metric}" if test_primary_metric != "accuracy" else "category_accuracy"
    if test_cat_key in test_result:
        for cat, val in test_result[test_cat_key].items():
            test_metrics[f"test/{test_primary_metric}_{cat}"] = val
    exp_logger.log_metrics(test_metrics, step=global_step)

    # Save final history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Log final summary and finish
    summary = {
        "best_epoch": history.get("best_epoch"),
        "best_val_loss": history.get("best_val_loss"),
        "test_loss": history.get("test_loss"),
    }
    test_metric_hist_key = f"test_{test_primary_metric}"
    if test_metric_hist_key in history:
        summary[test_metric_hist_key] = history[test_metric_hist_key]
    test_cat_hist_key = f"test_category_{test_primary_metric}" if test_primary_metric != "accuracy" else "test_category_accuracy"
    if test_cat_hist_key in history:
        summary[test_cat_hist_key] = history[test_cat_hist_key]
    exp_logger.log_summary(summary)
    exp_logger.finish()

    return history


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train TS-LLM on any registered dataset",
        epilog="All model, dataset, training and runtime settings are defined in the "
               "YAML config file. See configs/experiments/ for examples.",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config (e.g. configs/experiments/ts_haystack_llama.yaml)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (overrides config file)",
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="Load --checkpoint and evaluate --split with "
                             "generations; no training.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["val", "test"],
                        default="val")
    parser.add_argument("--out-json", type=str, default=None,
                        help="Where to write the summary metrics JSON.")

    args = parser.parse_args()

    # Load config from YAML
    config = ExperimentConfig.from_yaml(args.config)

    # Apply --no-wandb override
    if args.no_wandb:
        wandb_cfg = config.training.logging.setdefault("wandb", {})
        wandb_cfg["enabled"] = False

    if args.eval_only:
        if not args.checkpoint:
            raise SystemExit("--eval-only requires --checkpoint")
        # Research evals are strict by default: a single missing example
        # aborts instead of shipping a metric on a smaller denominator.
        os.environ.setdefault("TSLM_EVAL_STRICT", "1")
        device = get_device()
        model = load_model(config, device)
        # Inject LoRA adapters (and trainable flags) exactly as in training,
        # or the checkpoint's lora_* keys have no modules to land in.
        configure_training_mode(model, config)
        eos_token = model.get_eos_token()
        (_, val_loader, test_loader,
         _, val_dataset, test_dataset) = create_dataloaders(config, eos_token=eos_token)
        load_checkpoint(args.checkpoint, model)
        loader, dataset = ((val_loader, val_dataset) if args.split == "val"
                           else (test_loader, test_dataset))
        out_dir = Path(config.runtime.output_dir) / config.dataset.name / \
            (config.runtime.run_name or "evalonly")
        # TSLM_EVAL_TAG distinguishes re-evals (e.g. on the pinned HF test)
        # from historical output_logs files instead of overwriting them.
        eval_tag = os.environ.get("TSLM_EVAL_TAG", "")
        split_tag = (f"evalonly_{args.split}_{Path(args.checkpoint).stem}"
                     + (f"_{eval_tag}" if eval_tag else ""))
        result = validate(
            model, loader, epoch=0,
            dataset=dataset,
            output_dir=out_dir,
            split_name=split_tag,
            log_outputs=True,
            max_new_tokens=config.training.eval_max_new_tokens,
        )
        summary = {k: v for k, v in result.items()}
        summary["checkpoint"] = args.checkpoint
        summary["split"] = args.split
        print("EVALONLY_RESULT:", json.dumps(
            {k: v for k, v in summary.items() if k != "outputs"}))
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(summary, f, indent=2)
        return

    # Train
    train(config)


if __name__ == "__main__":
    main()
