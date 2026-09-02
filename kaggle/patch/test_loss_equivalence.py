#!/usr/bin/env python3
"""GPU regression test for WaveletTSModel's memory-efficient loss.

Run this from the deployed ``~/timerlm/sft/tslm`` tree after copying the
patched ``wavelet.py`` into ``src/models``.  The test intentionally needs the
real AnomalyXL parquet files and Qwen checkpoint, so it is not a unit test for
the torch-free development checkout.

It checks three real one-/two-channel examples against the inherited full-
logit loss, then runs forward+backward on the longest real 16-channel
``localize_all_channels`` example and reports CUDA peak memory.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
from dataclasses import dataclass
from typing import Any

import torch

import train


GIB = 1024**3


@dataclass(frozen=True)
class SampleRef:
    split: str
    dataset: Any
    index: int
    sample_id: str
    category: str
    length: int
    n_channels: int

    def describe(self) -> str:
        return (
            f"split={self.split} index={self.index} id={self.sample_id} "
            f"category={self.category} length={self.length} "
            f"channels={self.n_channels}"
        )


def _metadata_refs(datasets: list[tuple[str, Any]]) -> list[SampleRef]:
    """Read metadata columns only; do not deserialize every time series."""
    refs: list[SampleRef] = []
    columns = ["id", "category", "length", "n_channels"]
    for split, dataset in datasets:
        raw = getattr(dataset, "_raw_dataset", None)
        if raw is None:
            raise RuntimeError(
                "This test requires lazy_loading=True so metadata can be scanned "
                "without materializing all channel arrays"
            )
        metadata = raw.select_columns(columns)
        for index in range(len(metadata)):
            row = metadata[index]
            refs.append(
                SampleRef(
                    split=split,
                    dataset=dataset,
                    index=index,
                    sample_id=str(row["id"]),
                    category=str(row["category"]),
                    length=int(row["length"]),
                    n_channels=int(row["n_channels"]),
                )
            )
    return refs


def _pick_samples(refs: list[SampleRef]) -> tuple[list[SampleRef], SampleRef]:
    # Use long examples for coverage, but keep the legacy/full-logit checks at
    # one or two channels where their sequence-wide logits fit on the A40.
    by_channels = {
        n: sorted(
            (ref for ref in refs if ref.n_channels == n),
            key=lambda ref: (-ref.length, ref.split, ref.index),
        )
        for n in (1, 2)
    }
    if len(by_channels[1]) < 2 or not by_channels[2]:
        raise RuntimeError(
            "Need at least two mono and one two-channel real samples for the "
            "equivalence check"
        )
    equivalence_refs = [by_channels[1][0], by_channels[1][1], by_channels[2][0]]

    long_refs = [
        ref
        for ref in refs
        if ref.n_channels == 16 and ref.category == "localize_all_channels"
    ]
    if not long_refs:
        raise RuntimeError(
            "No 16-channel localize_all_channels sample found in train/val/test"
        )
    backward_ref = max(long_refs, key=lambda ref: (ref.length, ref.split, -ref.index))
    return equivalence_refs, backward_ref


def _clear_cuda() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def _run_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    memory_efficient: bool,
) -> tuple[float, float]:
    _clear_cuda()
    torch.cuda.reset_peak_memory_stats()
    # Use no_grad rather than inference_mode: gradient checkpointing's
    # enable_input_require_grads hook is allowed to mark embedding outputs in
    # no_grad, but some torch releases reject that mutation on inference tensors.
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        output = model(
            **batch,
            memory_efficient_loss=memory_efficient,
        )
        if memory_efficient:
            expected_rows = int((batch["labels"][..., 1:] != -100).sum())
            actual_rows = int(output["logits"].shape[-2])
            if actual_rows != expected_rows:
                raise AssertionError(
                    f"Expected {expected_rows} supervised logit rows, got {actual_rows}"
                )
        loss = float(output["loss"].detach().float())
    torch.cuda.synchronize()
    peak_gib = torch.cuda.max_memory_allocated() / GIB
    del output
    return loss, peak_gib


def _check_equivalence(
    model: torch.nn.Module,
    refs: list[SampleRef],
    rtol: float,
) -> None:
    model.eval()
    print("\n[1/2] Loss equivalence against inherited full-logit path")
    for number, ref in enumerate(refs, start=1):
        sample = ref.dataset[ref.index]
        batch = model.prepare_batch([sample], training=True)
        sequence_tokens = int(batch["input_ids"].shape[1])
        supervised_tokens = int((batch["labels"][..., 1:] != -100).sum())

        new_loss, new_peak = _run_loss(model, batch, memory_efficient=True)
        old_loss, old_peak = _run_loss(model, batch, memory_efficient=False)
        if not math.isfinite(new_loss) or not math.isfinite(old_loss):
            raise AssertionError(
                f"non-finite loss for {ref.describe()}: "
                f"new={new_loss}, old={old_loss}"
            )
        denominator = max(abs(old_loss), torch.finfo(torch.float32).tiny)
        relative_error = abs(new_loss - old_loss) / denominator
        if relative_error > rtol:
            raise AssertionError(
                f"loss mismatch for {ref.describe()}: new={new_loss:.9g}, "
                f"old={old_loss:.9g}, rel_err={relative_error:.3e} > {rtol:.1e}"
            )

        print(
            f"  sample {number}: {ref.describe()} seq={sequence_tokens} "
            f"supervised={supervised_tokens} new={new_loss:.7f} "
            f"old={old_loss:.7f} rel_err={relative_error:.3e} "
            f"peak_new={new_peak:.2f}GiB peak_old={old_peak:.2f}GiB"
        )
        del batch, sample
        _clear_cuda()


def _run_long_backward(model: torch.nn.Module, ref: SampleRef) -> None:
    print("\n[2/2] Memory-efficient 16-channel forward+backward")
    print(f"  target: {ref.describe()}")
    sample = ref.dataset[ref.index]
    batch = model.prepare_batch([sample], training=True)
    sequence_tokens = int(batch["input_ids"].shape[1])
    supervised_tokens = int((batch["labels"][..., 1:] != -100).sum())

    model.train()
    model.zero_grad(set_to_none=True)
    _clear_cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**batch)
        actual_rows = int(output["logits"].shape[-2])
        if actual_rows != supervised_tokens:
            raise AssertionError(
                f"Expected {supervised_tokens} supervised logit rows, got {actual_rows}"
            )
        loss = output["loss"]
    del output
    loss_value = float(loss.detach().float())
    if not math.isfinite(loss_value):
        raise AssertionError(f"16-channel loss is non-finite: {loss_value}")
    loss.backward()
    torch.cuda.synchronize()

    peak_allocated = torch.cuda.max_memory_allocated() / GIB
    peak_reserved = torch.cuda.max_memory_reserved() / GIB
    grad_tensors = sum(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if grad_tensors == 0:
        raise AssertionError("Backward completed but produced no trainable gradients")

    print(
        f"  PASS: seq={sequence_tokens} supervised={supervised_tokens} "
        f"loss={loss_value:.7f} "
        f"peak_allocated={peak_allocated:.2f}GiB "
        f"peak_reserved={peak_reserved:.2f}GiB grad_tensors={grad_tensors}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/wavelet_precise_va1.yaml",
        help="Deployed Wavelet experiment YAML",
    )
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This regression test requires a CUDA GPU")

    # Match the failing A40 run unless the caller deliberately overrides one
    # of these settings in the environment.
    os.environ.setdefault("TSLM_TORCH_DTYPE", "bfloat16")
    os.environ.setdefault("TSLM_ATTN_IMPL", "sdpa")
    os.environ.setdefault("TSLM_GRAD_CKPT", "1")

    config = train.ExperimentConfig.from_yaml(args.config)
    if config.model.architecture != "wavelet":
        raise ValueError(f"Expected wavelet architecture, got {config.model.architecture!r}")
    if config.training.batch_size != 1:
        raise ValueError(f"Expected training.batch_size=1, got {config.training.batch_size}")

    device = "cuda"
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    print(f"config: {args.config}")
    model = train.load_model(config, device)
    parameter_groups = train.configure_training_mode(model, config)
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Training configuration left no trainable parameters")
    del parameter_groups

    loaders = train.create_dataloaders(config, model.get_eos_token())
    train_loader, val_loader, test_loader = loaders[:3]
    raw_datasets = loaders[3:]
    datasets = list(zip(("train", "validation", "test"), raw_datasets))
    del loaders, train_loader, val_loader, test_loader

    refs = _metadata_refs(datasets)
    equivalence_refs, backward_ref = _pick_samples(refs)
    _check_equivalence(model, equivalence_refs, args.rtol)
    _run_long_backward(model, backward_ref)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
