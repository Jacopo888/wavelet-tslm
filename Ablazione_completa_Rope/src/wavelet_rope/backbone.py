"""Pinned Qwen3.5-4B backbone loader: offline snapshot, frozen base, wildcard wrapping, resolved LoRA.

Kept free of frontend/data imports so the wildcard gate can run before the rest of the package exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .qwen_temporal_rope import WildcardContext, lora_target_names, wrap_full_attention_layers


@dataclass
class BackboneBundle:
    model: nn.Module
    text: nn.Module
    lm_head: nn.Module
    tokenizer: Any
    ctx: WildcardContext
    wrapped_layers: list[int]
    lora_targets: list[str]
    lora_parameter_names: list[str]
    snapshot_dir: str


def load_backbone(snapshot_dir: str | Path, *, backend: str, pad_to: int, device: torch.device | str,
                  dtype: torch.dtype = torch.bfloat16, lora_r: int = 16, lora_alpha: int = 32,
                  lora_dropout: float = 0.0, gradient_checkpointing: bool = True, flex_dynamic: bool = True,
                  ) -> BackboneBundle:
    """Load the pinned Qwen3.5-4B snapshot offline, freeze it, wrap the 8 full-attention layers, inject LoRA."""
    from peft import LoraConfig, inject_adapter_in_model
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    snapshot_dir = str(snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        snapshot_dir, local_files_only=True, dtype=dtype, attn_implementation="sdpa")
    # The vision tower is not part of the design; drop it before moving to the accelerator.
    if hasattr(model.model, "visual"):
        del model.model.visual
        model.model.visual = None
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    text = model.model.language_model
    ctx = WildcardContext(backend=backend, pad_to=pad_to, flex_dynamic=flex_dynamic)
    wrapped = wrap_full_attention_layers(text, ctx)
    targets = lora_target_names(text)
    config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                        target_modules=targets)
    inject_adapter_in_model(config, text)
    lora_names = [name for name, _ in text.named_parameters() if "lora_" in name]
    expected = 2 * len(targets)
    if len(lora_names) != expected:
        raise RuntimeError(f"expected {expected} LoRA parameter tensors, found {len(lora_names)}")
    for name, parameter in text.named_parameters():
        parameter.requires_grad_("lora_" in name)
        if "lora_" in name and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    if gradient_checkpointing:
        text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return BackboneBundle(model=model, text=text, lm_head=model.lm_head, tokenizer=tokenizer, ctx=ctx,
                          wrapped_layers=wrapped, lora_targets=targets, lora_parameter_names=lora_names,
                          snapshot_dir=snapshot_dir)


__all__ = ["BackboneBundle", "load_backbone"]
