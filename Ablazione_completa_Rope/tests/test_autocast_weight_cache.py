"""Regression test for the fp32-weight autocast cache."""

from __future__ import annotations

import pytest
import torch
from torch import nn


def _run(device: torch.device, *, separate_contexts: bool, cache_enabled: bool = True):
    linear = nn.Linear(4, 2, bias=False, device=device, dtype=torch.float32)
    x = torch.tensor([[0.25, -0.5, 1.5, 2.0]], device=device, dtype=torch.float32)
    delta = torch.zeros_like(linear.weight)
    delta[0, 0] = 8.0
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 2.0, -2.0]], device=device))

    def forward():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=cache_enabled):
            return linear(x)

    if separate_contexts:
        first = forward()
        with torch.no_grad():
            linear.weight.add_(delta)
        second = forward()
    else:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=cache_enabled):
            first = linear(x)
            with torch.no_grad():
                linear.weight.add_(delta)
            second = linear(x)
    return first, second


def test_autocast_weight_cache_requires_a_fresh_context_after_weight_mutation() -> None:
    device = torch.device("cpu")
    try:
        wrong = _run(device, separate_contexts=False)
    except RuntimeError:
        wrong = None
    if wrong is None or not torch.equal(wrong[0], wrong[1]):
        if not torch.cuda.is_available():
            pytest.skip("CPU autocast does not expose the weight-cache behavior and CUDA is unavailable")
        device = torch.device("cuda")
        wrong = _run(device, separate_contexts=False)

    assert torch.equal(wrong[0], wrong[1])
    fresh = _run(device, separate_contexts=True)
    uncached = _run(device, separate_contexts=False, cache_enabled=False)
    assert not torch.equal(fresh[0], fresh[1])
    assert not torch.equal(uncached[0], uncached[1])
