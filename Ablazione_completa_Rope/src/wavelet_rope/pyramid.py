"""Fixed FP32 Haar pyramid with absolute support geometry.

The transform follows the prototype design: odd boundaries are extended by
replicating the last value, while observation counts and padding remain separate.
Every detail band ``D_j`` and the final approximation ``A_J`` retain half-open
supports in original recording coordinates and FP64 anchor times.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor

from .signal_contract import SignalView, ViewKind


MAX_LEVELS = 12
GLOBAL_APPROXIMATION_TARGET = 64
LOCAL_APPROXIMATION_TARGET = 128
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def pyramid_levels(length: int, kind: ViewKind | str) -> int:
    """Return the architecture-fixed decomposition depth for a view."""

    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("length must be an integer")
    if length < 1:
        raise ValueError("length must be positive")
    kind = ViewKind(kind)
    target = (
        GLOBAL_APPROXIMATION_TARGET
        if kind is ViewKind.GLOBAL
        else LOCAL_APPROXIMATION_TARGET
    )
    requested = math.ceil(math.log2(max(2.0, length / target)))
    return max(1, min(MAX_LEVELS, requested))


@dataclass(frozen=True, slots=True)
class HaarBand:
    """One Haar band and the geometry of each coefficient.

    ``usable`` is the strict AND of the child validity masks.  Consequently,
    replicated boundary padding never increments ``observed_count`` and makes
    the coefficient unusable.  Invalid coefficient payloads are zero; support,
    count, padding, and anchor metadata remain available beside that mask.
    """

    kind: Literal["detail", "approximation"]
    level: int
    values: Tensor
    usable: Tensor
    observed_count: Tensor
    support_start: Tensor
    support_end: Tensor
    anchor_time: Tensor
    nominal_width: int
    padding_count: Tensor

    def __post_init__(self) -> None:
        if self.kind not in ("detail", "approximation"):
            raise ValueError(f"unknown Haar band kind {self.kind!r}")
        if self.level < 1 or self.nominal_width != 2**self.level:
            raise ValueError("nominal_width must equal 2**level")
        if self.values.ndim != 2 or self.values.dtype != torch.float32:
            raise TypeError("band values must be float32 with shape [C, N]")
        channels, coefficients = self.values.shape
        matrix_shape = (channels, coefficients)
        vector_shape = (coefficients,)
        for name in ("usable", "observed_count", "anchor_time"):
            value = getattr(self, name)
            if tuple(value.shape) != matrix_shape:
                raise ValueError(f"{name} must have shape {matrix_shape}")
        for name in ("support_start", "support_end", "padding_count"):
            value = getattr(self, name)
            if tuple(value.shape) != vector_shape:
                raise ValueError(f"{name} must have shape {vector_shape}")
        if self.usable.dtype != torch.bool:
            raise TypeError("usable must be bool")
        if self.observed_count.dtype != torch.int64:
            raise TypeError("observed_count must be int64")
        if self.anchor_time.dtype != torch.float64:
            raise TypeError("anchor_time must be float64")
        if any(
            value.dtype != torch.int64
            for value in (self.support_start, self.support_end, self.padding_count)
        ):
            raise TypeError("support and padding tensors must be int64")
        devices = {
            value.device
            for value in (
                self.values,
                self.usable,
                self.observed_count,
                self.support_start,
                self.support_end,
                self.anchor_time,
                self.padding_count,
            )
        }
        if len(devices) != 1:
            raise ValueError("all band tensors must be on the same device")

        support_length = self.support_end - self.support_start
        if not bool((support_length > 0).all()):
            raise ValueError("every coefficient must intersect a non-empty support")
        if not torch.equal(
            self.padding_count,
            torch.full_like(support_length, self.nominal_width) - support_length,
        ):
            raise ValueError("padding_count is inconsistent with the support geometry")
        if bool((self.observed_count < 0).any()) or bool(
            (self.observed_count > support_length.unsqueeze(0)).any()
        ):
            raise ValueError("observed_count is outside its support")
        expected_usable = self.observed_count.eq(self.nominal_width)
        if not torch.equal(self.usable, expected_usable):
            raise ValueError("usable must be the strict AND of the nominal child mask")
        if bool((self.observed_count > 0).any()):
            if not bool(torch.isfinite(self.anchor_time[self.observed_count > 0]).all()):
                raise ValueError("observed coefficients require finite anchor times")
        if bool((self.observed_count == 0).any()):
            if not bool((self.anchor_time[self.observed_count == 0] == 0).all()):
                raise ValueError("unobserved coefficients require zero anchor payloads")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("band payloads must be finite")
        if bool((self.values[~self.usable] != 0).any()):
            raise ValueError("unusable coefficient payloads must be zero")

    @property
    def support_length(self) -> Tensor:
        return self.support_end - self.support_start

    @property
    def fully_nominally_observed(self) -> Tensor:
        """Mask requiring no missing samples and no replicated boundary padding."""

        return self.observed_count.eq(self.nominal_width)

    @property
    def observed_fraction_of_support(self) -> Tensor:
        """Observed real samples divided by the intersected support length."""

        return self.observed_count.to(torch.float32) / self.support_length.unsqueeze(0)

    @property
    def observed_fraction_of_nominal(self) -> Tensor:
        """Observed samples divided by the nominal ``2**level`` block width."""

        return self.observed_count.to(torch.float32) / float(self.nominal_width)


@dataclass(frozen=True, slots=True)
class HaarPyramid:
    """All ``D_1..D_J`` bands plus ``A_J`` for one signal view."""

    view_kind: ViewKind
    trace: Literal["global", "local"]
    start: int
    end: int
    details: tuple[HaarBand, ...]
    approximation: HaarBand

    def __post_init__(self) -> None:
        if self.trace not in ("global", "local"):
            raise ValueError(f"unknown normalization trace {self.trace!r}")
        if not self.details:
            raise ValueError("a Haar pyramid must contain at least one detail band")
        for expected_level, band in enumerate(self.details, start=1):
            if band.kind != "detail" or band.level != expected_level:
                raise ValueError("detail bands must be ordered D_1 through D_J")
        if self.approximation.kind != "approximation":
            raise ValueError("the final band must be an approximation")
        if self.approximation.level != len(self.details):
            raise ValueError("A_J and the detail tuple must use the same J")
        if not self.start < self.end:
            raise ValueError("pyramid interval must be non-empty")

    @property
    def levels(self) -> int:
        return len(self.details)

    @property
    def length(self) -> int:
        return self.end - self.start

    def detail(self, level: int) -> HaarBand:
        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError("level must be an integer")
        if not 1 <= level <= self.levels:
            raise ValueError(f"level must be in [1, {self.levels}]")
        return self.details[level - 1]


def _band_geometry(view: SignalView, level: int) -> dict[str, Tensor | int]:
    width = 2**level
    channels, length = view.valid.shape
    coefficients = math.ceil(length / width)
    padded_length = coefficients * width
    padding = padded_length - length

    valid = view.valid
    times = view.sample_times.unsqueeze(0).expand(channels, -1)
    if padding:
        valid = torch.cat(
            (
                valid,
                torch.zeros((channels, padding), dtype=torch.bool, device=valid.device),
            ),
            dim=1,
        )
        times = torch.cat(
            (
                times,
                torch.zeros(
                    (channels, padding), dtype=torch.float64, device=times.device
                ),
            ),
            dim=1,
        )

    valid_blocks = valid.reshape(channels, coefficients, width)
    time_blocks = times.reshape(channels, coefficients, width)
    observed_count = valid_blocks.sum(dim=-1, dtype=torch.int64)
    anchor_time = torch.where(
        valid_blocks,
        time_blocks,
        torch.zeros_like(time_blocks),
    ).sum(dim=-1) / observed_count.clamp_min(1).to(torch.float64)
    anchor_time = torch.where(
        observed_count > 0,
        anchor_time,
        torch.zeros_like(anchor_time),
    ).to(torch.float64)

    support_start = (
        torch.arange(coefficients, dtype=torch.int64, device=view.values.device) * width
        + view.start
    )
    support_end = torch.minimum(
        support_start + width,
        torch.full_like(support_start, view.end),
    )
    support_length = support_end - support_start
    padding_count = torch.full_like(support_length, width) - support_length
    usable = observed_count.eq(width)
    return {
        "usable": usable,
        "observed_count": observed_count,
        "support_start": support_start,
        "support_end": support_end,
        "anchor_time": anchor_time,
        "nominal_width": width,
        "padding_count": padding_count,
    }


def build_haar_pyramid(
    view: SignalView,
    *,
    trace: Literal["global", "local"] = "global",
) -> HaarPyramid:
    """Build the fixed Haar pyramid for one normalized view trace.

    A local view exposes both normalization traces.  Calling this function once
    for ``trace="global"`` and once for ``trace="local"`` keeps that decision
    explicit for the later local stem instead of silently concatenating them.
    """

    if trace == "global":
        values = view.global_normalized
    elif trace == "local":
        if view.kind is not ViewKind.LOCAL or view.local_normalized is None:
            raise ValueError("the local normalization trace exists only on local views")
        values = view.local_normalized
    else:
        raise ValueError(f"trace must be 'global' or 'local', got {trace!r}")

    levels = pyramid_levels(view.length, view.kind)
    # Invalid real positions carry a finite neutral storage value.  No usable
    # coefficient depends on it because its entire support is masked below.
    approximation = torch.where(
        view.valid,
        values.to(torch.float32),
        torch.zeros_like(values, dtype=torch.float32),
    )
    approximation_valid = view.valid
    details: list[HaarBand] = []

    # Explicitly disable autocast: the specification requires the DWT itself to
    # run in FP32 even when the surrounding perceptual model runs in BF16.
    with torch.autocast(device_type=approximation.device.type, enabled=False):
        for level in range(1, levels + 1):
            if approximation.shape[-1] % 2:
                approximation = torch.cat(
                    (approximation, approximation[..., -1:]), dim=-1
                )
                approximation_valid = torch.cat(
                    (
                        approximation_valid,
                        torch.zeros(
                            (*approximation_valid.shape[:-1], 1),
                            dtype=torch.bool,
                            device=approximation_valid.device,
                        ),
                    ),
                    dim=-1,
                )
            even = approximation[..., 0::2]
            odd = approximation[..., 1::2]
            pair_valid = (
                approximation_valid[..., 0::2]
                & approximation_valid[..., 1::2]
            )
            raw_detail = ((even - odd) * _INV_SQRT2).to(torch.float32)
            raw_approximation = ((even + odd) * _INV_SQRT2).to(torch.float32)
            detail_values = torch.where(
                pair_valid, raw_detail, torch.zeros_like(raw_detail)
            )
            approximation = torch.where(
                pair_valid,
                raw_approximation,
                torch.zeros_like(raw_approximation),
            )
            approximation_valid = pair_valid
            geometry = _band_geometry(view, level)
            if not torch.equal(pair_valid, geometry["usable"]):
                raise RuntimeError("recursive validity disagrees with support geometry")
            details.append(
                HaarBand(
                    kind="detail",
                    level=level,
                    values=detail_values,
                    **geometry,
                )
            )

    final_level = levels
    final_band = HaarBand(
        kind="approximation",
        level=final_level,
        values=approximation,
        **_band_geometry(view, final_level),
    )
    return HaarPyramid(
        view_kind=view.kind,
        trace=trace,
        start=view.start,
        end=view.end,
        details=tuple(details),
        approximation=final_band,
    )


__all__ = [
    "GLOBAL_APPROXIMATION_TARGET",
    "HaarBand",
    "HaarPyramid",
    "LOCAL_APPROXIMATION_TARGET",
    "MAX_LEVELS",
    "build_haar_pyramid",
    "pyramid_levels",
]
