"""Canonical signal, normalization, and view contracts for WR-Agent.

This module deliberately has no dataset or model dependencies.  Dataset adapters
must convert their rows into :class:`SignalEpisode` once; every global or local
view is then derived from that episode so coordinates and timestamps cannot be
silently rebased.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Literal, Sequence

import torch
from torch import Tensor


MAX_CHANNELS = 16
MAD_TO_SIGMA = 1.4826
RELATIVE_SCALE_FLOOR = 1e-6
NO_VALID_OBSERVATION = "NO_VALID_OBSERVATION"


class SignalContractError(ValueError):
    """A stable, machine-readable failure at the signal boundary."""

    code = "SIGNAL_CONTRACT_ERROR"


class NoValidObservationError(SignalContractError):
    """Raised when a requested channel/view has no observable sample."""

    code = NO_VALID_OBSERVATION

    def __init__(self, channel_ids: Sequence[str]):
        self.channel_ids = tuple(channel_ids)
        super().__init__(
            f"{self.code}: no valid samples for channel(s) "
            + ", ".join(repr(item) for item in self.channel_ids)
        )


class ViewKind(str, Enum):
    """The two perceptual views defined by the architecture."""

    GLOBAL = "global"
    LOCAL = "local"


class ViewStatus(str, Enum):
    """Whether every selected channel has a local observation."""

    COMPLETE = "complete"
    PARTIAL = "partial"


def _require_shape(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")


def _median_1d(values: Tensor) -> Tensor:
    """Return the conventional median, averaging the two middle values."""

    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("median requires a non-empty one-dimensional tensor")
    ordered = values.sort().values
    midpoint = ordered.numel() // 2
    if ordered.numel() % 2:
        return ordered[midpoint]
    left = ordered[midpoint - 1]
    right = ordered[midpoint]
    return left + (right - left) * 0.5


@dataclass(frozen=True, slots=True)
class SignalEpisode:
    """One uniformly sampled multichannel recording.

    ``values`` and ``valid`` use channel-major layout ``[C, L]``.  The task is
    intentionally opaque here: the dataset adapter owns its text/category/schema
    representation, while this module validates only the signal contract.
    """

    values: Tensor
    valid: Tensor
    channel_ids: tuple[str, ...]
    sample_times: Tensor
    task: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, Tensor) or self.values.ndim != 2:
            raise TypeError("values must be a torch.Tensor with shape [C, L]")
        if self.values.dtype != torch.float32:
            raise TypeError(f"values must be float32, got {self.values.dtype}")
        channels, length = self.values.shape
        if not 1 <= channels <= MAX_CHANNELS:
            raise ValueError(f"channel count must be in [1, {MAX_CHANNELS}], got {channels}")
        if length < 1:
            raise ValueError("a signal episode must contain at least one sample")

        if not isinstance(self.valid, Tensor):
            raise TypeError("valid must be a torch.Tensor")
        _require_shape("valid", self.valid, (channels, length))
        if self.valid.dtype != torch.bool:
            raise TypeError(f"valid must be bool, got {self.valid.dtype}")
        if self.valid.device != self.values.device:
            raise ValueError("values and valid must be on the same device")

        channel_ids = tuple(self.channel_ids)
        object.__setattr__(self, "channel_ids", channel_ids)
        if len(channel_ids) != channels:
            raise ValueError(
                f"channel_ids must contain {channels} entries, got {len(channel_ids)}"
            )
        if any(not isinstance(item, str) or not item for item in channel_ids):
            raise ValueError("every channel id must be a non-empty string")
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError("channel ids must be unique within an episode")

        if not isinstance(self.sample_times, Tensor) or self.sample_times.ndim != 1:
            raise TypeError("sample_times must be a torch.Tensor with shape [L]")
        _require_shape("sample_times", self.sample_times, (length,))
        if self.sample_times.dtype != torch.float64:
            raise TypeError(f"sample_times must be float64, got {self.sample_times.dtype}")
        if self.sample_times.device != self.values.device:
            raise ValueError("values and sample_times must be on the same device")
        if not bool(torch.isfinite(self.sample_times).all()):
            raise ValueError("sample_times must be finite")

        if length > 1:
            deltas = torch.diff(self.sample_times)
            if not bool((deltas > 0).all()):
                raise ValueError("sample_times must be strictly increasing")
            reference = deltas[0]
            atol = max(1e-12, abs(float(reference)) * 1e-12)
            if not torch.allclose(
                deltas,
                torch.full_like(deltas, reference),
                rtol=1e-9,
                atol=atol,
            ):
                raise ValueError("sample_times must lie on one uniform grid")

        valid_values = self.values[self.valid]
        if valid_values.numel() and not bool(torch.isfinite(valid_values).all()):
            raise ValueError("values marked valid must be finite")

    @classmethod
    def from_index_grid(
        cls,
        *,
        values: Tensor,
        valid: Tensor,
        channel_ids: Sequence[str],
        dt: float,
        origin: float = 0.0,
        task: Any = None,
    ) -> "SignalEpisode":
        """Build absolute sample times from a declared uniform index grid."""

        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and strictly positive")
        if not math.isfinite(origin):
            raise ValueError("origin must be finite")
        if not isinstance(values, Tensor) or values.ndim != 2:
            raise TypeError("values must be a torch.Tensor with shape [C, L]")
        times = (
            torch.arange(values.shape[1], dtype=torch.float64, device=values.device)
            * float(dt)
            + float(origin)
        )
        return cls(values, valid, tuple(channel_ids), times, task)

    @property
    def n_channels(self) -> int:
        return int(self.values.shape[0])

    @property
    def length(self) -> int:
        return int(self.values.shape[1])

    @property
    def dt(self) -> float | None:
        if self.length < 2:
            return None
        return float(self.sample_times[1] - self.sample_times[0])


@dataclass(frozen=True, slots=True)
class ChannelNormalization:
    """Robust per-channel statistics and the epsilon used to obtain them."""

    channel_ids: tuple[str, ...]
    median: Tensor
    scale: Tensor
    minimum: Tensor
    maximum: Tensor
    valid_count: Tensor
    available: Tensor
    epsilon: Tensor
    constant_channel: Tensor
    scale_floor_used: Tensor
    scope: Literal["recording", "view"]
    source_length: int

    def __post_init__(self) -> None:
        channels = len(self.channel_ids)
        for name in ("median", "scale", "minimum", "maximum"):
            value = getattr(self, name)
            _require_shape(name, value, (channels,))
            if value.dtype != torch.float32:
                raise TypeError(f"{name} must be float32, got {value.dtype}")
        _require_shape("epsilon", self.epsilon, (channels,))
        if self.epsilon.dtype != torch.float64:
            raise TypeError(f"epsilon must be float64, got {self.epsilon.dtype}")
        _require_shape("valid_count", self.valid_count, (channels,))
        if self.valid_count.dtype != torch.int64:
            raise TypeError("valid_count must be int64")
        _require_shape("available", self.available, (channels,))
        if self.available.dtype != torch.bool:
            raise TypeError("available must be bool")
        for name in ("constant_channel", "scale_floor_used"):
            value = getattr(self, name)
            _require_shape(name, value, (channels,))
            if value.dtype != torch.bool:
                raise TypeError(f"{name} must be bool")
        if self.scope not in ("recording", "view"):
            raise ValueError(f"unknown normalization scope {self.scope!r}")
        if self.source_length < 1:
            raise ValueError("source_length must be positive")

    def select(self, indices: Sequence[int]) -> "ChannelNormalization":
        index = torch.as_tensor(indices, dtype=torch.int64, device=self.median.device)
        if index.ndim != 1:
            raise ValueError("normalization indices must be one-dimensional")
        ids = tuple(self.channel_ids[int(i)] for i in index.tolist())
        return ChannelNormalization(
            channel_ids=ids,
            median=self.median.index_select(0, index),
            scale=self.scale.index_select(0, index),
            minimum=self.minimum.index_select(0, index),
            maximum=self.maximum.index_select(0, index),
            valid_count=self.valid_count.index_select(0, index),
            available=self.available.index_select(0, index),
            epsilon=self.epsilon.index_select(0, index),
            constant_channel=self.constant_channel.index_select(0, index),
            scale_floor_used=self.scale_floor_used.index_select(0, index),
            scope=self.scope,
            source_length=self.source_length,
        )


def _compute_stats(
    values: Tensor,
    valid: Tensor,
    channel_ids: tuple[str, ...],
    *,
    scope: Literal["recording", "view"],
) -> ChannelNormalization:
    channels, length = values.shape
    medians: list[Tensor] = []
    scales: list[Tensor] = []
    minima: list[Tensor] = []
    maxima: list[Tensor] = []
    counts: list[int] = []
    epsilons: list[Tensor] = []
    availability: list[bool] = []
    constants: list[Tensor] = []
    floor_flags: list[Tensor] = []

    # C is capped at 16, and the loop avoids padding invalid values into a
    # reduction whose numerical meaning would depend on their placeholders.
    empty_channels = [
        channel_ids[channel]
        for channel in range(channels)
        if not bool(valid[channel].any())
    ]
    if len(empty_channels) == channels or (scope == "recording" and empty_channels):
        raise NoValidObservationError(empty_channels)

    for channel in range(channels):
        observed = values[channel, valid[channel]].to(dtype=torch.float32)
        if observed.numel() == 0:
            zero_fp32 = torch.zeros((), dtype=torch.float32, device=values.device)
            zero_fp64 = torch.zeros((), dtype=torch.float64, device=values.device)
            medians.append(zero_fp32)
            scales.append(zero_fp32)
            minima.append(zero_fp32)
            maxima.append(zero_fp32)
            counts.append(0)
            epsilons.append(zero_fp64)
            availability.append(False)
            constants.append(torch.tensor(False, device=values.device))
            floor_flags.append(torch.tensor(False, device=values.device))
            continue
        median = _median_1d(observed)
        mad = _median_1d((observed - median).abs())
        robust_scale = mad * MAD_TO_SIGMA
        max_abs = observed.to(torch.float64).abs().amax()
        epsilon = torch.maximum(
            max_abs * RELATIVE_SCALE_FLOOR,
            torch.tensor(
                torch.finfo(torch.float32).tiny,
                dtype=torch.float64,
                device=values.device,
            ),
        )
        robust_scale_fp64 = robust_scale.to(torch.float64)
        floor_used = robust_scale_fp64 <= epsilon
        scale = torch.maximum(robust_scale_fp64, epsilon).to(torch.float32)
        minimum = observed.min()
        maximum = observed.max()
        medians.append(median)
        scales.append(scale)
        minima.append(minimum)
        maxima.append(maximum)
        counts.append(observed.numel())
        epsilons.append(epsilon)
        availability.append(True)
        constants.append(minimum == maximum)
        floor_flags.append(floor_used)

    return ChannelNormalization(
        channel_ids=channel_ids,
        median=torch.stack(medians).to(torch.float32),
        scale=torch.stack(scales).to(torch.float32),
        minimum=torch.stack(minima).to(torch.float32),
        maximum=torch.stack(maxima).to(torch.float32),
        valid_count=torch.tensor(counts, dtype=torch.int64, device=values.device),
        available=torch.tensor(availability, dtype=torch.bool, device=values.device),
        epsilon=torch.stack(epsilons).to(torch.float64),
        constant_channel=torch.stack(constants).to(torch.bool),
        scale_floor_used=torch.stack(floor_flags).to(torch.bool),
        scope=scope,
        source_length=length,
    )


def compute_recording_normalization(
    episode: SignalEpisode,
) -> ChannelNormalization:
    """Compute full-recording robust statistics in FP32.

    Epsilon is computed per channel in FP64 as
    ``max(1e-6 * max_abs_valid, finfo(float32).tiny)``.
    """

    return _compute_stats(
        episode.values,
        episode.valid,
        episode.channel_ids,
        scope="recording",
    )


def apply_normalization(values: Tensor, valid: Tensor, stats: ChannelNormalization) -> Tensor:
    """Apply channel statistics, returning FP32 with invalid positions zeroed."""

    if values.ndim != 2:
        raise ValueError("values must have shape [C, L]")
    _require_shape("valid", valid, tuple(values.shape))
    if values.dtype != torch.float32 or valid.dtype != torch.bool:
        raise TypeError("normalization requires float32 values and a bool valid mask")
    if values.device != stats.median.device or valid.device != values.device:
        raise ValueError("values, valid, and statistics must be on the same device")
    if values.shape[0] != len(stats.channel_ids):
        raise ValueError("statistics do not match the number of value channels")
    safe_scale = torch.where(
        stats.available,
        stats.scale,
        torch.ones_like(stats.scale),
    )
    normalized = (values - stats.median[:, None]) / safe_scale[:, None]
    # The zero is only a finite storage value.  The validity mask remains the
    # authority, and pyramid coefficients touching an invalid real sample are
    # marked unusable.
    usable = valid & stats.available[:, None]
    return torch.where(usable, normalized, torch.zeros_like(normalized)).to(torch.float32)


@dataclass(frozen=True, slots=True)
class SignalView:
    """A global recording view or a local crop with absolute coordinates."""

    kind: ViewKind
    values: Tensor
    valid: Tensor
    channel_ids: tuple[str, ...]
    channel_indices: tuple[int, ...]
    sample_times: Tensor
    start: int
    end: int
    recording_length: int
    global_normalized: Tensor
    global_stats: ChannelNormalization
    local_normalized: Tensor | None = None
    local_stats: ChannelNormalization | None = None

    def __post_init__(self) -> None:
        channels = len(self.channel_ids)
        length = self.end - self.start
        if not 0 <= self.start < self.end <= self.recording_length:
            raise ValueError("view interval must be inside the recording and half-open")
        _require_shape("values", self.values, (channels, length))
        _require_shape("valid", self.valid, (channels, length))
        _require_shape("sample_times", self.sample_times, (length,))
        _require_shape("global_normalized", self.global_normalized, (channels, length))
        if self.kind is ViewKind.GLOBAL:
            if self.start != 0 or self.end != self.recording_length:
                raise ValueError("a global view must cover the full recording")
            if self.local_normalized is not None or self.local_stats is not None:
                raise ValueError("a global view must not carry local normalization")
        elif self.kind is ViewKind.LOCAL:
            if self.local_normalized is None or self.local_stats is None:
                raise ValueError("a local view requires its secondary normalization")
            _require_shape("local_normalized", self.local_normalized, (channels, length))
        else:  # pragma: no cover - Enum prevents this through the public builder.
            raise ValueError(f"unknown view kind {self.kind!r}")

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def local_to_global_scale(self) -> Tensor | None:
        if self.local_stats is None:
            return None
        ratio = self.local_stats.scale / self.global_stats.scale
        return torch.where(
            self.local_stats.available,
            ratio,
            torch.zeros_like(ratio),
        )

    @property
    def local_normalization_available(self) -> Tensor | None:
        """Mask for the zero-filled local statistics and scale ratios."""

        return None if self.local_stats is None else self.local_stats.available

    @property
    def status(self) -> ViewStatus:
        if self.local_stats is None or bool(self.local_stats.available.all()):
            return ViewStatus.COMPLETE
        return ViewStatus.PARTIAL


def make_view(
    episode: SignalEpisode,
    recording_stats: ChannelNormalization,
    *,
    kind: ViewKind | str,
    start: int = 0,
    end: int | None = None,
    channels: Sequence[str] | None = None,
) -> SignalView:
    """Create a view without ever rebasing its indices or timestamps."""

    kind = ViewKind(kind)
    if end is None:
        end = episode.length
    if isinstance(start, bool) or isinstance(end, bool):
        raise TypeError("view coordinates must be integers, not booleans")
    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError("view coordinates must be integers")
    if not 0 <= start < end <= episode.length:
        raise ValueError(
            f"view interval must satisfy 0 <= start < end <= {episode.length}"
        )
    if recording_stats.scope != "recording":
        raise ValueError("recording_stats must have recording scope")
    if (
        recording_stats.channel_ids != episode.channel_ids
        or recording_stats.source_length != episode.length
    ):
        raise ValueError("recording statistics do not belong to this episode contract")

    requested = tuple(episode.channel_ids if channels is None else channels)
    if not requested:
        raise ValueError("a view must contain at least one channel")
    if len(set(requested)) != len(requested):
        raise ValueError("view channels must not contain duplicates")
    lookup = {name: index for index, name in enumerate(episode.channel_ids)}
    unknown = [name for name in requested if name not in lookup]
    if unknown:
        raise ValueError(f"unknown view channels: {unknown}")
    indices = tuple(lookup[name] for name in requested)

    if kind is ViewKind.GLOBAL:
        if start != 0 or end != episode.length or requested != episode.channel_ids:
            raise ValueError("the global view must include the full recording and all channels")

    index = torch.tensor(indices, dtype=torch.int64, device=episode.values.device)
    values = episode.values.index_select(0, index)[:, start:end]
    valid = episode.valid.index_select(0, index)[:, start:end]
    selected_global_stats = recording_stats.select(indices)
    global_values = apply_normalization(values, valid, selected_global_stats)

    local_stats = None
    local_values = None
    if kind is ViewKind.LOCAL:
        local_stats = _compute_stats(
            values,
            valid,
            requested,
            scope="view",
        )
        local_values = apply_normalization(values, valid, local_stats)

    return SignalView(
        kind=kind,
        values=values,
        valid=valid,
        channel_ids=requested,
        channel_indices=indices,
        sample_times=episode.sample_times[start:end],
        start=start,
        end=end,
        recording_length=episode.length,
        global_normalized=global_values,
        global_stats=selected_global_stats,
        local_normalized=local_values,
        local_stats=local_stats,
    )


__all__ = [
    "ChannelNormalization",
    "MAD_TO_SIGMA",
    "MAX_CHANNELS",
    "NO_VALID_OBSERVATION",
    "NoValidObservationError",
    "RELATIVE_SCALE_FLOOR",
    "SignalEpisode",
    "SignalContractError",
    "SignalView",
    "ViewKind",
    "ViewStatus",
    "apply_normalization",
    "compute_recording_normalization",
    "make_view",
]
