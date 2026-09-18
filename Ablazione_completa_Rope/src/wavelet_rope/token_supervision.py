"""Linear, gold-free supervision for the RMS-normalised soft tokens."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .haar_selector import _normalised_energy_scores
from .tokenizer import ViewTokens


DEFAULT_SOFT_DIM = 2560
DEFAULT_N_BINS = 64
MAD_TO_SIGMA = 1.4826
PEAK_SCALE_FLOOR = 1e-6


def _token_config(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either the token-supervision block or a full experiment spec."""

    training = spec.get("training")
    if isinstance(training, Mapping) and "token_supervision" in training:
        value = training["token_supervision"]
        if not isinstance(value, Mapping):
            raise TypeError("training.token_supervision must be a mapping")
        config = dict(value)
        tokenizer = spec.get("tokenizer")
        if "n_bins" not in config and isinstance(tokenizer, Mapping):
            global_per_channel = tokenizer.get("global_per_channel")
            if isinstance(global_per_channel, Mapping) and "A" in global_per_channel:
                config["n_bins"] = int(global_per_channel["A"])
        return config
    return spec


def load_token_supervision_stats(path: str | Path) -> dict[str, Any]:
    """Load and validate the fixed train statistics used by the supervision heads."""

    stats_path = Path(path)
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"token supervision is enabled but stats_path does not exist: {stats_path}"
        )
    try:
        document = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read token supervision statistics from {stats_path}") from error
    if not isinstance(document, Mapping):
        raise ValueError(f"token supervision statistics must be a JSON object: {stats_path}")
    for name in ("position", "energy", "peak"):
        item = document.get(name)
        if not isinstance(item, Mapping):
            raise ValueError(f"token supervision statistics are missing {name}.mean/std")
        for field in ("mean", "std"):
            value = item.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"token supervision statistics {name}.{field} must be finite")
        if float(item["std"]) <= 0.0:
            raise ValueError(f"token supervision statistics {name}.std must be positive")
    n_bins = document.get("n_bins")
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("token supervision statistics n_bins must be a positive integer")
    return dict(document)


def _stats_tensors(stats: Mapping[str, Any]) -> tuple[Tensor, Tensor, int]:
    means = torch.tensor(
        [float(stats[name]["mean"]) for name in ("position", "energy", "peak")],
        dtype=torch.float32,
    )
    stds = torch.tensor(
        [float(stats[name]["std"]) for name in ("position", "energy", "peak")],
        dtype=torch.float32,
    )
    if not bool(torch.isfinite(means).all()) or not bool(torch.isfinite(stds).all()) or bool((stds <= 0).any()):
        raise ValueError("token supervision statistics must have finite means and positive standard deviations")
    n_bins = int(stats["n_bins"])
    return means, stds, n_bins


class TokenSupervisionHeads(nn.Module):
    """Two linear heads on the projector's RMS-normalised soft tokens."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        *,
        soft_dim: int = DEFAULT_SOFT_DIM,
        target_rms: float | None = None,
        stats: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        config = _token_config(spec)
        backbone = spec.get("backbone")
        if isinstance(backbone, Mapping) and "projector_output_dim" in backbone:
            soft_dim = int(backbone["projector_output_dim"])

        raw_inputs = config.get("inputs", ("soft",))
        if isinstance(raw_inputs, (str, bytes)):
            raise TypeError("token_supervision.inputs must be a list containing only 'soft'")
        inputs = tuple(str(name) for name in raw_inputs)
        if inputs != ("soft",):
            raise ValueError("token supervision accepts only inputs=['soft']; Z supervision is disabled")

        hidden = int(config.get("hidden", 0))
        if hidden != 0:
            raise ValueError("token supervision is linear-only; hidden must be zero")
        if int(soft_dim) < 1:
            raise ValueError("token supervision soft dimension must be positive")

        n_bins = int(config.get("n_bins", DEFAULT_N_BINS))
        if n_bins < 1:
            raise ValueError("token_supervision.n_bins must be positive")
        weights = {
            "position": float(config.get("position_weight", 1.0)),
            "energy": float(config.get("energy_weight", 1.0)),
            "peak": float(config.get("peak_weight", 0.5)),
            "rank": float(config.get("rank_weight", 0.2)),
        }
        if any(value < 0.0 for value in weights.values()):
            raise ValueError("token supervision weights must be non-negative")
        if target_rms is not None and (not math.isfinite(float(target_rms)) or float(target_rms) <= 0.0):
            raise ValueError("projector target_rms must be finite and positive")

        if stats is None:
            stats_path = config.get("stats_path")
            if not stats_path:
                raise FileNotFoundError(
                    "token supervision is enabled but training.token_supervision.stats_path is missing"
                )
            stats = load_token_supervision_stats(stats_path)
        means, stds, stats_bins = _stats_tensors(stats)
        if stats_bins != n_bins:
            raise ValueError(
                f"token supervision statistics use n_bins={stats_bins}, expected configured n_bins={n_bins}"
            )

        self.inputs = ("soft",)
        self.input_dims = {"soft": int(soft_dim)}
        self.n_bins = n_bins
        self.stats_n_bins = stats_bins
        self.position_weight = weights["position"]
        self.energy_weight = weights["energy"]
        self.peak_weight = weights["peak"]
        self.rank_weight = weights["rank"]
        self.target_rms = float(target_rms) if target_rms is not None else None
        self.position_head = nn.Linear(int(soft_dim), 1)
        self.energy_peak_head = nn.Linear(int(soft_dim), 2)
        self.register_buffer("target_means", means)
        self.register_buffer("target_stds", stds)

    def _normalise_soft(self, soft: Tensor) -> Tensor:
        if not isinstance(soft, Tensor) or soft.ndim != 2:
            raise ValueError("soft must have shape [N, D]")
        if int(soft.shape[-1]) != self.input_dims["soft"]:
            raise ValueError(
                f"soft has dim {soft.shape[-1]}, expected {self.input_dims['soft']}"
            )
        if self.target_rms is None:
            return soft
        return soft.float() / float(self.target_rms)

    @staticmethod
    def _unwrap_soft(soft: Tensor | Mapping[str, Tensor]) -> Tensor:
        if isinstance(soft, Mapping):
            unexpected = set(soft).difference({"soft"})
            if unexpected:
                raise ValueError(
                    "token supervision accepts only the soft tensor; metadata and Z supervision are disabled"
                )
            value = soft.get("soft")
            if value is None:
                raise ValueError("missing token-supervision input 'soft'")
            return value
        return soft

    def forward(self, soft: Tensor | Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Predict standardized position, energy and peak targets."""

        value = self._normalise_soft(self._unwrap_soft(soft))
        energy_peak = self.energy_peak_head(value)
        return {
            "position": self.position_head(value).squeeze(-1),
            "energy": energy_peak[..., 0],
            "peak": energy_peak[..., 1],
        }

    @staticmethod
    def _vector(
        value: Tensor,
        name: str,
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.ndim == 2 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.ndim != 1:
            raise ValueError(f"{name} must have shape [N]")
        value = value.to(device=device)
        return value if dtype is None else value.to(dtype=dtype)

    @staticmethod
    def _group_mean(values: Tensor, group: Tensor, n_groups: int) -> Tensor:
        sums = values.new_zeros(n_groups)
        counts = values.new_zeros(n_groups)
        sums.index_add_(0, group, values)
        counts.index_add_(0, group, torch.ones_like(values))
        return sums / counts.clamp_min(1.0)

    @classmethod
    def _nested_loss(cls, error: Tensor, view: Tensor, channel: Tensor, bins: Tensor) -> Tensor:
        if error.numel() == 0:
            return error.sum() * 0.0
        channel_base = max(1, int(channel.max().item()) + 1)
        bin_base = max(1, int(bins.max().item()) + 1)
        group_key = (view * channel_base + channel) * bin_base + bins
        unique, token_group = torch.unique(group_key, sorted=True, return_inverse=True)
        group_error = cls._group_mean(error, token_group, int(unique.numel()))
        group_view = unique // (channel_base * bin_base)
        group_channel = (unique // bin_base) % channel_base

        channel_key = group_view * channel_base + group_channel
        channel_unique, group_channel_index = torch.unique(channel_key, sorted=True, return_inverse=True)
        channel_error = cls._group_mean(group_error, group_channel_index, int(channel_unique.numel()))
        view_index = channel_unique // channel_base
        view_unique, channel_view_index = torch.unique(view_index, sorted=True, return_inverse=True)
        view_error = cls._group_mean(channel_error, channel_view_index, int(view_unique.numel()))
        return view_error.mean()

    @staticmethod
    def _rank_loss(group_pred_energy: Tensor, group_target_energy: Tensor, group_view: Tensor) -> Tensor:
        """Rank teacher top-3 bins against the most dangerous predicted competitors."""

        if group_pred_energy.numel() == 0:
            return group_pred_energy.sum() * 0.0

        view_losses: list[Tensor] = []
        for view_id in torch.unique(group_view, sorted=True).tolist():
            view_mask = group_view == int(view_id)
            count = int(view_mask.sum().item())
            if count < 4:
                continue

            view_pred = group_pred_energy[view_mask]
            view_target = group_target_energy[view_mask]
            target_top = torch.argsort(view_target, descending=True, stable=True)[:3]

            excluded = torch.ones(count, dtype=torch.bool, device=view_mask.device)
            excluded[target_top] = False
            excluded_indices = torch.nonzero(excluded, as_tuple=False).squeeze(1)
            predicted_order = torch.argsort(
                view_pred[excluded_indices], descending=True, stable=True
            )[:16]
            competitors = excluded_indices[predicted_order]

            unequal_targets = (
                view_target[target_top, None] != view_target[competitors][None, :]
            )
            if not bool(unequal_targets.any()):
                continue
            deltas = view_pred[competitors][None, :] - view_pred[target_top][:, None]
            view_losses.append(F.softplus(deltas[unequal_targets]).mean())

        if not view_losses:
            return group_pred_energy.sum() * 0.0
        return torch.stack(view_losses).mean()

    @staticmethod
    def _pearson(prediction: Tensor, target: Tensor) -> float:
        if prediction.numel() < 2:
            return 0.0
        x = prediction.detach().float().double()
        y = target.detach().float().double()
        x = x - x.mean()
        y = y - y.mean()
        denominator = torch.sqrt(x.square().sum() * y.square().sum())
        if float(denominator) <= torch.finfo(torch.float64).eps:
            return 0.0
        return float((x * y).sum() / denominator)

    @staticmethod
    def _zero_metrics() -> dict[str, float]:
        return {
            "position_loss": 0.0,
            "energy_loss": 0.0,
            "peak_loss": 0.0,
            "rank_loss": 0.0,
            "position_pearson": 0.0,
            "position_mae_bins": 0.0,
            "energy_pearson": 0.0,
            "peak_pearson": 0.0,
            "top3_recall": 0.0,
            "energy_balanced_accuracy": 0.0,
        }

    def loss(
        self,
        soft: Tensor | Mapping[str, Tensor],
        position_target: Tensor | None = None,
        energy_target: Tensor | None = None,
        peak_target: Tensor | None = None,
        valid: Tensor | None = None,
        *,
        channel_index: Tensor | None = None,
        bin_index: Tensor | None = None,
        view_index: Tensor | None = None,
        position_bin_scale: Tensor | float | None = None,
        targets: Mapping[str, Tensor] | None = None,
        position_weight: float | None = None,
        energy_weight: float | None = None,
        peak_weight: float | None = None,
        rank_weight: float | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute nested token/bin/channel/view loss and raw-unit diagnostics."""

        if targets is not None:
            position_target = position_target if position_target is not None else targets.get("position")
            energy_target = energy_target if energy_target is not None else targets.get("energy")
            peak_target = peak_target if peak_target is not None else targets.get("peak")
            valid = valid if valid is not None else targets.get("valid")
        if position_target is None or energy_target is None or peak_target is None:
            raise ValueError("position, energy and peak targets are required")

        soft_tensor = self._unwrap_soft(soft)
        predictions = self.forward(soft_tensor)
        device = soft_tensor.device
        position = self._vector(position_target, "position_target", device=device, dtype=torch.float32)
        energy = self._vector(energy_target, "energy_target", device=device, dtype=torch.float32)
        peak = self._vector(peak_target, "peak_target", device=device, dtype=torch.float32)
        n_tokens = int(soft_tensor.shape[0])
        if not (position.numel() == energy.numel() == peak.numel() == n_tokens):
            raise ValueError("soft and all targets must have the same token count")
        if valid is None:
            valid_mask = torch.ones(n_tokens, dtype=torch.bool, device=device)
        else:
            valid_mask = self._vector(valid, "valid", device=device).bool()
            if valid_mask.numel() != n_tokens:
                raise ValueError("valid must have the same token count as soft")

        if channel_index is None:
            channel = torch.zeros(n_tokens, dtype=torch.int64, device=device)
        else:
            channel = self._vector(channel_index, "channel_index", device=device).to(torch.int64)
        if bin_index is None:
            bins = torch.arange(n_tokens, dtype=torch.int64, device=device)
        else:
            bins = self._vector(bin_index, "bin_index", device=device).to(torch.int64)
        if view_index is None:
            views = torch.zeros(n_tokens, dtype=torch.int64, device=device)
        else:
            views = self._vector(view_index, "view_index", device=device).to(torch.int64)
        if channel.numel() != n_tokens or bins.numel() != n_tokens or views.numel() != n_tokens:
            raise ValueError("group indices must have the same token count as soft")
        if bool((channel < 0).any()) or bool((bins < 0).any()) or bool((views < 0).any()):
            raise ValueError("group indices must be non-negative")

        if position_bin_scale is None:
            bin_scale = torch.ones(n_tokens, dtype=torch.float32, device=device)
        elif isinstance(position_bin_scale, Tensor):
            bin_scale = self._vector(position_bin_scale, "position_bin_scale", device=device, dtype=torch.float32)
        else:
            bin_scale = torch.full((n_tokens,), float(position_bin_scale), dtype=torch.float32, device=device)
        if bin_scale.numel() != n_tokens or not bool(torch.isfinite(bin_scale).all()) or bool((bin_scale <= 0).any()):
            raise ValueError("position_bin_scale must be finite and positive")

        if not bool(valid_mask.any()):
            zero = soft_tensor.sum() * 0.0
            return zero, self._zero_metrics()

        pred = {name: value[valid_mask].float() for name, value in predictions.items()}
        position = position[valid_mask]
        energy = energy[valid_mask]
        peak = peak[valid_mask]
        channel = channel[valid_mask]
        bins = bins[valid_mask]
        views = views[valid_mask]
        bin_scale = bin_scale[valid_mask]

        means = self.target_means.to(device=device)
        stds = self.target_stds.to(device=device)
        target_std = {
            "position": (position - means[0]) / stds[0],
            "energy": (energy - means[1]) / stds[1],
            "peak": (peak - means[2]) / stds[2],
        }
        errors = {
            name: (pred[name] - target_std[name]).square()
            for name in ("position", "energy", "peak")
        }
        component_weights = {
            "position": self.position_weight if position_weight is None else float(position_weight),
            "energy": self.energy_weight if energy_weight is None else float(energy_weight),
            "peak": self.peak_weight if peak_weight is None else float(peak_weight),
            "rank": self.rank_weight if rank_weight is None else float(rank_weight),
        }
        if any(value < 0.0 for value in component_weights.values()):
            raise ValueError("token supervision weights must be non-negative")
        component_losses = {
            name: self._nested_loss(errors[name], views, channel, bins)
            for name in ("position", "energy", "peak")
        }
        total = sum(
            (component_losses[name] * component_weights[name] for name in component_losses),
            component_losses["position"] * 0.0,
        )

        channel_base = max(1, int(channel.max().item()) + 1)
        bin_base = max(1, int(bins.max().item()) + 1)
        group_key = (views * channel_base + channel) * bin_base + bins
        unique, token_group = torch.unique(group_key, sorted=True, return_inverse=True)
        n_groups = int(unique.numel())
        group_view = unique // (channel_base * bin_base)
        group_pred = {
            name: self._group_mean(pred[name], token_group, n_groups)
            for name in ("position", "energy", "peak")
        }
        group_target = {
            "position": self._group_mean(position, token_group, n_groups),
            "energy": self._group_mean(energy, token_group, n_groups),
            "peak": self._group_mean(peak, token_group, n_groups),
        }
        group_scale = self._group_mean(bin_scale, token_group, n_groups)
        raw_pred = {
            name: group_pred[name] * stds[index] + means[index]
            for index, name in enumerate(("position", "energy", "peak"))
        }
        rank_loss = self._rank_loss(
            group_pred["energy"], group_target["energy"], group_view
        )
        if component_weights["rank"] != 0.0:
            total = total + component_weights["rank"] * rank_loss
        position_mae_bins = float(
            ((raw_pred["position"] - group_target["position"]).abs() * group_scale).mean().detach()
        )
        energy_pearson = self._pearson(raw_pred["energy"], group_target["energy"])
        peak_pearson = self._pearson(raw_pred["peak"], group_target["peak"])
        position_pearson = self._pearson(raw_pred["position"], group_target["position"])

        top3: list[Tensor] = []
        for view_id in torch.unique(group_view, sorted=True).tolist():
            mask = group_view == int(view_id)
            count = int(mask.sum().item())
            if count == 0:
                continue
            k = min(3, count)
            target_top = torch.argsort(group_target["energy"][mask], descending=True, stable=True)[:k]
            prediction_top = torch.argsort(raw_pred["energy"][mask], descending=True, stable=True)[:k]
            top3.append((target_top[:, None] == prediction_top[None, :]).any(dim=1).float().mean())
        top3_recall = float(torch.stack(top3).mean().detach()) if top3 else 0.0
        metrics = {
            "position_loss": float(component_losses["position"].detach()),
            "energy_loss": float(component_losses["energy"].detach()),
            "peak_loss": float(component_losses["peak"].detach()),
            "rank_loss": float(rank_loss.detach()),
            "position_pearson": position_pearson,
            "position_mae_bins": position_mae_bins,
            "energy_pearson": energy_pearson,
            "peak_pearson": peak_pearson,
            "top3_recall": top3_recall,
            "energy_balanced_accuracy": 0.0,
        }
        return total, metrics


def _view_tensor(tokens: ViewTokens, value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must have shape [C, L]")
    view_length = int(tokens.end - tokens.start)
    if int(value.shape[1]) == view_length:
        return value
    if int(value.shape[1]) >= int(tokens.end):
        return value[:, int(tokens.start):int(tokens.end)]
    raise ValueError(f"{name} does not cover the token view")


def _trace_for_view(
    tokens: ViewTokens,
    normalized_trace: Tensor,
    *,
    channel_indices: Sequence[int] | None,
) -> Tensor:
    trace = _view_tensor(tokens, normalized_trace, name="normalized_trace")
    if channel_indices is None:
        return trace
    selected = tuple(int(index) for index in channel_indices)
    if not selected:
        raise ValueError("channel_indices must not be empty")
    if int(trace.shape[0]) == len(selected):
        return trace
    if min(selected, default=0) < 0 or max(selected, default=-1) >= int(trace.shape[0]):
        raise ValueError("channel_indices do not match normalized_trace")
    index = torch.as_tensor(selected, dtype=torch.int64, device=trace.device)
    return trace.index_select(0, index)


def _valid_for_view(
    tokens: ViewTokens,
    valid_trace: Tensor | None,
    trace: Tensor,
    *,
    channel_indices: Sequence[int] | None,
) -> Tensor:
    if valid_trace is None:
        return torch.isfinite(trace)
    valid = _view_tensor(tokens, valid_trace, name="valid_trace").to(dtype=torch.bool, device=trace.device)
    if channel_indices is not None and int(valid.shape[0]) != len(tuple(channel_indices)):
        selected = tuple(int(index) for index in channel_indices)
        if min(selected, default=0) < 0 or max(selected, default=-1) >= int(valid.shape[0]):
            raise ValueError("channel_indices do not match valid_trace")
        index = torch.as_tensor(selected, dtype=torch.int64, device=valid.device)
        valid = valid.index_select(0, index)
    if valid.shape != trace.shape:
        raise ValueError("valid_trace must align with normalized_trace")
    return valid


def token_bin_indices(tokens: ViewTokens, n_bins: int, *, ordered: bool = False) -> Tensor:
    """Return token bins on the declared ceil-width reading grid."""

    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    length = int(tokens.end - tokens.start)
    if length < 1:
        raise ValueError("token view must have positive length")
    width = max(1, math.ceil(length / n_bins))
    centres = (tokens.support_start.to(torch.float64) + tokens.support_end.to(torch.float64)) * 0.5
    bins = torch.floor((centres - float(tokens.start)) / float(width)).to(torch.int64)
    bins = bins.clamp(0, n_bins - 1)
    return bins[tokens.sort_order] if ordered else bins


def reading_bin_width(tokens: ViewTokens, n_bins: int) -> int:
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    return max(1, math.ceil((int(tokens.end) - int(tokens.start)) / n_bins))


def _peak_scores(trace: Tensor, valid: Tensor, n_bins: int) -> Tensor:
    channels, length = map(int, trace.shape)
    width = max(1, math.ceil(length / int(n_bins)))
    sample_bins = (torch.arange(length, dtype=torch.int64, device=trace.device) // width).clamp(0, n_bins - 1)
    scores = torch.zeros((channels, n_bins), dtype=torch.float64, device=trace.device)
    for channel in range(channels):
        observed = trace[channel][valid[channel]]
        if observed.numel() == 0:
            continue
        median = observed.median()
        mad = (observed - median).abs().median()
        scale = torch.maximum(
            mad * MAD_TO_SIGMA,
            torch.tensor(PEAK_SCALE_FLOOR, dtype=observed.dtype, device=observed.device),
        )
        deviation = torch.where(
            valid[channel],
            (trace[channel] - median).abs() / scale,
            torch.zeros_like(trace[channel]),
        ).to(torch.float64)
        scores[channel].scatter_reduce_(0, sample_bins, deviation, reduce="amax", include_self=True)
    return scores


def token_supervision_targets(
    tokens: ViewTokens,
    normalized_trace: Tensor,
    recording_length: int | None = None,
    *,
    n_bins: int = DEFAULT_N_BINS,
    valid_trace: Tensor | None = None,
    valid: Tensor | None = None,
    channel_indices: Sequence[int] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build raw ``(position, energy, peak, valid)`` targets in sorted-slot order."""

    if not isinstance(tokens, ViewTokens):
        raise TypeError("tokens must be a ViewTokens instance")
    if recording_length is None:
        recording_length = int(tokens.end)
    recording_length = int(recording_length)
    if recording_length < 1:
        raise ValueError("recording_length must be positive")
    n_bins = int(n_bins)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if valid is not None:
        if valid_trace is not None:
            raise ValueError("pass only one of valid_trace and valid")
        valid_trace = valid

    trace = _trace_for_view(tokens, normalized_trace, channel_indices=channel_indices)
    valid_trace_aligned = _valid_for_view(
        tokens, valid_trace, trace, channel_indices=channel_indices
    )
    if not bool(torch.isfinite(trace[valid_trace_aligned]).all()):
        raise ValueError("normalized_trace must be finite at valid samples")
    filled = torch.where(valid_trace_aligned, trace, torch.zeros_like(trace)).to(torch.float32)
    device = filled.device

    bins = token_bin_indices(tokens, n_bins).to(device=device)
    width = reading_bin_width(tokens, n_bins)
    bin_centres = float(tokens.start) + (bins.to(torch.float64) + 0.5) * float(width)
    position = (bin_centres / float(recording_length)).to(torch.float32)
    channels = tokens.channel_index.to(device=device, dtype=torch.int64)
    if bool((channels < 0).any()):
        raise ValueError("token channel indices must be non-negative")
    if channel_indices is not None:
        selected = tuple(int(index) for index in channel_indices)
        max_channel = max(
            int(channels.max().item()) if channels.numel() else -1,
            max(selected, default=-1),
        )
        local_channels = torch.full(
            (max_channel + 1,), -1, dtype=torch.int64, device=device
        )
        selected_tensor = torch.as_tensor(selected, dtype=torch.int64, device=device)
        local_channels[selected_tensor] = torch.arange(len(selected), dtype=torch.int64, device=device)
        if channels.numel() and int(channels.max()) >= int(local_channels.numel()):
            raise ValueError("token channels do not match channel_indices")
        channels = local_channels[channels]
        if bool((channels < 0).any()):
            raise ValueError("token channels do not belong to the view")
    if channels.numel() and int(channels.max()) >= int(trace.shape[0]):
        raise ValueError("token channels do not match normalized_trace")

    with torch.no_grad():
        energy_scores = _normalised_energy_scores(filled, n_bins)
        peak_scores = _peak_scores(filled, valid_trace_aligned, n_bins)
    energy_by_token = torch.log1p(energy_scores[channels, bins]).to(torch.float32)
    peak_by_token = torch.log1p(peak_scores[channels, bins]).to(torch.float32)
    valid = tokens.valid.to(device=device, dtype=torch.bool)
    order = tokens.sort_order.to(device=device, dtype=torch.int64)
    return position[order], energy_by_token[order], peak_by_token[order], valid[order]


__all__ = [
    "TokenSupervisionHeads",
    "load_token_supervision_stats",
    "reading_bin_width",
    "token_bin_indices",
    "token_supervision_targets",
]
