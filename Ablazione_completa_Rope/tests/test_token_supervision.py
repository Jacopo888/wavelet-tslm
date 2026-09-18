"""CPU-only tests for the definitive P1-e token supervision contract."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from wavelet_rope.signal_contract import ViewKind
from wavelet_rope.token_supervision import TokenSupervisionHeads, token_supervision_targets
from wavelet_rope.tokenizer import ViewTokens


def _tokens() -> ViewTokens:
    n = 4
    return ViewTokens(
        descriptors=torch.zeros(n, 512),
        valid=torch.ones(n, dtype=torch.bool),
        anchor_time=torch.arange(n, dtype=torch.float64),
        support_start=torch.tensor([0, 16, 40, 56], dtype=torch.int64),
        support_end=torch.tensor([4, 20, 44, 60], dtype=torch.int64),
        channel_index=torch.zeros(n, dtype=torch.int64),
        family=torch.zeros(n, dtype=torch.int64),
        level=torch.zeros(n, dtype=torch.int64),
        view_kind=ViewKind.GLOBAL,
        start=0,
        end=64,
        raw_bin_features=torch.zeros(1, 4, 64),
        raw_bin_valid=torch.ones(1, 4, dtype=torch.bool),
        raw_bin_edges=torch.tensor([0, 16, 32, 48, 64], dtype=torch.int64),
        sort_order=torch.arange(n, dtype=torch.int64),
        coefficient_index=torch.arange(n, dtype=torch.int64),
    )


def _stats() -> dict:
    return {
        "position": {"mean": 0.5, "std": 0.25},
        "energy": {"mean": 0.1, "std": 0.7},
        "peak": {"mean": 0.2, "std": 1.1},
        "rows": [0, 1],
        "n_bins": 16,
    }


def _spec(**extra: object) -> dict:
    spec = {
        "enabled": True,
        "inputs": ["soft"],
        "hidden": 0,
        "n_bins": 16,
        "position_weight": 1.0,
        "energy_weight": 1.0,
        "peak_weight": 0.5,
    }
    spec.update(extra)
    return spec


def _rank_case(
    target_energy: list[float],
    predicted_energy: list[float],
    *,
    rank_weight: float = 1.0,
) -> tuple[TokenSupervisionHeads, torch.Tensor, torch.Tensor, dict[str, float]]:
    if len(target_energy) != len(predicted_energy):
        raise ValueError("the ranking case must have equal target and prediction lengths")
    heads = TokenSupervisionHeads(
        _spec(position_weight=0.0, energy_weight=0.0, peak_weight=0.0, rank_weight=rank_weight),
        soft_dim=1,
        stats=_stats(),
    )
    with torch.no_grad():
        heads.position_head.weight.zero_()
        heads.position_head.bias.zero_()
        heads.energy_peak_head.weight.zero_()
        heads.energy_peak_head.bias.zero_()
        heads.energy_peak_head.weight[0, 0] = 1.0
    soft = torch.tensor(predicted_energy, dtype=torch.float32).reshape(-1, 1).requires_grad_()
    target = torch.tensor(target_energy, dtype=torch.float32)
    n_tokens = len(target_energy)
    loss, metrics = heads.loss(
        soft,
        torch.zeros(n_tokens),
        target,
        torch.zeros(n_tokens),
        torch.ones(n_tokens, dtype=torch.bool),
        channel_index=torch.zeros(n_tokens, dtype=torch.int64),
        bin_index=torch.arange(n_tokens, dtype=torch.int64),
        view_index=torch.zeros(n_tokens, dtype=torch.int64),
        position_bin_scale=torch.ones(n_tokens),
    )
    return heads, soft, loss, metrics


def _legacy_mse_loss(
    heads: TokenSupervisionHeads,
    soft: torch.Tensor,
    position: torch.Tensor,
    energy: torch.Tensor,
    peak: torch.Tensor,
    channel: torch.Tensor,
    bins: torch.Tensor,
    views: torch.Tensor,
) -> torch.Tensor:
    predictions = {name: value.float() for name, value in heads.forward(soft).items()}
    means = heads.target_means.to(device=soft.device)
    stds = heads.target_stds.to(device=soft.device)
    targets = {
        "position": (position - means[0]) / stds[0],
        "energy": (energy - means[1]) / stds[1],
        "peak": (peak - means[2]) / stds[2],
    }
    errors = {name: (predictions[name] - targets[name]).square() for name in targets}
    return sum(
        (
            heads._nested_loss(errors[name], views, channel, bins) * weight
            for name, weight in (
                ("position", heads.position_weight),
                ("energy", heads.energy_weight),
                ("peak", heads.peak_weight),
            )
        ),
        errors["position"].sum() * 0.0,
    )


def _targets() -> tuple[torch.Tensor, ...]:
    trace = torch.zeros(1, 64)
    trace[0, 42] = 10.0
    return token_supervision_targets(
        _tokens(), trace, recording_length=64, n_bins=16,
        valid_trace=torch.ones_like(trace, dtype=torch.bool),
    )


def test_targets_use_declared_bins_and_detect_known_peak() -> None:
    position, energy, peak, valid = _targets()

    assert torch.equal(valid, torch.ones(4, dtype=torch.bool))
    assert torch.all(position[1:] > position[:-1])
    assert torch.allclose(position, torch.tensor([2.0, 18.0, 42.0, 58.0]) / 64.0)
    assert int(energy.argmax()) == 2
    assert int(peak.argmax()) == 2
    assert float(energy[2]) > float(energy[[0, 1, 3]].max())
    assert float(peak[2]) > float(peak[[0, 1, 3]].max())


def test_targets_remap_absolute_channels_in_a_local_view() -> None:
    tokens = copy.deepcopy(_tokens())
    tokens.channel_index = torch.ones_like(tokens.channel_index)
    trace = torch.zeros(1, 64)
    trace[0, 42] = 10.0
    _, energy, peak, valid = token_supervision_targets(
        tokens,
        trace,
        recording_length=64,
        n_bins=16,
        valid_trace=torch.ones_like(trace, dtype=torch.bool),
        channel_indices=(1,),
    )

    assert bool(valid.all())
    assert int(energy.argmax()) == 2
    assert int(peak.argmax()) == 2


def test_heads_are_two_linear_heads_on_soft_only() -> None:
    heads = TokenSupervisionHeads(_spec(), soft_dim=6, target_rms=0.013, stats=_stats())

    assert isinstance(heads.position_head, nn.Linear)
    assert heads.position_head.out_features == 1
    assert isinstance(heads.energy_peak_head, nn.Linear)
    assert heads.energy_peak_head.out_features == 2
    assert heads.rank_weight == pytest.approx(0.2)
    assert not any(isinstance(module, (nn.Sequential, nn.GELU)) for module in heads.modules())
    assert "target_means" in heads.state_dict() and "target_stds" in heads.state_dict()
    assert all("target_" not in name for name, _ in heads.named_parameters())
    with pytest.raises(ValueError, match="Z supervision"):
        heads({"soft": torch.zeros(2, 6), "z": torch.zeros(2, 5)})


def test_nested_reduction_is_invariant_to_duplicate_tokens_in_a_bin() -> None:
    torch.manual_seed(11)
    heads = TokenSupervisionHeads(_spec(), soft_dim=6, target_rms=0.013, stats=_stats())
    soft = torch.randn(4, 6)
    position, energy, peak, valid = _targets()
    channel = torch.zeros(4, dtype=torch.int64)
    bins = torch.tensor([0, 4, 10, 14], dtype=torch.int64)
    views = torch.zeros(4, dtype=torch.int64)
    scale = torch.ones(4)
    first, _ = heads.loss(soft, position, energy, peak, valid, channel_index=channel, bin_index=bins,
                          view_index=views, position_bin_scale=scale)

    duplicate = slice(2, 3)
    second, _ = heads.loss(
        torch.cat((soft, soft[duplicate]), dim=0),
        torch.cat((position, position[duplicate]), dim=0),
        torch.cat((energy, energy[duplicate]), dim=0),
        torch.cat((peak, peak[duplicate]), dim=0),
        torch.cat((valid, valid[duplicate]), dim=0),
        channel_index=torch.cat((channel, channel[duplicate]), dim=0),
        bin_index=torch.cat((bins, bins[duplicate]), dim=0),
        view_index=torch.cat((views, views[duplicate]), dim=0),
        position_bin_scale=torch.cat((scale, scale[duplicate]), dim=0),
    )
    assert torch.allclose(first, second)


def test_loss_is_finite_and_backpropagates_to_soft_and_heads() -> None:
    torch.manual_seed(19)
    heads = TokenSupervisionHeads(_spec(), soft_dim=6, target_rms=0.013, stats=_stats())
    soft = torch.randn(4, 6, requires_grad=True)
    position, energy, peak, valid = _targets()
    loss, metrics = heads.loss(
        soft,
        position,
        energy,
        peak,
        valid,
        channel_index=torch.zeros(4, dtype=torch.int64),
        bin_index=torch.tensor([0, 4, 10, 14]),
        view_index=torch.zeros(4, dtype=torch.int64),
        position_bin_scale=torch.ones(4),
    )

    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert {"position_loss", "energy_loss", "peak_loss", "position_mae_bins", "energy_pearson",
            "peak_pearson", "top3_recall"}.issubset(metrics)
    loss.backward()
    assert soft.grad is not None and torch.isfinite(soft.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in heads.parameters() if parameter.requires_grad)


def test_disabled_supervision_preserves_parameters_and_base_loss() -> None:
    torch.manual_seed(3)
    base = nn.Linear(5, 1)

    def optional(enabled: bool) -> nn.Module:
        module = nn.Module()
        module.base = copy.deepcopy(base)
        if enabled:
            module.token_supervision = TokenSupervisionHeads(_spec(), soft_dim=6, stats=_stats())
        return module

    disabled = optional(False)
    enabled = optional(True)
    disabled_loss = disabled.base(torch.ones(4, 5)).square().mean()
    base_loss = base(torch.ones(4, 5)).square().mean()

    assert sum(parameter.numel() for parameter in disabled.parameters()) == sum(
        parameter.numel() for parameter in base.parameters()
    )
    assert sum(parameter.numel() for parameter in enabled.parameters()) > sum(
        parameter.numel() for parameter in base.parameters()
    )
    assert torch.equal(disabled_loss, base_loss)


def test_missing_stats_path_fails_closed() -> None:
    with pytest.raises(FileNotFoundError, match="stats_path"):
        TokenSupervisionHeads(_spec(stats_path="does-not-exist/token_stats.json"), soft_dim=6)


def test_rank_loss_is_small_for_correct_top3_order_and_grows_when_a_competitor_wins() -> None:
    heads, _, correct_loss, correct_metrics = _rank_case(
        [3.0, 2.0, 1.0, 0.0, -1.0],
        [10.0, 9.0, 8.0, -10.0, -11.0],
    )
    assert heads.rank_weight == pytest.approx(1.0)
    assert correct_metrics["rank_loss"] < 1e-4
    assert float(correct_loss.detach()) < 1e-4

    _, _, wrong_loss, wrong_metrics = _rank_case(
        [3.0, 2.0, 1.0, 0.0, -1.0],
        [-10.0, -9.0, -8.0, 10.0, -11.0],
    )
    assert wrong_metrics["rank_loss"] > correct_metrics["rank_loss"] + 1.0
    assert float(wrong_loss.detach()) > float(correct_loss.detach()) + 1.0


def test_rank_loss_excludes_equal_teacher_energies() -> None:
    _, _, _, metrics = _rank_case(
        [3.0, 2.0, 1.0, 1.0],
        [200.0, 199.0, -100.0, 100.0],
    )

    assert metrics["rank_loss"] < 1e-6


def test_rank_weight_zero_matches_the_previous_mse_total_and_gradients() -> None:
    torch.manual_seed(29)
    spec = _spec(rank_weight=0.0)
    heads = TokenSupervisionHeads(spec, soft_dim=3, stats=_stats())
    legacy_heads = copy.deepcopy(heads)
    soft_value = torch.randn(6, 3)
    position = torch.linspace(0.0, 1.0, 6)
    energy = torch.tensor([0.0, 0.4, 0.8, 1.2, 0.2, 0.6])
    peak = torch.linspace(1.0, 0.0, 6)
    channel = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.int64)
    bins = torch.tensor([0, 1, 0, 1, 2, 2], dtype=torch.int64)
    views = torch.zeros(6, dtype=torch.int64)

    soft = soft_value.clone().requires_grad_()
    loss, _ = heads.loss(
        soft,
        position,
        energy,
        peak,
        channel_index=channel,
        bin_index=bins,
        view_index=views,
    )
    legacy_soft = soft_value.clone().requires_grad_()
    legacy_loss = _legacy_mse_loss(
        legacy_heads, legacy_soft, position, energy, peak, channel, bins, views
    )
    assert torch.equal(loss, legacy_loss)

    loss.backward()
    legacy_loss.backward()
    assert torch.equal(soft.grad, legacy_soft.grad)
    for parameter, legacy_parameter in zip(heads.parameters(), legacy_heads.parameters()):
        assert torch.equal(parameter.grad, legacy_parameter.grad)


def test_rank_loss_skips_views_with_fewer_than_four_occupied_bins() -> None:
    _, soft, loss, metrics = _rank_case([2.0, 1.0], [10.0, -10.0])

    assert torch.isfinite(loss)
    assert metrics["rank_loss"] == 0.0
    loss.backward()
    assert soft.grad is not None and torch.isfinite(soft.grad).all()


def test_rank_loss_reaches_the_energy_head() -> None:
    heads, _, loss, metrics = _rank_case(
        [3.0, 2.0, 1.0, 0.0],
        [-10.0, -9.0, -8.0, 10.0],
    )

    assert metrics["rank_loss"] > 0.0
    loss.backward()
    gradient = heads.energy_peak_head.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert float(gradient[0].abs().sum()) > 0.0
