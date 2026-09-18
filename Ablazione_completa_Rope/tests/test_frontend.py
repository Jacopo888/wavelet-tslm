"""Contract tests for the perceptual front-end (deep branches + tokenizer)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from wavelet_rope.deep_branches import DeepBranches, ResidualTemporalBlock
from wavelet_rope.pyramid import build_haar_pyramid
from wavelet_rope.signal_contract import (
    SignalEpisode,
    ViewKind,
    compute_recording_normalization,
    make_view,
)
from wavelet_rope.tokenizer import (
    FAMILY_A,
    FAMILY_D_COVERAGE,
    FAMILY_D_SALIENT,
    FAMILY_RAW,
    N_FAMILIES,
    PerceptualFrontend,
    ViewTokens,
    robust_band_scale,
    select_salient_coefficients,
    uniform_bin_edges,
)


ROOT = Path(__file__).resolve().parents[1]
DT = 0.5
ORIGIN = 100.0


@pytest.fixture(scope="module")
def spec() -> dict:
    with open(ROOT / "prototype.spec.json", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture()
def frontend(spec: dict) -> PerceptualFrontend:
    torch.manual_seed(0)
    return PerceptualFrontend(spec)


def make_episode(
    channels: int = 2,
    length: int = 1024,
    *,
    seed: int = 1,
    hole: tuple[int, int, int] | None = None,
    spikes: tuple[int, ...] = (),
    values: torch.Tensor | None = None,
) -> SignalEpisode:
    generator = torch.Generator().manual_seed(seed)
    if values is None:
        t = torch.arange(length, dtype=torch.float32)
        base = torch.stack(
            [torch.sin(t / (37.0 + 11.0 * c)) + 0.05 * c for c in range(channels)]
        )
        values = base + 0.1 * torch.randn(channels, length, generator=generator)
        for position in spikes:
            values[:, position : position + 3] += 6.0
    valid = torch.ones(channels, length, dtype=torch.bool)
    if hole is not None:
        channel, lo, hi = hole
        valid[channel, lo:hi] = False
    ids = tuple(f"ch{c}" for c in range(channels))
    return SignalEpisode.from_index_grid(
        values=values.to(torch.float32), valid=valid, channel_ids=ids, dt=DT, origin=ORIGIN
    )


def encode(frontend: PerceptualFrontend, episode: SignalEpisode, kind: ViewKind, start=0, end=None):
    stats = compute_recording_normalization(episode)
    view = make_view(episode, stats, kind=kind, start=start, end=end)
    return view, frontend.encode_view(view, recording_length=episode.length)


def _assert_clean(tokens: ViewTokens) -> None:
    assert torch.isfinite(tokens.descriptors).all()
    assert torch.isfinite(tokens.raw_bin_features).all()
    invalid = ~tokens.valid
    assert (tokens.descriptors[invalid] == 0).all()
    assert (tokens.anchor_time[invalid] == 0).all()
    assert (tokens.support_start[invalid] == 0).all() and (tokens.support_end[invalid] == 0).all()
    assert (tokens.raw_bin_features[~tokens.raw_bin_valid] == 0).all()


# --------------------------------------------------------------------------- #
# 1. Shapes and masks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("length", [300, 1024, 4096])
def test_shapes_and_masks_global(frontend: PerceptualFrontend, length: int) -> None:
    episode = make_episode(2, length)
    view, tokens = encode(frontend, episode, ViewKind.GLOBAL)
    quota = frontend.tokenizer.quota(ViewKind.GLOBAL)
    n_tokens = 2 * N_FAMILIES * quota
    assert tokens.descriptors.shape == (n_tokens, 512)
    assert tokens.valid.shape == (n_tokens,) and tokens.valid.dtype == torch.bool
    assert tokens.anchor_time.dtype == torch.float64
    assert tokens.support_start.dtype == tokens.support_end.dtype == torch.int64
    assert tokens.family.unique().tolist() == [FAMILY_A, FAMILY_D_COVERAGE, FAMILY_D_SALIENT, FAMILY_RAW]
    assert 0 < tokens.n_valid <= n_tokens
    assert tokens.raw_bin_features.shape == (2, quota, 64)
    assert tokens.raw_bin_valid.shape == (2, quota)
    assert tokens.raw_bin_edges.shape == (quota + 1,)
    assert int(tokens.raw_bin_edges[0]) == 0 and int(tokens.raw_bin_edges[-1]) == length
    assert tokens.sort_order.shape == (n_tokens,)
    assert torch.equal(tokens.sort_order.sort().values, torch.arange(n_tokens))
    assert tokens.channel_index.unique().tolist() == [0, 1]
    # A tokens carry J, salient tokens carry 1..J, uniform D summaries and raw carry 0.
    levels_a = tokens.level[(tokens.family == FAMILY_A) & tokens.valid].unique().tolist()
    assert levels_a == [build_haar_pyramid(view).levels]
    assert (tokens.level[tokens.family == FAMILY_RAW] == 0).all()
    _assert_clean(tokens)
    # valid tokens come first in sort order, invalid ones last
    ordered_valid = tokens.valid[tokens.sort_order]
    n_valid = tokens.n_valid
    assert ordered_valid[:n_valid].all() and not ordered_valid[n_valid:].any()
    ordered_times = tokens.anchor_time[tokens.sort_order][:n_valid]
    assert (ordered_times[1:] >= ordered_times[:-1]).all()
    # every uniform-bin family fills at most one token per bin per channel
    for family in (FAMILY_A, FAMILY_D_COVERAGE, FAMILY_RAW):
        mask = (tokens.family == family) & tokens.valid
        keys = tokens.channel_index[mask] * quota + tokens.coefficient_index[mask]
        assert keys.unique().numel() == keys.numel()


def test_shapes_local_quota(frontend: PerceptualFrontend) -> None:
    episode = make_episode(2, 1024)
    _, tokens = encode(frontend, episode, ViewKind.LOCAL, start=256, end=768)
    quota = frontend.tokenizer.quota(ViewKind.LOCAL)
    assert quota == 128
    assert tokens.descriptors.shape == (2 * N_FAMILIES * quota, 512)
    assert tokens.view_kind is ViewKind.LOCAL and (tokens.start, tokens.end) == (256, 768)
    assert tokens.raw_bin_features.shape == (2, quota, 64)
    _assert_clean(tokens)


# --------------------------------------------------------------------------- #
# 2. Geometry
# --------------------------------------------------------------------------- #


def test_geometry_local_view_absolute_coordinates(frontend: PerceptualFrontend) -> None:
    length = 1024
    episode = make_episode(2, length, spikes=(300, 331, 517, 700))
    start, end = 256, 768  # aligned to 2**4, so Haar supports coincide with the global ones
    local_view, local_tokens = encode(frontend, episode, ViewKind.LOCAL, start=start, end=end)
    global_view, global_tokens = encode(frontend, episode, ViewKind.GLOBAL)
    valid = local_tokens.valid
    assert (local_tokens.support_start[valid] >= start).all()
    assert (local_tokens.support_end[valid] <= end).all()
    assert (local_tokens.support_end[valid] > local_tokens.support_start[valid]).all()
    t_lo = float(episode.sample_times[start])
    t_hi = float(episode.sample_times[end - 1])
    assert (local_tokens.anchor_time[valid] >= t_lo).all()
    assert (local_tokens.anchor_time[valid] <= t_hi).all()
    # a sample keeps its absolute time across views: uniform bins of the local view
    # must sit at the exact recording clock of their centre samples
    uniform = valid & (local_tokens.family == FAMILY_RAW)
    lo = local_tokens.support_start[uniform]
    hi = local_tokens.support_end[uniform]
    expected = (episode.sample_times[lo] + episode.sample_times[hi - 1]) * 0.5
    assert torch.equal(local_tokens.anchor_time[uniform], expected)

    global_pyramid = build_haar_pyramid(global_view, trace="global")
    local_pyramid = build_haar_pyramid(local_view, trace="global")
    salient = valid & (local_tokens.family == FAMILY_D_SALIENT)
    assert salient.any()
    for slot in salient.nonzero().squeeze(1).tolist():
        level = int(local_tokens.level[slot])
        support_start = int(local_tokens.support_start[slot])
        channel = int(local_tokens.channel_index[slot])
        k_local = (support_start - start) >> level
        k_global = support_start >> level
        assert int(local_tokens.coefficient_index[slot]) == k_local
        band_local = local_pyramid.detail(level)
        band_global = global_pyramid.detail(level)
        assert int(band_local.support_start[k_local]) == support_start
        assert int(band_global.support_start[k_global]) == support_start
        assert float(local_tokens.anchor_time[slot]) == float(band_local.anchor_time[channel, k_local])
        assert float(local_tokens.anchor_time[slot]) == float(band_global.anchor_time[channel, k_global])

    # the same coefficient selected in both views shares its absolute anchor time
    def salient_map(tokens: ViewTokens) -> dict[tuple[int, int, int], float]:
        mask = tokens.valid & (tokens.family == FAMILY_D_SALIENT)
        out = {}
        for slot in mask.nonzero().squeeze(1).tolist():
            key = (int(tokens.channel_index[slot]), int(tokens.level[slot]), int(tokens.support_start[slot]))
            out[key] = float(tokens.anchor_time[slot])
        return out

    local_map = salient_map(local_tokens)
    global_map = salient_map(global_tokens)
    shared = set(local_map) & set(global_map)
    assert shared, "the spikes inside the crop must be salient in both views"
    for key in shared:
        assert local_map[key] == global_map[key]


def test_geometry_odd_length_supports_inside_recording(frontend: PerceptualFrontend) -> None:
    episode = make_episode(2, 300)
    _, tokens = encode(frontend, episode, ViewKind.GLOBAL)
    valid = tokens.valid
    assert (tokens.support_start[valid] >= 0).all() and (tokens.support_end[valid] <= 300).all()
    assert (tokens.anchor_time[valid] >= ORIGIN).all()
    assert (tokens.anchor_time[valid] <= ORIGIN + 299 * DT).all()


# --------------------------------------------------------------------------- #
# 3. Missing data
# --------------------------------------------------------------------------- #


def test_missing_data_hole(frontend: PerceptualFrontend) -> None:
    length = 1024
    lo, hi = 400, 520
    episode = make_episode(2, length, hole=(0, lo, hi), spikes=(450, 700))
    _, tokens = encode(frontend, episode, ViewKind.GLOBAL)
    _assert_clean(tokens)
    quota = frontend.tokenizer.quota(ViewKind.GLOBAL)
    ch0 = tokens.channel_index == 0
    # salient tokens never cover the hole (Haar usable rule)
    salient = tokens.valid & ch0 & (tokens.family == FAMILY_D_SALIENT)
    assert salient.any()
    intersects = (tokens.support_start[salient] < hi) & (tokens.support_end[salient] > lo)
    assert not intersects.any()
    # raw bins fully inside the hole are invalid slots; bins outside are valid
    edges = tokens.raw_bin_edges
    inside = (edges[:-1] >= lo) & (edges[1:] <= hi)
    outside = (edges[1:] <= lo) | (edges[:-1] >= hi)
    assert inside.any()
    assert not tokens.raw_bin_valid[0, inside].any()
    assert tokens.raw_bin_valid[0, outside].all()
    assert tokens.raw_bin_valid[1].all()
    for family in (FAMILY_A, FAMILY_D_COVERAGE, FAMILY_RAW):
        mask = ch0 & (tokens.family == family)
        bins = tokens.coefficient_index[mask]
        assert not tokens.valid[mask][inside[bins]].any()
        assert tokens.valid[mask][outside[bins]].all()
    # channel 1 is untouched: every uniform slot is valid
    assert tokens.valid[(tokens.channel_index == 1) & (tokens.family != FAMILY_D_SALIENT)].all()
    # the masked samples do not influence any token: changing them leaves everything equal
    altered = episode.values.clone()
    altered[0, lo:hi] = 1e4
    altered_episode = make_episode(2, length, hole=(0, lo, hi), values=altered)
    _, altered_tokens = encode(frontend, altered_episode, ViewKind.GLOBAL)
    assert torch.equal(tokens.valid, altered_tokens.valid)
    assert torch.equal(tokens.descriptors, altered_tokens.descriptors)
    assert torch.equal(tokens.anchor_time, altered_tokens.anchor_time)
    assert torch.equal(tokens.sort_order, altered_tokens.sort_order)


def test_local_view_with_empty_channel_is_fully_masked(frontend: PerceptualFrontend) -> None:
    episode = make_episode(2, 1024, hole=(1, 256, 768))
    _, tokens = encode(frontend, episode, ViewKind.LOCAL, start=256, end=768)
    _assert_clean(tokens)
    assert not tokens.valid[tokens.channel_index == 1].any()
    assert tokens.valid[(tokens.channel_index == 0) & (tokens.family == FAMILY_RAW)].all()


# --------------------------------------------------------------------------- #
# 4. Salient selection
# --------------------------------------------------------------------------- #


def test_select_salient_first_pass_and_refill() -> None:
    generator = torch.Generator().manual_seed(3)
    count, n_bins, quota = 500, 32, 64
    score = torch.rand(count, generator=generator)
    level = torch.randint(1, 5, (count,), generator=generator)
    index = torch.arange(count)
    bin_id = torch.randint(0, n_bins, (count,), generator=generator)
    chosen, n_first = select_salient_coefficients(score, level, index, bin_id, quota)
    assert chosen.numel() == quota
    assert chosen.unique().numel() == quota
    first = chosen[:n_first]
    assert n_first == min(quota, sum(min(2, int((bin_id == b).sum())) for b in range(n_bins)))
    assert torch.bincount(bin_id[first], minlength=n_bins).max() <= 2
    for b in range(n_bins):
        members = (bin_id == b).nonzero().squeeze(1)
        best = members[score[members].argsort(descending=True)][:2]
        assert set(best.tolist()) <= set(first.tolist())
    refill = chosen[n_first:]
    remaining_scores = score.clone()
    remaining_scores[first] = -1.0
    expected_refill = remaining_scores.argsort(descending=True)[: refill.numel()]
    assert torch.equal(refill, expected_refill)
    # fewer candidates than the quota: take them all, no duplicates
    chosen_small, _ = select_salient_coefficients(score[:10], level[:10], index[:10], bin_id[:10], quota)
    assert chosen_small.numel() == 10 and chosen_small.unique().numel() == 10
    # ties are broken by (level, index)
    tie_score = torch.ones(4)
    tie_level = torch.tensor([2, 1, 1, 3])
    tie_index = torch.tensor([0, 5, 2, 1])
    tie_bin = torch.zeros(4, dtype=torch.int64)
    tie_chosen, tie_first = select_salient_coefficients(tie_score, tie_level, tie_index, tie_bin, 4)
    assert tie_first == 2 and tie_chosen.tolist() == [2, 1, 0, 3]


@pytest.mark.parametrize("kind,start,end", [(ViewKind.GLOBAL, 0, None), (ViewKind.LOCAL, 128, 896)])
def test_salient_tokens_end_to_end(frontend: PerceptualFrontend, kind: ViewKind, start: int, end) -> None:
    episode = make_episode(2, 1024, hole=(0, 300, 340), spikes=(200, 600))
    view, tokens = encode(frontend, episode, kind, start=start, end=end)
    pyramid = build_haar_pyramid(view, trace="global")
    quota = frontend.tokenizer.quota(kind)
    n_bins = frontend.tokenizer.salience_bins(kind)
    sal_edges = uniform_bin_edges(view.start, view.end, n_bins, torch.device("cpu"))
    for channel in range(2):
        mask = tokens.valid & (tokens.channel_index == channel) & (tokens.family == FAMILY_D_SALIENT)
        usable = sum(int(band.usable[channel].sum()) for band in pyramid.details)
        assert int(mask.sum()) == min(quota, usable)
        keys = tokens.level[mask] * 10**6 + tokens.coefficient_index[mask]
        assert keys.unique().numel() == keys.numel()
        # every selected coefficient is usable and its token anchors on the band anchor
        for slot in mask.nonzero().squeeze(1).tolist():
            band = pyramid.detail(int(tokens.level[slot]))
            k = int(tokens.coefficient_index[slot])
            assert bool(band.usable[channel, k])
            assert float(tokens.anchor_time[slot]) == float(band.anchor_time[channel, k])
        # first pass guarantees coverage: each salience bin with >= 2 usable coefficients
        # receives at least two tokens
        centre = (tokens.support_start[mask] + tokens.support_end[mask] - 1).to(torch.float64) * 0.5
        token_bins = torch.searchsorted(sal_edges.to(torch.float64), centre, right=True) - 1
        got = torch.bincount(token_bins, minlength=n_bins)
        available = torch.zeros(n_bins, dtype=torch.int64)
        for band in pyramid.details:
            c = (band.support_start + band.support_end - 1).to(torch.float64) * 0.5
            b = torch.searchsorted(sal_edges.to(torch.float64), c, right=True) - 1
            available += torch.bincount(b[band.usable[channel]], minlength=n_bins)
        assert (got >= torch.minimum(available, torch.full_like(available, 2))).all()


# --------------------------------------------------------------------------- #
# 5. Gradients
# --------------------------------------------------------------------------- #


def _grad_state(module: torch.nn.Module) -> dict[str, bool]:
    return {
        name: parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for name, parameter in module.named_parameters()
    }


def test_gradients_reach_every_trunk_and_adapters_only_locally(frontend: PerceptualFrontend) -> None:
    episode = make_episode(2, 1024, spikes=(200, 600))
    for kind, start, end in ((ViewKind.GLOBAL, 0, None), (ViewKind.LOCAL, 256, 768)):
        frontend.zero_grad(set_to_none=True)
        _, tokens = encode(frontend, episode, kind, start=start, end=end)
        loss = tokens.descriptors[tokens.valid].sum()
        loss.backward()
        state = _grad_state(frontend)
        trunks = {
            "A": [n for n in state if n.startswith(("branches.a_stem", "branches.a_blocks"))],
            "D": [n for n in state if n.startswith(("branches.d_stem", "branches.d_blocks", "branches.level_embedding"))],
            "raw": [n for n in state if n.startswith(("branches.raw_stem", "branches.raw_blocks"))],
            "fusion": [n for n in state if n.startswith(("tokenizer.fuse_in", "tokenizer.fuse_out"))],
            "pool": [n for n in state if n.startswith(("tokenizer.a_pool", "tokenizer.d_pool", "tokenizer.raw_pool"))],
            "embeddings": [n for n in state if n.startswith(("tokenizer.family_embedding", "tokenizer.channel_embedding", "tokenizer.meta_proj"))],
        }
        for group, names in trunks.items():
            assert names, group
            assert all(state[n] for n in names), (kind, group, [n for n in names if not state[n]])
        adapters = [n for n in state if "adapter" in n]
        assert adapters
        if kind is ViewKind.LOCAL:
            assert all(state[n] for n in adapters), [n for n in adapters if not state[n]]
        else:
            assert not any(state[n] for n in adapters)
        # the auxiliary head sees gradients through the raw summaries
        frontend.zero_grad(set_to_none=True)
        _, tokens = encode(frontend, episode, kind, start=start, end=end)
        aux = frontend.aux_logits(tokens)
        assert aux.shape == tokens.raw_bin_valid.shape
        aux[tokens.raw_bin_valid].sum().backward()
        state = _grad_state(frontend)
        assert state["aux_head.weight"]
        assert all(state[n] for n in trunks["raw"])


def test_residual_block_masks_and_film() -> None:
    torch.manual_seed(0)
    block = ResidualTemporalBlock(dilation=2)
    h = torch.randn(3, 40, 64)
    valid = torch.ones(3, 40, dtype=torch.bool)
    valid[1, 5:15] = False
    out = block(h, valid)
    assert out.shape == h.shape and torch.isfinite(out).all()
    assert (out[1, 5:15] == 0).all()
    gamma = torch.full((3, 1, 64), 0.5)
    beta = torch.full((3, 1, 64), -0.25)
    film_out = block(h, valid, film=(gamma, beta))
    assert not torch.equal(film_out, out)
    assert (film_out[1, 5:15] == 0).all()
    # a valid position far from the hole is unaffected by the hole content
    h2 = h.clone()
    h2[1, 5:15] = 100.0
    assert torch.allclose(block(h2, valid), out)


# --------------------------------------------------------------------------- #
# 6. Determinism
# --------------------------------------------------------------------------- #


def test_determinism(frontend: PerceptualFrontend) -> None:
    episode = make_episode(2, 1024, hole=(1, 100, 140), spikes=(200, 600))
    for kind, start, end in ((ViewKind.GLOBAL, 0, None), (ViewKind.LOCAL, 256, 768)):
        _, first = encode(frontend, episode, kind, start=start, end=end)
        _, second = encode(frontend, episode, kind, start=start, end=end)
        assert torch.equal(first.descriptors, second.descriptors)
        assert torch.equal(first.valid, second.valid)
        assert torch.equal(first.anchor_time, second.anchor_time)
        assert torch.equal(first.support_start, second.support_start)
        assert torch.equal(first.support_end, second.support_end)
        assert torch.equal(first.level, second.level)
        assert torch.equal(first.sort_order, second.sort_order)
        assert torch.equal(first.raw_bin_features, second.raw_bin_features)


# --------------------------------------------------------------------------- #
# 7. Content sensitivity
# --------------------------------------------------------------------------- #


def test_content_sensitivity(frontend: PerceptualFrontend) -> None:
    base = make_episode(2, 1024, hole=(0, 400, 520), seed=1)
    other = make_episode(2, 1024, hole=(0, 400, 520), seed=7)
    assert torch.equal(base.valid, other.valid)
    _, tokens_a = encode(frontend, base, ViewKind.GLOBAL)
    _, tokens_b = encode(frontend, other, ViewKind.GLOBAL)
    assert torch.equal(tokens_a.valid[tokens_a.family != FAMILY_D_SALIENT], tokens_b.valid[tokens_b.family != FAMILY_D_SALIENT])
    both = tokens_a.valid & tokens_b.valid
    difference = (tokens_a.descriptors - tokens_b.descriptors).abs().amax(dim=-1)
    assert (difference[both] > 0).all()
    assert (difference[~both & ~(tokens_a.valid | tokens_b.valid)] == 0).all()


# --------------------------------------------------------------------------- #
# 8. Sort order
# --------------------------------------------------------------------------- #


def _first_significant_difference(keys: list[torch.Tensor]) -> torch.Tensor:
    """Difference of consecutive rows on the most significant key that differs."""

    out = torch.zeros(keys[0].numel() - 1, dtype=torch.float64)
    for key in reversed(keys):  # least significant first, most significant last
        step = key.to(torch.float64)
        delta = step[1:] - step[:-1]
        out = torch.where(delta != 0, delta, out)
    return out


@pytest.mark.parametrize("kind,start,end", [(ViewKind.GLOBAL, 0, None), (ViewKind.LOCAL, 128, 896)])
def test_sort_order_is_lexicographic(frontend: PerceptualFrontend, kind: ViewKind, start: int, end) -> None:
    episode = make_episode(2, 1024, hole=(0, 400, 520), spikes=(200, 600))
    _, tokens = encode(frontend, episode, kind, start=start, end=end)
    order = tokens.sort_order
    assert order.dtype == torch.int64
    assert torch.equal(order.sort().values, torch.arange(tokens.n_tokens))
    n_valid = tokens.n_valid
    # the hole (and the quota that no channel can fill) leaves empty slots behind
    assert 0 < n_valid < tokens.n_tokens
    ordered_valid = tokens.valid[order]
    assert ordered_valid[:n_valid].all() and not ordered_valid[n_valid:].any()
    keys = [
        tokens.anchor_time[order][:n_valid],
        tokens.channel_index[order][:n_valid],
        tokens.family[order][:n_valid],
        tokens.level[order][:n_valid],
        tokens.coefficient_index[order][:n_valid],
    ]
    difference = _first_significant_difference(keys)
    # (anchor_time, channel, family, level, index) is a strict total order on the
    # valid tokens and the permutation is sorted by it
    assert (difference > 0).all()
    # the same key computed by an independent stable lexsort agrees slot by slot
    reference = sorted(
        (int(slot) for slot in tokens.valid.nonzero().squeeze(1).tolist()),
        key=lambda slot: (
            float(tokens.anchor_time[slot]),
            int(tokens.channel_index[slot]),
            int(tokens.family[slot]),
            int(tokens.level[slot]),
            int(tokens.coefficient_index[slot]),
        ),
    )
    assert order[:n_valid].tolist() == reference


# --------------------------------------------------------------------------- #
# 9. End to end with the quotas of prototype.spec.json
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("crop_channels", [("ch0", "ch1"), ("ch1",)])
def test_end_to_end_spec_quotas(frontend: PerceptualFrontend, spec: dict, crop_channels) -> None:
    quotas = spec["tokenizer"]
    assert {quotas["global_per_channel"][name] for name in ("A", "D_coverage", "D_salient", "raw")} == {64}
    assert {quotas["local_per_channel"][name] for name in ("A", "D_coverage", "D_salient", "raw")} == {128}

    length = 4096
    episode = make_episode(2, length, spikes=(700, 2048, 3300))
    stats = compute_recording_normalization(episode)

    global_view = make_view(episode, stats, kind=ViewKind.GLOBAL)
    global_tokens = frontend.encode_view(global_view, recording_length=length)
    assert global_tokens.descriptors.shape == (2 * N_FAMILIES * 64, 512)
    assert global_tokens.raw_bin_features.shape == (2, 64, 64)
    assert frontend.aux_logits(global_tokens).shape == (2, 64)
    assert int(global_tokens.raw_bin_edges[0]) == 0
    assert int(global_tokens.raw_bin_edges[-1]) == length
    # 256 perceptual tokens per channel at most, exactly as in the DESIGN 6.2 table
    for channel in range(2):
        assert int((global_tokens.valid & (global_tokens.channel_index == channel)).sum()) <= 256
    _assert_clean(global_tokens)

    crop_start, crop_end = 1536, 2560  # 1024 samples around the middle spike
    local_view = make_view(
        episode, stats, kind=ViewKind.LOCAL, start=crop_start, end=crop_end, channels=crop_channels
    )
    local_tokens = frontend.encode_view(local_view, recording_length=length)
    n_crop_channels = len(crop_channels)
    assert local_tokens.descriptors.shape == (n_crop_channels * N_FAMILIES * 128, 512)
    assert local_tokens.raw_bin_features.shape == (n_crop_channels, 128, 64)
    assert frontend.aux_logits(local_tokens).shape == (n_crop_channels, 128)
    assert int(local_tokens.raw_bin_edges[0]) == crop_start
    assert int(local_tokens.raw_bin_edges[-1]) == crop_end
    assert local_tokens.channel_index.unique().tolist() == [
        episode.channel_ids.index(name) for name in crop_channels
    ]
    valid = local_tokens.valid
    assert (local_tokens.support_start[valid] >= crop_start).all()
    assert (local_tokens.support_end[valid] <= crop_end).all()
    for channel in range(n_crop_channels):
        index = episode.channel_ids.index(crop_channels[channel])
        assert int((valid & (local_tokens.channel_index == index)).sum()) <= 512
    _assert_clean(local_tokens)

    # the attributes WRAgentModel.encode_episode reads must all be present and usable
    for tokens in (global_tokens, local_tokens):
        n_valid = tokens.n_valid
        order = tokens.sort_order[:n_valid]
        assert bool(tokens.valid[order].all())
        assert tokens.descriptors[order].shape == (n_valid, 512)
        assert tokens.anchor_time[order].shape == (n_valid,)
        assert tokens.channel_index[order].shape == (n_valid,)
        assert tokens.raw_bin_valid.shape == frontend.aux_logits(tokens).shape
        assert tokens.raw_bin_edges.numel() == tokens.raw_bin_valid.shape[1] + 1


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #


def test_robust_band_scale_rules() -> None:
    values = torch.tensor(
        [
            [1.0, -1.0, 3.0, 100.0],
            [5.0, 5.0, 5.0, 5.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 3.0],
        ]
    )
    usable = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, True],
            [False, False, False, False],
            [True, True, True, True],
        ]
    )
    scale = robust_band_scale(values, usable)
    assert scale.dtype == torch.float32 and scale.shape == (4,)
    # channel 0: usable coefficients {1, -1, 3}: median 1, |V - 1| = {0, 2, 2},
    # MAD = median{0, 2, 2} = 2, so the robust scale is 1.4826 * 2.
    assert scale[0] == pytest.approx(2.0 * 1.4826, rel=1e-6)
    # channel 3: {0, 1, 2, 3}: median 1.5 (mean of the two middle values),
    # |V - 1.5| = {1.5, .5, .5, 1.5}, MAD = 1 -> exactly 1.4826.
    assert scale[3] == pytest.approx(1.4826, rel=1e-6)
    # constant band: MAD 0, numerical guard epsilon = max(1e-6 * 5, tiny), then the absolute floor 1e-3 in
    # normalised units (Astra 15/09, scala) wins.
    assert scale[1] == pytest.approx(1e-3, rel=1e-3)
    # no usable coefficient: scale 1 (its slots are invalid anyway)
    assert float(scale[2]) == 1.0
    # unusable coefficients never enter the statistic
    altered = values.clone()
    altered[0, 3] = -1e6
    assert torch.equal(robust_band_scale(altered, usable), scale)


def test_deep_branches_are_shared_across_channels(spec: dict) -> None:
    torch.manual_seed(0)
    branches = DeepBranches(spec)
    feats = torch.randn(2, 30, 9)
    valid = torch.ones(2, 30, dtype=torch.bool)
    out = branches.forward_A(feats, valid, False)
    swapped = branches.forward_A(feats.flip(0), valid, False)
    assert torch.allclose(out.flip(0), swapped, atol=1e-6)
    with pytest.raises(ValueError):
        branches.forward_D(torch.randn(1, 4, 11), torch.ones(1, 4, dtype=torch.bool), 0, False)


def test_fourier_position_resolves_neighbouring_samples_where_the_scalar_cannot() -> None:
    """Il punto della codifica multi-frequenza: distinguere posizioni che lo scalare confonde.

    position/L a 131.072 campioni, sotto autocast bf16, non risolve nulla sotto i ~585 campioni a fine serie --
    contro un'anomalia mediana di 137.  Le stesse posizioni devono invece separarsi nettamente nelle feature di
    Fourier, ed e' per questo che il calcolo sta in float64.
    """

    import math

    from wavelet_rope.tokenizer import PerceptualTokenizer

    probe = PerceptualTokenizer.__new__(PerceptualTokenizer)
    probe.position_fourier_freqs = 16
    probe.position_proj = None
    probe.position_freqs = (2.0 ** torch.arange(16, dtype=torch.float64)) * math.pi

    length = 131072
    far_end = float(length - 1000)
    positions = torch.tensor([far_end, far_end + 137.0], dtype=torch.float64)

    # lo scalare: identico in bf16, che e' il modo di fallire gia' documentato
    scalar = (positions / length).to(torch.bfloat16)
    assert scalar[0] == scalar[1], "il caso di fallimento dello scalare e' cambiato: rivedere il test"

    # le feature di Fourier, in float64: le due posizioni devono essere ben separate
    angle = (positions / length)[:, None] * probe.position_freqs[None, :]
    features = torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1).to(torch.float32)
    distance = float((features[0] - features[1]).norm())
    assert distance > 0.5, f"137 campioni separati da solo {distance:.4f}"


def test_fourier_position_widens_the_fusion_input_and_is_off_by_default() -> None:
    """Le feature devono entrare NELL'INGRESSO della fusione, non essere sommate al contenuto normalizzato.

    Il primo tentativo le sommava dopo ``fuse_norm`` e il train del piolo crollava da 20/32 a 3/32: 32 dimensioni
    di ampiezza 1 sopra un vettore normalizzato dominano il contenuto.  va1 costruisce invece il token da
    posizione e valore insieme, ed e' quella la forma da riprodurre.  Il test guarda la larghezza dello stem,
    che e' la firma osservabile della differenza.
    """

    import json
    from pathlib import Path

    from wavelet_rope.tokenizer import FUSION_INPUT_DIM, PerceptualFrontend

    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / "prototype.spec.va1rz.json").read_text(encoding="utf-8"))
    assert int(spec["tokenizer"].get("position_fourier_freqs", 0)) == 0, "il braccio base deve restare spento"
    assert PerceptualFrontend(spec).tokenizer.fuse_in.in_features == FUSION_INPUT_DIM

    acceso = json.loads((root / "prototype.spec.va1ethpos.json").read_text(encoding="utf-8"))
    freqs = int(acceso["tokenizer"]["position_fourier_freqs"])
    assert freqs == 16
    assert PerceptualFrontend(acceso).tokenizer.fuse_in.in_features == FUSION_INPUT_DIM + 2 * freqs
