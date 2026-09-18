"""Train-only token-substitution augmentation.

The augmentation deliberately lives outside the perceptual-gate code.  The
gate's ``SUBSTITUTED`` condition is a frozen evaluation perturbation with a
different meaning; this module builds a real training episode and lets the
normal encoder reconstruct its tokens and targets from that episode.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch

from .data import GoldInterval, RowMeta
from .signal_contract import SignalEpisode


NO_COMPATIBLE_TRAIN_DONOR = "no_compatible_train_donor"


@dataclass(frozen=True, slots=True)
class DonorSelection:
    """A donor row and its lazily loaded episode."""

    meta: RowMeta
    episode: SignalEpisode


@dataclass(frozen=True, slots=True)
class SubstitutionDecision:
    """The deterministic decision made for one training example."""

    requested: bool
    substituted: bool
    donor: DonorSelection | None = None
    reason: str | None = None

    @property
    def donor_id(self) -> str | None:
        return None if self.donor is None else self.donor.meta.id

    @property
    def donor_row_index(self) -> int | None:
        return None if self.donor is None else int(self.donor.meta.row_index)


def validate_substitution_share(value: Any) -> float:
    """Validate and return ``training.substitution_share`` in ``[0, 1]``."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("training.substitution_share must be a real number in [0, 1]")
    share = float(value)
    if not math.isfinite(share) or not 0.0 <= share <= 1.0:
        raise ValueError("training.substitution_share must be finite and in [0, 1]")
    return share


def example_rng(seed: int, pass_index: int, row_index: int) -> random.Random:
    """Return the same per-example RNG scheme used by the trainer."""

    return random.Random(f"{int(seed)}:{int(pass_index)}:{int(row_index)}")


def _effective_dt(episode: SignalEpisode) -> float:
    # ``encode_episode`` uses 1.0 when a one-sample episode has no measurable
    # delta, so compatibility must use the same effective cadence.
    return 1.0 if episode.dt is None else float(episode.dt)


def donor_compatibility_reason(
    receiver_meta: RowMeta,
    receiver: SignalEpisode,
    donor_meta: RowMeta,
    donor: SignalEpisode,
) -> str | None:
    """Return a stable incompatibility reason, or ``None`` when compatible.

    Validity compatibility is intentionally directional: every sample shown
    by the receiver must also be finite/valid in the donor.  Extra donor-valid
    samples are harmless because the substituted episode keeps the receiver's
    mask exactly.
    """

    if donor_meta.split != "train":
        return "donor_not_from_train"
    if donor_meta.row_index == receiver_meta.row_index:
        return "donor_is_receiver_row"
    if receiver.length != receiver_meta.length or receiver.channel_ids != receiver_meta.channel_names:
        return "receiver_meta_episode_contract_mismatch"
    if donor.length != donor_meta.length or donor.channel_ids != donor_meta.channel_names:
        return "donor_meta_episode_contract_mismatch"
    if donor_meta.length != receiver_meta.length or donor.length != receiver.length:
        return "length_mismatch"
    if donor_meta.n_channels != receiver_meta.n_channels or donor.n_channels != receiver.n_channels:
        return "channel_count_mismatch"
    if donor_meta.channel_names != receiver_meta.channel_names:
        return "channel_names_mismatch"
    if donor.channel_ids != receiver.channel_ids:
        return "episode_channel_names_mismatch"
    if not math.isclose(_effective_dt(receiver), _effective_dt(donor), rel_tol=1e-9, abs_tol=1e-12):
        return "dt_mismatch"
    if donor.valid.shape != receiver.valid.shape:
        return "validity_shape_mismatch"

    donor_valid = donor.valid.to(device=receiver.valid.device)
    if bool((receiver.valid & ~donor_valid).any()):
        return "donor_validity_does_not_cover_receiver"
    donor_values = donor.values.to(device=receiver.valid.device)
    if not bool(torch.isfinite(donor_values[receiver.valid]).all()):
        return "donor_values_nonfinite_under_receiver_mask"
    return None


def select_substitution_donor(
    metas: Sequence[RowMeta],
    receiver_meta: RowMeta,
    receiver: SignalEpisode,
    episode_getter: Callable[[int], SignalEpisode],
    *,
    rng: random.Random,
) -> DonorSelection | None:
    """Select one deterministic compatible donor from the train rows.

    Metadata filters are applied before loading signal values.  The remaining
    row indices are sorted and shuffled with the supplied per-example RNG;
    candidates are loaded in that deterministic order until the first one
    whose validity covers the receiver's valid samples.
    """

    if not hasattr(rng, "shuffle"):
        raise TypeError("rng must provide shuffle()")
    candidates = [
        meta
        for meta in metas
        if meta.split == "train"
        and meta.row_index != receiver_meta.row_index
        and meta.length == receiver_meta.length
        and meta.n_channels == receiver_meta.n_channels
        and meta.channel_names == receiver_meta.channel_names
    ]
    candidates.sort(key=lambda meta: (int(meta.row_index), meta.id))
    rng.shuffle(candidates)
    for donor_meta in candidates:
        donor = episode_getter(int(donor_meta.row_index))
        if donor_compatibility_reason(receiver_meta, receiver, donor_meta, donor) is None:
            return DonorSelection(meta=donor_meta, episode=donor)
    return None


def prepare_substitution(
    share: float,
    *,
    metas: Sequence[RowMeta],
    receiver_meta: RowMeta,
    receiver: SignalEpisode,
    episode_getter: Callable[[int], SignalEpisode],
    rng: random.Random,
) -> SubstitutionDecision:
    """Apply the per-example share and, if requested, choose a donor."""

    share = validate_substitution_share(share)
    if share == 0.0:
        return SubstitutionDecision(requested=False, substituted=False)
    if not hasattr(rng, "random"):
        raise TypeError("rng must provide random()")
    requested = bool(rng.random() < share)
    if not requested:
        return SubstitutionDecision(requested=False, substituted=False)
    donor = select_substitution_donor(
        metas,
        receiver_meta,
        receiver,
        episode_getter,
        rng=rng,
    )
    if donor is None:
        return SubstitutionDecision(
            requested=True,
            substituted=False,
            reason=NO_COMPATIBLE_TRAIN_DONOR,
        )
    return SubstitutionDecision(requested=True, substituted=True, donor=donor)


def make_substituted_episode(receiver: SignalEpisode, donor: SignalEpisode) -> SignalEpisode:
    """Use donor values with receiver validity, channel IDs and sample times."""

    # The public constructor is intentionally usable with bare episodes (as in
    # CPU tests), so perform the episode-level checks directly rather than
    # requiring RowMeta objects.
    if receiver.length != donor.length:
        raise ValueError("substitution requires equal recording lengths")
    if receiver.n_channels != donor.n_channels:
        raise ValueError("substitution requires equal channel counts")
    if receiver.channel_ids != donor.channel_ids:
        raise ValueError("substitution requires the same ordered channel IDs")
    if not math.isclose(_effective_dt(receiver), _effective_dt(donor), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("substitution requires equal dt")
    donor_valid = donor.valid.to(device=receiver.valid.device)
    if bool((receiver.valid & ~donor_valid).any()):
        raise ValueError("donor validity does not cover the receiver mask")
    donor_values = donor.values.to(device=receiver.values.device)
    if not bool(torch.isfinite(donor_values[receiver.valid]).all()):
        raise ValueError("donor values are nonfinite under the receiver mask")
    return SignalEpisode(
        values=donor_values.clone(),
        valid=receiver.valid.clone(),
        channel_ids=receiver.channel_ids,
        sample_times=receiver.sample_times.clone(),
        task=receiver.task,
    )


def mask_gold_intervals(
    intervals: Sequence[GoldInterval],
    receiver: SignalEpisode,
) -> tuple[GoldInterval, ...]:
    """Restrict donor annotations to the receiver's observed samples.

    The output may split one annotation into several contiguous runs when the
    receiver mask contains holes.  Coordinates stay absolute, so occupancy
    targets remain aligned with the receiver grid.
    """

    channel_index = {name: index for index, name in enumerate(receiver.channel_ids)}
    valid = receiver.valid.detach().to(device="cpu")
    result: list[GoldInterval] = []
    for interval in intervals:
        index = channel_index.get(interval.channel)
        if index is None:
            continue
        start = max(0, int(interval.start))
        end = min(receiver.length, int(interval.end))
        if start >= end:
            continue
        flags = valid[index, start:end].tolist()
        run_start: int | None = None
        for offset, is_valid in enumerate(flags):
            position = start + offset
            if is_valid and run_start is None:
                run_start = position
            elif not is_valid and run_start is not None:
                result.append(GoldInterval(interval.channel, run_start, position, interval.kind))
                run_start = None
        if run_start is not None:
            result.append(GoldInterval(interval.channel, run_start, end, interval.kind))
    return tuple(result)


# Short aliases make the contract easy to discover for callers/tests.
substitute_episode = make_substituted_episode
masked_gold_intervals = mask_gold_intervals


__all__ = [
    "DonorSelection",
    "NO_COMPATIBLE_TRAIN_DONOR",
    "SubstitutionDecision",
    "donor_compatibility_reason",
    "example_rng",
    "make_substituted_episode",
    "mask_gold_intervals",
    "masked_gold_intervals",
    "prepare_substitution",
    "select_substitution_donor",
    "substitute_episode",
    "validate_substitution_share",
]
