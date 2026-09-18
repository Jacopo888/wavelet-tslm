"""Programmatic P2 trajectory teacher.

The teacher uses annotations only to choose training targets and windows.  Text
placed before each action is computed from the global or local signal views
already acquired at that point.  ``ViewRef`` keeps the resulting turns
backbone-free: the model resolves it to the exact number of valid soft tokens
after its learned frontend has encoded the same action prefix.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import math
import random
import re
from typing import Any, Sequence

import torch
from torch import Tensor

from .data import GoldInterval, RowMeta, ZoomAction
from .haar_selector import blind_candidates, window_containing_point
from .signal_contract import SignalEpisode
from .templates import SEG_ACTION, SEG_FINAL, ConversationBuilder, Turn, ViewRef


TRAJECTORY_KINDS: tuple[str, ...] = (
    "final_no_tool",
    "one_zoom_correct",
    "wrong_then_recover",
    "refine_boundaries",
    "two_channels",
    "negative_stop",
    "multi_event",
)

TRAJECTORY_QUOTAS: dict[str, float] = {
    "final_no_tool": 0.15,
    "one_zoom_correct": 0.25,
    "wrong_then_recover": 0.15,
    "refine_boundaries": 0.10,
    "two_channels": 0.15,
    "negative_stop": 0.10,
    "multi_event": 0.10,
}


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    exact = {key: total * value for key, value in weights.items()}
    result = {key: int(math.floor(value)) for key, value in exact.items()}
    left = total - sum(result.values())
    for key in sorted(weights, key=lambda name: (-(exact[name] - result[name]), name))[:left]:
        result[key] += 1
    return result


def _seed(*parts: object) -> int:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _usable_gold(meta: RowMeta) -> tuple[GoldInterval, ...]:
    names = set(meta.channel_names)
    return tuple(
        item for item in meta.gold_intervals
        if item.channel in names and 0 <= item.start < item.end <= meta.length
    )


def _answer_is_absent(answer: str) -> bool:
    """Conservative no-anomaly detector used only for kind compatibility."""

    text = str(answer).lower()
    return bool(
        re.search(r'"present"\s*:\s*false', text)
        or "no anomaly" in text
        or "absent" in text
        or "no_anomaly" in text
    )


def _has_structural_gold_free_window(meta: RowMeta) -> bool:
    """Cheap compatibility test that does not inspect signal values."""

    gold = _usable_gold(meta)
    if not gold:
        return True
    # A one-channel 256-sample window is enough for the P2 selector.  Check
    # whether at least one channel has an uncovered gap of that width.  Rows
    # shorter than 256 have only the whole-series exception and are not useful
    # wrong-window examples when they contain a gold event.
    width = min(256, meta.length)
    if meta.length < 256:
        return False
    for channel in meta.channel_names:
        spans = sorted((g.start, g.end) for g in gold if g.channel == channel)
        cursor = 0
        for start, end in spans:
            if start - cursor >= width:
                return True
            cursor = max(cursor, end)
        if meta.length - cursor >= width:
            return True
    return False


def compatible_kinds(meta: RowMeta) -> tuple[str, ...]:
    """Return trajectory kinds whose structural prerequisites a row satisfies."""

    gold = _usable_gold(meta)
    distinct = {(g.channel, g.start, g.end) for g in gold}
    result = ["final_no_tool"]
    if gold:
        result.append("one_zoom_correct")
        if _has_structural_gold_free_window(meta):
            result.append("wrong_then_recover")
        if meta.length >= 512:
            result.append("refine_boundaries")
    if len(meta.channel_names) >= 2:
        result.append("two_channels")
    # A negative local crop is useful even on a positive row when the final is
    # grounded in the global evidence.  The no-anomaly/absent rows are the
    # canonical cases; positive rows are admitted when a truly gold-free
    # candidate exists so the teacher never claims that a local void proves
    # global absence.
    if not gold or _answer_is_absent(meta.answer) or _has_structural_gold_free_window(meta):
        result.append("negative_stop")
    if len(distinct) >= 2 and meta.length >= 256:
        result.append("multi_event")
    return tuple(kind for kind in TRAJECTORY_KINDS if kind in result)


@dataclass
class Trajectory:
    """A complete teacher episode in conversation order."""

    turns: list[Turn]
    actions: list[ZoomAction]
    kind: str
    injected_error: bool = False
    injected_error_action: ZoomAction | None = None
    injected_error_turn_index: int | None = None
    offsets: list[tuple[int, int]] = field(default_factory=list)
    requested_kind: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_actions(self) -> list[ZoomAction]:
        return self.actions

    @property
    def supervision_flags(self) -> list[bool]:
        return [bool(turn.supervise) for turn in self.turns]

    @property
    def action_supervision(self) -> list[bool]:
        return [
            bool(turn.supervise)
            for turn in self.turns
            if turn.role in ("assistant_action", "assistant_reasoning_action")
        ]

    @property
    def supervised_turns(self) -> list[Turn]:
        return [turn for turn in self.turns if turn.supervise]

    @property
    def supervised_segments(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for turn in self.turns:
            if not turn.supervise:
                continue
            if turn.role == "assistant_reasoning_action":
                if turn.supervise_reasoning is not False:
                    counts["reasoning"] += 1
                counts["action"] += 1
            elif turn.role == "assistant_reasoning_final":
                if turn.supervise_reasoning is not False:
                    counts["reasoning"] += 1
                if turn.supervise_final is not False:
                    counts["final"] += 1
            elif turn.role == "assistant_action":
                counts["action"] += 1
            elif turn.role == "assistant_final":
                if turn.supervise_final is not False:
                    counts["final"] += 1
        return dict(counts)

    @property
    def injected_wrong_action(self) -> ZoomAction | None:
        return self.injected_error_action

    @property
    def injected_error_action_index(self) -> int | None:
        if self.injected_error_action is None:
            return None
        try:
            return self.actions.index(self.injected_error_action)
        except ValueError:  # pragma: no cover - defensive for hand-built records
            return None

    @property
    def offsets_used(self) -> list[tuple[int, int]]:
        return self.offsets


class TrajectoryKindAssigner:
    """Stratify agentic trajectory kinds within category-compatible rows."""

    def __init__(self, quotas: dict[str, float] | None = None) -> None:
        self.quotas = dict(quotas or TRAJECTORY_QUOTAS)
        if set(self.quotas) != set(TRAJECTORY_KINDS):
            raise ValueError("trajectory quotas must contain exactly the seven P2 kinds")
        total = sum(float(value) for value in self.quotas.values())
        if abs(total - 1.0) > 1e-6 or any(float(value) < 0 for value in self.quotas.values()):
            raise ValueError("trajectory quotas must be non-negative and sum to one")
        self.last_report: dict[str, Any] = {}

    def assign(
        self,
        metas: Sequence[RowMeta],
        *,
        seed: int,
        pass_index: int,
        row_indices: Sequence[int] | None = None,
    ) -> list[tuple[int, str]]:
        by_index = {meta.row_index: meta for meta in metas}
        if len(by_index) != len(metas):
            raise ValueError("metas must contain unique row_index values")
        if row_indices is None:
            selected = list(by_index)
        else:
            selected = [int(index) for index in row_indices]
            if len(set(selected)) != len(selected) or any(index not in by_index for index in selected):
                raise ValueError("row_indices must be unique indices present in metas")
        categories: dict[str, list[int]] = {}
        for index in selected:
            categories.setdefault(by_index[index].category, []).append(index)

        assignments: list[tuple[int, str]] = []
        requested_counts: Counter[str] = Counter()
        for category in sorted(categories):
            rows = sorted(categories[category])
            random.Random(_seed("wr-trajectory-kind", seed, pass_index, category)).shuffle(rows)
            quotas = _largest_remainder(len(rows), self.quotas)
            unassigned = set(rows)
            # Constrained kinds get first choice.  This prevents a broad
            # final_no_tool pool from consuming rows that can support only a
            # multichannel or multi-event trajectory.
            kinds_by_candidates = sorted(
                TRAJECTORY_KINDS,
                key=lambda kind: (
                    sum(kind in compatible_kinds(by_index[index]) for index in rows),
                    TRAJECTORY_KINDS.index(kind),
                ),
            )
            deficits = dict(quotas)
            for kind in kinds_by_candidates:
                if deficits[kind] <= 0:
                    continue
                candidates = [
                    index for index in rows
                    if index in unassigned and kind in compatible_kinds(by_index[index])
                ]
                random.Random(_seed("wr-trajectory-pool", seed, pass_index, category, kind)).shuffle(candidates)
                for index in candidates[: deficits[kind]]:
                    assignments.append((index, kind))
                    requested_counts[kind] += 1
                    unassigned.remove(index)
                    deficits[kind] -= 1
            # Reassign unfilled quotas and any remaining rows to the most
            # under-quota compatible kind, with a deterministic tie break.
            for index in rows:
                if index not in unassigned:
                    continue
                options = [kind for kind in TRAJECTORY_KINDS if kind in compatible_kinds(by_index[index])]
                if not options:
                    raise RuntimeError(f"row {index} has no compatible trajectory kind")
                kind = min(options, key=lambda name: (-deficits[name], TRAJECTORY_KINDS.index(name)))
                assignments.append((index, kind))
                requested_counts[kind] += 1
                deficits[kind] = max(0, deficits[kind] - 1)
                unassigned.remove(index)

        random.Random(_seed("wr-trajectory-order", seed, pass_index)).shuffle(assignments)
        self.last_report = self.report(assignments)
        return assignments

    def report(self, assignments: Sequence[tuple[int, str]] | None = None) -> dict[str, Any]:
        if assignments is None and self.last_report:
            return dict(self.last_report)
        counts = Counter(kind for _index, kind in (assignments or []))
        total = sum(counts.values())
        return {
            "counts": {kind: int(counts.get(kind, 0)) for kind in TRAJECTORY_KINDS},
            "fractions": {
                kind: (counts.get(kind, 0) / total if total else 0.0)
                for kind in TRAJECTORY_KINDS
            },
            "declared_quotas": dict(self.quotas),
            "total": total,
        }

    @staticmethod
    def compatible(meta: RowMeta) -> tuple[str, ...]:
        return compatible_kinds(meta)


class TrajectoryTeacher:
    """Build deterministic, annotation-planned but observation-grounded turns."""

    def __init__(
        self,
        kind: str | None = None,
        *,
        max_calls: int = 6,
        max_views: int = 8,
        max_reasoning_tokens: int = 384,
        tokenizer: Any | None = None,
        width_rule: str = "log_uniform",
    ) -> None:
        if width_rule not in ("log_uniform", "coarse_to_fine"):
            raise ValueError(f"unknown width rule {width_rule!r}")
        self.width_rule = width_rule
        if kind is not None and kind not in TRAJECTORY_KINDS:
            raise ValueError(f"unknown trajectory kind {kind!r}")
        if isinstance(max_calls, bool) or int(max_calls) < 0:
            raise ValueError("max_calls must be non-negative")
        if isinstance(max_views, bool) or int(max_views) < 0:
            raise ValueError("max_views must be non-negative")
        if isinstance(max_reasoning_tokens, bool) or int(max_reasoning_tokens) < 0:
            raise ValueError("max_reasoning_tokens must be non-negative")
        self.kind = kind
        self.max_calls = int(max_calls)
        self.max_views = int(max_views)
        self.max_reasoning_tokens = int(max_reasoning_tokens)
        self.tokenizer = tokenizer
        self._realised: Counter[str] = Counter()

    @property
    def realised_distribution(self) -> dict[str, int]:
        return {kind: int(self._realised.get(kind, 0)) for kind in TRAJECTORY_KINDS}

    @property
    def realized_distribution(self) -> dict[str, int]:
        """US-spelling alias for callers that use ``realized`` in logs."""

        return self.realised_distribution

    def report_realised_distribution(self) -> dict[str, Any]:
        total = sum(self._realised.values())
        return {
            "counts": self.realised_distribution,
            "fractions": {
                kind: self._realised.get(kind, 0) / total if total else 0.0
                for kind in TRAJECTORY_KINDS
            },
            "declared_quotas": dict(TRAJECTORY_QUOTAS),
            "total": total,
        }

    # ------------------------------------------------------------------ text
    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            try:
                encoded = self.tokenizer(text, add_special_tokens=False)
                ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
                return len(ids)
            except Exception:  # pragma: no cover - conservative fallback
                pass
        words = len(re.findall(r"\S+", text))
        return max(1, int(math.ceil(words * 1.5))) if text.strip() else 0

    def _cap_reasoning(self, text: str) -> str:
        if self.max_reasoning_tokens <= 0:
            return ""
        # The literal closure is supervised as part of SEG_REASONING, so the
        # body must leave room for it in the per-decision cap.
        closure_budget = self._count_tokens("\n</think>\n\n")
        body_budget = max(0, self.max_reasoning_tokens - closure_budget)
        if self._count_tokens(text) <= body_budget:
            return text
        if self.tokenizer is not None:
            try:
                encoded = self.tokenizer(text, add_special_tokens=False)
                ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
                return self.tokenizer.decode(ids[: body_budget], skip_special_tokens=False).strip()
            except Exception:  # pragma: no cover
                pass
        words = text.split()
        keep = max(0, int(body_budget / 1.5))
        return " ".join(words[:keep])

    def _safe_values(self, episode: SignalEpisode, channels: Sequence[str], start: int, end: int) -> tuple[Tensor, Tensor]:
        indices = [episode.channel_ids.index(channel) for channel in channels]
        values = episode.values[indices, start:end].to(torch.float64)
        valid = episode.valid[indices, start:end]
        return values, valid

    def _view_reasoning(
        self,
        episode: SignalEpisode,
        *,
        acquired: Sequence[ZoomAction],
        next_action: ZoomAction | None,
        final: bool = False,
        global_only: bool = False,
    ) -> str:
        """Describe only signal quantities available in acquired views."""

        if acquired and (not final or not global_only):
            view = acquired[-1]
            start, end = view.start, view.end
            channels = view.channels
            label = f"After the local view [{start},{end})"
        else:
            start, end = 0, episode.length
            channels = episode.channel_ids
            label = "Considering the global view" if acquired else "From the global view"

        values, valid = self._safe_values(episode, channels, start, end)
        clauses: list[str] = []
        deviation_peaks: list[tuple[float, str, int]] = []
        for slot, channel in enumerate(channels):
            observed = values[slot][valid[slot]]
            if observed.numel() == 0:
                continue
            median = observed.median()
            mad = (observed - median).abs().median()
            scale = max(float(mad * 1.4826), float(observed.abs().amax()) * 1e-6, 1e-6)
            deviation = (observed - median).abs() / scale
            extreme = int(valid[slot].nonzero().squeeze(1)[int(deviation.argmax())]) + start
            deviation_peaks.append((float(deviation.max()), channel, extreme))
            clauses.append(
                f"{channel} robust range {float(observed.min()):.3g}..{float(observed.max()):.3g}, "
                f"largest deviation near {extreme}"
            )

        if deviation_peaks:
            _peak, peak_channel, peak_index = max(
                deviation_peaks, key=lambda item: (item[0], item[1], -item[2])
            )
            clauses.append(f"largest robust deviation is in {peak_channel} near {peak_index}")

        # A small, absolute Haar bucket summary is computable at every view
        # resolution.  It intentionally says where energy is, not why it is.
        try:
            selector_values = torch.where(valid, values, torch.zeros_like(values))
            view_length = end - start
            selector_width = min(max(1, view_length), 512)
            candidates = blind_candidates(selector_values, selector_width, min(2, len(channels) * 2))
            bucket_count = min(view_length, max(8, math.ceil(view_length / max(1, selector_width))))
            edges = [index * view_length // bucket_count for index in range(bucket_count + 1)]
            summaries: list[str] = []
            for channel_index, centre in candidates[:2]:
                bucket = min(
                    range(bucket_count),
                    key=lambda index: (abs((edges[index] + edges[index + 1]) // 2 - centre), index),
                )
                summaries.append(
                    f"{channels[channel_index]}[{start + edges[bucket]},{start + edges[bucket + 1]})"
                )
            energy_text = ", ".join(summaries)
        except (TypeError, ValueError, RuntimeError):  # very short/degenerate view
            energy_text = ""
        if energy_text:
            clauses.append(f"Haar-energy concentrations are around {energy_text}")
        if not clauses:
            clauses.append("the acquired samples contain no valid numeric evidence")

        if final:
            prefix = f"{label} "
            suffix = " The available evidence is sufficient for the requested answer."
        elif next_action is None:
            prefix = f"{label} "
            suffix = " I will answer from the evidence already acquired."
        else:
            remaining_calls = max(0, self.max_calls - len(acquired))
            remaining_views = max(0, self.max_views - sum(len(a.channels) for a in acquired))
            prefix = f"{label} "
            suffix = (
                f" Coverage is {len(acquired)} local views with {remaining_calls} calls and "
                f"{remaining_views} channel views remaining; I will inspect another region."
            )
        return self._cap_reasoning(prefix + "; ".join(clauses) + "." + suffix)

    # -------------------------------------------------------------- windows
    def _width(self, length: int, rng: random.Random, *, level: int = 0) -> int:
        """Window width for one zoom.
    
        ``log_uniform`` samples in [256, min(8192, L)] regardless of L.  That is what the Z1-bis traces did, and
        it is why the anomaly sat inside the window 80% of the time at L=16.384 but only 32% at L=131.072.
    
        ``coarse_to_fine`` is the rule Astra fixed on 16/09: the first window scales with the recording,
        ``clip(L/16, 1024, 8192)``, and each refinement divides it by four down to a floor of 256.
        """
    
        if length < 256:
            return length
        if self.width_rule == "coarse_to_fine":
            first = min(8192, max(1024, length // 16))
            return max(256, min(length, first // (4 ** max(0, int(level)))))
        high = min(8192, length)
        if high <= 256:
            return high
        return max(256, min(high, int(round(math.exp(rng.uniform(math.log(256), math.log(high)))))))

    @staticmethod
    def _overlaps(action: ZoomAction, gold: Sequence[GoldInterval]) -> bool:
        return any(
            item.channel in action.channels and max(action.start, item.start) < min(action.end, item.end)
            for item in gold
        )

    def _start_for_interval(
        self,
        interval: GoldInterval,
        width: int,
        length: int,
        rng: random.Random,
        *,
        contain: bool = True,
    ) -> int:
        width = min(width, length)
        a, b = interval.start, interval.end
        low = max(0, b - width) if contain else max(0, a - width + 1)
        high = min(a, length - width) if contain else min(b - 1, length - width)
        if low > high:
            # A crop narrower than an annotated event cannot contain it
            # wholly, but it should still overlap the event rather than
            # degrade to an unrelated random window.
            low = max(0, a - width + 1)
            high = min(b - 1, length - width)
        if low > high:
            low, high = 0, length - width
        choices = list(range(low, high + 1))
        if len(choices) > 1:
            centred = (a + b - width) / 2.0
            non_centre = [item for item in choices if abs(item - centred) > 0.5]
            if non_centre:
                choices = non_centre
        return int(rng.choice(choices))

    def _action_for_interval(
        self,
        interval: GoldInterval,
        meta: RowMeta,
        rng: random.Random,
        *,
        channels: Sequence[str] | None = None,
        width: int | None = None,
        contain: bool = True,
    ) -> ZoomAction:
        width = self._width(meta.length, rng) if width is None else min(int(width), meta.length)
        start = self._start_for_interval(interval, width, meta.length, rng, contain=contain)
        selected = tuple(channels or (interval.channel,))
        return ZoomAction(selected, start, start + width)

    def _wrong_action(self, meta: RowMeta, episode: SignalEpisode, rng: random.Random) -> ZoomAction:
        requested_width = self._width(meta.length, rng)
        widths = [requested_width]
        if meta.length >= 256:
            # A log-uniform draw can span the whole recording.  If that draw
            # leaves no gold-free window, retry at the minimum legal width;
            # the error trajectory must remain genuinely negative whenever
            # the row has any legal negative window.
            widths.extend([256, min(512, meta.length), min(1024, meta.length)])
        for width in dict.fromkeys(widths):
            try:
                selector_values = torch.where(episode.valid, episode.values, torch.zeros_like(episode.values))
                candidates = blind_candidates(selector_values, width, max(32, len(meta.channel_names) * 8))
            except (TypeError, ValueError):
                candidates = []
            # Candidate rank is deterministic; the random draw changes only
            # the containing window, retaining the W1 anti-centering property.
            for channel_index, centre in candidates:
                if channel_index >= len(meta.channel_names):
                    continue
                start, end = window_containing_point(meta.length, min(meta.length - 1, centre), width, rng)
                action = ZoomAction((meta.channel_names[channel_index],), start, end)
                if not self._overlaps(action, _usable_gold(meta)):
                    return action
        width = widths[-1]
        # Deterministic bounded fallback for a signal whose strongest buckets
        # are all annotated.  This path is rare; compatibility normally avoids
        # such a row, but the teacher still returns a valid action.
        # Include both borders and gold-derived gap boundaries in addition to
        # the coarse scan.  This remains blind to the signal values and makes
        # the structural promise exact even when ``length - width`` is not a
        # multiple of the scan stride.
        for candidate_width in dict.fromkeys(widths):
            candidate_width = min(candidate_width, meta.length)
            starts = {
                0,
                max(0, meta.length - candidate_width),
            }
            stride = max(1, candidate_width // 4)
            starts.update(range(0, max(1, meta.length - candidate_width + 1), stride))
            for item in _usable_gold(meta):
                starts.update({
                    max(0, min(meta.length - candidate_width, item.start - candidate_width + 1)),
                    max(0, min(meta.length - candidate_width, item.end)),
                })
            for start in sorted(starts):
                for channel in meta.channel_names:
                    action = ZoomAction((channel,), start, start + candidate_width)
                    if not self._overlaps(action, _usable_gold(meta)):
                        return action
        channel = meta.channel_names[0]
        return ZoomAction((channel,), 0, width)

    def _correct_action(self, meta: RowMeta, rng: random.Random, *, event: GoldInterval | None = None, width: int | None = None) -> ZoomAction:
        gold = _usable_gold(meta)
        if not gold:
            raise ValueError("a correct zoom requires a gold interval")
        return self._action_for_interval(event or rng.choice(gold), meta, rng, width=width)

    # --------------------------------------------------------------- building
    def _base_turns(self, meta: RowMeta) -> list[Turn]:
        return [
            Turn("system", [ConversationBuilder.system_prompt("agentic", max_calls=self.max_calls, max_views=self.max_views)]),
            Turn("user", [ViewRef(0, "global", meta.question)]),
        ]

    def _append_action(
        self,
        turns: list[Turn],
        actions: list[ZoomAction],
        action: ZoomAction,
        reasoning: str,
        *,
        supervise: bool,
    ) -> None:
        turns.append(Turn("assistant_reasoning_action", [self._cap_reasoning(reasoning), action.to_json()], SEG_ACTION, supervise))
        actions.append(action)
        turns.append(Turn("user", [ViewRef(len(actions), "observation")]))

    def _append_final(self, turns: list[Turn], meta: RowMeta, reasoning: str) -> None:
        turns.append(Turn("assistant_reasoning_final", [self._cap_reasoning(reasoning), meta.answer], SEG_FINAL, True))

    def _resolve_kind(self, meta: RowMeta, requested: str | None) -> str:
        compatible = compatible_kinds(meta)
        def fits_budget(kind: str) -> bool:
            if kind == "final_no_tool":
                return True
            if kind in ("one_zoom_correct", "negative_stop"):
                return self.max_calls >= 1 and self.max_views >= 1
            if kind in ("wrong_then_recover", "refine_boundaries", "multi_event"):
                return self.max_calls >= 2 and self.max_views >= 2
            if kind == "two_channels":
                return self.max_calls >= 1 and self.max_views >= 2
            return False

        budget_compatible = tuple(kind for kind in compatible if fits_budget(kind))
        if requested is not None and requested in budget_compatible:
            return requested
        # Stable fallback priority: retain the requested kind if it can be
        # made meaningful only after a more general route; otherwise use the
        # first compatible kind in quota order.
        for kind in TRAJECTORY_KINDS:
            if kind in budget_compatible:
                return kind
        raise RuntimeError(f"row {meta.id!r} has no compatible trajectory kind")

    def build(
        self,
        meta: RowMeta,
        episode: SignalEpisode,
        rng: random.Random,
        *,
        kind: str | None = None,
    ) -> Trajectory:
        if not isinstance(rng, random.Random) and not hasattr(rng, "choice"):
            raise TypeError("rng must be a random.Random-compatible object")
        if episode.length != meta.length or episode.channel_ids != meta.channel_names:
            raise ValueError("meta and episode do not share the signal contract")
        requested = kind if kind is not None else self.kind
        actual = self._resolve_kind(meta, requested)
        turns = self._base_turns(meta)
        actions: list[ZoomAction] = []
        offsets: list[tuple[int, int]] = []
        gold = _usable_gold(meta)
        injected: ZoomAction | None = None
        injected_turn: int | None = None

        if actual == "final_no_tool":
            reasoning = self._view_reasoning(episode, acquired=(), next_action=None, final=True)
            self._append_final(turns, meta, reasoning)

        elif actual == "one_zoom_correct":
            event = rng.choice(gold)
            action = self._correct_action(meta, rng, event=event)
            reasoning = self._view_reasoning(episode, acquired=(), next_action=action)
            self._append_action(turns, actions, action, reasoning, supervise=True)
            offsets.append((action.start, action.end))
            local_reasoning = self._view_reasoning(episode, acquired=actions, next_action=None, final=True)
            self._append_final(turns, meta, local_reasoning)

        elif actual == "wrong_then_recover":
            wrong = self._wrong_action(meta, episode, rng)
            injected = wrong
            reasoning = self._view_reasoning(episode, acquired=(), next_action=wrong)
            # Both the deliberate mistake and its introductory rationale are
            # retained in context but excluded from CE.
            injected_turn = len(turns)
            self._append_action(turns, actions, wrong, reasoning, supervise=False)
            offsets.append((wrong.start, wrong.end))
            recovery = self._correct_action(meta, rng)
            recovery_reasoning = self._view_reasoning(episode, acquired=actions, next_action=recovery)
            self._append_action(turns, actions, recovery, recovery_reasoning, supervise=True)
            offsets.append((recovery.start, recovery.end))
            final_reasoning = self._view_reasoning(episode, acquired=actions, next_action=None, final=True)
            self._append_final(turns, meta, final_reasoning)

        elif actual == "refine_boundaries":
            event = rng.choice(gold)
            first_width = self._width(meta.length, rng)
            # Make room for a genuinely narrower second view on the synthetic
            # and benchmark length bands where this kind is assigned.
            if first_width <= 256 and meta.length >= 512:
                first_width = min(meta.length, 512)
            first = self._correct_action(meta, rng, event=event, width=first_width)
            reasoning = self._view_reasoning(episode, acquired=(), next_action=first)
            self._append_action(turns, actions, first, reasoning, supervise=True)
            offsets.append((first.start, first.end))
            high = max(256, event.end - event.start)
            second_width = max(256, min(first.width - 1, high))
            if second_width >= first.width:
                second_width = max(1, first.width // 2)
            second = self._correct_action(meta, rng, event=event, width=second_width)
            if second.width >= first.width and meta.length >= 512:
                second = ZoomAction((event.channel,), max(0, min(event.start, meta.length - 256)),
                                    max(0, min(event.start, meta.length - 256)) + 256)
            refine_reasoning = self._view_reasoning(episode, acquired=actions, next_action=second)
            self._append_action(turns, actions, second, refine_reasoning, supervise=True)
            offsets.append((second.start, second.end))
            self._append_final(turns, meta, self._view_reasoning(episode, acquired=actions, next_action=None, final=True))

        elif actual == "two_channels":
            event = rng.choice(gold) if gold else None
            names = list(meta.channel_names)
            base = event.channel if event is not None else names[0]
            other = next(name for name in names if name != base)
            width = self._width(meta.length, rng)
            synchronous = rng.random() < 0.5 or self.max_calls < 2 or self.max_views < 2
            if not synchronous and meta.length > 256 and width >= meta.length:
                width = meta.length - 1
            if synchronous:
                # Synchronous multichannel observation.
                if event is None:
                    start = rng.randint(0, meta.length - width)
                else:
                    start = self._start_for_interval(event, width, meta.length, rng)
                action = ZoomAction((base, other), start, start + width)
                reasoning = self._view_reasoning(episode, acquired=(), next_action=action)
                self._append_action(turns, actions, action, reasoning, supervise=True)
                offsets.append((action.start, action.end))
            else:
                # Two separate reads deliberately use different offsets.  If
                # both channels have annotations, use their own events; for a
                # single-event row the second offset is still a valid negative
                # comparison view, and remains annotation-free in rationale.
                if event is not None:
                    base_action = self._correct_action(meta, rng, event=event, width=width)
                else:
                    start = rng.randint(0, meta.length - width)
                    base_action = ZoomAction((base,), start, start + width)
                if base_action.end > meta.length:
                    base_action = ZoomAction((base,), meta.length - base_action.width, meta.length)
                other_event = next((item for item in gold if item.channel == other), None)
                if other_event is not None:
                    other_action = self._correct_action(meta, rng, event=other_event, width=base_action.width)
                else:
                    if base_action.width < meta.length:
                        delta = min(
                            meta.length - base_action.width,
                            base_action.start + max(1, base_action.width // 3),
                        )
                    else:
                        delta = 0
                    other_action = ZoomAction((other,), delta, delta + base_action.width)
                if other_action.start == base_action.start and meta.length > base_action.width:
                    upper = meta.length - base_action.width
                    shift = other_action.start + 1 if other_action.start < upper else other_action.start - 1
                    shift = max(0, min(upper, shift))
                    other_action = ZoomAction((other,), shift, shift + base_action.width)
                first_reasoning = self._view_reasoning(episode, acquired=(), next_action=base_action)
                self._append_action(turns, actions, base_action, first_reasoning, supervise=True)
                second_reasoning = self._view_reasoning(episode, acquired=actions, next_action=other_action)
                self._append_action(turns, actions, other_action, second_reasoning, supervise=True)
                offsets.extend([(base_action.start, base_action.end), (other_action.start, other_action.end)])
            self._append_final(turns, meta, self._view_reasoning(episode, acquired=actions, next_action=None, final=True))

        elif actual == "negative_stop":
            action = self._wrong_action(meta, episode, rng)
            reasoning = self._view_reasoning(episode, acquired=(), next_action=action)
            self._append_action(turns, actions, action, reasoning, supervise=True)
            offsets.append((action.start, action.end))
            # A locally negative crop is not evidence of global absence; keep
            # this final rationale grounded in the full-resolution global
            # measurements rather than treating the crop as a verdict.
            final_reasoning = self._view_reasoning(
                episode, acquired=actions, next_action=None, final=True, global_only=True
            )
            self._append_final(turns, meta, final_reasoning)

        elif actual == "multi_event":
            events = sorted(gold, key=lambda item: (item.channel, item.start, item.end))
            target_events = events[: min(3, self.max_calls)]
            for event in target_events:
                action = self._correct_action(meta, rng, event=event)
                if action in actions:
                    # A full-recording crop can otherwise make two distinct
                    # annotations serialize to the same action.  Prefer a
                    # narrower view around the event; partial containment is
                    # allowed for this refinement/exploration trajectory.
                    alternatives = [
                        min(action.width - 1, max(256, event.end - event.start + 1)),
                        256,
                    ]
                    for candidate_width in dict.fromkeys(
                        width for width in alternatives if 0 < width < action.width
                    ):
                        candidate = self._action_for_interval(
                            event, meta, rng, width=candidate_width, contain=False
                        )
                        if candidate not in actions:
                            action = candidate
                            break
                reasoning = self._view_reasoning(episode, acquired=actions, next_action=action)
                self._append_action(turns, actions, action, reasoning, supervise=True)
                offsets.append((action.start, action.end))
            self._append_final(turns, meta, self._view_reasoning(episode, acquired=actions, next_action=None, final=True))

        else:  # pragma: no cover - _resolve_kind validates the enum
            raise AssertionError(actual)

        trajectory = Trajectory(
            turns=turns,
            actions=actions,
            kind=actual,
            injected_error=injected is not None,
            injected_error_action=injected,
            injected_error_turn_index=injected_turn,
            offsets=offsets,
            requested_kind=requested,
            metadata={"compatible_kinds": list(compatible_kinds(meta))},
        )
        self._realised[actual] += 1
        return trajectory

    def build_for_kind(self, meta: RowMeta, episode: SignalEpisode, kind: str, rng: random.Random) -> Trajectory:
        return self.build(meta, episode, rng, kind=kind)


__all__ = [
    "TRAJECTORY_KINDS",
    "TRAJECTORY_QUOTAS",
    "Trajectory",
    "TrajectoryKindAssigner",
    "TrajectoryTeacher",
    "compatible_kinds",
]
