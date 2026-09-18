"""WR-Agent-4B-Wildcard-v1: perceptual frontend + mixer + projector + wildcard Qwen3.5 backbone.

One episode is encoded view by view (DESIGN §7.3): the global view first, then every zoom in acquisition
order, each appended to the perceptual memory with immutable ordinals and shared physical time.  The soft
tokens of a view replace the ``<|image_pad|>`` placeholders of its block in the conversation; the backbone
receives per-token modality/time metadata through the shared :class:`WildcardContext`.

Training loss (DESIGN §10.3-10.4): cross-entropy averaged separately per assistant segment and combined
with declared weights, plus the auxiliary event-occupancy BCE on the raw uniform-bin summaries (0.1).
The logits are computed only at supervised positions (memory-efficient loss, as in va1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .data import GoldInterval, RowMeta, ZoomAction, occupancy_targets, parse_zoom_action
from .perceptual_rope import PerceptualMemory, PerceptualMixer, Projector, soft_token_decorrelation
from .reading import episode_reading
from .backbone import BackboneBundle, load_backbone
from .signal_contract import SignalEpisode, ViewKind, compute_recording_normalization, make_view
from .token_supervision import TokenSupervisionHeads, reading_bin_width, token_bin_indices, token_supervision_targets
from .templates import (
    IGNORE_INDEX,
    SEG_ACTION,
    SEG_FINAL,
    SEG_REASONING,
    BuiltSequence,
    ConversationBuilder,
    Turn,
    ViewRef,
    ViewHeader,
)
from .tokenizer import PerceptualFrontend, ViewTokens

SEGMENT_NAMES = {SEG_REASONING: "reasoning", SEG_ACTION: "action", SEG_FINAL: "final"}


@dataclass
class EncodedView:
    kind: str
    channel_ids: tuple[str, ...]
    start: int
    end: int
    soft: Tensor  # [N, 2560]
    tau: Tensor  # [N] float32
    header: ViewHeader
    aux_logits: Tensor  # [C, Q]
    aux_valid: Tensor  # [C, Q] bool
    aux_edges: Tensor  # [Q+1] int64
    n_tokens_per_channel: tuple[int, ...]
    z: Tensor | None = None  # [N, 512] mixer output, valid tokens in sorted order
    token_position_targets: Tensor | None = None  # [N_slots] sorted-slot targets
    token_energy_targets: Tensor | None = None  # [N_slots] sorted-slot targets
    token_peak_targets: Tensor | None = None  # [N_slots] sorted-slot targets
    token_valid: Tensor | None = None  # [N_slots] validity mask in sorted-slot order
    token_sort_order: Tensor | None = None  # [N_slots] original ViewTokens permutation
    token_channel_index: Tensor | None = None  # [N_slots] sorted-slot channels
    # [N_valid] slot of each valid soft token inside ``channel_ids``.  Needed to emit one labelled
    # placeholder block per channel (the ETH/OpenTSLM prompt layout) instead of a single time-ordered
    # block: the soft tokens stay in their (t, ch, fam, lvl, k) order, and the block simply selects its own.
    channel_slot: Tensor | None = None
    token_bin_index: Tensor | None = None  # [N_slots] sorted-slot reading bins
    token_position_bin_scale: Tensor | None = None  # [N_slots] recording samples per reading bin

    @property
    def position_targets(self) -> Tensor | None:
        return self.token_position_targets

    @property
    def energy_targets(self) -> Tensor | None:
        return self.token_energy_targets

    @property
    def peak_targets(self) -> Tensor | None:
        return self.token_peak_targets

    @property
    def valid_mask(self) -> Tensor | None:
        return self.token_valid


@dataclass
class EpisodeEncoding:
    views: list[EncodedView]
    memory: PerceptualMemory
    n_signal_tokens: int


@dataclass
class LossReport:
    loss: Tensor
    ce_by_segment: dict[str, float]
    ce_combined: float
    aux_loss: float
    aux_bins: int
    supervised_tokens: int
    sequence_length: int
    n_signal_tokens: int
    n_views: int
    soft_rms: float = 0.0  # per-dimension RMS of the soft tokens, averaged over views (drift monitor)
    decorrelation: float = 0.0  # mean squared off-diagonal cosine among soft tokens within views
    token_pos_loss: float = 0.0
    token_energy_loss: float = 0.0
    token_rank_loss: float = 0.0
    token_pos_pearson: float = 0.0
    token_energy_balanced_accuracy: float = 0.0
    token_peak_loss: float = 0.0
    token_pos_mae_bins: float = 0.0
    token_energy_pearson: float = 0.0
    token_peak_pearson: float = 0.0
    token_top3_recall: float = 0.0


class WRAgentModel(nn.Module):
    def __init__(self, backbone: BackboneBundle, spec: dict[str, Any], *, max_calls: int = 6, max_views: int = 8,
                 segment_weights: dict[str, float] | None = None, aux_weight: float = 0.1) -> None:
        super().__init__()
        self.bb = backbone
        self.frontend = PerceptualFrontend(spec)
        self.mixer = PerceptualMixer()
        embed = backbone.text.embed_tokens.weight
        target_rms = float(embed.detach().float().pow(2).mean().sqrt())  # text-embedding RMS per dimension
        self.projector = Projector(int(spec["backbone"]["projector_output_dim"]), target_rms=target_rms)
        self._soft_rms: list[float] = []
        token_spec = spec.get("training", {}).get("token_supervision")
        if token_spec is not None and not isinstance(token_spec, Mapping):
            raise TypeError("training.token_supervision must be a mapping")
        if token_spec is not None and bool(token_spec.get("enabled", False)):
            token_spec = dict(token_spec)
            header_bins = int(spec["tokenizer"]["global_per_channel"]["A"])
            token_spec.setdefault("n_bins", header_bins)
            if int(token_spec["n_bins"]) != header_bins:
                raise ValueError("training.token_supervision.n_bins must match the global ViewHeader.reading_bins")
            self.token_supervision: TokenSupervisionHeads | None = TokenSupervisionHeads(
                token_spec,
                soft_dim=int(spec["backbone"]["projector_output_dim"]),
                target_rms=self.projector.target_rms,
            )
            self.token_position_weight = float(token_spec.get("position_weight", 1.0))
            self.token_energy_weight = float(token_spec.get("energy_weight", 1.0))
            self.token_peak_weight = float(token_spec.get("peak_weight", 0.5))
            self.token_rank_weight = float(token_spec.get("rank_weight", 0.2))
        else:
            self.token_supervision = None
            self.token_position_weight = 0.0
            self.token_energy_weight = 0.0
            self.token_peak_weight = 0.0
            self.token_rank_weight = 0.0
        # Astra 15/09 (sentinella D2/D4): decorrelation penalty weight and the supervised reading segment.
        self.decorrelation_weight = float(spec["training"].get("decorrelation_weight", 0.0))
        self.ce_weight = float(spec["training"].get("ce_weight", 1.0))
        if self.ce_weight < 0.0:
            raise ValueError("training.ce_weight must be non-negative")
        self.reading_segment = bool(spec["training"].get("reading_segment", False))
        self.reading_format = str(spec["training"].get("reading_format", "verbose"))
        # Astra 16/09: "reduced" removes min/max from the global header (the measure_magnitude shortcut).
        self.header_stats = str(spec["training"].get("header_stats", "full"))
        # "eth": the AnomalyXL/OpenTSLM prompt verbatim -- one labelled block per channel, zero statistics.
        self.reading_bins = {"global": int(spec["tokenizer"]["global_per_channel"]["A"]),
                             "local": int(spec["tokenizer"]["local_per_channel"]["A"])}
        self.builder = ConversationBuilder(backbone.tokenizer)
        self.max_calls, self.max_views = max_calls, max_views
        self.segment_weights = segment_weights or {"reasoning": 0.2, "action": 1.0, "final": 1.0}
        self.aux_weight = aux_weight
        self.spec = spec

    # ----------------------------------------------------------------- parameters
    @property
    def device(self) -> torch.device:
        return self.projector.proj.weight.device

    def trainable_groups(self) -> dict[str, list[nn.Parameter]]:
        frontend = [p for p in self.frontend.parameters() if p.requires_grad]
        frontend += [p for p in self.mixer.parameters() if p.requires_grad]
        frontend += [p for p in self.projector.parameters() if p.requires_grad]
        if self.token_supervision is not None:
            frontend += [p for p in self.token_supervision.parameters() if p.requires_grad]
        lora = [p for name, p in self.bb.text.named_parameters() if p.requires_grad and "lora_" in name]
        return {"frontend": frontend, "lora": lora}

    def trainable_state_dict(self) -> dict[str, Tensor]:
        state: dict[str, Tensor] = {}
        modules = [("frontend", self.frontend), ("mixer", self.mixer), ("projector", self.projector)]
        if self.token_supervision is not None:
            modules.append(("token_supervision", self.token_supervision))
        for prefix, module in modules:
            for name, value in module.state_dict().items():
                state[f"{prefix}.{name}"] = value.detach().cpu()
        for name, parameter in self.bb.text.named_parameters():
            if "lora_" in name:
                state[f"lora.{name}"] = parameter.detach().cpu()
        return state

    def load_trainable_state_dict(self, state: dict[str, Tensor]) -> None:
        modules = [("frontend", self.frontend), ("mixer", self.mixer), ("projector", self.projector)]
        if self.token_supervision is not None:
            prefix = "token_supervision"
            sub = {k[len(prefix) + 1:]: v for k, v in state.items() if k.startswith(prefix + ".")}
            expected = set(self.token_supervision.state_dict())
            missing = sorted(expected.difference(sub))
            if missing:
                raise RuntimeError(
                    "token_supervision state mismatch: enabled heads are missing checkpoint keys "
                    f"{missing[:3]}"
                )
            modules.append((prefix, self.token_supervision))
        for prefix, module in modules:
            sub = {k[len(prefix) + 1:]: v for k, v in state.items() if k.startswith(prefix + ".")}
            module.load_state_dict(sub, strict=True)
        lora = {k[5:]: v for k, v in state.items() if k.startswith("lora.")}
        params = dict(self.bb.text.named_parameters())
        missing = [k for k in lora if k not in params]
        if missing or len(lora) != len(self.bb.lora_parameter_names):
            raise RuntimeError(f"LoRA state mismatch: missing={missing[:3]} n={len(lora)}")
        with torch.no_grad():
            for name, value in lora.items():
                params[name].copy_(value.to(params[name].device, params[name].dtype))

    # ----------------------------------------------------------------- perception
    def encode_episode(self, episode: SignalEpisode, actions: Sequence[ZoomAction], *,
                       gold: Sequence[GoldInterval] | None = None,
                       header_overrides: dict[int, ViewHeader] | None = None) -> EpisodeEncoding:
        if len(actions) > self.max_calls:
            raise ValueError("too many zoom actions for the declared budget")
        if header_overrides is not None:
            invalid_indices = [
                index for index in header_overrides
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= len(actions)
            ]
            if invalid_indices:
                raise ValueError(f"header override view index out of range: {invalid_indices}")
        stats = compute_recording_normalization(episode)
        dt = episode.dt if episode.dt is not None else 1.0
        t_origin = float(episode.sample_times[0])
        memory = PerceptualMemory(t_origin=t_origin, dt_ref=dt)
        self._soft_rms = []
        views: list[EncodedView] = []
        remaining_calls, remaining_views = self.max_calls, self.max_views

        def encode(view, *, kind: str, observation_index: int) -> EncodedView:
            tokens: ViewTokens = self.frontend.encode_view(view, recording_length=episode.length)
            n_valid = int(tokens.valid.sum().item())
            order = tokens.sort_order[:n_valid]
            if n_valid and not bool(tokens.valid[order].all()):
                raise RuntimeError("sort_order must list valid tokens first")
            h = tokens.descriptors[order]
            anchors = tokens.anchor_time[order]
            channel_index = tokens.channel_index[order]
            mem_view = memory.append(h, anchors, kind=kind, start=view.start, end=view.end,
                                     channel_ids=view.channel_ids)
            h_mem, p_mem, tau_mem = memory.stacked()
            z = self.mixer(h[None], mem_view.p[None], mem_view.tau[None], h_mem[None], p_mem[None], tau_mem[None])[0]
            soft = self.projector(z)
            self._soft_rms.append(float(soft.detach().float().pow(2).mean().sqrt()) if soft.numel() else 0.0)
            counts = tuple(int((channel_index == idx).sum().item()) for idx in view.channel_indices)
            channel_slot = torch.zeros_like(channel_index)
            for slot_of, episode_index in enumerate(view.channel_indices):
                channel_slot[channel_index == episode_index] = slot_of
            stats_view = view.global_stats
            header = ViewHeader(
                kind=kind, channel_ids=view.channel_ids, start=view.start, end=view.end,
                recording_length=episode.length, dt=dt, tokens_total=n_valid, tokens_per_channel=counts,
                median=tuple(stats_view.median.tolist()), scale=tuple(stats_view.scale.tolist()),
                minimum=tuple(stats_view.minimum.tolist()), maximum=tuple(stats_view.maximum.tolist()),
                valid_count=tuple(int(v) for v in view.valid.sum(dim=1).tolist()),
                local_median=None if view.local_stats is None else tuple(view.local_stats.median.tolist()),
                local_scale=None if view.local_stats is None else tuple(view.local_stats.scale.tolist()),
                local_available=None if view.local_stats is None else tuple(view.local_stats.available.tolist()),
                observation_index=observation_index, remaining_calls=remaining_calls,
                remaining_channel_views=remaining_views,
                reading_bins=(getattr(self, "reading_bins", None) or {}).get(
                    "global" if kind == "global" else "local"))
            if header_overrides is not None and observation_index in header_overrides:
                override = header_overrides[observation_index]
                if not isinstance(override, ViewHeader):
                    raise TypeError("header overrides must contain ViewHeader values")
                # A frozen header may change the text, but never the number of
                # placeholders available for the actual perceptual payload.
                if override.tokens_total != n_valid:
                    raise AssertionError(
                        f"header override for view {observation_index} has "
                        f"tokens_total={override.tokens_total}, expected {n_valid}"
                    )
                if tuple(override.tokens_per_channel) != counts:
                    raise AssertionError(
                        f"header override for view {observation_index} has "
                        f"tokens_per_channel={override.tokens_per_channel}, expected {counts}"
                    )
                header = override
            if getattr(self, "ce_weight", 1.0) > 0.0:
                aux_logits = self.frontend.aux_logits(tokens)
            else:
                aux_logits = torch.zeros(
                    tokens.raw_bin_features.shape[:2], dtype=soft.dtype, device=soft.device
                )
            token_position_targets = None
            token_energy_targets = None
            token_peak_targets = None
            token_valid = None
            token_channel_index = None
            token_bin_index = None
            token_position_bin_scale = None
            if getattr(self, "token_supervision", None) is not None:
                n_bins = int(header.reading_bins or self.token_supervision.n_bins)
                (
                    token_position_targets,
                    token_energy_targets,
                    token_peak_targets,
                    token_valid,
                ) = token_supervision_targets(
                    tokens,
                    view.global_normalized,
                    episode.length,
                    n_bins=n_bins,
                    valid_trace=view.valid,
                    channel_indices=view.channel_indices,
                )
                token_channel_index = tokens.channel_index[tokens.sort_order]
                token_bin_index = token_bin_indices(tokens, n_bins, ordered=True)
                bin_width = reading_bin_width(tokens, n_bins)
                token_position_bin_scale = torch.full(
                    (tokens.sort_order.numel(),),
                    float(episode.length) / float(bin_width),
                    dtype=torch.float32,
                    device=soft.device,
                )
            return EncodedView(kind=kind, channel_ids=view.channel_ids, start=view.start, end=view.end,
                               soft=soft, tau=mem_view.tau, header=header, aux_logits=aux_logits,
                               aux_valid=tokens.raw_bin_valid, aux_edges=tokens.raw_bin_edges,
                               n_tokens_per_channel=counts, z=z, channel_slot=channel_slot,
                               token_position_targets=token_position_targets,
                               token_energy_targets=token_energy_targets, token_peak_targets=token_peak_targets,
                               token_valid=token_valid, token_sort_order=tokens.sort_order,
                               token_channel_index=token_channel_index, token_bin_index=token_bin_index,
                               token_position_bin_scale=token_position_bin_scale)

        global_view = make_view(episode, stats, kind=ViewKind.GLOBAL)
        views.append(encode(global_view, kind="global", observation_index=0))
        for index, action in enumerate(actions, start=1):
            remaining_calls -= 1
            remaining_views -= len(action.channels)
            if remaining_calls < 0 or remaining_views < 0:
                raise ValueError("zoom budget exhausted")
            local_view = make_view(episode, stats, kind=ViewKind.LOCAL, start=action.start, end=action.end,
                                   channels=action.channels)
            views.append(encode(local_view, kind="local", observation_index=index))
        return EpisodeEncoding(views=views, memory=memory, n_signal_tokens=sum(v.soft.shape[0] for v in views))

    # ----------------------------------------------------------------- conversation
    def _materialize_turns(
        self,
        meta: RowMeta,
        encoding: EpisodeEncoding,
        turns: Sequence[Turn],
        *,
        oracle_text: str | None = None,
    ) -> list[Turn]:
        """Resolve backbone-free teacher view references to encoded view pieces."""

        materialized: list[Turn] = []
        oracle_added = False
        for turn in turns:
            pieces: list[Any] = []
            for piece in turn.pieces:
                if isinstance(piece, ViewRef):
                    index = int(piece.view_index)
                    if not 0 <= index < len(encoding.views):
                        raise ValueError(f"view reference {index} is outside the encoding")
                    header = encoding.views[index].header
                    if piece.kind == "global":
                        pieces.extend(self.builder.global_pieces(
                            header, piece.question if piece.question is not None else meta.question, index,
                            oracle_text=oracle_text if not oracle_added else None,
                            **self._header_flags(),
                        ))
                        oracle_added = True
                    elif piece.kind in ("observation", "local"):
                        pieces.extend(self.builder.observation_pieces(header, index))
                    else:
                        raise ValueError(f"unknown view reference kind {piece.kind!r}")
                elif (
                    isinstance(piece, (tuple, list)) and len(piece) == 4
                    and piece[0] == "view"
                    and (piece[3] is None or int(piece[3]) < 0)
                ):
                    index, slot = int(piece[1]), int(piece[2])
                    if not 0 <= index < len(encoding.views):
                        raise ValueError(f"view reference {index} is outside the encoding")
                    view = encoding.views[index]
                    if slot < 0:
                        count = int(view.soft.shape[0])
                    else:
                        count = int(view.n_tokens_per_channel[slot])
                    pieces.append(("view", index, slot, count))
                else:
                    pieces.append(piece)
            materialized.append(Turn(
                turn.role,
                pieces,
                turn.segment,
                turn.supervise,
                turn.supervise_reasoning,
                turn.supervise_final,
            ))
        return materialized

    def build_from_turns(
        self,
        meta: RowMeta,
        encoding: EpisodeEncoding,
        turns: Sequence[Turn],
        *,
        generation: bool = False,
        thinking: bool = False,
        header_overrides: dict[int, ViewHeader] | None = None,
        supervise_final: bool | None = None,
        oracle_text: str | None = None,
    ) -> BuiltSequence:
        """Build a sequence from an already planned conversation.

        ``TrajectoryTeacher`` supplies ``ViewRef`` pieces because the number of
        valid perceptual tokens is learned/frontend-dependent.  Resolving them
        here also guarantees that a teacher action list and the encoding use
        exactly the same view insertion order.
        """

        if header_overrides is not None:
            invalid_indices = [
                index for index in header_overrides
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(encoding.views)
            ]
            if invalid_indices:
                raise ValueError(f"header override view index out of range: {invalid_indices}")
            for index, override in header_overrides.items():
                if not isinstance(override, ViewHeader):
                    raise TypeError("header overrides must contain ViewHeader values")
                view = encoding.views[index]
                if override.tokens_total != int(view.soft.shape[0]):
                    raise ValueError("header override has an invalid token count")
                if tuple(override.tokens_per_channel) != tuple(view.n_tokens_per_channel):
                    raise ValueError("header override has invalid per-channel token counts")
                view.header = override
        effective_turns = list(turns)
        if supervise_final is not None:
            effective_turns = [
                replace(turn, supervise_final=bool(supervise_final))
                if turn.role in ("assistant_final", "assistant_reasoning_final")
                else turn
                for turn in effective_turns
            ]
        resolved = self._materialize_turns(meta, encoding, effective_turns, oracle_text=oracle_text)
        return self.builder.build(resolved, generation_prompt=generation, thinking=thinking)

    def build_sequence(
        self,
        meta: RowMeta,
        encoding: EpisodeEncoding,
        turns: Sequence[Turn] | None = None,
        *,
        mode: str = "agentic",
        actions: Sequence[ZoomAction] = (),
        answer: str | None = None,
        supervise_action: bool = True,
        generation: bool = False,
        thinking: bool = False,
        header_overrides: dict[int, ViewHeader] | None = None,
        reading: str | None = None,
        supervise_final: bool = True,
        oracle_text: str | None = None,
    ) -> BuiltSequence:
        if turns is not None:
            return self.build_from_turns(
                meta, encoding, turns, generation=generation, thinking=thinking,
                header_overrides=header_overrides,
                supervise_final=supervise_final,
                oracle_text=oracle_text,
            )

        def header_for(view_index: int) -> ViewHeader:
            header = encoding.views[view_index].header
            if header_overrides is None or view_index not in header_overrides:
                return header
            override = header_overrides[view_index]
            if not isinstance(override, ViewHeader):
                raise TypeError("header overrides must contain ViewHeader values")
            if override.tokens_total != encoding.views[view_index].soft.shape[0]:
                raise AssertionError(
                    f"header override for view {view_index} has an invalid token count"
                )
            if tuple(override.tokens_per_channel) != encoding.views[view_index].n_tokens_per_channel:
                raise AssertionError(
                    f"header override for view {view_index} has invalid per-channel token counts"
                )
            return override

        if header_overrides is not None:
            invalid_indices = [
                index for index in header_overrides
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(encoding.views)
            ]
            if invalid_indices:
                raise ValueError(f"header override view index out of range: {invalid_indices}")
            for index in header_overrides:
                encoding.views[index].header = header_for(index)

        turns = [Turn("system", [self.builder.system_prompt(
            mode, max_calls=self.max_calls, max_views=self.max_views,
            eth_layout=self._header_flags()["eth_layout"])])]
        turns.append(Turn(
            "user",
            self.builder.global_pieces(header_for(0), meta.question, 0, oracle_text=oracle_text,
                                       **self._header_flags()),
        ))
        for index, action in enumerate(actions, start=1):
            turns.append(Turn("assistant_action", [action.to_json()], SEG_ACTION))
            turns.append(Turn("user", self.builder.observation_pieces(header_for(index), index)))
        if not generation:
            if answer is None:
                raise ValueError("a training sequence needs the final answer")
            if reading is not None:
                # Astra 15/09 (sentinella D4): the programmatic reading is the supervised <think> segment.
                turns.append(Turn(
                    "assistant_reasoning_final", [reading, answer], SEG_FINAL,
                    supervise_final=supervise_final,
                ))
            else:
                turns.append(Turn("assistant_final", [answer], SEG_FINAL, supervise_final=supervise_final))
        return self.builder.build(
            turns, supervise_action=supervise_action, generation_prompt=generation, thinking=thinking,
            flat=self._flat_layout(mode)
        )

    def _header_flags(self) -> dict[str, bool]:
        """How the global header is rendered, from ``training.header_stats``.

        full     median, scale, min, max, valid count, token count, reading grid (our own design)
        reduced  the same without min and max (Astra 16/09, opzione C)
        raw      only the token count per channel
        eth      the AnomalyXL/OpenTSLM prompt verbatim: one labelled block per channel, no statistic at all
        """

        mode = getattr(self, "header_stats", "full")
        return {"reduced_stats": mode in ("reduced", "raw"),
                "raw_stats": mode == "raw",
                "eth_layout": mode == "eth"}

    def _flat_layout(self, mode: str) -> bool:
        """The ETH prompt is one flat string: no chat template, no roles, no turn delimiters.

        Only ``direct`` has a flat form -- the agentic and fixed-crop modes ARE conversations.  When P2 starts,
        the chat template comes back and Qwen meets it on the format its own post-training used, so nothing has
        to be taught here that the backbone does not already know.
        """

        return self._header_flags()["eth_layout"] and mode == "direct"

    def _inputs(self, seq: BuiltSequence, encoding: EpisodeEncoding) -> tuple[Tensor, Tensor, Tensor]:
        device = self.device
        ids = seq.input_ids.to(device)
        embeds = self.bb.text.embed_tokens(ids[None]).clone()  # [1, S, 2560]
        tau = torch.zeros(seq.length, dtype=torch.float32, device=device)
        for view_index, slot, positions in seq.view_slots:
            view = encoding.views[view_index]
            positions = positions.to(device)
            # ``slot < 0`` is the whole view in one time-ordered block (our own layout).  A non-negative slot
            # is one channel of the view, in the ETH/OpenTSLM layout where every channel gets its own labelled
            # block; the tokens keep their (t, fam, lvl, k) order inside the block, only the grouping changes.
            if slot < 0:
                soft, taus = view.soft, view.tau
            else:
                if view.channel_slot is None:
                    raise RuntimeError("per-channel blocks need the view's channel_slot")
                selected = (view.channel_slot == slot).nonzero().squeeze(1)
                soft, taus = view.soft[selected], view.tau[selected]
            if positions.numel() != soft.shape[0]:
                raise RuntimeError("placeholder count does not match the view's soft tokens")
            embeds[0, positions] = soft.to(embeds.dtype)
            tau[positions] = taus.to(device)
        is_signal = seq.is_signal.to(device)[None]
        return embeds, is_signal, tau[None]

    # ----------------------------------------------------------------- losses
    def token_supervision_loss(self, encoding: EpisodeEncoding) -> tuple[Tensor, dict[str, float]]:
        """Aggregate per-token supervision over all encoded views."""

        if self.token_supervision is None:
            return torch.zeros((), device=self.device), TokenSupervisionHeads._zero_metrics()
        softs: list[Tensor] = []
        position_targets: list[Tensor] = []
        energy_targets: list[Tensor] = []
        peak_targets: list[Tensor] = []
        channel_indices: list[Tensor] = []
        bin_indices: list[Tensor] = []
        view_indices: list[Tensor] = []
        position_bin_scales: list[Tensor] = []
        for view_index, view in enumerate(encoding.views):
            fields = (
                view.token_position_targets,
                view.token_energy_targets,
                view.token_peak_targets,
                view.token_channel_index,
                view.token_bin_index,
                view.token_position_bin_scale,
            )
            if any(field is None for field in fields):
                raise RuntimeError("encoded view is missing token-supervision tensors")
            n_tokens = int(view.soft.shape[0])
            if n_tokens:
                softs.append(view.soft)
                valid = view.token_valid.to(device=view.soft.device) if view.token_valid is not None else None
                if valid is not None and (valid.ndim != 1 or int(valid[:n_tokens].sum()) != n_tokens):
                    raise RuntimeError("token supervision mask is not aligned with encoded tokens")
                position_targets.append(view.token_position_targets[:n_tokens].to(view.soft.device))
                energy_targets.append(view.token_energy_targets[:n_tokens].to(view.soft.device))
                peak_targets.append(view.token_peak_targets[:n_tokens].to(view.soft.device))
                channel_indices.append(view.token_channel_index[:n_tokens].to(view.soft.device))
                bin_indices.append(view.token_bin_index[:n_tokens].to(view.soft.device))
                position_bin_scales.append(view.token_position_bin_scale[:n_tokens].to(view.soft.device))
                view_indices.append(torch.full((n_tokens,), view_index, dtype=torch.int64, device=view.soft.device))
        if not softs:
            return torch.zeros((), device=self.device), TokenSupervisionHeads._zero_metrics()
        return self.token_supervision.loss(
            torch.cat(softs, dim=0),
            torch.cat(position_targets, dim=0),
            torch.cat(energy_targets, dim=0),
            torch.cat(peak_targets, dim=0),
            channel_index=torch.cat(channel_indices, dim=0),
            bin_index=torch.cat(bin_indices, dim=0),
            view_index=torch.cat(view_indices, dim=0),
            position_bin_scale=torch.cat(position_bin_scales, dim=0),
            position_weight=self.token_position_weight,
            energy_weight=self.token_energy_weight,
            peak_weight=self.token_peak_weight,
            rank_weight=self.token_rank_weight,
        )

    def forward_loss(
        self,
        meta: RowMeta,
        episode: SignalEpisode,
        *,
        mode: str = "agentic",
        actions: Sequence[ZoomAction] = (),
        answer: str | None = None,
        gold: Sequence[GoldInterval] = (),
        supervise_action: bool = True,
        turns: Sequence[Turn] | None = None,
        reading: str | None = None,
        header_overrides: dict[int, ViewHeader] | None = None,
        supervise_final: bool = True,
        oracle_text: str | None = None,
    ) -> LossReport:
        if header_overrides is None:
            encoding = self.encode_episode(episode, actions, gold=gold)
        else:
            encoding = self.encode_episode(
                episode,
                actions,
                gold=gold,
                header_overrides=header_overrides,
            )
        decor = (self.soft_token_decorrelation(encoding) if self.decorrelation_weight > 0
                 else torch.zeros((), device=self.device))
        if self.token_supervision is None:
            token_loss = torch.zeros((), device=self.device)
            token_metrics = TokenSupervisionHeads._zero_metrics()
        else:
            token_loss, token_metrics = self.token_supervision_loss(encoding)

        # P1-e O1: perceptual losses must not build or execute a backbone sequence.
        if getattr(self, "ce_weight", 1.0) == 0.0:
            loss = token_loss + self.decorrelation_weight * decor
            return LossReport(
                loss=loss,
                ce_by_segment={},
                ce_combined=0.0,
                aux_loss=0.0,
                aux_bins=0,
                supervised_tokens=0,
                sequence_length=0,
                n_signal_tokens=encoding.n_signal_tokens,
                n_views=len(encoding.views),
                soft_rms=(sum(self._soft_rms) / len(self._soft_rms)) if self._soft_rms else 0.0,
                decorrelation=float(decor.detach()),
                token_pos_loss=token_metrics["position_loss"],
                token_energy_loss=token_metrics["energy_loss"],
                token_rank_loss=token_metrics["rank_loss"],
                token_pos_pearson=token_metrics["position_pearson"],
                token_energy_balanced_accuracy=token_metrics["energy_balanced_accuracy"],
                token_peak_loss=token_metrics["peak_loss"],
                token_pos_mae_bins=token_metrics["position_mae_bins"],
                token_energy_pearson=token_metrics["energy_pearson"],
                token_peak_pearson=token_metrics["peak_pearson"],
                token_top3_recall=token_metrics["top3_recall"],
            )
        if turns is None:
            if answer is None:
                raise ValueError("a training sequence needs the final answer")
            seq = self.build_sequence(
                meta, encoding, mode=mode, actions=actions, answer=answer,
                supervise_action=supervise_action, reading=reading,
                header_overrides=header_overrides, supervise_final=supervise_final,
                oracle_text=oracle_text,
            )
        else:
            seq = self.build_from_turns(
                meta,
                encoding,
                turns,
                generation=False,
                thinking=True,
                header_overrides=header_overrides,
                supervise_final=supervise_final,
            )
        embeds, is_signal, tau = self._inputs(seq, encoding)
        self.bb.ctx.set_tokens(is_signal, tau)
        hidden = self.bb.text(inputs_embeds=embeds, use_cache=False).last_hidden_state[0]  # [S, 2560]

        labels = seq.labels.to(self.device)
        segments = seq.segment.to(self.device)
        supervised = torch.nonzero(labels != IGNORE_INDEX).squeeze(1)
        if (
            (not supervised.numel() and supervise_final)
            or (supervised.numel() and int(supervised.min()) == 0)
        ):
            raise RuntimeError("no supervised position (or supervised at position 0)")
        if supervised.numel():
            logits = self.bb.lm_head(hidden[supervised - 1]).float()
            token_ce = F.cross_entropy(logits, labels[supervised], reduction="none")
            seg_ids = segments[supervised]
        else:
            token_ce = torch.empty(0, dtype=torch.float32, device=hidden.device)
            seg_ids = torch.empty(0, dtype=torch.int64, device=hidden.device)
        ce_by_segment: dict[str, float] = {}
        combined = torch.zeros((), device=self.device)
        for seg_id, name in SEGMENT_NAMES.items():
            mask = seg_ids == seg_id
            if bool(mask.any()):
                mean = token_ce[mask].mean()
                ce_by_segment[name] = float(mean.detach())
                combined = combined + float(self.segment_weights[name]) * mean
        aux_terms: list[Tensor] = []
        aux_bins = 0
        for view in encoding.views:
            targets = torch.stack([
                occupancy_targets(gold, channel, view.aux_edges.cpu()) for channel in view.channel_ids
            ]).to(self.device)
            valid = view.aux_valid
            if bool(valid.any()):
                bce = F.binary_cross_entropy_with_logits(view.aux_logits.float()[valid], targets[valid])
                aux_terms.append(bce)
                aux_bins += int(valid.sum())
        aux = torch.stack(aux_terms).mean() if aux_terms else torch.zeros((), device=self.device)
        loss = self.ce_weight * combined + self.aux_weight * aux + self.decorrelation_weight * decor + token_loss
        return LossReport(loss=loss, ce_by_segment=ce_by_segment, ce_combined=float(combined.detach()),
                          aux_loss=float(aux.detach()), aux_bins=aux_bins, supervised_tokens=int(supervised.numel()),
                          sequence_length=seq.length, n_signal_tokens=seq.n_signal_tokens, n_views=len(encoding.views),
                          soft_rms=(sum(self._soft_rms) / len(self._soft_rms)) if self._soft_rms else 0.0,
                          decorrelation=float(decor.detach()),
                          token_pos_loss=token_metrics["position_loss"],
                          token_energy_loss=token_metrics["energy_loss"],
                          token_rank_loss=token_metrics["rank_loss"],
                          token_pos_pearson=token_metrics["position_pearson"],
                          token_energy_balanced_accuracy=token_metrics["energy_balanced_accuracy"],
                          token_peak_loss=token_metrics["peak_loss"],
                          token_pos_mae_bins=token_metrics["position_mae_bins"],
                          token_energy_pearson=token_metrics["energy_pearson"],
                          token_peak_pearson=token_metrics["peak_pearson"],
                          token_top3_recall=token_metrics["top3_recall"])

    def soft_token_decorrelation(self, encoding: EpisodeEncoding) -> Tensor:
        """Astra 15/09 (sentinella D2): mean squared off-diagonal cosine among the soft tokens of each view."""

        return soft_token_decorrelation([view.soft for view in encoding.views]).to(self.device)

    def reading_text(self, episode: SignalEpisode, actions: Sequence[ZoomAction] = ()) -> str:
        """Programmatic reading of the LAST acquired view at its token resolution (Astra 15/09, sentinella D4)."""

        if actions:
            last = actions[-1]
            return episode_reading(episode, channels=last.channels, start=last.start, end=last.end,
                                   n_bins=self.reading_bins["local"], kind="local", fmt=self.reading_format)
        return episode_reading(episode, n_bins=self.reading_bins["global"], kind="global", fmt=self.reading_format)

    # ----------------------------------------------------------------- generation
    def _forced_closure_ids(self) -> list[int]:
        """Ids of the closure the generator forces when the reading hits its budget."""

        return [int(t) for t in self.bb.tokenizer("\n</think>\n\n", add_special_tokens=False).input_ids]

    def _closure_tail_ids(self) -> list[int]:
        """Ids that follow a naturally generated closing think tag in the training format."""

        return [int(t) for t in self.bb.tokenizer("\n\n", add_special_tokens=False).input_ids]

    @torch.no_grad()
    def generate(self, meta: RowMeta, episode: SignalEpisode, *, mode: str, actions: Sequence[ZoomAction] = (),
                 max_new_tokens: int = 768, use_cache: bool = True,
                 header_overrides: dict[int, ViewHeader] | None = None,
                 thinking: bool | None = None, max_reasoning_tokens: int = 384,
                 oracle_text: str | None = None) -> dict[str, Any]:
        """Direct / fixed-crop generation.  With ``thinking`` (default: the spec's ``reading_segment``) the model first
        writes ITS OWN reading inside ``<think>`` (cut at ``max_reasoning_tokens`` with a forced closure), then the
        final answer with ``max_new_tokens`` (Astra 15/09, sentinella D4: the reading is generated in every
        condition, never supplied).  ``text`` is the content after ``</think>``; ``reading`` the generated reading."""

        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

        thinking = self.reading_segment if thinking is None else bool(thinking)
        encoding = self.encode_episode(episode, actions, header_overrides=header_overrides)
        seq = self.build_sequence(meta, encoding, mode=mode, actions=actions, answer=None,
                                  supervise_action=False, generation=True, thinking=thinking,
                                  header_overrides=header_overrides, oracle_text=oracle_text)
        embeds, is_signal, tau = self._inputs(seq, encoding)
        eos = {self.builder.im_end_id, self.bb.tokenizer.eos_token_id}
        think_close = int(self.bb.tokenizer.convert_tokens_to_ids("</think>"))
        text = self.bb.text
        cache = Qwen3_5DynamicCache(config=text.config) if use_cache else None
        state = {"embeds": embeds, "signal": is_signal, "tau": tau, "position": seq.length}

        def first_step() -> int:
            self.bb.ctx.set_tokens(is_signal, tau)
            if use_cache:
                out = text(inputs_embeds=embeds, past_key_values=cache, use_cache=True,
                           cache_position=torch.arange(seq.length, device=self.device))
            else:
                out = text(inputs_embeds=embeds, use_cache=False)
            return int(self.bb.lm_head(out.last_hidden_state[:, -1]).argmax(-1))

        def feed(token_id: int) -> int:
            step = text.embed_tokens(torch.tensor([[token_id]], device=self.device))
            zero_signal = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
            zero_tau = torch.zeros(1, 1, device=self.device)
            if use_cache:
                self.bb.ctx.set_tokens(zero_signal, zero_tau)
                out = text(inputs_embeds=step, past_key_values=cache, use_cache=True,
                           cache_position=torch.tensor([state["position"]], device=self.device))
                state["position"] += 1
            else:
                state["embeds"] = torch.cat([state["embeds"], step], dim=1)
                state["signal"] = torch.cat([state["signal"], zero_signal], 1)
                state["tau"] = torch.cat([state["tau"], zero_tau], 1)
                self.bb.ctx.set_tokens(state["signal"], state["tau"])
                out = text(inputs_embeds=state["embeds"], use_cache=False)
            return int(self.bb.lm_head(out.last_hidden_state[:, -1]).argmax(-1))

        reasoning: list[int] = []
        final: list[int] = []
        closure: list[int] = []  # ids of the think closure actually used (Astra 16/09: replayed verbatim)
        forced_closure = False
        finished = False
        next_id = first_step()
        if thinking:
            while True:
                if next_id == think_close:
                    closure = [think_close] + self._closure_tail_ids()
                    next_id = feed(next_id)
                    break
                if next_id in eos:
                    finished = True  # the turn closed inside the reading: no final answer
                    closure = self._forced_closure_ids()
                    break
                if len(reasoning) >= max_reasoning_tokens:
                    forced_closure = True
                    closure = self._forced_closure_ids()
                    for token_id in self.bb.tokenizer("\n</think>\n\n", add_special_tokens=False).input_ids:
                        next_id = feed(int(token_id))
                    break
                reasoning.append(next_id)
                next_id = feed(next_id)
        if not finished:
            for _ in range(max_new_tokens):
                final.append(next_id)
                if next_id in eos:
                    finished = True
                    break
                next_id = feed(next_id)
        content = [t for t in final if t not in eos]
        return {"text": self.bb.tokenizer.decode(content, skip_special_tokens=False).strip("\n"),
                "reading": self.bb.tokenizer.decode(reasoning, skip_special_tokens=False) if reasoning else "",
                "reasoning_ids": list(reasoning) + list(closure),
                "token_ids": reasoning + final, "reasoning_tokens": len(reasoning), "final_tokens": len(content),
                "forced_closure": forced_closure, "thinking": thinking,
                "prompt_length": seq.length, "n_signal_tokens": seq.n_signal_tokens, "finished": finished}

    # ------------------------------------------------------------ agentic loop
    def _count_generated_tokens(self, text: str) -> int:
        """Count generated text conservatively when a tokenizer is unavailable."""

        tokenizer = getattr(self, "bb", None)
        tokenizer = getattr(tokenizer, "tokenizer", None)
        if tokenizer is not None:
            try:
                encoded = tokenizer(text, add_special_tokens=False)
                ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
                return len(ids)
            except Exception:  # pragma: no cover - fallback is intentionally simple
                pass
        return max(1, int(math.ceil(len(text.split()) * 1.5))) if text.strip() else 0

    def _truncate_generated_text(self, text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        if self._count_generated_tokens(text) <= limit:
            return text
        tokenizer = getattr(getattr(self, "bb", None), "tokenizer", None)
        if tokenizer is not None:
            try:
                encoded = tokenizer(text, add_special_tokens=False)
                ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
                return tokenizer.decode(ids[:limit], skip_special_tokens=False)
            except Exception:  # pragma: no cover
                pass
        words = text.split()
        return " ".join(words[: max(1, int(limit / 1.5))])

    @staticmethod
    def _split_agentic_text(text: str) -> tuple[str, str, bool]:
        """Split one assistant completion at the actual think closure."""

        raw = str(text or "")
        if "<|im_start|>assistant" in raw:
            raw = raw.split("<|im_start|>assistant", 1)[1]
        if "<think>" in raw:
            raw = raw.split("<think>", 1)[1]
        closed = "</think>" in raw
        if closed:
            reasoning, content = raw.split("</think>", 1)
        else:
            # Thinking-on decoding is only parseable after the literal
            # closure.  Preserve the unfinished reasoning in the returned
            # turn, but do not mistake it for a final answer or a tool call.
            reasoning, content = raw, ""
        content = content.replace("<|im_end|>", "").strip()
        return reasoning.strip(), content, closed

    @staticmethod
    def _looks_like_tool_payload(content: str) -> bool:
        lowered = content.lower()
        stripped = content.lstrip()
        return (
            '"tool"' in lowered
            or "'tool'" in lowered
            or (stripped.startswith("{") and ("zoom" in lowered or "channels" in lowered))
        )

    @staticmethod
    def _normalise_decode_result(result: Any) -> dict[str, Any]:
        if isinstance(result, Mapping):
            text = result.get("text", result.get("content", ""))
            return {
                "text": str(text or ""),
                "token_ids": list(result.get("token_ids", result.get("ids", [])) or []),
                "finished": bool(result.get("finished", True)),
            }
        if isinstance(result, tuple) and result:
            return {"text": str(result[0] or ""), "token_ids": [], "finished": True}
        return {"text": str(result or ""), "token_ids": [], "finished": True}

    def _invoke_agentic_decoder(self, decoder: Callable[..., Any], **available: Any) -> Any:
        """Call a test decoder without imposing one brittle callback signature."""

        try:
            signature = inspect.signature(decoder)
        except (TypeError, ValueError):  # pragma: no cover - unusual C callables
            return decoder(available["turns"])
        parameters = signature.parameters
        aliases = {
            "sequence": "prompt",
            "built_sequence": "prompt",
            "seq": "prompt",
            "max_new_tokens": "max_tokens",
            "budget": "max_tokens",
            "available_tokens": "max_tokens",
            "remaining_tokens": "max_tokens",
            "allow_tool": "tool_available",
            "tools_available": "tool_available",
            "can_use_tool": "tool_available",
            "history": "turns",
        }
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            kwargs = dict(available)
            for name, parameter in parameters.items():
                source = aliases.get(name, name)
                if name not in kwargs and source in available and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
                    kwargs[name] = available[source]
            return decoder(**kwargs)
        kwargs: dict[str, Any] = {}
        positional: list[Any] = []
        for name, parameter in parameters.items():
            source = aliases.get(name, name)
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                if source in available:
                    positional.append(available[source])
            elif source in available:
                kwargs[name] = available[source]
        return decoder(*positional, **kwargs)

    def _new_agentic_cache(self) -> Any:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

        return Qwen3_5DynamicCache(config=self.bb.text.config)

    def _prime_agentic_cache(self, seq: BuiltSequence, encoding: EpisodeEncoding, cache: Any, position: int = 0) -> dict[str, Any]:
        embeds, is_signal, tau = self._inputs(seq, encoding)
        self.bb.ctx.set_tokens(is_signal, tau)
        out = self.bb.text(
            inputs_embeds=embeds,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(position, position + seq.length, device=self.device),
        )
        next_id = int(self.bb.lm_head(out.last_hidden_state[:, -1]).argmax(-1))
        return {"cache": cache, "position": position + seq.length, "next_id": next_id}

    def _append_agentic_turn_to_cache(
        self,
        meta: RowMeta,
        encoding: EpisodeEncoding,
        turn: Turn,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one user turn plus the thinking-on assistant opener."""

        seq = self.build_from_turns(meta, encoding, [turn], generation=True, thinking=True)
        return self._prime_agentic_cache(seq, encoding, state["cache"], int(state["position"]))

    def _decode_agentic_turn_from_cache(
        self,
        state: dict[str, Any],
        *,
        max_new_tokens: int,
        max_reasoning_tokens: int = 384,
    ) -> dict[str, Any]:
        """Greedy cached decoding of one reasoning/action-or-final turn."""

        if max_new_tokens <= 0:
            return {"text": "", "token_ids": [], "finished": False, "state": state}
        eos = {self.builder.im_end_id, self.bb.tokenizer.eos_token_id}
        think_end = self.bb.tokenizer.convert_tokens_to_ids("</think>")
        generated: list[int] = []
        next_id = int(state["next_id"])
        cache = state["cache"]
        position = int(state["position"])
        reasoning_body_tokens = 0
        closed = False
        for _ in range(int(max_new_tokens)):
            # Reserve one generated token for the literal think closure.  If
            # the backbone keeps reasoning past the per-decision limit, force
            # that delimiter at the boundary so the cache and the serialized
            # history remain in the same conversation state.  The normal
            # callback/fake-decoder path applies the equivalent text cap after
            # splitting.
            if (
                not closed
                and max_reasoning_tokens > 0
                and next_id not in eos
                and next_id != think_end
                and reasoning_body_tokens >= max(0, int(max_reasoning_tokens) - 1)
            ):
                next_id = int(think_end)
            generated.append(next_id)
            if next_id in eos:
                # The sampled end token was not part of the cache that
                # produced its logits.  Insert it before a later user
                # observation so the cache contains the literal assistant
                # turn terminator (including the template's trailing
                # newline) as well as the decoded text.
                newline = self.bb.tokenizer("\n", add_special_tokens=False)
                newline_ids = newline.input_ids if hasattr(newline, "input_ids") else newline["input_ids"]
                for token_id in [next_id, *[int(item) for item in newline_ids]]:
                    step = self.bb.text.embed_tokens(torch.tensor([[token_id]], device=self.device))
                    self.bb.ctx.set_tokens(
                        torch.zeros(1, 1, dtype=torch.bool, device=self.device),
                        torch.zeros(1, 1, dtype=torch.float32, device=self.device),
                    )
                    out = self.bb.text(
                        inputs_embeds=step,
                        past_key_values=cache,
                        use_cache=True,
                        cache_position=torch.tensor([position], device=self.device),
                    )
                    next_id = int(self.bb.lm_head(out.last_hidden_state[:, -1]).argmax(-1))
                    position += 1
                break
            if next_id == think_end:
                closed = True
            elif not closed:
                reasoning_body_tokens += 1
            step = self.bb.text.embed_tokens(torch.tensor([[next_id]], device=self.device))
            self.bb.ctx.set_tokens(
                torch.zeros(1, 1, dtype=torch.bool, device=self.device),
                torch.zeros(1, 1, dtype=torch.float32, device=self.device),
            )
            out = self.bb.text(
                inputs_embeds=step,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([position], device=self.device),
            )
            next_id = int(self.bb.lm_head(out.last_hidden_state[:, -1]).argmax(-1))
            position += 1
        content_ids = [token for token in generated if token not in eos]
        text = self.bb.tokenizer.decode(content_ids, skip_special_tokens=False)
        return {
            "text": text,
            "token_ids": generated,
            "finished": bool(generated and generated[-1] in eos),
            "state": {"cache": cache, "position": position, "next_id": next_id},
        }

    def _agentic_turn_prompt(
        self,
        meta: RowMeta,
        encoding: EpisodeEncoding,
        turns: Sequence[Turn],
    ) -> BuiltSequence | None:
        try:
            return self.build_from_turns(meta, encoding, turns, generation=True, thinking=True)
        except (AttributeError, TypeError):
            # A CPU fake may intentionally provide only a decoder and a tiny
            # encoding object.  It still receives the ordered Turn list.
            return None
        except ValueError:
            # A real run must fail closed on the declared text/context budget;
            # only the backbone-free fake path is allowed to omit a full
            # sequence object.
            if hasattr(self, "bb"):
                raise
            return None

    @torch.no_grad()
    def generate_agentic(
        self,
        meta: RowMeta,
        episode: SignalEpisode,
        *,
        max_calls: int | None = None,
        max_views: int | None = None,
        max_assistant_tokens: int = 4096,
        final_reserve: int = 768,
        max_reasoning_tokens: int = 384,
        decode: Callable[..., Any] | None = None,
        decode_step: Callable[..., Any] | None = None,
        generator: Callable[..., Any] | None = None,
        decode_fn: Callable[..., Any] | None = None,
        decoder: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Run the causal multi-turn zoom protocol with an injectable decoder.

        ``decode``/``decode_step``/``generator`` are deliberately equivalent
        hooks for CPU protocol tests.  Without a hook the pinned Qwen backbone
        is decoded greedily with one dynamic KV cache; every new observation is
        appended to that same cache before the next assistant decision.
        """

        hooks = (decode, decode_step, generator, decode_fn, decoder)
        if sum(item is not None for item in hooks) > 1:
            raise ValueError("provide at most one agentic decoder hook")
        decoder = decode or decode_step or generator or decode_fn or decoder
        if decoder is None:
            decoder = getattr(self, "agentic_decoder", None)
        if decoder is None:
            decoder = getattr(self, "_decode_agentic_turn", None)
        max_calls = self.max_calls if max_calls is None else int(max_calls)
        max_views = self.max_views if max_views is None else int(max_views)
        for name, value in (
            ("max_calls", max_calls), ("max_views", max_views),
            ("max_assistant_tokens", max_assistant_tokens),
            ("final_reserve", final_reserve), ("max_reasoning_tokens", max_reasoning_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if max_calls > self.max_calls or max_views > self.max_views:
            raise ValueError("agentic budgets exceed the model declaration")
        declared_assistant = int(getattr(self.builder, "max_assistant", 4096))
        if max_assistant_tokens > declared_assistant:
            raise ValueError("max_assistant_tokens exceeds the model declaration")
        if final_reserve > max_assistant_tokens:
            raise ValueError("final_reserve must fit inside max_assistant_tokens")

        # Only the global observation is encoded before the first decision.
        encoding = self.encode_episode(episode, ())
        global_header = replace(
            encoding.views[0].header,
            remaining_calls=max_calls,
            remaining_channel_views=max_views,
        )
        turns: list[Turn] = [
            Turn("system", [self.builder.system_prompt("agentic", max_calls=max_calls, max_views=max_views)]),
            Turn("user", self.builder.global_pieces(global_header, meta.question, 0, **self._header_flags())),
        ]
        initial_prompt = self._agentic_turn_prompt(meta, encoding, turns)

        calls = 0
        local_views = 0
        used_assistant = 0
        prompt_tokens = int(initial_prompt.n_text_tokens) if initial_prompt is not None else 0
        soft_tokens = int(encoding.n_signal_tokens)
        reasoning_tokens = 0
        action_tokens = 0
        final_tokens = 0
        executed_actions: list[ZoomAction] = []
        invalid_actions: list[dict[str, Any]] = []
        duplicate_actions: list[dict[str, Any]] = []
        generated_token_ids: list[int] = []
        final_text = ""
        finished = False
        existing_observations: dict[tuple[tuple[str, ...], int, int], int] = {}

        def action_key(action: ZoomAction) -> tuple[tuple[str, ...], int, int]:
            # Channel order has no meaning in the shared-window tool contract;
            # a reversed channel list is still the same already-executed view.
            return tuple(sorted(action.channels)), action.start, action.end
        disabled_turn_added = False

        def append_user_turn(turn: Turn, *, count_prompt: bool = True) -> None:
            nonlocal prompt_tokens
            turns.append(turn)
            if count_prompt:
                try:
                    prompt_tokens += self.builder.build([turn]).n_text_tokens
                except (AttributeError, TypeError, ValueError):
                    prompt_tokens += self._count_generated_tokens("".join(str(p) for p in turn.pieces if isinstance(p, str)))

        # A separate cache state keeps the callback path independent of
        # heavyweight Qwen objects.
        cache_state: dict[str, Any] | None = None
        if decoder is None:
            cache_state = self._prime_agentic_cache(initial_prompt, encoding, self._new_agentic_cache()) if initial_prompt is not None else None

        def append_context(turn: Turn) -> None:
            nonlocal cache_state
            if decoder is None and cache_state is not None:
                cache_state = self._append_agentic_turn_to_cache(meta, encoding, turn, cache_state)

        def append_tool_unavailable(reason: str) -> None:
            nonlocal disabled_turn_added
            if disabled_turn_added:
                return
            turn = Turn("user", [
                f"Tool no longer available: {reason}. Provide the final answer from the observations already acquired."
            ])
            append_user_turn(turn)
            append_context(turn)
            disabled_turn_added = True

        def remaining_tool_available() -> bool:
            return (
                not disabled_turn_added
                and calls < max_calls
                and local_views < max_views
                and max_assistant_tokens - used_assistant >= final_reserve
            )

        # The initial reserve check is observable in the conversation and is
        # important for very small CPU budget tests.
        if not remaining_tool_available():
            append_tool_unavailable("the final token reserve is required")

        # A valid run has at most one final turn plus bounded tool attempts.
        # The extra allowance handles a malformed completion followed by a
        # valid final without permitting an unbounded fake decoder loop.
        max_decisions = max(2, 2 * (max_calls + 1) + 2)
        for _decision_index in range(max_decisions):
            if not remaining_tool_available() and not disabled_turn_added:
                append_tool_unavailable("the call/view budget or final token reserve is exhausted")
            available_tokens = max(0, max_assistant_tokens - used_assistant)
            prompt = self._agentic_turn_prompt(meta, encoding, turns)
            callback_kwargs = {
                "model": self,
                "meta": meta,
                "episode": episode,
                "turns": list(turns),
                "encoding": encoding,
                "prompt": prompt,
                "sequence": prompt,
                "max_tokens": available_tokens,
                "max_new_tokens": available_tokens,
                "max_reasoning_tokens": max_reasoning_tokens,
                "final_reserve": final_reserve,
                "calls": calls,
                "local_channel_views": local_views,
                "tool_available": remaining_tool_available(),
                "cache": cache_state,
                "state": cache_state,
                "decision_index": _decision_index,
            }
            if decoder is not None:
                raw_result = self._invoke_agentic_decoder(decoder, **callback_kwargs)
            elif cache_state is not None:
                raw_result = self._decode_agentic_turn_from_cache(
                    cache_state,
                    max_new_tokens=available_tokens,
                    max_reasoning_tokens=max_reasoning_tokens,
                )
                cache_state = raw_result.pop("state")
            else:
                raw_result = {"text": "", "finished": False, "token_ids": []}
            result = self._normalise_decode_result(raw_result)
            generated_token_ids.extend(int(item) for item in result.get("token_ids", []) if isinstance(item, int))
            reasoning, content, closed = self._split_agentic_text(result["text"])

            closure_tokens = self._count_generated_tokens("\n</think>\n\n") if closed else 0
            body_limit = max(0, max_reasoning_tokens - closure_tokens)
            reasoning = self._truncate_generated_text(reasoning, body_limit)
            reasoning_n = self._count_generated_tokens(reasoning) + (closure_tokens if closed else 0)
            if reasoning_n > available_tokens:
                # A fake decoder may ignore the advertised budget, and a
                # real decoder can reach the global limit while still in its
                # think block.  Preserve the closure only when it fits and
                # never let bookkeeping exceed the total assistant budget.
                closure_fit = closure_tokens if closed and closure_tokens <= available_tokens else 0
                reasoning = self._truncate_generated_text(
                    reasoning, max(0, available_tokens - closure_fit)
                )
                reasoning_n = self._count_generated_tokens(reasoning) + closure_fit
                if reasoning_n > available_tokens:
                    reasoning = ""
                    reasoning_n = 0

            parsed = parse_zoom_action(
                content, channel_names=episode.channel_ids, length=episode.length
            )
            is_tool_attempt = parsed is not None or self._looks_like_tool_payload(content)

            if is_tool_attempt:
                # A generated tool turn is kept in the history even when it is
                # rejected; the runner must not silently repair its text.
                original_content = content
                action_n = self._count_generated_tokens(content) + self._count_generated_tokens("<|im_end|>")
                total_turn_tokens = reasoning_n + action_n
                if total_turn_tokens > available_tokens:
                    action_n_limit = max(0, available_tokens - reasoning_n)
                    content = self._truncate_generated_text(content, action_n_limit)
                    action_n = self._count_generated_tokens(content) + self._count_generated_tokens("<|im_end|>") if content else 0
                    total_turn_tokens = reasoning_n + action_n
                    if content != original_content:
                        reparsed = parse_zoom_action(
                            content, channel_names=episode.channel_ids, length=episode.length
                        )
                        if reparsed != parsed:
                            parsed = None
                assistant_turn = Turn(
                    "assistant_reasoning_action", [reasoning, content], SEG_ACTION, True
                )
                turns.append(assistant_turn)
                used_assistant += total_turn_tokens
                reasoning_tokens += reasoning_n
                action_tokens += action_n

                if not remaining_tool_available():
                    invalid_actions.append({
                        "text": content, "reason": "tool unavailable or final reserve required",
                        "turn_index": len(turns) - 1,
                    })
                    append_tool_unavailable("the call/view budget or final reserve is exhausted")
                    continue

                if parsed is None:
                    calls += 1
                    record = {
                        "text": content,
                        "reason": "invalid zoom action",
                        "turn_index": len(turns) - 1,
                        "calls": calls,
                    }
                    invalid_actions.append(record)
                    error = Turn("user", self.builder.error_observation_pieces(
                        "invalid zoom action",
                        remaining_calls=max(0, max_calls - calls),
                        remaining_channel_views=max(0, max_views - local_views),
                    ))
                    append_user_turn(error)
                    append_context(error)
                    if calls >= max_calls:
                        append_tool_unavailable("no zoom calls remain")
                    continue

                parsed_key = action_key(parsed)
                if parsed_key in existing_observations:
                    calls += 1
                    observation_index = existing_observations[parsed_key]
                    record = {
                        "text": content,
                        "action": parsed,
                        "reason": "duplicate zoom action",
                        "existing_observation": observation_index,
                        "observation_index": observation_index,
                        "turn_index": len(turns) - 1,
                        "calls": calls,
                    }
                    duplicate_actions.append(record)
                    error = Turn("user", self.builder.error_observation_pieces(
                        "duplicate zoom action",
                        remaining_calls=max(0, max_calls - calls),
                        remaining_channel_views=max(0, max_views - local_views),
                        duplicate_of=observation_index,
                    ))
                    append_user_turn(error)
                    append_context(error)
                    if calls >= max_calls:
                        append_tool_unavailable("no zoom calls remain")
                    continue

                if local_views + len(parsed.channels) > max_views:
                    calls += 1
                    reason = "channel-view budget exceeded"
                    invalid_actions.append({
                        "text": content, "action": parsed, "reason": reason,
                        "turn_index": len(turns) - 1, "calls": calls,
                    })
                    error = Turn("user", self.builder.error_observation_pieces(
                        reason,
                        remaining_calls=max(0, max_calls - calls),
                        remaining_channel_views=max(0, max_views - local_views),
                    ))
                    append_user_turn(error)
                    append_context(error)
                    if calls >= max_calls or local_views >= max_views:
                        append_tool_unavailable("no channel views remain")
                    continue

                # The action is valid, unique, and affordable.  Re-encoding the
                # prefix gives the new view the same frontend/mixer/projector
                # path and memory ordinal it would have at this insertion point.
                if max_assistant_tokens - (used_assistant + final_reserve) < 0:
                    append_tool_unavailable("the final token reserve is required")
                    invalid_actions.append({
                        "text": content, "action": parsed,
                        "reason": "final reserve would be violated",
                        "turn_index": len(turns) - 1,
                    })
                    continue
                calls += 1
                local_views += len(parsed.channels)
                executed_actions.append(parsed)
                candidate_encoding = self.encode_episode(episode, executed_actions)
                if int(candidate_encoding.n_signal_tokens) > int(self.builder.max_signal):
                    calls += 1
                    reason = "signal-token budget exceeded"
                    invalid_actions.append({
                        "text": content, "action": parsed, "reason": reason,
                        "turn_index": len(turns) - 1, "calls": calls,
                    })
                    error = Turn("user", self.builder.error_observation_pieces(
                        reason,
                        remaining_calls=max(0, max_calls - calls),
                        remaining_channel_views=max(0, max_views - local_views),
                    ))
                    append_user_turn(error)
                    append_context(error)
                    if calls >= max_calls:
                        append_tool_unavailable("no zoom calls remain")
                    continue
                encoding = candidate_encoding
                view_index = len(encoding.views) - 1
                header = replace(
                    encoding.views[view_index].header,
                    remaining_calls=max(0, max_calls - calls),
                    remaining_channel_views=max(0, max_views - local_views),
                )
                encoding.views[view_index].header = header
                observation = Turn("user", self.builder.observation_pieces(header, view_index))
                existing_observations[action_key(parsed)] = view_index
                append_user_turn(observation)
                append_context(observation)
                soft_tokens = int(encoding.n_signal_tokens)
                if calls >= max_calls or local_views >= max_views:
                    append_tool_unavailable("the call/view budget is exhausted")
                elif max_assistant_tokens - used_assistant < final_reserve:
                    append_tool_unavailable("the final token reserve is required")
                continue

            # Anything not recognisably a zoom payload is the final answer.
            final_n = self._count_generated_tokens(content) + self._count_generated_tokens("<|im_end|>")
            if reasoning_n + final_n > available_tokens:
                final_n_limit = max(0, available_tokens - reasoning_n)
                content = self._truncate_generated_text(content, final_n_limit)
                final_n = self._count_generated_tokens(content) + self._count_generated_tokens("<|im_end|>") if content else 0
            turns.append(Turn("assistant_reasoning_final", [reasoning, content], SEG_FINAL, True))
            used_assistant += reasoning_n + final_n
            reasoning_tokens += reasoning_n
            final_tokens += final_n
            final_text = content
            finished = bool(result.get("finished", True)) and bool(content)
            break

        token_counts = {
            "prompt": int(prompt_tokens),
            "reasoning": int(reasoning_tokens),
            "action": int(action_tokens),
            "final": int(final_tokens),
            "assistant": int(used_assistant),
            "soft": int(soft_tokens),
            "prompt_tokens": int(prompt_tokens),
            "reasoning_tokens": int(reasoning_tokens),
            "action_tokens": int(action_tokens),
            "final_tokens": int(final_tokens),
            "soft_tokens": int(soft_tokens),
        }
        return {
            "text": final_text,
            "final_text": final_text,
            "final": final_text,
            "turns": turns,
            "executed_actions": executed_actions,
            "actions": executed_actions,
            "invalid_actions": invalid_actions,
            "invalid_action_texts": [record["text"] for record in invalid_actions],
            "duplicate_actions": duplicate_actions,
            "duplicates": duplicate_actions,
            "duplicate_action_texts": [record["text"] for record in duplicate_actions],
            "token_counts": token_counts,
            "prompt_tokens": token_counts["prompt"],
            "reasoning_tokens": token_counts["reasoning"],
            "action_tokens": token_counts["action"],
            "final_tokens": token_counts["final"],
            "assistant_tokens": token_counts["assistant"],
            "soft_tokens": token_counts["soft"],
            "n_signal_tokens": token_counts["soft"],
            "calls": int(calls),
            "local_channel_views": int(local_views),
            "generated_token_ids": generated_token_ids,
            "finished": bool(finished),
        }


__all__ = ["BackboneBundle", "EncodedView", "EpisodeEncoding", "LossReport", "WRAgentModel", "load_backbone"]
