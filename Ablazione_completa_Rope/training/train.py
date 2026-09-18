#!/usr/bin/env python3
"""SFT trainer for WR-Agent-4B-Wildcard-v1 (DESIGN §10): fail-closed, resumable, coverage-counted.

One pass processes every train id exactly once; the mode of each id (direct / fixed_crop / agentic) is assigned
stratified per category by :class:`ModeAssigner`.  Microbatch 1, four microbatches per optimizer step, AdamW with two
parameter groups (new modules, LoRA), linear warmup then cosine decay, gradient clipping 1.0.  In P1 the wildcard
curriculum interpolates the T logits during the first 10% of the planned optimizer steps (lambda 0 -> 1).

Nothing is skipped silently: any failure on an example aborts the run with the example id in a FAILED marker, and
the coverage counters (per category and mode) are written with every checkpoint.  Checkpoints are published only at
optimizer boundaries through :class:`AtomicCheckpointStore`; resume restores model, optimizer, schedulers, RNG and the
sample cursor of the current pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wavelet_rope.backbone import load_backbone  # noqa: E402
from wavelet_rope.checkpoint import AtomicCheckpointStore  # noqa: E402
from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import AnomalyXLSplit, ModeAssigner, TeacherCropSampler, stratified_subset  # noqa: E402
from wavelet_rope.manifest import atomic_write_json, build_experiment_manifest, write_experiment_manifest  # noqa: E402
from wavelet_rope.model import WRAgentModel  # noqa: E402
from wavelet_rope.run_guard import RunLease  # noqa: E402
from wavelet_rope.scoring import REQUIRED_KEYS, parse_structured_pred, score_precise  # noqa: E402
from wavelet_rope.substitution import (  # noqa: E402
    SubstitutionDecision,
    make_substituted_episode,
    mask_gold_intervals,
    prepare_substitution,
    validate_substitution_share,
)
from wavelet_rope.trajectories import TrajectoryKindAssigner, TrajectoryTeacher  # noqa: E402

PHASE_LORA_LR = {"P1": "lora_learning_rate_P1", "P2": "lora_learning_rate_P2", "P3": "lora_learning_rate_P2"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def crop_kind(meta, action) -> str:
    """Effective relation between a teacher crop and the gold intervals of its channels (Astra 15/09 §4):
    positive_full (an event entirely inside), positive_partial (an event crosses a border), negative (no overlap)."""
    kinds = set()
    for g in meta.gold_intervals:
        if g.channel not in action.channels or g.end <= g.start:
            continue
        overlap = min(g.end, action.end) - max(g.start, action.start)
        if overlap <= 0:
            continue
        kinds.add("positive_full" if (g.start >= action.start and g.end <= action.end) else "positive_partial")
    if "positive_full" in kinds:
        return "positive_full"
    return "positive_partial" if kinds else "negative"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="resolved TOML (no REQUIRED left)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--resume", default="auto", help="auto | none | <checkpoint dir>")
    ap.add_argument("--freeze-frontend", action="store_true",
                    help="freeze frontend, mixer, projector and supervision heads (kept in eval) and train the LoRA only (Astra 15/09, V1 of P1-f: can the backbone learn to read a FIXED representation?)")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="override the warmup length in optimizer steps (0 = the spec ratio)")
    ap.add_argument("--constant-lr", action="store_true",
                    help="after the warmup keep the peak learning rate constant instead of the cosine decay")
    ap.add_argument("--wildcard-lambda", type=float, default=None,
                    help="force a constant wildcard RoPE lambda (geometric control: 0 against 1)")
    ap.add_argument("--resume-foreign", default=None,
                    help="resume the FULL training state (weights, optimizer moments, RNG, sampling cursor) from a "
                         "checkpoint of ANOTHER experiment fingerprint, e.g. after a source change; both are recorded")
    ap.add_argument("--extend-steps", type=int, default=0,
                    help="resume and train N MORE optimizer steps under a FRESH schedule (Astra 15/09, D1 of the P1-e extension): warmup --extend-warmup then cosine over the N added steps, optimizer moments preserved")
    ap.add_argument("--extend-warmup", type=int, default=18, help="warmup steps of the extension schedule")
    ap.add_argument("--skip-curriculum", action="store_true",
                    help="wildcard lambda = 1 from the first step (phase B of P1-d initialised from phase A)")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint dir of ANOTHER experiment whose trainable weights initialise this run (e.g. P2 from the "
                         "selected P1 checkpoint); only used on a fresh start, recorded in trainer_state.init_from")
    ap.add_argument("--max-optimizer-steps", type=int, default=0, help="stop early (smoke); 0 = full plan")
    ap.add_argument("--limit-examples", type=int, default=0, help="use only the first N assigned examples per pass")
    ap.add_argument("--val-every", type=int, default=0, help="optimizer steps between quick validations (0 = pass end only)")
    ap.add_argument("--val-subset", type=int, default=104, help="stratified validation rows for quick probes")
    ap.add_argument("--val-full-at-pass-end", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=768, help="final-answer reserve (DESIGN 9.4); official validation uses 768")
    ap.add_argument("--flex-dynamic", type=int, default=1)
    ap.add_argument("--checkpointing-min-tokens", type=int, default=0,
                    help="0 = backbone gradient checkpointing as in config; >0 disables it (the prototype runs microbatch 1 on 192 GB)")
    return ap.parse_args()


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cfg = load_runtime_config(args.config)
        self.cfg.require_resolved()
        raw = self.cfg.as_dict()
        self.raw = raw
        self.spec = dict(self.cfg.spec)
        self.phase = raw["phase"]
        self.run_dir = Path(raw["run_root"]) / args.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        self.manifest = build_experiment_manifest(self.cfg, project_root=PROJECT_ROOT)
        self.fingerprint = self.manifest.experiment_fingerprint
        manifest_path = self.run_dir / "EXPERIMENT_MANIFEST.json"
        if not manifest_path.is_file():
            write_experiment_manifest(manifest_path, self.manifest)
        else:
            existing = json.loads(manifest_path.read_text(encoding="utf-8")).get("experiment_fingerprint")
            if existing != self.fingerprint:
                # never overwrite the manifest of the run that produced the checkpoints (e.g. evaluation with a
                # newer tool tree): keep both, keyed by fingerprint
                write_experiment_manifest(self.run_dir / f"EXPERIMENT_MANIFEST.{self.fingerprint[:12]}.json", self.manifest)
        self.store = AtomicCheckpointStore(self.run_dir / "checkpoints", self.fingerprint)
        self.device = torch.device("cuda:0")
        self.seed = int(raw["seed"])
        training = self.spec["training"]
        self.ce_weight = float(training.get("ce_weight", 1.0))
        if self.ce_weight < 0.0:
            raise ValueError("training.ce_weight must be non-negative")
        self.substitution_share = validate_substitution_share(training.get("substitution_share", 0.0))
        self.accum = int(training["gradient_accumulation_steps"])
        self.mix = dict(training[f"{self.phase}_mix"])
        self.passes = int(training["phase_passes"][self.phase])
        self.examples_per_pass = int(training["examples_per_pass"])
        self.steps_per_pass = math.ceil(self.examples_per_pass / self.accum)
        self.total_steps = self.steps_per_pass * self.passes
        if int(getattr(self.args, "max_optimizer_steps", 0) or 0) > 0:
            # Astra 15/09 D1: a bounded phase (P1-d A) schedules warmup 3 % and cosine over ITS OWN horizon.
            self.total_steps = min(self.total_steps, int(self.args.max_optimizer_steps))
        curriculum = self.spec["backbone"]["temporal_rope"]["curriculum"]
        self.curriculum_steps = (int(round(float(curriculum["transition_fraction_of_P1_optimizer_steps"]) * self.steps_per_pass))
                                 if self.phase == curriculum["phase"] else 0)
        if getattr(self.args, "skip_curriculum", False):
            self.curriculum_steps = 0  # Astra 15/09 D4: phase B keeps lambda = 1 (no RoPE curriculum restart)
        self.log_every = int(raw["logging"]["every_optimizer_steps"])
        self.ckpt_every = int(raw["checkpoint"]["every_optimizer_steps"])
        self.keep_last = int(raw["checkpoint"]["keep_last"])
        self._log_file = open(self.run_dir / "logs" / "train.jsonl", "a", encoding="utf-8")

    # ------------------------------------------------------------------ setup
    def log(self, record: dict[str, Any]) -> None:
        record = {"utc": utc_now(), **record}
        self._log_file.write(json.dumps(record, default=str) + "\n")
        self._log_file.flush()

    def say(self, msg: str) -> None:
        print(f"[train {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    def setup_data(self) -> None:
        root = Path(self.raw["dataset"]["root"])
        ds = self.raw["dataset"]
        self.train = AnomalyXLSplit(root / ds["train_relative_path"], "train", int(ds["train_rows"]))
        self.val = AnomalyXLSplit(root / ds["validation_relative_path"], "validation", int(ds["validation_rows"]))
        self.metas = self.train.metas()
        if len(self.metas) != self.examples_per_pass:
            raise RuntimeError(f"train split has {len(self.metas)} rows, plan expects {self.examples_per_pass}")
        self.assigner = ModeAssigner()
        self.cropper = TeacherCropSampler()
        agent = self.spec["agent"]
        self.trajectory_kind_assigner = TrajectoryKindAssigner()
        self.trajectory_teacher = TrajectoryTeacher(
            max_calls=int(agent["max_calls"]),
            max_views=int(agent["max_local_channel_views"]),
            max_reasoning_tokens=int(agent["max_reasoning_tokens_per_decision"]),
            # Astra 16/09: "coarse_to_fine" ties the first window to the recording length instead of drawing it
            # log-uniformly in [256, 8192] regardless of L, which is what left the anomaly inside the window only
            # 32% of the time at L=131.072 in the Z1-bis traces.  Default keeps the old rule.
            width_rule=str(self.spec["training"].get("trajectory_width_rule", "log_uniform")),
        )
        self._trajectory_kinds: dict[int, str] = {}
        self.say(f"train {len(self.metas)} rows {self.train.category_counts()}; val {len(self.val)} rows")

    def setup_model(self) -> None:
        rt = self.raw["runtime"]
        snapshot = self.raw["backbone"]["snapshot"]["root"]
        gc_always = bool(rt["gradient_checkpointing"]) and self.args.checkpointing_min_tokens == 0
        self.bb = load_backbone(snapshot, backend=rt["attention_backend"], pad_to=int(rt["attention_qk_pad_to"]),
                                device=self.device, gradient_checkpointing=gc_always,
                                flex_dynamic=bool(self.args.flex_dynamic))
        training = self.spec["training"]
        self.model = WRAgentModel(self.bb, self.spec, max_calls=int(self.spec["agent"]["max_calls"]),
                                  max_views=int(self.spec["agent"]["max_local_channel_views"]),
                                  segment_weights=dict(training["segment_loss_weights"]),
                                  aux_weight=float(training["auxiliary_event_occupancy_loss_weight"]))
        # The teacher is backbone-free during planning, but once the pinned
        # tokenizer is loaded use its exact token count for the 384-token
        # per-decision rationale cap.
        self.trajectory_teacher.tokenizer = self.bb.tokenizer
        self.reading_segment = bool(training.get("reading_segment", False))  # Astra 15/09, sentinella D4
        self.model.frontend.to(self.device)
        self.model.mixer.to(self.device)
        self.model.projector.to(self.device)
        if self.model.token_supervision is not None:
            self.model.token_supervision.to(self.device)
        opt = self.raw["optimizer"]
        lora_lr = float(training[PHASE_LORA_LR[self.phase]])
        if lora_lr <= 0.0:
            # Astra 15/09 D1 (P1-d phase A): LoRA frozen with requires_grad=False; the backward still flows through the
            # backbone to the soft tokens.  The LoRA tensors stay in the checkpoint (identity adapters).
            for name, parameter in self.bb.text.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad_(False)
        if getattr(self.args, "freeze_frontend", False):
            for module in (self.model.frontend, self.model.mixer, self.model.projector,
                           getattr(self.model, "token_supervision", None)):
                if module is None:
                    continue
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            self.say("frontend, mixer, projector and supervision heads FROZEN (eval): only the LoRA trains")
        groups = self.model.trainable_groups()
        param_groups = []
        if groups["frontend"]:
            param_groups.append({"params": groups["frontend"], "lr": float(training["frontend_learning_rate"]),
                                 "name": "frontend"})
        if groups["lora"]:
            param_groups.append({"params": groups["lora"], "lr": lora_lr, "name": "lora"})
        if not param_groups:
            raise ValueError("no trainable parameter group: check freeze-frontend and the learning rates")
        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(float(opt["beta1"]), float(opt["beta2"])), eps=float(opt["epsilon"]),
            weight_decay=float(training["weight_decay"]))
        self.base_lrs = [g["lr"] for g in self.optimizer.param_groups]
        self.warmup_steps = int(getattr(self.args, "warmup_steps", 0) or
                                round(float(training["warmup_ratio"]) * self.total_steps))
        n_front = sum(p.numel() for p in groups["frontend"])
        n_lora = sum(p.numel() for p in groups["lora"])
        self.say(f"trainable: frontend {n_front / 1e6:.2f}M, lora {n_lora / 1e6:.2f}M; wrapped layers {self.bb.wrapped_layers}")
        atomic_write_json(self.run_dir / "TRAINABLE_PARAMETERS.json", {
            "frontend": [n for n, p in self.model.named_parameters() if p.requires_grad and not n.startswith("bb")],
            "lora": self.bb.lora_parameter_names, "n_frontend": n_front, "n_lora": n_lora,
            "lora_targets": self.bb.lora_targets, "wrapped_layers": self.bb.wrapped_layers})
        self.state = {"experiment_fingerprint": self.fingerprint, "phase": self.phase, "pass_index": 0,
                      "optimizer_step": 0, "sample_cursor": 0, "micro_step_in_accumulation": 0,
                      "wildcard_lambda": 0.0 if self.curriculum_steps else 1.0, "coverage": {},
                      "plan": {"passes": self.passes, "steps_per_pass": self.steps_per_pass, "total_steps": self.total_steps,
                               "warmup_steps": self.warmup_steps, "curriculum_steps": self.curriculum_steps, "mix": self.mix}}

    # ------------------------------------------------------------------ schedules
    def lr_factor(self, step: int) -> float:
        # An extension restarts the schedule at the resumed step: warmup, then cosine over the ADDED steps only.
        origin = getattr(self, 'schedule_origin', 0)
        local = step - origin
        if self.warmup_steps and local < self.warmup_steps:
            return (local + 1) / self.warmup_steps
        if getattr(self.args, "constant_lr", False):
            return 1.0
        progress = (local - self.warmup_steps) / max(1, (self.total_steps - origin) - self.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    def apply_schedules(self, step: int) -> None:
        factor = self.lr_factor(step)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * factor
        forced = getattr(self.args, "wildcard_lambda", None)
        lam = float(forced) if forced is not None else (1.0 if not self.curriculum_steps
                                                        else min(1.0, step / self.curriculum_steps))
        self.bb.ctx.lam = lam
        self.state["wildcard_lambda"] = lam

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self) -> None:
        from safetensors.torch import save_file

        state = dict(self.state)
        state["saved_at_utc"] = utc_now()

        def w_model(path: Path) -> None:
            save_file({k: v.contiguous() for k, v in self.model.trainable_state_dict().items()}, str(path))

        def w_opt(path: Path) -> None:
            torch.save(self.optimizer.state_dict(), path)

        def w_sched(path: Path) -> None:
            torch.save({"base_lrs": self.base_lrs, "lambda": self.state["wildcard_lambda"],
                        "optimizer_step": self.state["optimizer_step"]}, path)

        def w_rng(path: Path) -> None:
            torch.save({"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                        "python": random.getstate()}, path)

        ref = self.store.save(trainer_state=state, artifact_writers={
            "model.safetensors": w_model, "optimizer.pt": w_opt, "schedulers.pt": w_sched, "rng.pt": w_rng})
        self.say(f"checkpoint {ref.path.name} saved (step {self.state['optimizer_step']}, cursor {self.state['sample_cursor']})")
        self.prune_checkpoints()

    def prune_checkpoints(self) -> None:
        import shutil
        dirs = sorted(p for p in self.store.root.iterdir() if p.is_dir() and (p / "COMPLETE").is_file())
        keep = set(dirs[-self.keep_last:])
        protected = {p for p in dirs if (p / "PASS_END").is_file()}
        for path in dirs:
            if path not in keep and path not in protected:
                shutil.rmtree(path)

    def resume(self) -> None:
        from safetensors.torch import load_file

        foreign = getattr(self.args, "resume_foreign", None)
        if foreign:
            from wavelet_rope.checkpoint import inspect_foreign_checkpoint
            ref = inspect_foreign_checkpoint(foreign)
            self.say(f"resuming ACROSS experiments from {ref.path} "
                     f"(checkpoint fingerprint {ref.trainer_state['experiment_fingerprint'][:12]}, "
                     f"this tool tree {self.fingerprint[:12]})")
            self.state["resumed_foreign"] = {"path": str(ref.path), "optimizer_step": ref.optimizer_step,
                                             "checkpoint_fingerprint": ref.trainer_state["experiment_fingerprint"],
                                             "tool_fingerprint": self.fingerprint}
            atomic_write_json(self.run_dir / "RESUMED_FOREIGN.json", self.state["resumed_foreign"])
        else:
            ref = self.store.resolve_resume(self.args.resume)
        if ref is None:
            if getattr(self.args, "init_from", None):
                from safetensors.torch import load_file as _load
                from wavelet_rope.checkpoint import inspect_foreign_checkpoint
                init = inspect_foreign_checkpoint(self.args.init_from)
                incoming = dict(_load(str(init.path / "model.safetensors")))
                # A foreign checkpoint predates modules added later (P1-e token-supervision heads and their fixed
                # target statistics): those keep their fresh initialisation, everything else must be present.
                current = self.model.trainable_state_dict()
                fresh = sorted(k for k in current if k not in incoming and k.startswith("token_supervision."))
                for key in fresh:
                    incoming[key] = current[key]
                if fresh:
                    self.say(f"init-from: {len(fresh)} token-supervision tensors keep their fresh initialisation")
                self.model.load_trainable_state_dict(incoming)
                self.state["init_from"] = {"path": str(init.path), "optimizer_step": init.optimizer_step,
                                           "experiment_fingerprint": init.trainer_state["experiment_fingerprint"],
                                           "phase": init.trainer_state.get("phase"), "lambda": init.trainer_state.get("wildcard_lambda")}
                atomic_write_json(self.run_dir / "INIT_FROM.json", self.state["init_from"])
                self.say(f"fresh start initialised from {init.path} (phase {init.trainer_state.get('phase')}, "
                         f"fingerprint {init.trainer_state['experiment_fingerprint'][:12]})")
            else:
                self.say("fresh start")
            return
        incoming = dict(load_file(str(ref.path / "model.safetensors")))
        current = self.model.trainable_state_dict()
        fresh = sorted(k for k in current if k not in incoming)
        for key in fresh:
            incoming[key] = current[key]
        if fresh:
            self.say(f"resume: {len(fresh)} tensors absent from the checkpoint keep their fresh initialisation")
        self.model.load_trainable_state_dict(incoming)
        self.optimizer.load_state_dict(torch.load(ref.path / "optimizer.pt", map_location="cpu"))
        rng = torch.load(ref.path / "rng.pt", map_location="cpu", weights_only=False)
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        random.setstate(rng["python"])
        st = dict(ref.trainer_state)
        for key in ("pass_index", "optimizer_step", "sample_cursor", "micro_step_in_accumulation", "wildcard_lambda", "coverage"):
            self.state[key] = st[key]
        self.say(f"resumed from {ref.path.name}: pass {st['pass_index']} step {st['optimizer_step']} cursor {st['sample_cursor']}")
        extra = int(getattr(self.args, "extend_steps", 0) or 0)
        if extra > 0:
            # Astra 15/09 (D1 of the P1-e extension): keep weights, heads, statistics and Adam moments, restart the
            # schedule over the ADDED steps only (short warmup, then cosine) and keep sampling where it stopped.
            self.schedule_origin = int(self.state["optimizer_step"])
            self.total_steps = self.schedule_origin + extra
            self.warmup_steps = int(getattr(self.args, "extend_warmup", 18))
            self.args.max_optimizer_steps = self.total_steps
            self.state["plan"] = dict(self.state.get("plan", {}), extension={"origin": self.schedule_origin,
                                      "added_steps": extra, "warmup": self.warmup_steps, "total": self.total_steps})
            self.say(f"extension: {extra} more steps from {self.schedule_origin} (warmup {self.warmup_steps}, "
                     f"peak lr {self.base_lrs[0]:.2e}, cosine to {self.total_steps})")

    # ------------------------------------------------------------------ examples
    def example_rng(self, pass_index: int, row_index: int) -> random.Random:
        return random.Random(f"{self.seed}:{pass_index}:{row_index}")

    def _prepare_substitution(
        self,
        pass_index: int,
        row_index: int,
        meta,
        receiver_episode,
    ) -> tuple[SubstitutionDecision, Any, tuple]:
        """Choose the O2 donor and return the episode/gold shown to the model."""

        if self.substitution_share == 0.0:
            # Keep the zero/absent-key path free of donor RNG draws and extra
            # parquet reads, preserving the pre-O2 training behaviour.
            return (
                SubstitutionDecision(requested=False, substituted=False),
                receiver_episode,
                meta.gold_intervals,
            )

        decision = prepare_substitution(
            self.substitution_share,
            metas=self.metas,
            receiver_meta=meta,
            receiver=receiver_episode,
            episode_getter=self.train.episode,
            rng=self.example_rng(pass_index, row_index),
        )
        if not decision.substituted:
            return decision, receiver_episode, meta.gold_intervals
        assert decision.donor is not None  # guarded by ``substituted``
        shown = make_substituted_episode(receiver_episode, decision.donor.episode)
        shown_gold = mask_gold_intervals(decision.donor.meta.gold_intervals, receiver_episode)
        return decision, shown, shown_gold

    @staticmethod
    def _receiver_header_overrides(model, receiver_episode, actions) -> dict[int, Any]:
        """Encode the true receiver once and retain only its text headers."""

        # Header construction is metadata-only for O2.  Avoid retaining a
        # second autograd graph while still using the exact current frontend
        # and the same action prefix as the substituted encoding.
        with torch.no_grad():
            receiver_encoding = model.encode_episode(receiver_episode, actions)
        return {index: replace(view.header) for index, view in enumerate(receiver_encoding.views)}

    def train_example(self, pass_index: int, row_index: int, mode: str) -> dict[str, Any]:
        meta = self.train.meta(row_index)
        receiver_episode = self.train.episode(row_index)
        substitution, episode, supervision_gold = self._prepare_substitution(
            pass_index, row_index, meta, receiver_episode
        )
        actions = []
        trajectory = None
        reading = None
        header_overrides = None
        action_rng = self.example_rng(pass_index, row_index)
        if mode == "fixed_crop":
            actions = [self.cropper.sample(meta, action_rng)]
        elif mode == "agentic":
            trajectory = self.trajectory_teacher.build(
                meta,
                episode,
                action_rng,
                kind=self._trajectory_kinds.get(row_index),
            )
            if substitution.substituted:
                trajectory = replace(
                    trajectory,
                    turns=[
                        replace(turn, supervise_final=False)
                        if turn.role in ("assistant_final", "assistant_reasoning_final")
                        else turn
                        for turn in trajectory.turns
                    ],
                )
            actions = trajectory.actions
        elif mode != "direct":
            raise NotImplementedError(f"mode {mode} is not part of phase {self.phase} yet")
        episode = type(episode)(episode.values.to(self.device), episode.valid.to(self.device), episode.channel_ids,
                                episode.sample_times.to(self.device), episode.task)
        if substitution.substituted:
            receiver_episode = type(receiver_episode)(
                receiver_episode.values.to(self.device),
                receiver_episode.valid.to(self.device),
                receiver_episode.channel_ids,
                receiver_episode.sample_times.to(self.device),
                receiver_episode.task,
            )
            header_overrides = self._receiver_header_overrides(self.model, receiver_episode, actions)
        substitution_loss_kwargs = (
            {"header_overrides": header_overrides, "supervise_final": False}
            if substitution.substituted
            else {}
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if trajectory is None:
                # Keep the P1 direct/fixed-crop path byte-for-byte in spirit:
                # teacher actions remain in context but are not CE targets.
                if self.reading_segment and self.ce_weight > 0.0:
                    reading = self.model.reading_text(episode, actions)
                report = self.model.forward_loss(meta, episode, mode=mode, actions=actions, answer=meta.answer,
                                                 gold=supervision_gold, supervise_action=False, reading=reading,
                                                 **substitution_loss_kwargs)
            else:
                report = self.model.forward_loss(
                    meta,
                    episode,
                    mode="agentic",
                    turns=trajectory.turns,
                    actions=trajectory.actions,
                    answer=meta.answer,
                    gold=supervision_gold,
                    **substitution_loss_kwargs,
                )
        (report.loss / self.accum).backward()
        record = {"id": meta.id, "category": meta.category, "mode": mode, "loss": float(report.loss.detach()),
                "ce": report.ce_by_segment, "aux": report.aux_loss, "tokens": report.sequence_length,
                "signal_tokens": report.n_signal_tokens, "views": report.n_views, "actions": [a.to_json() for a in actions],
                "substitution_requested": substitution.requested,
                "substituted": substitution.substituted,
                "donor_id": substitution.donor_id,
                "substitution_reason": substitution.reason,
                "crop_kind": crop_kind(meta, actions[0]) if actions else None, "soft_rms": round(report.soft_rms, 6),
                "decorrelation": round(report.decorrelation, 6),
                "token_pos_loss": round(report.token_pos_loss, 6),
                "token_energy_loss": round(report.token_energy_loss, 6),
                "token_pos_pearson": round(report.token_pos_pearson, 6),
                "token_peak_loss": round(report.token_peak_loss, 6), "token_rank_loss": round(report.token_rank_loss, 6),
                "token_pos_mae_bins": round(report.token_pos_mae_bins, 6),
                "token_energy_pearson": round(report.token_energy_pearson, 6),
                "token_peak_pearson": round(report.token_peak_pearson, 6),
                "token_top3_recall": round(report.token_top3_recall, 6),
                "token_energy_balanced_accuracy": round(report.token_energy_balanced_accuracy, 6),
                "reading_chars": len(reading) if reading else 0}
        if trajectory is not None:
            record.update({
                "trajectory_kind": trajectory.kind,
                "num_actions": len(trajectory.actions),
                "n_actions": len(trajectory.actions),
                "supervised_segments": trajectory.supervised_segments,
                "injected_error": bool(trajectory.injected_error),
                "injected_error_flag": bool(trajectory.injected_error),
            })
        return record

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def validate(self, *, rows: list[int], tag: str) -> dict[str, Any]:
        self.model.eval()
        self.bb.text.eval()
        per_cat: dict[str, list[float]] = {}
        invalid = 0
        records = []
        t0 = time.time()
        for row in rows:
            meta = self.val.meta(row)
            episode = self.val.episode(row)
            episode = type(episode)(episode.values.to(self.device), episode.valid.to(self.device), episode.channel_ids,
                                    episode.sample_times.to(self.device), episode.task)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen = self.model.generate(meta, episode, mode="direct", max_new_tokens=self.args.max_new_tokens)
            scored = score_precise(gen["text"], meta.answer)
            invalid += int(parse_structured_pred(gen["text"], REQUIRED_KEYS[meta.category]) is None)
            per_cat.setdefault(meta.category, []).append(float(scored["primary"]))
            records.append({"id": meta.id, "category": meta.category, "primary": scored["primary"], "pred": gen["text"][:300],
                            "gold": meta.answer, "prompt_tokens": gen["prompt_length"],
                            "reading": (gen.get("reading") or "")[:300], "reasoning_tokens": gen.get("reasoning_tokens"),
                            "forced_closure": gen.get("forced_closure")})
        summary = {k: round(sum(v) / len(v), 4) for k, v in per_cat.items()}
        overall = round(sum(sum(v) for v in per_cat.values()) / max(1, sum(len(v) for v in per_cat.values())), 4)
        fixed_crop_loss = self.validate_fixed_crop_loss(rows)
        result = {"tag": tag, "rows": len(rows), "primary": overall, "by_category": summary, "invalid_format": invalid,
                  "invalid_format_rate": round(invalid / max(1, len(rows)), 4), "fixed_crop_loss": fixed_crop_loss,
                  "seconds": round(time.time() - t0, 1), "optimizer_step": self.state["optimizer_step"]}
        atomic_write_json(self.run_dir / "logs" / f"val_{tag}_step{self.state['optimizer_step']:06d}.json",
                          {"summary": result, "records": records})
        self.log({"event": "validation", **result})
        self.say(f"VAL[{tag}] primary {overall} {summary} invalid {invalid} ({result['seconds']}s)")
        self.model.train()
        self.bb.text.train()
        return result

    @torch.no_grad()
    def validate_fixed_crop_loss(self, rows: list[int]) -> float:
        """Tie-break metric (Astra 15/09 §3): teacher-crop reading loss on validation rows, seeded crops."""
        losses = []
        for row in rows:
            meta = self.val.meta(row)
            actions = [self.cropper.sample(meta, random.Random(f"val:{self.seed}:{row}"))]
            episode = self.val.episode(row)
            episode = type(episode)(episode.values.to(self.device), episode.valid.to(self.device), episode.channel_ids,
                                    episode.sample_times.to(self.device), episode.task)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                report = self.model.forward_loss(meta, episode, mode="fixed_crop", actions=actions, answer=meta.answer,
                                                 gold=meta.gold_intervals, supervise_action=False)
            losses.append(float(report.loss))
        return round(sum(losses) / max(1, len(losses)), 4)

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        self.setup_data()
        self.setup_model()
        self.resume()
        self.model.train()
        self.bb.text.train()
        coverage = Counter({tuple(k.split("|")): v for k, v in self.state.get("coverage", {}).items()})
        step = int(self.state["optimizer_step"])
        micro = int(self.state["micro_step_in_accumulation"])
        self.apply_schedules(step)
        val_rows = stratified_subset(self.val.metas(), self.args.val_subset, self.seed)
        started = time.time()
        examples_done = 0
        for pass_index in range(int(self.state["pass_index"]), self.passes):
            assignments = self.assigner.assign(self.metas, self.mix, seed=self.seed, pass_index=pass_index)
            agentic_rows = [row for row, mode in assignments if mode == "agentic"]
            if agentic_rows:
                kind_assignments = self.trajectory_kind_assigner.assign(
                    self.metas,
                    seed=self.seed,
                    pass_index=pass_index,
                    row_indices=agentic_rows,
                )
                self._trajectory_kinds = dict(kind_assignments)
                trajectory_report = self.trajectory_kind_assigner.report(kind_assignments)
                self.state["trajectory_distribution"] = trajectory_report
                self.log({
                    "event": "trajectory_distribution",
                    "pass": pass_index,
                    "rows": len(agentic_rows),
                    "report": trajectory_report,
                })
            else:
                self._trajectory_kinds = {}
            if self.args.limit_examples:
                assignments = assignments[: self.args.limit_examples]
            cursor = int(self.state["sample_cursor"]) if pass_index == int(self.state["pass_index"]) else 0
            self.say(f"pass {pass_index}: {len(assignments)} examples, resuming at cursor {cursor}")
            for position in range(cursor, len(assignments)):
                row_index, mode = assignments[position]
                t0 = time.time()
                try:
                    rec = self.train_example(pass_index, row_index, mode)
                except Exception as exc:  # fail closed: no silent skip
                    marker = {"utc": utc_now(), "pass": pass_index, "position": position, "row_index": row_index, "mode": mode,
                              "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]}
                    atomic_write_json(self.run_dir / "FAILED", marker)
                    self.log({"event": "failed", **marker})
                    raise
                rec.update({"event": "example", "pass": pass_index, "position": position, "seconds": round(time.time() - t0, 2),
                            "lambda": self.state["wildcard_lambda"], "optimizer_step": step})
                self.log(rec)
                coverage[(rec["category"], mode)] += 1
                micro += 1
                examples_done += 1
                self.state["sample_cursor"] = position + 1
                self.state["pass_index"] = pass_index
                self.state["micro_step_in_accumulation"] = micro
                if micro == self.accum:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for g in self.model.trainable_groups().values() for p in g],
                        float(self.spec["training"]["gradient_clip"]))
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    step += 1
                    micro = 0
                    self.state["optimizer_step"] = step
                    self.state["micro_step_in_accumulation"] = 0
                    self.apply_schedules(step)
                    self.state["coverage"] = {"|".join(k): v for k, v in coverage.items()}
                    if step % self.log_every == 0:
                        elapsed = time.time() - started
                        # Peak allocation decides which GPUs can host this run: the MI300X reports 35.8 GB of
                        # reserved VRAM, but that is allocator cache on a 192 GB card, not the requirement.  va1
                        # fitted the same task in 24 GB on a Vultr A40, so the true peak is what we need to know.
                        peak_gib = torch.cuda.max_memory_allocated() / 2 ** 30
                        self.say(f"step {step}/{self.total_steps} loss {rec['loss']:.4f} grad {float(grad_norm):.3f} "
                                 f"lambda {self.state['wildcard_lambda']:.3f} lr {self.optimizer.param_groups[0]['lr']:.2e}/"
                                 f"{(self.optimizer.param_groups[1]['lr'] if len(self.optimizer.param_groups) > 1 else 0.0):.2e} {elapsed / max(1, examples_done):.1f}s/ex "
                                 f"tokens {rec['tokens']} peak {peak_gib:.1f}GiB")
                        self.log({"event": "step", "optimizer_step": step, "grad_norm": float(grad_norm),
                                  "lambda": self.state["wildcard_lambda"], "lr": [g["lr"] for g in self.optimizer.param_groups],
                                  "seconds_per_example": elapsed / max(1, examples_done),
                                  "peak_gib": round(peak_gib, 3)})
                    if step % self.ckpt_every == 0:
                        self.save_checkpoint()
                    if self.args.val_every and step % self.args.val_every == 0:
                        self.validate(rows=val_rows, tag="quick")
                    if self.args.max_optimizer_steps and step >= self.args.max_optimizer_steps:
                        self.say("max optimizer steps reached; saving and stopping")
                        # The periodic save may already have written this exact step (the horizon is often a multiple of
                        # the checkpoint interval): saving twice raises "checkpoint already exists".
                        if step % self.ckpt_every != 0:
                            self.save_checkpoint()
                        return
            # pass end: flush a partial accumulation is NOT done (checkpoints are optimizer boundaries only)
            if micro != 0:
                self.say(f"pass end with {micro} pending microbatches: stepping once with the partial accumulation")
                torch.nn.utils.clip_grad_norm_([p for g in self.model.trainable_groups().values() for p in g],
                                               float(self.spec["training"]["gradient_clip"]))
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                step += 1
                micro = 0
                self.state["optimizer_step"] = step
                self.state["micro_step_in_accumulation"] = 0
                self.apply_schedules(step)
            self.state["coverage"] = {"|".join(k): v for k, v in coverage.items()}
            self.state["pass_index"] = pass_index + 1
            self.state["sample_cursor"] = 0
            # A pass boundary is often a multiple of the checkpoint interval, so the periodic save may already
            # have written this exact step; that copy carries the OLD pass bookkeeping and is superseded here.
            superseded = self.store.root / f"step_{step:012d}"
            if superseded.exists():
                self.say(f"pass end coincides with the periodic checkpoint at step {step}; rewriting it")
                shutil.rmtree(superseded)
            self.save_checkpoint()
            latest = self.store.latest_complete()
            if latest is not None:
                (latest.path / "PASS_END").write_text(json.dumps({"pass": pass_index, "coverage": self.state["coverage"]}))
            self.say(f"pass {pass_index} complete; coverage {dict(coverage)}")
            if self.args.val_full_at_pass_end:
                self.validate(rows=list(range(len(self.val))), tag=f"full_pass{pass_index}")
            else:
                self.validate(rows=val_rows, tag=f"quick_pass{pass_index}")
        atomic_write_json(self.run_dir / "DONE", {"utc": utc_now(), "optimizer_step": step, "coverage": self.state["coverage"]})
        self.say("training plan complete")


def main() -> int:
    args = parse_args()
    trainer = Trainer(args)
    with RunLease(trainer.run_dir, trainer.fingerprint):
        trainer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
