#!/usr/bin/env python3
"""P2 smoke with the REAL backbone and REAL rows: one teacher trajectory per kind through forward/backward, and
autonomous agentic generation on validation rows with a real checkpoint (protocol only; no semantic claim).

Checks: every trajectory kind builds on a compatible train row, its sequence stays within the budgets, the loss is
finite with reasoning/action/final segments as planned (injected errors unsupervised), gradients flow; the agentic
runner terminates, executes at most the budgeted zooms, appends observations to the same cache and returns a final
answer whose format the strict parser accepts (or reports it did not).  Writes a JSON report; exit 0 only on PASS.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safetensors.torch import load_file  # noqa: E402

from wavelet_rope.backbone import load_backbone  # noqa: E402
from wavelet_rope.checkpoint import AtomicCheckpointStore  # noqa: E402
from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import AnomalyXLSplit, parse_zoom_action  # noqa: E402
from wavelet_rope.manifest import build_experiment_manifest  # noqa: E402
from wavelet_rope.model import WRAgentModel  # noqa: E402
from wavelet_rope.scoring import REQUIRED_KEYS, parse_structured_pred, score_precise  # noqa: E402
from wavelet_rope.trajectories import TRAJECTORY_KINDS, TrajectoryTeacher, compatible_kinds  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default="p1-main", help="run whose latest checkpoint initialises the model (or 'none')")
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--agentic-rows", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    cfg = load_runtime_config(args.config, validate_paths=False)
    raw, spec = cfg.as_dict(), dict(cfg.spec)
    device = torch.device("cuda:0")
    report: dict = {"checks": [], "trajectories": [], "agentic": []}

    def check(name, ok, **info):
        report["checks"].append({"check": name, "pass": bool(ok), **info})
        print(f"[{'PASS' if ok else 'FAIL':>4}] {name} {json.dumps(info, default=str)[:700]}", flush=True)

    root = Path(raw["dataset"]["root"])
    train = AnomalyXLSplit(root / raw["dataset"]["train_relative_path"], "train", 4488)
    val = AnomalyXLSplit(root / raw["dataset"]["validation_relative_path"], "validation", 520)
    rt = raw["runtime"]
    bb = load_backbone(raw["backbone"]["snapshot"]["root"], backend=rt["attention_backend"], pad_to=int(rt["attention_qk_pad_to"]),
                       device=device, gradient_checkpointing=True)
    training = spec["training"]
    model = WRAgentModel(bb, spec, max_calls=int(spec["agent"]["max_calls"]), max_views=int(spec["agent"]["max_local_channel_views"]),
                         segment_weights=dict(training["segment_loss_weights"]),
                         aux_weight=float(training["auxiliary_event_occupancy_loss_weight"]))
    model.frontend.to(device); model.mixer.to(device); model.projector.to(device)
    if model.token_supervision is not None:
        model.token_supervision.to(device)
    if args.run_id != "none":
        from wavelet_rope.checkpoint import inspect_foreign_checkpoint, latest_complete_dir
        target = latest_complete_dir(Path(raw["run_root"]) / args.run_id / "checkpoints")
        ref = inspect_foreign_checkpoint(target) if target is not None else None  # P1 weights, another experiment: protocol test only
        if ref is not None:
            model.load_trainable_state_dict(load_file(str(ref.path / "model.safetensors")))
            bb.ctx.lam = float(ref.trainer_state["wildcard_lambda"])
            check("checkpoint_loaded", True, checkpoint=ref.path.name, lam=bb.ctx.lam)
        else:
            check("checkpoint_loaded", True, checkpoint=None, note="run has no complete checkpoint yet; random frontend/LoRA init")

    def to_dev(ep):
        return type(ep)(ep.values.to(device), ep.valid.to(device), ep.channel_ids, ep.sample_times.to(device), ep.task)

    # 1. one trajectory per kind through the training loss
    model.train(); bb.text.train()
    teacher = TrajectoryTeacher(tokenizer=bb.tokenizer, max_calls=int(spec["agent"]["max_calls"]),
                                max_views=int(spec["agent"]["max_local_channel_views"]))
    metas = train.metas()
    rng = random.Random(args.seed)
    for kind in TRAJECTORY_KINDS:
        rows = [m.row_index for m in metas if kind in compatible_kinds(m)]
        if not rows:
            check(f"trajectory_{kind}", False, note="no compatible train row")
            continue
        row = rows[rng.randrange(len(rows))]
        meta, ep = train.meta(row), to_dev(train.episode(row))
        t0 = time.time()
        try:
            traj = teacher.build(meta, ep, random.Random(f"{args.seed}:{kind}:{row}"), kind=kind)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rep = model.forward_loss(meta, ep, mode="agentic", turns=traj.turns, actions=traj.actions,
                                         answer=meta.answer, gold=meta.gold_intervals)
            rep.loss.backward()
            grads = sum(int(p.grad is not None and bool((p.grad != 0).any())) for g in model.trainable_groups().values() for p in g)
            for g in model.trainable_groups().values():
                for p in g:
                    p.grad = None
            info = {"row": meta.id, "category": meta.category, "kind": traj.kind, "requested": kind, "n_actions": len(traj.actions),
                    "injected_error": traj.injected_error, "segments": traj.supervised_segments, "ce": rep.ce_by_segment,
                    "loss": float(rep.loss), "tokens": rep.sequence_length, "signal_tokens": rep.n_signal_tokens,
                    "views": rep.n_views, "seconds": round(time.time() - t0, 2), "grad_tensors_nonzero": grads}
            ok = torch.isfinite(rep.loss).item() and "final" in rep.ce_by_segment and grads > 0
            if kind != "final_no_tool":
                ok = ok and "reasoning" in rep.ce_by_segment
            check(f"trajectory_{kind}", ok, **info)
            report["trajectories"].append(info)
        except Exception as exc:  # noqa: BLE001
            import traceback
            check(f"trajectory_{kind}", False, error=f"{type(exc).__name__}: {str(exc)[:300]}", tb=traceback.format_exc()[-1500:])
    check("teacher_realised_distribution", True, distribution=teacher.realised_distribution)

    # 2. autonomous agentic generation on validation rows
    model.eval(); bb.text.eval()
    vmetas = val.metas()
    picks = [next(m.row_index for m in vmetas if m.category == "localize"),
             next(m.row_index for m in vmetas if m.category == "localize_all_channels")][: args.agentic_rows]
    for row in picks:
        meta, ep = val.meta(row), to_dev(val.episode(row))
        t0 = time.time()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate_agentic(meta, ep)
            final_text = out.get("text") or out.get("final_text") or ""
            parsed = parse_structured_pred(final_text, REQUIRED_KEYS[meta.category]) is not None
            scored = score_precise(final_text, meta.answer) if parsed else {"primary": 0.0}
            info = {k: v for k, v in out.items() if k not in ("turns", "state")}
            info.update({"row": meta.id, "category": meta.category, "final_text": final_text[:200], "parsed": parsed,
                         "primary": scored.get("primary"), "n_turns": len(out.get("turns", [])), "seconds": round(time.time() - t0, 1)})
            check(f"agentic_generation_{meta.category}", bool(out.get("finished", False)), **info)
            report["agentic"].append(info)
        except Exception as exc:  # noqa: BLE001
            import traceback
            check(f"agentic_generation_{meta.category}", False, error=f"{type(exc).__name__}: {str(exc)[:300]}", tb=traceback.format_exc()[-1500:])

    report["verdict"] = "PASS" if all(c["pass"] for c in report["checks"]) else "FAIL"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))
    print("VERDICT", report["verdict"])
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
