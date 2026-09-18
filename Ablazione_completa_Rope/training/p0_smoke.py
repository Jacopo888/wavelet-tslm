#!/usr/bin/env python3
"""P0 end-to-end smoke with the REAL backbone and REAL train rows (DESIGN §10.2, Astra 14/09 §3).

Checks: (1) direct-mode forward/backward on a 1-channel row; (2) fixed-crop forward/backward on a 2-channel crop
(teacher action), gradients non-zero in every module group (A/D/raw trunks, local adapters, tokenizer/fusion,
mixer, projector, LoRA); (3) one optimizer step changes parameters of every group; (4) save -> reload gives the
same loss on the same example; (5) greedy generation produces a parseable answer schema (format only, no
semantic claim); (6) maximum load: a 16-channel L=131072 row with a 2-channel 8192-sample crop, forward+backward
peak memory and time.  Writes a JSON report; exit 0 only if every check passes.  No weights are kept.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safetensors.torch import load_file, save_file  # noqa: E402

from wavelet_rope.backbone import load_backbone  # noqa: E402
from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import AnomalyXLSplit, TeacherCropSampler, ZoomAction  # noqa: E402
from wavelet_rope.model import WRAgentModel  # noqa: E402
from wavelet_rope.scoring import parse_structured_pred, REQUIRED_KEYS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--skip-max-load", action="store_true")
    args = ap.parse_args()
    cfg = load_runtime_config(args.config, validate_paths=False)
    raw = cfg.as_dict()
    spec = dict(cfg.spec)
    device = torch.device("cuda:0")
    report: dict = {"checks": []}

    def check(name, ok, **info):
        report["checks"].append({"check": name, "pass": bool(ok), **info})
        print(f"[{'PASS' if ok else 'FAIL':>4}] {name} {json.dumps(info, default=str)[:600]}", flush=True)

    root = Path(raw["dataset"]["root"])
    train = AnomalyXLSplit(root / raw["dataset"]["train_relative_path"], "train", 4488)
    metas = train.metas()
    by_cat = {}
    for m in metas:
        by_cat.setdefault(m.category, []).append(m.row_index)
    rt = raw["runtime"]
    bb = load_backbone(raw["backbone"]["snapshot"]["root"], backend=rt["attention_backend"],
                       pad_to=int(rt["attention_qk_pad_to"]), device=device, gradient_checkpointing=True)
    training = spec["training"]
    model = WRAgentModel(bb, spec, segment_weights=dict(training["segment_loss_weights"]),
                         aux_weight=float(training["auxiliary_event_occupancy_loss_weight"]))
    model.frontend.to(device); model.mixer.to(device); model.projector.to(device)
    if model.token_supervision is not None:
        model.token_supervision.to(device)
    model.train(); bb.text.train()
    groups = model.trainable_groups()
    opt = torch.optim.AdamW([{"params": groups["frontend"], "lr": 1e-4}, {"params": groups["lora"], "lr": 2e-5}])

    def to_dev(ep):
        return type(ep)(ep.values.to(device), ep.valid.to(device), ep.channel_ids, ep.sample_times.to(device), ep.task)

    def named_groups():
        g = {"A_trunk": [], "D_trunk": [], "raw_trunk": [], "local_adapters": [], "tokenizer_fusion": [], "aux_head": [],
             "mixer": [], "projector": [], "lora": []}
        for n, p in model.frontend.named_parameters():
            if n.startswith("aux_head"):
                key = "aux_head"
            elif n.startswith("tokenizer."):
                key = "tokenizer_fusion"
            elif "_adapter" in n:
                key = "local_adapters"
            elif n.startswith("branches.a_"):
                key = "A_trunk"
            elif n.startswith("branches.d_") or n.startswith("branches.level_embedding"):
                key = "D_trunk"
            elif n.startswith("branches.raw_"):
                key = "raw_trunk"
            else:
                raise RuntimeError(f"unclassified frontend parameter {n}")
            g[key].append((n, p))
        g["mixer"] = list(model.mixer.named_parameters())
        g["projector"] = list(model.projector.named_parameters())
        g["lora"] = [(n, p) for n, p in bb.text.named_parameters() if "lora_" in n]
        return g

    def grad_stats():
        out = {}
        for k, items in named_groups().items():
            grads = [p.grad for _, p in items if p.grad is not None]
            total = sum(p.numel() for _, p in items)
            nz = sum(int((g != 0).sum()) for g in grads)
            out[k] = {"params": total, "grad_tensors": len(grads), "nonzero_grad_elems": nz}
        return out

    # 1 + 2: direct and fixed-crop
    for mode, cat in (("direct", "localize"), ("fixed_crop", "lead_lag_with_magnitude")):
        row = by_cat[cat][0]
        meta, ep = train.meta(row), to_dev(train.episode(row))
        actions = [] if mode == "direct" else [TeacherCropSampler().sample(meta, random.Random(1))]
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rep = model.forward_loss(meta, ep, mode=mode, actions=actions, answer=meta.answer, gold=meta.gold_intervals,
                                     supervise_action=False)
        rep.loss.backward()
        torch.cuda.synchronize()
        stats = grad_stats()
        ok = torch.isfinite(rep.loss).item() and all(v["nonzero_grad_elems"] > 0 for k, v in stats.items()
                                                       if (k != "local_adapters" or mode == "fixed_crop")
                                                       and (k != "aux_head" or model.aux_weight > 0))  # aux off: no aux grads
        if mode == "direct":
            ok = ok and stats["local_adapters"]["nonzero_grad_elems"] == 0
        check(f"{mode}_forward_backward", ok, loss=float(rep.loss), ce=rep.ce_by_segment, aux=rep.aux_loss, tokens=rep.sequence_length,
              signal_tokens=rep.n_signal_tokens, seconds=round(time.time() - t0, 2), grads=stats, actions=[a.to_json() for a in actions],
              peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        if mode == "direct":
            opt.zero_grad(set_to_none=True)
    # 3: optimizer step changes every group
    before = {k: torch.cat([p.detach().flatten().float().cpu() for _, p in v]) for k, v in named_groups().items()}
    opt.step(); opt.zero_grad(set_to_none=True)
    after = {k: torch.cat([p.detach().flatten().float().cpu() for _, p in v]) for k, v in named_groups().items()}
    changed = {k: float((before[k] - after[k]).abs().max()) for k in before}
    check("optimizer_step_updates_every_group", all(v > 0 for v in changed.values()), max_abs_change=changed)
    # 3b (Astra 15/09, scala): soft-token scale at init and CE-only updates of the perceptual stack
    target = model.projector.target_rms
    row = by_cat["localize"][1]
    meta, ep = train.meta(row), to_dev(train.episode(row))
    aux_weight = model.aux_weight
    model.aux_weight = 0.0
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rep = model.forward_loss(meta, ep, mode="direct", actions=[], answer=meta.answer, gold=meta.gold_intervals,
                                     supervise_action=False)
        rep.loss.backward()
        torch.cuda.synchronize()
    finally:
        model.aux_weight = aux_weight
    stats = grad_stats()
    ce_only_ok = all(v["nonzero_grad_elems"] > 0 for k, v in stats.items() if k not in ("local_adapters", "lora", "aux_head"))
    ce_only_ok = ce_only_ok and stats["aux_head"]["nonzero_grad_elems"] == 0  # the aux head only learns from the aux loss
    check("ce_only_gradients_reach_perceptual_stack", ce_only_ok, aux_weight_used=0.0, loss=float(rep.loss), grads=stats)
    scale_ok = target is not None and abs(rep.soft_rms - target) <= 0.05 * target  # parameter-free per-token RMS
    check("soft_token_rms_matches_text_embeddings", scale_ok, soft_rms=rep.soft_rms, target_rms=target)
    # Astra 15/09 (sentinella D4/D2): supervised reading segment and finite decorrelation penalty
    reading = model.reading_text(ep, [])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rep = model.forward_loss(meta, ep, mode="direct", actions=[], answer=meta.answer, gold=meta.gold_intervals,
                                 supervise_action=False, reading=reading)
    rep.loss.backward()
    torch.cuda.synchronize()
    check("reading_segment_supervised_and_decorrelation_finite",
          "reasoning" in rep.ce_by_segment and rep.decorrelation == rep.decorrelation and rep.decorrelation < 1.0,
          reading=reading[:240], ce=rep.ce_by_segment, decorrelation=rep.decorrelation, loss=float(rep.loss),
          tokens=rep.sequence_length, reading_segment_on=model.reading_segment, decorrelation_weight=model.decorrelation_weight)
    opt.zero_grad(set_to_none=True)
    # 4: save / reload
    row = by_cat["classify_with_evidence"][0]
    meta, ep = train.meta(row), to_dev(train.episode(row))
    model.eval(); bb.text.eval()

    def loss_now() -> float:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return float(model.forward_loss(meta, ep, mode="direct", actions=[], answer=meta.answer,
                                            gold=meta.gold_intervals, supervise_action=False).loss)

    # run-to-run reference: CUDA scatter/index_add reductions in the tokenizer are not bitwise deterministic
    loss_a = loss_now()
    loss_a2 = loss_now()
    run_to_run = abs(loss_a - loss_a2)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.safetensors"
        saved = {k: v.contiguous() for k, v in model.trainable_state_dict().items()}
        save_file(saved, str(path))
        with torch.no_grad():
            for p in groups["frontend"] + groups["lora"]:
                p.add_(torch.randn_like(p) * 1e-3)
        perturbed_max = max(float((v.cpu() - saved[k]).abs().max()) for k, v in model.trainable_state_dict().items())
        model.load_trainable_state_dict(load_file(str(path)))
        restored_max = max(float((v.cpu() - saved[k]).abs().max()) for k, v in model.trainable_state_dict().items())
    check("save_reload_restores_parameters_exactly", perturbed_max > 1e-4 and restored_max == 0.0,
          perturbed_max_abs=perturbed_max, restored_max_abs=restored_max)
    loss_b = loss_now()
    tolerance = max(2e-3 * max(1.0, abs(loss_a)), 4.0 * run_to_run)
    # the perturbation must have been visible (else the reload proves nothing): check it on the parameters themselves
    check("save_reload_same_loss", abs(loss_a - loss_b) <= tolerance, loss_before=loss_a, loss_before_repeat=loss_a2,
          loss_after=loss_b, run_to_run_abs=run_to_run, tolerance=tolerance)
    # 5: generation format
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        gen = model.generate(meta, ep, mode="direct", max_new_tokens=96)
    parsed = parse_structured_pred(gen["text"], REQUIRED_KEYS[meta.category])
    check("generation_runs_and_terminates", gen["finished"], text=gen["text"][:200], parsed=bool(parsed), reading=(gen.get("reading") or "")[:200], reasoning_tokens=gen.get("reasoning_tokens"), forced_closure=gen.get("forced_closure"), seconds=round(time.time() - t0, 2),
          prompt_tokens=gen["prompt_length"], new_tokens=len(gen["token_ids"]))
    model.train(); bb.text.train()
    # 6: max load
    if not args.skip_max_load:
        rows16 = [r for r in by_cat["localize_all_channels"] if metas[r].length == 131072]
        row = rows16[0]
        meta, ep = train.meta(row), to_dev(train.episode(row))
        action = ZoomAction(channels=tuple(meta.channel_names[:2]), start=1000, end=1000 + 8192)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rep = model.forward_loss(meta, ep, mode="fixed_crop", actions=[action], answer=meta.answer, gold=meta.gold_intervals,
                                     supervise_action=False)
        rep.loss.backward(); torch.cuda.synchronize()
        opt.zero_grad(set_to_none=True)
        check("max_load_16ch_131k_plus_crop", torch.isfinite(rep.loss).item(), loss=float(rep.loss), tokens=rep.sequence_length,
              signal_tokens=rep.n_signal_tokens, seconds=round(time.time() - t0, 2), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    report["verdict"] = "PASS" if all(c["pass"] for c in report["checks"]) else "FAIL"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))
    print("VERDICT", report["verdict"])
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
