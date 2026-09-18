#!/usr/bin/env python3
"""Linear probes on the perceptual representations (Astra 15/09, P1-d D4 fallback): is the information THERE at all?

Three frozen representations of the global view of each recording, taken from one checkpoint (or from the random
initialisation with ``--checkpoint none``):
  H0    fused descriptors, before the mixer      [N, 512]
  Z     mixer output                             [N, 512]
  SOFT  projector output (what the backbone sees) [N, 2560]

Per-token targets, all computable from the signal at the resolution the tokenizer represents:
  position   the token's own bin index on the reading grid (64 bins) -- ridge regression on the normalised position,
             reported as R^2 and as the median absolute error in bins;
  top_bin    whether the token's bin is one of the three highest Haar-energy bins of its channel (logistic probe,
             reported as AUC-free balanced accuracy at the 0.5 threshold and as average precision).
Recordings are split disjointly into train and validation, so nothing is shared between fit and evaluation.
A probe that succeeds on H0/Z and fails on SOFT implicates the projector; success everywhere moves the diagnosis to the
interface with the backbone; a negative probe alone does not prove the information is absent (Astra).

Usage:
  python tools/linear_probe_tokens.py --config configs/p1dA_mi300x.toml --checkpoint <dir|none> --out <json> [--rows 120]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safetensors.torch import load_file  # noqa: E402

from wavelet_rope.checkpoint import inspect_foreign_checkpoint  # noqa: E402
from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import AnomalyXLSplit, stratified_subset  # noqa: E402
from wavelet_rope.haar_selector import _normalised_energy_scores  # noqa: E402
from wavelet_rope.perceptual_rope import PerceptualMemory, PerceptualMixer, Projector  # noqa: E402
from wavelet_rope.signal_contract import ViewKind, compute_recording_normalization, make_view  # noqa: E402
from wavelet_rope.tokenizer import PerceptualFrontend  # noqa: E402

TOP_BINS = 3


def ridge(x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """Closed-form ridge with an intercept column; x [N, D], y [N] -> weights [D + 1]."""

    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    xb = torch.cat((x, ones), dim=1)
    d = xb.shape[1]
    penalty = torch.eye(d, dtype=xb.dtype, device=xb.device)
    penalty[-1, -1] = 0.0  # never shrink the intercept: with a large lambda that alone destroys the fit
    a = xb.T @ xb + lam * penalty
    return torch.linalg.solve(a, xb.T @ y)


def apply_ridge(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.cat((x, torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)), dim=1) @ w


def logistic(x: torch.Tensor, y: torch.Tensor, *, steps: int = 400, lam: float = 1e-3) -> torch.Tensor:
    """L2-regularised logistic regression by full-batch LBFGS-free gradient descent (deterministic)."""

    xb = torch.cat((x, torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)), dim=1)
    w = torch.zeros(xb.shape[1], dtype=xb.dtype, device=xb.device, requires_grad=True)
    pos = float(y.mean().clamp(1e-6, 1 - 1e-6))
    weight = torch.where(y > 0.5, torch.full_like(y, 0.5 / pos), torch.full_like(y, 0.5 / (1.0 - pos)))
    opt = torch.optim.LBFGS([w], max_iter=steps, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xb @ w, y, weight=weight) + lam * w[:-1].pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="checkpoint directory, or 'none' for the random initialisation")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=120)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train-share", type=float, default=0.7)
    ap.add_argument("--ridge-lambda", type=float, default=10.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    t0 = time.time()

    cfg = load_runtime_config(args.config, validate_paths=False)
    raw, spec = cfg.as_dict(), dict(cfg.spec)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    frontend = PerceptualFrontend(spec).to(device).eval()
    mixer = PerceptualMixer().to(device).eval()
    embed_rms = None
    projector = Projector(int(spec["backbone"]["projector_output_dim"]), target_rms=0.013028954155743122).to(device).eval()
    step = None
    if args.checkpoint != "none":
        ref = inspect_foreign_checkpoint(args.checkpoint)
        state = load_file(str(ref.path / "model.safetensors"))
        step = ref.optimizer_step
        for prefix, module in (("frontend", frontend), ("mixer", mixer), ("projector", projector)):
            module.load_state_dict({k[len(prefix) + 1:]: v.to(device) for k, v in state.items()
                                    if k.startswith(prefix + ".")}, strict=True)

    root = Path(raw["dataset"]["root"])
    val = AnomalyXLSplit(root / raw["dataset"]["validation_relative_path"], "validation", 520)
    rows = stratified_subset(val.metas(), args.rows, args.seed)
    n_bins = int(spec["tokenizer"]["global_per_channel"]["A"])

    features = {"H0": [], "Z": [], "SOFT": []}
    targets = {"position": [], "top_bin": []}
    groups: list[int] = []
    with torch.no_grad():
        for order, row in enumerate(rows):
            episode = val.episode(row)
            episode = type(episode)(episode.values.to(device), episode.valid.to(device), episode.channel_ids,
                                    episode.sample_times.to(device), episode.task)
            stats = compute_recording_normalization(episode)
            view = make_view(episode, stats, kind=ViewKind.GLOBAL)
            tokens = frontend.encode_view(view, recording_length=episode.length)
            n_valid = int(tokens.valid.sum())
            if n_valid == 0:
                continue
            index = tokens.sort_order[:n_valid]
            h0 = tokens.descriptors[index].float()
            memory = PerceptualMemory(t_origin=float(episode.sample_times[0]),
                                      dt_ref=episode.dt if episode.dt is not None else 1.0)
            mem_view = memory.append(h0, tokens.anchor_time[index], kind="global", start=view.start, end=view.end,
                                     channel_ids=view.channel_ids)
            h_mem, p_mem, tau_mem = memory.stacked()
            z = mixer(h0[None], mem_view.p[None], mem_view.tau[None], h_mem[None], p_mem[None], tau_mem[None])[0].float()
            soft = projector(z).float()
            # per-token targets, from the signal only
            normalised = torch.where(episode.valid, (episode.values.float() - stats.median[:, None].to(device))
                                     / stats.scale[:, None].to(device), torch.zeros_like(episode.values.float()))
            scores = _normalised_energy_scores(normalised.cpu(), n_bins)  # [C, n_bins]
            top = {c: set(torch.argsort(scores[c], descending=True)[:TOP_BINS].tolist()) for c in range(scores.shape[0])}
            start = torch.tensor([float(s) for s in tokens.support_start[index].tolist()], device=device)
            end = torch.tensor([float(e) for e in tokens.support_end[index].tolist()], device=device)
            centre = ((start + end) * 0.5).clamp(0, episode.length - 1)
            position = centre / float(episode.length)
            bins = (position * n_bins).floor().clamp(0, n_bins - 1).to(torch.int64)
            channels = tokens.channel_index[index].to(torch.int64)
            is_top = torch.tensor([1.0 if int(b) in top.get(int(c), set()) else 0.0
                                   for b, c in zip(bins.tolist(), channels.tolist())], device=device)
            features["H0"].append(h0.cpu()); features["Z"].append(z.cpu()); features["SOFT"].append(soft.cpu())
            targets["position"].append(position.cpu()); targets["top_bin"].append(is_top.cpu())
            groups.append(order)
            if (order + 1) % 20 == 0:
                print(f"  encoded {order + 1}/{len(rows)} recordings", flush=True)

    n_train = int(len(groups) * args.train_share)
    report: dict = {"checkpoint": args.checkpoint, "optimizer_step": step, "n_recordings": len(groups),
                    "n_train_recordings": n_train, "n_bins": n_bins, "seed": args.seed, "probes": {}}
    for name in ("H0", "Z", "SOFT"):
        xs = features[name]
        xtr = torch.cat(xs[:n_train]).double(); xva = torch.cat(xs[n_train:]).double()
        mu = xtr.mean(0, keepdim=True)
        sd = xtr.std(0, keepdim=True)
        # A dimension that is (nearly) constant on the training tokens must not be blown up on validation: give it
        # scale 1 so that, after centring, it simply contributes ~0.
        sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
        xtr = (xtr - mu) / sd; xva = (xva - mu) / sd
        ytr = torch.cat(targets["position"][:n_train]).double(); yva = torch.cat(targets["position"][n_train:]).double()
        # Ridge strength chosen on an INNER split of the training recordings (never on the evaluation tokens).
        # Tune on a prefix of complete recordings, never on a token prefix.
        inner_recordings = max(1, int(n_train * 0.8))
        inner_recordings = min(inner_recordings, max(1, n_train - 1))
        inner = sum(int(item.shape[0]) for item in xs[:inner_recordings])
        best = (float("inf"), args.ridge_lambda)
        if inner < int(xtr.shape[0]):
            for lam in (1.0, 10.0, 1e2, 1e3, 1e4, 1e5):
                wi = ridge(xtr[:inner], ytr[:inner], lam)
                err = float(((ytr[inner:] - apply_ridge(xtr[inner:], wi)) ** 2).mean())
                if err < best[0]:
                    best = (err, lam)
        lam = best[1]
        w = ridge(xtr, ytr, lam)
        pred = apply_ridge(xva, w)
        ss_res = float(((yva - pred) ** 2).sum()); ss_tot = float(((yva - yva.mean()) ** 2).sum())
        median_bins = float(((yva - pred).abs() * n_bins).median())
        fitted = apply_ridge(xtr, w)
        train_r2 = 1.0 - float(((ytr - fitted) ** 2).sum()) / max(float(((ytr - ytr.mean()) ** 2).sum()), 1e-12)
        pc = torch.stack((pred, yva))
        pearson = float(torch.corrcoef(pc)[0, 1])
        ttr = torch.cat(targets["top_bin"][:n_train]).double(); tva = torch.cat(targets["top_bin"][n_train:]).double()
        wl = logistic(xtr, ttr)
        logits = torch.cat((xva, torch.ones((xva.shape[0], 1), dtype=xva.dtype)), dim=1) @ wl
        hit = (logits > 0).double()
        tpr = float(hit[tva > 0.5].mean()) if float(tva.sum()) else float("nan")
        tnr = float((1 - hit)[tva < 0.5].mean()) if float((1 - tva).sum()) else float("nan")
        report["probes"][name] = {
            "position_r2": 1.0 - ss_res / max(ss_tot, 1e-12),
            "position_r2_train": train_r2,
            "position_pearson_val": pearson,
            "position_median_error_bins": median_bins,
            "top_bin_balanced_accuracy": 0.5 * (tpr + tnr),
            "top_bin_base_rate": float(tva.mean()),
            "n_tokens_train": int(xtr.shape[0]), "n_tokens_val": int(xva.shape[0]), "dim": int(xtr.shape[1]),
            "ridge_lambda": lam,
        }
        p = report["probes"][name]
        print(f"{name:5s} position R2 val {p['position_r2']:+.4f} train {p['position_r2_train']:+.4f} "
              f"pearson {p['position_pearson_val']:+.3f} (median error {p['position_median_error_bins']:.2f} bins) | "
              f"top-{TOP_BINS} bin balanced accuracy {p['top_bin_balanced_accuracy']:.3f} "
              f"(base rate {p['top_bin_base_rate']:.3f})", flush=True)
    report["seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("LINEAR_PROBE", json.dumps({k: v for k, v in report.items() if k != "probes"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
