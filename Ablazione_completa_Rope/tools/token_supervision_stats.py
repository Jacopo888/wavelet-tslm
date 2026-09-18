#!/usr/bin/env python3
"""Generate fixed train statistics for P1-e token supervision.

The selected rows are a deterministic category-stratified subset.  Statistics
are computed from global-view tokens and the signal only; gold annotations are
never read by this tool.

Usage:
  python tools/token_supervision_stats.py --config <config.toml> --out <stats.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import AnomalyXLSplit, stratified_subset  # noqa: E402
from wavelet_rope.signal_contract import ViewKind, compute_recording_normalization, make_view  # noqa: E402
from wavelet_rope.token_supervision import token_supervision_targets  # noqa: E402
from wavelet_rope.tokenizer import PerceptualFrontend  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="resolved TOML containing the train split")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1024, help="stratified rows; 0 means the complete train split")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--n-bins", type=int, default=0, help="override the global reading grid")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rows < 0:
        raise SystemExit("--rows must be non-negative")
    cfg = load_runtime_config(args.config, validate_paths=False)
    raw, spec = cfg.as_dict(), dict(cfg.spec)
    training = spec.get("training", {})
    token_config = training.get("token_supervision", {}) if isinstance(training, dict) else {}
    header_bins = int(spec["tokenizer"]["global_per_channel"]["A"])
    configured_bins = int(token_config.get("n_bins", header_bins))
    if configured_bins != header_bins:
        raise SystemExit("token_supervision.n_bins must match the global ViewHeader.reading_bins")
    n_bins = configured_bins if args.n_bins <= 0 else int(args.n_bins)
    if n_bins != header_bins:
        raise SystemExit("--n-bins must match the global ViewHeader.reading_bins")
    if n_bins < 1:
        raise SystemExit("n_bins must be positive")

    root = Path(raw["dataset"]["root"])
    train = AnomalyXLSplit(root / raw["dataset"]["train_relative_path"], "train", int(raw["dataset"]["train_rows"]))
    count = len(train) if args.rows == 0 else args.rows
    rows = stratified_subset(train.metas(), count, args.seed)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    frontend = PerceptualFrontend(spec).to(device).eval()

    collected: list[Tensor] = []
    with torch.no_grad():
        for position, row in enumerate(rows, start=1):
            episode = train.episode(row)
            episode = type(episode)(episode.values.to(device), episode.valid.to(device), episode.channel_ids,
                                    episode.sample_times.to(device), None)
            recording_stats = compute_recording_normalization(episode)
            view = make_view(episode, recording_stats, kind=ViewKind.GLOBAL)
            tokens = frontend.encode_view(view, recording_length=episode.length)
            target_position, target_energy, target_peak, valid = token_supervision_targets(
                tokens,
                view.global_normalized,
                episode.length,
                n_bins=n_bins,
                valid_trace=view.valid,
                channel_indices=view.channel_indices,
            )
            if bool(valid.any()):
                values = torch.stack(
                    (target_position[valid], target_energy[valid], target_peak[valid]), dim=1
                ).cpu()
                collected.append(values)
            if position % 100 == 0:
                print(f"  encoded {position}/{len(rows)} recordings", flush=True)

    if not collected:
        raise RuntimeError("the selected train subset produced no valid token targets")
    values = torch.cat(collected, dim=0).double()
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(1e-6)
    document = {
        "position": {"mean": float(mean[0]), "std": float(std[0])},
        "energy": {"mean": float(mean[1]), "std": float(std[1])},
        "peak": {"mean": float(mean[2]), "std": float(std[2])},
        "rows": rows,
        "n_bins": n_bins,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"TOKEN_SUPERVISION_STATS rows={len(rows)} tokens={values.shape[0]} n_bins={n_bins} out={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
