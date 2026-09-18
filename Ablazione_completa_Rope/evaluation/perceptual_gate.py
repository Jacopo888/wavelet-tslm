#!/usr/bin/env python3
"""Run the frozen validation perceptual-content gate.

The 4B backbone is intentionally loaded only here.  The reusable panel,
perturbation, bootstrap, and verdict code lives in ``wavelet_rope`` so it can
be tested on CPU with small fake models.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wavelet_rope.backbone import load_backbone  # noqa: E402
from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import AnomalyXLSplit  # noqa: E402
from wavelet_rope.model import WRAgentModel  # noqa: E402
from wavelet_rope.perceptual_gate import (  # noqa: E402
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_CROPS_PER_ROW,
    PanelCrop,
    build_panel,
    evaluate_gate,
    load_panel,
    save_panel,
    substitution_sources,
    write_gate_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="checkpoint directory or model.safetensors")
    parser.add_argument("--config", required=True, help="resolved runtime TOML")
    parser.add_argument("--panel", default=None, help="panel JSON to build or reuse")
    parser.add_argument("--rows", type=int, default=0, help="validation rows when building (0 = all eligible)")
    parser.add_argument("--crops-per-row", type=int, default=DEFAULT_CROPS_PER_ROW)
    parser.add_argument("--negative-share", type=float, default=1.0 / DEFAULT_CROPS_PER_ROW)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0", help="evaluation device; defaults to training's cuda:0")
    parser.add_argument("--flex-dynamic", type=int, default=1)
    return parser.parse_args()


def _load_model(config, checkpoint: str | Path, device: torch.device, *, flex_dynamic: bool) -> WRAgentModel:
    raw = config.as_dict()
    spec = dict(config.spec)
    runtime = raw["runtime"]
    snapshot = raw["backbone"]["snapshot"]["root"]
    bb = load_backbone(
        snapshot,
        backend=runtime["attention_backend"],
        pad_to=int(runtime["attention_qk_pad_to"]),
        device=device,
        dtype=torch.bfloat16,
        gradient_checkpointing=bool(runtime["gradient_checkpointing"]),
        flex_dynamic=flex_dynamic,
    )
    training = spec["training"]
    model = WRAgentModel(
        bb,
        spec,
        max_calls=int(spec["agent"]["max_calls"]),
        max_views=int(spec["agent"]["max_local_channel_views"]),
        segment_weights=dict(training["segment_loss_weights"]),
        aux_weight=float(training["auxiliary_event_occupancy_loss_weight"]),
    )
    model.frontend.to(device)
    model.mixer.to(device)
    model.projector.to(device)

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "model.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint model.safetensors at {checkpoint_path}")
    from safetensors.torch import load_file

    model.load_trainable_state_dict(load_file(str(checkpoint_path), device="cpu"))
    model.eval()
    model.bb.text.eval()
    return model


def _panel_path(value: str | None, out_dir: Path) -> Path:
    if value is None:
        return out_dir / "panel.json"
    path = Path(value)
    return path / "panel.json" if path.is_dir() else path


def _complete_panel(panel: list[PanelCrop], validation: AnomalyXLSplit, seed: int) -> list[PanelCrop]:
    completed: list[PanelCrop] = []
    for crop in panel:
        try:
            meta = validation.meta(crop.row_index)
        except (IndexError, ValueError):
            matches = [item for item in validation.metas() if item.id == crop.row_id]
            if len(matches) != 1:
                raise ValueError(f"panel row {crop.row_id!r} is not uniquely present in validation")
            meta = matches[0]
            crop = replace(crop, row_index=meta.row_index)
        sources = dict(crop.substitution_sources)
        if any(channel not in sources for channel in crop.channels):
            sources.update(substitution_sources(meta, crop, seed=seed))
        completed.append(replace(crop, substitution_sources=sources))
    return completed


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_runtime_config(args.config, validate_paths=False)
    cfg.require_resolved()
    raw = cfg.as_dict()
    seed = int(raw["seed"] if args.seed is None else args.seed)

    dataset = raw["dataset"]
    validation_path = Path(dataset["root"]) / dataset["validation_relative_path"]
    validation = AnomalyXLSplit(
        validation_path,
        "validation",
        int(dataset.get("validation_rows", 0)) or None,
    )
    panel_path = _panel_path(args.panel, out_dir)
    panel_reused = panel_path.is_file()
    if panel_reused:
        panel = load_panel(panel_path)
    else:
        panel = build_panel(
            validation.metas(),
            seed=seed,
            rows=None if args.rows == 0 else args.rows,
            crops_per_row=args.crops_per_row,
            negative_share=args.negative_share,
        )
    had_all_sources = all(
        all(channel in crop.substitution_sources for channel in crop.channels) for crop in panel
    )
    panel = _complete_panel(panel, validation, seed)
    # Persist any legacy panel entries whose per-channel source windows were
    # not recorded yet; subsequent checkpoints then use exactly the same panel.
    if not panel_reused or not had_all_sources:
        save_panel(panel_path, panel, seed=seed, crops_per_row=args.crops_per_row)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the perceptual-gate CLI requires the remote CUDA backbone; CUDA is unavailable")
    model = _load_model(cfg, args.checkpoint, device, flex_dynamic=bool(args.flex_dynamic))
    report = evaluate_gate(
        model,
        validation,
        panel,
        device=device,
        seed=seed,
        n_resamples=args.bootstrap_resamples,
        max_new_tokens=args.max_new_tokens,
        autocast_dtype=torch.bfloat16 if device.type == "cuda" else None,
    )
    report["checkpoint"] = str(Path(args.checkpoint).resolve())
    report["config"] = str(Path(args.config).resolve())
    report["panel"] = str(panel_path.resolve())
    report["validation_parquet"] = str(validation_path.resolve())
    report_path, summary_path = write_gate_outputs(out_dir, report)
    print(f"VERDICT: {report['verdict']}")
    print(f"Wrote {report_path}")
    print(f"Wrote {summary_path}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
