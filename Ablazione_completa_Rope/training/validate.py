#!/usr/bin/env python3
"""Official validation of one checkpoint (Astra 15/09 §3): direct-mode generation with 768 final tokens on the
validation split (all 520 rows by default), primary per category via the vendored scorer, invalid formats counted
with the strict parser, and the tie-break metric (teacher-crop reading loss on the same rows).

Reuses the Trainer machinery so data, model, checkpoint resolution and device handling are identical to training.
Never touches the test split.  Usage:
  python training/validate.py --config configs/p1_mi300x.toml --run-id p1-main --checkpoint auto [--rows 520] [--tag official]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from safetensors.torch import load_file  # noqa: E402

from train import Trainer  # noqa: E402
from wavelet_rope.data import stratified_subset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--checkpoint", default="auto", help="auto = latest complete checkpoint of the run | <dir>")
    ap.add_argument("--rows", type=int, default=0, help="0 = every validation row; N = stratified subset")
    ap.add_argument("--tag", default="official")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--flex-dynamic", type=int, default=1)
    args = ap.parse_args()
    trainer_args = argparse.Namespace(config=args.config, run_id=args.run_id, resume="none", max_optimizer_steps=0,
                                      limit_examples=0, val_every=0, val_subset=0, val_full_at_pass_end=0,
                                      max_new_tokens=args.max_new_tokens, flex_dynamic=args.flex_dynamic,
                                      checkpointing_min_tokens=0)
    trainer = Trainer(trainer_args)
    trainer.setup_data()
    trainer.setup_model()
    from wavelet_rope.checkpoint import inspect_foreign_checkpoint, latest_complete_dir
    from wavelet_rope.errors import FingerprintMismatchError

    target = latest_complete_dir(trainer.store.root) if args.checkpoint == "auto" else Path(args.checkpoint)
    if target is None:
        raise SystemExit("no complete checkpoint found for this run")
    try:
        ref = trainer.store.validate(target)
        foreign = None
    except FingerprintMismatchError:
        # The tool tree changed after the run started (source hash is part of the experiment fingerprint):
        # evaluation is still legitimate, but the two fingerprints are recorded explicitly (Astra 14/09: manifests).
        ref = inspect_foreign_checkpoint(target)
        foreign = {"checkpoint_fingerprint": ref.trainer_state["experiment_fingerprint"],
                   "tool_fingerprint": trainer.fingerprint}
        trainer.say(f"WARNING: checkpoint {target.name} belongs to fingerprint "
                    f"{foreign['checkpoint_fingerprint'][:12]}, evaluation tool tree is {trainer.fingerprint[:12]} (recorded)")
    trainer.model.load_trainable_state_dict(load_file(str(ref.path / "model.safetensors")))
    trainer.log({"event": "validation_checkpoint", "checkpoint": ref.path.name, "foreign_fingerprint": foreign})
    trainer.state["optimizer_step"] = ref.optimizer_step
    trainer.bb.ctx.lam = float(ref.trainer_state["wildcard_lambda"])
    trainer.say(f"validating {ref.path.name} (lambda {trainer.bb.ctx.lam}) on "
                f"{'all' if not args.rows else args.rows} validation rows, {args.max_new_tokens} final tokens")
    rows = list(range(len(trainer.val))) if not args.rows else stratified_subset(trainer.val.metas(), args.rows, trainer.seed)
    with torch.no_grad():
        result = trainer.validate(rows=rows, tag=f"{args.tag}_{ref.path.name}")
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
