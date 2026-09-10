"""Exact-load check for a trainable-only TSLM checkpoint (CPU, no GPU needed).

Usage (on the node, from /root/timerlm/sft/tslm, same env as the eval worker):
  CUDA_VISIBLE_DEVICES= TSLM_TORCH_DTYPE=bfloat16 TSLM_SAVE_TRAINABLE_ONLY=1 \
  TSLM_QWEN_REVISION=<rev> ~/venv/bin/python check_va1_exact_load.py <config.yaml> <ckpt.pt> [out.json]

Builds the model exactly as train.py --eval-only does (load_model + configure_training_mode),
then checks that the set of trainable parameter names/shapes is IDENTICAL to the checkpoint's
model_state keys/shapes, that every checkpoint tensor lands in the model, that no trainable
tensor would silently keep its init value (the strict=False path in load_checkpoint), and that
no tensor contains NaN/Inf. Exit 0 only if everything matches.
"""
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from train import ExperimentConfig, configure_training_mode, load_model  # noqa: E402


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    cfg_path, ckpt_path = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    config = ExperimentConfig.from_yaml(cfg_path)
    model = load_model(config, "cpu")
    configure_training_mode(model, config)
    trainable = {n: tuple(p.shape) for n, p in model.named_parameters() if p.requires_grad}
    model_keys = set(model.state_dict().keys())

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ms = ck["model_state"]
    ck_shapes = {k: tuple(v.shape) for k, v in ms.items()}

    missing_in_ckpt = sorted(set(trainable) - set(ck_shapes))     # trainable but not saved -> would keep init
    extra_in_ckpt = sorted(set(ck_shapes) - set(trainable))        # saved but not trainable
    not_in_model = sorted(k for k in ck_shapes if k not in model_keys)
    shape_mismatch = sorted(k for k in set(ck_shapes) & set(trainable) if ck_shapes[k] != trainable[k])
    non_finite = sorted(k for k, v in ms.items() if not torch.isfinite(v.float()).all())

    missing, unexpected = model.load_state_dict(ms, strict=False)
    trainable_kept_from_init = sorted(k for k in missing if k in trainable)

    n_params_ckpt = int(sum(v.numel() for v in ms.values()))
    n_params_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    report = {
        "config": cfg_path,
        "checkpoint": ckpt_path,
        "checkpoint_sha256": sha256(ckpt_path),
        "checkpoint_meta": {k: ck.get(k) for k in ("epoch", "batch_idx", "val_loss", "best_val_metric", "timestamp")},
        "tensors_in_checkpoint": len(ms),
        "trainable_tensors_in_model": len(trainable),
        "params_in_checkpoint": n_params_ckpt,
        "trainable_params_in_model": n_params_trainable,
        "frozen_tensors_kept_from_init": len(missing) - len(trainable_kept_from_init),
        "by_prefix": {},
        "missing_in_ckpt (trainable, would keep init)": missing_in_ckpt,
        "extra_in_ckpt (saved but frozen)": extra_in_ckpt,
        "not_in_model": not_in_model,
        "shape_mismatch": shape_mismatch,
        "unexpected_on_load": list(unexpected),
        "trainable_kept_from_init_on_load": trainable_kept_from_init,
        "non_finite": non_finite,
        "dtypes": sorted({str(v.dtype) for v in ms.values()}),
    }
    for k in ms:
        p = k.split(".")[0]
        report["by_prefix"][p] = report["by_prefix"].get(p, 0) + 1
    ok = not (missing_in_ckpt or extra_in_ckpt or not_in_model or shape_mismatch or unexpected
              or trainable_kept_from_init or non_finite) and n_params_ckpt == n_params_trainable
    report["EXACT_LOAD_OK"] = bool(ok)
    text = json.dumps(report, indent=1, default=str)
    print(text)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
