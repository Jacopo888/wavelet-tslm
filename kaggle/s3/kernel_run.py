"""TimeRLM wavelet variant - chained training, session 3.

Same proven recipe as the budget run (2xT4, fp32 device_map=auto, SDPA,
gradient checkpointing, frozen LM embeddings, slim checkpoints), plus the
session-chaining pieces:

- warm start: weights-only resume from the session-2 checkpoint, mounted via
  kernel_sources ("timerlm-wavelet-s2" output) and auto-discovered;
- fresh 2000-sample train subset (TSLM_SUBSET_SEED), while val/test rows are
  drawn with a fixed RNG so metrics stay comparable session-over-session;
- no val generations (TSLM_VAL_GEN=0): the presentability check is the final
  test eval; saves ~1.7h per session;
- wall-clock training budget as a belt-and-braces stop before the 12h wall.
"""

import os
import shutil
import subprocess
import sys
import time

WORK = "/kaggle/working"
REPO = os.path.join(WORK, "timerlm")

SESSION = "s3"
SUBSET_SEED = "1043"          # unique per session: s2=1042, s3=1043, ...
TRAIN_BUDGET_S = "30600"      # 8.5h cap: covers 2000 steps even at ~15s/step

OVERLAY = {
    "wavelet_encoder.py": "sft/tslm/src/models/wavelet_encoder.py",
    "wavelet.py": "sft/tslm/src/models/wavelet.py",
    "qwen_backbone.py": "sft/tslm/src/models/qwen_backbone.py",
    "qa_dataset.py": "sft/tslm/src/datasets/anomaly_xl/qa_dataset.py",
    "train.py": "sft/tslm/train.py",
    "wavelet_precise.yaml": "sft/tslm/configs/wavelet_precise.yaml",
    "wavelet_precise_s3.yaml": "sft/tslm/configs/wavelet_precise_s3.yaml",
}

TRANSFORMERS_GIT = "git+https://github.com/huggingface/transformers@5c1c72b"


def sh(cmd, cwd=None):
    print(">>", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def find_patch_dir(marker="wavelet_encoder.py", timeout_s=240):
    deadline = time.time() + timeout_s
    while True:
        print("/kaggle/input contents:",
              os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "MISSING",
              flush=True)
        for root, _dirs, files in os.walk("/kaggle/input"):
            if marker in files:
                print("patch dir found:", root, flush=True)
                return root
        if time.time() > deadline:
            raise RuntimeError("patch dataset not found under /kaggle/input")
        print("patch not mounted yet; retrying in 30s", flush=True)
        time.sleep(30)


def find_prev_checkpoint():
    """Locate the previous session's checkpoint in the mounted kernel output.

    Prefers best_model.pt, falls back to latest_checkpoint.pt / periodic
    checkpoints so a quota-killed previous session is still resumable.
    Fails loudly rather than silently training from scratch.
    """
    best, latest, periodic = [], [], []
    for root, _dirs, files in os.walk("/kaggle/input"):
        if "wavelet" not in root:
            continue
        for f in files:
            p = os.path.join(root, f)
            if f == "best_model.pt":
                best.append(p)
            elif f == "latest_checkpoint.pt":
                latest.append(p)
            elif f.startswith("checkpoint_epoch_") and f.endswith(".pt"):
                periodic.append(p)
    for pool, label in ((best, "best"), (latest, "latest"), (periodic, "periodic")):
        if pool:
            path = max(pool, key=os.path.getmtime)
            print(f"resume checkpoint found ({label}): {path}", flush=True)
            return path
    raise RuntimeError(
        "no previous wavelet checkpoint found under /kaggle/input - "
        "refusing to silently train from scratch"
    )


def run():
    os.chdir(WORK)
    patch = find_patch_dir()
    prev_ckpt = find_prev_checkpoint()
    sh(f"git clone --depth 1 https://github.com/OpenTSLM/TimeRLM {REPO}")
    for src, dst in OVERLAY.items():
        dst_path = os.path.join(REPO, dst)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(os.path.join(patch, src), dst_path)
        print("overlay:", dst, flush=True)

    print("== GPU ==", flush=True)
    sh("nvidia-smi")

    sh(f"{sys.executable} -m pip install -q -e {REPO}/anomalyXL")
    sh(f'{sys.executable} -m pip install -q "transformers @ {TRANSFORMERS_GIT}" '
       f"peft==0.18.1 einops einops-exts datasets pyyaml wandb open_clip_torch")
    sh(f"{sys.executable} -m pip install -q --no-deps "
       f"chronos-forecasting==2.2.2 open-flamingo")
    check = ("import torch; "
             "print('torch', torch.__version__, torch.cuda.get_device_name(0)); "
             "x = torch.zeros(8, device='cuda'); print('cuda ok', (x + 1).sum().item()); "
             "import transformers; print('transformers', transformers.__version__)")
    sh(f'{sys.executable} -c "{check}"')

    sh(f"{sys.executable} scripts/generate_anomalyxl_sweep.py "
       f"--out-root datasets --precise-only --seed 42",
       cwd=os.path.join(REPO, "anomalyXL"))

    tslm = os.path.join(REPO, "sft", "tslm")
    sh(f"{sys.executable} fetch_data.py", cwd=tslm)

    sh("free -m")
    sh("df -h /kaggle/working /root")

    os.environ["TSLM_DEVICE_MAP"] = "auto"
    os.environ["TSLM_SAVE_TRAINABLE_ONLY"] = "1"
    os.environ["TSLM_ATTN_IMPL"] = "sdpa"
    os.environ["TSLM_GRAD_CKPT"] = "1"
    os.environ["TSLM_RESUME_FROM"] = prev_ckpt
    os.environ["TSLM_RESUME_WEIGHTS_ONLY"] = "1"
    os.environ["TSLM_VAL_GEN"] = "0"
    os.environ["TSLM_SUBSET_SEED"] = SUBSET_SEED
    os.environ["TSLM_MAX_TRAIN_SECONDS"] = TRAIN_BUDGET_S
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    sh(f"{sys.executable} train.py --config configs/wavelet_precise_{SESSION}.yaml "
       f"--no-wandb", cwd=tslm)


def salvage_results():
    results = os.path.join(REPO, "sft", "tslm", "results")
    if os.path.isdir(results):
        shutil.copytree(results, os.path.join(WORK, "results_out"),
                        dirs_exist_ok=True)
        for root, _dirs, files in os.walk(os.path.join(WORK, "results_out")):
            for f in files:
                print("result file:",
                      os.path.relpath(os.path.join(root, f), WORK), flush=True)


def main():
    try:
        run()
        print(f"WAVELET {SESSION.upper()} RUN COMPLETE", flush=True)
    finally:
        salvage_results()
        shutil.rmtree(REPO, ignore_errors=True)


if __name__ == "__main__":
    main()
