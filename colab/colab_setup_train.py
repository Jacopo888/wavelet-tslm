"""On-VM setup + training launcher for Colab wavelet sessions.

Run with:  colab exec -s <session> -f colab_setup_train.py

Expects three files uploaded to the VM beforehand:
  /content/session.json        {"config_file": "wavelet_precise_c1.yaml",
                                "subset_seed": 1044, "train_budget_s": 10800}
  /content/patch.tar.gz        tar of esperimenti/kaggle/patch/ (overlay files)
  /content/prev_best_model.pt  newest checkpoint from the previous session

Does the same setup as the Kaggle kernels (clone, overlay, deps, gold seed-42
benchmark, HF train shards), then launches train.py DETACHED via nohup-style
Popen writing to /content/train.log, and returns. Poll with poll_train.py;
on exit the wrapper writes /content/train_done.txt with the exit code.

Colab specifics vs Kaggle: single T4 15GB -> fp16 + AMP (TSLM_AMP=1), no
TSLM_DEVICE_MAP; torch is the preinstalled Colab build (already CUDA-matched).
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile

WORK = "/content"
REPO = os.path.join(WORK, "timerlm")
PATCH = os.path.join(WORK, "patch")

OVERLAY = {
    "wavelet_encoder.py": "sft/tslm/src/models/wavelet_encoder.py",
    "wavelet.py": "sft/tslm/src/models/wavelet.py",
    "qwen_backbone.py": "sft/tslm/src/models/qwen_backbone.py",
    "qa_dataset.py": "sft/tslm/src/datasets/anomaly_xl/qa_dataset.py",
    "train.py": "sft/tslm/train.py",
}

TRANSFORMERS_GIT = "git+https://github.com/huggingface/transformers@5c1c72b"


def sh(cmd, cwd=None):
    print(">>", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def main():
    with open(os.path.join(WORK, "session.json")) as f:
        sess = json.load(f)
    config_file = sess["config_file"]
    for required in ("patch.tar.gz", "prev_best_model.pt"):
        if not os.path.exists(os.path.join(WORK, required)):
            raise RuntimeError(f"missing /content/{required} - upload it first")

    if os.path.isdir(PATCH):
        shutil.rmtree(PATCH)
    os.makedirs(PATCH)
    with tarfile.open(os.path.join(WORK, "patch.tar.gz")) as tar:
        tar.extractall(PATCH)

    if not os.path.isdir(REPO):
        sh(f"git clone --depth 1 https://github.com/OpenTSLM/TimeRLM {REPO}")

    for src, dst in OVERLAY.items():
        dst_path = os.path.join(REPO, dst)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(os.path.join(PATCH, src), dst_path)
        print("overlay:", dst, flush=True)
    # All configs travel too, so any wavelet_precise_*.yaml is selectable.
    cfg_dir = os.path.join(REPO, "sft", "tslm", "configs")
    for f in os.listdir(PATCH):
        if f.endswith(".yaml"):
            shutil.copy(os.path.join(PATCH, f), os.path.join(cfg_dir, f))
            print("overlay config:", f, flush=True)

    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")

    sh(f"{sys.executable} -m pip install -q -e {REPO}/anomalyXL")
    sh(f'{sys.executable} -m pip install -q "transformers @ {TRANSFORMERS_GIT}" '
       f"peft==0.18.1 einops einops-exts datasets pyyaml wandb open_clip_torch")
    sh(f"{sys.executable} -m pip install -q --no-deps "
       f"chronos-forecasting==2.2.2 open-flamingo")
    sh(f'{sys.executable} -c "import torch; print(torch.__version__, '
       f"torch.cuda.get_device_name(0)); import transformers; "
       f'print(transformers.__version__)"')

    sh(f"{sys.executable} scripts/generate_anomalyxl_sweep.py "
       f"--out-root datasets --precise-only --seed 42",
       cwd=os.path.join(REPO, "anomalyXL"))

    tslm = os.path.join(REPO, "sft", "tslm")
    sh(f"{sys.executable} fetch_data.py", cwd=tslm)

    env = dict(os.environ)
    env.update({
        "TSLM_TORCH_DTYPE": "float16",
        "TSLM_AMP": "1",
        "TSLM_ATTN_IMPL": "sdpa",
        "TSLM_GRAD_CKPT": "1",
        "TSLM_SAVE_TRAINABLE_ONLY": "1",
        "TSLM_VAL_GEN": "0",
        "TSLM_RESUME_FROM": os.path.join(WORK, "prev_best_model.pt"),
        "TSLM_RESUME_WEIGHTS_ONLY": "1",
        "TSLM_SUBSET_SEED": str(sess["subset_seed"]),
        "TSLM_MAX_TRAIN_SECONDS": str(sess["train_budget_s"]),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    })

    for stale in ("train_done.txt",):
        p = os.path.join(WORK, stale)
        if os.path.exists(p):
            os.remove(p)

    wrapper = (
        f"cd {tslm} && "
        f"{sys.executable} train.py --config configs/{config_file} --no-wandb "
        f">> /content/train.log 2>&1; "
        f"echo EXIT:$? > /content/train_done.txt"
    )
    proc = subprocess.Popen(
        ["bash", "-c", wrapper],
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"LAUNCHED pid={proc.pid} config={config_file} "
          f"seed={sess['subset_seed']} budget={sess['train_budget_s']}s",
          flush=True)


if __name__ == "__main__":
    main()
