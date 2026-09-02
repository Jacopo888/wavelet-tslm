"""TimeRLM ChatTS anchor - Kaggle batch kernel (budget run).

Same pipeline and recipe as the wavelet budget kernel; trains the official
ChatTS variant (chronos-2 frozen encoder) with the identical reduced budget
(1500 x 1 epoch, frozen LM embeddings) as the controlled anchor. Cells too
long for uniform patching may be skipped by the trainer's per-batch guard -
their count is itself a finding.
"""

import os
import shutil
import subprocess
import sys
import time

WORK = "/kaggle/working"
REPO = os.path.join(WORK, "timerlm")

OVERLAY = {
    "wavelet_encoder.py": "sft/tslm/src/models/wavelet_encoder.py",
    "wavelet.py": "sft/tslm/src/models/wavelet.py",
    "qwen_backbone.py": "sft/tslm/src/models/qwen_backbone.py",
    "qa_dataset.py": "sft/tslm/src/datasets/anomaly_xl/qa_dataset.py",
    "train.py": "sft/tslm/train.py",
    "chatts_precise_budget.yaml": "sft/tslm/configs/chatts_precise_budget.yaml",
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


def run():
    os.chdir(WORK)
    patch = find_patch_dir()
    sh(f"git clone --depth 1 https://github.com/OpenTSLM/TimeRLM {REPO}")
    for src, dst in OVERLAY.items():
        dst_path = os.path.join(REPO, dst)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(os.path.join(patch, src), dst_path)
        print("overlay:", dst, flush=True)

    print("== GPU ==", flush=True)
    sh("nvidia-smi")

    sh(f"{sys.executable} -m pip install -q uv")
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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    sh(f"{sys.executable} train.py --config configs/chatts_precise_budget.yaml "
       f"--no-wandb", cwd=tslm)


def salvage_results():
    results = os.path.join(REPO, "sft", "tslm", "results")
    if not os.path.isdir(results):
        return
    dest = os.path.join(WORK, "results_out")
    try:
        shutil.copytree(results, dest, dirs_exist_ok=True)
    except Exception as exc:
        print("full salvage failed, retrying without checkpoints:", exc, flush=True)
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(results, dest, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.pt"))
    for root, _dirs, files in os.walk(dest):
        for f in files:
            print("result file:", os.path.relpath(os.path.join(root, f), WORK),
                  flush=True)


def main():
    try:
        run()
        print("CHATTS BUDGET RUN COMPLETE", flush=True)
    finally:
        salvage_results()
        shutil.rmtree(REPO, ignore_errors=True)


if __name__ == "__main__":
    main()
