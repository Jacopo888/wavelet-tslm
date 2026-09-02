"""Modal training app for the wavelet TSLM session chain.

One Modal function = one chained session (same semantics as the Kaggle
kernels): weights-only warm start from /vol/checkpoints/prev_best_model.pt,
fresh 2000-sample subset, fixed 200-row val/test, fp16+AMP on a single L4.

Persistent Volume "wavelet-vol" mounted at /vol:
  /vol/checkpoints/prev_best_model.pt   input checkpoint (modal volume put)
  /vol/hf                               HF cache (Qwen weights, once)
  /vol/data                             gold benchmark + train shards cache
  /vol/sessions/<name>                  output dir (checkpoints, metrics, log)
  /vol/sessions/<name>/DONE             exit marker

Launch (detached survives the local client):
  modal run --detach modal_train.py --session m1 --subset-seed 1044 --budget-s 14400

The patch overlay (train.py with AMP, wavelet model files, configs) is baked
into the image from esperimenti/kaggle/patch via add_local_dir.
"""

import modal

app = modal.App("wavelet-train")

PATCH_LOCAL = r"C:\Users\jacop\Proposta ETH FINALE\wavelet-tslm\kaggle\patch"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "numpy", "pandas", "pyarrow", "scipy", "psutil")
    .pip_install(
        "transformers @ git+https://github.com/huggingface/transformers@5c1c72b",
        "peft==0.18.1", "einops", "einops-exts", "datasets", "pyyaml",
        "wandb", "open_clip_torch", "huggingface-hub",
    )
    .pip_install("chronos-forecasting==2.2.2", "open-flamingo",
                 extra_options="--no-deps")
    .add_local_dir(PATCH_LOCAL, "/patch")
)

vol = modal.Volume.from_name("wavelet-vol", create_if_missing=True)

OVERLAY = {
    "wavelet_encoder.py": "sft/tslm/src/models/wavelet_encoder.py",
    "wavelet.py": "sft/tslm/src/models/wavelet.py",
    "qwen_backbone.py": "sft/tslm/src/models/qwen_backbone.py",
    "qa_dataset.py": "sft/tslm/src/datasets/anomaly_xl/qa_dataset.py",
    "train.py": "sft/tslm/train.py",
    "eval_integrity.py": "sft/tslm/eval_integrity.py",
    "fetch_data.py": "sft/tslm/fetch_data.py",
    "preflight_data.py": "sft/tslm/preflight_data.py",
}

# The pinned AnomalyXL revision (must match kaggle/patch/fetch_data.py).
HF_REVISION = "df363a467f855e55aaad414b0bdd58590829df9a"

REPO = "/root/timerlm"


def _sh(cmd, cwd=None, env=None):
    import subprocess
    print(">>", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=env)


def _sync_dir(src, dst):
    import shutil
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _materialize_data(repo, sys_executable):
    """Deterministic data provenance (integrity pass 02/09): train, val AND
    gold test all come from the PINNED HF revision via the patched
    fetch_data.py, which writes MANIFEST.json and asserts split
    disjointness. The volume cache is trusted only when its manifest
    matches the pin; pre-integrity caches (no manifest, generated gold)
    are refreshed from HF on first use."""
    import json as _json
    import os

    ds_root = os.path.join(repo, "anomalyXL", "datasets")
    train_cache = "/vol/data/anomalyxl-train"
    gold_cache = "/vol/data/gold-precise"
    manifest = os.path.join(train_cache, "MANIFEST.json")
    pinned = False
    if os.path.isfile(manifest):
        try:
            pinned = _json.load(open(manifest)).get("revision") == HF_REVISION
        except Exception:
            pinned = False
    if pinned and os.path.isdir(gold_cache):
        print(f"data: volume cache @ pinned {HF_REVISION[:12]}", flush=True)
        _sync_dir(train_cache, os.path.join(ds_root, "anomalyxl-train"))
        _sync_dir(gold_cache, os.path.join(ds_root, "anomalyxl-precise"))
    else:
        print("data: cache missing or unpinned -> fetching HF @ pin", flush=True)
        _sh(f"{sys_executable} fetch_data.py --variant precise",
            cwd=os.path.join(repo, "sft", "tslm"))
        _sync_dir(os.path.join(ds_root, "anomalyxl-train"), train_cache)
        _sync_dir(os.path.join(ds_root, "anomalyxl-precise"), gold_cache)
    # Anti-leakage gate on whatever ended up in ds_root (cache or fresh).
    _sh(f"{sys_executable} preflight_data.py {ds_root} --variant precise",
        cwd=os.path.join(repo, "sft", "tslm"))
    vol.commit()


@app.function(gpu="A100-80GB", image=image, volumes={"/vol": vol},
              memory=24576, cpu=4, timeout=86400)
def train_session(session: str, config_file: str, subset_seed: int,
                  budget_s: int, resume_rel: str, max_steps: int = 0,
                  sched_total: int = 0):
    import os
    import shutil
    import subprocess
    import sys
    import threading

    os.environ["HF_HOME"] = "/vol/hf"
    os.makedirs("/vol/hf", exist_ok=True)
    sess_dir = f"/vol/sessions/{session}"
    os.makedirs(sess_dir, exist_ok=True)

    ckpt = None
    if resume_rel:
        ckpt = os.path.join("/vol", resume_rel)
        if not os.path.exists(ckpt):
            raise RuntimeError(f"missing {ckpt} - modal volume put it first")
    else:
        print("from-scratch run: no resume checkpoint", flush=True)

    _sh(f"git clone --depth 1 https://github.com/OpenTSLM/TimeRLM {REPO}")
    for src, dst in OVERLAY.items():
        dst_path = os.path.join(REPO, dst)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(os.path.join("/patch", src), dst_path)
    cfg_dir = os.path.join(REPO, "sft", "tslm", "configs")
    for f in os.listdir("/patch"):
        if f.endswith(".yaml"):
            shutil.copy(os.path.join("/patch", f), os.path.join(cfg_dir, f))

    _sh(f"{sys.executable} -m pip install -q -e {REPO}/anomalyXL")

    # Benchmark + train shards: pinned HF revision, manifest-verified cache.
    _materialize_data(REPO, sys.executable)

    env = dict(os.environ)
    env.update({
        # bf16, not fp16: Qwen's forward activations overflow fp16 (every m1
        # batch produced inf/nan loss); bf16 has fp32's range and needs no
        # GradScaler. A100/L4 support it natively.
        "TSLM_TORCH_DTYPE": "bfloat16",
        "TSLM_AMP": "1",
        "TSLM_AMP_DTYPE": "bfloat16",
        "TSLM_ATTN_IMPL": "sdpa",
        # LESSON m30 (02/09): checkpointing ON from batch 0, ALWAYS. With
        # ckpt off, the allocator fragments on the 131k mono-channel cells
        # and every 16ch cell then OOMs even through the fallback (2,770
        # samples silently lost). Full coverage is mandatory: the run
        # aborts at the FIRST lost sample instead of shipping a partial
        # model as "complete".
        "TSLM_GRAD_CKPT": "1",
        "TSLM_CKPT_FALLBACK": "1",
        "TSLM_REQUIRE_FULL_COVERAGE": "1",
        "TSLM_EVAL_STRICT": "1",
        "TSLM_KEEP_EPOCH_CKPTS": "1",
        "TSLM_SAVE_TRAINABLE_ONLY": "1",
        "TSLM_SAVE_OPT_STATE": "1",
        "TSLM_VAL_GEN": "0",
        "TSLM_SUBSET_SEED": str(subset_seed),
        "TSLM_MAX_TRAIN_SECONDS": str(budget_s),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    })
    if ckpt:
        env["TSLM_RESUME_FROM"] = ckpt
        env["TSLM_RESUME_WEIGHTS_ONLY"] = "1"
    if max_steps:
        env["TSLM_MAX_TRAIN_STEPS"] = str(max_steps)
    if sched_total:
        env["TSLM_SCHED_TOTAL_STEPS"] = str(sched_total)

    # Commit the volume every 10 min so checkpoints/logs survive a crash.
    stop_commits = threading.Event()

    def committer():
        while not stop_commits.wait(600):
            try:
                vol.commit()
                print("[vol] periodic commit", flush=True)
            except Exception as e:
                print("[vol] commit failed:", e, flush=True)

    threading.Thread(target=committer, daemon=True).start()

    tslm = os.path.join(REPO, "sft", "tslm")
    log_path = os.path.join(sess_dir, "train.log")
    print(f"starting train.py (config {config_file}, seed {subset_seed}, "
          f"budget {budget_s}s) -> {log_path}", flush=True)
    rc = -1
    try:
        with open(log_path, "a") as logf:
            proc = subprocess.Popen(
                [sys.executable, "train.py", "--config", f"configs/{config_file}",
                 "--no-wandb"],
                cwd=tslm, env=env, stdout=logf, stderr=subprocess.STDOUT,
            )
            rc = proc.wait()
    finally:
        stop_commits.set()
        with open(os.path.join(sess_dir, "DONE"), "w") as f:
            f.write(f"EXIT:{rc}\n")
        vol.commit()
    print(f"train.py exited with {rc}", flush=True)
    if rc != 0:
        with open(log_path, errors="replace") as f:
            print("--- train.log tail ---", flush=True)
            print("\n".join(f.read().split("\n")[-40:]), flush=True)
        raise SystemExit(rc)


@app.function(gpu="A100-40GB", image=image, volumes={"/vol": vol},
              memory=24576, cpu=4, timeout=14400)
def select_and_eval(session: str, config_file: str):
    """Upstream-style selection: eval every epoch_*_end.pt on val WITH
    generations, pick best by primary, run the 800-row test on the winner."""
    import glob
    import json as _json
    import os
    import shutil
    import subprocess
    import sys

    os.environ["HF_HOME"] = "/vol/hf"
    sess_dir = f"/vol/sessions/{session}"
    ck_dirs = glob.glob(f"{sess_dir}/anomaly_xl_precise/*/checkpoints")
    if not ck_dirs:
        raise RuntimeError(f"no checkpoints dir under {sess_dir}")
    ck_dir = ck_dirs[0]

    _sh(f"git clone --depth 1 https://github.com/OpenTSLM/TimeRLM {REPO}")
    for src, dst in OVERLAY.items():
        dst_path = os.path.join(REPO, dst)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(os.path.join("/patch", src), dst_path)
    cfg_dir = os.path.join(REPO, "sft", "tslm", "configs")
    for f in os.listdir("/patch"):
        if f.endswith(".yaml"):
            shutil.copy(os.path.join("/patch", f), os.path.join(cfg_dir, f))
    _sh(f"{sys.executable} -m pip install -q -e {REPO}/anomalyXL")
    _materialize_data(REPO, sys.executable)

    env = dict(os.environ)
    env.update({
        "TSLM_TORCH_DTYPE": "bfloat16",
        "TSLM_AMP": "1",
        "TSLM_AMP_DTYPE": "bfloat16",
        "TSLM_ATTN_IMPL": "sdpa",
        "TSLM_SAVE_TRAINABLE_ONLY": "1",
        # Eval must account for every example or die (integrity pass 02/09).
        "TSLM_EVAL_STRICT": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    })
    tslm = os.path.join(REPO, "sft", "tslm")
    outdir = f"{sess_dir}/select"
    os.makedirs(outdir, exist_ok=True)

    primaries = {}
    for ck in sorted(glob.glob(os.path.join(ck_dir, "epoch_*_end.pt"))):
        ep = os.path.basename(ck).split("_")[1]
        out = f"{outdir}/ep{ep}_val.json"
        if not os.path.exists(out):
            print(f"=== val eval epoch {ep} ===", flush=True)
            subprocess.run(
                [sys.executable, "train.py", "--config", f"configs/{config_file}",
                 "--no-wandb", "--eval-only", "--checkpoint", ck,
                 "--split", "val", "--out-json", out],
                cwd=tslm, env=env, check=True)
            vol.commit()
        d = _json.load(open(out))
        primaries[ep] = float(d.get("primary", 0.0))
        print(f"epoch {ep}: val primary = {primaries[ep]:.4f}", flush=True)

    best = max(primaries, key=primaries.get)
    print("VAL PRIMARIES:", primaries, "-> BEST EPOCH:", best, flush=True)

    test_out = f"{outdir}/best_ep{best}_test.json"
    if not os.path.exists(test_out):
        print(f"=== FINAL test eval on epoch {best} ===", flush=True)
        subprocess.run(
            [sys.executable, "train.py", "--config", f"configs/{config_file}",
             "--no-wandb", "--eval-only",
             "--checkpoint", os.path.join(ck_dir, f"epoch_{best}_end.pt"),
             "--split", "test", "--out-json", test_out],
            cwd=tslm, env=env, check=True)
    with open(f"{outdir}/DONE", "w") as f:
        f.write(f"BEST_EPOCH:{best}\n")
    vol.commit()
    print("SELECT+EVAL COMPLETE", flush=True)


@app.function(gpu="A100-40GB", image=image, volumes={"/vol": vol},
              memory=24576, cpu=4, timeout=14400)
def reeval(session: str, config_file: str, checkpoint_rel: str, tag: str):
    """Integrity re-eval (02/09): score an existing checkpoint on the PINNED
    official HF test (the historical evals used a regenerated seed-42 gold
    whose input series differ from the published shard — ids/answers match,
    series fingerprint does not). Writes aggregate JSON + per-example JSONL
    under sessions/<session>/reeval/, never overwriting historical logs."""
    import os
    import shutil
    import subprocess
    import sys

    os.environ["HF_HOME"] = "/vol/hf"
    ck = os.path.join("/vol", checkpoint_rel)
    if not os.path.exists(ck):
        raise RuntimeError(f"missing checkpoint {ck}")

    _sh(f"git clone --depth 1 https://github.com/OpenTSLM/TimeRLM {REPO}")
    for src, dst in OVERLAY.items():
        dst_path = os.path.join(REPO, dst)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy(os.path.join("/patch", src), dst_path)
    cfg_dir = os.path.join(REPO, "sft", "tslm", "configs")
    for f in os.listdir("/patch"):
        if f.endswith(".yaml"):
            shutil.copy(os.path.join("/patch", f), os.path.join(cfg_dir, f))
    _sh(f"{sys.executable} -m pip install -q -e {REPO}/anomalyXL")
    _materialize_data(REPO, sys.executable)   # refreshes cache to the HF pin

    env = dict(os.environ)
    env.update({
        "TSLM_TORCH_DTYPE": "bfloat16",
        "TSLM_AMP": "1",
        "TSLM_AMP_DTYPE": "bfloat16",
        "TSLM_ATTN_IMPL": "sdpa",
        "TSLM_SAVE_TRAINABLE_ONLY": "1",
        "TSLM_EVAL_STRICT": "1",
        "TSLM_EVAL_TAG": tag,
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    })
    outdir = f"/vol/sessions/{session}/reeval"
    os.makedirs(outdir, exist_ok=True)
    out_json = f"{outdir}/{os.path.basename(checkpoint_rel)}_{tag}_test.json"
    subprocess.run(
        [sys.executable, "train.py", "--config", f"configs/{config_file}",
         "--no-wandb", "--eval-only", "--checkpoint", ck,
         "--split", "test", "--out-json", out_json],
        cwd=os.path.join(REPO, "sft", "tslm"), env=env, check=True)
    vol.commit()
    print(f"REEVAL DONE -> {out_json}", flush=True)


@app.local_entrypoint()
def main(session: str = "full1",
         config_file: str = "wavelet_precise_full.yaml",
         subset_seed: int = 42,
         budget_s: int = 43200,
         resume_rel: str = "",
         max_steps: int = 0,
         sched_total: int = 0):
    train_session.remote(session, config_file, subset_seed, budget_s,
                         resume_rel, max_steps, sched_total)


@app.local_entrypoint()
def reeval_main():
    m30 = ("sessions/m30/anomaly_xl_precise/anomaly_xl_wavelet_precise_m30"
           "/checkpoints/epoch_4_end.pt")
    full1 = ("sessions/full1/anomaly_xl_precise/anomaly_xl_wavelet_precise_full"
             "/checkpoints/latest_checkpoint.pt")
    h1 = reeval.spawn("m30", "wavelet_precise_m30.yaml", m30, "hfpin")
    h2 = reeval.spawn("full1", "wavelet_precise_full.yaml", full1, "hfpin")
    h1.get()
    h2.get()
    print("BOTH REEVALS COMPLETE", flush=True)
