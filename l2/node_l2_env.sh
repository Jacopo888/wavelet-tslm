#!/usr/bin/env bash
# Level 2 (GRPO, prime-rl) environment on the Vultr A40-24Q node — pinned, fail-fast, idempotent.
#
# What it does (run as root on wavelet-l1, from anywhere):
#   1. prime-rl at the pin (docs/PRIME_RL_PIN.md) with its 4 submodules, own uv venv
#      (~/timerlm/prime-rl/.venv, python 3.12), locked resolve + the FA2 extra only.
#   2. anomalyxl + timeseries_qa + timeseries_qa_v1 editable into that venv (setup.sh recipe).
#   3. rlm-harness venv (host-local IPython agent used by eval_v1_rlm.py).
#   4. Pinned data exposed as directories holding data.parquet (sha256-verified against
#      the L1 MANIFEST) under ~/timerlm/l2/data/{val,test}.
#   5. Policy model snapshot (Qwen/Qwen3.5-2B) at an explicit HF revision, byte-identical
#      for both arms, under ~/timerlm/l2/model_snapshot.
#   6. GPU/stack assertions in the prime-rl venv (cuda, bf16, sm_86, flash_attn, vllm, fla).
#   7. ~/L2_ENV_MANIFEST.json with every SHA/version/path, then "L2_ENV_OK".
#
# The Level-1 venv (~/venv, torch 2.6.0+cu124) is NEVER touched; the script asserts it is
# unchanged at the end. Nothing here needs the GPU to be idle except the final asserts.
#
# Env overrides: L2_MODEL_ID (default Qwen/Qwen3.5-2B), L2_MODEL_REVISION (default: resolve
# main once and freeze it in the manifest), PRIME_RL_REF (default: the documented pin).
set -euo pipefail

TIMERLM="${TIMERLM:-$HOME/timerlm}"
PR="$TIMERLM/prime-rl"
# NOT the pin in docs/PRIME_RL_PIN.md (da53d4c06, 2026-08-11): that commit's verifiers submodule
# post-dates two breaking v1 API changes (RolloutContext -> ModelContext on 2026-07-06, per-task
# behaviour moved from Taskset onto Task on 2026-07-10) and prime-rl itself dropped the paper
# config's schema (orchestrator.advantage / student.client / train.env / max_inflight_rollouts)
# on 2026-06-27 and 2026-07-29. c5afcaa2e (2026-06-26) is the last upstream commit where the
# TimeRLM v1 env imports AND the paper's RL config both load unmodified (verified commit by
# commit, see l2/PROTOCOL-L2.md). Same torch 2.11.0+cu128 / flash-attn 2.8.3 as the later pin;
# vLLM 0.23.0.
PIN="${PRIME_RL_REF:-c5afcaa2e2db6751e0f2eae437afa6bd532fa87f}"
L2="$TIMERLM/l2"
MODEL_ID="${L2_MODEL_ID:-Qwen/Qwen3.5-2B}"
MANIFEST="$HOME/L2_ENV_MANIFEST.json"
L1_MANIFEST="$TIMERLM/anomalyXL/datasets/anomalyxl-train/MANIFEST.json"
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DISABLE_PROGRESS_BARS=1

log() { echo "[l2-env $(date -u +%H:%M:%S)] $*"; }
die() { echo "[l2-env] FATAL: $*" >&2; exit 1; }

# ---------------------------------------------------------------- 0. preconditions
[ -d "$TIMERLM/.git" ] || die "$TIMERLM is not the TimeRLM checkout"
[ -f "$L1_MANIFEST" ] || die "pinned data manifest missing: $L1_MANIFEST (run vultr/node_env.sh first)"
command -v uv >/dev/null || die "uv missing (expected ~/.local/bin/uv)"
command -v nvidia-smi >/dev/null || die "nvidia-smi missing"
L1_PY="$HOME/venv/bin/python"
L1_TORCH_BEFORE="$([ -x "$L1_PY" ] && "$L1_PY" -c 'import torch;print(torch.__version__)' 2>/dev/null || echo none)"
log "L1 venv torch before: $L1_TORCH_BEFORE"
mkdir -p "$L2/data/val" "$L2/data/test" "$L2/logs"

# ---------------------------------------------------------------- 0b. wavelet overlay onto the TimeRLM checkout
# The wavelet files (rlm skill, v0 tool, note variant, RL eval script) are untracked in the
# upstream checkout; the tracked copies live in l2/overlay/timerlm/ and are scp'd to
# ~/l2/kit/overlay by the launcher. Re-applied on every run (idempotent, sha256-logged).
OVERLAY="${L2_OVERLAY:-$HOME/l2/kit/overlay/timerlm}"
[ -d "$OVERLAY" ] || die "overlay missing: $OVERLAY (scp l2/overlay/timerlm there first)"
( cd "$OVERLAY" && find . -type f | sort | while read -r f; do
    mkdir -p "$TIMERLM/$(dirname "$f")"
    cp "$f" "$TIMERLM/$f"
    echo "overlay $f $(sha256sum "$TIMERLM/$f" | cut -c1-12)"
  done )
[ -f "$TIMERLM/rlm-harness/src/rlm/skills/wavelet_zoom.py" ] || die "overlay did not deliver wavelet_zoom.py"

# ---------------------------------------------------------------- 1. prime-rl at the pin
if [ ! -d "$PR/.git" ]; then
  log "cloning prime-rl"
  git clone https://github.com/PrimeIntellect-ai/prime-rl.git "$PR"
fi
git -C "$PR" fetch --quiet origin "$PIN" || true
git -C "$PR" checkout --quiet "$PIN"
[ "$(git -C "$PR" rev-parse HEAD)" = "$PIN" ] || die "prime-rl HEAD != pin"
log "prime-rl at $PIN; initialising the 4 submodules"
git -C "$PR" -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init \
  deps/verifiers deps/renderers deps/pydantic-config deps/research-environments
SUBS="$(git -C "$PR" submodule status deps/verifiers deps/renderers deps/pydantic-config deps/research-environments)"
echo "$SUBS"
echo "$SUBS" | grep -qE '^[+-]' && die "a submodule is not cleanly checked out (see +/- above)"

# ---------------------------------------------------------------- 1b. prime-rl overlay (env-gated gloo backend for vGPU)
# The pinned checkout is kept pristine except for l2/overlay/prime-rl/ (see the file header
# comments): PRIME_RL_DIST_BACKEND override in trainer/utils.py, needed because GRID vGPUs have
# no CUDA memory pools and NCCL >= 2.27 cannot initialise. Applied idempotently, diff logged.
PR_OVERLAY="${L2_PR_OVERLAY:-$HOME/l2/kit/overlay/prime-rl}"
if [ -d "$PR_OVERLAY" ]; then
  ( cd "$PR_OVERLAY" && find . -type f | sort | while read -r f; do cp "$f" "$PR/$f"; echo "prime-rl overlay $f"; done )
  git -C "$PR" diff --stat | tail -3
  grep -q "PRIME_RL_DIST_BACKEND" "$PR/src/prime_rl/trainer/utils.py" || die "prime-rl overlay not applied"
else
  die "prime-rl overlay missing: $PR_OVERLAY (scp l2/overlay/prime-rl there first)"
fi

# ---------------------------------------------------------------- 2. locked venv + FA2 extra
cd "$PR"
log "uv sync --frozen --extra flash-attn (heavy: torch cu128 + vllm + fla + flash-attn wheel)"
uv python install 3.12 >/dev/null 2>&1 || true
uv sync --frozen --extra flash-attn --python 3.12
VENV_PY="$PR/.venv/bin/python"
[ -x "$VENV_PY" ] || die "prime-rl venv python missing"

log "installing anomalyxl + timeseries_qa (+v1) editable into the prime-rl venv"
uv pip install --python "$VENV_PY" -e "$TIMERLM/anomalyXL"
uv pip install --python "$VENV_PY" --no-deps -e "$TIMERLM/environments/timeseries_qa"
uv pip install --python "$VENV_PY" --no-deps -e "$TIMERLM/environments/timeseries_qa_v1"
# timeseries_qa's own runtime deps that the locked prime-rl resolve does not carry
# (--no-deps above is deliberate: a plain install would re-resolve `verifiers` from PyPI
# and clobber the editable deps/verifiers submodule). Pinned so both arms see the same.
uv pip install --python "$VENV_PY" "matplotlib==3.10.8" "datasets>=2.19.0" "huggingface-hub>=0.23"
"$VENV_PY" -c "import anomalyxl, timeseries_qa, timeseries_qa_v1, prime_rl, verifiers, matplotlib, datasets; print('imports OK')"
"$VENV_PY" -c "import verifiers, os; p=os.path.realpath(verifiers.__file__); assert '/prime-rl/deps/verifiers/' in p, p; print('verifiers editable OK:', p)"

# ---------------------------------------------------------------- 3. rlm-harness venv
if [ ! -x "$TIMERLM/rlm-harness/.venv/bin/python" ]; then
  uv venv --python 3.12 "$TIMERLM/rlm-harness/.venv"
fi
uv pip install --python "$TIMERLM/rlm-harness/.venv/bin/python" -e "$TIMERLM/rlm-harness"
"$TIMERLM/rlm-harness/.venv/bin/python" -c "import rlm, numpy, scipy; from rlm.skills import available_builtin_skills as a; s=a(); assert 'wavelet_zoom' in s, s; import importlib.metadata as m; v=m.version('mcp'); assert v.split('.')[0]=='1', 'mcp must be <2 (got '+v+')'; print('rlm-harness OK, skills:', s, 'mcp', v)"
# The rollout program (rlm_program.py) is a PEP-723 uv script resolved by the runtime from
# rlm-harness/pyproject.toml: pre-resolve it once here so the first rollout does not pay for
# it and so a resolution failure surfaces now, not inside a training step.
( cd "$TIMERLM" && sed "s#file://RLM_HARNESS_PATH#file://$TIMERLM/rlm-harness#" environments/timeseries_qa_v1/timeseries_qa_v1/rlm_program.py > /tmp/rlm_program_resolved.py && uv run --script /tmp/rlm_program_resolved.py --help >/dev/null 2>&1 || uv run --script /tmp/rlm_program_resolved.py --help 2>&1 | tail -5 )
uv run --script /tmp/rlm_program_resolved.py --help 2>&1 | grep -q -- "--base-url" || die "rlm_program uv-script resolution failed"
log "rlm_program uv-script resolves"

# ---------------------------------------------------------------- 4. pinned data as data.parquet dirs
link_split() {  # <split-key in manifest> <dest dir>
  local key="$1" dest="$2"
  local rel sha
  rel="$("$VENV_PY" -c "import json,sys; m=json.load(open('$L1_MANIFEST')); f=m['files']['$key']; print(f['path'])")"
  sha="$("$VENV_PY" -c "import json,sys; m=json.load(open('$L1_MANIFEST')); f=m['files']['$key']; print(f['sha256'])")"
  local src="$TIMERLM/$rel"
  [ -f "$src" ] || die "pinned parquet missing: $src"
  local got; got="$(sha256sum "$src" | cut -d' ' -f1)"
  [ "$got" = "$sha" ] || die "sha256 mismatch for $key: $got != $sha"
  ln -sfn "$src" "$dest/data.parquet"
  log "linked $key -> $dest/data.parquet (sha256 verified)"
}
link_split "precise/val" "$L2/data/val"
link_split "precise/test" "$L2/data/test"
"$VENV_PY" - <<PY
import pyarrow.parquet as pq
for split, n in (("val", 520), ("test", 800)):
    rows = pq.ParquetFile("$L2/data/%s/data.parquet" % split).metadata.num_rows
    assert rows == n, (split, rows, n)
print("data rows OK: val 520, test 800")
PY

# ---------------------------------------------------------------- 5. model snapshot at a frozen revision
REV_FILE="$L2/model_snapshot/.revision"
if [ -n "${L2_MODEL_REVISION:-}" ]; then
  REV="$L2_MODEL_REVISION"
elif [ -f "$REV_FILE" ]; then
  REV="$(cat "$REV_FILE")"
else
  REV="$("$VENV_PY" -c "from huggingface_hub import HfApi; print(HfApi().model_info('$MODEL_ID').sha)")"
fi
log "model $MODEL_ID @ $REV"
"$VENV_PY" - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL_ID", revision="$REV", local_dir="$L2/model_snapshot",
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "merges.txt", "vocab.json", "tokenizer*"])
import os, json
assert os.path.exists(os.path.join(p, "config.json")), p
print("snapshot at", p)
PY
echo "$REV" > "$REV_FILE"
MODEL_SHA="$(cd "$L2/model_snapshot" && sha256sum *.safetensors | sha256sum | cut -d' ' -f1)"

# ---------------------------------------------------------------- 6. stack asserts in the prime-rl venv
log "stack asserts"
"$VENV_PY" - <<'PY'
import torch, importlib, sys
assert torch.cuda.is_available(), "cuda not available in prime-rl venv"
cap = torch.cuda.get_device_capability(0)
assert cap == (8, 6), f"unexpected compute capability {cap} (A40 = sm_86)"
assert torch.version.cuda.startswith("12."), torch.version.cuda
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); (a @ a).sum().item()
import flash_attn, vllm, fla, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "| flash_attn", flash_attn.__version__,
      "| vllm", vllm.__version__, "| fla", getattr(fla, "__version__", "?"), "| transformers", transformers.__version__)
# FA2 kernel really runs on this GPU (sm_80 cubin on sm_86)
from flash_attn import flash_attn_func
q = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
o = flash_attn_func(q, q, q, causal=True); torch.cuda.synchronize(); assert o.shape == q.shape
print("flash_attn_func OK")
PY

# ---------------------------------------------------------------- 7. manifest
"$VENV_PY" - <<PY
import json, subprocess, torch, vllm, flash_attn, transformers, fla
def sh(*a): return subprocess.check_output(a, text=True).strip()
subs = {}
for s in ("deps/verifiers", "deps/renderers", "deps/pydantic-config", "deps/research-environments"):
    subs[s] = sh("git", "-C", "$PR", "rev-parse", f"HEAD:{s}")
m = {
  "timerlm_commit": sh("git", "-C", "$TIMERLM", "rev-parse", "HEAD"),
  "prime_rl_commit": sh("git", "-C", "$PR", "rev-parse", "HEAD"),
  "prime_rl_submodules": subs,
  "python": "$("$VENV_PY" -c 'import sys;print(sys.version.split()[0])')",
  "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "vllm": vllm.__version__,
  "flash_attn": flash_attn.__version__, "transformers": transformers.__version__,
  "fla": getattr(fla, "__version__", None),
  "driver": sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
  "gpu": sh("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"),
  "model_id": "$MODEL_ID", "model_revision": "$REV", "model_snapshot": "$L2/model_snapshot",
  "model_safetensors_sha256_of_sha256s": "$MODEL_SHA",
  "data": json.load(open("$L1_MANIFEST")),
  "data_dirs": {"val": "$L2/data/val", "test": "$L2/data/test"},
  "l1_venv_torch_before": "$L1_TORCH_BEFORE",
}
json.dump(m, open("$MANIFEST", "w"), indent=1)
print(json.dumps({k: m[k] for k in ("prime_rl_commit", "torch", "vllm", "flash_attn", "model_revision")}))
PY

# ---------------------------------------------------------------- 8. L1 venv untouched
L1_TORCH_AFTER="$([ -x "$L1_PY" ] && "$L1_PY" -c 'import torch;print(torch.__version__)' 2>/dev/null || echo none)"
[ "$L1_TORCH_BEFORE" = "$L1_TORCH_AFTER" ] || die "L1 venv changed: $L1_TORCH_BEFORE -> $L1_TORCH_AFTER"
log "L1 venv untouched ($L1_TORCH_AFTER); manifest at $MANIFEST"
echo "L2_ENV_OK"
