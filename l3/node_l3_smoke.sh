#!/usr/bin/env bash
# Fail-closed a-d smoke on one A40-24Q, using only the Level-1 venv.
set -Eeuo pipefail

ROOT="$HOME/timerlm/sft/tslm"
PY="$HOME/venv/bin/python"
L3="$ROOT/l3"
CONFIG="$L3/configs/l3_smoke.yaml"
VA1_CONFIG="$ROOT/configs/wavelet_precise_va1.yaml"
VAL_CONFIG="$ROOT/configs/wavelet_precise_va1_valprobe.yaml"
CHECKPOINT="$ROOT/results/anomaly_xl_precise/anomaly_xl_wavelet_va1/checkpoints/epoch_5_end.pt"
SMOKE_ROOT="$HOME/l3/smoke"
BASELINE_PROBE="$HOME/evalq/va1_ep5_valprobe80.json"
GPU="${L3_GPU:-0}"
PEAK_LIMIT_MIB=22528
CURRENT_STAGE="startup"

if [ -e "$SMOKE_ROOT" ]; then
  echo "refusing to reuse existing smoke directory: $SMOKE_ROOT" >&2
  exit 1
fi
mkdir -p "$SMOKE_ROOT"

fail() {
  local reason="$*"
  set +e
  rm -f "$SMOKE_ROOT/SMOKE_OK"
  L3_FAIL_STAGE="$CURRENT_STAGE" L3_FAIL_REASON="$reason" "$PY" - "$SMOKE_ROOT/FAILED" <<'PYFAIL'
import json, os, sys
from datetime import datetime, timezone
with open(sys.argv[1], "x", encoding="utf-8") as handle:
    json.dump({
        "status": "FAILED",
        "stage": os.environ["L3_FAIL_STAGE"],
        "reason": os.environ["L3_FAIL_REASON"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }, handle, indent=2)
    handle.write("\n")
PYFAIL
  echo "L3_SMOKE_FAILED [$CURRENT_STAGE]: $reason" >&2
  exit 1
}

on_err() {
  local rc=$?
  fail "unexpected command failure at line $1 (exit $rc)"
}
trap 'on_err $LINENO' ERR

[ -x "$PY" ] || fail "missing Level-1 interpreter $PY"
[ -f "$ROOT/train.py" ] || fail "missing $ROOT/train.py"
[ -f "$L3/grpo.py" ] || fail "missing $L3/grpo.py"
[ -f "$CONFIG" ] || fail "missing $CONFIG"
[ -f "$VA1_CONFIG" ] || fail "missing va1 config"
[ -f "$VAL_CONFIG" ] || fail "missing validation probe config"
[ -f "$CHECKPOINT" ] || fail "missing epoch_5_end.pt"
[ -f "$BASELINE_PROBE" ] || fail "missing fixed va1 validation-probe baseline"

EXPECTED_SHA="cd7238ae81da1afca9f225a99e9af4261be829798a39985a427d88a098bf2531"
OBSERVED_SHA=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
[ "$OBSERVED_SHA" = "$EXPECTED_SHA" ] || fail "epoch_5_end.pt SHA mismatch: $OBSERVED_SHA"

# Native allocator only: expandable_segments uses unsupported VMM on this vGPU.
unset PYTORCH_CUDA_ALLOC_CONF
export TSLM_TORCH_DTYPE=bfloat16
export TSLM_AMP=1
export TSLM_AMP_DTYPE=bfloat16
export TSLM_ATTN_IMPL=sdpa
export TSLM_SAVE_TRAINABLE_ONLY=1
export TSLM_EVAL_STRICT=1
export TSLM_GRAD_CKPT=1
export TSLM_QWEN_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT

CURRENT_STAGE="a_cpu_tests"
cd "$ROOT"
if ! CUDA_VISIBLE_DEVICES="" "$PY" -m pytest l3/tests -q >"$SMOKE_ROOT/stage_a_pytest.log" 2>&1; then
  fail "CPU tests failed; see stage_a_pytest.log"
fi
"$PY" - "$SMOKE_ROOT/stage_a.json" <<'PYA'
import json, sys
with open(sys.argv[1], "x", encoding="utf-8") as handle:
    json.dump({"stage": "a_cpu_tests", "ok": True}, handle, indent=2)
    handle.write("\n")
PYA

CURRENT_STAGE="gpu_preflight"
GPU_NAME=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader | head -1)
GPU_TOTAL=$(nvidia-smi -i "$GPU" --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
case "$GPU_NAME" in *A40*) ;; *) fail "expected A40, got $GPU_NAME" ;; esac
case "$GPU_TOTAL" in ''|*[!0-9]*) fail "cannot parse GPU total memory: $GPU_TOTAL" ;; esac
[ "$GPU_TOTAL" -gt "$PEAK_LIMIT_MIB" ] || fail "GPU total memory is below gate limit"
GPU_PROCS=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
if printf '%s\n' "$GPU_PROCS" | grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$'; then
  fail "GPU already has compute processes: $GPU_PROCS"
fi

CURRENT_STAGE="b_init_identity"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/grpo.py init-check \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --out "$SMOKE_ROOT/stage_b.json" >"$SMOKE_ROOT/stage_b.log" 2>&1
"$PY" - "$SMOKE_ROOT/stage_b.json" <<'PYB'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
assert d["ok"] and len(d["rows"]) == 5
assert all(r["tokens_equal"] and r["text_equal"] for r in d["rows"])
assert all(r["max_logprob_diff"] == 0.0 and r["ratio_all_exact_one"] for r in d["rows"])
PYB

CURRENT_STAGE="c0_memory_gate"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/grpo.py train \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --mode memory-gate --run-dir "$SMOKE_ROOT/c0" >"$SMOKE_ROOT/stage_c0.log" 2>&1
cp "$SMOKE_ROOT/c0/summary.json" "$SMOKE_ROOT/stage_c0.json"
"$PY" - "$SMOKE_ROOT/stage_c0.json" "$PEAK_LIMIT_MIB" <<'PYC0'
import json, math, sys
d=json.load(open(sys.argv[1], encoding="utf-8")); limit=float(sys.argv[2])
assert d["status"] == "DONE" and d["steps"] == 1 and d["rollouts"] == 4
assert d["parsable"] + d["unparsable"] == 4
assert d["ratio_all_exact_one"] and d["peak_mem_mib"] < limit
assert d["checkpoint_tensor_count"] == 32 and d["eval_composed_tensor_count"] == 42
assert math.isfinite(d["reward_mean"]) and math.isfinite(d["kl_mean"])
PYC0

CURRENT_STAGE="c_twenty_steps"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/grpo.py train \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --mode smoke --run-dir "$SMOKE_ROOT/c" >"$SMOKE_ROOT/stage_c.log" 2>&1
cp "$SMOKE_ROOT/c/summary.json" "$SMOKE_ROOT/stage_c.json"
"$PY" - "$SMOKE_ROOT/stage_c.json" <<'PYC'
import json, math, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
expected={"classify_with_evidence","measure_magnitude","localize","localize_all_channels","lead_lag_with_magnitude"}
assert d["status"] == "DONE" and d["steps"] == 20 and d["rollouts"] == 80
assert d["parsable"] + d["unparsable"] == 80
assert d["ratio_all_exact_one"] and d["positive_grad_steps"] > 0
assert d["checkpoint_tensor_count"] == 32 and d["eval_composed_tensor_count"] == 42
assert set(d["by_category"]) == expected
assert all(math.isfinite(d[k]) for k in ("reward_mean","kl_mean","max_grad_norm","peak_mem_mib","seconds"))
rows=[json.loads(line) for line in open(sys.argv[1].replace("stage_c.json","c/steps.jsonl"), encoding="utf-8")]
required={"rollouts","parsable","unparsable","reward_mean","reward_std","by_category","zero_variance_groups","kl_mean","ratio_mean","grad_norm","tokens_prompt","tokens_answer","peak_mem_mib","s_per_step"}
assert len(rows) == 20 and all(required <= set(row) for row in rows)
assert all(row["ratio_mean"] == 1.0 and row["rollouts"] == 4 for row in rows)
PYC

CURRENT_STAGE="d_validation_probe"
RL_CHECKPOINT="$SMOKE_ROOT/c/lora_checkpoint_after_20_steps.pt"
EVAL_CHECKPOINT="$SMOKE_ROOT/c/eval_composed_after_20_steps.pt"
[ -f "$RL_CHECKPOINT" ] || fail "missing LoRA-only post-RL checkpoint"
[ -f "$EVAL_CHECKPOINT" ] || fail "missing composed eval artifact"
export TSLM_EVAL_TAG="l3_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py \
  --config "$VAL_CONFIG" --no-wandb --eval-only \
  --checkpoint "$EVAL_CHECKPOINT" --split val \
  --out-json "$SMOKE_ROOT/stage_d.json" >"$SMOKE_ROOT/stage_d.log" 2>&1
"$PY" - "$SMOKE_ROOT/stage_d.json" "$BASELINE_PROBE" <<'PYD'
import json, math, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
b=json.load(open(sys.argv[2], encoding="utf-8"))
assert d["split"] == "val" and d["samples_evaluated"] == 80
assert math.isfinite(float(d["primary"])) and float(d["primary"]) >= 0.10
assert b["samples_evaluated"] == 80 and math.isfinite(float(b["primary"]))
PYD

CURRENT_STAGE="finalize"
"$PY" - "$SMOKE_ROOT" <<'PYDONE'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
root=Path(sys.argv[1])
stages={name: json.load(open(root/f"stage_{name}.json", encoding="utf-8")) for name in ("a","b","c0","c","d")}
document={"status":"SMOKE_OK","created_utc":datetime.now(timezone.utc).isoformat(),"stages":stages}
with open(root/"stages.json", "x", encoding="utf-8") as handle:
    json.dump(document, handle, indent=2); handle.write("\n")
with open(root/"SMOKE_OK", "x", encoding="utf-8") as handle:
    json.dump({"status":"SMOKE_OK","stages_file":str(root/"stages.json")}, handle, indent=2); handle.write("\n")
PYDONE
trap - ERR
echo "L3_SMOKE_OK $SMOKE_ROOT/SMOKE_OK"
