#!/usr/bin/env bash
# Z1-bis only: traces -> one SFT epoch -> perceptual gate -> operational gate.
set -Eeuo pipefail

ROOT="${L3_ROOT:-$HOME/timerlm/sft/tslm}"
PY="${L3_PYTHON:-$HOME/venv/bin/python}"
L3="$ROOT/l3"
CONFIG="$L3/configs/l3_zoom_z1bis.yaml"
VA1_CHECKPOINT="${L3_VA1_CHECKPOINT:-$ROOT/results/anomaly_xl_precise/anomaly_xl_wavelet_va1/checkpoints/epoch_5_end.pt}"
TRAIN_PARQUET="${L3_TRAIN_PARQUET:-$HOME/timerlm/anomalyXL/datasets/anomalyxl-train/anomalyxl-precise/train.parquet}"
PREVIOUS_ROOT="${L3_PREVIOUS_ZOOM_ROOT:-$HOME/l3/zoom_20260911c}"
REFERENCE_GATE_ROWS="${L3_REFERENCE_GATE_ROWS:-$PREVIOUS_ROOT/Z1/gate/rows.jsonl}"
VA1_VAL_JSONL="${L3_VA1_VAL_JSONL:-$PREVIOUS_ROOT/Z0/val520_no_zoom.jsonl}"
PREVIOUS_VZ1_VAL_JSONL="${L3_PREVIOUS_VZ1_VAL_JSONL:-$PREVIOUS_ROOT/Z1/diag/val520_vz1_no_zoom.jsonl}"
EXPECTED_VA1_SHA="cd7238ae81da1afca9f225a99e9af4261be829798a39985a427d88a098bf2531"
EXPECTED_TRAIN_SHA="b8571d886772a55613c2e8851a774e6a561e19c8272227f5abb70e2d76e70b7d"
GPU="${L3_GPU:-0}"
RUN_ROOT="${L3_Z1BIS_RUN_ROOT:-$HOME/l3/z1bis_$(date -u +%Y%m%dT%H%M%SZ)}"
TRACE_TIMEOUT_SECONDS="${L3_Z1BIS_TRACE_TIMEOUT_SECONDS:-900}"
TRACE_PEAK_LIMIT_MIB="${L3_Z1BIS_TRACE_PEAK_LIMIT_MIB:-4096}"

# This is deliberately not inferred from git state. Claude's reviewed-diff handoff
# must explicitly set the interlock before any GPU work can start.
[ "${L3_Z1BIS_DIFF_REVIEWED:-}" = "CLAUDE_APPROVED" ] || {
  echo "refusing to start: set L3_Z1BIS_DIFF_REVIEWED=CLAUDE_APPROVED after diff review" >&2
  exit 1
}

if [ "${1:-}" != "--worker" ] && [ -z "${TMUX:-}" ]; then
  command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
  [ ! -e "$RUN_ROOT" ] || { echo "refusing to reuse run root: $RUN_ROOT" >&2; exit 1; }
  mkdir -p "$RUN_ROOT"
  SESSION="l3-z1bis-$(date -u +%Y%m%dT%H%M%SZ)"
  tmux new-session -d -s "$SESSION" \
    "env L3_Z1BIS_DIFF_REVIEWED=CLAUDE_APPROVED L3_Z1BIS_RUN_ROOT='$RUN_ROOT' L3_GPU='$GPU' L3_ROOT='$ROOT' L3_PYTHON='$PY' L3_VA1_CHECKPOINT='$VA1_CHECKPOINT' L3_TRAIN_PARQUET='$TRAIN_PARQUET' L3_PREVIOUS_ZOOM_ROOT='$PREVIOUS_ROOT' L3_REFERENCE_GATE_ROWS='$REFERENCE_GATE_ROWS' L3_VA1_VAL_JSONL='$VA1_VAL_JSONL' L3_PREVIOUS_VZ1_VAL_JSONL='$PREVIOUS_VZ1_VAL_JSONL' L3_Z1BIS_TRACE_TIMEOUT_SECONDS='$TRACE_TIMEOUT_SECONDS' L3_Z1BIS_TRACE_PEAK_LIMIT_MIB='$TRACE_PEAK_LIMIT_MIB' bash '$L3/node_l3_z1bis.sh' --worker >'$RUN_ROOT/worker.log' 2>&1"
  echo "L3_Z1BIS_DETACHED session=$SESSION run_root=$RUN_ROOT log=$RUN_ROOT/worker.log"
  exit 0
fi

START_EPOCH=$(date +%s)
CURRENT_PHASE=startup
CURRENT_DIR=""

failure() {
  local rc="$1" line="$2"
  set +e
  if [ -n "$CURRENT_DIR" ] && [ -d "$CURRENT_DIR" ]; then
    rm -f "$CURRENT_DIR/OK"
    L3_PHASE="$CURRENT_PHASE" L3_RC="$rc" L3_LINE="$line" "$PY" - "$CURRENT_DIR/FAILED" <<'PY'
import json,os,sys
from datetime import datetime,timezone
d={"status":"FAILED","phase":os.environ["L3_PHASE"],"exit_code":int(os.environ["L3_RC"]),
   "line":int(os.environ["L3_LINE"]),"created_utc":datetime.now(timezone.utc).isoformat()}
with open(sys.argv[1],"w",encoding="utf-8") as h: json.dump(d,h,indent=2); h.write("\n")
PY
  fi
  echo "L3_Z1BIS_FAILED phase=$CURRENT_PHASE line=$line rc=$rc" >&2
  exit "$rc"
}
trap 'failure $? $LINENO' ERR

start_phase() {
  CURRENT_PHASE="$1"
  CURRENT_DIR="$RUN_ROOT/$1"
  [ ! -e "$CURRENT_DIR" ] || { echo "refusing to reuse phase directory: $CURRENT_DIR" >&2; return 1; }
  mkdir "$CURRENT_DIR"
  printf 'L3_Z1BIS_STATUS phase=%s elapsed_s=%s\n' "$CURRENT_PHASE" "$(( $(date +%s) - START_EPOCH ))" | tee -a "$CURRENT_DIR/phase.log"
}

finish_phase() {
  [ -f "$CURRENT_DIR/summary.json" ]
  "$PY" - "$CURRENT_DIR/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8")); assert d.get("status") in {"DONE","OK"},d
PY
  : > "$CURRENT_DIR/OK"
}

logged() { "$@" 2>&1 | tee -a "$CURRENT_DIR/phase.log"; }
require_ok() { [ -f "$RUN_ROOT/$1/OK" ] || { echo "missing $1/OK" >&2; return 1; }; }

run_trace_planner_guarded() {
  local started now rss_kib state rc
  [ -r /proc/self/status ] || {
    echo "trace planner RSS guard requires readable /proc" >&2
    return 125
  }
  env CUDA_VISIBLE_DEVICES="" "$PY" l3/zoom_traces.py \
    --config "$CONFIG" --out-dir "$CURRENT_DIR/plan" &
  local trace_pid=$!
  started=$(date +%s)
  while [ -r "/proc/$trace_pid/stat" ]; do
    state=$(awk '{print $3}' "/proc/$trace_pid/stat")
    [ "$state" != "Z" ] || break
    now=$(date +%s)
    if [ "$((now - started))" -ge "$TRACE_TIMEOUT_SECONDS" ]; then
      echo "trace planner exceeded timeout: ${TRACE_TIMEOUT_SECONDS}s" >&2
      kill -TERM "$trace_pid" 2>/dev/null || true
      wait "$trace_pid" 2>/dev/null || true
      return 124
    fi
    rss_kib=$(awk '/^VmRSS:/ {print $2}' "/proc/$trace_pid/status")
    if [ -z "$rss_kib" ]; then
      state=$(awk '{print $3}' "/proc/$trace_pid/stat" 2>/dev/null || true)
      if [ -z "$state" ] || [ "$state" = "Z" ]; then
        break
      fi
      echo "cannot read trace planner RSS from /proc/$trace_pid/status" >&2
      kill -TERM "$trace_pid" 2>/dev/null || true
      wait "$trace_pid" 2>/dev/null || true
      return 125
    fi
    if [ "$rss_kib" -gt "$((TRACE_PEAK_LIMIT_MIB * 1024))" ]; then
      echo "trace planner exceeded RSS limit: ${rss_kib} KiB > ${TRACE_PEAK_LIMIT_MIB} MiB" >&2
      kill -TERM "$trace_pid" 2>/dev/null || true
      wait "$trace_pid" 2>/dev/null || true
      return 137
    fi
    sleep 1
  done
  if wait "$trace_pid"; then
    return 0
  else
    rc=$?
    return "$rc"
  fi
}

for required in "$PY" "$CONFIG" "$L3/zoom.py" "$L3/zoom_sft.py" \
  "$L3/zoom_traces.py" "$L3/tools/diag_crop_content.py" \
  "$L3/tools/gate_z1bis_operational.py" "$VA1_CHECKPOINT" "$TRAIN_PARQUET" \
  "$REFERENCE_GATE_ROWS" "$VA1_VAL_JSONL" "$PREVIOUS_VZ1_VAL_JSONL"; do
  [ -e "$required" ] || { echo "missing required input: $required" >&2; exit 1; }
done
[ -x "$PY" ]
case "$TRACE_TIMEOUT_SECONDS:$TRACE_PEAK_LIMIT_MIB" in
  *[!0-9:]*|:*|*:) echo "trace guards must be positive integer values" >&2; exit 1 ;;
esac
[ "$TRACE_TIMEOUT_SECONDS" -gt 0 ] && [ "$TRACE_PEAK_LIMIT_MIB" -gt 0 ]
[ "$(sha256sum "$VA1_CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_VA1_SHA" ]
[ "$(sha256sum "$TRAIN_PARQUET" | awk '{print $1}')" = "$EXPECTED_TRAIN_SHA" ]
[ "$(wc -l < "$VA1_VAL_JSONL")" -eq 520 ]
[ "$(wc -l < "$PREVIOUS_VZ1_VAL_JSONL")" -eq 520 ]
GPU_NAME=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *A40*) ;; *) echo "expected A40, got $GPU_NAME" >&2; exit 1 ;; esac
GPU_PROCS=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
if printf '%s\n' "$GPU_PROCS" | grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$'; then
  echo "GPU already has compute processes: $GPU_PROCS" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
git rev-parse HEAD > "$RUN_ROOT/KIT_COMMIT.txt"
git status --porcelain -- l3 >> "$RUN_ROOT/KIT_COMMIT.txt"
unset PYTORCH_CUDA_ALLOC_CONF WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
export TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16
export TSLM_ATTN_IMPL=sdpa TSLM_SAVE_TRAINABLE_ONLY=1 TSLM_GRAD_CKPT=1
export TSLM_QWEN_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

start_phase preflight
logged "$PY" -m pytest l3/tests -q
logged "$PY" - "$CONFIG" <<'PY'
import sys
from l3.grpo import load_l3_config
c=load_l3_config(sys.argv[1]); assert c["zoom_sft"]["trace_contract"]=="z1bis_v1"
assert c["model"]["encoder"]["freeze"] is True
PY
"$PY" - "$CURRENT_DIR/summary.json" <<'PY'
import json,sys
json.dump({"status":"DONE","cpu_tests":"passed","training_started":False},open(sys.argv[1],"w"),indent=2); open(sys.argv[1],"a").write("\n")
PY
finish_phase

require_ok preflight
start_phase traces
logged run_trace_planner_guarded
"$PY" - "$CURRENT_DIR/plan/summary.json" "$CURRENT_DIR/summary.json" \
  "$TRACE_TIMEOUT_SECONDS" "$TRACE_PEAK_LIMIT_MIB" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8")); assert d["status"]=="DONE" and d["pseudo_recovery_count"]==0
elapsed=d.get("planner_elapsed_seconds"); peak=d.get("planner_peak_rss_mib")
assert isinstance(elapsed,(int,float)) and elapsed <= int(sys.argv[3]),elapsed
assert isinstance(peak,(int,float)) and peak <= int(sys.argv[4]),peak
json.dump(d,open(sys.argv[2],"w",encoding="utf-8"),indent=2); open(sys.argv[2],"a").write("\n")
PY
finish_phase

require_ok traces
start_phase sft
logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/zoom_sft.py \
  --config "$CONFIG" --checkpoint "$VA1_CHECKPOINT" \
  --traces "$RUN_ROOT/traces/plan/traces.jsonl" --run-dir "$CURRENT_DIR/train"
"$PY" - "$CURRENT_DIR/train/summary.json" "$CURRENT_DIR/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8")); assert d["status"]=="DONE" and d["steps"]==4488
assert d["trace_summary"]["pseudo_recovery_count"]==0
json.dump(d,open(sys.argv[2],"w",encoding="utf-8"),indent=2); open(sys.argv[2],"a").write("\n")
PY
finish_phase

require_ok sft
start_phase perceptual_gate
VZ1BIS=$("$PY" -c "import json; print(json.load(open('$RUN_ROOT/sft/summary.json'))['checkpoint'])")
VZ1BIS_SHA=$("$PY" -c "import json; print(json.load(open('$RUN_ROOT/sft/summary.json'))['checkpoint_sha256'])")
[ "$(sha256sum "$VZ1BIS" | awk '{print $1}')" = "$VZ1BIS_SHA" ]
logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/tools/diag_crop_content.py \
  --config "$CONFIG" --checkpoint "$VZ1BIS" --reference-rows "$REFERENCE_GATE_ROWS" \
  --out-dir "$CURRENT_DIR/gate"
cp "$CURRENT_DIR/gate/summary.json" "$CURRENT_DIR/summary.json"
finish_phase

require_ok perceptual_gate
start_phase operational_gate
logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/zoom.py eval \
  --config "$CONFIG" --checkpoint "$VZ1BIS" --split val --mode policy \
  --out-json "$CURRENT_DIR/val520_policy.json" --out-jsonl "$CURRENT_DIR/val520_policy.jsonl"
logged "$PY" l3/tools/gate_z1bis_operational.py \
  --config "$CONFIG" --candidate "$CURRENT_DIR/val520_policy.jsonl" \
  --va1-reference "$VA1_VAL_JSONL" --previous-vz1-reference "$PREVIOUS_VZ1_VAL_JSONL" \
  --out "$CURRENT_DIR/summary.json"
finish_phase

trap - ERR
echo "L3_Z1BIS_OK run_root=$RUN_ROOT elapsed_s=$(( $(date +%s) - START_EPOCH ))"
