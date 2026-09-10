#!/usr/bin/env bash
# Fail-closed L3 point-5 chain.  Run inside tmux, or let this script detach itself.
set -Eeuo pipefail

ROOT="$HOME/timerlm/sft/tslm"
PY="$HOME/venv/bin/python"
L3="$ROOT/l3"
CONFIG="$L3/configs/l3_longrun.yaml"
VA1_CONFIG="$ROOT/configs/wavelet_precise_va1.yaml"
PROBE_CONFIG="$ROOT/configs/wavelet_precise_va1_valprobe.yaml"
CHECKPOINT="$ROOT/results/anomaly_xl_precise/anomaly_xl_wavelet_va1/checkpoints/epoch_5_end.pt"
REFERENCE_TEST_JSONL="$ROOT/results/anomaly_xl_precise/anomaly_xl_wavelet_va1/output_logs/evalonly_test_epoch_5_end_epoch_0.jsonl"
TRAIN_PARQUET="$ROOT/anomalyXL/datasets/anomalyxl-train/anomalyxl-precise/train.parquet"
GPU="${L3_GPU:-0}"
EXPECTED_CHECKPOINT_SHA="cd7238ae81da1afca9f225a99e9af4261be829798a39985a427d88a098bf2531"
EXPECTED_TRAIN_SHA="b8571d886772a55613c2e8851a774e6a561e19c8272227f5abb70e2d76e70b7d"
PEAK_LIMIT_MIB=22528
TOTAL_ESTIMATE_HOURS=58
RUN_ROOT="${L3_RUN_ROOT:-$HOME/l3/longrun_$(date -u +%Y%m%dT%H%M%SZ)}"

if [ "${1:-}" != "--worker" ] && [ -z "${TMUX:-}" ]; then
  command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
  SESSION="l3-longrun-$(date -u +%Y%m%dT%H%M%SZ)"
  tmux new-session -d -s "$SESSION" \
    "env L3_RUN_ROOT='$RUN_ROOT' L3_GPU='$GPU' bash '$L3/node_l3_longrun.sh' --worker"
  echo "L3_LONGRUN_DETACHED session=$SESSION run_root=$RUN_ROOT"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

START_EPOCH=$(date +%s)
CURRENT_PHASE="startup"
CURRENT_DIR=""

elapsed_status() {
  local now elapsed remaining
  now=$(date +%s)
  elapsed=$((now - START_EPOCH))
  remaining=$((TOTAL_ESTIMATE_HOURS * 3600 - elapsed))
  [ "$remaining" -gt 0 ] || remaining=0
  printf 'L3_STATUS phase=%s elapsed_h=%.2f estimated_remaining_h=%.2f\n' \
    "$CURRENT_PHASE" "$((elapsed / 36))e-2" "$((remaining / 36))e-2"
}

write_failure() {
  local rc="$1" line="$2"
  local phase_root="$RUN_ROOT/$CURRENT_PHASE"
  set +e
  if [ -n "$CURRENT_DIR" ] && [ -d "$CURRENT_DIR" ]; then
    rm -f "$CURRENT_DIR/OK" "$phase_root/OK"
    L3_FAILURE_PHASE="$CURRENT_PHASE" L3_FAILURE_RC="$rc" L3_FAILURE_LINE="$line" \
      "$PY" - "$CURRENT_DIR/summary.json" "$CURRENT_DIR/FAILED" <<'PY'
import json, os, sys
from datetime import datetime, timezone
document = {
    "status": "FAILED", "phase": os.environ["L3_FAILURE_PHASE"],
    "exit_code": int(os.environ["L3_FAILURE_RC"]),
    "line": int(os.environ["L3_FAILURE_LINE"]),
    "reason": "command failed; see phase.log and nested runner FAILED marker",
    "created_utc": datetime.now(timezone.utc).isoformat(),
}
for target in sys.argv[1:]:
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2); handle.write("\n")
PY
    if [ "$CURRENT_DIR" != "$phase_root" ] && [ -d "$phase_root" ]; then
      cp "$CURRENT_DIR/summary.json" "$phase_root/summary.json"
      cp "$CURRENT_DIR/FAILED" "$phase_root/FAILED"
    fi
  fi
  echo "L3_LONGRUN_FAILED phase=$CURRENT_PHASE line=$line rc=$rc" >&2
  exit "$rc"
}
trap 'write_failure $? $LINENO' ERR

start_phase() {
  CURRENT_PHASE="$1"
  CURRENT_DIR="$RUN_ROOT/$1"
  [ ! -e "$CURRENT_DIR" ] || { echo "refusing to reuse phase directory $CURRENT_DIR" >&2; return 1; }
  mkdir "$CURRENT_DIR"
  elapsed_status | tee -a "$CURRENT_DIR/phase.log"
}

finish_phase() {
  [ -f "$CURRENT_DIR/summary.json" ] || { echo "phase summary missing" >&2; return 1; }
  "$PY" - "$CURRENT_DIR/summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1], encoding="utf-8"))
assert d.get("status") in {"DONE", "OK"}, d
PY
  : > "$CURRENT_DIR/OK"
  elapsed_status | tee -a "$CURRENT_DIR/phase.log"
}

logged() {
  "$@" 2>&1 | tee -a "$CURRENT_DIR/phase.log"
}

for required in "$PY" "$L3/grpo.py" "$L3/paired_eval_l3.py" "$CONFIG" \
                "$VA1_CONFIG" "$PROBE_CONFIG" "$CHECKPOINT" "$TRAIN_PARQUET"; do
  [ -e "$required" ] || { echo "missing required input: $required" >&2; exit 1; }
done
[ -x "$PY" ] || { echo "Level-1 venv interpreter is not executable" >&2; exit 1; }
[ ! -e "$RUN_ROOT" ] || { echo "refusing to reuse run root: $RUN_ROOT" >&2; exit 1; }

OBSERVED=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
[ "$OBSERVED" = "$EXPECTED_CHECKPOINT_SHA" ] || { echo "initial checkpoint SHA mismatch" >&2; exit 1; }
OBSERVED=$(sha256sum "$TRAIN_PARQUET" | awk '{print $1}')
[ "$OBSERVED" = "$EXPECTED_TRAIN_SHA" ] || { echo "train parquet SHA mismatch" >&2; exit 1; }

GPU_NAME=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader | head -1)
GPU_TOTAL=$(nvidia-smi -i "$GPU" --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
case "$GPU_NAME" in *A40*) ;; *) echo "expected A40, got $GPU_NAME" >&2; exit 1 ;; esac
[ "$GPU_TOTAL" -gt "$PEAK_LIMIT_MIB" ] || { echo "GPU memory is below the gate" >&2; exit 1; }
GPU_PROCS=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
if printf '%s\n' "$GPU_PROCS" | grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$'; then
  echo "GPU already has compute processes: $GPU_PROCS" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
cd "$ROOT"
{
  git rev-parse HEAD
  if [ -n "$(git status --porcelain -- l3)" ]; then echo "DIRTY_L3"; fi
} > "$RUN_ROOT/KIT_COMMIT.txt"
"$PY" - "$CONFIG" "$RUN_ROOT/resolved_config.json" <<'PY'
import json, sys
from l3.grpo import load_l3_config
with open(sys.argv[2], "x", encoding="utf-8") as h:
    json.dump(load_l3_config(sys.argv[1]), h, indent=2, sort_keys=True); h.write("\n")
PY
"$PY" - "$RUN_ROOT" "$CHECKPOINT" "$TRAIN_PARQUET" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
root, checkpoint, train = map(Path, sys.argv[1:])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
d={
  "format":"l3_node_longrun_manifest_v1", "created_utc":datetime.now(timezone.utc).isoformat(),
  "initial_checkpoint":str(checkpoint), "initial_checkpoint_sha256":sha(checkpoint),
  "train_parquet":str(train), "train_sha256":sha(train),
  "kit_commit_file":str(root/"KIT_COMMIT.txt"), "kit_commit_file_sha256":sha(root/"KIT_COMMIT.txt"),
  "resolved_config":str(root/"resolved_config.json"), "resolved_config_sha256":sha(root/"resolved_config.json"),
  "seeds":{"pool":20260911,"main":20260911,"replicas":[20260912,20260913]},
}
with open(root/"RUN_MANIFEST.json","x",encoding="utf-8") as h:
    json.dump(d,h,indent=2,sort_keys=True); h.write("\n")
PY

# Native allocator, one process, bf16/SDPA/gradient checkpointing.
unset PYTORCH_CUDA_ALLOC_CONF WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
export TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16
export TSLM_ATTN_IMPL=sdpa TSLM_SAVE_TRAINABLE_ONLY=1 TSLM_GRAD_CKPT=1
export TSLM_QWEN_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# L0: worst-shape P2/G8 memory gate.
start_phase L0
logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/grpo.py train \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --mode longrun-memory \
  --run-dir "$CURRENT_DIR/train"
cp "$CURRENT_DIR/train/summary.json" "$CURRENT_DIR/summary.json"
"$PY" - "$CURRENT_DIR/summary.json" "$PEAK_LIMIT_MIB" <<'PY'
import json, math, sys
d=json.load(open(sys.argv[1],encoding="utf-8"))
assert d["steps"]==1 and d["rollouts"]==16
assert d["parsable"]+d["unparsable"]==16
assert d["peak_mem_mib"] < float(sys.argv[2])
assert all(math.isfinite(float(d[k])) for k in ("reward_mean","kl_mean","peak_mem_mib"))
PY
finish_phase

# L1: frozen train-only pool, T=0.7, at most 200 G8 candidates.
start_phase L1
logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/grpo.py pool \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --run-dir "$CURRENT_DIR/filter"
cp "$CURRENT_DIR/filter/summary.json" "$CURRENT_DIR/summary.json"
"$PY" - "$RUN_ROOT/RUN_MANIFEST.json" "$CURRENT_DIR/filter/pool.json" <<'PY'
import hashlib,json,sys
manifest=json.load(open(sys.argv[1],encoding="utf-8")); pool=sys.argv[2]
manifest["pool"]={"path":pool,"sha256":hashlib.sha256(open(pool,"rb").read()).hexdigest()}
with open(sys.argv[1],"w",encoding="utf-8") as h:
    json.dump(manifest,h,indent=2,sort_keys=True); h.write("\n")
PY
finish_phase
POOL="$RUN_ROOT/L1/filter/pool.json"

eval_checkpoint() {
  local train_dir="$1" step="$2" kind="$3" eval_config="$4" strict="$5" expected="$6" split="$7"
  local checkpoint="$train_dir/checkpoints/step_$(printf '%06d' "$step")_composed.pt" run_tag
  local eval_dir="$CURRENT_DIR/eval" raw sealed before
  mkdir -p "$eval_dir"
  raw="$eval_dir/${kind}_step_$(printf '%06d' "$step")_raw.json"
  sealed="$eval_dir/${kind}_step_$(printf '%06d' "$step").json"
  run_tag=$(basename "$(dirname "$train_dir")")
  before=$(sha256sum "$checkpoint" | awk '{print $1}')
  logged env TSLM_EVAL_STRICT="$strict" TSLM_EVAL_TAG="l3_${run_tag}_${kind}_${step}" \
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" train.py --config "$eval_config" --no-wandb \
    --eval-only --checkpoint "$checkpoint" --split "$split" --out-json "$raw"
  logged "$PY" l3/grpo.py seal-eval --checkpoint "$checkpoint" \
    --manifest "$train_dir/RUN_MANIFEST.json" --eval-json "$raw" --out "$sealed" \
    --before-sha256 "$before" --split "$split" --expected-rows "$expected"
  LAST_EVAL="$sealed"
}

update_selection() {
  local train_dir="$1" val_json="$2" step="$3"
  "$PY" - "$CURRENT_DIR" "$train_dir" "$val_json" "$step" <<'PY'
import json,sys
from pathlib import Path
phase,train,val_path,step=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),int(sys.argv[4])
history_path=phase/"val_history.json"
history=json.load(open(history_path,encoding="utf-8")) if history_path.exists() else []
result=json.load(open(val_path,encoding="utf-8"))
primary=float(result["primary"])
prior_max=max((x["primary"] for x in history),default=None)
stale=0 if prior_max is None or primary >= prior_max+0.005 else (history[-1]["stale"]+1)
history.append({"step":step,"primary":primary,"sealed_eval":str(val_path),"stale":stale})
max_primary=max(x["primary"] for x in history)
selected=next(x for x in history if x["primary"] >= max_primary-0.002)
manifest=json.load(open(train/"RUN_MANIFEST.json",encoding="utf-8"))
record=next(x for x in manifest["checkpoints"] if x["step"]==selected["step"])
selection={
 "status":"FROZEN" if stale>=2 or step==400 else "PROVISIONAL",
 "selected_step":selected["step"],"selected_primary":selected["primary"],
 "maximum_observed_primary":max_primary,"tie_tolerance":0.002,
 "minimum_improvement":0.005,"consecutive_stale_validations":stale,
 "checkpoint":str((train/record["composed_file"]).resolve()),
 "checkpoint_sha256":record["composed_sha256"],"history":history,
}
with open(history_path,"w",encoding="utf-8") as h: json.dump(history,h,indent=2); h.write("\n")
with open(phase/"selection.json","w",encoding="utf-8") as h: json.dump(selection,h,indent=2); h.write("\n")
PY
}

run_seed_phase() {
  local seed="$1" train_dir="$CURRENT_DIR/train" step stale
  for step in $(seq 25 25 400); do
    if [ "$step" -eq 25 ]; then
      logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/grpo.py train \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" --mode longrun --pool "$POOL" \
        --seed "$seed" --target-step "$step" --run-dir "$train_dir"
    else
      logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/grpo.py train \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" --mode longrun --pool "$POOL" \
        --seed "$seed" --target-step "$step" --resume "$train_dir"
    fi
    eval_checkpoint "$train_dir" "$step" probe80 "$PROBE_CONFIG" 0 80 val
    if [ $((step % 100)) -eq 0 ]; then
      eval_checkpoint "$train_dir" "$step" val520 "$VA1_CONFIG" 1 520 val
      update_selection "$train_dir" "$LAST_EVAL" "$step"
      stale=$("$PY" -c "import json; print(json.load(open('$CURRENT_DIR/selection.json'))['consecutive_stale_validations'])")
      if [ "$stale" -ge 2 ]; then
        echo "L3_EARLY_STOP seed=$seed step=$step stale=$stale" | tee -a "$CURRENT_DIR/phase.log"
        break
      fi
    fi
    elapsed_status | tee -a "$CURRENT_DIR/phase.log"
  done
  "$PY" - "$CURRENT_DIR" "$seed" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); selection=json.load(open(p/"selection.json",encoding="utf-8"))
d={"status":"DONE","seed":int(sys.argv[2]),"selected_step":selection["selected_step"],
   "selected_primary":selection["selected_primary"],"selection":str(p/"selection.json")}
with open(p/"summary.json","w",encoding="utf-8") as h: json.dump(d,h,indent=2); h.write("\n")
PY
}

# L2: predesignated main seed; full validation alone selects and stops training.
start_phase L2
run_seed_phase 20260911
finish_phase

# L3: the only test access, after L2/OK and frozen L2 selection.
start_phase L3
[ -f "$RUN_ROOT/L2/OK" ]
[ -f "$REFERENCE_TEST_JSONL" ]
SELECTED_CHECKPOINT=$("$PY" -c "import json; print(json.load(open('$RUN_ROOT/L2/selection.json'))['checkpoint'])")
SELECTED_STEP=$("$PY" -c "import json; print(json.load(open('$RUN_ROOT/L2/selection.json'))['selected_step'])")
SELECTED_SHA=$("$PY" -c "import json; print(json.load(open('$RUN_ROOT/L2/selection.json'))['checkpoint_sha256'])")
[ "$(sha256sum "$SELECTED_CHECKPOINT" | awk '{print $1}')" = "$SELECTED_SHA" ]
mkdir "$CURRENT_DIR/eval"
TEST_RAW="$CURRENT_DIR/eval/test800_raw.json"
TEST_SEALED="$CURRENT_DIR/eval/test800.json"
BEFORE=$(sha256sum "$SELECTED_CHECKPOINT" | awk '{print $1}')
logged env TSLM_EVAL_STRICT=1 TSLM_EVAL_TAG="l3_L3_test800" CUDA_VISIBLE_DEVICES="$GPU" \
  "$PY" train.py --config "$VA1_CONFIG" --no-wandb --eval-only \
  --checkpoint "$SELECTED_CHECKPOINT" --split test --out-json "$TEST_RAW"
logged "$PY" l3/grpo.py seal-eval --checkpoint "$SELECTED_CHECKPOINT" \
  --manifest "$RUN_ROOT/L2/train/RUN_MANIFEST.json" --eval-json "$TEST_RAW" --out "$TEST_SEALED" \
  --before-sha256 "$BEFORE" --split test --expected-rows 800
CANDIDATE_JSONL=$("$PY" -c "import json; print(json.load(open('$TEST_SEALED'))['eval_jsonl'])")
logged "$PY" l3/paired_eval_l3.py --reference "$REFERENCE_TEST_JSONL" \
  --candidate "$CANDIDATE_JSONL" --split test --seed 20260911 \
  --out-json "$CURRENT_DIR/paired_test.json" --out-markdown "$CURRENT_DIR/paired_test.md"
"$PY" - "$CURRENT_DIR" "$SELECTED_STEP" "$TEST_SEALED" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d={"status":"DONE","selected_step":int(sys.argv[2]),
 "test_eval":sys.argv[3],"paired_json":str(p/"paired_test.json"),"paired_markdown":str(p/"paired_test.md")}
with open(p/"summary.json","x",encoding="utf-8") as h: json.dump(d,h,indent=2); h.write("\n")
PY
finish_phase

# L4: two untouched-seed replicas, same frozen pool, validation only.
start_phase L4
for seed in 20260912 20260913; do
  REPLICA_DIR="$CURRENT_DIR/seed_$seed"
  mkdir "$REPLICA_DIR"
  SAVED_DIR="$CURRENT_DIR"
  CURRENT_DIR="$REPLICA_DIR"
  : > "$CURRENT_DIR/phase.log"
  run_seed_phase "$seed"
  : > "$CURRENT_DIR/OK"
  CURRENT_DIR="$SAVED_DIR"
done
"$PY" - "$CURRENT_DIR" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); rows=[]
for seed in (20260912,20260913):
 d=json.load(open(p/f"seed_{seed}"/"summary.json",encoding="utf-8")); rows.append(d)
with open(p/"summary.json","x",encoding="utf-8") as h:
 json.dump({"status":"DONE","replicas":rows},h,indent=2); h.write("\n")
PY
finish_phase

trap - ERR
CURRENT_PHASE=complete
elapsed_status
echo "L3_LONGRUN_OK run_root=$RUN_ROOT"
