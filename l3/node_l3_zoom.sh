#!/usr/bin/env bash
# Fail-closed L3 point-4 chain. The detached worker always redirects to worker.log.
set -Eeuo pipefail

ROOT="$HOME/timerlm/sft/tslm"
PY="$HOME/venv/bin/python"
L3="$ROOT/l3"
VA1_CONFIG="$L3/configs/l3_zoom_z0.yaml"
Z1_CONFIG="$L3/configs/l3_zoom_z1.yaml"
Z2_CONFIG="$L3/configs/l3_zoom_z2.yaml"
VA1_CHECKPOINT="$ROOT/results/anomaly_xl_precise/anomaly_xl_wavelet_va1/checkpoints/epoch_5_end.pt"
VA1_TEST_JSONL="${L3_VA1_TEST_JSONL:-$ROOT/results/anomaly_xl_precise/anomaly_xl_wavelet_va1/output_logs/evalonly_test_epoch_5_end_epoch_0.jsonl}"
RL_TEST_JSONL="${L3_RL_TEST_JSONL:-$ROOT/results/anomaly_xl_precise/anomaly_xl_wavelet_va1/output_logs/evalonly_test_step_000100_composed_l3_L3_test800_epoch_0.jsonl}"
TRAIN_PARQUET="${L3_TRAIN_PARQUET:-$HOME/timerlm/anomalyXL/datasets/anomalyxl-train/anomalyxl-precise/train.parquet}"
EXPECTED_VA1_SHA="cd7238ae81da1afca9f225a99e9af4261be829798a39985a427d88a098bf2531"
EXPECTED_TRAIN_SHA="b8571d886772a55613c2e8851a774e6a561e19c8272227f5abb70e2d76e70b7d"
GPU="${L3_GPU:-0}"
PEAK_LIMIT_MIB=22528
RUN_ROOT="${L3_ZOOM_RUN_ROOT:-$HOME/l3/zoom_$(date -u +%Y%m%dT%H%M%SZ)}"

if [ "${1:-}" != "--worker" ] && [ -z "${TMUX:-}" ]; then
  command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
  [ ! -e "$RUN_ROOT" ] || { echo "refusing to reuse run root: $RUN_ROOT" >&2; exit 1; }
  mkdir -p "$RUN_ROOT"
  SESSION="l3-zoom-$(date -u +%Y%m%dT%H%M%SZ)"
  tmux new-session -d -s "$SESSION" \
    "env L3_ZOOM_RUN_ROOT='$RUN_ROOT' L3_GPU='$GPU' bash '$L3/node_l3_zoom.sh' --worker >'$RUN_ROOT/worker.log' 2>&1"
  echo "L3_ZOOM_DETACHED session=$SESSION run_root=$RUN_ROOT log=$RUN_ROOT/worker.log"
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
import json, os, sys
from datetime import datetime, timezone
d={"status":"FAILED","phase":os.environ["L3_PHASE"],"exit_code":int(os.environ["L3_RC"]),
   "line":int(os.environ["L3_LINE"]),"created_utc":datetime.now(timezone.utc).isoformat()}
with open(sys.argv[1],"w",encoding="utf-8") as h: json.dump(d,h,indent=2); h.write("\n")
PY
  fi
  echo "L3_ZOOM_FAILED phase=$CURRENT_PHASE line=$line rc=$rc" >&2
  exit "$rc"
}
trap 'failure $? $LINENO' ERR

start_phase() {
  CURRENT_PHASE="$1"; CURRENT_DIR="$RUN_ROOT/$1"
  [ ! -e "$CURRENT_DIR" ] || { echo "refusing to reuse phase directory: $CURRENT_DIR" >&2; return 1; }
  mkdir "$CURRENT_DIR"
  printf 'L3_ZOOM_STATUS phase=%s elapsed_s=%s\n' "$CURRENT_PHASE" "$(( $(date +%s) - START_EPOCH ))" | tee -a "$CURRENT_DIR/phase.log"
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

for required in "$PY" "$L3/zoom.py" "$L3/zoom_sft.py" "$L3/grpo.py" \
                "$L3/paired_eval_l3.py" "$VA1_CONFIG" "$Z1_CONFIG" "$Z2_CONFIG" \
                "$VA1_CHECKPOINT" "$TRAIN_PARQUET" "$VA1_TEST_JSONL" "$RL_TEST_JSONL"; do
  [ -e "$required" ] || { echo "missing required input: $required" >&2; exit 1; }
done
[ -x "$PY" ]
for jsonl in "$VA1_TEST_JSONL" "$RL_TEST_JSONL"; do
  rows=$(wc -l < "$jsonl")
  [ "$rows" -eq 800 ] || { echo "expected 800 rows in $jsonl, got $rows" >&2; exit 1; }
done
[ "$(sha256sum "$VA1_CHECKPOINT" | awk '{print $1}')" = "$EXPECTED_VA1_SHA" ]
[ "$(sha256sum "$TRAIN_PARQUET" | awk '{print $1}')" = "$EXPECTED_TRAIN_SHA" ]
GPU_NAME=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in *A40*) ;; *) echo "expected A40, got $GPU_NAME" >&2; exit 1 ;; esac
GPU_PROCS=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
if printf '%s\n' "$GPU_PROCS" | grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$'; then
  echo "GPU already has compute processes: $GPU_PROCS" >&2; exit 1
fi

mkdir -p "$RUN_ROOT"
cd "$ROOT"
# zoom.py / zoom_sft.py import the l3 package; running them as scripts puts l3/ (not ROOT) first on sys.path.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
git rev-parse HEAD > "$RUN_ROOT/KIT_COMMIT.txt"
git status --porcelain -- l3 >> "$RUN_ROOT/KIT_COMMIT.txt"
unset PYTORCH_CUDA_ALLOC_CONF WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
export TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16
export TSLM_ATTN_IMPL=sdpa TSLM_SAVE_TRAINABLE_ONLY=1 TSLM_GRAD_CKPT=1
export TSLM_QWEN_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# CPU gate is part of Z0 and runs before loading the GPU model.
start_phase Z0
logged "$PY" -m pytest l3/tests -q
for scope in probe80 val520; do
  max_args=()
  [ "$scope" != probe80 ] || max_args=(--max-samples 80)
  for mode in no_zoom oracle heuristic; do
    logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/zoom.py eval \
      --config "$VA1_CONFIG" --checkpoint "$VA1_CHECKPOINT" --split val --mode "$mode" \
      --out-json "$CURRENT_DIR/${scope}_${mode}.json" \
      --out-jsonl "$CURRENT_DIR/${scope}_${mode}.jsonl" "${max_args[@]}"
  done
done
"$PY" - "$CURRENT_DIR" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); docs={m:json.load(open(p/f"val520_{m}.json",encoding="utf-8"))
                         for m in ("no_zoom","oracle","heuristic")}
for d in docs.values(): assert d["samples_evaluated"]==520
cats=("localize","localize_all_channels")
reads=all(docs["oracle"]["per_category"][c]["primary"] > docs["no_zoom"]["per_category"][c]["primary"] for c in cats)
out={"status":"DONE","oracle_improves_both_localization_categories":reads,
     "flag_z1_required":not reads,"val520":{m:d["primary"] for m,d in docs.items()}}
json.dump(out,open(p/"summary.json","w",encoding="utf-8"),indent=2); open(p/"summary.json","a").write("\n")
PY
finish_phase

# Z1: one complete SFT epoch; every row contributes action and answer loss.
require_ok Z0
start_phase Z1
logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/zoom_sft.py \
  --config "$Z1_CONFIG" --checkpoint "$VA1_CHECKPOINT" --run-dir "$CURRENT_DIR/train"
VZ1=$("$PY" -c "import json; print(json.load(open('$CURRENT_DIR/train/summary.json'))['checkpoint'])")
VZ1_SHA=$("$PY" -c "import json; print(json.load(open('$CURRENT_DIR/train/summary.json'))['checkpoint_sha256'])")
[ "$(sha256sum "$VZ1" | awk '{print $1}')" = "$VZ1_SHA" ]
logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/zoom.py eval \
  --config "$Z1_CONFIG" --checkpoint "$VZ1" --split val --mode policy \
  --out-json "$CURRENT_DIR/val520_policy.json" --out-jsonl "$CURRENT_DIR/val520_policy.jsonl"
"$PY" - "$CURRENT_DIR" "$RUN_ROOT/Z0/val520_no_zoom.json" "$VZ1" "$VZ1_SHA" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); z=json.load(open(p/"val520_policy.json",encoding="utf-8")); b=json.load(open(sys.argv[2],encoding="utf-8"))
assert z["samples_evaluated"]==520 and z["primary"]>=0.3105
for c in ("localize","localize_all_channels"):
    assert z["per_category"][c]["primary"] >= b["per_category"][c]["primary"]
train=json.load(open(p/"train"/"summary.json",encoding="utf-8")); assert train["steps"]==4488
out={"status":"DONE","checkpoint":sys.argv[3],"checkpoint_sha256":sys.argv[4],
     "primary":z["primary"],"zoom_rate":z["zoom_rate"],"zoom_invalid":z["zoom_invalid"]}
json.dump(out,open(p/"summary.json","w",encoding="utf-8"),indent=2); open(p/"summary.json","a").write("\n")
PY
finish_phase

# Z2: identity, worst-shape memory gate, then resumable 25-step chunks.
require_ok Z1
start_phase Z2
VZ1=$("$PY" -c "import json; print(json.load(open('$RUN_ROOT/Z1/summary.json'))['checkpoint'])")
VZ1_SHA=$("$PY" -c "import json; print(json.load(open('$RUN_ROOT/Z1/summary.json'))['checkpoint_sha256'])")
[ "$(sha256sum "$VZ1" | awk '{print $1}')" = "$VZ1_SHA" ]
logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/grpo.py init-check \
  --config "$Z2_CONFIG" --checkpoint "$VZ1" --out "$CURRENT_DIR/init_identity.json"
logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/grpo.py train \
  --config "$Z2_CONFIG" --checkpoint "$VZ1" --mode zoom-memory --run-dir "$CURRENT_DIR/memory"
"$PY" - "$CURRENT_DIR/memory/summary.json" "$PEAK_LIMIT_MIB" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding="utf-8")); assert d["steps"]==1 and d["rollouts"]==16
assert d["peak_mem_mib"] < float(sys.argv[2]) and d["parsable"]+d["unparsable"]==16
PY

TRAIN_DIR="$CURRENT_DIR/train"
for step in $(seq 25 25 200); do
  if [ "$step" -eq 25 ]; then
    logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/grpo.py train \
      --config "$Z2_CONFIG" --checkpoint "$VZ1" --mode zoom --target-step "$step" --run-dir "$TRAIN_DIR"
  else
    logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/grpo.py train \
      --config "$Z2_CONFIG" --checkpoint "$VZ1" --mode zoom --target-step "$step" --resume "$TRAIN_DIR"
  fi
  if [ $((step % 50)) -eq 0 ]; then
    CKPT="$TRAIN_DIR/checkpoints/step_$(printf '%06d' "$step")_composed.pt"
    logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/zoom.py eval \
      --config "$Z2_CONFIG" --checkpoint "$CKPT" --split val --mode policy \
      --out-json "$CURRENT_DIR/val520_step_${step}.json" \
      --out-jsonl "$CURRENT_DIR/val520_step_${step}.jsonl"
    "$PY" - "$CURRENT_DIR" "$step" "$CKPT" <<'PY'
import hashlib,json,sys
from pathlib import Path
p,step,ckpt=Path(sys.argv[1]),int(sys.argv[2]),Path(sys.argv[3]); hp=p/"val_history.json"
h=json.load(open(hp,encoding="utf-8")) if hp.exists() else []
d=json.load(open(p/f"val520_step_{step}.json",encoding="utf-8")); prior=max((x["primary"] for x in h),default=None)
stale=0 if prior is None or d["primary"]>=prior+0.005 else h[-1]["stale"]+1
h.append({"step":step,"primary":d["primary"],"stale":stale,"checkpoint":str(ckpt),
          "checkpoint_sha256":hashlib.sha256(ckpt.read_bytes()).hexdigest()})
best=max(x["primary"] for x in h); chosen=next(x for x in h if x["primary"]>=best-0.002)
sel={"status":"FROZEN" if stale>=2 or step==200 else "PROVISIONAL","history":h,
     "selected_step":chosen["step"],"selected_primary":chosen["primary"],
     "checkpoint":chosen["checkpoint"],"checkpoint_sha256":chosen["checkpoint_sha256"],"stale":stale}
json.dump(h,open(hp,"w",encoding="utf-8"),indent=2); open(hp,"a").write("\n")
json.dump(sel,open(p/"selection.json","w",encoding="utf-8"),indent=2); open(p/"selection.json","a").write("\n")
PY
    STALE=$("$PY" -c "import json; print(json.load(open('$CURRENT_DIR/selection.json'))['stale'])")
    [ "$STALE" -lt 2 ] || break
  fi
done
"$PY" - "$CURRENT_DIR" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); s=json.load(open(p/"selection.json",encoding="utf-8")); assert s["status"]=="FROZEN"
json.dump({"status":"DONE",**s},open(p/"summary.json","w",encoding="utf-8"),indent=2); open(p/"summary.json","a").write("\n")
PY
finish_phase

# Z3: the only test access in this script; two zoom arms, one read each.
require_ok Z2
start_phase Z3
[ -f "$VA1_TEST_JSONL" ] && [ -f "$RL_TEST_JSONL" ]
VZ1=$("$PY" -c "import json; print(json.load(open('$RUN_ROOT/Z1/summary.json'))['checkpoint'])")
SELECTED=$("$PY" -c "import json; print(json.load(open('$RUN_ROOT/Z2/selection.json'))['checkpoint'])")
for arm in vz1 vz1_rl; do
  CKPT="$VZ1"; [ "$arm" != vz1_rl ] || CKPT="$SELECTED"
  logged env CUDA_VISIBLE_DEVICES="$GPU" "$PY" l3/zoom.py eval \
    --config "$Z2_CONFIG" --checkpoint "$CKPT" --split test --mode policy \
    --out-json "$CURRENT_DIR/${arm}_test800.json" --out-jsonl "$CURRENT_DIR/${arm}_test800.jsonl"
  for control in va1 va1_rl; do
    REF="$VA1_TEST_JSONL"; [ "$control" != va1_rl ] || REF="$RL_TEST_JSONL"
    logged "$PY" l3/paired_eval_l3.py --reference "$REF" \
      --candidate "$CURRENT_DIR/${arm}_test800.jsonl" --split test --seed 20260911 \
      --out-json "$CURRENT_DIR/${arm}_vs_${control}.json" \
      --out-markdown "$CURRENT_DIR/${arm}_vs_${control}.md"
  done
done
"$PY" - "$CURRENT_DIR" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); arms={a:json.load(open(p/f"{a}_test800.json",encoding="utf-8")) for a in ("vz1","vz1_rl")}
for d in arms.values(): assert d["samples_evaluated"]==800
out={"status":"DONE","arms":{a:{"primary":d["primary"],"zoom_rate":d["zoom_rate"],
     "zoom_invalid":d["zoom_invalid"],"lead_lag_direction_accuracy":
     d["per_category"]["lead_lag_with_magnitude"].get("direction_correct",0.0)} for a,d in arms.items()}}
json.dump(out,open(p/"summary.json","w",encoding="utf-8"),indent=2); open(p/"summary.json","a").write("\n")
PY
finish_phase

trap - ERR
echo "L3_ZOOM_OK run_root=$RUN_ROOT elapsed_s=$(( $(date +%s) - START_EPOCH ))"
