#!/usr/bin/env bash
# W2: recupero mirato della lettura multicanale, partendo da vw1.
# tracce -> SFT 1 epoca (solo LoRA) -> lettura su 3 crop per ID a finestre identiche
# -> operativo su val520. Criteri in PROTOCOL-L3.md, nessun accesso al test 800.
set -Eeuo pipefail

ROOT="${L3_ROOT:-$HOME/timerlm/sft/tslm}"
PY="${L3_PYTHON:-$HOME/venv/bin/python}"
L3="$ROOT/l3"
W2_CONFIG="$L3/configs/l3_zoom_w2.yaml"
Z1BIS_CONFIG="$L3/configs/l3_zoom_z1bis.yaml"
W1_CHECKPOINT="${L3_W1_CHECKPOINT:-$HOME/l3/night_20260913T005001Z/w1_sft/train/vz1bis_epoch_1_end.pt}"
W1_SHA="${L3_W1_SHA:-8221cd4aba0a9101374ec551ba31f239020586bb6fcc1257a407abdee748ed10}"
W1_POLICY_ROWS="${L3_W1_POLICY_ROWS:-$HOME/l3/night_20260913T005001Z/w1_operational/val520_policy.jsonl}"
REFERENCE_GATE_ROWS="${L3_REFERENCE_GATE_ROWS:-$HOME/l3/zoom_20260911c/Z1/gate/rows.jsonl}"
VA1_VAL_JSONL="${L3_VA1_VAL_JSONL:-$HOME/l3/zoom_20260911c/Z0/val520_no_zoom.jsonl}"
# Letture di riferimento già su disco: Z1-bis e W1 sugli stessi crop, offset 0/1/2.
Z1BIS_READ_0="${L3_Z1BIS_READ_0:-$HOME/l3/z1bis_20260912/perceptual_gate/gate/rows.jsonl}"
Z1BIS_READ_1="${L3_Z1BIS_READ_1:-$HOME/l3/sep_z1bis_off1/rows.jsonl}"
Z1BIS_READ_2="${L3_Z1BIS_READ_2:-$HOME/l3/sep_z1bis_off2/rows.jsonl}"
W1_READ_0="${L3_W1_READ_0:-$HOME/l3/w1_perceptual/rows.jsonl}"
W1_READ_1="${L3_W1_READ_1:-$HOME/l3/sep_w1_off1/rows.jsonl}"
W1_READ_2="${L3_W1_READ_2:-$HOME/l3/sep_w1_off2/rows.jsonl}"
GPU="${L3_GPU:-0}"
RUN_ROOT="${L3_W2_RUN_ROOT:-$HOME/l3/w2_$(date -u +%Y%m%dT%H%M%SZ)}"
TRACE_TIMEOUT_SECONDS="${L3_W2_TRACE_TIMEOUT_SECONDS:-3600}"
TRACE_PEAK_LIMIT_MIB="${L3_W2_TRACE_PEAK_LIMIT_MIB:-8192}"
DEADLINE_HOURS="${L3_W2_DEADLINE_HOURS:-12}"

[ "${L3_W2_DIFF_REVIEWED:-}" = "CLAUDE_APPROVED" ] || {
  echo "refusing to start: set L3_W2_DIFF_REVIEWED=CLAUDE_APPROVED after diff review" >&2
  exit 1
}

if [ "${1:-}" != "--worker" ] && [ -z "${TMUX:-}" ]; then
  command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
  [ ! -e "$RUN_ROOT" ] || { echo "refusing to reuse run root: $RUN_ROOT" >&2; exit 1; }
  mkdir -p "$RUN_ROOT"
  SESSION="l3-w2-$(date -u +%Y%m%dT%H%M%SZ)"
  tmux new-session -d -s "$SESSION" \
    "env L3_W2_DIFF_REVIEWED=CLAUDE_APPROVED L3_W2_RUN_ROOT='$RUN_ROOT' L3_GPU='$GPU' L3_ROOT='$ROOT' L3_PYTHON='$PY' L3_W1_CHECKPOINT='$W1_CHECKPOINT' L3_W1_POLICY_ROWS='$W1_POLICY_ROWS' L3_W2_DEADLINE_HOURS='$DEADLINE_HOURS' bash '$L3/node_l3_w2.sh' --worker >'$RUN_ROOT/worker.log' 2>&1"
  echo "L3_W2_DETACHED session=$SESSION run_root=$RUN_ROOT log=$RUN_ROOT/worker.log"
  exit 0
fi

START_EPOCH=$(date +%s)
DEADLINE_EPOCH=$(( START_EPOCH + DEADLINE_HOURS * 3600 ))
CURRENT_PHASE=startup
CURRENT_DIR=""

failure() {
  local rc="$1" line="$2"
  set +e
  if [ -n "$CURRENT_DIR" ] && [ -d "$CURRENT_DIR" ]; then
    rm -f "$CURRENT_DIR/OK"
    printf '{"status":"FAILED","phase":"%s","exit_code":%s,"line":%s}\n' \
      "$CURRENT_PHASE" "$rc" "$line" >"$CURRENT_DIR/FAILED"
  fi
  echo "L3_W2_FAILED phase=$CURRENT_PHASE line=$line rc=$rc" >&2
  exit "$rc"
}
trap 'failure $? $LINENO' ERR

start_phase() {
  CURRENT_PHASE="$1"
  CURRENT_DIR="$RUN_ROOT/$1"
  [ ! -e "$CURRENT_DIR" ] || { echo "refusing to reuse phase: $CURRENT_DIR" >&2; return 1; }
  mkdir -p "$CURRENT_DIR"
  printf 'L3_W2_STATUS phase=%s elapsed_s=%s\n' "$CURRENT_PHASE" \
    "$(( $(date +%s) - START_EPOCH ))" | tee -a "$CURRENT_DIR/phase.log"
}

finish_phase() {
  : >"$CURRENT_DIR/OK"
  printf 'L3_W2_PHASE_OK phase=%s elapsed_s=%s\n' "$CURRENT_PHASE" \
    "$(( $(date +%s) - START_EPOCH ))" | tee -a "$CURRENT_DIR/phase.log"
}

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
unset PYTORCH_CUDA_ALLOC_CONF WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
export TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16
export TSLM_ATTN_IMPL=sdpa TSLM_SAVE_TRAINABLE_ONLY=1 TSLM_GRAD_CKPT=1
export TSLM_QWEN_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# ---- preflight -------------------------------------------------------------
start_phase preflight
for required in "$W2_CONFIG" "$Z1BIS_CONFIG" "$W1_CHECKPOINT" "$W1_POLICY_ROWS" \
                "$REFERENCE_GATE_ROWS" "$VA1_VAL_JSONL" "$Z1BIS_READ_0" "$Z1BIS_READ_1" \
                "$Z1BIS_READ_2" "$W1_READ_0" "$W1_READ_1" "$W1_READ_2"; do
  [ -e "$required" ] || { echo "missing prerequisite: $required" >&2; exit 1; }
done
[ "$(sha256sum "$W1_CHECKPOINT" | awk '{print $1}')" = "$W1_SHA" ] || {
  echo "W1 checkpoint SHA mismatch" >&2; exit 1; }
git -C "$ROOT" rev-parse HEAD > "$RUN_ROOT/KIT_COMMIT.txt" 2>/dev/null || true
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest "$L3/tests/test_l3_w1.py" \
  "$L3/tests/test_l3_z1bis.py" "$L3/tests/test_l3_selectors.py" -q \
  2>&1 | tee "$CURRENT_DIR/pytest.log"
finish_phase

# ---- tracce ----------------------------------------------------------------
start_phase traces
set +e
CUDA_VISIBLE_DEVICES="" timeout --signal=TERM "$TRACE_TIMEOUT_SECONDS" \
  "$PY" "$L3/zoom_traces_w1.py" --config "$W2_CONFIG" \
  --out-dir "$CURRENT_DIR/plan" --seed 20260914 2>&1 | tee "$CURRENT_DIR/traces.log"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" -eq 0 ] || { echo "W2 trace planning failed rc=$RC" >&2; exit "$RC"; }
"$PY" - "$CURRENT_DIR/plan/summary.json" "$TRACE_PEAK_LIMIT_MIB" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["status"] == "DONE", d["status"]
peak = d.get("planner_peak_rss_mib")
assert peak is None or peak <= float(sys.argv[2]), peak
block = d["w1"]
declared = block["perceptual_replay_declared"]
assert declared and declared["categories"] == ["localize_all_channels"], declared
rows = block["perceptual_replay_rows"]
assert rows > 0, "perceptual replay produced no rows"
print(f"W2 perceptual_replay={rows} righe | wavelet_rate={block['wavelet_rate']:.4f}")
PYEOF
finish_phase

# ---- SFT -------------------------------------------------------------------
start_phase sft
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/zoom_sft.py" \
  --config "$W2_CONFIG" --checkpoint "$W1_CHECKPOINT" \
  --traces "$RUN_ROOT/traces/plan/traces.jsonl" --run-dir "$CURRENT_DIR/train" \
  2>&1 | tee "$CURRENT_DIR/sft.log"
cp "$CURRENT_DIR/train/summary.json" "$CURRENT_DIR/summary.json"
VW2=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['checkpoint'])" \
  "$CURRENT_DIR/summary.json")
echo "$VW2" > "$RUN_ROOT/CHECKPOINT.txt"
finish_phase

# ---- obiettivo 1: lettura sugli stessi crop --------------------------------
start_phase reading
VW2=$(cat "$RUN_ROOT/CHECKPOINT.txt")
for offset in 0 1 2; do
  OUT="$CURRENT_DIR/off$offset"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/tools/diag_crop_content.py" \
    --config "$Z1BIS_CONFIG" --checkpoint "$VW2" \
    --reference-rows "$REFERENCE_GATE_ROWS" --out-dir "$OUT" \
    --seed-offset "$offset" 2>&1 | tee -a "$CURRENT_DIR/reading.log"
done
"$PY" "$L3/tools/reader_vs_windows.py" \
  --pair "$W1_READ_0:$CURRENT_DIR/off0/rows.jsonl" \
  --pair "$W1_READ_1:$CURRENT_DIR/off1/rows.jsonl" \
  --pair "$W1_READ_2:$CURRENT_DIR/off2/rows.jsonl" \
  --out "$CURRENT_DIR/vs_w1.json" 2>&1 | tee -a "$CURRENT_DIR/reading.log"
"$PY" "$L3/tools/reader_vs_windows.py" \
  --pair "$Z1BIS_READ_0:$CURRENT_DIR/off0/rows.jsonl" \
  --pair "$Z1BIS_READ_1:$CURRENT_DIR/off1/rows.jsonl" \
  --pair "$Z1BIS_READ_2:$CURRENT_DIR/off2/rows.jsonl" \
  --out "$CURRENT_DIR/vs_z1bis.json" 2>&1 | tee -a "$CURRENT_DIR/reading.log"
finish_phase

# ---- obiettivo 2: policy autonoma su val520 --------------------------------
start_phase operational
VW2=$(cat "$RUN_ROOT/CHECKPOINT.txt")
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/zoom.py" eval \
  --config "$W2_CONFIG" --checkpoint "$VW2" --split val --mode policy \
  --out-json "$CURRENT_DIR/val520_policy.json" \
  --out-jsonl "$CURRENT_DIR/val520_policy.jsonl" 2>&1 | tee "$CURRENT_DIR/eval.log"
set +e
CUDA_VISIBLE_DEVICES="" "$PY" "$L3/tools/gate_z1bis_operational.py" \
  --config "$W2_CONFIG" --candidate "$CURRENT_DIR/val520_policy.jsonl" \
  --va1-reference "$VA1_VAL_JSONL" --previous-vz1-reference "$W1_POLICY_ROWS" \
  --out "$CURRENT_DIR/gate.json" 2>&1 | tee "$CURRENT_DIR/gate.log"
GATE_RC=${PIPESTATUS[0]}
set -e
printf '{"gate_exit_code":%s}\n' "$GATE_RC" > "$CURRENT_DIR/gate_exit.json"
CUDA_VISIBLE_DEVICES="" "$PY" "$L3/tools/compare_selectors.py" \
  --config "$W2_CONFIG" --policy-rows "$CURRENT_DIR/val520_policy.jsonl" \
  --out-dir "$CURRENT_DIR/geometry" 2>&1 | tee "$CURRENT_DIR/geometry.log"
finish_phase

# ---- verdetto dichiarato ---------------------------------------------------
start_phase verdict
"$PY" "$L3/tools/w2_verdict.py" \
  --vs-w1 "$RUN_ROOT/reading/vs_w1.json" \
  --vs-z1bis "$RUN_ROOT/reading/vs_z1bis.json" \
  --policy-json "$RUN_ROOT/operational/val520_policy.json" \
  --policy-rows "$RUN_ROOT/operational/val520_policy.jsonl" \
  --geometry "$RUN_ROOT/operational/geometry/summary.json" \
  --gate "$RUN_ROOT/operational/gate.json" \
  --out "$CURRENT_DIR/VERDICT.json" 2>&1 | tee "$CURRENT_DIR/verdict.log"
cp "$CURRENT_DIR/VERDICT.json" "$RUN_ROOT/VERDICT.json"
finish_phase

printf 'L3_W2_OK run_root=%s elapsed_s=%s\n' "$RUN_ROOT" \
  "$(( $(date +%s) - START_EPOCH ))"
