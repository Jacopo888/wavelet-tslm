#!/usr/bin/env bash
# W3 (replay percettivo su tutte le categorie, da vw2) e, in coda, il test 800:
# UNA lettura per braccio, disegno congelato prima del run. Criteri in PROTOCOL-L3.md.
set -Eeuo pipefail

ROOT="${L3_ROOT:-$HOME/timerlm/sft/tslm}"
PY="${L3_PYTHON:-$HOME/venv/bin/python}"
L3="$ROOT/l3"
W3_CONFIG="$L3/configs/l3_zoom_w3.yaml"
W2_CONFIG="$L3/configs/l3_zoom_w2.yaml"
Z1BIS_CONFIG="$L3/configs/l3_zoom_z1bis.yaml"
W2_RUN="${L3_W2_RUN:-$HOME/l3/w2_20260913T133901Z}"
W2_CHECKPOINT="${L3_W2_CHECKPOINT:-$W2_RUN/sft/train/vz1bis_epoch_1_end.pt}"
W2_SHA="${L3_W2_SHA:-9f48ca99490b1dd8b8fb9b8a075495e2be38dfb7ea2da7df51dfe6a55e629964}"
W2_POLICY_ROWS="${L3_W2_POLICY_ROWS:-$W2_RUN/operational/val520_policy.jsonl}"
Z1BIS_CHECKPOINT="${L3_Z1BIS_CHECKPOINT:-$HOME/l3/z1bis_20260912/sft/train/vz1bis_epoch_1_end.pt}"
Z1BIS_SHA="${L3_Z1BIS_SHA:-924110a5a344dc802c8eee646320f0c7035572c69e1125c714a3db737e8fd23f}"
REFERENCE_GATE_ROWS="${L3_REFERENCE_GATE_ROWS:-$HOME/l3/zoom_20260911c/Z1/gate/rows.jsonl}"
VA1_VAL_JSONL="${L3_VA1_VAL_JSONL:-$HOME/l3/zoom_20260911c/Z0/val520_no_zoom.jsonl}"
VA1_TEST_JSONL="${L3_VA1_TEST_JSONL:-$ROOT/results/anomaly_xl_precise/anomaly_xl_wavelet_va1/output_logs/evalonly_test_epoch_5_end_epoch_0.jsonl}"
VA1_TEST_SHA_PREFIX="${L3_VA1_TEST_SHA_PREFIX:-67212566a7396ea8}"
Z1BIS_READ_0="${L3_Z1BIS_READ_0:-$HOME/l3/z1bis_20260912/perceptual_gate/gate/rows.jsonl}"
Z1BIS_READ_1="${L3_Z1BIS_READ_1:-$HOME/l3/sep_z1bis_off1/rows.jsonl}"
Z1BIS_READ_2="${L3_Z1BIS_READ_2:-$HOME/l3/sep_z1bis_off2/rows.jsonl}"
W2_READ_0="${L3_W2_READ_0:-$W2_RUN/reading/off0/rows.jsonl}"
W2_READ_1="${L3_W2_READ_1:-$W2_RUN/reading/off1/rows.jsonl}"
W2_READ_2="${L3_W2_READ_2:-$W2_RUN/reading/off2/rows.jsonl}"
GPU="${L3_GPU:-0}"
RUN_ROOT="${L3_W3_RUN_ROOT:-$HOME/l3/w3_$(date -u +%Y%m%dT%H%M%SZ)}"
TRACE_TIMEOUT_SECONDS="${L3_W3_TRACE_TIMEOUT_SECONDS:-3600}"
TRACE_PEAK_LIMIT_MIB="${L3_W3_TRACE_PEAK_LIMIT_MIB:-8192}"
W2_VAL_PRIMARY="${L3_W2_VAL_PRIMARY:-0.3747}"
MIN_PRIMARY="${L3_W3_MIN_PRIMARY:-0.3647}"
MIN_CONTAINED="${L3_W3_MIN_CONTAINED:-0.609}"
TEST_ARMS="${L3_TEST_ARMS:-vz1bis vw2 vw3}"

[ "${L3_W3_DIFF_REVIEWED:-}" = "CLAUDE_APPROVED" ] || {
  echo "refusing to start: set L3_W3_DIFF_REVIEWED=CLAUDE_APPROVED after diff review" >&2
  exit 1
}

if [ "${1:-}" != "--worker" ] && [ -z "${TMUX:-}" ]; then
  command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
  [ ! -e "$RUN_ROOT" ] || { echo "refusing to reuse run root: $RUN_ROOT" >&2; exit 1; }
  mkdir -p "$RUN_ROOT"
  SESSION="l3-w3-$(date -u +%Y%m%dT%H%M%SZ)"
  tmux new-session -d -s "$SESSION" \
    "env L3_W3_DIFF_REVIEWED=CLAUDE_APPROVED L3_W3_RUN_ROOT='$RUN_ROOT' L3_GPU='$GPU' L3_ROOT='$ROOT' L3_PYTHON='$PY' L3_W2_RUN='$W2_RUN' L3_TEST_ARMS='$TEST_ARMS' bash '$L3/node_l3_w3.sh' --worker >'$RUN_ROOT/worker.log' 2>&1"
  echo "L3_W3_DETACHED session=$SESSION run_root=$RUN_ROOT log=$RUN_ROOT/worker.log"
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
    printf '{"status":"FAILED","phase":"%s","exit_code":%s,"line":%s}\n' \
      "$CURRENT_PHASE" "$rc" "$line" >"$CURRENT_DIR/FAILED"
  fi
  echo "L3_W3_FAILED phase=$CURRENT_PHASE line=$line rc=$rc" >&2
  exit "$rc"
}
trap 'failure $? $LINENO' ERR

start_phase() {
  CURRENT_PHASE="$1"
  CURRENT_DIR="$RUN_ROOT/$1"
  [ ! -e "$CURRENT_DIR" ] || { echo "refusing to reuse phase: $CURRENT_DIR" >&2; return 1; }
  mkdir -p "$CURRENT_DIR"
  printf 'L3_W3_STATUS phase=%s elapsed_s=%s\n' "$CURRENT_PHASE" \
    "$(( $(date +%s) - START_EPOCH ))" | tee -a "$CURRENT_DIR/phase.log"
}

finish_phase() {
  : >"$CURRENT_DIR/OK"
  printf 'L3_W3_PHASE_OK phase=%s elapsed_s=%s\n' "$CURRENT_PHASE" \
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
for required in "$W3_CONFIG" "$W2_CONFIG" "$Z1BIS_CONFIG" "$W2_CHECKPOINT" "$Z1BIS_CHECKPOINT" \
                "$W2_POLICY_ROWS" "$REFERENCE_GATE_ROWS" "$VA1_VAL_JSONL" "$VA1_TEST_JSONL" \
                "$Z1BIS_READ_0" "$Z1BIS_READ_1" "$Z1BIS_READ_2" \
                "$W2_READ_0" "$W2_READ_1" "$W2_READ_2"; do
  [ -e "$required" ] || { echo "missing prerequisite: $required" >&2; exit 1; }
done
[ "$(sha256sum "$W2_CHECKPOINT" | awk '{print $1}')" = "$W2_SHA" ] || {
  echo "W2 checkpoint SHA mismatch" >&2; exit 1; }
[ "$(sha256sum "$Z1BIS_CHECKPOINT" | awk '{print $1}')" = "$Z1BIS_SHA" ] || {
  echo "Z1-bis checkpoint SHA mismatch" >&2; exit 1; }
case "$(sha256sum "$VA1_TEST_JSONL" | awk '{print $1}')" in
  "$VA1_TEST_SHA_PREFIX"*) ;;
  *) echo "va1 test rows SHA mismatch" >&2; exit 1 ;;
esac
[ "$(wc -l < "$VA1_TEST_JSONL")" -eq 800 ] || { echo "va1 test rows != 800" >&2; exit 1; }
# Il test non deve essere mai stato letto con questi bracci.
for stale in "$HOME"/l3/*/test800 "$HOME"/l3/*/*test800*; do
  [ ! -e "$stale" ] || { echo "test800 output already exists: $stale" >&2; exit 1; }
done
git -C "$ROOT" rev-parse HEAD > "$RUN_ROOT/KIT_COMMIT.txt" 2>/dev/null || true
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest "$L3/tests/test_l3_w1.py" \
  "$L3/tests/test_l3_z1bis.py" "$L3/tests/test_l3_decision.py" -q \
  2>&1 | tee "$CURRENT_DIR/pytest.log"
finish_phase

# ---- tracce ----------------------------------------------------------------
start_phase traces
set +e
CUDA_VISIBLE_DEVICES="" timeout --signal=TERM "$TRACE_TIMEOUT_SECONDS" \
  "$PY" "$L3/zoom_traces_w1.py" --config "$W3_CONFIG" \
  --out-dir "$CURRENT_DIR/plan" --seed 20260915 2>&1 | tee "$CURRENT_DIR/traces.log"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" -eq 0 ] || { echo "W3 trace planning failed rc=$RC" >&2; exit "$RC"; }
"$PY" - "$CURRENT_DIR/plan/summary.json" "$TRACE_PEAK_LIMIT_MIB" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["status"] == "DONE", d["status"]
peak = d.get("planner_peak_rss_mib")
assert peak is None or peak <= float(sys.argv[2]), peak
block = d["w1"]
declared = block["perceptual_replay_declared"]
assert declared and sorted(declared["categories"]) == [
    "classify_with_evidence", "localize", "localize_all_channels"], declared
assert block["perceptual_replay_rows"] > 0
print(f"W3 perceptual_replay={block['perceptual_replay_rows']} righe | "
      f"wavelet_rate={block['wavelet_rate']:.4f}")
PYEOF
finish_phase

# ---- SFT da vw2 --------------------------------------------------------------
start_phase sft
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/zoom_sft.py" \
  --config "$W3_CONFIG" --checkpoint "$W2_CHECKPOINT" \
  --traces "$RUN_ROOT/traces/plan/traces.jsonl" --run-dir "$CURRENT_DIR/train" \
  2>&1 | tee "$CURRENT_DIR/sft.log"
cp "$CURRENT_DIR/train/summary.json" "$CURRENT_DIR/summary.json"
VW3=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['checkpoint'])" \
  "$CURRENT_DIR/summary.json")
echo "$VW3" > "$RUN_ROOT/CHECKPOINT.txt"
finish_phase

# ---- obiettivo 1: lettura sugli stessi crop ---------------------------------
start_phase reading
VW3=$(cat "$RUN_ROOT/CHECKPOINT.txt")
for offset in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/tools/diag_crop_content.py" \
    --config "$Z1BIS_CONFIG" --checkpoint "$VW3" \
    --reference-rows "$REFERENCE_GATE_ROWS" --out-dir "$CURRENT_DIR/off$offset" \
    --seed-offset "$offset" 2>&1 | tee -a "$CURRENT_DIR/reading.log"
done
"$PY" "$L3/tools/reader_vs_windows.py" \
  --pair "$W2_READ_0:$CURRENT_DIR/off0/rows.jsonl" \
  --pair "$W2_READ_1:$CURRENT_DIR/off1/rows.jsonl" \
  --pair "$W2_READ_2:$CURRENT_DIR/off2/rows.jsonl" \
  --out "$CURRENT_DIR/vs_w2.json" 2>&1 | tee -a "$CURRENT_DIR/reading.log"
"$PY" "$L3/tools/reader_vs_windows.py" \
  --pair "$Z1BIS_READ_0:$CURRENT_DIR/off0/rows.jsonl" \
  --pair "$Z1BIS_READ_1:$CURRENT_DIR/off1/rows.jsonl" \
  --pair "$Z1BIS_READ_2:$CURRENT_DIR/off2/rows.jsonl" \
  --out "$CURRENT_DIR/vs_z1bis.json" 2>&1 | tee -a "$CURRENT_DIR/reading.log"
finish_phase

# ---- obiettivo 2: policy autonoma su val520 ----------------------------------
start_phase operational
VW3=$(cat "$RUN_ROOT/CHECKPOINT.txt")
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/zoom.py" eval \
  --config "$W3_CONFIG" --checkpoint "$VW3" --split val --mode policy \
  --out-json "$CURRENT_DIR/val520_policy.json" \
  --out-jsonl "$CURRENT_DIR/val520_policy.jsonl" 2>&1 | tee "$CURRENT_DIR/eval.log"
set +e
CUDA_VISIBLE_DEVICES="" "$PY" "$L3/tools/gate_z1bis_operational.py" \
  --config "$W3_CONFIG" --candidate "$CURRENT_DIR/val520_policy.jsonl" \
  --va1-reference "$VA1_VAL_JSONL" --previous-vz1-reference "$W2_POLICY_ROWS" \
  --out "$CURRENT_DIR/gate.json" 2>&1 | tee "$CURRENT_DIR/gate.log"
GATE_RC=${PIPESTATUS[0]}
set -e
printf '{"gate_exit_code":%s}\n' "$GATE_RC" > "$CURRENT_DIR/gate_exit.json"
CUDA_VISIBLE_DEVICES="" "$PY" "$L3/tools/compare_selectors.py" \
  --config "$W3_CONFIG" --policy-rows "$CURRENT_DIR/val520_policy.jsonl" \
  --out-dir "$CURRENT_DIR/geometry" 2>&1 | tee "$CURRENT_DIR/geometry.log"
finish_phase

# ---- verdetto dichiarato ------------------------------------------------------
start_phase verdict
"$PY" "$L3/tools/w2_verdict.py" --mode w3 \
  --vs-w1 "$RUN_ROOT/reading/vs_w2.json" \
  --vs-z1bis "$RUN_ROOT/reading/vs_z1bis.json" \
  --policy-json "$RUN_ROOT/operational/val520_policy.json" \
  --policy-rows "$RUN_ROOT/operational/val520_policy.jsonl" \
  --geometry "$RUN_ROOT/operational/geometry/summary.json" \
  --gate "$RUN_ROOT/operational/gate.json" \
  --min-primary "$MIN_PRIMARY" --min-contained-rate "$MIN_CONTAINED" \
  --start-primary "$W2_VAL_PRIMARY" \
  --out "$CURRENT_DIR/VERDICT.json" 2>&1 | tee "$CURRENT_DIR/verdict.log"
cp "$CURRENT_DIR/VERDICT.json" "$RUN_ROOT/VERDICT.json"
HEADLINE=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['headline_arm'])" \
  "$RUN_ROOT/VERDICT.json")
echo "$HEADLINE" > "$RUN_ROOT/HEADLINE_ARM.txt"
finish_phase

# ---- test 800: una lettura per braccio, disegno congelato --------------------
start_phase test800
VW3=$(cat "$RUN_ROOT/CHECKPOINT.txt")
for arm in $TEST_ARMS; do
  case "$arm" in
    vz1bis) CKPT="$Z1BIS_CONFIG"; CK="$Z1BIS_CHECKPOINT" ;;
    vw2)    CKPT="$W2_CONFIG";    CK="$W2_CHECKPOINT" ;;
    vw3)    CKPT="$W3_CONFIG";    CK="$VW3" ;;
    *) echo "unknown arm: $arm" >&2; exit 1 ;;
  esac
  [ ! -e "$CURRENT_DIR/${arm}_test800.jsonl" ] || { echo "arm $arm already read" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/zoom.py" eval \
    --config "$CKPT" --checkpoint "$CK" --split test --mode policy \
    --out-json "$CURRENT_DIR/${arm}_test800.json" \
    --out-jsonl "$CURRENT_DIR/${arm}_test800.jsonl" 2>&1 | tee -a "$CURRENT_DIR/test.log"
  "$PY" "$L3/paired_eval_l3.py" --reference "$VA1_TEST_JSONL" \
    --candidate "$CURRENT_DIR/${arm}_test800.jsonl" --split test --seed 20260915 \
    --out-json "$CURRENT_DIR/${arm}_vs_va1.json" \
    --out-markdown "$CURRENT_DIR/${arm}_vs_va1.md" 2>&1 | tee -a "$CURRENT_DIR/test.log"
done
# Confronti appaiati fra i bracci zoom, tutti contro vz1bis e vw3 contro vw2.
for pair in "vw2:vz1bis" "vw3:vz1bis" "vw3:vw2"; do
  cand="${pair%%:*}"; ref="${pair##*:}"
  [ -f "$CURRENT_DIR/${cand}_test800.jsonl" ] && [ -f "$CURRENT_DIR/${ref}_test800.jsonl" ] || continue
  "$PY" "$L3/paired_eval_l3.py" --reference "$CURRENT_DIR/${ref}_test800.jsonl" \
    --candidate "$CURRENT_DIR/${cand}_test800.jsonl" --split test --seed 20260915 \
    --out-json "$CURRENT_DIR/${cand}_vs_${ref}.json" \
    --out-markdown "$CURRENT_DIR/${cand}_vs_${ref}.md" 2>&1 | tee -a "$CURRENT_DIR/test.log"
done
"$PY" - "$CURRENT_DIR" "$TEST_ARMS" "$RUN_ROOT/HEADLINE_ARM.txt" <<'PYEOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); arms = sys.argv[2].split()
headline = Path(sys.argv[3]).read_text(encoding="utf-8").strip()
out = {"status": "DONE", "headline_arm": "vw3" if headline == "new" else "vw2", "arms": {}}
for arm in arms:
    d = json.load(open(root / f"{arm}_test800.json", encoding="utf-8"))
    assert d["samples_evaluated"] == 800, (arm, d["samples_evaluated"])
    out["arms"][arm] = {"primary": d["primary"], "zoom_rate": d.get("zoom_rate"),
                        "zoom_invalid": d.get("zoom_invalid"),
                        "per_category": {k: v.get("primary") for k, v in d["per_category"].items()}}
json.dump(out, open(root / "summary.json", "w", encoding="utf-8"), indent=2)
open(root / "summary.json", "a").write("\n")
for arm, block in out["arms"].items():
    print(f"TEST800 {arm}: primary={block['primary']:.4f} zoom_rate={block['zoom_rate']}")
print(f"TEST800 headline={out['headline_arm']}")
PYEOF
finish_phase

printf 'L3_W3_OK run_root=%s headline=%s elapsed_s=%s\n' "$RUN_ROOT" \
  "$(cat "$RUN_ROOT/HEADLINE_ARM.txt")" "$(( $(date +%s) - START_EPOCH ))"
