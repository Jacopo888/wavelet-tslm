#!/usr/bin/env bash
# Catena notturna condizionale: punteggio dei selettori -> decisione dichiarata ->
# un solo ramo fra GRPO, imitazione del selettore e solo-report.
# Nessun accesso al test 800, nessuna modifica dell'architettura.
set -Eeuo pipefail

ROOT="${L3_ROOT:-$HOME/timerlm/sft/tslm}"
PY="${L3_PYTHON:-$HOME/venv/bin/python}"
L3="$ROOT/l3"
Z1BIS_CONFIG="$L3/configs/l3_zoom_z1bis.yaml"
W1_CONFIG="$L3/configs/l3_zoom_w1.yaml"
WGRPO_CONFIG="$L3/configs/l3_zoom_wgrpo.yaml"
Z1BIS_ROOT="${L3_Z1BIS_ROOT:-$HOME/l3/z1bis_20260912}"
CHECKPOINT="${L3_Z1BIS_CHECKPOINT:-$Z1BIS_ROOT/sft/train/vz1bis_epoch_1_end.pt}"
CHECKPOINT_SHA="${L3_Z1BIS_CHECKPOINT_SHA:-924110a5a344dc802c8eee646320f0c7035572c69e1125c714a3db737e8fd23f}"
VA1_CHECKPOINT="${L3_VA1_CHECKPOINT:-$ROOT/results/anomaly_xl_precise/anomaly_xl_wavelet_va1/checkpoints/epoch_5_end.pt}"
VA1_SHA="${L3_VA1_SHA:-cd7238ae81da1afca9f225a99e9af4261be829798a39985a427d88a098bf2531}"
POLICY_ROWS="${L3_POLICY_ROWS:-$Z1BIS_ROOT/operational_gate/val520_policy.jsonl}"
CONTENT_DIR="${L3_CONTENT_DIR:-$HOME/l3/selectors_20260912T231746Z/selfwindow/content}"
PREVIOUS_ROOT="${L3_PREVIOUS_ZOOM_ROOT:-$HOME/l3/zoom_20260911c}"
VA1_VAL_JSONL="${L3_VA1_VAL_JSONL:-$PREVIOUS_ROOT/Z0/val520_no_zoom.jsonl}"
GPU="${L3_GPU:-0}"
RUN_ROOT="${L3_NIGHT_RUN_ROOT:-$HOME/l3/night_$(date -u +%Y%m%dT%H%M%SZ)}"
DEADLINE_HOURS="${L3_NIGHT_DEADLINE_HOURS:-9}"
CONTENT_WAIT_MINUTES="${L3_NIGHT_CONTENT_WAIT_MINUTES:-75}"
TRACE_TIMEOUT_SECONDS="${L3_NIGHT_TRACE_TIMEOUT_SECONDS:-3600}"
TRACE_PEAK_LIMIT_MIB="${L3_NIGHT_TRACE_PEAK_LIMIT_MIB:-8192}"
MIN_CONTENT_DELTA="${L3_NIGHT_MIN_CONTENT_DELTA:-0.03}"
MIN_WINDOW_DELTA="${L3_NIGHT_MIN_WINDOW_DELTA:-0.05}"
MIN_SELECTOR_DELTA="${L3_NIGHT_MIN_SELECTOR_DELTA:-0.03}"

[ "${L3_NIGHT_DIFF_REVIEWED:-}" = "CLAUDE_APPROVED" ] || {
  echo "refusing to start: set L3_NIGHT_DIFF_REVIEWED=CLAUDE_APPROVED after diff review" >&2
  exit 1
}

if [ "${1:-}" != "--worker" ] && [ -z "${TMUX:-}" ]; then
  command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
  [ ! -e "$RUN_ROOT" ] || { echo "refusing to reuse run root: $RUN_ROOT" >&2; exit 1; }
  mkdir -p "$RUN_ROOT"
  SESSION="l3-night-$(date -u +%Y%m%dT%H%M%SZ)"
  tmux new-session -d -s "$SESSION" \
    "env L3_NIGHT_DIFF_REVIEWED=CLAUDE_APPROVED L3_NIGHT_RUN_ROOT='$RUN_ROOT' L3_GPU='$GPU' L3_ROOT='$ROOT' L3_PYTHON='$PY' L3_Z1BIS_ROOT='$Z1BIS_ROOT' L3_Z1BIS_CHECKPOINT='$CHECKPOINT' L3_VA1_CHECKPOINT='$VA1_CHECKPOINT' L3_POLICY_ROWS='$POLICY_ROWS' L3_CONTENT_DIR='$CONTENT_DIR' L3_PREVIOUS_ZOOM_ROOT='$PREVIOUS_ROOT' L3_NIGHT_DEADLINE_HOURS='$DEADLINE_HOURS' L3_NIGHT_CONTENT_WAIT_MINUTES='$CONTENT_WAIT_MINUTES' bash '$L3/node_l3_night.sh' --worker >'$RUN_ROOT/worker.log' 2>&1"
  echo "L3_NIGHT_DETACHED session=$SESSION run_root=$RUN_ROOT log=$RUN_ROOT/worker.log"
  exit 0
fi

START_EPOCH=$(date +%s)
DEADLINE_EPOCH=$(( START_EPOCH + $(printf '%.0f' "$(echo "$DEADLINE_HOURS * 3600" | bc -l)") ))
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
  echo "L3_NIGHT_FAILED phase=$CURRENT_PHASE line=$line rc=$rc" >&2
  exit "$rc"
}
trap 'failure $? $LINENO' ERR

start_phase() {
  CURRENT_PHASE="$1"
  CURRENT_DIR="$RUN_ROOT/$1"
  [ ! -e "$CURRENT_DIR" ] || { echo "refusing to reuse phase directory: $CURRENT_DIR" >&2; return 1; }
  mkdir -p "$CURRENT_DIR"
  printf 'L3_NIGHT_STATUS phase=%s elapsed_s=%s\n' \
    "$CURRENT_PHASE" "$(( $(date +%s) - START_EPOCH ))" | tee -a "$CURRENT_DIR/phase.log"
}

finish_phase() {
  : >"$CURRENT_DIR/OK"
  printf 'L3_NIGHT_PHASE_OK phase=%s elapsed_s=%s\n' \
    "$CURRENT_PHASE" "$(( $(date +%s) - START_EPOCH ))" | tee -a "$CURRENT_DIR/phase.log"
}

budget_left_hours() {
  echo "scale=2; ($DEADLINE_EPOCH - $(date +%s)) / 3600" | bc -l
}

require_budget() {
  local needed="$1" label="$2"
  local left
  left=$(budget_left_hours)
  if [ "$(echo "$left < $needed" | bc -l)" = "1" ]; then
    echo "L3_NIGHT_BUDGET_STOP label=$label left_h=$left needed_h=$needed" \
      | tee -a "$RUN_ROOT/BUDGET_STOP"
    return 1
  fi
  printf 'L3_NIGHT_BUDGET label=%s left_h=%s needed_h=%s\n' "$label" "$left" "$needed"
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
for required in "$Z1BIS_CONFIG" "$W1_CONFIG" "$WGRPO_CONFIG" "$CHECKPOINT" \
                "$VA1_CHECKPOINT" "$POLICY_ROWS" "$VA1_VAL_JSONL" \
                "$L3/tools/score_selectors.py" "$L3/tools/decide_branch.py" \
                "$L3/zoom_traces_w1.py" "$L3/zoom_sft.py" "$L3/grpo.py"; do
  [ -e "$required" ] || { echo "missing prerequisite: $required" >&2; exit 1; }
done
[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$CHECKPOINT_SHA" ] || {
  echo "vz1bis checkpoint SHA mismatch" >&2; exit 1; }
[ "$(sha256sum "$VA1_CHECKPOINT" | awk '{print $1}')" = "$VA1_SHA" ] || {
  echo "va1 checkpoint SHA mismatch" >&2; exit 1; }
git -C "$ROOT" rev-parse HEAD > "$RUN_ROOT/KIT_COMMIT.txt" 2>/dev/null || echo unknown > "$RUN_ROOT/KIT_COMMIT.txt"
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest "$L3/tests/test_l3_selectors.py" \
  "$L3/tests/test_l3_w1.py" -q 2>&1 | tee "$CURRENT_DIR/pytest.log"
finish_phase

# ---- attesa del gate del contenuto (fase A, in corso su GPU) ---------------
start_phase wait_content
DEADLINE_WAIT=$(( $(date +%s) + CONTENT_WAIT_MINUTES * 60 ))
while [ ! -e "$CONTENT_DIR/summary.json" ]; do
  if [ "$(date +%s)" -ge "$DEADLINE_WAIT" ]; then
    echo "content gate did not finish within $CONTENT_WAIT_MINUTES minutes" >&2
    exit 1
  fi
  sleep 60
done
# La GPU deve essere libera: la fase A scrive il summary prima di uscire.
while pgrep -f "[g]ate_selfwindow_content.py" >/dev/null; do sleep 20; done
cp "$CONTENT_DIR/summary.json" "$CURRENT_DIR/content_summary.json"
finish_phase

# ---- punteggio dei selettori (GPU) ----------------------------------------
require_budget 1.5 scores || { echo "L3_NIGHT_OK_NO_BUDGET"; exit 0; }
start_phase scores
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/tools/score_selectors.py" \
  --config "$Z1BIS_CONFIG" --checkpoint "$CHECKPOINT" --policy-rows "$POLICY_ROWS" \
  --baseline-rows "$CONTENT_DIR/rows.jsonl" --out-dir "$CURRENT_DIR/scores" \
  2>&1 | tee "$CURRENT_DIR/scores.log"
finish_phase

# ---- decisione -------------------------------------------------------------
start_phase decision
"$PY" "$L3/tools/decide_branch.py" \
  --content-summary "$RUN_ROOT/wait_content/content_summary.json" \
  --scores-summary "$RUN_ROOT/scores/scores/summary.json" \
  --min-content-delta "$MIN_CONTENT_DELTA" \
  --min-window-delta "$MIN_WINDOW_DELTA" \
  --min-selector-delta "$MIN_SELECTOR_DELTA" \
  --out "$CURRENT_DIR/DECISION.json" 2>&1 | tee "$CURRENT_DIR/decision.log"
BRANCH=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['branch'])" \
  "$CURRENT_DIR/DECISION.json")
echo "$BRANCH" > "$RUN_ROOT/BRANCH.txt"
cp "$CURRENT_DIR/DECISION.json" "$RUN_ROOT/DECISION.json"
finish_phase

# ---- ramo ------------------------------------------------------------------
case "$BRANCH" in
  report_only)
    start_phase experiment_report_only
    printf '{"status":"DONE","branch":"report_only"}\n' > "$CURRENT_DIR/summary.json"
    finish_phase
    ;;

  selector_sft)
    require_budget 8.0 selector_sft || { echo "L3_NIGHT_OK_NO_BUDGET"; exit 0; }
    start_phase traces_w1
    set +e
    CUDA_VISIBLE_DEVICES="" timeout --signal=TERM "$TRACE_TIMEOUT_SECONDS" \
      "$PY" "$L3/zoom_traces_w1.py" --config "$W1_CONFIG" \
      --out-dir "$CURRENT_DIR/plan" --seed 20260913 2>&1 | tee "$CURRENT_DIR/traces.log"
    RC=${PIPESTATUS[0]}
    set -e
    [ "$RC" -eq 0 ] || { echo "W1 trace planning failed rc=$RC" >&2; exit "$RC"; }
    "$PY" - "$CURRENT_DIR/plan/summary.json" "$TRACE_PEAK_LIMIT_MIB" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["status"] == "DONE", d["status"]
peak = d.get("planner_peak_rss_mib")
assert peak is None or peak <= float(sys.argv[2]), peak
print(f"W1 wavelet_rate={d['w1']['wavelet_rate']:.4f} "
      f"gold_fallback={d['w1']['gold_fallback_rate']:.4f}")
PYEOF
    finish_phase

    require_budget 7.6 w1_sft || { echo "L3_NIGHT_OK_NO_BUDGET"; exit 0; }
    start_phase w1_sft
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/zoom_sft.py" \
      --config "$W1_CONFIG" --checkpoint "$VA1_CHECKPOINT" \
      --traces "$RUN_ROOT/traces_w1/plan/traces.jsonl" --run-dir "$CURRENT_DIR/train" \
      2>&1 | tee "$CURRENT_DIR/sft.log"
    cp "$CURRENT_DIR/train/summary.json" "$CURRENT_DIR/summary.json"
    finish_phase

    start_phase w1_operational
    VW1=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['checkpoint'])" \
      "$RUN_ROOT/w1_sft/summary.json")
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/zoom.py" eval \
      --config "$W1_CONFIG" --checkpoint "$VW1" --split val --mode policy \
      --out-json "$CURRENT_DIR/val520_policy.json" \
      --out-jsonl "$CURRENT_DIR/val520_policy.jsonl" 2>&1 | tee "$CURRENT_DIR/eval.log"
    set +e
    CUDA_VISIBLE_DEVICES="" "$PY" "$L3/tools/gate_z1bis_operational.py" \
      --config "$W1_CONFIG" --candidate "$CURRENT_DIR/val520_policy.jsonl" \
      --va1-reference "$VA1_VAL_JSONL" --previous-vz1-reference "$POLICY_ROWS" \
      --out "$CURRENT_DIR/gate.json" 2>&1 | tee "$CURRENT_DIR/gate.log"
    GATE_RC=${PIPESTATUS[0]}
    set -e
    # Un gate non superato e' un risultato, non un errore della catena: si registra.
    printf '{"gate_exit_code":%s}\n' "$GATE_RC" > "$CURRENT_DIR/gate_exit.json"
    finish_phase

    start_phase w1_geometry
    # La geometria delle finestre della nuova policy e' la misura decisiva del ramo:
    # il contenimento va confrontato con il 55,3 % di vz1bis.
    CUDA_VISIBLE_DEVICES="" "$PY" "$L3/tools/compare_selectors.py" \
      --config "$W1_CONFIG" --policy-rows "$RUN_ROOT/w1_operational/val520_policy.jsonl" \
      --out-dir "$CURRENT_DIR/geometry" 2>&1 | tee "$CURRENT_DIR/geometry.log"
    finish_phase
    ;;

  grpo)
    require_budget 6.5 grpo || { echo "L3_NIGHT_OK_NO_BUDGET"; exit 0; }
    start_phase wgrpo
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/grpo.py" init-check \
      --config "$WGRPO_CONFIG" --checkpoint "$CHECKPOINT" \
      --out "$CURRENT_DIR/init_identity.json" 2>&1 | tee "$CURRENT_DIR/init.log"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/grpo.py" train \
      --config "$WGRPO_CONFIG" --checkpoint "$CHECKPOINT" --mode zoom-memory \
      --run-dir "$CURRENT_DIR/memory" 2>&1 | tee "$CURRENT_DIR/memory.log"
    # Step 0 fra i candidati, col suo primary su val520 gia' misurato.
    "$PY" - "$CURRENT_DIR/val_history.json" "$POLICY_ROWS" "$CHECKPOINT" "$CHECKPOINT_SHA" <<'PYEOF'
import json, statistics, sys
rows = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
primary = statistics.fmean(float(row["primary"]) for row in rows)
history = [{"step": 0, "primary": primary, "stale": 0, "checkpoint": sys.argv[3],
            "checkpoint_sha256": sys.argv[4], "source": "z1bis operational gate"}]
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(history, handle, indent=2)
    handle.write("\n")
print(f"step0 primary={primary:.4f} rows={len(rows)}")
PYEOF
    TRAIN_DIR="$CURRENT_DIR/train"
    for step in $(seq 25 25 150); do
      require_budget 1.2 "grpo_step_$step" || break
      if [ "$step" -eq 25 ]; then
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/grpo.py" train \
          --config "$WGRPO_CONFIG" --checkpoint "$CHECKPOINT" --mode zoom \
          --target-step "$step" --run-dir "$TRAIN_DIR" 2>&1 | tee -a "$CURRENT_DIR/train.log"
      else
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/grpo.py" train \
          --config "$WGRPO_CONFIG" --checkpoint "$CHECKPOINT" --mode zoom \
          --target-step "$step" --resume "$TRAIN_DIR" 2>&1 | tee -a "$CURRENT_DIR/train.log"
      fi
      if [ $((step % 50)) -eq 0 ]; then
        CKPT="$TRAIN_DIR/checkpoints/step_$(printf '%06d' "$step")_composed.pt"
        CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/zoom.py" eval \
          --config "$WGRPO_CONFIG" --checkpoint "$CKPT" --split val --mode policy \
          --out-json "$CURRENT_DIR/val520_step_${step}.json" \
          --out-jsonl "$CURRENT_DIR/val520_step_${step}.jsonl" \
          2>&1 | tee -a "$CURRENT_DIR/eval.log"
        "$PY" - "$CURRENT_DIR" "$step" "$CKPT" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
root, step, ckpt = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
history_path = root / "val_history.json"
history = json.loads(history_path.read_text(encoding="utf-8"))
scored = json.loads((root / f"val520_step_{step}.json").read_text(encoding="utf-8"))
prior = max(entry["primary"] for entry in history)
stale = 0 if scored["primary"] >= prior + 0.005 else history[-1]["stale"] + 1
history.append({"step": step, "primary": scored["primary"], "stale": stale,
                "checkpoint": str(ckpt),
                "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()})
best = max(entry["primary"] for entry in history)
chosen = next(entry for entry in history if entry["primary"] >= best - 0.002)
selection = {"status": "FROZEN" if stale >= 2 or step == 150 else "PROVISIONAL",
             "history": history, "selected_step": chosen["step"],
             "selected_primary": chosen["primary"],
             "checkpoint": chosen["checkpoint"],
             "checkpoint_sha256": chosen["checkpoint_sha256"], "stale": stale,
             "step0_kept_as_candidate": True}
history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
(root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n",
                                     encoding="utf-8")
print(f"step={step} primary={scored['primary']:.4f} selected={chosen['step']} "
      f"stale={stale}")
PYEOF
        STALE=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['stale'])" \
          "$CURRENT_DIR/selection.json")
        [ "$STALE" -lt 2 ] || break
      fi
    done
    if [ -f "$CURRENT_DIR/selection.json" ]; then
      cp "$CURRENT_DIR/selection.json" "$CURRENT_DIR/summary.json"
    else
      printf '{"status":"STOPPED","reason":"budget before the first validation"}\n' \
        > "$CURRENT_DIR/summary.json"
    fi
    finish_phase
    ;;

  *)
    echo "unknown branch: $BRANCH" >&2
    exit 1
    ;;
esac

printf 'L3_NIGHT_OK branch=%s run_root=%s elapsed_s=%s\n' \
  "$BRANCH" "$RUN_ROOT" "$(( $(date +%s) - START_EPOCH ))"
