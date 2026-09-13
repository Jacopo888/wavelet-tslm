#!/usr/bin/env bash
# Pilot GRPO sulla scelta della finestra. Attende il verdetto di W2, sceglie il
# checkpoint di partenza secondo la tabella dichiarata, poi: init-check, gate di
# memoria, chunk da 25 fino a 150 con val520 ogni 50, selezione coi vincoli
# dichiarati (step 0 incluso), verifica finale della lettura sul solo checkpoint
# scelto. Criteri in PROTOCOL-L3.md. Nessun accesso al test 800.
set -Eeuo pipefail

ROOT="${L3_ROOT:-$HOME/timerlm/sft/tslm}"
PY="${L3_PYTHON:-$HOME/venv/bin/python}"
L3="$ROOT/l3"
TEMPLATE="$L3/configs/l3_zoom_w2grpo.yaml"
Z1BIS_CONFIG="$L3/configs/l3_zoom_z1bis.yaml"
W2_RUN="${L3_W2_RUN:-$HOME/l3/w2_20260913T133901Z}"
W1_CHECKPOINT="${L3_W1_CHECKPOINT:-$HOME/l3/night_20260913T005001Z/w1_sft/train/vz1bis_epoch_1_end.pt}"
W1_POLICY_ROWS="${L3_W1_POLICY_ROWS:-$HOME/l3/night_20260913T005001Z/w1_operational/val520_policy.jsonl}"
REFERENCE_GATE_ROWS="${L3_REFERENCE_GATE_ROWS:-$HOME/l3/zoom_20260911c/Z1/gate/rows.jsonl}"
VA1_VAL_JSONL="${L3_VA1_VAL_JSONL:-$HOME/l3/zoom_20260911c/Z0/val520_no_zoom.jsonl}"
Z1BIS_READ_0="${L3_Z1BIS_READ_0:-$HOME/l3/z1bis_20260912/perceptual_gate/gate/rows.jsonl}"
Z1BIS_READ_1="${L3_Z1BIS_READ_1:-$HOME/l3/sep_z1bis_off1/rows.jsonl}"
Z1BIS_READ_2="${L3_Z1BIS_READ_2:-$HOME/l3/sep_z1bis_off2/rows.jsonl}"
W1_READ_0="${L3_W1_READ_0:-$HOME/l3/w1_perceptual/rows.jsonl}"
W1_READ_1="${L3_W1_READ_1:-$HOME/l3/sep_w1_off1/rows.jsonl}"
W1_READ_2="${L3_W1_READ_2:-$HOME/l3/sep_w1_off2/rows.jsonl}"
GPU="${L3_GPU:-0}"
RUN_ROOT="${L3_GRPO_RUN_ROOT:-$HOME/l3/grpo_$(date -u +%Y%m%dT%H%M%SZ)}"
VERDICT_WAIT_MINUTES="${L3_GRPO_VERDICT_WAIT_MINUTES:-360}"
DEADLINE_HOURS="${L3_GRPO_DEADLINE_HOURS:-14}"
MAX_STEP="${L3_GRPO_MAX_STEP:-150}"
MULTICHANNEL_TOLERANCE="${L3_GRPO_MULTICHANNEL_TOLERANCE:-0.02}"

[ "${L3_GRPO_DIFF_REVIEWED:-}" = "CLAUDE_APPROVED" ] || {
  echo "refusing to start: set L3_GRPO_DIFF_REVIEWED=CLAUDE_APPROVED after diff review" >&2
  exit 1
}

if [ "${1:-}" != "--worker" ] && [ -z "${TMUX:-}" ]; then
  command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
  [ ! -e "$RUN_ROOT" ] || { echo "refusing to reuse run root: $RUN_ROOT" >&2; exit 1; }
  mkdir -p "$RUN_ROOT"
  SESSION="l3-grpo-$(date -u +%Y%m%dT%H%M%SZ)"
  tmux new-session -d -s "$SESSION" \
    "env L3_GRPO_DIFF_REVIEWED=CLAUDE_APPROVED L3_GRPO_RUN_ROOT='$RUN_ROOT' L3_GPU='$GPU' L3_ROOT='$ROOT' L3_PYTHON='$PY' L3_W2_RUN='$W2_RUN' L3_W1_CHECKPOINT='$W1_CHECKPOINT' L3_W1_POLICY_ROWS='$W1_POLICY_ROWS' L3_GRPO_VERDICT_WAIT_MINUTES='$VERDICT_WAIT_MINUTES' L3_GRPO_DEADLINE_HOURS='$DEADLINE_HOURS' L3_GRPO_MAX_STEP='$MAX_STEP' bash '$L3/node_l3_grpo.sh' --worker >'$RUN_ROOT/worker.log' 2>&1"
  echo "L3_GRPO_DETACHED session=$SESSION run_root=$RUN_ROOT log=$RUN_ROOT/worker.log"
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
  echo "L3_GRPO_FAILED phase=$CURRENT_PHASE line=$line rc=$rc" >&2
  exit "$rc"
}
trap 'failure $? $LINENO' ERR

start_phase() {
  CURRENT_PHASE="$1"
  CURRENT_DIR="$RUN_ROOT/$1"
  [ ! -e "$CURRENT_DIR" ] || { echo "refusing to reuse phase: $CURRENT_DIR" >&2; return 1; }
  mkdir -p "$CURRENT_DIR"
  printf 'L3_GRPO_STATUS phase=%s elapsed_s=%s\n' "$CURRENT_PHASE" \
    "$(( $(date +%s) - START_EPOCH ))" | tee -a "$CURRENT_DIR/phase.log"
}

finish_phase() {
  : >"$CURRENT_DIR/OK"
  printf 'L3_GRPO_PHASE_OK phase=%s elapsed_s=%s\n' "$CURRENT_PHASE" \
    "$(( $(date +%s) - START_EPOCH ))" | tee -a "$CURRENT_DIR/phase.log"
}

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
unset PYTORCH_CUDA_ALLOC_CONF WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
export TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16
export TSLM_ATTN_IMPL=sdpa TSLM_SAVE_TRAINABLE_ONLY=1 TSLM_GRAD_CKPT=1
export TSLM_QWEN_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# ---- attesa del verdetto di W2 e scelta della partenza ---------------------
start_phase start_point
DEADLINE_WAIT=$(( $(date +%s) + VERDICT_WAIT_MINUTES * 60 ))
while [ ! -f "$W2_RUN/VERDICT.json" ]; do
  if ls "$W2_RUN"/*/FAILED >/dev/null 2>&1; then
    echo "la catena W2 e' fallita: nessun pilot parte" >&2
    exit 1
  fi
  if [ "$(date +%s)" -ge "$DEADLINE_WAIT" ]; then
    echo "verdetto W2 non arrivato entro $VERDICT_WAIT_MINUTES minuti" >&2
    exit 1
  fi
  sleep 120
done
while pgrep -f "[d]iag_crop_content.py|[z]oom.py eval|[c]ompare_selectors.py" >/dev/null; do
  sleep 30
done
VERDICT=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['verdict'])" \
  "$W2_RUN/VERDICT.json")
case "$VERDICT" in
  w2_becomes_candidate)
    START_CKPT=$(cat "$W2_RUN/CHECKPOINT.txt")
    START_LABEL=vw2
    START_POLICY_ROWS="$W2_RUN/operational/val520_policy.jsonl"
    START_READ_0="$W2_RUN/reading/off0/rows.jsonl"
    START_READ_1="$W2_RUN/reading/off1/rows.jsonl"
    START_READ_2="$W2_RUN/reading/off2/rows.jsonl"
    ;;
  keep_w1_reading_only|keep_w1_grpo_with_constraint)
    START_CKPT="$W1_CHECKPOINT"
    START_LABEL=vw1
    START_POLICY_ROWS="$W1_POLICY_ROWS"
    START_READ_0="$W1_READ_0"
    START_READ_1="$W1_READ_1"
    START_READ_2="$W1_READ_2"
    ;;
  *)
    echo "verdetto non riconosciuto: $VERDICT" >&2; exit 1 ;;
esac
[ -f "$START_CKPT" ] || { echo "checkpoint di partenza assente: $START_CKPT" >&2; exit 1; }
START_SHA=$(sha256sum "$START_CKPT" | awk '{print $1}')
CONFIG="$RUN_ROOT/config_grpo.yaml"
sed "s|^  initial_checkpoint_sha256: .*|  initial_checkpoint_sha256: $START_SHA|" \
  "$TEMPLATE" > "$CONFIG"
grep -q "initial_checkpoint_sha256: $START_SHA" "$CONFIG"
{
  echo "verdict=$VERDICT"
  echo "start_label=$START_LABEL"
  echo "start_checkpoint=$START_CKPT"
  echo "start_sha256=$START_SHA"
  echo "config=$CONFIG"
} | tee "$CURRENT_DIR/start_point.txt"
finish_phase

# ---- guard: identita' iniziale e memoria -----------------------------------
start_phase guards
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/grpo.py" init-check \
  --config "$CONFIG" --checkpoint "$START_CKPT" \
  --out "$CURRENT_DIR/init_identity.json" 2>&1 | tee "$CURRENT_DIR/init.log"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/grpo.py" train \
  --config "$CONFIG" --checkpoint "$START_CKPT" --mode zoom-memory \
  --run-dir "$CURRENT_DIR/memory" 2>&1 | tee "$CURRENT_DIR/memory.log"
"$PY" - "$CURRENT_DIR/memory/summary.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["steps"] == 1 and d["rollouts"] == 16, d
assert d["peak_mem_mib"] < 22528, d["peak_mem_mib"]
assert d["parsable"] + d["unparsable"] == 16, d
print(f"gate memoria: picco {d['peak_mem_mib']:.0f} MiB, parsabili {d['parsable']}/16")
PYEOF
finish_phase

# ---- step 0 fra i candidati -------------------------------------------------
start_phase training
"$PY" - "$CURRENT_DIR/val_history.json" "$START_POLICY_ROWS" "$START_CKPT" \
  "$START_SHA" <<'PYEOF'
import json, statistics, sys
from collections import defaultdict
rows = [json.loads(line) for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
by_category = defaultdict(list)
for row in rows:
    by_category[str(row["category"])].append(float(row["primary"]))
history = [{
    "step": 0,
    "primary": statistics.fmean(float(row["primary"]) for row in rows),
    "per_category": {k: statistics.fmean(v) for k, v in by_category.items()},
    "invalid_rate": sum(1 for r in rows if r.get("zoom_invalid")) / len(rows),
    "empty_rate": sum(1 for r in rows if r.get("empty_final")) / len(rows),
    "stale": 0, "selectable": True,
    "checkpoint": sys.argv[3], "checkpoint_sha256": sys.argv[4],
}]
json.dump(history, open(sys.argv[1], "w", encoding="utf-8"), indent=2)
open(sys.argv[1], "a").write("\n")
print(f"step0 primary={history[0]['primary']:.4f} "
      f"multicanale={history[0]['per_category'].get('localize_all_channels', 0):.4f}")
PYEOF
TRAIN_DIR="$CURRENT_DIR/train"
for step in $(seq 25 25 "$MAX_STEP"); do
  if [ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]; then
    echo "L3_GRPO_DEADLINE raggiunta prima dello step $step" | tee -a "$CURRENT_DIR/phase.log"
    break
  fi
  if [ "$step" -eq 25 ]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/grpo.py" train \
      --config "$CONFIG" --checkpoint "$START_CKPT" --mode zoom \
      --target-step "$step" --run-dir "$TRAIN_DIR" 2>&1 | tee -a "$CURRENT_DIR/train.log"
  else
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/grpo.py" train \
      --config "$CONFIG" --checkpoint "$START_CKPT" --mode zoom \
      --target-step "$step" --resume "$TRAIN_DIR" 2>&1 | tee -a "$CURRENT_DIR/train.log"
  fi
  if [ $((step % 50)) -ne 0 ]; then continue; fi
  CKPT="$TRAIN_DIR/checkpoints/step_$(printf '%06d' "$step")_composed.pt"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/zoom.py" eval \
    --config "$CONFIG" --checkpoint "$CKPT" --split val --mode policy \
    --out-json "$CURRENT_DIR/val520_step_${step}.json" \
    --out-jsonl "$CURRENT_DIR/val520_step_${step}.jsonl" \
    2>&1 | tee -a "$CURRENT_DIR/eval.log"
  "$PY" - "$CURRENT_DIR" "$step" "$CKPT" "$MULTICHANNEL_TOLERANCE" "$MAX_STEP" <<'PYEOF'
import hashlib, json, statistics, sys
from collections import defaultdict
from pathlib import Path
root, step, ckpt = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
tolerance = float(sys.argv[4])
history = json.loads((root / "val_history.json").read_text(encoding="utf-8"))
rows = [json.loads(line) for line in
        (root / f"val520_step_{step}.jsonl").open(encoding="utf-8") if line.strip()]
by_category = defaultdict(list)
for row in rows:
    by_category[str(row["category"])].append(float(row["primary"]))
entry = {
    "step": step,
    "primary": statistics.fmean(float(row["primary"]) for row in rows),
    "per_category": {k: statistics.fmean(v) for k, v in by_category.items()},
    "invalid_rate": sum(1 for r in rows if r.get("zoom_invalid")) / len(rows),
    "empty_rate": sum(1 for r in rows if r.get("empty_final")) / len(rows),
    "checkpoint": str(ckpt),
    "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
}
start = history[0]
multichannel_start = start["per_category"].get("localize_all_channels", 0.0)
entry["selectable"] = bool(
    entry["primary"] >= start["primary"]
    and entry["per_category"].get("localize_all_channels", 0.0)
    >= multichannel_start - tolerance
    and entry["invalid_rate"] <= 0.02 and entry["empty_rate"] <= 0.02)
best_prior = max(item["primary"] for item in history)
entry["stale"] = 0 if entry["primary"] >= best_prior + 0.005 else history[-1]["stale"] + 1
history.append(entry)
selectable = [item for item in history if item.get("selectable")]
best = max(item["primary"] for item in selectable)
chosen = next(item for item in selectable if item["primary"] >= best - 0.002)
selection = {
    "status": ("FROZEN" if entry["stale"] >= 2 or step >= int(sys.argv[5])
               else "PROVISIONAL"),
    "history": history, "selected_step": chosen["step"],
    "selected_primary": chosen["primary"],
    "checkpoint": chosen["checkpoint"], "checkpoint_sha256": chosen["checkpoint_sha256"],
    "stale": entry["stale"], "step0_kept_as_candidate": True,
    "multichannel_tolerance": tolerance,
}
(root / "val_history.json").write_text(json.dumps(history, indent=2) + "\n",
                                       encoding="utf-8")
(root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n",
                                     encoding="utf-8")
print(f"step={step} primary={entry['primary']:.4f} "
      f"multicanale={entry['per_category'].get('localize_all_channels', 0.0):.4f} "
      f"selezionabile={entry['selectable']} scelto={chosen['step']} "
      f"stale={entry['stale']}")
PYEOF
  STALE=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['stale'])" \
    "$CURRENT_DIR/selection.json")
  [ "$STALE" -lt 2 ] || { echo "arresto per stagnazione allo step $step"; break; }
done
[ -f "$CURRENT_DIR/selection.json" ] || printf '{"selected_step":0,"checkpoint":"%s","checkpoint_sha256":"%s","note":"nessuna validazione completata"}\n' \
  "$START_CKPT" "$START_SHA" > "$CURRENT_DIR/selection.json"
cp "$CURRENT_DIR/selection.json" "$RUN_ROOT/SELECTION.json"
finish_phase

# ---- verifica della lettura sul solo checkpoint scelto ---------------------
start_phase reading_check
SELECTED=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['checkpoint'])" \
  "$RUN_ROOT/SELECTION.json")
SELECTED_STEP=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['selected_step'])" \
  "$RUN_ROOT/SELECTION.json")
if [ "$SELECTED_STEP" = "0" ]; then
  printf '{"status":"SKIPPED","reason":"il candidato e lo step 0: nessun nuovo checkpoint da verificare"}\n' \
    > "$CURRENT_DIR/summary.json"
else
  for offset in 0 1 2; do
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/tools/diag_crop_content.py" \
      --config "$Z1BIS_CONFIG" --checkpoint "$SELECTED" \
      --reference-rows "$REFERENCE_GATE_ROWS" --out-dir "$CURRENT_DIR/off$offset" \
      --seed-offset "$offset" 2>&1 | tee -a "$CURRENT_DIR/reading.log"
  done
  "$PY" "$L3/tools/reader_vs_windows.py" \
    --pair "$START_READ_0:$CURRENT_DIR/off0/rows.jsonl" \
    --pair "$START_READ_1:$CURRENT_DIR/off1/rows.jsonl" \
    --pair "$START_READ_2:$CURRENT_DIR/off2/rows.jsonl" \
    --out "$CURRENT_DIR/vs_start.json" 2>&1 | tee -a "$CURRENT_DIR/reading.log"
  "$PY" "$L3/tools/reader_vs_windows.py" \
    --pair "$Z1BIS_READ_0:$CURRENT_DIR/off0/rows.jsonl" \
    --pair "$Z1BIS_READ_1:$CURRENT_DIR/off1/rows.jsonl" \
    --pair "$Z1BIS_READ_2:$CURRENT_DIR/off2/rows.jsonl" \
    --out "$CURRENT_DIR/vs_z1bis.json" 2>&1 | tee -a "$CURRENT_DIR/reading.log"
  "$PY" - "$CURRENT_DIR/vs_start.json" "$CURRENT_DIR/summary.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
multichannel = d["per_category"]["localize_all_channels"]["reading"]
rejected = multichannel["mean"] < -0.03 and multichannel["ci95"][1] < 0.0
out = {"status": "REJECTED" if rejected else "ACCEPTED",
       "multichannel_reading_vs_start": multichannel,
       "criterion": "respinto se < -0.03 con CI95 che esclude lo zero"}
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), indent=2)
open(sys.argv[2], "a").write("\n")
print(f"lettura multicanale vs partenza: {multichannel['mean']:+.4f} "
      f"CI[{multichannel['ci95'][0]:+.4f}; {multichannel['ci95'][1]:+.4f}] "
      f"-> {out['status']}")
PYEOF
fi
cp "$CURRENT_DIR/summary.json" "$RUN_ROOT/READING_CHECK.json"
finish_phase

printf 'L3_GRPO_OK run_root=%s verdict=%s start=%s elapsed_s=%s\n' \
  "$RUN_ROOT" "$VERDICT" "$START_LABEL" "$(( $(date +%s) - START_EPOCH ))"
