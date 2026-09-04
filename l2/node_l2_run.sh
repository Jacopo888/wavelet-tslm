#!/usr/bin/env bash
# L2 pilot run driver — one arm, two phases, fail-closed. Usage: node_l2_run.sh <code|wavelet>
#
#   phase 1 (steps 1..35): negatives withheld  -> ANOMALYXL_P_PRESENT=1.0 ANOMALYXL_ALLCH_MIN_K=1
#   phase 2 (steps 36..60): resume from step_35 with both vars UNSET (benchmark 80/20 split)
# exactly the paper's two-phase recipe (220/400 -> 35/60), driven from the LAUNCH env of the
# `rl` process (the generator reads the vars at import inside the env-server process, which
# inherits this shell; setting them in harness.env would silently do nothing).
# Arm `wavelet` additionally exports TIMERLM_NOTE_VARIANT=wavelet (task-note sentence); the
# RLM_SKILLS harness entry lives in its config. Everything else is identical between arms.
#
# Gates: env manifest + SMOKE_OK (stages a b c) must exist; a run dir with DONE/FAILED is
# never reused silently. Every wait is bounded. Any process death, missing checkpoint, or
# failed resume writes FAILED with the reason and exits non-zero. Coverage: l2_monitor.py
# follows the run (tmux l2mon) and the post-hoc pass must report zero violations for DONE.
set -euo pipefail
ARM="${1:-}"; [ "$ARM" = code ] || [ "$ARM" = wavelet ] || { echo "usage: $0 <code|wavelet>"; exit 2; }
TIMERLM="${TIMERLM:-$HOME/timerlm}"; PR="$TIMERLM/prime-rl"; L2="$TIMERLM/l2"
KIT="${L2_KIT:-$HOME/l2/kit/l2}"; LOGS="$HOME/l2/logs"; RUN="$HOME/l2/runs/$ARM"
MANIFEST="$HOME/L2_ENV_MANIFEST.json"; SMOKE_OK="$HOME/l2/smoke/SMOKE_OK"
PHASE1_STEPS="${L2_PHASE1_STEPS:-35}"; TOTAL_STEPS="${L2_TOTAL_STEPS:-60}"
PHASE_WALL_H="${L2_PHASE_WALL_H:-45}"          # bounded wait per phase (hours)
EXPECT_ROLLOUTS="${L2_EXPECT_ROLLOUTS:-64}"
PORT=8000; VENV_PY="$PR/.venv/bin/python"
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
# vGPU: no CUDA memory pools -> NCCL cannot init; single-rank FSDP2 on gloo (see overlay/prime-rl)
export PRIME_RL_DIST_BACKEND="cpu:gloo,cuda:gloo"
# vGPU: no CUDA VMM -> the launcher must not set expandable_segments for the trainer (overlay/prime-rl rl.py)
export PRIME_RL_TRAINER_ALLOC_CONF=""
# flash-linear-attention routes GatedDeltaNet chunk kernels to TileLang (JIT via nvcc, absent here): force the Triton path
export FLA_TILELANG=0
# torch._grouped_mm backward asserts 16-byte strides here: plain per-adapter matmul (overlay/prime-rl multi_linear.py)
export PRIME_RL_LORA_GROUPED_MM=0
mkdir -p "$LOGS" "$HOME/l2/runs"

log() { echo "[l2-run/$ARM $(date -u +%H:%M:%S)] $*" | tee -a "$LOGS/run_${ARM}_driver.log"; }
fail() { log "FAILED: $*"; mkdir -p "$RUN"; echo "$(date -u +%FT%TZ) $*" > "$RUN/FAILED"; tmux kill-session -t "l2$ARM" 2>/dev/null || true; tmux kill-session -t l2mon 2>/dev/null || true; exit 1; }

# ------------------------------------------------------------------ gates
[ -f "$MANIFEST" ] || fail "no env manifest (run node_l2_env.sh)"
[ -f "$SMOKE_OK" ] || fail "no SMOKE_OK (run node_l2_smoke.sh a b c d first)"
for s in a b c; do grep -q " $s" "$SMOKE_OK" || grep -q "SMOKE_OK $s" "$SMOKE_OK" || grep -qE "(^| )$s( |$)" "$SMOKE_OK" || fail "smoke stage $s not recorded in SMOKE_OK"; done
[ "$SMOKE_OK" -nt "$MANIFEST" ] || fail "SMOKE_OK is older than the env manifest: re-run the smoke"
[ -x "$VENV_PY" ] || fail "prime-rl venv missing"
mkdir -p "$L2/configs" && cp "$KIT"/configs/*.toml "$L2/configs/"
CFG="../l2/configs/l2_pilot_${ARM}.toml"
[ -f "$L2/configs/l2_pilot_${ARM}.toml" ] || fail "config missing"
if [ -e "$RUN/DONE" ] || [ -e "$RUN/FAILED" ]; then
  [ "${L2_FORCE:-0}" = 1 ] || fail "$RUN already has DONE/FAILED; set L2_FORCE=1 to wipe and rerun"
fi
rm -rf "$RUN"; mkdir -p "$RUN"
tmux kill-session -t "l2$ARM" 2>/dev/null || true; tmux kill-session -t l2mon 2>/dev/null || true

# ------------------------------------------------------------------ server (shared GPU)
server_up() { curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }
if ! server_up; then
  tmux kill-session -t l2srv 2>/dev/null || true
  tmux new-session -d -s l2srv "cd '$PR' && uv run --no-sync inference @ ../l2/configs/inference_l2.toml > '$LOGS/inference_${ARM}.log' 2>&1; echo EXIT:\$? >> '$LOGS/inference_${ARM}.log'"
  t=0; until server_up; do sleep 5; t=$((t+5)); grep -q "EXIT:" "$LOGS/inference_${ARM}.log" 2>/dev/null && fail "server died at startup"; [ $t -ge 900 ] && fail "server not healthy after 900s"; done
  log "server healthy after ${t}s"
fi
MODEL_NAME="$("$VENV_PY" -c "import tomllib;print(tomllib.load(open('$L2/configs/inference_l2.toml','rb'))['model']['name'])")"
curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL_NAME" || fail "server does not serve $MODEL_NAME"

# ------------------------------------------------------------------ helpers
probe_negatives() {  # prints the effective P_PRESENT the env-server would see in THIS env
  "$VENV_PY" -c "import anomalyxl.questions.localize as l, os; print(l._P_PRESENT, os.environ.get('ANOMALYXL_ALLCH_MIN_K','unset'))"
}
run_phase() {  # $1 phase label, $2.. extra rl args; expects env already exported
  local phase="$1"; shift
  local plog="$LOGS/${ARM}_${phase}.log" exitf="$RUN/${phase}.EXIT"
  rm -f "$exitf"
  tmux new-session -d -s "l2$ARM" "cd '$PR' && uv run --no-sync rl @ $CFG --output-dir '$RUN' $* > '$plog' 2>&1; echo EXIT:\$? > '$exitf'"
  log "$phase launched: rl @ $CFG $* (tmux l2$ARM, log $plog)"
  local t=0 max=$((PHASE_WALL_H*3600))
  while [ ! -f "$exitf" ]; do
    sleep 60; t=$((t+60))
    server_up || fail "$phase: inference server stopped answering"
    tmux has-session -t "l2$ARM" 2>/dev/null || { sleep 5; [ -f "$exitf" ] || fail "$phase: rl session vanished without EXIT"; }
    [ $t -ge $max ] && fail "$phase: exceeded ${PHASE_WALL_H}h wall limit"
    if [ $((t % 1800)) -eq 0 ]; then
      "$VENV_PY" "$KIT/l2_monitor.py" "$RUN" --expect-rollouts "$EXPECT_ROLLOUTS" 2>/dev/null | tail -1 | tee -a "$LOGS/run_${ARM}_driver.log" || true
    fi
  done
  local code; code="$(cat "$exitf")"
  [ "$code" = "EXIT:0" ] || { tail -40 "$plog"; fail "$phase: rl exited $code"; }
  log "$phase finished EXIT:0"
}
require_ckpt() {  # $1 step — at this pin the trainer checkpoint lives in checkpoints/step_N/trainer
  [ -d "$RUN/checkpoints/step_$1/trainer" ] || fail "missing $RUN/checkpoints/step_$1/trainer"
  ls "$RUN/checkpoints/step_$1/trainer"/* >/dev/null 2>&1 || fail "empty trainer checkpoint step_$1"
  [ -f "$RUN/weights/step_$1/model.safetensors" ] || fail "missing weights/step_$1/model.safetensors"
  log "checkpoint step_$1 present (trainer + HF weights)"
}

# ------------------------------------------------------------------ phase 1: negatives OFF
export ANOMALYXL_P_PRESENT=1.0 ANOMALYXL_ALLCH_MIN_K=1
[ "$ARM" = wavelet ] && export TIMERLM_NOTE_VARIANT=wavelet || export TIMERLM_NOTE_VARIANT=default
read -r P1 K1 <<<"$(probe_negatives)"; [ "$P1" = "1.0" ] && [ "$K1" = "1" ] || fail "phase-1 env probe: P_PRESENT=$P1 MIN_K=$K1"
log "phase-1 env verified (P_PRESENT=$P1, ALLCH_MIN_K=$K1, NOTE_VARIANT=$TIMERLM_NOTE_VARIANT)"
tmux new-session -d -s l2mon "'$VENV_PY' '$KIT/l2_monitor.py' '$RUN' --follow --interval 120 --jsonl '$RUN/coverage.jsonl' --expect-rollouts $EXPECT_ROLLOUTS > '$LOGS/${ARM}_coverage.log' 2>&1"
cat > "$RUN/RUN_MANIFEST_L2.json" <<JSON
{"arm": "$ARM", "config": "$CFG", "phase1_steps": $PHASE1_STEPS, "total_steps": $TOTAL_STEPS,
 "phase1_env": {"ANOMALYXL_P_PRESENT": "1.0", "ANOMALYXL_ALLCH_MIN_K": "1", "TIMERLM_NOTE_VARIANT": "$TIMERLM_NOTE_VARIANT"},
 "phase2_env": {"ANOMALYXL_P_PRESENT": "unset", "ANOMALYXL_ALLCH_MIN_K": "unset", "TIMERLM_NOTE_VARIANT": "$TIMERLM_NOTE_VARIANT"},
 "started_utc": "$(date -u +%FT%TZ)", "env_manifest": "$MANIFEST", "smoke_ok": "$(cat "$SMOKE_OK")"}
JSON
run_phase phase1 --max-steps "$PHASE1_STEPS"
require_ckpt "$PHASE1_STEPS"

# ------------------------------------------------------------------ phase 2: negatives ON (resume)
unset ANOMALYXL_P_PRESENT ANOMALYXL_ALLCH_MIN_K
read -r P2 K2 <<<"$(probe_negatives)"; [ "$P2" = "0.8" ] && [ "$K2" = "unset" ] || fail "phase-2 env probe: P_PRESENT=$P2 MIN_K=$K2"
log "phase-2 env verified (P_PRESENT=$P2, ALLCH_MIN_K unset)"
run_phase phase2 --max-steps "$TOTAL_STEPS" --ckpt.resume-step "$PHASE1_STEPS"
grep -qiE "resum" "$LOGS/${ARM}_phase2.log" || fail "phase 2 log shows no resume"
require_ckpt "$TOTAL_STEPS"
[ -d "$RUN/weights/step_$TOTAL_STEPS" ] || fail "no HF weights at weights/step_$TOTAL_STEPS (needed for the final eval)"

# ------------------------------------------------------------------ finalize
tmux kill-session -t l2mon 2>/dev/null || true
"$VENV_PY" "$KIT/l2_monitor.py" "$RUN" --jsonl "$RUN/coverage_final.jsonl" --expect-rollouts "$EXPECT_ROLLOUTS" > "$RUN/coverage_final.txt" 2>&1 || fail "post-hoc coverage reported violations (see $RUN/coverage_final.txt)"
NSTEPS=$(grep -c "^\[L2-Coverage\] step" "$RUN/coverage_final.txt" || true)
[ "$NSTEPS" -ge "$TOTAL_STEPS" ] || fail "coverage has $NSTEPS steps, expected $TOTAL_STEPS"
cp "$MANIFEST" "$RUN/L2_ENV_MANIFEST.json"; cp "$L2/configs/l2_pilot_${ARM}.toml" "$L2/configs/inference_l2.toml" "$RUN/"
"$VENV_PY" - "$RUN/RUN_MANIFEST_L2.json" <<PY
import json, sys, datetime
p = sys.argv[1]; m = json.load(open(p)); m["finished_utc"] = datetime.datetime.utcnow().isoformat() + "Z"; m["steps_covered"] = $NSTEPS
json.dump(m, open(p, "w"), indent=1)
PY
echo "EXIT:0 $(date -u +%FT%TZ)" > "$RUN/DONE"
log "DONE: $TOTAL_STEPS steps, coverage $NSTEPS/$TOTAL_STEPS, weights at $RUN/weights/step_$TOTAL_STEPS"
