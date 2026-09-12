#!/usr/bin/env bash
# Punti 1-2 del piano di Codex a pesi congelati: geometria dei selettori (CPU) e
# controllo del contenuto sulle finestre scelte dalla policy (GPU). Nessun training.
set -Eeuo pipefail

ROOT="${L3_ROOT:-$HOME/timerlm/sft/tslm}"
PY="${L3_PYTHON:-$HOME/venv/bin/python}"
L3="$ROOT/l3"
CONFIG="$L3/configs/l3_zoom_z1bis.yaml"
Z1BIS_ROOT="${L3_Z1BIS_ROOT:-$HOME/l3/z1bis_20260912}"
CHECKPOINT="${L3_Z1BIS_CHECKPOINT:-$Z1BIS_ROOT/sft/train/vz1bis_epoch_1_end.pt}"
POLICY_ROWS="${L3_POLICY_ROWS:-$Z1BIS_ROOT/operational_gate/val520_policy.jsonl}"
EXPECTED_CHECKPOINT_SHA="${L3_Z1BIS_CHECKPOINT_SHA:-924110a5a344dc802c8eee646320f0c7035572c69e1125c714a3db737e8fd23f}"
GPU="${L3_GPU:-0}"
RUN_ROOT="${L3_SELECTORS_RUN_ROOT:-$HOME/l3/selectors_$(date -u +%Y%m%dT%H%M%SZ)}"
# Soglie dichiarate prima del run (PROTOCOL-L3.md, sezione selettori).
MIN_DELTA="${L3_SELFWINDOW_MIN_DELTA:-0.03}"
SEED="${L3_SELECTORS_SEED:-20260913}"
CPU_TIMEOUT_SECONDS="${L3_SELECTORS_CPU_TIMEOUT_SECONDS:-3600}"
CPU_PEAK_LIMIT_MIB="${L3_SELECTORS_CPU_PEAK_LIMIT_MIB:-6144}"

[ "${L3_SELECTORS_DIFF_REVIEWED:-}" = "CLAUDE_APPROVED" ] || {
  echo "refusing to start: set L3_SELECTORS_DIFF_REVIEWED=CLAUDE_APPROVED after diff review" >&2
  exit 1
}

if [ "${1:-}" != "--worker" ] && [ -z "${TMUX:-}" ]; then
  command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
  [ ! -e "$RUN_ROOT" ] || { echo "refusing to reuse run root: $RUN_ROOT" >&2; exit 1; }
  mkdir -p "$RUN_ROOT"
  SESSION="l3-selectors-$(date -u +%Y%m%dT%H%M%SZ)"
  tmux new-session -d -s "$SESSION" \
    "env L3_SELECTORS_DIFF_REVIEWED=CLAUDE_APPROVED L3_SELECTORS_RUN_ROOT='$RUN_ROOT' L3_GPU='$GPU' L3_ROOT='$ROOT' L3_PYTHON='$PY' L3_Z1BIS_ROOT='$Z1BIS_ROOT' L3_Z1BIS_CHECKPOINT='$CHECKPOINT' L3_POLICY_ROWS='$POLICY_ROWS' L3_SELFWINDOW_MIN_DELTA='$MIN_DELTA' L3_SELECTORS_SEED='$SEED' L3_SELECTORS_CPU_TIMEOUT_SECONDS='$CPU_TIMEOUT_SECONDS' L3_SELECTORS_CPU_PEAK_LIMIT_MIB='$CPU_PEAK_LIMIT_MIB' bash '$L3/node_l3_selectors.sh' --worker >'$RUN_ROOT/worker.log' 2>&1"
  echo "L3_SELECTORS_DETACHED session=$SESSION run_root=$RUN_ROOT log=$RUN_ROOT/worker.log"
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
  echo "L3_SELECTORS_FAILED phase=$CURRENT_PHASE line=$line rc=$rc" >&2
  exit "$rc"
}
trap 'failure $? $LINENO' ERR

start_phase() {
  CURRENT_PHASE="$1"
  CURRENT_DIR="$RUN_ROOT/$1"
  [ ! -e "$CURRENT_DIR" ] || { echo "refusing to reuse phase directory: $CURRENT_DIR" >&2; return 1; }
  mkdir "$CURRENT_DIR"
  printf 'L3_SELECTORS_STATUS phase=%s elapsed_s=%s\n' \
    "$CURRENT_PHASE" "$(( $(date +%s) - START_EPOCH ))" | tee -a "$CURRENT_DIR/phase.log"
}

finish_phase() {
  : >"$CURRENT_DIR/OK"
  printf 'L3_SELECTORS_PHASE_OK phase=%s elapsed_s=%s\n' \
    "$CURRENT_PHASE" "$(( $(date +%s) - START_EPOCH ))" | tee -a "$CURRENT_DIR/phase.log"
}

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
# Stesso ambiente esatto della catena Z1-bis: il checkpoint contiene solo i tensori
# allenabili (TSLM_SAVE_TRAINABLE_ONLY) e dtype, attenzione e revisione devono
# coincidere, altrimenti la misura non e' confrontabile con il gate percettivo.
unset PYTORCH_CUDA_ALLOC_CONF WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
export TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16
export TSLM_ATTN_IMPL=sdpa TSLM_SAVE_TRAINABLE_ONLY=1 TSLM_GRAD_CKPT=1
export TSLM_QWEN_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# ---- preflight -------------------------------------------------------------
start_phase preflight
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG" >&2; exit 1; }
[ -f "$CHECKPOINT" ] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 1; }
[ -f "$POLICY_ROWS" ] || { echo "missing policy rows: $POLICY_ROWS" >&2; exit 1; }
ACTUAL_SHA=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
[ "$ACTUAL_SHA" = "$EXPECTED_CHECKPOINT_SHA" ] || {
  echo "checkpoint SHA mismatch: $ACTUAL_SHA" >&2; exit 1; }
POLICY_SHA=$(sha256sum "$POLICY_ROWS" | awk '{print $1}')
ACTED=$(grep -c '"zoom": {' "$POLICY_ROWS" || true)
{
  echo "checkpoint_sha256=$ACTUAL_SHA"
  echo "policy_rows_sha256=$POLICY_SHA"
  echo "policy_acted_rows=$ACTED"
  echo "kit_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "min_delta=$MIN_DELTA seed=$SEED"
} | tee "$CURRENT_DIR/preflight.txt"
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest "$L3/tests/test_l3_selectors.py" -q \
  2>&1 | tee "$CURRENT_DIR/pytest.log"
finish_phase

# ---- geometria dei selettori (solo CPU) ------------------------------------
start_phase selectors
set +e
CUDA_VISIBLE_DEVICES="" timeout --signal=TERM "$CPU_TIMEOUT_SECONDS" \
  "$PY" "$L3/tools/compare_selectors.py" \
  --config "$CONFIG" --policy-rows "$POLICY_ROWS" \
  --out-dir "$CURRENT_DIR/geometry" 2>&1 | tee "$CURRENT_DIR/selectors.log"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" -eq 0 ] || { echo "selector geometry failed rc=$RC" >&2; exit "$RC"; }
PEAK=$("$PY" -c "import json,sys; print(int(json.load(open(sys.argv[1]))['runtime'].get('peak_rss_mib',0)))" \
  "$CURRENT_DIR/geometry/summary.json")
[ "$PEAK" -le "$CPU_PEAK_LIMIT_MIB" ] || {
  echo "selector phase exceeded the declared peak: ${PEAK} MiB" >&2; exit 1; }
finish_phase

# ---- contenuto sulle finestre autonome (GPU) -------------------------------
start_phase selfwindow
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$L3/tools/gate_selfwindow_content.py" \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --policy-rows "$POLICY_ROWS" \
  --out-dir "$CURRENT_DIR/content" --min-delta "$MIN_DELTA" --seed "$SEED" \
  2>&1 | tee "$CURRENT_DIR/selfwindow.log"
finish_phase

printf 'L3_SELECTORS_OK run_root=%s elapsed_s=%s\n' \
  "$RUN_ROOT" "$(( $(date +%s) - START_EPOCH ))"
