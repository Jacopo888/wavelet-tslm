#!/usr/bin/env bash
# Schedule-faithful validation-only pilot.  This launcher never invokes the
# five-epoch worker and never evaluates the test split.
set -Eeuo pipefail

PILOT_NAME="${1:?uso: node_pilot_v3.sh <pilot_name> <config.yaml> <n_epochs>}"
INPUT_CONFIG="${2:?uso: node_pilot_v3.sh <pilot_name> <config.yaml> <n_epochs>}"
N_EPOCHS="${3:?uso: node_pilot_v3.sh <pilot_name> <config.yaml> <n_epochs>}"
case "$PILOT_NAME" in *[!a-zA-Z0-9_-]*|'') echo "pilot_name non valido"; exit 2;; esac
case "$N_EPOCHS" in ''|*[!0-9]*) echo "n_epochs deve essere intero"; exit 2;; esac
[ "$N_EPOCHS" -ge 1 ] && [ "$N_EPOCHS" -le 5 ] || {
  echo "n_epochs deve essere fra 1 e 5 (scheduler originale a 5 epoche)"; exit 2;
}

ROOT="$HOME/timerlm/sft/tslm"
PY="$HOME/venv/bin/python"
GPU="${V3_GPU:-0}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_NAME="pilot_${PILOT_NAME}_${TIMESTAMP}"
RUN_DIR="$ROOT/results/anomaly_xl_precise/$RUN_NAME"
LOG_DIR="$HOME/logs"
EVAL_DIR="$RUN_DIR/pilot_eval"
INIT_STATE="$HOME/pilots/init_common.pt"
FORWARD_AUDIT="$HOME/patch/forward_audit.py"
PAIRED_EVAL="$HOME/patch/paired_eval.py"
TRAIN_LOG="$LOG_DIR/${RUN_NAME}.train.log"
WORKER_LOG="$LOG_DIR/${RUN_NAME}.eval.log"
FAILED="$RUN_DIR/PILOT_FAILED"
DONE="$RUN_DIR/PILOT_OK"
TRAIN_CONFIG="$RUN_DIR/pilot_train.yaml"
PROBE_CONFIG="$RUN_DIR/pilot_probe80.yaml"
TRAIN_PID=""
EVAL_PID=""

fail() {
  local reason="$*"
  echo "$reason" | tee "$FAILED" >&2
  [ -n "$TRAIN_PID" ] && kill "$TRAIN_PID" 2>/dev/null || true
  [ -n "$EVAL_PID" ] && kill "$EVAL_PID" 2>/dev/null || true
  exit 1
}
trap 'fail "errore inatteso alla riga $LINENO"' ERR

cd "$ROOT"
[ -x "$PY" ] || fail "python assente: $PY"
[ -f "$FORWARD_AUDIT" ] || fail "forward_audit.py assente: rieseguire ship_v3.sh e node_env_v3.sh"
[ -f "$PAIRED_EVAL" ] || fail "paired_eval.py assente: rieseguire ship_v3.sh"
[ -f "$HOME/RUN_MANIFEST_V3.json" ] || fail "manca RUN_MANIFEST_V3.json"
if [ ! -f "$INPUT_CONFIG" ]; then
  [ -f "$ROOT/$INPUT_CONFIG" ] && INPUT_CONFIG="$ROOT/$INPUT_CONFIG"
fi
[ -f "$INPUT_CONFIG" ] || fail "config assente: $INPUT_CONFIG"
[ ! -e "$RUN_DIR" ] || fail "run dir gia esistente: $RUN_DIR"
mkdir -p "$RUN_DIR" "$EVAL_DIR" "$LOG_DIR" "$HOME/pilots"

# First pilot is the no-token-gate reference.  Later pilots load the same
# strict common keys; only a newly present identity token gate is permitted.
if [ ! -f "$INIT_STATE" ]; then
  TOKEN_GATE=$($PY - "$INPUT_CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))
print(str(bool(cfg['model']['encoder'].get('token_gate', False))).lower())
PY
)
  [ "$TOKEN_GATE" = "false" ] || fail "init_common.pt assente: lancia prima il pilot cross senza token gate"
fi

# Produce immutable run/probe configs without changing scheduler or training
# data size.  max_samples=800 means the fixed validation probe is 80 rows.
$PY - "$INPUT_CONFIG" "$TRAIN_CONFIG" "$PROBE_CONFIG" "$RUN_NAME" <<'PY'
import copy, sys, yaml
source, train_out, probe_out, run_name = sys.argv[1:]
cfg = yaml.safe_load(open(source, encoding='utf-8'))
if cfg['training']['num_epochs'] != 5 or float(cfg['training']['warmup_ratio']) != 0.03:
    raise SystemExit('pilot reference must retain num_epochs=5 and warmup_ratio=0.03')
cfg['name'] = run_name
cfg['runtime']['output_dir'] = 'results'
cfg['runtime']['run_name'] = run_name
cfg['runtime']['max_samples'] = None
probe = copy.deepcopy(cfg)
probe['runtime']['max_samples'] = 800
probe['runtime']['eval_samples_ratio'] = 0.1
with open(train_out, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
with open(probe_out, 'w', encoding='utf-8') as f:
    yaml.safe_dump(probe, f, sort_keys=False)
PY

cp "$HOME/RUN_MANIFEST_V3.json" "$RUN_DIR/RUN_MANIFEST.json"
$PY - "$RUN_DIR" "$INPUT_CONFIG" "$N_EPOCHS" "$INIT_STATE" \
  "$FORWARD_AUDIT" "$PAIRED_EVAL" "$HOME/node_pilot_v3.sh" <<'PY'
import hashlib, json, pathlib, sys
run_dir, source, epochs, init_state, forward_audit, paired_eval, launcher = sys.argv[1:]
run_dir = pathlib.Path(run_dir)
base = json.load(open(run_dir/'RUN_MANIFEST.json', encoding='utf-8'))
files = [run_dir/'pilot_train.yaml', run_dir/'pilot_probe80.yaml', pathlib.Path(source),
         pathlib.Path(forward_audit), pathlib.Path(paired_eval), pathlib.Path(launcher)]
base['pilot'] = {
    'n_epochs': int(epochs), 'init_common': init_state,
    'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    'train_steps_per_epoch': 4488, 'scheduler_total_steps': 22440,
    'warmup_steps': 673, 'eval_split': 'val',
}
json.dump(base, open(run_dir/'PILOT_MANIFEST.json', 'w', encoding='utf-8'), indent=2)
PY

$PY preflight_data.py "$HOME/timerlm/anomalyXL/datasets" --variant precise
QWEN_REV=$($PY -c "import json,os;print(json.load(open(os.path.expanduser('~/RUN_MANIFEST_V3.json')))['qwen_revision'])")
[ -n "$QWEN_REV" ] || fail "qwen revision vuota"

COMMON_ENV=(
  TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16
  TSLM_ATTN_IMPL=sdpa TSLM_GRAD_CKPT=1 TSLM_CKPT_FALLBACK=1
  TSLM_EVAL_STRICT=1 TSLM_QWEN_REVISION="$QWEN_REV"
  TSLM_SAVE_TRAINABLE_ONLY=1 HF_HUB_DISABLE_PROGRESS_BARS=1
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="$GPU"
  HIP_VISIBLE_DEVICES="$GPU" ROCR_VISIBLE_DEVICES="$GPU"
)

wait_for_log() {
  local pattern="$1"
  while ! tr '\r' '\n' < "$TRAIN_LOG" 2>/dev/null | grep -Fq "$pattern"; do
    kill -0 "$TRAIN_PID" 2>/dev/null || {
      tr '\r' '\n' < "$TRAIN_LOG" | tail -80 >&2 || true
      return 1
    }
    sleep 20
  done
}

eval_checkpoint() {
  local tag="$1" checkpoint="$2"
  local on_json="$EVAL_DIR/${tag}_generation_on.json"
  local off_json="$EVAL_DIR/${tag}_generation_off.json"
  local roll_json="$EVAL_DIR/${tag}_generation_roll.json"
  local forward_json="$EVAL_DIR/${tag}_teacher_forcing.json"
  env "${COMMON_ENV[@]}" TSLM_EVAL_STRICT=0 TSLM_V3_COLLECT_CONTEXT_STATS=1 TSLM_EVAL_TAG="${tag}_on" \
    "$PY" train.py --config "$PROBE_CONFIG" --no-wandb --eval-only \
    --checkpoint "$checkpoint" --split val --out-json "$on_json"
  env "${COMMON_ENV[@]}" TSLM_EVAL_STRICT=0 TSLM_V3_COLLECT_CONTEXT_STATS=1 TSLM_V3_CONTEXT_EVAL=off TSLM_EVAL_TAG="${tag}_off" \
    "$PY" train.py --config "$PROBE_CONFIG" --no-wandb --eval-only \
    --checkpoint "$checkpoint" --split val --out-json "$off_json"
  env "${COMMON_ENV[@]}" TSLM_EVAL_STRICT=0 TSLM_V3_COLLECT_CONTEXT_STATS=1 TSLM_V3_CONTEXT_EVAL=roll_tokens TSLM_EVAL_TAG="${tag}_roll" \
    "$PY" train.py --config "$PROBE_CONFIG" --no-wandb --eval-only \
    --checkpoint "$checkpoint" --split val --out-json "$roll_json"
  env "${COMMON_ENV[@]}" "$PY" "$FORWARD_AUDIT" \
    --pilot-config "$PROBE_CONFIG" --checkpoint "$checkpoint" \
    --out-json "$forward_json" --n 80
  "$PY" - "$on_json" "$off_json" "$roll_json" "$forward_json" <<'PY'
import json, math, sys
on, off, roll, forward = [json.load(open(path, encoding='utf-8')) for path in sys.argv[1:]]
for label, doc in [('on', on), ('off', off), ('roll', roll)]:
    assert doc['split'] == 'val', (label, doc.get('split'))
    assert doc['samples_evaluated'] == 80, (label, doc.get('samples_evaluated'))
    assert math.isfinite(float(doc['primary'])), (label, doc.get('primary'))
    assert isinstance(doc.get('category_primary'), dict) and doc['category_primary'], label
assert forward['split'] == 'val' and forward['n'] == 80
assert set(forward['conditions']) == {'on', 'off', 'roll_tokens'}
required_context_keys = {
    f'{metric}_{group}'
    for metric in (
        'token_variable_fraction', 'mean_token_cosine',
        'input_series_relative',
    )
    for group in ('total', 'coarse', 'detail')
}
assert required_context_keys <= set(forward['context_final_mean']), (
    required_context_keys - set(forward['context_final_mean'])
)
assert forward['context_final_mean']['token_variable_fraction_total'] is not None
bands = forward['context_final_mean'].get('by_base_band')
assert isinstance(bands, dict) and bands
required_band_keys = {
    'token_variable_fraction', 'mean_token_cosine',
    'group_mean_rms_over_h_rms', 'input_series_relative',
}
assert all(required_band_keys <= set(stats) for stats in bands.values()), bands
PY
}

eval_worker() {
  wait_for_log "[Checkpoint] Saved at epoch 1, batch 500"
  eval_checkpoint step500 "$RUN_DIR/checkpoints/checkpoint_epoch_1_batch_500.pt"
  wait_for_log "[Checkpoint] Saved at epoch 1, batch 2000"
  eval_checkpoint step2000 "$RUN_DIR/checkpoints/checkpoint_epoch_1_batch_2000.pt"
  for epoch in $(seq 1 "$N_EPOCHS"); do
    wait_for_log "[EpochCkpt] kept epoch_${epoch}_end.pt"
    checkpoint="$RUN_DIR/checkpoints/epoch_${epoch}_end.pt"
    eval_checkpoint "epoch${epoch}" "$checkpoint"
    env "${COMMON_ENV[@]}" TSLM_V3_COLLECT_CONTEXT_STATS=1 TSLM_EVAL_TAG="epoch${epoch}_full_val_on" \
      "$PY" train.py --config "$TRAIN_CONFIG" --no-wandb --eval-only \
      --checkpoint "$checkpoint" --split val \
      --out-json "$EVAL_DIR/epoch${epoch}_full_val_on.json"
    env "${COMMON_ENV[@]}" TSLM_V3_COLLECT_CONTEXT_STATS=1 TSLM_V3_CONTEXT_EVAL=off TSLM_EVAL_TAG="epoch${epoch}_full_val_off" \
      "$PY" train.py --config "$TRAIN_CONFIG" --no-wandb --eval-only \
      --checkpoint "$checkpoint" --split val \
      --out-json "$EVAL_DIR/epoch${epoch}_full_val_off.json"
    full_roll_args=()
    if [ "${V3_FULL_VAL_ROLL:-0}" = "1" ]; then
      env "${COMMON_ENV[@]}" TSLM_V3_COLLECT_CONTEXT_STATS=1 TSLM_V3_CONTEXT_EVAL=roll_tokens TSLM_EVAL_TAG="epoch${epoch}_full_val_roll" \
        "$PY" train.py --config "$TRAIN_CONFIG" --no-wandb --eval-only \
        --checkpoint "$checkpoint" --split val \
        --out-json "$EVAL_DIR/epoch${epoch}_full_val_roll.json"
      full_roll_args=(--full-roll "$EVAL_DIR/epoch${epoch}_full_val_roll.json")
    fi
    "$PY" "$PAIRED_EVAL" --out "$EVAL_DIR/epoch${epoch}_paired.json" \
      --generation-on "$EVAL_DIR/epoch${epoch}_generation_on.json" \
      --generation-off "$EVAL_DIR/epoch${epoch}_generation_off.json" \
      --generation-roll "$EVAL_DIR/epoch${epoch}_generation_roll.json" \
      --teacher-forcing "$EVAL_DIR/epoch${epoch}_teacher_forcing.json" \
      --full-on "$EVAL_DIR/epoch${epoch}_full_val_on.json" \
      --full-off "$EVAL_DIR/epoch${epoch}_full_val_off.json" \
      "${full_roll_args[@]}"
  done
}

TRAIN_STEPS=$((N_EPOCHS * 4488))
env "${COMMON_ENV[@]}" \
  TSLM_INIT_STATE_PATH="$INIT_STATE" TSLM_DATA_ORDER_SEED=42 TSLM_SUBSET_SEED=42 \
  TSLM_MAX_TRAIN_STEPS="$TRAIN_STEPS" TSLM_SCHED_TOTAL_STEPS=22440 \
  TSLM_CHECKPOINT_STEPS=500,2000 TSLM_SKIP_FINAL_TEST=1 \
  TSLM_KEEP_EPOCH_CKPTS=1 TSLM_VAL_GEN=0 TSLM_V3_GATE_LOG_EVERY=100 \
  TSLM_V3_GRAD_LOG=1 TSLM_V3_CTX_LOG_EVERY=100 \
  "$PY" train.py --config "$TRAIN_CONFIG" --no-wandb > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
eval_worker > "$WORKER_LOG" 2>&1 &
EVAL_PID=$!

set +e
wait "$TRAIN_PID"; TRAIN_RC=$?
wait "$EVAL_PID"; EVAL_RC=$?
set -e
[ "$TRAIN_RC" -eq 0 ] || fail "training fallito rc=$TRAIN_RC; log $TRAIN_LOG"
[ "$EVAL_RC" -eq 0 ] || fail "eval val fallita rc=$EVAL_RC; log $WORKER_LOG"

for epoch in $(seq 1 "$N_EPOCHS"); do
  count=$(tr '\r' '\n' < "$TRAIN_LOG" | grep -Ec "^[[:space:]]*\[Coverage\] epoch ${epoch}: trained 4488, failed 0, expected 4488[[:space:]]*$" || true)
  [ "$count" -eq 1 ] || fail "coverage epoca $epoch non esatta (righe=$count)"
  [ -s "$EVAL_DIR/epoch${epoch}_full_val_on.json" ] || fail "val intera epoca $epoch assente"
  [ -s "$EVAL_DIR/epoch${epoch}_full_val_off.json" ] || fail "val intera off epoca $epoch assente"
  [ -s "$EVAL_DIR/epoch${epoch}_generation_roll.json" ] || fail "probe roll epoca $epoch assente"
  [ -s "$EVAL_DIR/epoch${epoch}_paired.json" ] || fail "confronto appaiato epoca $epoch assente"
  "$PY" - "$EVAL_DIR/epoch${epoch}_full_val_on.json" "$EVAL_DIR/epoch${epoch}_full_val_off.json" "$EVAL_DIR/epoch${epoch}_paired.json" <<'PY'
import json, math, sys
on, off, paired = [json.load(open(path, encoding='utf-8')) for path in sys.argv[1:]]
for doc in (on, off):
    assert doc['split'] == 'val' and doc['samples_evaluated'] == 520, doc
    assert math.isfinite(float(doc['primary']))
    assert isinstance(doc.get('category_primary'), dict) and doc['category_primary']
assert paired['bootstrap']['n_resamples'] == 10000
assert paired['full_val']['n'] == 520
assert 'on_minus_off' in paired['full_val']['comparisons']
for section in ('generation_probe80', 'teacher_forcing_probe80'):
    assert set(paired[section]['comparisons']) == {
        'on_minus_off', 'on_minus_roll', 'off_minus_roll'
    }, paired[section]
PY
done
grep -Fq "[StepBudget] completed at epoch boundary: global_step=${TRAIN_STEPS}, cap=${TRAIN_STEPS}" "$TRAIN_LOG" || \
  fail "arresto al confine epoca non verificato"

"$PY" - "$RUN_DIR/PILOT_MANIFEST.json" "$INIT_STATE" <<'PY'
import hashlib, json, pathlib, sys
manifest_path, init_path = map(pathlib.Path, sys.argv[1:])
doc = json.load(open(manifest_path, encoding='utf-8'))
digest = hashlib.sha256()
with init_path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
        digest.update(chunk)
doc['pilot']['init_common_sha256'] = digest.hexdigest()
json.dump(doc, open(manifest_path, 'w', encoding='utf-8'), indent=2)
PY

printf 'PILOT_OK name=%s run_dir=%s epochs=%s\n' "$PILOT_NAME" "$RUN_DIR" "$N_EPOCHS" | tee "$DONE"
echo "training log: $TRAIN_LOG"
echo "evaluation log: $WORKER_LOG"
