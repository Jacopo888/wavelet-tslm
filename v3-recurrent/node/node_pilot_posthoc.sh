#!/usr/bin/env bash
# Add missing v3 pilot evaluations without modifying training or existing files.
set -Eeuo pipefail

RUN_DIR="${1:?uso: node_pilot_posthoc.sh <run_dir> <epoch>}"
EPOCH="${2:?uso: node_pilot_posthoc.sh <run_dir> <epoch>}"
case "$EPOCH" in ''|*[!0-9]*) echo "epoch deve essere intero"; exit 2;; esac

ROOT="$HOME/timerlm/sft/tslm"
PY="$HOME/venv/bin/python"
GPU="${V3_GPU:-0}"
FORWARD_AUDIT="$HOME/patch/forward_audit.py"
PAIRED_EVAL="$HOME/patch/paired_eval.py"

fail() { echo "$*" >&2; exit 1; }
trap 'fail "errore inatteso alla riga $LINENO"' ERR

[ -d "$RUN_DIR" ] || {
  [ -d "$ROOT/results/anomaly_xl_precise/$RUN_DIR" ] && \
    RUN_DIR="$ROOT/results/anomaly_xl_precise/$RUN_DIR"
}
[ -d "$RUN_DIR" ] || fail "run dir assente: $RUN_DIR"
RUN_DIR=$(cd "$RUN_DIR" && pwd -P)
TRAIN_CONFIG="$RUN_DIR/pilot_train.yaml"
PROBE_CONFIG="$RUN_DIR/pilot_probe80.yaml"
CHECKPOINT="$RUN_DIR/checkpoints/epoch_${EPOCH}_end.pt"
EVAL_DIR="$RUN_DIR/pilot_eval"
[ -x "$PY" ] || fail "python assente: $PY"
[ -f "$TRAIN_CONFIG" ] || fail "config assente: $TRAIN_CONFIG"
[ -f "$PROBE_CONFIG" ] || fail "config assente: $PROBE_CONFIG"
[ -f "$CHECKPOINT" ] || fail "checkpoint assente: $CHECKPOINT"
[ -f "$FORWARD_AUDIT" ] || fail "forward_audit.py assente: rieseguire ship_v3.sh"
[ -f "$PAIRED_EVAL" ] || fail "paired_eval.py assente: rieseguire ship_v3.sh"
mkdir -p "$EVAL_DIR"

cd "$ROOT"
MANIFEST="$RUN_DIR/RUN_MANIFEST.json"
[ -f "$MANIFEST" ] || MANIFEST="$HOME/RUN_MANIFEST_V3.json"
[ -f "$MANIFEST" ] || fail "manifest qwen assente"
QWEN_REV=$($PY - "$MANIFEST" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8'))['qwen_revision'])
PY
)

COMMON_ENV=(
  TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16
  TSLM_ATTN_IMPL=sdpa TSLM_GRAD_CKPT=1 TSLM_CKPT_FALLBACK=1
  TSLM_EVAL_STRICT=1 TSLM_QWEN_REVISION="$QWEN_REV"
  TSLM_SAVE_TRAINABLE_ONLY=1 HF_HUB_DISABLE_PROGRESS_BARS=1
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="$GPU"
  HIP_VISIBLE_DEVICES="$GPU" ROCR_VISIBLE_DEVICES="$GPU"
)

run_generation_if_missing() {
  local output="$1" config="$2" condition="$3" tag="$4"
  [ ! -e "$output" ] || { echo "KEEP $output"; return; }
  local intervention=()
  [ "$condition" = "on" ] || intervention=(TSLM_V3_CONTEXT_EVAL="$condition")
  # Il sondaggio sugli 80 (max_samples 800) non copre i 520: strict solo sulla val intera.
  local strict=1
  [ "$config" = "$PROBE_CONFIG" ] && strict=0
  env "${COMMON_ENV[@]}" TSLM_EVAL_STRICT="$strict" "${intervention[@]}" \
    TSLM_V3_COLLECT_CONTEXT_STATS=1 TSLM_EVAL_TAG="${tag}_posthoc" \
    "$PY" train.py --config "$config" --no-wandb --eval-only \
    --checkpoint "$CHECKPOINT" --split val --out-json "$output"
}

TAG="epoch${EPOCH}"
GEN_ON="$EVAL_DIR/${TAG}_generation_on.json"
GEN_OFF="$EVAL_DIR/${TAG}_generation_off.json"
GEN_ROLL="$EVAL_DIR/${TAG}_generation_roll.json"
FULL_ON="$EVAL_DIR/${TAG}_full_val_on.json"
FULL_OFF="$EVAL_DIR/${TAG}_full_val_off.json"
FULL_ROLL="$EVAL_DIR/${TAG}_full_val_roll.json"
PAIR="$EVAL_DIR/${TAG}_paired.json"

run_generation_if_missing "$GEN_ON" "$PROBE_CONFIG" on "${TAG}_on"
run_generation_if_missing "$GEN_OFF" "$PROBE_CONFIG" off "${TAG}_off"
run_generation_if_missing "$GEN_ROLL" "$PROBE_CONFIG" roll_tokens "${TAG}_roll"

TF_CANONICAL="$EVAL_DIR/${TAG}_teacher_forcing.json"
TF_JSON="$TF_CANONICAL"
if [ -e "$TF_CANONICAL" ]; then
  if ! "$PY" - "$TF_CANONICAL" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
bands = d.get('context_final_mean', {}).get('by_base_band')
assert set(d.get('conditions', {})) == {'on', 'off', 'roll_tokens'}
assert isinstance(bands, dict) and bands
PY
  then
    TF_JSON="$EVAL_DIR/${TAG}_teacher_forcing_context_eval.json"
  fi
fi
if [ ! -e "$TF_JSON" ]; then
  env "${COMMON_ENV[@]}" "$PY" "$FORWARD_AUDIT" \
    --pilot-config "$PROBE_CONFIG" --checkpoint "$CHECKPOINT" \
    --out-json "$TF_JSON" --n 80
else
  echo "KEEP $TF_JSON"
fi

run_generation_if_missing "$FULL_ON" "$TRAIN_CONFIG" on "${TAG}_full_val_on"
run_generation_if_missing "$FULL_OFF" "$TRAIN_CONFIG" off "${TAG}_full_val_off"
FULL_ROLL_ARGS=()
if [ "${V3_FULL_VAL_ROLL:-0}" = "1" ]; then
  run_generation_if_missing "$FULL_ROLL" "$TRAIN_CONFIG" roll_tokens "${TAG}_full_val_roll"
fi
if [ -e "$FULL_ROLL" ]; then
  FULL_ROLL_ARGS=(--full-roll "$FULL_ROLL")
fi

if [ ! -e "$PAIR" ]; then
  "$PY" "$PAIRED_EVAL" --out "$PAIR" \
    --generation-on "$GEN_ON" --generation-off "$GEN_OFF" \
    --generation-roll "$GEN_ROLL" --teacher-forcing "$TF_JSON" \
    --full-on "$FULL_ON" --full-off "$FULL_OFF" \
    "${FULL_ROLL_ARGS[@]}"
else
  echo "KEEP $PAIR"
fi

"$PY" - "$PAIR" "$TF_JSON" <<'PY'
import json, sys
paired, tf = [json.load(open(path, encoding='utf-8')) for path in sys.argv[1:]]
assert paired['bootstrap'] == {'n_resamples': 10000, 'seed': 20260909, 'ci': 0.95}
assert paired['generation_probe80']['n'] == paired['teacher_forcing_probe80']['n'] == 80
assert paired['full_val']['n'] == 520
bands = tf['context_final_mean']['by_base_band']
required = {'token_variable_fraction', 'mean_token_cosine',
            'group_mean_rms_over_h_rms', 'input_series_relative'}
assert bands and all(required <= set(row) for row in bands.values())
PY

printf 'POSTHOC_OK run_dir=%s epoch=%s paired=%s\n' "$RUN_DIR" "$EPOCH" "$PAIR"
