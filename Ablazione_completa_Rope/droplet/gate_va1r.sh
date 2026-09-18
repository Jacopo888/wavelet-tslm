#!/usr/bin/env bash
# Gate percettivo di va1r (Astra 16/09): primary del compito con token veri contro sostituiti e azzerati.
# Uso: bash droplet/gate_va1r.sh <run-id> <step> [righe] [extra args di tools/oracle_text_probe.py]
# La soglia dichiarata prima e' quella di Z1-bis: delta >= +0,05 con CI95 che esclude lo zero.
set -Eeuo pipefail
RUN_ID="${1:?run-id}"
STEP="${2:?optimizer step, es. 4488}"
ROWS="${3:-60}"
shift 3 2>/dev/null || shift $# || true
CODE_DIR="${CODE_DIR:-/workspace/code}"
CONFIG="${CONFIG:-configs/va1r_mi300x.toml}"
CKPT="$(printf '/workspace/runs/%s/checkpoints/step_%012d' "$RUN_ID" "$STEP")"
OUT="/workspace/runs/${RUN_ID}/oracle_probe_step_$(printf '%012d' "$STEP").json"
LOG="/root/wavelet-rope/logs/gate_${RUN_ID}_${STEP}.log"

if [ ! -d "/root/wavelet-rope/runs/${RUN_ID}/checkpoints/$(printf 'step_%012d' "$STEP")" ]; then
  echo "checkpoint step ${STEP} of ${RUN_ID} does not exist yet" >&2
  exit 2
fi
docker exec -w "${CODE_DIR}" -e FLA_TILELANG=0 wrope \
  python tools/oracle_text_probe.py --config "${CONFIG}" --checkpoint "${CKPT}" \
  --out "${OUT}" --rows "${ROWS}" "$@" 2>&1 \
  | grep --line-buffered -vE 'redis|warnings.warn|Triton version' | tee -a "${LOG}" || true
echo "gate done; report ${OUT}, log ${LOG}"
