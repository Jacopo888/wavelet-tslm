#!/usr/bin/env bash
# Lancia (o riprende) il training P1 nel container `wrope` dentro tmux, con log su /root/wavelet-rope/logs.
# Uso: [CODE_DIR=/workspace/code_p2] bash droplet/run_p1.sh <run-id> [extra args di training/train.py]
# Riprendere = rilanciare lo stesso run-id (resume auto dall'ultimo checkpoint completo).
set -Eeuo pipefail
RUN_ID="${1:?run-id}"; shift || true
SESSION="p1-${RUN_ID}"
CODE_DIR="${CODE_DIR:-/workspace/code}"  # container path of the tool tree (P1-b: /workspace/code_p2)
CONFIG="${CONFIG:-configs/p1_mi300x.toml}"  # resolved config inside the tool tree (P1-c: configs/p1c_mi300x.toml)
LOG="/root/wavelet-rope/logs/train_${RUN_ID}.log"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "tmux session $SESSION already running"; exit 1; fi
tmux new-session -d -s "$SESSION" "docker exec -w ${CODE_DIR} -e FLA_TILELANG=0 wrope python training/train.py --config ${CONFIG} --run-id ${RUN_ID} --resume auto $* 2>&1 | grep --line-buffered -vE 'redis|warnings.warn|Triton version' | tee -a ${LOG}"
echo "started tmux session $SESSION; log $LOG"
