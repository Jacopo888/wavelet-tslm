#!/usr/bin/env bash
# Fase A di P1-d (Astra 15/09 D1): esegue tools/reading_probe.py su ogni checkpoint completo (100, 200, ... fino a MAX)
# della run indicata e stampa il verdetto. Si ferma al primo PASS (stampa PROBE_PASS) o dopo MAX (PROBE_NO_PASS).
# Non ferma il training: la decisione resta a chi legge.
# Uso: CODE_DIR=/workspace/code_p2 CONFIG=configs/p1dA_mi300x.toml bash droplet/probe_watch.sh <run-id> [max=500] [every=100] [rows=24]
set -Eeuo pipefail
RUN="${1:?run-id}"; MAX="${2:-500}"; EVERY="${3:-100}"; ROWS="${4:-24}"
CODE_DIR="${CODE_DIR:-/workspace/code_p2}"; CONFIG="${CONFIG:-configs/p1dA_mi300x.toml}"
R=/root/wavelet-rope/runs/$RUN; L=/root/wavelet-rope/logs
step=$EVERY
while [ "$step" -le "$MAX" ]; do
  TARGET=$(printf "step_%012d" "$step")
  echo "== waiting for $TARGET"
  until [ -f "$R/checkpoints/$TARGET/COMPLETE" ]; do
    if ! tmux has-session -t "p1-$RUN" 2>/dev/null && [ ! -f "$R/checkpoints/$TARGET/COMPLETE" ]; then
      echo "run ended before $TARGET"; echo "PROBE_NO_PASS $RUN (run ended)"; exit 2
    fi
    sleep 60
  done
  OUT=/workspace/runs/$RUN/probe_$TARGET.json
  if [ ! -f "$R/probe_$TARGET.json" ]; then
    docker exec -w "$CODE_DIR" -e FLA_TILELANG=0 wrope python tools/reading_probe.py --config "$CONFIG" \
      --checkpoint /workspace/runs/$RUN/checkpoints/$TARGET --out "$OUT" --rows "$ROWS" 2>&1 \
      | grep -vE "redis|warn|Triton|UserWarning|return F|attn_output|__recompiles|attn = F" \
      | tee "$L/probe_${RUN}_$TARGET.log" | grep -E "^\[|READING_PROBE" || true
  fi
  if grep -q '"verdict": "PASS"' "$R/probe_$TARGET.json" 2>/dev/null; then
    echo "PROBE_PASS $RUN $TARGET"; exit 0
  fi
  echo "PROBE_FAIL $RUN $TARGET"
  step=$((step + EVERY))
done
echo "PROBE_NO_PASS $RUN (reached $MAX)"
exit 1
