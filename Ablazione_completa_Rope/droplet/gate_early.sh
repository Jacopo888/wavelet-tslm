#!/usr/bin/env bash
# Gate percettivo precoce (Astra 15/09, sezione scala): attende il primo checkpoint COMPLETO del passo indicato,
# esegue il gate §12.2 sul pannello congelato ridotto (r20, 60 crop) e stampa il verdetto. Non ferma la run da solo:
# la decisione di abort resta a chi legge il verdetto.
# Uso: [CODE_DIR=/workspace/code_p2] bash droplet/gate_early.sh <run-id> [step=400] [panel=/workspace/runs/gate/panel_r20.json]
set -Eeuo pipefail
RUN="${1:?run-id}"; STEP="${2:-400}"; PANEL="${3:-/workspace/runs/gate/panel_r20.json}"
CODE_DIR="${CODE_DIR:-/workspace/code}"  # container path of the tool tree that trained the run (P1-b: /workspace/code_p2)
CONFIG="${CONFIG:-configs/p1_mi300x.toml}"  # resolved config inside the tool tree (P1-c: configs/p1c_mi300x.toml)
R=/root/wavelet-rope/runs/$RUN; L=/root/wavelet-rope/logs
TARGET=$(printf "step_%012d" "$STEP")
echo "== waiting for $R/checkpoints/$TARGET/COMPLETE"
until [ -f "$R/checkpoints/$TARGET/COMPLETE" ]; do
  if [ -f "$R/DONE" ] || ! tmux has-session -t "p1-$RUN" 2>/dev/null; then
    if [ ! -f "$R/checkpoints/$TARGET/COMPLETE" ]; then echo "run ended before $TARGET"; exit 2; fi
  fi
  sleep 60
done
OUT=/workspace/runs/$RUN/gate_r20_$TARGET
if [ ! -f "$R/gate_r20_$TARGET/gate_report.json" ]; then
  echo "== gate r20 on $TARGET"
  docker exec -w ${CODE_DIR} -e FLA_TILELANG=0 wrope python evaluation/perceptual_gate.py \
    --checkpoint /workspace/runs/$RUN/checkpoints/$TARGET --config ${CONFIG} \
    --panel "$PANEL" --seed 42 --out-dir "$OUT" --max-new-tokens 768 2>&1 \
    | grep -vE "redis|warn|Triton|UserWarning|return F|attn_output|__recompiles" \
    | tee "$L/gate_early_${RUN}_$TARGET.log" | tail -5 || true  # the gate CLI exits 1 on FAIL: keep going
fi
cat "$R/gate_r20_$TARGET/gate_summary.md" 2>/dev/null || true
echo "GATE_EARLY_DONE $RUN $TARGET"
