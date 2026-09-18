#!/usr/bin/env bash
# Catena post-P1 (Astra 15/09 §3 e §6): validazione ufficiale del checkpoint finale a 768 token con tie-break
# fixed-crop e formato via parser; gate percettivo §12.2 sul pannello congelato a 120 righe; riepilogo della run.
# Uso: bash droplet/post_p1.sh <run-id>   (idempotente per passo: salta i pezzi già prodotti)
set -Eeuo pipefail
RUN="${1:?run-id}"; R=/root/wavelet-rope/runs/$RUN; L=/root/wavelet-rope/logs
CONFIG="${CONFIG:-configs/p1_mi300x.toml}"  # resolved config inside the tool tree (P1-c: configs/p1c_mi300x.toml)
mkdir -p "$R"; exec 9>"$R/.post_p1.lock"; flock -n 9 || { echo "post-P1 chain already running for $RUN"; exit 0; }
X() { docker exec -w /workspace/code -e FLA_TILELANG=0 wrope "$@"; }
CK=$(ls -d $R/checkpoints/step_* 2>/dev/null | sort | tail -1); CKB=$(basename "$CK")
echo "== post-P1 on $RUN, checkpoint $CKB"
if ls $R/logs/val_full_pass*.json >/dev/null 2>&1; then
  echo "== [1/3] official validation already produced by the trainer at pass end (val_full_pass*, 520 rows, 768 tokens): skipped"
elif ! ls $R/logs/val_official_${CKB}_step*.json >/dev/null 2>&1; then
  echo "== [1/3] official validation (768 tokens, all 520 rows)"
  X python training/validate.py --config ${CONFIG} --run-id $RUN --checkpoint auto --tag official 2>&1 | grep -vE "redis|warn|Triton|UserWarning|return F|attn_output" | tee $L/post_${RUN}_validate.log | tail -3
fi
if [ ! -f $R/gate_r120/gate_report.json ]; then
  echo "== [2/3] perceptual gate, 120-row frozen panel"
  X python evaluation/perceptual_gate.py --checkpoint /workspace/runs/$RUN/checkpoints/$CKB --config ${CONFIG} --panel /workspace/runs/gate/panel_r120.json --seed 42 --out-dir /workspace/runs/$RUN/gate_r120 --max-new-tokens 768 2>&1 | grep -vE "redis|warn|Triton|UserWarning|return F|attn_output|__recompiles" | tee $L/post_${RUN}_gate.log | tail -5 || true  # the gate CLI exits 1 on FAIL: keep going
fi
echo "== [3/3] run report"
X python tools/report_run.py /workspace/runs/$RUN --train-parquet /scratch/datasets/va1_layout/anomalyXL/datasets/anomalyxl-train/anomalyxl-precise/train.parquet --out /workspace/runs/$RUN/REPORT.json 2>&1 | grep -vE "redis|warn" | tee $L/post_${RUN}_report.log
cat $R/gate_r120/gate_summary.md 2>/dev/null || true
echo "POST_P1_DONE"
