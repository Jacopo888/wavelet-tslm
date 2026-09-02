#!/bin/bash
# Worker single-GPU: attende la fine del training va1 (stessa GPU 0), poi
# val con generazioni su ogni epoch_*_end.pt -> best per PRIMARIA -> test 800.
# Eval strict: un esempio mancante abortisce (TSLM_EVAL_STRICT=1).
set -u
cd ~/timerlm/sft/tslm
ENV="TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16 \
TSLM_ATTN_IMPL=sdpa TSLM_SAVE_TRAINABLE_ONLY=1 TSLM_EVAL_STRICT=1 \
HF_HUB_DISABLE_PROGRESS_BARS=1"

while [ ! -f ~/logs/va1.done ]; do sleep 60; done
echo "training concluso ($(cat ~/logs/va1.done)), parte la selezione"

run="results/anomaly_xl_precise/anomaly_xl_wavelet_va1"
ckdir="$run/checkpoints"
for ck in "$ckdir"/epoch_*_end.pt; do
  [ -e "$ck" ] || continue
  ep=$(basename "$ck" | grep -oE '[0-9]+')
  out=~/evalq/va1_ep${ep}_val.json
  [ -f "$out" ] && continue
  echo "val eval epoch $ep"
  env $ENV CUDA_VISIBLE_DEVICES=0 ~/venv/bin/python train.py \
    --config configs/wavelet_precise_va1.yaml --no-wandb \
    --eval-only --checkpoint "$ck" --split val --out-json "$out" \
    >> ~/logs/eval_va1_ep${ep}.log 2>&1
done

best=$(~/venv/bin/python - <<'EOF'
import glob
import json
import os
c = []
for f in glob.glob(os.path.expanduser("~/evalq/va1_ep*_val.json")):
    d = json.load(open(f))
    c.append((d.get("primary", 0.0), f))
c.sort(reverse=True)
print(c[0][1].split("_ep")[1].split("_")[0])
EOF
)
echo "BEST_EPOCH=$best (per val primary)"
env $ENV CUDA_VISIBLE_DEVICES=0 ~/venv/bin/python train.py \
  --config configs/wavelet_precise_va1.yaml --no-wandb \
  --eval-only --checkpoint "$ckdir/epoch_${best}_end.pt" --split test \
  --out-json ~/evalq/va1_FINAL_test.json >> ~/logs/eval_va1_final.log 2>&1
echo "FINAL_DONE"
