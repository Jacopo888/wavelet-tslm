#!/bin/bash
# Worker single-GPU FAIL-CLOSED (review 02/09, bloccante #4): ogni condizione
# di successo e' verificata POSITIVAMENTE; qualunque fallimento produce
# ~/evalq/WORKER_FAILED con la ragione, MAI un FINAL_DONE di comodo.
set -u
cd ~/timerlm/sft/tslm
ENV="TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16 \
TSLM_ATTN_IMPL=sdpa TSLM_SAVE_TRAINABLE_ONLY=1 TSLM_EVAL_STRICT=1 \
HF_HUB_DISABLE_PROGRESS_BARS=1"
EXPECTED_EPOCHS=5

fail() {
  echo "WORKER_FAILED: $1"
  echo "$1" > ~/evalq/WORKER_FAILED
  exit 1
}

while [ ! -f ~/logs/va1.done ]; do sleep 60; done
train_exit=$(cat ~/logs/va1.done)
[ "$train_exit" = "EXIT:0" ] || fail "training terminato con $train_exit (serve EXIT:0)"
echo "training concluso con EXIT:0, parte la selezione"

run="results/anomaly_xl_precise/anomaly_xl_wavelet_va1"
ckdir="$run/checkpoints"
n_ck=$(ls "$ckdir"/epoch_*_end.pt 2>/dev/null | wc -l)
[ "$n_ck" -eq "$EXPECTED_EPOCHS" ] || \
  fail "trovati $n_ck checkpoint epoch_*_end.pt, attesi $EXPECTED_EPOCHS"

for ck in "$ckdir"/epoch_*_end.pt; do
  ep=$(basename "$ck" | grep -oE '[0-9]+')
  out=~/evalq/va1_ep${ep}_val.json
  [ -f "$out" ] && { echo "val epoch $ep gia' presente"; continue; }
  echo "val eval epoch $ep"
  if ! env $ENV CUDA_VISIBLE_DEVICES=0 ~/venv/bin/python train.py \
      --config configs/wavelet_precise_va1.yaml --no-wandb \
      --eval-only --checkpoint "$ck" --split val --out-json "$out" \
      >> ~/logs/eval_va1_ep${ep}.log 2>&1; then
    fail "val eval epoch $ep uscita con errore (log: eval_va1_ep${ep}.log)"
  fi
  [ -f "$out" ] || fail "val eval epoch $ep: nessun out-json scritto"
done

n_val=$(ls ~/evalq/va1_ep*_val.json 2>/dev/null | wc -l)
[ "$n_val" -eq "$EXPECTED_EPOCHS" ] || \
  fail "presenti $n_val val json, attesi $EXPECTED_EPOCHS"

best=$(~/venv/bin/python - <<'EOF'
import glob
import json
import os
import sys
files = glob.glob(os.path.expanduser("~/evalq/va1_ep*_val.json"))
if len(files) != 5:
    sys.exit(f"attesi 5 val json, trovati {len(files)}")
c = []
for f in files:
    d = json.load(open(f))
    if "primary" not in d:
        sys.exit(f"{f} senza campo primary")
    c.append((d["primary"], f))
c.sort(reverse=True)
print(c[0][1].split("_ep")[1].split("_")[0])
EOF
) || fail "selezione best fallita: $best"
[ -n "$best" ] || fail "best epoch vuoto"
echo "BEST_EPOCH=$best (per val primary)"

if ! env $ENV CUDA_VISIBLE_DEVICES=0 ~/venv/bin/python train.py \
    --config configs/wavelet_precise_va1.yaml --no-wandb \
    --eval-only --checkpoint "$ckdir/epoch_${best}_end.pt" --split test \
    --out-json ~/evalq/va1_FINAL_test.json >> ~/logs/eval_va1_final.log 2>&1; then
  fail "test finale (epoch $best) uscito con errore"
fi
[ -f ~/evalq/va1_FINAL_test.json ] || fail "test finale senza out-json"
echo "BEST_EPOCH=$best" > ~/evalq/BEST_EPOCH
echo "FINAL_DONE"
