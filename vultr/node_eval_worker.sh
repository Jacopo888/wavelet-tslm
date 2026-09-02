#!/bin/bash
# Eval worker su una GPU dedicata (arg 1 = indice GPU).
# Consuma: checkpoint epoch_*_end.pt dei run va1..va3 -> eval val con
# generazioni (selezione upstream best-by-primary). Quando un seed ha tutte e
# 5 le epoche valutate -> sceglie il best e lancia l'eval finale test (800).
# Lock via mkdir (atomico su fs locale). Esce quando tutti i seed hanno FINAL.
set -u
GPU=$1
cd ~/timerlm/sft/tslm
ENV="TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16 \
TSLM_ATTN_IMPL=sdpa TSLM_SAVE_TRAINABLE_ONLY=1 HF_HUB_DISABLE_PROGRESS_BARS=1"

while true; do
  progress=0
  alldone=1
  for i in 1 2 3; do
    run="results/anomaly_xl_precise/anomaly_xl_wavelet_va$i"
    ckdir="$run/checkpoints"
    [ -d "$ckdir" ] || { alldone=0; continue; }

    # 1) val eval per ogni checkpoint di fine epoca non ancora valutato
    for ck in "$ckdir"/epoch_*_end.pt; do
      [ -e "$ck" ] || continue
      ep=$(basename "$ck" | grep -oE '[0-9]+')
      out=~/evalq/va${i}_ep${ep}_val.json
      lock=~/evalq/va${i}_ep${ep}.lock
      [ -f "$out" ] && continue
      mkdir "$lock" 2>/dev/null || continue
      echo "[gpu$GPU] val eval va$i epoch $ep"
      env $ENV CUDA_VISIBLE_DEVICES=$GPU ~/venv/bin/python train.py \
        --config configs/wavelet_precise_va$i.yaml --no-wandb \
        --eval-only --checkpoint "$ck" --split val --out-json "$out" \
        >> ~/logs/eval_va${i}_ep${ep}.log 2>&1 || rm -rf "$lock"
      progress=1
    done

    # 2) se il training e' finito e le 5 val ci sono -> best + test finale
    if [ -f ~/logs/va$i.done ] && [ $(ls ~/evalq/va${i}_ep*_val.json 2>/dev/null | wc -l) -ge 5 ]; then
      final=~/evalq/va${i}_FINAL_test.json
      flock=~/evalq/va${i}_FINAL.lock
      if [ ! -f "$final" ] && mkdir "$flock" 2>/dev/null; then
        best=$(~/venv/bin/python - <<EOF
import json, glob
c = []
for f in glob.glob("$HOME/evalq/va${i}_ep*_val.json"):
    d = json.load(open(f))
    c.append((d.get("primary", 0.0), f))
c.sort(reverse=True)
print(c[0][1].split("_ep")[1].split("_")[0])
EOF
)
        echo "[gpu$GPU] FINAL test va$i (best epoch $best)"
        env $ENV CUDA_VISIBLE_DEVICES=$GPU ~/venv/bin/python train.py \
          --config configs/wavelet_precise_va$i.yaml --no-wandb \
          --eval-only --checkpoint "$ckdir/epoch_${best}_end.pt" --split test \
          --out-json "$final" >> ~/logs/eval_va${i}_final.log 2>&1 || rm -rf "$flock"
        progress=1
      fi
    fi
    [ -f ~/evalq/va${i}_FINAL_test.json ] || alldone=0
  done
  [ "$alldone" = "1" ] && { echo "[gpu$GPU] tutti i FINAL presenti, esco"; break; }
  [ "$progress" = "0" ] && sleep 120
done
