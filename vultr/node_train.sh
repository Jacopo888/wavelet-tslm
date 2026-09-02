#!/bin/bash
# Fase 4: lancia i 3 training (GPU 0,1,2) + 3 eval worker (GPU 3,4,5) in tmux.
set -e
cd ~/timerlm/sft/tslm

# GRAD_CKPT=1 SEMPRE (lezione m30: senza, le celle 16ch = 4096 token percettivi
# falliscono in OOM anche col fallback, e il run perde il 12% dei dati).
# REQUIRE_FULL_COVERAGE=1: qualunque campione fallito ABORTISCE il run.
COMMON_ENV="TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16 \
TSLM_ATTN_IMPL=sdpa TSLM_GRAD_CKPT=1 TSLM_CKPT_FALLBACK=1 \
TSLM_REQUIRE_FULL_COVERAGE=1 TSLM_SAVE_TRAINABLE_ONLY=1 \
TSLM_SAVE_OPT_STATE=1 TSLM_VAL_GEN=0 TSLM_KEEP_EPOCH_CKPTS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1"

for i in 1 2 3; do
  gpu=$((i-1))
  tmux new-session -d -s va$i \
    "cd ~/timerlm/sft/tslm && env $COMMON_ENV CUDA_VISIBLE_DEVICES=$gpu \
     ~/venv/bin/python train.py --config configs/wavelet_precise_va$i.yaml \
     --no-wandb > ~/logs/va$i.log 2>&1; echo EXIT:\$? > ~/logs/va$i.done"
done

# GPU 7 riservata al peer (ordine Jacopo 01/09: "niente modal, aspetta" -> il
# patto condiviso puo' tornare attivo). Eval worker su GPU 3-6; se la 7 viene
# liberata esplicitamente, aggiungere un sesto worker a mano.
for w in 1 2 3 4; do
  gpu=$((w+2))
  tmux new-session -d -s evw$w \
    "bash ~/node_eval_worker.sh $gpu > ~/logs/evw$w.log 2>&1"
done
tmux ls
echo "TRAIN_LAUNCHED"
