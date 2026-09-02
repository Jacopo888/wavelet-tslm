#!/bin/bash
# Lancio SINGLE-GPU (A40 24GB vGPU, istanza wavelet-l1): re-run L1 va1
# (seed 42) con coverage garantita; selezione upstream + test 800 partono
# in un worker che ATTENDE la fine del training (stessa GPU).
set -e
cd ~/timerlm/sft/tslm

# Gate anti-leakage: nessun training parte se gli split non sono puliti.
~/venv/bin/python preflight_data.py ~/timerlm/anomalyXL/datasets --variant precise

# GRAD_CKPT=1 SEMPRE (lezione m30) + fail-fast di copertura + eval strict.
COMMON_ENV="TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16 \
TSLM_ATTN_IMPL=sdpa TSLM_GRAD_CKPT=1 TSLM_CKPT_FALLBACK=1 \
TSLM_REQUIRE_FULL_COVERAGE=1 TSLM_EVAL_STRICT=1 TSLM_SAVE_TRAINABLE_ONLY=1 \
TSLM_SAVE_OPT_STATE=1 TSLM_VAL_GEN=0 TSLM_KEEP_EPOCH_CKPTS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1"

mkdir -p ~/logs ~/evalq
tmux new-session -d -s va1 \
  "cd ~/timerlm/sft/tslm && env $COMMON_ENV CUDA_VISIBLE_DEVICES=0 \
   ~/venv/bin/python train.py --config configs/wavelet_precise_va1.yaml \
   --no-wandb > ~/logs/va1.log 2>&1; echo EXIT:\$? > ~/logs/va1.done"
tmux new-session -d -s evw1 "bash ~/node_eval_worker_single.sh > ~/logs/evw1.log 2>&1"
tmux ls
echo "TRAIN_LAUNCHED"
