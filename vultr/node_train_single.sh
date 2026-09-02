#!/bin/bash
# Lancio SINGLE-GPU (A40 24GB vGPU, istanza wavelet-l1): re-run L1 va1
# (seed 42) con coverage garantita; selezione upstream + test 800 nel worker
# fail-closed che attende la fine del training (stessa GPU).
set -e
cd ~/timerlm/sft/tslm

# Gate anti-leakage: nessun training parte se gli split non sono puliti.
~/venv/bin/python preflight_data.py ~/timerlm/anomalyXL/datasets --variant precise

RUN_DIR=results/anomaly_xl_precise/anomaly_xl_wavelet_va1

# Stato stale (review 02/09, 2° giro): NON solo marker ed evalq — anche la
# RUN DIR. Checkpoint vecchi farebbero auto-resume da latest_checkpoint
# (start_epoch=6, zero epoche eseguite) e il worker selezionerebbe su
# artefatti vecchi dichiarando un falso successo.
tmux kill-server 2>/dev/null || true
rm -f ~/logs/va1.done ~/evalq/WORKER_FAILED ~/evalq/BEST_EPOCH \
      ~/evalq/va1_ep*_val.json ~/evalq/va1_FINAL_test.json
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR" ~/logs ~/evalq

# Manifest di run INSIEME agli artefatti (checkpoint slim ricostruibili).
cp ~/RUN_MANIFEST.json "$RUN_DIR/RUN_MANIFEST.json"

# Revision Qwen pinnata = quella registrata nel manifest (e ora davvero
# passata a model+tokenizer via TSLM_QWEN_REVISION).
QWEN_REV=$(~/venv/bin/python -c "import json,os;print(json.load(open(os.path.expanduser('~/RUN_MANIFEST.json')))['qwen_revision'])")
[ -n "$QWEN_REV" ]

# Full-coverage run: budget parziali VIETATI (train.py abortisce comunque
# se li trova insieme a REQUIRE_FULL_COVERAGE=1).
unset TSLM_MAX_TRAIN_SECONDS TSLM_MAX_TRAIN_STEPS

# GRAD_CKPT=1 SEMPRE (lezione m30) + fail-fast di copertura + eval strict.
# SKIP_FINAL_TEST: il test interno best-by-val-loss sprecherebbe ~800
# generazioni su un modello che la selezione best-by-primary puo' scartare.
COMMON_ENV="TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16 \
TSLM_ATTN_IMPL=sdpa TSLM_GRAD_CKPT=1 TSLM_CKPT_FALLBACK=1 \
TSLM_REQUIRE_FULL_COVERAGE=1 TSLM_EVAL_STRICT=1 TSLM_SKIP_FINAL_TEST=1 \
TSLM_QWEN_REVISION=$QWEN_REV \
TSLM_SAVE_TRAINABLE_ONLY=1 \
TSLM_SAVE_OPT_STATE=1 TSLM_VAL_GEN=0 TSLM_KEEP_EPOCH_CKPTS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1"

tmux new-session -d -s va1 \
  "cd ~/timerlm/sft/tslm && env $COMMON_ENV CUDA_VISIBLE_DEVICES=0 \
   ~/venv/bin/python train.py --config configs/wavelet_precise_va1.yaml \
   --no-wandb > ~/logs/va1.log 2>&1; echo EXIT:\$? > ~/logs/va1.done"
tmux new-session -d -s evw1 \
  "env TSLM_QWEN_REVISION=$QWEN_REV bash ~/node_eval_worker_single.sh > ~/logs/evw1.log 2>&1"
tmux ls
echo "TRAIN_LAUNCHED qwen_rev=$QWEN_REV"
