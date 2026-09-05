#!/bin/bash
# Lancio di UNA run v3 (v3tied | v3untied | v3sdepth) con il protocollo di va1
# (vultr/node_train_single.sh), parametrizzato per nome run. Coverage garantita
# (TSLM_REQUIRE_FULL_COVERAGE=1), GRAD_CKPT=1 sempre, eval strict, selezione
# best-by-val-primary + test 800 nel worker fail-closed (node_eval_worker_v3.sh).
# Su GPU >= 80GB (MI300X) piu' run in parallelo: una chiamata per run, con
# V3_GPU=<indice> (stessa GPU va bene: 3 run x ~20GB) e sessioni tmux separate.
set -e
RUN="${1:?uso: node_train_v3.sh v3tied|v3untied|v3sdepth}"
case "$RUN" in v3tied|v3untied|v3sdepth) ;; *) echo "run sconosciuta: $RUN"; exit 2 ;; esac
cd ~/timerlm/sft/tslm
[ -f ~/RUN_MANIFEST_V3.json ] || { echo "manca ~/RUN_MANIFEST_V3.json: eseguire node_env_v3.sh"; exit 3; }

# Gate anti-leakage: nessun training parte se gli split non sono puliti.
~/venv/bin/python preflight_data.py ~/timerlm/anomalyXL/datasets --variant precise

CFG=configs/wavelet_precise_${RUN}.yaml
[ -f "$CFG" ] || { echo "manca $CFG"; exit 3; }
RUN_DIR=results/anomaly_xl_precise/anomaly_xl_wavelet_${RUN}

# Stato stale SOLO di questa run (mai riusare una cartella di run; le altre
# sessioni tmux — altre run in parallelo — non si toccano).
tmux kill-session -t "$RUN" 2>/dev/null || true
tmux kill-session -t "evw_$RUN" 2>/dev/null || true
rm -f ~/logs/${RUN}.done ~/evalq/${RUN}_WORKER_FAILED ~/evalq/${RUN}_BEST_EPOCH \
      ~/evalq/${RUN}_ep*_val*.json ~/evalq/${RUN}_FINAL_test.json
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR" ~/logs ~/evalq
cp ~/RUN_MANIFEST_V3.json "$RUN_DIR/RUN_MANIFEST.json"

QWEN_REV=$(~/venv/bin/python -c "import json,os;print(json.load(open(os.path.expanduser('~/RUN_MANIFEST_V3.json')))['qwen_revision'])")
[ -n "$QWEN_REV" ]

# Full-coverage run: budget parziali VIETATI.
unset TSLM_MAX_TRAIN_SECONDS TSLM_MAX_TRAIN_STEPS TSLM_V3_DEPTH
GPU="${V3_GPU:-0}"

COMMON_ENV="TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16 \
TSLM_ATTN_IMPL=sdpa TSLM_GRAD_CKPT=1 TSLM_CKPT_FALLBACK=1 \
TSLM_REQUIRE_FULL_COVERAGE=1 TSLM_EVAL_STRICT=1 TSLM_SKIP_FINAL_TEST=1 \
TSLM_QWEN_REVISION=$QWEN_REV \
TSLM_SAVE_TRAINABLE_ONLY=1 \
TSLM_SAVE_OPT_STATE=1 TSLM_VAL_GEN=0 TSLM_KEEP_EPOCH_CKPTS=1 \
HF_HUB_DISABLE_PROGRESS_BARS=1"
# vGPU Vultr: NIENTE expandable_segments (VMM non supportata dai profili GRID).

tmux new-session -d -s "$RUN" \
  "cd ~/timerlm/sft/tslm && env $COMMON_ENV CUDA_VISIBLE_DEVICES=$GPU \
   ~/venv/bin/python train.py --config $CFG --no-wandb > ~/logs/${RUN}.log 2>&1; \
   echo EXIT:\$? > ~/logs/${RUN}.done"
tmux new-session -d -s "evw_$RUN" \
  "env RUN=$RUN V3_GPU=$GPU TSLM_QWEN_REVISION=$QWEN_REV \
   bash ~/node_eval_worker_v3.sh > ~/logs/evw_${RUN}.log 2>&1"
tmux ls
echo "TRAIN_LAUNCHED run=$RUN gpu=$GPU qwen_rev=$QWEN_REV"
