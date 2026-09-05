#!/bin/bash
# Worker FAIL-CLOSED per una run v3 (vultr/node_eval_worker_single.sh
# parametrizzato per RUN): attende ${RUN}.done con liveness, richiede EXIT:0 e
# l'insieme esatto epoch_1..5_end.pt, val eval per epoca, best-by-primary,
# test 800 a profondita' piena. Poi, se V3_DEPTH_SWEEP=1 (default per
# v3sdepth), la curva metrica-vs-profondita' SULLA VAL con il best checkpoint
# (TSLM_V3_DEPTH=r): il test resta uno solo, a profondita' piena.
# Ogni fallimento scrive ~/evalq/${RUN}_WORKER_FAILED con la ragione.
set -u
RUN="${RUN:?RUN mancante (v3tied|v3untied|v3sdepth)}"
GPU="${V3_GPU:-0}"
cd ~/timerlm/sft/tslm
CFG=configs/wavelet_precise_${RUN}.yaml
ENV="TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16 \
TSLM_ATTN_IMPL=sdpa TSLM_SAVE_TRAINABLE_ONLY=1 TSLM_EVAL_STRICT=1 \
TSLM_QWEN_REVISION=${TSLM_QWEN_REVISION:-} \
HF_HUB_DISABLE_PROGRESS_BARS=1"
EXPECTED_EPOCHS=5
MAX_WAIT_HOURS=72
SWEEP_DEFAULT=0; [ "$RUN" = "v3sdepth" ] && SWEEP_DEFAULT=1
DEPTH_SWEEP="${V3_DEPTH_SWEEP:-$SWEEP_DEFAULT}"
DEPTHS="${V3_DEPTHS:-1 2 3 4 6 8 11}"

fail() {
  echo "WORKER_FAILED: $1"
  echo "$1" > ~/evalq/${RUN}_WORKER_FAILED
  exit 1
}

waited=0
while [ ! -f ~/logs/${RUN}.done ]; do
  if ! tmux has-session -t "$RUN" 2>/dev/null; then
    sleep 30
    [ -f ~/logs/${RUN}.done ] && break
    fail "sessione tmux $RUN morta senza scrivere ${RUN}.done"
  fi
  sleep 60
  waited=$((waited + 60))
  [ "$waited" -ge $((MAX_WAIT_HOURS * 3600)) ] && fail "training oltre ${MAX_WAIT_HOURS}h senza ${RUN}.done"
done
train_exit=$(cat ~/logs/${RUN}.done)
[ "$train_exit" = "EXIT:0" ] || fail "training terminato con $train_exit (serve EXIT:0)"
echo "training $RUN concluso con EXIT:0, parte la selezione"

run="results/anomaly_xl_precise/anomaly_xl_wavelet_${RUN}"
ckdir="$run/checkpoints"
for ep in $(seq 1 $EXPECTED_EPOCHS); do
  [ -f "$ckdir/epoch_${ep}_end.pt" ] || fail "manca il checkpoint epoch_${ep}_end.pt"
done
n_ck=$(ls "$ckdir"/epoch_*_end.pt 2>/dev/null | wc -l)
[ "$n_ck" -eq "$EXPECTED_EPOCHS" ] || fail "trovati $n_ck checkpoint, attesi $EXPECTED_EPOCHS"

# Coverage per epoca DAL LOG (regola d'oro): 5 righe trained 4488 / failed 0.
n_cov=$(grep -c "\[Coverage\].*trained 4488" ~/logs/${RUN}.log || true)
n_fail=$(grep -c "\[Coverage\].*failed [1-9]" ~/logs/${RUN}.log || true)
[ "$n_cov" -ge "$EXPECTED_EPOCHS" ] || fail "coverage: solo $n_cov epoche con trained 4488"
[ "$n_fail" -eq 0 ] || fail "coverage: $n_fail righe con failed>0"

unset TSLM_V3_DEPTH
for ck in "$ckdir"/epoch_*_end.pt; do
  ep=$(basename "$ck" | grep -oE '[0-9]+')
  out=~/evalq/${RUN}_ep${ep}_val.json
  [ -f "$out" ] && { echo "val epoch $ep gia' presente"; continue; }
  echo "val eval epoch $ep"
  env $ENV CUDA_VISIBLE_DEVICES=$GPU ~/venv/bin/python train.py --config $CFG --no-wandb \
      --eval-only --checkpoint "$ck" --split val --out-json "$out" \
      >> ~/logs/eval_${RUN}_ep${ep}.log 2>&1 || fail "val eval epoch $ep fallita (log eval_${RUN}_ep${ep}.log)"
  [ -f "$out" ] || fail "val eval epoch $ep: nessun out-json"
done
n_val=$(ls ~/evalq/${RUN}_ep*_val.json 2>/dev/null | wc -l)
[ "$n_val" -eq "$EXPECTED_EPOCHS" ] || fail "presenti $n_val val json, attesi $EXPECTED_EPOCHS"

best=$(RUN=$RUN ~/venv/bin/python - <<'EOF'
import glob, json, os, sys
run = os.environ["RUN"]
files = glob.glob(os.path.expanduser(f"~/evalq/{run}_ep*_val.json"))
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

env $ENV CUDA_VISIBLE_DEVICES=$GPU ~/venv/bin/python train.py --config $CFG --no-wandb \
    --eval-only --checkpoint "$ckdir/epoch_${best}_end.pt" --split test \
    --out-json ~/evalq/${RUN}_FINAL_test.json >> ~/logs/eval_${RUN}_final.log 2>&1 \
  || fail "test finale (epoch $best) fallito"
[ -f ~/evalq/${RUN}_FINAL_test.json ] || fail "test finale senza out-json"
echo "BEST_EPOCH=$best" > ~/evalq/${RUN}_BEST_EPOCH
echo "FINAL_DONE"

if [ "$DEPTH_SWEEP" = "1" ]; then
  # Curva metrica-vs-profondita' sulla VAL (best checkpoint): r iterazioni del
  # loop (r=1 = solo banda grossolana; r>=steps = profondita' piena).
  for r in $DEPTHS; do
    out=~/evalq/${RUN}_ep${best}_val_depth${r}.json
    [ -f "$out" ] && continue
    echo "depth sweep r=$r"
    env $ENV TSLM_V3_DEPTH=$r CUDA_VISIBLE_DEVICES=$GPU ~/venv/bin/python train.py --config $CFG --no-wandb \
        --eval-only --checkpoint "$ckdir/epoch_${best}_end.pt" --split val --out-json "$out" \
        >> ~/logs/eval_${RUN}_depth${r}.log 2>&1 || fail "depth sweep r=$r fallito"
    [ -f "$out" ] || fail "depth sweep r=$r senza out-json"
  done
  echo "DEPTH_SWEEP_DONE"
fi
