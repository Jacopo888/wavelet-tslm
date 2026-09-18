#!/usr/bin/env bash
# L2 final evaluation on the pinned OFFICIAL test (800 rows) through the training-protocol
# replay (scripts/eval_v1_rlm.py: rlm harness, per-example JSONL, resumable).
#   node_l2_eval.sh <code|wavelet|base> [--skill]
#     code|wavelet : serve ~/l2/runs/<arm>/weights/step_<TOTAL> (merged HF weights of the run)
#     base         : serve the untouched Qwen3.5-2B snapshot (zero-shot reference)
#     --skill      : wavelet protocol (RLM_SKILLS=wavelet_zoom + the wavelet note sentence);
#                    default = paper protocol. For arm `wavelet` --skill is implied.
# The harness budget replays the TRAINING-TIME budget (RLM_MAX_TOKENS=16384, RLM_MAX_OUTPUT=4096,
# RLM_MAX_TURNS=15, summarize 22000) — completion/output/context budgets remain below paper eval.
# Output: ~/l2/eval/<tag>/{test800.jsonl, workspaces/, summary.json, DONE|FAILED}. DONE only with
# 800/800 rows; errors/timeouts are counted, never hidden (they score 0 like the protocol says).
set -euo pipefail
TARGET="${1:-}"; SKILL=0; [ "${2:-}" = "--skill" ] && SKILL=1
case "$TARGET" in code|wavelet|base) ;; *) echo "usage: $0 <code|wavelet|base> [--skill]"; exit 2;; esac
[ "$TARGET" = wavelet ] && SKILL=1
TIMERLM="${TIMERLM:-$HOME/timerlm}"; PR="$TIMERLM/prime-rl"; L2="$TIMERLM/l2"
KIT="${L2_KIT:-$HOME/l2/kit/l2}"; LOGS="$HOME/l2/logs"; TOTAL_STEPS="${L2_TOTAL_STEPS:-60}"
TAG="${TARGET}$([ $SKILL = 1 ] && echo "-skill" || echo "-plain")"; OUT="$HOME/l2/eval/$TAG"
VENV_PY="$PR/.venv/bin/python"; PORT="${L2_EVAL_PORT:-8010}"; BASE_URL="http://127.0.0.1:$PORT/v1"
WALL_H="${L2_EVAL_WALL_H:-30}"; CONC="${L2_EVAL_CONCURRENCY:-16}"
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
# vGPU: no CUDA memory pools -> NCCL cannot init; single-rank FSDP2 on gloo (see overlay/prime-rl)
export PRIME_RL_DIST_BACKEND="cpu:gloo,cuda:gloo"
# vGPU: no CUDA VMM -> the launcher must not set expandable_segments for the trainer (overlay/prime-rl rl.py)
export PRIME_RL_TRAINER_ALLOC_CONF=""
# flash-linear-attention routes GatedDeltaNet chunk kernels to TileLang (JIT via nvcc, absent here): force the Triton path
export FLA_TILELANG=0
# torch._grouped_mm backward asserts 16-byte strides here: plain per-adapter matmul (overlay/prime-rl multi_linear.py)
export PRIME_RL_LORA_GROUPED_MM=0
mkdir -p "$OUT" "$LOGS"
log() { echo "[l2-eval/$TAG $(date -u +%H:%M:%S)] $*" | tee -a "$OUT/driver.log"; }
fail() { log "FAILED: $*"; echo "$(date -u +%FT%TZ) $*" > "$OUT/FAILED"; tmux kill-session -t l2evalsrv 2>/dev/null || true; exit 1; }
[ -e "$OUT/DONE" ] && { log "already DONE"; exit 0; }
rm -f "$OUT/FAILED"

# ------------------------------------------------------------------ model dir to serve
SNAP="$L2/model_snapshot"
if [ "$TARGET" = base ]; then
  MODEL_DIR="$SNAP"
else
  RUN="$HOME/l2/runs/$TARGET"; [ -f "$RUN/DONE" ] || fail "run $RUN has no DONE"
  W="$RUN/weights/step_$TOTAL_STEPS"; [ -d "$W" ] || fail "no weights at $W"
  MODEL_DIR="$OUT/model"; rm -rf "$MODEL_DIR"; mkdir -p "$MODEL_DIR"
  # merged weights from the run + config/tokenizer from the pinned snapshot when the run dir
  # lacks them (prime-rl writes weights for /update_weights, not always a full HF folder)
  for f in "$SNAP"/*.json "$SNAP"/tokenizer* "$SNAP"/merges.txt "$SNAP"/vocab.json "$SNAP"/*.jinja; do [ -e "$f" ] && ln -sf "$f" "$MODEL_DIR/"; done
  for f in "$W"/*; do ln -sfn "$f" "$MODEL_DIR/$(basename "$f")"; done   # run files win
  # A merged checkpoint is one monolithic file. A stale shard index copied from the
  # base snapshot makes vLLM ignore it and look for base-model shard filenames.
  [ -f "$W/model.safetensors" ] && rm -f "$MODEL_DIR/model.safetensors.index.json"
  ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || fail "no safetensors in $W"
fi
log "serving $MODEL_DIR"

# ------------------------------------------------------------------ server (no LoRA needed: merged)
"$VENV_PY" - "$L2/configs/inference_l2.toml" "$OUT/inference_eval.toml" "$MODEL_DIR" "$PORT" "$OUT/runs" <<'PY'
import sys, tomllib, tomli_w
src, dst, model, port, outdir = sys.argv[1:6]
c = tomllib.load(open(src, "rb"))
c["model"]["name"] = model; c["server"]["port"] = int(port); c["enable_lora"] = False
c["output_dir"] = outdir; c.pop("max_lora_rank", None); c.pop("max_loras", None); c.pop("max_cpu_loras", None)
c["gpu_memory_utilization"] = 0.85   # the GPU is free during eval: more KV = more concurrency
tomli_w.dump(c, open(dst, "wb")); print("eval server config ->", dst)
PY
server_up() { curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }
tmux kill-session -t l2evalsrv 2>/dev/null || true
tmux new-session -d -s l2evalsrv "cd '$PR' && uv run --no-sync inference @ '$OUT/inference_eval.toml' > '$OUT/inference.log' 2>&1; echo EXIT:\$? >> '$OUT/inference.log'"
t=0; until server_up; do sleep 5; t=$((t+5)); grep -q "EXIT:" "$OUT/inference.log" 2>/dev/null && { tail -30 "$OUT/inference.log"; fail "eval server died"; }; [ $t -ge 900 ] && fail "eval server not healthy after 900s"; done
curl -sf "$BASE_URL/models" | grep -q "$MODEL_DIR" || fail "server does not list $MODEL_DIR"
log "server healthy after ${t}s"

# ------------------------------------------------------------------ rollouts (resumable)
ENVS=(RLM_MAX_TOKENS=16384 RLM_MAX_OUTPUT=4096 RLM_MAX_TURNS=15 RLM_SUMMARIZE_AT_TOKENS=22000
      RLM_PYTHON="$TIMERLM/rlm-harness/.venv/bin/python" OMP_NUM_THREADS=1 TIMERLM_NOTE_VARIANT=default)
[ $SKILL = 1 ] && ENVS+=(RLM_SKILLS=wavelet_zoom TIMERLM_NOTE_VARIANT=wavelet)
log "protocol env: ${ENVS[*]}"
( cd "$PR" && timeout "$((WALL_H*3600))" env "${ENVS[@]}" "$VENV_PY" "$TIMERLM/scripts/eval_v1_rlm.py" \
    --dataset anomalyxl --repo-or-path "$L2/data/test" --eval-size 800 \
    --base-url "$BASE_URL" --api-key x --model "$MODEL_DIR" --concurrency "$CONC" \
    --out "$OUT/test800.jsonl" ) > "$OUT/rollouts.log" 2>&1 || { rc=$?; tail -30 "$OUT/rollouts.log"; fail "eval rollouts exited $rc (timeout ${WALL_H}h or crash)"; }
tmux kill-session -t l2evalsrv 2>/dev/null || true

# ------------------------------------------------------------------ summary (800/800 or FAIL)
"$VENV_PY" - "$OUT/test800.jsonl" "$L2/data/test/data.parquet" "$OUT/summary.json" "$TAG" "$MODEL_DIR" <<'PY'
import json, sys, collections, glob, os
import pyarrow.parquet as pq
path, parquet, out, tag, model = sys.argv[1:6]
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
by_id = {}
for r in rows: by_id[r["example_id"]] = r          # last write wins (resume appends)
cats = pq.read_table(parquet, columns=["category"]).column("category").to_pylist()
n_expected = len(cats)
per = collections.defaultdict(list); errs = 0; timeouts = 0; turns = []
for i, cat in enumerate(cats):
    r = by_id.get(i)
    if r is None: continue
    per[cat].append(float(r.get("reward") or 0.0))
    if r.get("error"): errs += 1
    if r.get("error") and "timeout" in str(r["error"]): timeouts += 1
    if r.get("num_turns") is not None: turns.append(r["num_turns"])
cat_primary = {c: sum(v)/len(v) for c, v in per.items()}
n = sum(len(v) for v in per.values())
skill_calls = 0
for f in glob.glob(os.path.splitext(path)[0] + "/workspaces/*/session/messages.jsonl"):
    skill_calls += open(f, encoding="utf-8", errors="replace").read().count("wavelet_zoom(")
summary = {"tag": tag, "model": model, "n_rows": n, "n_expected": n_expected, "unique_ids": len(by_id),
           "primary": (sum(cat_primary.values())/len(cat_primary)) if cat_primary else None,
           "category_primary": cat_primary, "category_n": {c: len(v) for c, v in per.items()},
           "n_errors": errs, "n_timeouts": timeouts, "mean_turns": (sum(turns)/len(turns)) if turns else None,
           "wavelet_zoom_calls_total": skill_calls, "complete": n == n_expected}
json.dump(summary, open(out, "w"), indent=1); print(json.dumps(summary))
assert n == n_expected, f"only {n}/{n_expected} rows scored"
PY
echo "EXIT:0 $(date -u +%FT%TZ)" > "$OUT/DONE"
log "DONE -> $OUT/summary.json"
