#!/usr/bin/env bash
# Zero-shot learnability probe (run between smoke c and d, GPU otherwise idle):
#   N rollouts per arm through the training harness on the pinned val, then counts how often the
#   untrained policy produces ANY final answer / a non-zero reward / a wavelet_zoom call.
# GRPO only learns from groups with reward variance: if the base policy scores 0 on ~every
# rollout, the pilot has no signal and the protocol must change BEFORE spending GPU-days.
#   node_l2_zeroshot_probe.sh [N=32] [arms="code wavelet"]
set -euo pipefail
N="${1:-32}"; ARMS="${2:-code wavelet}"
TIMERLM="${TIMERLM:-$HOME/timerlm}"; PR="$TIMERLM/prime-rl"; L2="$TIMERLM/l2"
OUT="$HOME/l2/smoke/zeroshot"; LOGS="$HOME/l2/logs"; PORT=8000; BASE_URL="http://127.0.0.1:$PORT/v1"
MODEL_NAME="/root/timerlm/l2/model_snapshot"; VENV_PY="$PR/.venv/bin/python"
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
log() { echo "[zeroshot $(date -u +%H:%M:%S)] $*"; }
server_up() { curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }
if ! server_up; then
  tmux kill-session -t l2srv 2>/dev/null || true
  tmux new-session -d -s l2srv "cd '$PR' && uv run --no-sync inference @ ../l2/configs/inference_l2.toml > '$LOGS/inference_zeroshot.log' 2>&1"
  t=0; until server_up; do sleep 5; t=$((t+5)); [ $t -ge 900 ] && { echo "server not healthy"; exit 1; }; done
  log "server healthy after ${t}s"
fi
for arm in $ARMS; do
  envs=(RLM_MAX_TOKENS=16384 RLM_MAX_OUTPUT=4096 RLM_MAX_TURNS=15 RLM_SUMMARIZE_AT_TOKENS=22000
        RLM_PYTHON="$TIMERLM/rlm-harness/.venv/bin/python" OMP_NUM_THREADS=1 TIMERLM_NOTE_VARIANT=default)
  [ "$arm" = wavelet ] && envs+=(RLM_SKILLS=wavelet_zoom TIMERLM_NOTE_VARIANT=wavelet)
  out="$OUT/${arm}_${N}.jsonl"; rm -rf "$out" "${out%.jsonl}"
  t0=$(date +%s)
  ( cd "$PR" && env "${envs[@]}" "$VENV_PY" "$TIMERLM/scripts/eval_v1_rlm.py" \
      --dataset anomalyxl --repo-or-path "$L2/data/val" --eval-size 100 --limit "$N" \
      --base-url "$BASE_URL" --api-key x --model "$MODEL_NAME" --concurrency 16 --out "$out" ) > "$LOGS/zeroshot_${arm}.log" 2>&1 || { tail -20 "$LOGS/zeroshot_${arm}.log"; exit 1; }
  dt=$(( $(date +%s) - t0 ))
  "$VENV_PY" - "$arm" "$out" "$dt" "$OUT/${arm}_${N}_summary.json" <<'PY'
import json, sys, glob, os, collections
arm, path, dt, outp = sys.argv[1:5]
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
ws = os.path.splitext(path)[0] + "/workspaces"
answers = 0; skill_calls = 0; repl = 0; stop = collections.Counter(); out_tok = []
for sdir in glob.glob(ws + "/*/session"):
    meta = os.path.join(sdir, "meta.json")
    if os.path.exists(meta):
        m = json.load(open(meta, encoding="utf-8")); answers += int(bool(m.get("answer_preview")))
        stop[m.get("metrics", {}).get("stop_reason")] += 1; out_tok.append(m.get("usage", {}).get("completion_tokens", 0))
    mp = os.path.join(sdir, "messages.jsonl")
    if os.path.exists(mp):
        for line in open(mp, encoding="utf-8", errors="replace"):
            try: rec = json.loads(line)
            except json.JSONDecodeError: continue
            if rec.get("type") == "assistant":
                for tc in rec.get("tool_calls") or []:
                    if tc.get("name") == "ipython":
                        repl += 1; skill_calls += int("wavelet_zoom(" in str((tc.get("args") or {}).get("code", "")))
rew = [float(r.get("reward") or 0) for r in rows]
res = {"arm": arm, "n": len(rows), "wall_s": int(dt), "seconds_per_rollout_at_conc16": (int(dt) / max(1, len(rows))),
       "reward_mean": sum(rew) / max(1, len(rew)), "reward_nonzero": sum(1 for x in rew if x > 0),
       "errors": sum(1 for r in rows if r.get("error")), "mean_turns": sum((r.get("num_turns") or 0) for r in rows) / max(1, len(rows)),
       "final_answers_nonempty": answers, "stop_reasons": dict(stop), "repl_calls": repl, "wavelet_zoom_calls": skill_calls,
       "mean_completion_tokens": (sum(out_tok) / len(out_tok)) if out_tok else None}
json.dump(res, open(outp, "w"), indent=1); print(json.dumps(res))
PY
done
tmux kill-session -t l2srv 2>/dev/null || true
log "done -> $OUT"
