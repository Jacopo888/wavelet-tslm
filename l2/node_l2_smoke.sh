#!/usr/bin/env bash
# Level 2 smoke gate on the node. Stages (run all, or a subset: `node_l2_smoke.sh a b`):
#   a  policy server up (uv run inference @ inference_l2.toml), /health + /v1/models, and one
#      chat completion WITH a parsed tool call (qwen3_coder parser).
#   b  two real rollouts per arm through the exact training harness program (rlm_program.py via
#      scripts/eval_v1_rlm.py on the pinned local val): code arm must show REPL turns; wavelet arm
#      must show `wavelet_zoom(` in the session messages.
#   c  `rl --max-steps 2` at a tiny batch (smoke overlay) with a 1 s nvidia-smi sampler:
#      asserts the run completes, checkpoints step_2, and the GPU peak stays under L2_PEAK_MIB.
#   d  `rl --max-steps 2` at the REAL batch (64 rollouts/step): measures seconds/step and
#      rollouts/errors so the driver can project hours per arm.
# Every stage writes ~/l2/smoke/<stage>.json and prints PASS/FAIL. The machine is left clean
# (server killed, no GPU processes). Output dirs are wiped before use (no stale resume).
set -euo pipefail
TIMERLM="${TIMERLM:-$HOME/timerlm}"; PR="$TIMERLM/prime-rl"; L2="$TIMERLM/l2"
KIT="${L2_KIT:-$HOME/l2/kit/l2}"          # configs live in $KIT/configs
SMOKE="$HOME/l2/smoke"; LOGS="$HOME/l2/logs"
MANIFEST="$HOME/L2_ENV_MANIFEST.json"
PORT=8000; BASE_URL="http://127.0.0.1:$PORT/v1"
MODEL_NAME="/root/timerlm/l2/model_snapshot"   # served name == [model].name in every config
PEAK_MIB="${L2_PEAK_MIB:-23000}"
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
# vGPU: no CUDA memory pools -> NCCL cannot init; single-rank FSDP2 on gloo (see overlay/prime-rl)
export PRIME_RL_DIST_BACKEND="cpu:gloo,cuda:gloo"
# vGPU: no CUDA VMM -> the launcher must not set expandable_segments for the trainer (overlay/prime-rl rl.py)
export PRIME_RL_TRAINER_ALLOC_CONF=""
# flash-linear-attention routes GatedDeltaNet chunk kernels to TileLang (JIT via nvcc, absent here): force the Triton path
export FLA_TILELANG=0
# smoke only: C++ frames in trainer tracebacks (kernel-level errors are otherwise anonymous)
export TORCH_SHOW_CPP_STACKTRACES=1
mkdir -p "$SMOKE" "$LOGS"
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(a b c d)

log() { echo "[l2-smoke $(date -u +%H:%M:%S)] $*"; }
fail() { echo "[l2-smoke] FAIL: $*" >&2; cleanup; exit 1; }
VENV_PY="$PR/.venv/bin/python"
[ -f "$MANIFEST" ] || fail "env manifest missing; run node_l2_env.sh first"
[ -x "$VENV_PY" ] || fail "prime-rl venv missing"
[ -f "$KIT/configs/inference_l2.toml" ] || fail "kit configs missing under $KIT/configs"
# configs must be reachable as ../l2/configs from $PR: mirror the kit configs into $L2/configs
mkdir -p "$L2/configs" && cp "$KIT"/configs/*.toml "$L2/configs/"

SAMPLER_PID=""
start_sampler() {  # $1 = csv path
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$1" 2>/dev/null &
  SAMPLER_PID=$!
}
stop_sampler() { [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true; SAMPLER_PID=""; }
peak_of() { sort -n "$1" | tail -1; }

server_up() { curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }
start_server() {
  tmux kill-session -t l2srv 2>/dev/null || true
  tmux new-session -d -s l2srv "cd '$PR' && uv run --no-sync inference @ ../l2/configs/inference_l2.toml > '$LOGS/inference_smoke.log' 2>&1; echo EXIT:\$? >> '$LOGS/inference_smoke.log'"
  local t=0
  until server_up; do
    sleep 5; t=$((t+5))
    grep -q "EXIT:" "$LOGS/inference_smoke.log" 2>/dev/null && { tail -40 "$LOGS/inference_smoke.log"; fail "inference server died at startup"; }
    [ $t -ge 900 ] && fail "inference server not healthy after 900s"
  done
  log "server healthy after ${t}s"
}
stop_server() { tmux kill-session -t l2srv 2>/dev/null || true; sleep 3; }
cleanup() {
  stop_sampler; stop_server
  pkill -f "prime_rl" 2>/dev/null || true; pkill -f "vllm" 2>/dev/null || true; sleep 2
  local n; n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  log "cleanup: $n GPU processes left"
}
trap cleanup EXIT

has_stage() { local s; for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

# ------------------------------------------------------------------ a: server + tool call
if has_stage a; then
  log "stage a: server + tool call"
  start_sampler "$SMOKE/a_mem.csv"
  start_server
  "$VENV_PY" - "$BASE_URL" "$MODEL_NAME" "$SMOKE/a.json" <<'PY'
import json, sys, urllib.request
base, model, out = sys.argv[1:4]
def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    return json.load(urllib.request.urlopen(req, timeout=600))
models = json.load(urllib.request.urlopen(base + "/models", timeout=60))
ids = [m["id"] for m in models["data"]]
assert model in ids, f"served models {ids} do not contain {model!r}"
tools = [{"type": "function", "function": {"name": "call_python_repl", "description": "Run python code in a persistent IPython kernel and return its output.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
r = post("/chat/completions", {"model": model, "tools": tools, "tool_choice": "auto", "max_tokens": 512, "temperature": 0.0,
        "messages": [{"role": "user", "content": "Use the call_python_repl tool to compute 17*23 and print it. Call the tool, do not answer directly."}]})
msg = r["choices"][0]["message"]
tc = msg.get("tool_calls") or []
res = {"served_models": ids, "tool_calls": len(tc), "finish_reason": r["choices"][0].get("finish_reason"), "usage": r.get("usage"),
       "first_call": (tc[0]["function"] if tc else None), "content_head": (msg.get("content") or "")[:300], "pass": bool(tc)}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res)[:600])
assert tc, "no parsed tool call returned (parser/tool-choice problem)"
PY
  stop_sampler
  echo "{\"server_peak_mib\": $(peak_of "$SMOKE/a_mem.csv")}" > "$SMOKE/a_mem.json"
  log "stage a PASS (server peak $(peak_of "$SMOKE/a_mem.csv") MiB)"
fi

# ------------------------------------------------------------------ b: real rollouts per arm
if has_stage b; then
  log "stage b: harness rollouts per arm"
  server_up || start_server
  run_arm() {  # $1 arm
    local arm="$1" extra=()
    local out="$SMOKE/b_${arm}_rollouts.jsonl"; rm -rf "$out" "${out%.jsonl}"
    local envs=(RLM_MAX_TOKENS=16384 RLM_MAX_OUTPUT=4096 RLM_MAX_TURNS=12 RLM_SUMMARIZE_AT_TOKENS=22000
                RLM_PYTHON="$TIMERLM/rlm-harness/.venv/bin/python" OMP_NUM_THREADS=2)
    [ "$arm" = wavelet ] && envs+=(RLM_SKILLS=wavelet_zoom TIMERLM_NOTE_VARIANT=wavelet)
    ( cd "$PR" && env "${envs[@]}" "$VENV_PY" "$TIMERLM/scripts/eval_v1_rlm.py" \
        --dataset anomalyxl --repo-or-path "$L2/data/val" --eval-size 100 --limit 2 \
        --base-url "$BASE_URL" --api-key x --model "$MODEL_NAME" --concurrency 2 --out "$out" ) \
        > "$LOGS/smoke_b_${arm}.log" 2>&1 || { tail -30 "$LOGS/smoke_b_${arm}.log"; fail "arm $arm rollouts crashed"; }
    "$VENV_PY" - "$arm" "$out" "$SMOKE/b_${arm}.json" <<'PY'
import json, sys, glob, os
arm, path, out = sys.argv[1:4]
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
ws = os.path.splitext(path)[0] + "/workspaces"
# rlm session records: {"type":"assistant","tool_calls":[{"name":"ipython","args":{"code":...}}]} and
# {"type":"tool_result","tool":"ipython","content":...}; the skill, when enabled, is materialised as
# session/wavelet_zoom.py (that file is what makes the system prompt advertise it).
repl_calls = 0; skill_calls = 0; final_answers = 0; skill_files = 0; sessions = 0
for sdir in glob.glob(ws + "/*/session"):
    sessions += 1
    skill_files += int(os.path.exists(os.path.join(sdir, "wavelet_zoom.py")))
    mp = os.path.join(sdir, "messages.jsonl")
    if os.path.exists(mp):
        for line in open(mp, encoding="utf-8", errors="replace"):
            try: rec = json.loads(line)
            except json.JSONDecodeError: continue
            if rec.get("type") == "assistant":
                for tc in rec.get("tool_calls") or []:
                    if tc.get("name") == "ipython":
                        repl_calls += 1
                        if "wavelet_zoom(" in str((tc.get("args") or {}).get("code", "")): skill_calls += 1
    meta = os.path.join(sdir, "meta.json")
    if os.path.exists(meta):
        final_answers += int(bool(json.load(open(meta, encoding="utf-8")).get("answer_preview")))
res = {"arm": arm, "n": len(rows), "errors": [r.get("error") for r in rows], "turns": [r.get("num_turns") for r in rows],
       "rewards": [r.get("reward") for r in rows], "seconds": [r.get("seconds") for r in rows],
       "sessions": sessions, "repl_calls": repl_calls, "wavelet_zoom_calls": skill_calls,
       "skill_enabled_sessions": skill_files, "final_answers_nonempty": final_answers}
res["pass"] = (all(e is None for e in res["errors"]) and all((t or 0) >= 1 for t in res["turns"])
               and sessions == len(rows) and repl_calls >= 1)
if arm == "wavelet":
    res["pass"] = res["pass"] and skill_files == sessions
    res["note"] = "wavelet_zoom_calls==0 is allowed at zero-shot; the skill must be enabled in every session"
else:
    res["pass"] = res["pass"] and skill_files == 0
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res))
assert res["pass"], "stage b failed for arm " + arm
PY
  }
  run_arm code
  run_arm wavelet
  log "stage b PASS"
fi

# ------------------------------------------------------------------ c: rl 2 steps, tiny batch, memory peak
rl_run() {  # $1 name, $2 arm, $3 extra args...
  local name="$1" arm="$2"; shift 2
  local outdir="$HOME/l2/smoke/$name"; rm -rf "$outdir"
  server_up || start_server
  start_sampler "$SMOKE/${name}_mem.csv"
  local t0=$(date +%s)
  # bounded: a 2-step run that does not finish within L2_RL_SMOKE_WALL_S (default 90 min) is a
  # failure (e.g. every rollout erroring so the batch never fills), not something to wait on.
  ( cd "$PR" && timeout "${L2_RL_SMOKE_WALL_S:-5400}" uv run --no-sync rl @ "../l2/configs/l2_pilot_${arm}.toml" "$@" --output-dir "$outdir" ) \
      > "$LOGS/smoke_${name}.log" 2>&1 || { stop_sampler; tail -60 "$LOGS/smoke_${name}.log"; grep -aE "Rollout failed|WARNING|ERROR" "$outdir/logs/orchestrator.log" 2>/dev/null | tail -5 | cut -c1-300; fail "rl run $name exited non-zero or exceeded the wall limit"; }
  local dt=$(( $(date +%s) - t0 ))
  stop_sampler
  [ -d "$outdir/checkpoints/step_2" ] || { ls -la "$outdir" "$outdir/checkpoints" 2>/dev/null; fail "$name: no checkpoints/step_2"; }
  "$VENV_PY" "$KIT/l2_monitor.py" "$outdir" --jsonl "$SMOKE/${name}_coverage.jsonl" || fail "$name: monitor reported violations"
  "$VENV_PY" - "$outdir" "$SMOKE/${name}_mem.csv" "$dt" "$PEAK_MIB" "$SMOKE/${name}.json" <<'PY'
import json, sys, statistics
outdir, memcsv, dt, peak_lim, out = sys.argv[1:6]
mem = [int(x) for x in open(memcsv).read().split() if x.strip().isdigit()]
rows = [json.loads(l) for l in open(outdir + "/orchestrator_metrics.jsonl", encoding="utf-8") if l.strip()]
steps = {r["step"]: r for r in rows if "step" in r}
st = [steps[s].get("time/step") for s in sorted(steps)]
roll = [steps[s].get("progress/rollouts") for s in sorted(steps)]
err = [steps[s].get("dispatcher/errored/train") for s in sorted(steps)]
res = {"wall_s": int(dt), "steps": sorted(steps), "time_per_step_s": st, "rollouts_per_step": roll, "errored_train": err,
       "gpu_peak_mib": max(mem) if mem else None, "gpu_peak_limit_mib": int(peak_lim),
       "pass": bool(steps) and len(steps) >= 2 and (max(mem) if mem else 10**9) < int(peak_lim)}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res))
assert res["pass"], "stage failed: " + json.dumps(res)
PY
}
if has_stage c; then
  log "stage c: rl 2 steps tiny batch (memory gate)"
  rl_run c_rl_tiny code @ ../l2/configs/l2_smoke.toml
  log "stage c PASS"
fi

# ------------------------------------------------------------------ d: rl 2 steps, real batch (timing)
if has_stage d; then
  log "stage d: rl 2 steps real batch (timing)"
  rl_run d_rl_full code --max-steps 2 --ckpt.interval 2 --orchestrator.eval.interval 1000
  "$VENV_PY" - "$SMOKE/d_rl_full.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
ts = [t for t in r["time_per_step_s"] if t]
per = max(ts) if ts else None
print(json.dumps({"projected_hours_per_arm_60_steps": (per * 60 / 3600) if per else None, "seconds_per_step_max": per}))
PY
  log "stage d PASS"
fi
log "ALL REQUESTED STAGES PASSED: ${STAGES[*]}"
# SMOKE_OK accumulates the stages that passed (stages may be run in separate invocations);
# the run driver requires a, b and c to be recorded and the file to be newer than the env manifest.
prev=""; [ -f "$SMOKE/SMOKE_OK" ] && prev="$(sed 's/^SMOKE_OK//' "$SMOKE/SMOKE_OK")"
all="$(printf '%s\n' $prev "${STAGES[@]}" | grep -E '^[abcd]$' | sort -u | tr '\n' ' ')"
echo "SMOKE_OK ${all% }" > "$SMOKE/SMOKE_OK"
log "SMOKE_OK now records: ${all% }"
