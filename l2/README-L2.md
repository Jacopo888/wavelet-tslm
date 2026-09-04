# l2/ — kit Livello 2 (GRPO agentico, prime-rl) per il nodo Vultr `wavelet-l1`

Protocollo, deviazioni, pin e rischi: `PROTOCOL-L2.md`. File:

- `node_l2_env.sh` — ambiente: prime-rl al pin `c5afcaa2e` (venv proprio in `~/timerlm/prime-rl/.venv`), pacchetti TimeRLM editable, venv rlm-harness, overlay wavelet (`overlay/timerlm/`), dati pinnati come cartelle `data.parquet`, snapshot Qwen3.5-2B a revisione fissa, asserzioni GPU/stack, `~/L2_ENV_MANIFEST.json`. Mai tocca `~/venv` (L1).
- `configs/` — `inference_l2.toml` (server standalone), `l2_pilot_code.toml`, `l2_pilot_wavelet.toml` (identici salvo skill/nomi: test), `l2_smoke.toml` (overlay 2 step).
- `node_l2_smoke.sh [a b c d]` — gate misurato prima di spendere GPU (vedi PROTOCOL §4).
- `node_l2_run.sh <code|wavelet>` — driver a due fasi fail-closed, coverage per step.
- `node_l2_eval.sh <code|wavelet|base> [--skill]` — eval finale sul test ufficiale (800).
- `l2_monitor.py` — riga `[L2-Coverage]` per step dai rollout salvati dal pin (`<run>/run_default/rollouts/step_N/train_rollouts.jsonl`: categoria, reward, forced-final, gruppi a varianza zero, tool call, chiamate `wavelet_zoom`, token) + la riga `Step N | …` di `logs/orchestrator.log`.
- `overlay/prime-rl/` — tre override env-gated del pin (backend gloo, allocator del trainer, LoRA senza grouped_mm), applicati da `node_l2_env.sh`; unset = comportamento upstream.
- `tests/test_l2_kit.py` — test CPU (config, variante nota, monitor): `python -m pytest l2/tests -q`.
- `overlay/timerlm/` — copie tracciate dei file wavelet/RL che vivono nel checkout ignorato `timerlm/` (skill rlm, tool v0, `taskset.py` con la variante nota, `eval_v1_rlm.py`, pyproject dell'harness con `mcp<2`, PRIME_RL_PIN.md).

## Sequenza sul nodo

```bash
# dal PC (WSL): spedire il kit (l2/ intero) in ~/l2/kit e l'overlay in ~/l2/kit/overlay
# (scratchpad/node_l2_env_launch.sh e node_l2_smoke_launch.sh lo fanno)
bash ~/l2/kit/l2/node_l2_env.sh            # -> L2_ENV_OK, ~/L2_ENV_MANIFEST.json
bash ~/l2/kit/l2/node_l2_smoke.sh a b c d  # -> ~/l2/smoke/*.json, SMOKE_OK
bash ~/l2/kit/l2/node_l2_run.sh code       # -> ~/l2/runs/code/{DONE,coverage_final.txt,weights/step_60}
bash ~/l2/kit/l2/node_l2_run.sh wavelet
bash ~/l2/kit/l2/node_l2_eval.sh code; bash ~/l2/kit/l2/node_l2_eval.sh wavelet
bash ~/l2/kit/l2/node_l2_eval.sh base;  bash ~/l2/kit/l2/node_l2_eval.sh base --skill
```

Da scaricare a fine corsa: `~/l2/runs/<arm>/{RUN_MANIFEST_L2.json,L2_ENV_MANIFEST.json,coverage_final.*,orchestrator_metrics.jsonl,trainer_metrics.jsonl,logs/,configs/}`, `~/l2/eval/*/{summary.json,test800.jsonl}`, `~/l2/smoke/*.json`.

## Regole ereditate dal Livello 1

Contatori prima delle conclusioni (rollout per step, errori, forced-final, gruppi scartati);
niente budget silenziosi; cartelle di run mai riusate; ogni attesa è limitata; il codice
di scoring del benchmark non si tocca.
