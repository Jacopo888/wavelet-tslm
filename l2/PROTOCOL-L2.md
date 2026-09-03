# PROTOCOL L2 — pilot GRPO agentico con skill `wavelet_zoom` su una A40-24Q

Stato: kit scritto il 03/09/2026 (Claude, con il design iniziale di Codex interrotto dal suo
limite d'uso); da far rivedere prima di spendere i giorni di GPU dei due bracci.

## 1. Cosa si misura

Due bracci a seme, dati, budget e protocollo identici, che differiscono in una sola cosa:

| braccio | harness | nota nel prompt |
|---|---|---|
| `code` | RLM REPL (numpy/scipy) — il protocollo del paper TimeRLM | testo del paper, byte-identico |
| `wavelet` | idem + skill rlm `wavelet_zoom` pre-importata (`RLM_SKILLS=wavelet_zoom`) | testo del paper + una frase (`TIMERLM_NOTE_VARIANT=wavelet`) |

Ipotesi (dal pilot Livello 0): la piramide multirisoluzione lascia navigare serie fino a 131k
campioni in un budget di token compatto; sotto RL una policy piccola impara a usarla e
migliora reward/turni rispetto al braccio `code` a parità di budget.

Esito primario: reward sul test ufficiale (800 righe, HF `df363a46`) tramite replay del
protocollo di training (`scripts/eval_v1_rlm.py`), per braccio + due riferimenti zero-shot
(base con/senza skill). Esiti secondari: curva di reward per step (coverage.jsonl), turni medi,
token per rollout, tasso di forced-final, uso effettivo della skill (chiamate `wavelet_zoom(`
nei log di sessione).

## 2. Il pin di prime-rl (correzione a `docs/PRIME_RL_PIN.md`)

Il pin documentato (`da53d4c06`, 11/08) è scelto per il layout dei submodule, non per l'API:
il suo `verifiers` post-data due rotture (`RolloutContext` → `ModelContext` il 06/07;
setup/reward spostati da Taskset a Task il 10/07) e prime-rl stesso ha eliminato lo schema
del config del paper (`orchestrator.advantage`, `student.client`, `train.env`,
`max_inflight_rollouts`) il 27/06 e il 29/07. Verifica commit per commit (puntatore del
submodule + grep dei simboli importati dall'env): la finestra in cui env e config del paper
caricano intatti è 23/06–26/06. **Pin usato: `c5afcaa2e2db6751e0f2eae437afa6bd532fa87f`
(2026-06-26)**, verifiers `aa95826d`, torch 2.11.0+cu128, vLLM 0.23.0, flash-attn 2.8.3
(cubin sm_80, gira su sm_86), fla 0.5.2, transformers 5.6.2. Qwen3.5 denso via impl HF con
le patch varlen di prime-rl. Il lock del pin è usato con `uv sync --frozen`.

Altre correzioni di ambiente: `matplotlib`/`datasets` installati a mano (l'env v0 va con
`--no-deps` per non sovrascrivere il verifiers editable); `mcp<2` nel pyproject
dell'harness rlm (mcp 2.x ha rinominato `streamablehttp_client`).

## 3. Paper vs pilot (tutto ciò che cambia è imposto dall'hardware, e identico nei due bracci)

| parametro | paper (`anomalyxl_rlm_v5_ablation_fixed.toml`) | pilot | perché |
|---|---|---|---|
| GPU | 4× RTX PRO 6000 96 GB (2 trainer + 2 inferenza) | 1× A40-24Q vGPU, trainer e vLLM sulla stessa GPU | è quel che c'è |
| policy | Qwen3.5-4B, full-parameter | **Qwen3.5-2B** rev `15852e8c`, **LoRA r32/α64** su attn+MLP+GatedDeltaNet | il 4B non entra in 24 GB accanto a vLLM; LoRA per trainer leggero |
| lr | 7e-6 | 2e-5 | LoRA vuole un lr più alto del full-param |
| seq_len / compaction | 65536 / 45000 | 32768 / 22000 | memoria |
| RLM_MAX_TOKENS / OUTPUT / TURNS | 24576 / 8192 / 15 | 16384 / 4096 / 12 | finestra dimezzata |
| max_completion_tokens | 4096 | 2048 | idem |
| rollouts/step | 1024 (128×8) | **64 (8×8)** | tempo per step su vGPU |
| max_inflight | 128 | 16 (= max_num_seqs vLLM) | KV cache |
| step | 400 | **60** | ~1.5% del compute del paper: è un pilot |
| negativi | OFF fino a 220, poi ON (resume) | OFF fino a **35**, poi ON (resume da step_35) | stessa proporzione (58% vs 55%) |
| eval online | ogni 20 step, 50 esempi (val HF) | ogni 10, 32 esempi (val locale pinnata) | provenienza |
| length penalty | coef 0.1, gate off | uguale | |
| renderer / generator θ / celle | qwen3.5 / θ fisso / lunghezze fino a 131k | uguali (`run_seed` 20260903) | |
| server | 2× `vllm serve` standalone | 1× `uv run inference` standalone (util 0.42, prefix caching, LoRA hot-load) | |

Deviazioni che vanno dichiarate nel testo: modello 2B+LoRA, budget ridotto (step, rollouts,
finestra), eval finale con budget di training (16k/4096/12) invece di quello del paper.

Note di schema al pin (scoperte col `--dry-run`): `batch_size` conta i ROLLOUT per step e deve
essere divisibile per `group_size` (64 = 8 prompt × 8; nel fork del paper contava i prompt);
`[ckpt.weights]` non è un campo condiviso, quindi l'adapter LoRA non viene salvato a parte —
la valutazione usa i pesi HF fusi in `weights/step_N` (l'adapter resta nel checkpoint del trainer).

## 4. Gate: lo smoke test (`node_l2_smoke.sh`)

| stadio | cosa | PASS |
|---|---|---|
| a | server vLLM + `/v1/models` + una completion con tool call | tool call parsata dal parser `qwen3_coder` |
| b | 2 rollout reali per braccio via `eval_v1_rlm.py` (val pinnata) | 0 errori, ≥1 turno; braccio wavelet: skill pubblicizzata nel prompt di sistema (chiamate ≥0 a zero-shot) |
| c | `rl` 2 step, 2×4 rollout, sampler nvidia-smi 1 s | `checkpoints/step_2` presente, picco GPU < 23 000 MiB, monitor senza violazioni |
| d | `rl` 2 step al batch reale 8×8 | come c; misura s/step → ore per braccio = s/step × 60 / 3600 |

Solo con `SMOKE_OK` (più recente del manifest d'ambiente) il driver accetta di partire.

## 5. Esecuzione e copertura

`node_l2_run.sh <braccio>`: server → fase 1 (`ANOMALYXL_P_PRESENT=1.0 ANOMALYXL_ALLCH_MIN_K=1`
esportati nell'env di lancio, sonda che li verifica) → checkpoint step_35 obbligatorio → fase 2
(`--ckpt.resume-step 35 --max-steps 60`, sonda P_PRESENT=0.8) → `weights/step_60` → coverage
post-hoc su tutti i 60 step (`[L2-Coverage] step N: rollouts C/64, errored, forced_final,
reward, turni, token, zero-adv dropped, eval`) → `DONE`. Qualunque processo morto, attesa
oltre 45 h per fase, checkpoint mancante, tasso d'errore > 25% in uno step o rollout sotto
i 64 attesi → `FAILED` con motivo. Mai un DONE di comodo.

Stima tempo: dal paper e dalla vGPU (~L4), 64 rollout da ~12 turni con 16 in parallelo →
~25-35 min/step → 25-35 h per braccio, ~$25-30 di crediti l'uno; eval finale 800 righe ×
~4 target ≈ 6-8 h l'una. Lo smoke d sostituisce la stima con la misura.

## 6. Valutazione finale

`node_l2_eval.sh {code|wavelet|base} [--skill]` sul test ufficiale (800 righe locali pinnate):
`code` (protocollo paper), `wavelet` (skill), `base` e `base --skill` (zero-shot, riferimenti).
Righe JSONL per esempio, riepilogo per categoria, conteggio errori/timeout, chiamate skill;
DONE solo con 800/800. Confronto appaiato (bootstrap sui 800 id) fatto in locale come per L1.

## 7. Rischi aperti che solo lo smoke chiude

- vLLM 0.23 cu129 e FA2 cu128 sul driver 550: **chiusi dallo stadio a** (server sano in 210 s,
  tool call parsata, picco 9 GB a util 0.42).
- vGPU senza VMM: `enable_sleep_mode=false`; nessun `expandable_segments`.
- **NCCL non si inizializza sul vGPU (scoperto allo stadio c)**: la slice GRID non espone memory
  pool CUDA, VMM, eventi IPC (`cudaDeviceGetAttribute` → 0 per tutti; `cudaDeviceGetDefaultMemPool`
  → 801 not supported). NCCL ≥ 2.27 fallisce in `ncclCommInitRank` ("Cuda failure 'operation not
  supported'", nessuna variabile NCCL_* lo evita); NCCL 2.21 (torch 2.6) funziona ma torch 2.11
  linka simboli 2.28 (`ncclDevCommCreate`) e non si può degradare. Soluzione: trainer a rank
  singolo su **gloo** (`PRIME_RL_DIST_BACKEND=cpu:gloo,cuda:gloo`, override env-gated di 6 righe in
  `trainer/utils.py`, overlay `l2/overlay/prime-rl/`): FSDP2 + mixed precision + clip + AdamW
  verificati su gloo con tensori CUDA. Costo: collettive via CPU, ma a rank 1 sono no-op logici.
- **Il launcher `rl` forza `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` sul trainer**
  (`entrypoints/rl.py`): sul vGPU la prima allocazione fallisce ("CUDA driver error: operation not
  supported"). Override env-gated `PRIME_RL_TRAINER_ALLOC_CONF=""` (overlay `l2/overlay/prime-rl/`).
- Parser `qwen3_coder` con Qwen3.5-2B (il paper lo usa col 4B).
- Memoria: 0.42×24 GB per vLLM + trainer LoRA 2B a 32k con AC pieno; picco misurato in c.
- Weight sync LoRA via filesystem + `/load_lora_adapter` a ogni step: costo misurato in d.
- `impl=auto` → HF per Qwen3.5 denso a questo pin: AC pieno e LoRA sui `in_proj_*` da verificare nel log del trainer (`Applied LoRA to N modules`).
