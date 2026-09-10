# Brief per Codex — Livello 3, punto 3: smoke GRPO sulla policy multimodale va1 (4B, senza ricorrenza)

Stato: **rev. 2 del 10/09/2026** (Claude), dopo la revisione di Codex del brief rev. 1
(commit `0fb3a46`): corretti l'uso della validation nello smoke (P1), la separazione fra
old policy e reference (P1), il numero di rollout (P2) e fissate le scelte elencate al §4.
Codex scrive il codice, Claude controlla il diff ed esegue (regola del 05/09). Prima di
scrivere codice, Codex apre il file con una sezione **Decisioni di disegno** (§7) e la lascia nel PR.

## 1. Cosa vogliamo e perché

Pipeline del Livello 3 (ordine di Jacopo, 10/09):

`serie → WaveletTSEncoder va1 → proiettore → embedding wavelet → Qwen3.5-4B (LoRA) → politica RL`

I coefficienti wavelet entrano nel modello **come embedding** (il percettore del Livello 1), non
come testo. `wavelet_zoom` potrà restare uno strumento aggiuntivo (punto 4, fuori da questo brief).
Il kit `l2/` NON è riusabile: serve Qwen3.5-2B testuale via vLLM e non sa caricare il checkpoint
multimodale. Qui si costruisce un **loop GRPO minimale in puro HF/PyTorch**, single-turn, sulla
stessa architettura di va1 (**Qwen3.5-4B, non il 2B di L2**), modello unico in memoria (nessun
server, nessun trainer separato), che sostituisce i placeholder con gli embedding temporali sia nei
rollout sia nel calcolo dei log-prob.

Sequenza obbligata (Jacopo): (1) checkpoint salvati fuori dal nodo con SHA-256 ✔; (2) caricamento
esatto verificato ✔ e riproduzione **esatta** dello 0,2744 ✔ (10/09: primary
0,2743539445041715, 0/800 righe diverse, testo generato incluso → la decodifica greedy bf16 su
questo stack è deterministica); **(3) questo brief**; (4) solo dopo il ciclo agentico/tool;
(5) confronto `va1 SFT` vs `va1 + RL` su stessi dati e stesso checkpoint iniziale.

## 2. Fatti sul modello e sul checkpoint (verificati)

- Codice: `kaggle/patch/` è la fonte di verità (sul nodo: `/root/timerlm/sft/tslm/`, venv `~/venv`,
  torch 2.6.0+cu124, transformers 5.3.0.dev0, peft). `WaveletTSModel(ChatTSModel)`
  (`kaggle/patch/wavelet.py`): `prepare_batch(batch, training=False)` costruisce il prompt con
  `num_patches` token `<ts_pad>` per canale (256 per canale, 16 canali = 4096 token percettivi);
  `_encode_and_project` + `_merge_features` sostituiscono gli embedding dei placeholder;
  `generate(**gen_inputs, max_new_tokens)` chiama `llm_model.generate(inputs_embeds=…)`;
  la loss di training è memory-efficient (`logits_to_keep` sulle posizioni supervisionate,
  batch_size=1 obbligatorio).
- LoRA: `configure_training_mode(model, config)` → `model.backbone.apply_peft(...)` (PEFT,
  `inject_adapter_in_model`), poi ri-sblocca i gruppi `encoder` (286.208 param), `projector`
  (1.314.304), `backbone_lora` (1.835.008): 42 tensori, 3.435.520 parametri allenabili.
- Checkpoint iniziale = **`epoch_5_end.pt`** (sha256
  `cd7238ae81da1afca9f225a99e9af4261be829798a39985a427d88a098bf2531`, epoch 5, val_loss 0,3478),
  NON `best_model.pt`. Sul nodo: `/root/timerlm/sft/tslm/results/anomaly_xl_precise/anomaly_xl_wavelet_va1/checkpoints/`;
  copie: `/srv/backups/wavelet-before-airway-20260905/…` e WSL `~/va1_offnode/`; hash in
  `integrity/artifacts/va1/checkpoints_SHA256SUMS.txt`. `load_checkpoint` (train.py) con
  `TSLM_SAVE_TRAINABLE_ONLY=1` carica strict=False: verifica esatta in
  `integrity/artifacts/va1/exact_load_ep5.json` (42/42, 0 mancanti, 0 in più, tutto finito);
  script `l3/tools/check_va1_exact_load.py`.
- Env di eval che ha prodotto 0,2744 (`vultr/node_eval_worker_single.sh`):
  `TSLM_TORCH_DTYPE=bfloat16 TSLM_AMP=1 TSLM_AMP_DTYPE=bfloat16 TSLM_ATTN_IMPL=sdpa
  TSLM_SAVE_TRAINABLE_ONLY=1 TSLM_EVAL_STRICT=1 TSLM_QWEN_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
  `train.py --config configs/wavelet_precise_va1.yaml --eval-only --checkpoint … --split test`.
- Scoring ufficiale (NON si tocca): in `validate()` → `dataset.extract_answer(pred, sample)` e
  `dataset.evaluate_answer(pred_answer, sample)` → dict con `correct` e la metrica primaria della
  categoria (`AnomalyXL.score_metrics`, `timerlm/environments/timeseries_qa/ts_datasets/anomalyxl.py`).
  Generazione di eval: greedy, `max_new_tokens = config.training.eval_max_new_tokens`.
- Dati: HF `nz00shuuuu/AnomalyXL` rev `df363a467f855e55aaad414b0bdd58590829df9a` via
  `create_dataloaders(config, eos_token)` (train 4488 / val 520 / test 800; categorie
  classify_with_evidence, measure_magnitude, localize, localize_all_channels ×16ch,
  lead_lag_with_magnitude ×2ch; lunghezze 16384…131072).

## 3. Vincoli non negoziabili

- Nessuna modifica all'architettura, al percettore, ai config va1, allo scorer.
  `kaggle/patch/wavelet_encoder.py` e `wavelet.py` restano byte-identici; il codice RL vive in `l3/`.
- A init la policy RL deve essere **va1 bit-per-bit**: stesso `load_model` + `configure_training_mode`
  + `load_checkpoint(epoch_5_end.pt)`; test che lo dimostri (risposta greedy identica a va1 e
  log-prob identici alla reference al passo 0, vedi §5 stadio b).
- A40-24Q vGPU: batch 1, `TSLM_GRAD_CKPT=1`, bf16 AMP, `sdpa`; NIENTE `expandable_segments`
  (VMM non supportata), NIENTE NCCL/multi-processo, niente vLLM. Picco misurato < 22 GB anche sulla
  cella 16ch × 131072 (stadio c0).
- Contatori prima delle conclusioni: per step `rollouts`, `parsable`, `unparsable`,
  `reward_mean/std`, per categoria, `zero_variance_groups`, `kl_mean`, `ratio_mean` (deve valere 1
  con un solo aggiornamento per batch), `grad_norm`, `tokens_prompt/answer`, `peak_mem_mib`,
  `s/step`; un passo che perde rollout senza contarli è un bug. Niente budget silenziosi,
  cartelle di run mai riusate, fail-closed (marker DONE/FAILED).
- Test CPU eseguibili in locale con `py -m pytest l3/tests -q` (niente GPU in locale) e sul nodo.
- **Separazione degli split (P1 di Codex)**: i prompt di RL vengono SOLO dal **train split**
  (4488). La **validation** serve solo a selezionare/gate (stadio d sui probe80 fissi) e non
  entra mai nei rollout né nel gradiente. Il **test 800** pinnato non viene toccato fino alla
  selezione su validation del punto 5: nessun `--split test` in `l3/` prima di quel momento.

## 4. Scelte fissate (Codex, revisione del 10/09; non riaprirle)

| scelta | valore |
|---|---|
| modello | Qwen3.5-4B va1 (`epoch_5_end.pt`), stesso `load_model`/`configure_training_mode` di SFT |
| reference per la KL | **va1 `epoch_5_end.pt`**, congelata, conservata come copia CPU dei tensori allenabili (nel primo smoke bastano i 32 tensori LoRA, 1,8 M param) e applicata con uno swap/context manager; **non** "LoRA spenta", **non** la old policy |
| old policy vs reference | due oggetti distinti: la **old policy** è la policy che ha campionato (log-prob registrati al campionamento, senza grad, usati per il ratio PPO); la **reference** è va1 e serve solo alla KL. Con un solo aggiornamento per batch il ratio vale 1 al passo (test) e la struttura resta pronta per più epoche interne |
| parametri allenabili | **solo LoRA** (`backbone_lora`, 32 tensori); `encoder` e `projector` congelati (`requires_grad=False`); flag di config per sbloccarli in seguito, non nello smoke |
| campionamento | `G=4`, temperatura `0.7`, `top_p=0.95`, `do_sample=True`; `max_new_tokens = eval_max_new_tokens` di va1 |
| budget dello smoke | **1 prompt di train per step, 20 step, rollout sequenziali** → 80 rollout in tutto (+4 dello stadio c0) |
| reward | metrica primaria ufficiale della categoria via `extract_answer`/`evaluate_answer`; output non parsabile → reward **0** e contato in `unparsable`; al massimo una piccola **penalità di formato separata** (`format_penalty`, default 0, loggata a parte, mai mescolata nella metrica) |
| loss | policy gradient con ratio clippato (PPO/GRPO), vantaggi normalizzati nel gruppo (eps; gruppo a varianza zero → vantaggi 0, contato) + **KL token-level solo sui token della risposta**, stimatore k3, coefficiente iniziale `0.01`; nessuna penalità di lunghezza nello smoke |
| memoria | **una** cella 16 canali × 131072 come gate (stadio c0, 1 step, G=4), non a ogni step; picco loggato |
| test 800 | mai, fino alla selezione su validation (punto 5) |
| lr | LoRA `1e-5` (proposta; Codex motiva al §7), AdamW, niente scheduler nello smoke, clip grad 1.0 |

## 5. Deliverable

1. `l3/grpo.py` — loop GRPO single-turn su un `WaveletTSModel` già caricato:
   - per il prompt dello step: `G` completamenti campionati **in sequenza** da
     `model.generate(**gen_inputs)`; al campionamento si registrano i token della risposta e i
     log-prob della old policy (senza grad);
   - reward per completamento come da §4; vantaggi normalizzati per gruppo;
   - log-prob correnti del completamento con lo stesso `_merge_features` (prompt multimodale +
     completamento), un forward per completamento con grad ckpt, accumulo dei gradienti sui `G`;
   - forward della reference (swap dei tensori LoRA, `no_grad`) per la KL token-level sui soli
     token della risposta;
   - un aggiornamento per step; contatori del §3; checkpoint slim (solo tensori allenabili, come
     SFT) + `RUN_MANIFEST` con sha del checkpoint iniziale e config risolto.
2. `l3/configs/l3_smoke.yaml` (eredita `wavelet_precise_va1.yaml`; sezione `rl:` con i valori del
   §4: `G`, `temperature`, `top_p`, `lr`, `clip_eps`, `kl_coef`, `steps`, `prompts_per_step=1`,
   `trainable=lora`, `format_penalty=0`, `train_prompt_indices` fissi).
3. `l3/node_l3_smoke.sh` — sul nodo, nella venv L1 (`~/venv`), mai nella venv di prime-rl:
   - (a) test CPU;
   - (b) **init == va1**: greedy su 5 esempi fissi (indici fissi del **train** split, per non
     toccare la val) == output di va1 con lo stesso `--eval-only` sugli stessi 5; log-prob
     policy == log-prob reference al passo 0 (differenza 0) e `ratio == 1`;
   - (c0) **gate memoria**: 1 step su una cella 16ch × 131072 del train, G=4: nessun OOM, picco
     loggato < 22 GB, reward/KL finiti;
   - (c) **20 step su 20 prompt fissi del train** (1 per step, G=4, sequenziali) che coprono le 5
     categorie con lunghezze miste (incluse una lead_lag e una 16ch di lunghezza media): nessun
     NaN, reward e KL finiti, `ratio_mean` = 1, `grad_norm` > 0, `s/step`, contatori completi;
   - (d) **gate di non distruzione su validation**: eval-only greedy sui probe80 fissi
     (`configs/wavelet_precise_va1_valprobe.yaml`, 80 esempi di val) del checkpoint dopo i 20
     step: primary ≥ 0,10 (va1 ep5 sugli stessi 80: `~/evalq/va1_ep5_valprobe80.json`). È
     l'unico uso della validation nello smoke e non sceglie nulla;
   - scrive `~/l3/smoke/SMOKE_OK` solo se a-d passano; JSON con tutti i contatori per stadio.
4. `l3/tests/test_l3_kit.py` — config, reward sul parser ufficiale (parsabile / non parsabile /
   categoria sbagliata), normalizzazione dei vantaggi con gruppo a varianza zero, swap della
   reference (round-trip esatto), KL k3 su un caso a mano, maschera dei token di risposta,
   ratio = 1 con old policy = policy, contatori, split guard (nessun indice di val/test fra i
   `train_prompt_indices`).
5. `l3/PROTOCOL-L3.md` — cosa si misura al punto 5: `va1 SFT` (epoch_5_end.pt) vs `va1 + RL`
   sullo stesso test 800 pinnato con bootstrap appaiato 10k (`v3-recurrent/audit/paired_eval.py`
   esiste già), stesso `max_new_tokens`, stessa decodifica greedy (deterministica, verificato);
   selezione del checkpoint RL **solo** sulla validation; budget RL dichiarato in step, rollout e
   ore A40; deviazioni dichiarate.

## 6. Ordine di lavoro

1. Sezione **Decisioni di disegno** (§7) nel PR, con le alternative scartate.
2. Codice + test CPU (`py -m pytest l3/tests -q` deve passare in locale).
3. Claude spedisce sul nodo e lancia `node_l3_smoke.sh`; Codex legge i JSON e decide se il gate è
   passato. Nessun run lungo prima di `SMOKE_OK`.

Fuori dal brief ma da ricordare: prima di distruggere il nodo, copia dei sei checkpoint
essenziali su un supporto indipendente dal PC (la copia WSL sta sullo stesso disco fisico).

## 7. Decisioni di disegno (da compilare, Codex)

- Reference: come si conserva e si applica lo stato iniziale dei tensori LoRA (swap/context manager).
- Old policy: come si registrano i log-prob al campionamento e come si verifica ratio = 1.
- Loss: forma esatta (clip, stimatore KL, media per token vs per sequenza), motivazione del lr.
- Memoria: stima e misura per la cella 16ch × 131072 (4096 token percettivi + testo + G completamenti sequenziali).
- Reward: primaria ufficiale per categoria; come si contano i non parsabili; `format_penalty`.
- Scelta dei 20 indici di train e dei 5 dello stadio b (seed, copertura per categoria e lunghezza).
- Cosa NON è coperto: ciclo agentico/tool (punto 4), ricorrenza (v3), Livello 2 testuale,
  sblocco di encoder/proiettore.
