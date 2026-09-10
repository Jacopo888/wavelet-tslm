# Brief per Codex — Livello 3, punto 3: smoke GRPO sulla policy multimodale va1 (4B, senza ricorrenza)

Stato: bozza del 10/09/2026 (Claude). Codex scrive il codice, Claude controlla il diff ed esegue
(regola del 05/09). Prima di scrivere codice, Codex apre il file con una sezione **Decisioni di
disegno** (punto 6) e la lascia nel PR.

## 1. Cosa vogliamo e perché

Pipeline del Livello 3 (ordine di Jacopo, 10/09):

`serie → WaveletTSEncoder va1 → proiettore → embedding wavelet → Qwen3.5-4B (LoRA) → politica RL`

I coefficienti wavelet entrano nel modello **come embedding** (il percettore del Livello 1), non
come testo. `wavelet_zoom` potrà restare uno strumento aggiuntivo (punto 4, fuori da questo brief).
Il kit `l2/` NON è riusabile: serve Qwen3.5-2B testuale via vLLM e non sa caricare il checkpoint
multimodale. Qui si costruisce un **loop GRPO minimale in puro HF/PyTorch**, single-turn, sulla
stessa architettura di va1, che sostituisce i placeholder con gli embedding temporali sia nei
rollout sia nel calcolo dei log-prob.

Sequenza obbligata (Jacopo): (1) checkpoint salvati fuori dal nodo con SHA-256 ✔; (2) caricamento
esatto verificato ✔ e riproduzione dello 0,2744 (in corso, `~/evalq/va1_repro_test.json`);
**(3) questo brief**; (4) solo dopo il ciclo agentico/tool; (5) confronto `va1 SFT` vs `va1 + RL`
su stessi dati e stesso checkpoint iniziale.

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
  script `check_va1_exact_load.py` (scratchpad → da portare in `l3/tools/`).
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
  + `load_checkpoint(epoch_5_end.pt)`; test che lo dimostri (risposta greedy identica a va1 su 5
  esempi fissi di val, log-prob identici alla reference al passo 0).
- A40-24Q vGPU: batch 1, `TSLM_GRAD_CKPT=1`, bf16 AMP, `sdpa`; NIENTE `expandable_segments`
  (VMM non supportata), NIENTE NCCL/multi-processo, niente vLLM. Picco misurato < 22 GB anche sulla
  cella 16ch × 131072.
- Contatori prima delle conclusioni: per step `rollouts`, `parsable`, `reward_mean/std`,
  per categoria, `zero_variance_groups`, `kl`, `grad_norm`, `tokens`, `s/step`; un passo che
  perde rollout senza contarli è un bug. Niente budget silenziosi, cartelle di run mai riusate,
  fail-closed (marker DONE/FAILED).
- Test CPU eseguibili in locale con `py -m pytest l3/tests -q` (niente GPU in locale) e sul nodo.
- Il test 800 pinnato NON entra in nessuna scelta di disegno: RL sul train split, selezione
  sulla val (o sui probe80 fissi), test una volta sola al punto 5.

## 4. Deliverable

1. `l3/grpo.py` — loop GRPO single-turn su un `WaveletTSModel` già caricato:
   - per ogni prompt del batch: `G` completamenti campionati (`do_sample=True`, temperatura e
     top_p in config) da `model.generate(**gen_inputs)`; reward = metrica primaria ufficiale della
     categoria (`evaluate_answer`) + penalità di formato documentata per output non parsabile;
   - vantaggi normalizzati per gruppo; loss policy-gradient con ratio clippato (PPO/GRPO) e KL
     verso la **reference = va1 iniziale** (cioè i 42 tensori allenabili all'init, NON "LoRA
     spenta": Codex progetta uno swap economico dei 3,4 M parametri, es. copia in CPU e context
     manager); coefficiente KL e loss a livello sequenza o token: scelta motivata al punto 6;
   - log-prob del completamento calcolati con lo stesso `_merge_features` (prompt multimodale +
     completamento), un forward per completamento con grad ckpt, accumulo dei gradienti;
   - gruppo/i di parametri allenabili configurabili: `lora` sola oppure `lora+encoder+projector`
     (default proposto da Codex al punto 6), stesso split di `get_trainable_parameters()`;
   - checkpoint slim (solo tensori allenabili, come SFT) + `RUN_MANIFEST` con sha del checkpoint
     iniziale.
2. `l3/configs/l3_smoke.yaml` (eredita `wavelet_precise_va1.yaml`; sezione `rl:` con G,
   temperatura, top_p, lr per gruppo, clip, kl_coef, steps, prompts per step, `max_new_tokens`).
3. `l3/node_l3_smoke.sh` — sul nodo, nella venv L1 (`~/venv`), mai nella venv di prime-rl:
   (a) test CPU; (b) `V3_INIT_EQUALS_VA1`-style: greedy su 5 esempi fissi di val == output di
   va1 (stessi `output_logs` di `evalonly_val_epoch_5_end`), ratio log-prob = 1 al passo 0;
   (c) 20 step su 8 prompt fissi di val × G=4 che coprono le 5 categorie, incluse una cella 16ch
   e una lead_lag: nessun OOM (picco loggato), nessun NaN, reward e KL finiti, `s/step`;
   (d) eval-only sui probe80 fissi (`configs/wavelet_precise_va1_valprobe.yaml`, 80 esempi) del
   checkpoint dopo 20 step: deve restare ≥ 0,10 (va1 ep5 sugli stessi 80: vedi
   `~/evalq/va1_ep5_valprobe80.json`), cioè lo smoke non deve distruggere la policy;
   scrive `~/l3/smoke/SMOKE_OK` solo se a-d passano; JSON con tutti i contatori.
4. `l3/tests/test_l3_kit.py` — config, reward sul parser ufficiale (casi parsabile / non
   parsabile / categoria sbagliata), normalizzazione dei vantaggi con gruppo a varianza zero,
   swap della reference (round-trip esatto), contatori.
5. `l3/PROTOCOL-L3.md` — cosa si misura al punto 5: `va1 SFT` (epoch_5_end.pt) vs `va1 + RL`
   sullo stesso test 800 pinnato con bootstrap appaiato 10k (`v3-recurrent/audit/paired_eval.py`
   esiste già), stesso `max_new_tokens`, stessa decodifica greedy; budget RL dichiarato in step,
   rollout e ore A40; deviazioni dichiarate.

## 5. Ordine di lavoro

1. Sezione **Decisioni di disegno** (punto 6) nel PR, con le alternative scartate.
2. Codice + test CPU (`py -m pytest l3/tests -q` deve passare in locale).
3. Claude spedisce sul nodo e lancia `node_l3_smoke.sh`; Codex legge i JSON e decide se il gate è
   passato. Nessun run lungo prima di `SMOKE_OK`.

## 6. Decisioni di disegno (da compilare, Codex)

- Reference per la KL: come si conserva e si applica lo stato iniziale dei 42 tensori.
- Loss: GRPO a livello sequenza vs token; clip; `kl_coef`; eventuale penalità di lunghezza (no
  budget impliciti).
- Parametri allenabili in RL: solo LoRA vs LoRA+encoder+projector; lr per gruppo.
- Campionamento: G, temperatura, top_p; `max_new_tokens` (le risposte sono JSON corti).
- Memoria: stima e misura per la cella 16ch × 131072 (4096 token percettivi + testo + G completamenti).
- Reward: primaria ufficiale per categoria + penalità di formato; come si contano i non parsabili.
- Cosa NON è coperto: ciclo agentico/tool (punto 4), ricorrenza (v3), Livello 2 testuale.
