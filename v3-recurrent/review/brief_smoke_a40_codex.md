# Brief per Codex — smoke end-to-end v3 sulla A40 (scrivi tu lo script)

Repo: `<cartella-locale>\wavelet-tslm`. Non committare, niente rete. Il tuo
sandbox non esegue comandi: scrivi script semplici e verificabili a occhio; li eseguo io sul nodo
e ti riporto i log se serve un secondo giro.

## La GPU trovata (fatti misurati oggi, 05/09)
- Vultr `wavelet-l1`, **NVIDIA A40-24Q** (vGPU GRID, driver 550, CUDA 12.4), 24 576 MiB, 12 vCPU,
  60 GB RAM, 820 GB liberi. `~/venv` con torch 2.6.0+cu124, `~/timerlm/sft/tslm` al commit di va1
  con l'overlay v3 già applicato (`node_env_v3.sh` → `V3_ENV_OK`: `MODEL_CLASSES` contiene
  `wavelet_v3`, test CPU 18/18, smoke del solo encoder 16×131072 fwd+bwd: tied 314 MiB, untied
  425 MiB). `~/RUN_MANIFEST_V3.json` ha `qwen_revision`. Dati in `~/timerlm/anomalyXL/datasets`
  (train 4488, val 520, test 800; preflight anti-leakage OK).
- Trappole vGPU: NIENTE `expandable_segments` (VMM non supportata → "operation not supported");
  NCCL ≥2.27 non si inizializza (irrilevante: processo singolo); allocator nativo.
- Riferimento va1 su questa GPU: 22 440 step in ≈20 h (≈3,2 s/step medi) con
  `TSLM_GRAD_CKPT=1`; picco memoria non registrato.

## Cosa deve fare `v3-recurrent/node/node_smoke_v3.sh`
Uno smoke **end-to-end attraverso `train.py`** (Qwen3.5-4B LoRA r16 + encoder v3 + proiettore,
stesso `COMMON_ENV` di `node_train_v3.sh` salvo i budget), che dimostri PRIMA della run da 20 h:
1. training di pochi step reali con **`v3untied`** (il più pesante) e poi **`v3tied`**, includendo
   con certezza almeno una cella a **16 canali** (categoria all_channels) e una serie da
   **131 072** campioni. Leggi `kaggle/patch/train.py` e
   `timerlm/sft/tslm/src/datasets/anomaly_xl/qa_dataset.py` (copia in `kaggle/patch/qa_dataset.py`)
   per capire ordine/shuffle del dataloader, `runtime.max_samples`, `TSLM_SUBSET_SEED`,
   `TSLM_MAX_TRAIN_STEPS`/`TSLM_MAX_TRAIN_SECONDS` e il loro conflitto con
   `TSLM_REQUIRE_FULL_COVERAGE=1` (train.py abortisce se coesistono: lo smoke non deve
   usare REQUIRE_FULL_COVERAGE, oppure usa un sottoinsieme senza budget). Scegli il modo
   più semplice e deterministico per garantire i due casi difficili; se serve un hook in
   `train.py` deve essere env-gated, inerte di default, minimo e documentato (preferibile
   nessuna modifica; `wavelet_encoder.py`/`wavelet.py` v1 restano intoccabili).
2. run dir e run_name dedicati (es. `anomaly_xl_wavelet_v3smoke_<variant>`), config di smoke
   in `v3-recurrent/configs/` con nome esplicito (ship_v3.sh spedisce `configs/*.yaml` e
   `node/*.sh`; se metti file altrove aggiorna `ops/ship_v3.sh`), sezione `training:` uguale a
   va1 salvo ciò che lo smoke richiede; alla fine i run dir di smoke vengono rimossi.
3. misure: picco memoria GPU (sampler `nvidia-smi` in background a 2 s, come già fatto per gli
   smoke L2, oppure `torch.cuda.max_memory_allocated` se train.py lo stampa), tempo per step
   medio sui passi dopo il primo, tempo totale; presenza delle righe `[V3Gates]`
   (`TSLM_V3_GATE_LOG_EVERY` piccolo nello smoke) e della riga `[Coverage]`.
4. checkpoint: aprire con `torch.load` lo slim salvato dallo smoke e verificare che contenga
   `ts_encoder.latents`, `ts_encoder.step_embed.*`, `ts_encoder.gate_head.*`,
   `ts_encoder.blocks.*`, `ts_encoder.ln_out.*` e nessun `pos_freqs`; poi `--eval-only` su un
   piccolo sottoinsieme della val (`runtime.max_samples` o equivalente) con quel checkpoint,
   fino a un `EVALONLY_RESULT`, per provare tutto il percorso di eval con la v3 (la qualità
   non conta).
5. fail-closed: ogni condizione verificata positivamente; esito in `~/logs/v3smoke.ok` con JSON
   (variante, step, s/step, picco MiB, chiavi checkpoint, esito eval) oppure
   `~/logs/v3smoke.FAILED` con la ragione; stampa `SMOKE_OK` solo alla fine; durata totale
   attesa ≤ 30 min; abortisci se il picco supera 23 000 MiB.

## Consegna
- `v3-recurrent/node/node_smoke_v3.sh` (+ config di smoke se servono, + eventuale aggiornamento
  di `ops/ship_v3.sh` e `README.md` sezione "Sequenza sul nodo").
- `v3-recurrent/review/CODEX-SMOKE-NOTES.md`: come garantisci i casi 16 canali e 131k, cosa
  misuri, cosa deve leggere Claude nell'output prima di lanciare `node_train_v3.sh v3tied`.
