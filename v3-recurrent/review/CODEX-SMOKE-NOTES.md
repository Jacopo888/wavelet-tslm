# CODEX smoke notes — v3 end-to-end su A40

## Scelta deterministica dei casi difficili

`node_smoke_v3.sh` non dipende dall'ordine casuale del `DataLoader` e non aggiunge un hook di
selezione al trainer. Prima di caricare Qwen, apre il parquet train reale con `Dataset.from_parquet`,
legge solo `id`, `category`, `length` e `n_channels`, e costruisce
`anomalyXL/datasets/anomalyxl-v3smoke/train.parquet` con quattro righe:

1. tra le righe con `category == localize_all_channels` e `n_channels == 16`, quella
   con la massima lunghezza disponibile (131072 se presente), risolta deterministicamente per id
   e poi indice sorgente in caso di parità;
2. solo se il primo caso non è già lungo 131072, la riga lessicograficamente prima con
   `length == 131072`;
3. filler deterministici ordinati per lunghezza, numero canali, id e indice, fino a quattro righe.

Se la stessa riga soddisfa i primi due criteri viene inserita una volta sola. Sul parquet misurato
il primo criterio seleziona quindi il vero caso peggiore di memoria: 16 × 131072 campioni, ossia
4096 token percettivi più il testo. Lo script rilegge le righe complete scelte e verifica anche che
`channel_values` contenga esattamente 16 canali nel caso `hard_16` e vettori lunghi
131072 nel caso `hard_131072`. Scrive gli id e gli indici sorgente in
`~/logs/v3smoke.subset.json`.

Il mini-parquet contiene esattamente quattro righe, la config usa `max_samples: null` e una sola
epoca. Quindi `shuffle=True` cambia soltanto l'ordine: tutti e quattro i campioni raggiungono
`train.py`. Il gate richiede la riga `[Coverage] epoch 1: trained 4, failed 0, expected 4` e una
riga esatta di categoria `localize_all_channels` con almeno un successo. La validation è un parquet separato con
la riga deterministica più economica del val reale; l'eval-only deve rendicontare 1/1 esempi.

Non sono impostati né `TSLM_MAX_TRAIN_STEPS` né `TSLM_MAX_TRAIN_SECONDS`, e
`TSLM_REQUIRE_FULL_COVERAGE` viene esplicitamente rimosso dall'ambiente. Non esiste quindi il
conflitto budget/coverage di `train.py`; la copertura è comunque verificata positivamente dallo
script sul sottoinsieme interamente consumato.

## Cosa misura e cosa blocca

Ordine obbligatorio: `v3untied`, poi `v3tied`. Entrambe passano dal vero `train.py` con
Qwen3.5-4B, LoRA, encoder v3 e proiettore. Il `COMMON_ENV` replica `node_train_v3.sh`, salvo
l'assenza di full coverage e le aggiunte da smoke: modalità offline,
`TSLM_V3_GATE_LOG_EVERY=1` e `TSLM_STEP_TIMING=1`.

Il piccolo hook in `kaggle/patch/train.py` è inerte per default. Solo con
`TSLM_STEP_TIMING=1` sincronizza CUDA dopo ogni optimizer step e stampa
`[StepTiming] step=N seconds=S`. Lo smoke pretende i passi 1, 2, 3 e 4 senza buchi e calcola la
media sui passi 2–4. Registra inoltre wall time di training, wall time di eval e totale.

Un sampler `nvidia-smi` ogni 2 secondi resta attivo dall'inizio del training alla fine
dell'eval-only di ciascuna variante. Il massimo campionato deve essere al più 23000 MiB.
`PYTORCH_CUDA_ALLOC_CONF` viene rimosso per lasciare l'allocator nativo della vGPU. La GPU deve
essere una A40 e non deve avere altri processi compute all'avvio.

Per ogni checkpoint slim `best_model.pt`, `torch.load(..., weights_only=False)` deve trovare:

- `ts_encoder.latents`;
- almeno una chiave per ciascun prefisso `ts_encoder.step_embed.`, `ts_encoder.gate_head.`,
  `ts_encoder.blocks.` e `ts_encoder.ln_out.`;
- nessuna chiave contenente `pos_freqs`;
- il valore `tied` coerente con la variante.

Poi un processo nuovo esegue `train.py --eval-only --split val` con quel checkpoint. Sono
obbligatori `EVALONLY_RESULT`, `[EvalCoverage] ... 1/1 examples accounted for`, un batch, un
campione valutato e zero `eval_errors`. Qualunque fallback, chiave mancante, riga di coverage
mancante, assenza di `[V3Gates]`, errore di eval, picco eccessivo o timeout produce
`~/logs/v3smoke.FAILED` e nessun `SMOKE_OK`.

Il limite complessivo fisso è 1800 secondi. Run dir e mini-parquet vengono rimossi sia in
successo sia in errore; log, misure e manifest del sottoinsieme rimangono in `~/logs`.

## Cosa leggere prima della run da 20 ore

Eseguire:

```bash
bash ~/node_smoke_v3.sh
python3 -m json.tool ~/logs/v3smoke.ok
```

Claude deve autorizzare `bash ~/node_train_v3.sh v3tied` solo se:

- l'ultima riga dello script è esattamente `SMOKE_OK` e `~/logs/v3smoke.FAILED` non esiste;
- il JSON ha `status: OK`, `run_dirs_removed: true` e varianti nell'ordine
  `v3untied`, `v3tied`, entrambe con `pass: true` e `steps: 4`;
- `subset.checks.localize_all_channels_16_present`,
  `hard_16_has_max_available_length` e `length_131072_present` sono true;
- ciascuna variante mostra coverage 4/4, almeno una riga `localize_all_channels` riuscita,
  `v3_gate_lines > 0`, un valore finito `seconds_per_step_after_first` e
  `gpu_peak_mib <= 23000`;
- `checkpoint_has_pos_freqs` è false, tutte le cinque famiglie di chiavi sono presenti e
  `eval.EVALONLY_RESULT` è true con un campione e zero errori;
- `total_wall_s <= 1800`.

Per l'audit umano dei valori dei gate e delle righe originali restano:
`~/logs/v3smoke.v3untied.train.log`, `~/logs/v3smoke.v3tied.train.log` e
`~/logs/v3smoke.subset.json`. `ops/ship_v3.sh` non richiede modifiche: spedisce già
`v3-recurrent/configs/*.yaml` e `v3-recurrent/node/*.sh`.
