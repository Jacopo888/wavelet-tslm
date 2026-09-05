# Brief per Codex — variante v3 "percettore wavelet ricorrente" (tu sei l'autore del codice)

Repo: `C:\Users\jacop\Proposta ETH FINALE\wavelet-tslm` (Windows; Python con torch CPU = `py`,
cioè Python 3.14 + torch 2.12 cpu; `python` è 3.11 senza torch). Non committare. Non usare la rete.

## Contesto (leggi prima, nell'ordine)
1. `v3-recurrent/README.md` — il disegno della variante e le tre ablazioni. È la specifica.
2. `kaggle/patch/wavelet_encoder.py`, `kaggle/patch/wavelet.py` — la **v1** (riferimento va1,
   risultati in `vultr/RISULTATI-VA1.md`). **NON modificarli**: la v3 deve essere confrontata con
   va1 a protocollo identico e la v1 resta byte-identica.
3. `timerlm/sft/tslm/src/models/chatts.py` — modello base: `_encode_and_project` chiama
   `ts_encoder(time_series)` con `(C, L)` = tutti i canali di UN esempio (batch 1) e si aspetta
   `(C, N, D)`; il proiettore MLP porta D=512 → dim Qwen. `timerlm/sft/tslm/src/utils/config.py`.
4. `kaggle/patch/train.py` — training/eval (già contiene la registrazione di `wavelet_v3` in
   `MODEL_CLASSES`, 8 righe dopo il dict; verifica che basti). In particolare: come vengono salvati
   i checkpoint "slim" (`TSLM_SAVE_TRAINABLE_ONLY=1`), come `load_checkpoint` li ricarica in
   `--eval-only`, i gruppi di parametri/lr in `configure_training_mode` e in `chatts.py`
   (`groups["encoder"]`), `TSLM_GRAD_CKPT`, la loss memory-efficient in `wavelet.py`.
5. La bozza scritta da Claude come specifica eseguibile (test CPU 12/12 con
   `py v3-recurrent/patch/test_encoder_v3.py`): `v3-recurrent/patch/wavelet_encoder_v3.py`,
   `v3-recurrent/patch/wavelet_v3.py`, `v3-recurrent/patch/test_encoder_v3.py`,
   `v3-recurrent/configs/*.yaml`, `v3-recurrent/node/*.sh`.

## Mandato
Sei tu l'autore: riscrivi o correggi liberamente la bozza in `v3-recurrent/` finché il codice è
corretto, robusto e pronto per una run da 20 ore non presidiata (fail-fast, niente errori a
epoca 3). Claude controlla il diff. Vincoli fissi:
- contratto dell'encoder identico a v1 (forward `(B,L)->(B,256,512)`, `multi_variable`,
  `output_dim`, `freeze`, `patch_size`, `get_num_output_patches`); sequenza Qwen invariata;
- featurizzazione dei token identica a v1 con `pos_dtype: input` (parità per l'ablazione 1);
- nessuna modifica a `kaggle/patch/*` tranne, se necessario, le righe di registrazione in
  `train.py`; nessuna modifica alla sezione `training:` dei config; nessuna dipendenza nuova;
- le tre ablazioni restano: `tied` vs `untied`, `depth_mode: full|sdepth|noise_gate`,
  override `TSLM_V3_DEPTH` in eval.

## Cose da verificare esplicitamente (rispondi punto per punto nel report)
1. **Checkpoint**: con `TSLM_SAVE_TRAINABLE_ONLY=1` i parametri del nuovo encoder (`ts_encoder.*`,
   compresi `latents`, `step_embed`, `gate_head`, LayerNorm) finiscono nel checkpoint e vengono
   ricaricati in `--eval-only` (strict? chiavi?). Il buffer `pos_freqs` è non-persistente come in v1.
2. **Sostituzione dell'encoder in `from_config`** (`WaveletV3TSModel`: costruisce v1 e sostituisce
   `model.ts_encoder`): compatibile con `freeze_ts_encoder`, con i gruppi di parametri/lr del
   trainer, con `.to(device)` successivo in `load_model`, con `TSLM_DEVICE_MAP`? Se preferisci una
   costruzione diretta, fallo, ma senza duplicare logica in modo divergente da v1.
3. **Numerica bf16**: `nn.MultiheadAttention` in bf16 sotto autocast; LayerNorm; `torch.where`
   con latenti espansi; `topk` su `score` con NaN/inf impossibili? (`sigma + 1e-8` come v1);
   serie costanti (MAD = 0) e serie corte (L=64, levels=1).
4. **Memoria/tempo su A40 24 GB** con 16 canali × 131072 campioni, LoRA su Qwen3.5-4B,
   `TSLM_GRAD_CKPT=1`: l'encoder aggiunge ≤ 11 livelli × (16 × 1024 × 512) attivazioni bf16 per
   canale; stimare il picco e, se serve, checkpointing anche nel loop.
5. **`sdepth`**: `torch.randint` sul RNG globale è riproducibile con il seed del trainer? Cambia
   la sequenza RNG del training rispetto a `full` (accettabile, ma documentalo). Va bene una
   profondità per chiamata (= per esempio, dato batch 1)?
6. **`noise_gate`**: regola σ·√(2 ln n) sul massimo; cieca alle anomalie di varianza (dichiarato).
   Proponi, se la vedi, una variante energetica per livello (non obbligatoria, separata da flag).
7. **Gate appreso**: input `[mean z, mean e, stats]`, bias iniziale +2; rischio di collasso a 0
   per i livelli fini? Serve un termine di regolarizzazione? (non aggiungerlo se non necessario:
   solo motivare).
8. **Script nodo** (`v3-recurrent/node/*.sh`): coerenza con `vultr/node_env.sh`,
   `vultr/node_train_single.sh`, `vultr/node_eval_worker_single.sh` (gli originali di va1);
   il worker deve essere fail-closed; la curva di profondità va sulla val con il best checkpoint;
   `V3_ROCM=1` (MI300X: pin torch rocm6.3 in `node_env.sh`) è plausibile?
9. **Test**: estendi `test_encoder_v3.py` con ciò che ritieni necessario (es. round-trip
   state_dict, parità della featurizzazione con v1 su un livello, equivalenza `TSLM_V3_DEPTH`
   = steps con `full`). Tutti i test devono passare con `py v3-recurrent/patch/test_encoder_v3.py`.

## Consegna
- File modificati in `v3-recurrent/` (e al massimo la registrazione in `kaggle/patch/train.py`).
- `v3-recurrent/review/CODEX-REPORT.md`: elenco delle modifiche con motivazione, risposte ai
  punti 1–9, rischi residui, cosa Claude deve controllare prima del lancio.
