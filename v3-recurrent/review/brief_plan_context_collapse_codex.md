# Brief per Codex — eseguire PLAN-CONTEXT-COLLAPSE (parte ingegneristica), priorità dell'audit

Repo: `C:\Users\jacop\Proposta ETH FINALE\wavelet-tslm`. Non committare, niente rete. Modifica
direttamente i file (tu sei l'autore); io eseguo `py v3-recurrent/patch/test_encoder_v3.py`, smoke e
pilot sulla A40 stanotte. Jacopo ha approvato l'esecuzione del tuo piano `v3-recurrent/PLAN-CONTEXT-COLLAPSE.md`
(con l'ordine aggiornato dal tuo `AUDIT-CONTEXT-20260908.md`): prima un lettore che conservi
l'informazione locale, poi scala lenta e gate per token. Tutto ciò che qui non è specificato lo
decidi tu secondo il piano; riporta scelte e numeri in `CODEX-REPORT.md` (sezione 13) e aggiorna
`README.md` della cartella (alcune descrizioni sono storiche).

## Vincoli invariati
v1 (`kaggle/patch/wavelet_encoder.py`, `wavelet.py`) intatta; contratto 256×512; identità v3≡v1 con
residuo disabilitato (test esistenti); una sola piramide; `pos_dtype: input`; nessuna dipendenza nuova;
`train.py` solo con hook env-gated e inerti di default; test rapidi su CPU (`py`, torch 2.9 CPU).
Checkpoint `1b0f9fa`/`b2d6808` non devono caricarsi silenziosamente con optimizer incongruente.

## 1. Lettore che conserva struttura (priorità 1 dell'audit)
Config `model.encoder`:
- `reader_mode: full | cross_only` — `cross_only`: nel loop solo le letture cross-attention gated
  si accumulano in z (niente self-attention né MLP); `full`: come oggi.
- `self_attn_scale_init`, `mlp_scale_init` (default 1.0 = comportamento attuale): scalari appresi
  che moltiplicano le uscite di self-attention e MLP nel blocco (per blocco se untied). Con valori
  piccoli (es. 0.1) reintroduci le trasformazioni con contributo controllato.
- Diagnostiche del CONTESTO DA SOLO, per iterazione e a fine loop (detached, campionate): RMS di z
  prima della LN, RMS di h=LN(z), frazione variabile fra token `RMS(h-mean_token(h))/RMS(h)`,
  coseno medio fra token di h, dipendenza dall'input di h (serie diversa e traslazione), RMS di
  ogni componente (cross/self/MLP) per iterazione. Esposte su `last_context_stats` e loggate
  con `TSLM_V3_GATE_LOG_EVERY` (stessa riga `[V3Ctx]`), e nel trainer con l'hook già presente
  (`TSLM_V3_GRAD_LOG`) ogni N step configurabile via env (`TSLM_V3_CTX_LOG_EVERY`, default 0).
- Test: (a) `cross_only` e `full` con scale 1.0 riproducono a init `v3 == v1` con residuo
  disabilitato; (b) su serie sintetiche, con `cross_only` la frazione variabile fra token di h a
  init è ≥ 0,3 e dipende dall'input (>0,1); con `full` (scale 1.0) documenta il valore attuale; con
  scale 0,1 deve stare vicino a `cross_only`; (c) witness held-out in cui l'informazione target
  è ASSENTE dalla selezione v1 ma presente nella piramide (es. anomalia a bassa ampiezza su
  un livello che il top-k globale scarta, oppure struttura periodica in un livello intermedio):
  dopo K step sintetici sul modello encoder+probe lineare, `cross_only` migliora l'held-out
  rispetto a `base` sola; usa anche esempi dove il contesto non serve e misura on/off.

## 2. Uscita: scala lenta e gate per token (priorità 2-3)
- `context_scale` in gruppo optimizer proprio `context_scale` (`training.learning_rate.context_scale`,
  default = lr di `context` se assente → nessun cambio per i config esistenti).
- `token_gate: false|true` — gate per token dopo la LN: `g_i = 2*sigmoid(w^T[LN_noaffine(base_i), h_i] + b)`,
  una sola Linear condivisa, `w=0,b=0` (g=1 a init: forward identico alla v3 senza gate);
  gruppo `context_token_gate`. Test: g=1 replica l'uscita; g=0 su uno slot lascia il token base
  bit-per-bit; distribuzione/saturazione di g in `last_context_stats`.
- Smoke (`node_smoke_v3.sh`): rendi coerente il gate sulla scala con LR lento: pretendi update
  finiti e misurabili del lettore (`blocks_grad_norm`>0, `blocks_relative`≥1e-3, spostamento dei
  pesi dei blocchi fra step 0 e 4 > 0), NON lo spostamento di `context_scale`; aggiungi le
  diagnostiche del contesto da solo (frazione variabile, coseno) nel JSON e fai fallire se la
  frazione variabile a fine dei 4 step è < 1e-3 (collasso già a init).

## 3. Harness pilot sulla val (senza test)
`v3-recurrent/node/node_pilot_v3.sh <pilot_name> <config.yaml> <n_epochs>`:
- run dir unico `results/anomaly_xl_precise/pilot_<name>_<timestamp>`; NON usa `node_train_v3.sh`
  né il worker (che pretende 5 epoche e lancia il test); manifest/hash overlay come oggi.
- scheduler originale (22440 step, warmup 673): stessa `training:` di va1; fermare dopo
  `n_epochs` epoche complete (coverage 4488/0 per epoca verificata dal log come nel worker;
  scegli il meccanismo: `TSLM_MAX_TRAIN_STEPS=n*4488` senza `TSLM_REQUIRE_FULL_COVERAGE`, o un
  hook env-gated; documenta).
- **init comune**: prima del primo pilot salva lo stato iniziale dei parametri allenabili comuni
  (encoder v1, proiettore, LoRA, lettore) in `~/pilots/init_common.pt` con un config di
  riferimento e caricalo in ogni pilot (strict sulle chiavi comuni, i moduli nuovi — token gate —
  inizializzati alla loro identità); ordine dei campioni fissato (stesso seed e stesso
  `TSLM_SUBSET_SEED`). Env-gated in `train.py` (`TSLM_INIT_STATE_PATH`, inerte di default).
- valutazioni SOLO val: a step 500, 2000 e fine epoca → sondaggio sugli 80 esempi fissi
  (`max_samples 800`, come i nostri probe: generazione, primary per categoria) in modalità
  contesto acceso e spento (a=0 forzato via env, es. `TSLM_V3_CONTEXT_OFF=1`), + teacher forcing
  on/off sugli 80 (riusa `audit/20260908-ctx01/forward_audit.py` come modulo/CLI), + diagnostiche
  del contesto (frazione variabile, coseno, RMS) sul checkpoint; a fine epoca anche la val intera
  (520) con contesto acceso. Le eval possono girare in parallelo al training (GPU 24 GB: training
  12,8 GB + eval 9,6 GB, già provato) — oppure sequenziali a fine epoca; scegli e documenta.
- output `~/pilots/<name>/summary.json` + `PILOT_DONE`/`PILOT_FAILED` nel log; fail-closed.
- config pilot in `v3-recurrent/configs/pilot_*.yaml`: almeno
  `pilot_cross_only.yaml` (reader cross_only, scala 0,1, lr context 1e-3, scala lenta 1e-5),
  `pilot_full_small.yaml` (full, self/mlp scale 0,1, stesse lr),
  `pilot_full_small_gate.yaml` (come sopra + token_gate), `pilot_control.yaml` (v3 attuale = A).
  Aggiorna `ops/ship_v3.sh` se servono nuovi path (configs/pilot_*.yaml, node/node_pilot_v3.sh,
  audit/ script usati dal pilot).

## 4. Consegna
Test aggiornati (tutti verdi con `py`), `bash -n` degli script, `git diff --check`, report §13 con
scelte, numeri dei test (frazione variabile a init per full/cross_only/full-small) e istruzioni
esatte per lanciare i pilot nell'ordine consigliato per stanotte (≈5 h l'uno su A40).
