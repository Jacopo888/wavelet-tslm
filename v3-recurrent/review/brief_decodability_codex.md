# Brief per Codex — v3: la posizione fluisce nei token ma Qwen non la usa (analisi + ridisegno + test di decodificabilità)

Repo: `C:\Users\jacop\Proposta ETH FINALE\wavelet-tslm`. Non committare, niente rete. Modifica
direttamente i file; io controllo il diff, eseguo `py v3-recurrent/patch/test_encoder_v3.py`, poi
smoke e run. Jacopo vuole la tua analisi del codice nel report, non solo il fix.

## Fatti (07/09, run `v3tied` con il tuo fix del collasso, commit `cfd5256`, A40, epoca 1 conclusa)
- Training ok: coverage 4488/0, loss training fine ep1 0,446 (va1 0,451; run collassata 0,453),
  val loss teacher-forced 0,3966 (run collassata 0,3963): **la loss non discrimina**.
- Diagnostiche CPU su `epoch_1_end.pt` (script `scratchpad/diag_v3_tokens.py`, `diag_v3_position.py`,
  copie sul nodo in `~/`): coseno medio fra i 256 token 0,325 (init 0,05), std fra token 0,78;
  un'anomalia (+25σ, 40 campioni) al 15/50/85% della serie cambia esattamente gli slot
  35-42 / 125-132 / 214-221 (8/8 attesi) → **la posizione fluisce**; ma la variazione relativa
  dell'uscita è piccola e NON cresce col training: serie diversa 0,050 (step 1000) → 0,043 (fine
  ep1); anomalia 0,022-0,028 → 0,023-0,025.
- **Sondaggio della val a fine ep1** (80 esempi fissi, `max_samples: 800`, eval-only sul
  checkpoint, GPU): primary **0,053**; localize 0/21 con IoU 0,0 e risposte quasi costanti
  (`start 884 end 1006`, `start 978 end 1000`, `start 1000 end 1100`) indipendenti dal gold
  (gold anche a 25000, 55000, 63000); presence "present" 86% (= sempre presente);
  classify_with_evidence 0/15; lead_lag 0/7; measure_magnitude 0,085; localize_all_channels 0,18
  (casi vuoti). **Stesse risposte-moda della run collassata** (che sulla val intera faceva 0,026).
  va1 (encoder v1) alla stessa epoca: 0,17 sulla val intera, con IoU reali.
- Quindi: i token v3 contengono la posizione ma Qwen (LoRA r16 + proiettore MLP 512→dim) non
  la estrae entro un'epoca. In v1 ogni token = MLP([valore, Fourier(pos)]) + band_embed: il
  contenuto (valore) e la posizione sono ENTRAMBI nel token a scala piena e ogni token è già
  "un coefficiente grande" (rilevazione fatta dal top-k). In v3 dopo il tuo fix ogni token =
  `ln_out(z − 0,9·mean(z) + 0,85·anchor_i)`: l'ancora posizionale fissa domina la norma e il
  contenuto dipendente dall'input è ~3-4% (dopo LN). Ipotesi: Qwen deve imparare sia la mappa
  slot→posizione sia a leggere un residuo piccolo; troppo lento a parità di budget.

## Cosa fare
1. **Analisi**: leggi `wavelet_encoder_v3.py` e `kaggle/patch/wavelet_encoder.py` (v1) e spiega
   nel report perché i token v3 sono poco decodificabili rispetto a v1 (scala contenuto/ancora,
   LN finale, ampiezza delle letture della cross-attention con gate ~0,9, prior di località σ=2/256).
2. **Test di decodificabilità offline (CPU, senza Qwen)** in `test_encoder_v3.py` + funzione
   riusabile (es. `position_decodability(enc, n=…)`): serie sintetiche (rumore + anomalia a
   posizione casuale, lunghezze 4096-32768, tipi: spike, level shift, variance change), calcola i
   256 token con v1 (`WaveletTSEncoder`, stessa featurizzazione) e con v3, poi un **probe lineare
   / ridge** (chiuso, numpy/torch lstsq) dai token concatenati o da statistiche per slot alla
   posizione normalizzata dell'anomalia; metrica: MAE della posizione su un held-out. Requisito:
   v3 a init ≤ 1,5× la MAE di v1 a init, e il test deve FALLIRE sull'encoder attuale (commit
   `cfd5256`) se la tua analisi è giusta. Aggiungi anche la decodificabilità della *presenza*
   (anomalia sì/no) e della *magnitudo*.
3. **Ridisegna la scala contenuto/ancora** in modo principled, non con costanti a mano: ad es.
   `token_i = LN_c(z_i) + anchor_i` con `z_i` il residuo di contenuto e un scalare appreso
   (init 1) sul contenuto; oppure contenuto e ancora concatenati (256+256 dim) e proiettati a 512;
   oppure lettura iniziale v1-like: a init, `z_i` deve già contenere il coefficiente locale più
   grande (valore) del livello letto, così il token è "valore + posizione" come in v1; il loop
   ricorrente parte da lì e aggiunge il contesto multi-scala. Valuta anche il gate: a init ~0,88 è
   ok, ma verifica che la norma della lettura non sia schiacciata dall'`out_proj` dell'attenzione.
   Mantieni: contratto, 256 token, `pos_dtype: input`, ablazioni tied/untied/sdepth, `noise_gate`,
   `TSLM_V3_DEPTH`, `token_diversity`. La v1 resta intatta.
4. Il nuovo encoder deve passare: i 20 test attuali, il test di decodificabilità (v3 ≈ v1 o
   meglio a init E dopo pochi step sintetici), e le soglie dello smoke (`node_smoke_v3.sh`):
   aggiungi allo smoke la decodificabilità sul checkpoint addestrato (fallisce se v3 è molto
   peggio di v1).
5. Report in `CODEX-REPORT.md` (sezione nuova): analisi, scelta, numeri del probe v1 vs v3
   (vecchio e nuovo), rischi. Aggiorna `README.md` della cartella.

Vincolo di tempo: la GPU costa; se puoi, dai priorità a un fix semplice e misurabile.
