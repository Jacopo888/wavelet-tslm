# Brief per Codex — v3 "residuo su v1": a init v3 ≡ v1, il loop ricorrente aggiunge solo contesto

Repo: `<cartella-locale>\wavelet-tslm`. Non committare, niente rete. Modifica
direttamente i file (tu sei l'autore); io controllo il diff, eseguo `py v3-recurrent/patch/test_encoder_v3.py`,
smoke sul nodo, run. Metti l'analisi nel report (Jacopo la vuole).

## Perché cambiamo disegno (fatti)
Due iterazioni del percettore a latenti (commit `cfd5256`, poi `0d2484e` con content highway) non
hanno chiuso il divario di decodificabilità con v1: probe ridge a init, MAE posizione v1 0,078,
v3 0,112 → 0,105 (1,45× → 1,36×), presenza 0,94 vs 0,82-0,85. Sulla GPU la run `cfd5256`
dopo un'epoca intera risponde con finestre costanti (sondaggio val 80 esempi: primary 0,053,
localize IoU 0,0; va1 alla stessa epoca 0,17). Non possiamo permetterci una terza run da 23 h su
un disegno che a init è già peggio di v1. Nuovo principio: **il punto di partenza deve essere
esattamente v1; la ricorrenza può solo aggiungere**.

## Disegno richiesto ("v3 residuo", classe `RecurrentWaveletTSEncoder` stessa API/contratto)
1. **Token di base = token v1**: per ogni canale calcola ESATTAMENTE i 256 token di
   `WaveletTSEncoder` (importa/riusa il codice v1 senza duplicarlo: banda grossolana intera +
   top-k globale dei dettagli per |d|/σ, `value_proj`, `band_embed`, posizioni Fourier,
   `pad_token`, `pos_dtype: input`). Ogni token porta già valore + posizione + banda. Tieni
   traccia della posizione normalizzata `p_i` e della banda `b_i` di ciascun token base.
2. **Loop ricorrente sulla scala (invariato nello spirito)**: le 256 query sono i token base
   (più `step_embed` piccolo); a ogni iterazione leggono il livello corrente (banda grossolana,
   poi D_L … D_1, ciascuno con al massimo `level_topm` coefficienti per |d|/σ, stessa
   featurizzazione) con cross-attention + prior di località centrato sulla posizione REALE del
   token `p_i` (non su i/256), poi self-attention fra i 256 token e MLP; blocco condiviso
   (`tied`) o per livello (`untied`); gate per livello; `depth_mode` full|sdepth|noise_gate;
   `TSLM_V3_DEPTH`. Il contesto accumulato è un residuo `c_i`.
3. **Uscita = token base + residuo con scala appresa inizializzata a ZERO** (o quasi:
   `context_scale` init 0 → a init `out_i == v1_i` bit-per-bit; prevedi anche l'opzione
   init piccolo, es. 0,1, via config, documentandola). Nessuna LayerNorm che rimescoli il token
   base: il proiettore MLP di v1 riceve, a init, esattamente ciò che riceve in va1.
4. **Test obbligatori** (in `test_encoder_v3.py`): (a) a init, con `context_scale=0`, uscita v3
   == uscita v1 (`torch.equal` o allclose stretto, fp32 e bf16) sugli stessi input, incluse serie
   corte, 16 canali, L=131072; (b) il probe `position_decodability` a init dà v3 == v1; (c) dopo
   pochi step sintetici la scala del contesto si muove (gradiente non nullo su `context_scale` e
   sui blocchi) e la decodificabilità non peggiora oltre 1,1× v1; (d) tutti i test esistenti
   (diversità, depth override, sdepth, noise_gate, tied/untied, checkpoint slim, checkpointing,
   nonfinite, bf16, ROCm-neutral) restano validi; (e) `TSLM_V3_DEPTH=1` = solo banda grossolana
   letta come contesto (i token base restano interi).
5. **Smoke** (`node_smoke_v3.sh`): mantieni diversità e probe; aggiungi l'asserzione "a init
   v3 == v1" (CPU, sul modulo, prima del training) e nel JSON i valori di `context_scale` dopo i
   4 step.
6. **Config** (`configs/wavelet_precise_v3{tied,untied,sdepth}.yaml`): aggiungi i nuovi campi
   (es. `context_scale_init: 0.0`, `locality_sigma`), lascia `training:` identico a va1.
7. **README.md** della cartella: riscrivi "Architettura" e "Le tre ablazioni" per il disegno
   residuo: l'ablazione 1 diventa "selezione v1 + contesto ricorrente multi-scala vs selezione
   v1 sola" (va1 è il controllo esatto, stesso punto di partenza); 2 e 3 invariate. Nota nel
   README che il percettore a soli latenti (commit `cfd5256`/`0d2484e`) resta come lavoro futuro
   con i suoi numeri di decodificabilità. `CODEX-REPORT.md`: sezione 11 con analisi e misure.
8. Vincoli: v1 (`kaggle/patch/wavelet_encoder.py`, `wavelet.py`) intatta; `wavelet_v3.py`
   continua a sostituire solo `ts_encoder`; i checkpoint precedenti non devono caricare
   (chiavi diverse → errore chiaro); niente dipendenze nuove.

Priorità: correttezza e "a init == v1" verificato bit-per-bit. Poi efficienza (una sola volta la
piramide Haar per canale, condivisa fra selezione v1 e livelli del loop).
