# Brief per Codex — collasso dei 256 token dell'encoder v3 (`v3-recurrent/patch/wavelet_encoder_v3.py`)

Repo: `C:\Users\jacop\Proposta ETH FINALE\wavelet-tslm`. Non committare, niente rete. Modifica
direttamente i file (tu sei l'autore); io controllo il diff, eseguo i test con `py`, spedisco sul
nodo e rilancio. Jacopo chiede esplicitamente la tua analisi del codice: spiega la causa nel report.

## Fatti misurati (06/09, run `v3tied` completa sulla A40: 5 epoche, coverage 4488/0, loss finale 0,39)
- Val epoca 1 (520 esempi, 518 parsati): primary **0,0262** (va1 alla stessa epoca: 0,1703).
  Per categoria: localize 0,002, classify 0,0, measure_magnitude 0,085, localize_all_channels 0,0,
  lead_lag 0,03. Le risposte sono JSON corretti ma degeneri: finestre quasi sempre intorno a
  1000 (`start 950 end 1000`, `start 1010 end 1060`), magnitudo ≈ 1,0, `lead`/`lag_samples 1000`,
  in all_channels tutti i canali con lo stesso intervallo 970-1000: il modello risponde con una
  "moda" indipendente dalla serie.
- Diagnostica sull'encoder (script `scratchpad/diag_v3_tokens.py`, copia in
  `~/archive/v3tied_collapsed_init_20260906/diag_v3_tokens.py` sul nodo), serie 1×16384,
  CPU, eval:

  | encoder | cos medio fra i 256 token | std fra token / std totale | diff rel. con anomalia +30σ | diff rel. serie diversa | latents std | step_embed std |
  |---|---|---|---|---|---|---|
  | init casuale tied | **1,0000** | 0,0053 / 1,0000 | 0,0021 | 0,0008 | 0,0200 | 1,0098 |
  | epoch_1_end.pt | **1,0000** | 0,0016 / 1,0002 | 0,7426 | 0,0018 | 0,0199 | 1,0025 |
  | epoch_5_end.pt | **1,0000** | 0,0013 / 0,9948 | 1,6111 | 0,0094 | 0,0199 | 0,9952 |

  I 256 token in uscita sono lo STESSO vettore ripetuto 256 volte, a init e dopo 5 epoche; i
  latenti non crescono mai (std 0,0199). Il vettore reagisce all'energia globale (l'anomalia
  +30σ cambia l'uscita) ma non distingue realizzazioni diverse: Qwen vede un solo "token"
  globale, da cui presenza e magnitudo grossolana, nessuna posizione. Gate a epoca 5:
  [0,988, 0,124, 0,992, 1,0, …].
- Tutti i 18 test CPU passavano: nessuno misurava la diversità fra i token né la dipendenza
  dall'input.

## Causa (mia ipotesi, verificala sul codice)
`z_in = z + step_embed.weight[step]` con `latents` inizializzati a `randn*0.02` e `step_embed`
= `nn.Embedding` (init N(0,1)): dopo `ln_q` le 256 query differiscono di ~0,02 su norma ~1 →
letture (cross-attention) identiche per tutti i latenti → self-attention fra vettori identici =
identità → 256 uscite identiche. Il gradiente sui latenti è proporzionalmente minuscolo e con
lr 1e-4 non rompe mai la simmetria. È un difetto della bozza originale che la revisione non ha colto.

## Cosa fare
1. **Rompi la simmetria per costruzione**, scegliendo e motivando (nel report) tra:
   a. latenti inizializzati con **feature di posizione di Fourier** delle 256 posizioni
      `(i+0.5)/256` (stesse frequenze `pos_freqs` della featurizzazione dei token, proiettate a D
      con scala ~1) così a init ogni latente "possiede" una regione e l'attenzione parte locale
      (come i token grossolani di v1), poi impara a zoomare; step embedding a scala piccola
      (≈0,02) o iniettato sulle chiavi/valori del livello invece che sulle query;
   b. latenti `N(0,1)` (o trunc-normal std 1) e `step_embed` a scala 0,02;
   c. altro, se lo ritieni migliore. In ogni caso `pos_dtype: input`, contratto e config restano.
2. **Test che avrebbero intercettato il difetto** in `test_encoder_v3.py`, a init casuale e dopo
   pochi step di ottimizzazione su un obiettivo sintetico: (i) coseno medio fra i 256 token < 0,5
   e std fra token ≥ 0,3·std totale; (ii) due serie diverse → diff relativa > 0,1; (iii) la stessa
   serie con anomalia localizzata in due posizioni diverse → uscite diverse; (iv) norma del
   gradiente sui latenti non trascurabile rispetto a `value_proj` (rapporto > 1e-2). Aggiungi la
   diagnostica come funzione riusabile (`token_diversity(enc, x)`), usata anche dallo smoke.
3. **Smoke** (`node_smoke_v3.sh`): dopo il training di 4 step, esegui la diagnostica di diversità
   sullo slim checkpoint (`ts_encoder.*`) e fallisci se coseno medio > 0,9 o se l'uscita non
   dipende dall'input; scrivi i numeri nel JSON. Con la vecchia bozza questo smoke deve FALLIRE.
4. Non toccare `kaggle/patch/wavelet_encoder.py` / `wavelet.py` (v1). Aggiorna `README.md` della
   cartella (architettura: init dei latenti) e `CODEX-REPORT.md` con analisi, causa, fix.

Consegna: file modificati + report. Poi io: `py v3-recurrent/patch/test_encoder_v3.py` (tutti i
test), ship, `node_env_v3.sh`, `node_smoke_v3.sh`, e rilancio `node_train_v3.sh v3tied`.
