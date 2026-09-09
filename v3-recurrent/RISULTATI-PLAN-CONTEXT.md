# Risultati del piano "contesto v3" (PLAN-CONTEXT-COLLAPSE) — aggiornato man mano

Esecuzione notturna 09/09/2026 sulla A40 `wavelet-l1` (ordine Jacopo: "leggi ed esegui
PLAN-CONTEXT-COLLAPSE, ti lascio la notte attivo"). Codice: commit `6652295` (harness pilot,
reader_mode, scale, token gate, init comune), `b24050d` (`context_mode` local/global, diagnostiche
su tutti i token). Tutti i pilot: 1 epoca, scheduler originale (22.440 step, warmup 673), seed 42,
init comune `~/pilots/init_common.pt`, `reader_mode: cross_only`, scala del contesto init 0,1,
lr default 1e-4 / context 1e-3 / context_scale 1e-5, **valutazioni solo sulla val** (mai il test):
sondaggio sugli stessi 80 esempi (contesto acceso e spento, generazione), teacher forcing
acceso/spento sugli 80, val intera (520) a fine epoca con contesto acceso.

Riferimenti sugli stessi 80 esempi a fine epoca 1: va1 0,124; v3 residuo zero-init (= replica di
va1) 0,165. Val intera a fine epoca 1: va1 0,170; replica 0,135.

## Tabella dei pilot (val)

| pilot | contesto | step 500 on / off | step 2000 on / off | fine ep1 on / off (80) | TF Δ(on−off) fine ep1 | val intera on (520) | esito |
|---|---|---|---|---|---|---|---|
| 1 `cross` | cross-only, centraggio globale (`full`) | 0,032 / 0,036 | 0,065 / 0,067 | 0,095 / 0,093 | +0,0003 | **0,149** | nessun apporto |
| 2 `crosslocal` | cross-only, **solo locale** (centrato per banda) | 0,028 / … | … | … | … | … | in corso (07:16 UTC) |
| 3 `crossglobal` | cross-only, **solo globale** (medie per banda) | | | | | | in coda |
| — `fullsmall`, `gate` | | | | | | | rinviati: poco informativi dopo il pilot 1 |

## Pilot 1 `cross` — dettaglio (run `pilot_cross_20260909T014929Z`)

- Teacher forcing sugli 80: step 500 on 0,4686 / off 0,4693 (Δ −0,0008); step 2000 0,4398 /
  0,4393 (+0,0005); fine epoca 0,4330 / 0,4328 (+0,0003). Nessun vantaggio del contesto.
- Generazione fine epoca (80): on 0,0946, off 0,0931; per categoria identiche (magnitudo 0,214 /
  0,210, all_channels 0,182 / 0,182, localize 0 / 0). Val intera on: 0,1489 (localize 0,0,
  magnitudo 0,264, all_channels 0,333, lead_lag 0,251, classify 0,0).
- **Struttura del contesto su tutti i 256 token** (script `diag_ctx_full.py`, 3 serie sintetiche):
  a init frazione variabile 0,9997 in tutti i gruppi; step 500: slot grossolani coseno 0,23,
  dettaglio 0,12; **step 2000: grossolani coseno 0,99 (frazione variabile 0,08), dettaglio 0,75
  (0,50)**, differenza fra serie 0,83 → 0,38. Il centraggio globale (media su tutti i token
  nulla) nasconde un collasso a due cluster: un vettore per banda (identità di banda, già nota
  ai token base) e perdita del contenuto per token. Quindi il collasso non dipende da MLP e
  self-attention (assenti qui): è il segnale di training che premia un canale globale/per banda.
- gamma/beta della `context_norm` non spiegano nulla (gamma RMS 0,98, beta 0,02); scala 0,0997.
- Primo tentativo del pilot fallito per `TSLM_EVAL_STRICT=1` sul sondaggio (80/520): corretto
  nello harness (strict solo sulla val intera); training ucciso al passo 722, rilanciato.

## Pilot 2 `crosslocal` — dettaglio (run `pilot_crosslocal_20260909T071641Z`)

- step 500 on: 0,0276 (magnitudo 0,085). In attesa di off, TF, step 2000, fine epoca.

## Criteri di lettura (dal piano §5)

- `local` on > off (80 e val intera) e struttura per token conservata → il contesto locale porta
  informazione: confermare con 2 epoche e seed aggiuntivi prima delle run da 5 epoche.
- `local` on ≈ off → il contesto locale residuo non è utile a questo budget; `global` on vs off
  misura il valore del solo canale globale (atteso ≈ 0: le statistiche globali sono già nei
  token base).
- Tutti ≈ off → chiudere la variante con questa decomposizione come risultato negativo motivato.
