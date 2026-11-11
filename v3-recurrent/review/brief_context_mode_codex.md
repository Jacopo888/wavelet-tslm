# Brief per Codex — decomposizione del contesto: `context_mode: full | local | global` (+ config pilot)

Repo: `<cartella-locale>\wavelet-tslm`. Non committare, niente rete. Modifica
direttamente i file; io eseguo test, ship e pilot. Report in `CODEX-REPORT.md` §14; README aggiornato.

## Evidenza (pilot 1 `cross`, run `pilot_cross_20260909T014929Z`, in corso, 09/09)
Lettore `cross_only` (centraggio globale fra token, strength 1). Statistiche su TUTTI i 256 token
× 512 dim (script `~/diag_ctx_full.py`, 3 serie sintetiche L=32768 con spike al 20/50/80%):

| checkpoint | var_frac tutti | cos tutti | cos grossolani (slot 0-127) | cos dettaglio (128-255) | var_frac gross. | var_frac dett. | diff. fra serie |
|---|---|---|---|---|---|---|---|
| init | 0,9997 | −0,003 | −0,004 | −0,005 | 0,998 | 0,998 | 0,83 |
| step 500 | 0,996 | 0,005 | 0,23 | 0,12 | 0,88 | 0,93 | 0,83 |
| step 2000 | 0,998 | 0,001 | **0,99** | **0,75** | **0,08** | 0,50 | 0,38 |

Il centraggio globale nasconde un collasso a DUE cluster: gli slot grossolani convergono a un
vettore comune A, quelli di dettaglio a ≈ −A, il contenuto per token sparisce (prima nei
grossolani). Sondaggi 80: step 500 on 0,032 / off 0,036; step 2000 on 0,065 / off 0,067; teacher
forcing Δ(on−off) −0,0008 (step 500) e +0,0005 (step 2000). Quindi anche senza MLP/self-attention
il lettore impara un canale "globale/per banda" (ridondante con `band_embed`) e non informazione
locale: la loss premia statistiche di serie, non il dettaglio per token. La diagnostica
`_context_snapshot` usa una fetta `[:4,:64,:64]` (solo slot grossolani) e va corretta.

## Cosa fare
1. **`context_mode: full | local | global`** (default `full` = comportamento attuale, bit-per-bit):
   - `local`: prima della `context_norm` sottrai a z la media per GRUPPO DI BANDA (media sugli slot
     con lo stesso `base_band`, padding escluso; grossolana = banda 0, dettagli = banda ≥1 — decidi
     se per banda esatta o grossolana/dettaglio, motivando) → il contesto può portare SOLO
     variazione fra token dello stesso gruppo; nessun canale globale.
   - `global`: il contesto è SOLO la media per gruppo di banda (broadcast sugli slot del gruppo):
     nessuna variazione fra token. Misura il valore del canale globale da solo.
   - `full`: entrambi (oggi).
   Mantieni `reader_mode`, scale, token gate, `context_scale`, gruppi optimizer, init comune.
   Con `context_scale` 0 resta v3 ≡ v1. Test: `local` ha media per gruppo esattamente zero;
   `global` ha varianza fra token nel gruppo zero; `full` = `local` + `global` (allclose);
   witness: l'informazione locale (posizione) è decodificabile da `local` e non da `global`,
   la presenza/magnitudo globale da `global`.
2. **Diagnostiche su tutti i token**: `_context_snapshot` deve usare tutti i 256 slot e tutte le
   dimensioni (o un campionamento uniforme, non i primi 64), riportando var_frac/coseno TOTALI e
   PER GRUPPO (grossolani, dettaglio) e la differenza fra serie; stesse chiavi nel JSON dello
   harness e nello smoke (`[V3Ctx]`).
3. **Config pilot**: `wavelet_precise_v3pilot_crosslocal.yaml` (cross_only + local) e
   `wavelet_precise_v3pilot_crossglobal.yaml` (cross_only + global), stesse lr/init dei pilot
   attuali (scala 0,1, context 1e-3, context_scale 1e-5 se così è configurato — verifica e
   mantieni coerenza con `pilot_cross`). `ship_v3.sh` se servono path nuovi.
4. Test rapidi CPU tutti verdi (`py`), `bash -n`, `git diff --check`. Nient'altro.
