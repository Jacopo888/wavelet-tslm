# Video dimostrativo — dalla serie temporale agli embedding

`make_video.py` genera `wavelet_tokenization.mp4` (1760x990, 30 fps, ~75 s).
Nessun dato esterno: il segnale è sintetico e la decomposizione è calcolata
davvero, con lo stesso codice dell'encoder.

```bash
python make_video.py            # render del video
python make_video.py --stills   # 2 PNG per scena in stills/, per controllare i layout
```

Dipendenze: `numpy`, `matplotlib`, `ffmpeg` nel PATH.

## Cosa mostra, scena per scena

| # | scena | contenuto |
|---|-------|-----------|
| 1 | signal | serie grezza L = 16.384, anomalia lunga 144 campioni (0,9%): invisibile a figura intera, chiara nell'inset a zoom 114x |
| 2 | cost | patching uniforme: L/P = 1.024 token per canale, 8.192 per episodio; il riquadro rosso riporta il risultato misurato (ChatTS, 1500/1500 batch in OOM) |
| 3 | analysis | piramide di Haar: `d = (pari − dispari)/√2`, `a = (pari + dispari)/√2`, ricorsione; righe D1…D7 + A con le lunghezze reali |
| 4 | selection | criterio implementato (`score = |d| / 1.4826·MAD`, top-k) accanto al passo agentico previsto (`zoom(level=…, span=…)`); barra con la ripartizione reale dei 256 token |
| 5 | projection | la testina scorre sul layer scelto; per ogni coefficiente selezionato mostra valore, banda, posizione e score, poi `[valore | Fourier(pos) 16 freq] + band embedding → MLP 33→512` e la striscia di token che cresce |
| 6 | budget | token vs L in scala log: L/16 che sale contro i 256 piatti, con il punto OOM a L = 131k |
| 7 | answer | token wavelet interfogliati col testo nel prompt, Qwen3.5-4B + LoRA, risposta JSON in coordinate campione |

## Fedeltà al codice

Numeri e meccanica vengono da `esperimenti/kaggle/patch/wavelet_encoder.py` e
`wavelet_precise_full.yaml`: `num_tokens=256`, `coarse_blocks=128`,
`n_pos_freqs=16`, `output_dim=512`, quindi 7 livelli per L = 16.384 e un budget
di 128 coarse + 128 dettagli. `haar_pyramid` e la selezione top-k sono
riscritte in numpy con la stessa ricorsione e la stessa normalizzazione MAD.

Due licenze poetiche, entrambe dichiarate a schermo:

- **la testina che scorre.** Nel modello attuale `patch_size = 1`: ogni
  coefficiente selezionato è già un token, non c'è una finestra che scivola.
  L'animazione usa la testina per *mostrare* l'ordine di emissione — resta
  vera la sostanza (quali coefficienti diventano token e in che ordine).
- **gli embedding colorati.** La striscia heat è costruita con una proiezione
  a pesi casuali applicata alle feature vere (valore, Fourier della posizione,
  band embedding): illustra la forma del tensore, non i pesi addestrati.

## Modifiche facili

- **testi in italiano**: sono tutti nelle funzioni `s_*`, uno per riga.
- **durate**: il decoratore `@scene(step, secondi)` sopra ogni scena.
- **altra anomalia**: `A0`, `A1` e l'ampiezza in `make_signal()`; l'analisi e la
  selezione si ricalcolano da sole, compreso il livello su cui si zooma
  (`ZOOM_L` è dedotto, non scritto a mano).
- **scena risultati**: quando la run ufficiale chiude, si aggiunge una scena
  finale con la riga confrontabile con la Table 1.
