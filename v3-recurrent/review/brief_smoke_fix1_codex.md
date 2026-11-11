# Brief per Codex — fix 1 dello smoke A40 (`v3-recurrent/node/node_smoke_v3.sh`)

Repo: `<cartella-locale>\wavelet-tslm`. Non committare, niente rete. Modifica
direttamente i file (tu sei l'autore); io li spedisco sul nodo e li eseguo.

## Cosa è successo sul nodo (05/09 16:25 UTC)
`bash ~/node_smoke_v3.sh` fallisce subito nello stage `build_deterministic_subset`:

```
  File "<stdin>", line 38, in <module>
ValueError: min() iterable argument is empty
{"status": "FAILED", "stage": "build_deterministic_subset", "reason": "unexpected command failure at line 160 (exit 1)"}
```

Riga 38 del blocco Python = `hard16 = min(... row["category"] == "all_channels" and row["n_channels"] == 16 ...)`.
Il nome della categoria assunto non esiste nel parquet.

## Fatti misurati sul parquet reale (`anomalyxl-train/anomalyxl-precise/train.parquet`, 4488 righe)
- colonne: `id, tier, category, length, n_channels, n_channels_a, n_channels_b, question, options,
  answer, semantic_class, coupling_kind, coupling_lag, injection_kinds, injection_channels,
  injection_starts, injection_ends, channel_names, channel_values, seed`
- `category` × `n_channels` (conteggi): `classify_with_evidence`×1 = 1142,
  `measure_magnitude`×1 = 1121, `localize`×1 = 1120, **`localize_all_channels`×16 = 554**,
  `lead_lag_with_magnitude`×2 = 551.
- `length`: 131072 = 1400 righe, 32768 = 1388, 65536 = 856, 16384 = 844 (max = 131072).
- La riga `[Coverage]` per categoria di `train.py` stampa `    {category}: ok=N failed=M` con il nome
  reale della categoria (quindi `localize_all_channels: ok=1 failed=0`): la regex attuale
  `all_channels: ok=(\d+) failed=0` matcha per sottostringa, rendila esatta sul nome vero.

## Cosa fare
1. Usa i nomi reali delle categorie. Per il caso "16 canali" scegli la riga `localize_all_channels`
   con `n_channels == 16` e **la lunghezza massima disponibile (131072 se esiste)**: è il vero caso
   peggiore di memoria (16 × 131072 → 4096 token percettivi + testo). Poi, se quella riga non ha già
   `length == 131072`, aggiungi la prima riga da 131072; poi i filler fino a 4 righe come ora.
2. Aggiorna coerentemente il manifest del sottoinsieme (`hard_16`, `hard_131072`, `checks`) e
   `CODEX-SMOKE-NOTES.md` (nomi delle categorie, criterio di scelta).
3. Rivedi se altri punti dello script assumono nomi o valori non verificati (es. campi del JSON
   di `--eval-only`: `num_batches`, `samples_evaluated`, `eval_errors`, `split` esistono in
   `train.py`; `best_model.pt` e `[SkipFinalTest]` esistono). Non cambiare altro senza motivo.
