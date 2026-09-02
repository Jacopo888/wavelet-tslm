# Risultato àncora ChatTS — run budget Kaggle (2026-08-27)

Kernel `timerlm-chatts-budget`, COMPLETE alle 11:13 (log recuperato via `kernels_logs` il 2026-08-27 pomeriggio, dopo l'interruzione della sessione precedente per limite settimanale).

## Esito in una riga

**ChatTS non ha eseguito un solo passo di training: tutti i 1500 batch sono andati in CUDA OOM** ("Error at batch 0: CUDA out of memory", ripetuto 1500/1500 — `train/epoch_loss = 0.0`). La stessa ricetta, sullo stesso hardware, con la variante wavelet aveva completato l'epoca intera senza perdere un batch.

## Numeri (test su 150 righe gold, best "checkpoint" = modello mai addestrato)

| Metrica | Wavelet (addestrata) | ChatTS (0 step riusciti) |
|---|---|---|
| train loss (fine epoca) | 0.457 | 0.0 (nessun batch completato) |
| val loss / val primary | 0.389 / — | 2.194 / 0.032 |
| test loss | 0.402 | 2.213 |
| **test primary** | **0.069** | **0.053** |
| lead_lag_with_magnitude | 0.208 | 0.391 |
| measure_magnitude | 0.119 | 0.000 |
| localize | 0.000 | 0.037 |
| localize_all_channels | 0.048 | assente dalle metriche |
| classify_with_evidence | 0.000 | 0.000 |

Nota: i numeri ChatTS sono di fatto la baseline "percettore non addestrato" (proiettore a init casuale + Qwen3.5-4B base). Il lead_lag 0.391 viene dal formato della risposta indovinabile dal testo, non dalla percezione.

## Perché è il numero-tesi della proposta

I token percettivi di ChatTS crescono con la lunghezza della serie (L/16 patch per canale: celle da 16k→131k campioni = 1.024→8.192 token *per canale*, fino a 16 canali). Su 2×T4 (14.5 GiB l'una), con la stessa configurazione salva-memoria che alla wavelet basta e avanza (fp32 distribuito, SDPA, gradient checkpointing, embeddings congelati), non entra in memoria nemmeno un batch. Il budget fisso di 256 token della variante wavelet digerisce tutte le celle, incluse quelle da 131k. **"L'allocazione adattiva della risoluzione batte il patching uniforme" qui non è una differenza di punteggio: è la differenza fra addestrabile e non addestrabile a parità di hardware.**

Caveat onesto da riportare sempre accanto: su GPU più grandi ChatTS si addestra (il paper lo fa); il claim corretto è sul *costo di memoria in funzione di L*, non "ChatTS non funziona". Il loro stesso config ammette OOM sulle celle L=131k.

## Fonte

Log completo del kernel: scaricabile con `KaggleApi().kernels_logs('jacopodabundo/timerlm-chatts-budget')`. Metriche finali: `Summary: {'test_loss': 2.2130, 'test_primary': 0.0528, 'test_category_primary': {'measure_magnitude': 0.0, 'classify_with_evidence': 0.0, 'localize': 0.037, 'lead_lag_with_magnitude': 0.3911}}`.

## Prossimi passi possibili

1. **Àncora addestrabile a parità di hardware**: riprovare ChatTS con serie sottocampionate/troncate al suo massimo digeribile (misura il degrado da perdita di risoluzione) — oppure Flamingo (64 latenti fissi, profilo memoria simile al nostro; lo smoke era crashato per disco, fix già noto).
2. **Ablazioni wavelet** (quota Kaggle rigenerata a inizio settimana): budget token, con/senza coarse, Haar vs Db4.
3. **Fedeltà piena** (5008×5 epoche) su GPU a noleggio: ~15-20h di A100, ordine 30-60€.
