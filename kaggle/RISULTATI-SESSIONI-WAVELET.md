# Catena di sessioni Kaggle — variante wavelet (Livello 1)

Strategia dal 2026-08-27: su Kaggle si addestra solo la variante wavelet, a sessioni concatenate (ogni kernel monta l'output del precedente e riprende weights-only dal best_model.pt). A fine sessione il kernel valuta sul test e si giudica la presentabilità contro il blocco TSLM SFT della Table 1.

**Nota di comparabilità**: da s2 in poi val/test sono estrazioni FISSE (Random(42) dedicato, 200 righe ciascuna con max_samples=2000 — che per questo va tenuto a 2000 in ogni sessione). I numeri di s1 (budget) usavano un'estrazione diversa da 150 righe: stessa distribuzione, confronto solo indicativo.

## Tabella per sessione (test primary per categoria)

| Sessione | Step fatti (cum.) | train loss | val loss | test loss | primary | localize | classify | MMag | all_ch | lead_lag |
|---|---|---|---|---|---|---|---|---|---|---|
| s1 budget (150 righe) | 1500 (1500) | 0.457 | 0.389 | 0.402 | 0.069 | 0.000 | 0.000 | 0.119 | 0.048 | 0.208 |
| s2 (200 righe fisse) | 1335 (2835) | 0.406 | 0.391 | 0.399 | 0.064 | 0.000 | 0.000 | **0.186** | 0.000 | 0.104 |
| s3 (200 righe fisse) | 1773 (4608) | 0.404 | 0.389 | 0.396 | **0.115** | 0.000 | 0.000 | **0.268** | **0.091** | 0.237 |

Riferimenti full-budget (Table 1, 5008×5): blocco TSLM SFT — localize 0.003-0.008, classify 0.000, MMag 0.079-0.092, all_channels 0.230 (OpenTSLM; ChatTS OOM), lead_lag 0.350. Classical: 0.160/0.022/0.685/0.549/0.108. TimeRLM: 0.656/0.745/0.554/0.551/0.978.

## Lettura dopo s3 (2026-08-28)

- **Primary quasi raddoppiato sulle stesse 200 righe: 0.064 → 0.115.** La catena migliora monotonicamente sulle metriche che contano.
- **MMag 0.268 = ~3× il miglior TSLM full-budget pubblicato (0.092)** e ~metà di TimeRLM agentico (0.554).
- **all_channels ha rotto lo zero (0.091)**: la strada verso OpenTSLM 0.230 è aperta; è la categoria dove ChatTS pubblicato segna OOM.
- **lead_lag 0.237, in marcia verso l'asticella 0.350** (s2: 0.104).
- localize/classify ancora 0 (come tutti i TSLM SFT pubblicati: 0.000-0.008).
- Cumulato 4608 step ≈ 18% della ricetta ufficiale. VM di s3 ancora lenta (~17s/step): budget 8.5h → 1773/2000 step.
- Presentabilità: MMag ✓✓, all_ch in crescita ma sotto 0.230 ✗, lead_lag sotto 0.350 ✗.

## Lettura dopo s2

- **Resume verificato**: la val loss all'avvio di s2 (0.3889) coincide con quella finale di s1 — la catena di checkpoint funziona.
- **Il modello continua a imparare su dati freschi**: train loss 0.457→0.406 su un subset mai visto (seed 1042).
- **measure_magnitude 0.186**: già il doppio del miglior TSLM full-budget pubblicato (0.092). È la categoria "percezione pura" — coerente con la tesi.
- **localize/classify a 0**: come TUTTI i TSLM SFT pubblicati (0.000-0.008). Non è la debolezza nostra, è la debolezza della classe; il localize vero è il territorio del Livello 2 (agentico).
- **lead_lag 0.104 e all_channels 0.000**: sotto l'asticella (0.350 / 0.230). Servono più step; il cumulato è ~2.8k step contro i ~25k della ricetta ufficiale (~11%).
- **Presentabilità: non ancora.** Criterio: MMag sopra il blocco TSLM ✓, lead_lag ≥0.35 ✗, all_channels >0 ✗.
- Nota operativa: la VM di s2 andava a ~15s/step (vs ~9 di s1) → il budget di 5.5h ha fermato il training a 1335/2000. Da s3 il budget è 8.5h.

## Meccanica per la sessione N+1

1. Copiare `s<N>/` → `s<N+1>/`; in `kernel_run.py` aggiornare `SESSION`, `SUBSET_SEED` (+1), eventualmente `TRAIN_BUDGET_S`.
2. In `kernel-metadata.json`: id/title `timerlm-wavelet-s<N+1>`, `kernel_sources: ["jacopodabundo/timerlm-wavelet-s<N>"]`.
3. Nuovo `patch/wavelet_precise_s<N+1>.yaml` (run_name aggiornato, max_samples SEMPRE 2000) + voce OVERLAY.
4. `kaggle datasets version` sul patch, attendere status "ready", poi `kaggle kernels push` dalla cartella s<N+1>.
5. Se la sessione precedente è morta senza best_model.pt, il fallback trova latest/periodic; se l'output è vuoto, montare in kernel_sources anche la sessione ancora precedente.
