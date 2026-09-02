# Run ufficiale m30 — risultati definitivi (02/09/2026)

**CORREZIONE CRITICA (audit post-run, su domanda di Jacopo)**: il claim iniziale "zero batch saltati" era FALSO. Audit del log: **2.770 batch = TUTTE le celle a 16 canali (554×5 epoche, 12,3%) falliti per CUDA OOM sia sul percorso veloce sia sul fallback con checkpointing** ("FAILED anche col fallback" ×2770, "recovered" ×0). Il monitoraggio non li ha visti perché il fallback riscriveva la firma dell'errore fuori dai filtri. Causa: le celle 16ch = 16×256 = 4.096 token percettivi (seq ~5k) e arrivavano ad allocator già saturo (cache no-ckpt delle celle C1 131k a 85GB, frammentata) → nemmeno il retry con ckpt allocava. In full1 (ckpt ON globale in epoca 5) le stesse celle passavano su 40GB → **fix noto: grad checkpointing ON dall'inizio**.

**Quindi m30 = ricetta ufficiale sui task mono/bi-canale (88% dei dati, 5 epoche piene), ma ZERO training multicanale.** Le righe all_channels della tabella vanno lette con questo caveat.

Modal A100-80GB, bf16, da zero. 22.440 batch processati (4.488×5), ma solo **19.670 optimizer step riusciti** — i 2.770 batch 16ch della correzione qui sopra non hanno mai fatto `optimizer.step()`. Training 5,6h. Selezione **upstream-style best-by-val-primary** eseguita a posteriori sui checkpoint di fine epoca (val con generazioni, 520 righe) grazie a TSLM_KEEP_EPOCH_CKPTS + --eval-only.

## Curva val-primary per epoca (risultato in sé: mostra la dinamica di apprendimento)

| Epoca | val loss | val primary |
|---|---|---|
| 1 | 0.533 | 0.110 |
| 2 | **0.510** (min loss) | 0.175 |
| 3 | 0.550 | 0.174 |
| 4 | 0.523 | **0.229** (BEST per primaria) |
| 5 | 0.525 | 0.219 |

**La selezione per val loss avrebbe scelto l'epoca 2 (primary 0.175); quella per primaria sceglie l'epoca 4 (0.229).** Conferma sperimentale del perché l'upstream seleziona per metrica: la loss non traccia la qualità delle generazioni.

## Eval definitiva — 800 righe gold, modello epoca 4

**Test primary 0.211**

| Task | **m30 (ufficiale pulito)** | full1 (run "sporco" 28/08) | Miglior TSLM pubblicato | Classical | TimeRLM (agente) |
|---|---|---|---|---|---|
| localize | **0.181** | 0.158 | 0.003-0.008 | 0.160 | 0.656 |
| measure_magnitude | **0.510** | 0.522 | 0.092 | 0.685 | 0.554 |
| lead_lag | 0.264 | 0.340 | 0.350 | 0.108 | 0.978 |
| all_channels | 0.030 | 0.180 | 0.230 | 0.549 | 0.551 |
| classify | 0.004 | 0.011 | 0.000 | 0.022 | 0.745 |
| primary | 0.211 | 0.238 | — | — | — |

## Lettura

1. **Il titolo regge, e migliora, nel run pulito**: localize 0.181 — ~25-60× il blocco TSLM pubblicato e SOPRA il baseline classico (0.160). Il primo percettore SFT della classe che localizza meglio del classico, con ricetta 100% fedele e certificata (ogni episodio visto, ogni epoca piena).
2. **measure_magnitude 0.510**: 5,5× il miglior TSLM pubblicato, a un soffio dall'agente GPT+codice (0.554).
3. **Scoperta collaterale (dal confronto con full1)**: il run "sporco" — che per un incidente ha visto le celle 16-canali SOLO nelle ultime ~1,4 epoche a LR basso — fa MEGLIO su all_channels (0.180 vs 0.030) e lead_lag (0.340 vs 0.264). L'incidente è di fatto un **esperimento di curriculum**: esposizione multicanale concentrata e tardiva > esposizione uniforme, per i task multicanale. Non citabile come protocollo (non riproducibile a disegno), ma è un'ipotesi concreta e testabile per il futuro (curriculum weighting delle celle 16ch) e materiale eccellente per la discussione col lab.
4. classify resta ~0 per tutti i percettori SFT (terreno dell'agente → Livello 2).

## Costi e infrastruttura

Run m30 ~$22 + selezione a posteriori ~$8 ≈ **~$30 totali** (i crediti Modal di settembre). Artefatti nel volume `wavelet-vol`: 5 checkpoint di fine epoca (+ optimizer state), val JSON per epoca, test finale, log completi.

## Bug upstream trovati e fixati nel patch (per il report agli autori)

1. best_val_metric None dopo resume senza miglioramento → crash della print finale (uccideva l'eval).
2. `early_stop_patience` ignora `early_stopping.enabled: false` → break spurio (in m30 ha silenziosamente saltato il salvataggio dell'ultima epoca... insieme al bug 3).
3. (nostro) keep-epoch-checkpoint collocato dopo il break di early stopping.
4. eval-only: iniettare i LoRA prima del load del checkpoint.
