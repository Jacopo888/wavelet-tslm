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

## Lettura (RISCRITTA dopo l'integrity pass del 02/09 sera — vedi integrity/REPORT.md)

1. **localize 0.181 NON dimostra localizzazione** (decomposizione sui
   per-esempio, input ufficiali pinnati): la confusion è TP131/FP3/FN30/TN36
   — **detection di presenza genuina** (precision 0.978, recall 0.814,
   accuratezza 0.835, sopra entrambi i null: sempre-assente 0.195,
   sempre-presente 0.805) — ma l'**IoU sui 161 positivi è 0.0017**. Il
   primary è quasi tutto credito dei negativi (36/200=0.180). Nota: i TSLM
   pubblicati (0.003-0.008) stanno molto SOTTO il null 0.195 (allucinano
   anomalie); il confronto va comunque scoped alla famiglia Chronos+4B
   (Toto SFT 32B in Table 1 va citato a parte, con nota di classe).
   Fenotipo "rileva ma non piazza" coerente con l'ipotesi bf16-posizioni
   (→ variante v2 fp32, esperimento separato).
2. **measure_magnitude 0.510 CI95 [0.463, 0.556]**: il risultato forte e
   stabile — 5,5× il miglior TSLM pubblicato, vicino all'agente GPT+codice
   (0.554).
3. **Il "curriculum accidentale" di full1 è SGONFIATO**: il suo all_channels
   0.18-0.20 è interamente accuratezza sui 23 gold vuoti (18-20/23 corretti)
   con **F1 0.000 sui 77 gold non vuoti**. L'esposizione 16ch tardiva ha
   insegnato la calibrazione a rispondere "nessuna anomalia", NON la
   localizzazione multicanale. lead_lag: entrambi i run ≈ o sotto il null
   0.350 — nessuna evidenza di skill lead/lag.
4. classify resta ~0 per tutti i percettori SFT (terreno dell'agente → Livello 2).

**Numeri certificati sul test HF pinnato `df363a46` (02/09 sera, eval-only,
800/800, JSONL per-esempio)**: m30-ep4 primary **0.2156** (identico allo
storico su tutto tranne lead_lag 0.3043 — le serie rigenerate differivano
dall'HF solo nelle 100 righe lead_lag); full1-best-ep5 **0.2356**; full1
post-top-up **0.2417** (22.440/22.440 step — candidato full1 di
riferimento). Tabella completa in `integrity/REPORT.md`.

## Costi e infrastruttura

Run m30 ~$22 + selezione a posteriori ~$8 ≈ **~$30 totali** (i crediti Modal di settembre). Artefatti nel volume `wavelet-vol`: 5 checkpoint di fine epoca (+ optimizer state), val JSON per epoca, test finale, log completi.

## Bug upstream trovati e fixati nel patch (per il report agli autori)

1. best_val_metric None dopo resume senza miglioramento → crash della print finale (uccideva l'eval).
2. `early_stop_patience` ignora `early_stopping.enabled: false` → break spurio (in m30 ha silenziosamente saltato il salvataggio dell'ultima epoca... insieme al bug 3).
3. (nostro) keep-epoch-checkpoint collocato dopo il break di early stopping.
4. eval-only: iniettare i LoRA prima del load del checkpoint.
