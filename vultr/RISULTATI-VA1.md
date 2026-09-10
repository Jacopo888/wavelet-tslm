# RISULTATI va1 — re-run Livello 1 a copertura totale (Vultr A40-24Q, 02-03/09/2026)

Run riproduttivo della ricetta del paper (5 epoche × 4.488 esempi, batch 1, LoRA r16/α32 sul
backbone Qwen3.5-4B, LR 1e-4, warmup 0.03) con il percettore wavelet, **senza perdere un solo
esempio**: è il run che sostituisce m30 (che aveva perso tutte le 2.770 celle a 16 canali) come
riferimento del Livello 1.

## Provenienza (RUN_MANIFEST.json in `integrity/artifacts/va1/`)

| voce | valore |
|---|---|
| nodo | Vultr `wavelet-l1`, vcg-a40-12c-60g-24vram (A40-24Q vGPU, driver 550.90.07 / CUDA 12.4) |
| codice | TimeRLM commit `31fcd847` + overlay `kaggle/patch/` (repo commit `9dc6324`) |
| stack | torch 2.6.0+cu124, torchvision 0.21.0+cu124 (constraints su ogni pip), transformers 5.3.0.dev0 |
| backbone | `Qwen/Qwen3.5-4B` revisione `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (passata a from_pretrained) |
| dati | HF `nz00shuuuu/AnomalyXL` rev `df363a467f855e55aaad414b0bdd58590829df9a`, sha256 verificati dal preflight strict: train 4488 / val 520 / test 800 |
| config | `wavelet_precise_va1.yaml` (seed 42), env: bf16 + AMP bf16, sdpa, GRAD_CKPT=1, REQUIRE_FULL_COVERAGE=1, EVAL_STRICT=1, SKIP_FINAL_TEST=1, budget vietati, resume solo `epoch_*_end.pt` |
| loss | memory-efficient (logits solo sulle posizioni supervisionate, bit-identica al percorso upstream: rel_err 0) — necessaria per le celle 16ch su 24 GB |
| costo | ≈ 25 $ di crediti (training ~22.5 h + selezione/test ~2.7 h a 0.86 $/h) |

## Copertura (dal log di training, `logs/va1_train_summary.txt`)

```
[Coverage] epoch 1: trained 4488, failed 0, expected 4488
[Coverage] epoch 2: trained 4488, failed 0, expected 4488
[Coverage] epoch 3: trained 4488, failed 0, expected 4488
[Coverage] epoch 4: trained 4488, failed 0, expected 4488
[Coverage] epoch 5: trained 4488, failed 0, expected 4488
```

22.440/22.440 step, EXIT:0, cinque `epoch_N_end.pt` (solo pesi allenabili + stato optimizer).
Il test interno di fine training è stato saltato di proposito: il test si fa una sola volta,
sul checkpoint scelto dalla validazione.

## Selezione upstream: val con generazioni per epoca (520/520 righe, 0 errori)

| epoca | val primary | localize | classify | MMag | all_channels | lead_lag | val loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1703 | 0.000 | 0.000 | 0.378 | 0.167 | 0.360 | 0.3854 |
| 2 | 0.2257 | 0.197 | 0.023 | 0.418 | 0.153 | 0.308 | 0.3781 |
| 3 | 0.2482 | 0.201 | 0.013 | 0.483 | 0.304 | 0.213 | 0.3567 |
| 4 | 0.2925 | 0.232 | 0.048 | 0.557 | 0.355 | 0.237 | 0.3513 |
| **5** | **0.3105** | 0.244 | 0.090 | 0.578 | 0.358 | 0.238 | 0.3478 |

Best = epoca 5 (criterio del paper: primaria sulla val). Con la copertura multicanale la curva
sale fino in fondo (m30 si fermava a 0.229 all'epoca 4).

## Test ufficiale (800/800, HF pinnato, eval-only da `epoch_5_end.pt`, eval_errors 0, 800 parsed, 0 violazioni strict)

| Categoria | **va1 ep5** | m30 ep4 (hfpin) | full1 post-top-up (hfpin) | null baseline | miglior TSLM pubblicato |
|---|---:|---:|---:|---:|---:|
| primary | **0.2744** CI [0.247, 0.301] | 0.2156 | 0.2417 | — | — |
| localize | **0.2410** | 0.1814 | 0.1755 | 0.195 (sempre assente) | 0.008 |
| classify | **0.0642** | 0.0037 | 0.0089 | — | 0.000 |
| measure_magnitude | **0.5783** CI [0.529, 0.625] | 0.5102 | 0.5326 | — | 0.092 |
| all_channels | **0.2613** | 0.0300 | 0.2000 | 0.230 (sempre vuoto) | 0.230 (OpenTSLM) |
| lead_lag | 0.1664 | 0.3043 | 0.2993 | 0.350 (sempre independent) | 0.350 |

Bootstrap appaiato sugli stessi 800 esempi (seed 20260902, N=10000):
va1 − m30 = **+0.059** CI [0.036, 0.082]; va1 − full1-topup = **+0.033** CI [0.013, 0.052]
(entrambi significativi; per categoria: all_channels +0.231 vs m30, localize +0.06/+0.066,
MMag +0.068/+0.046, classify +0.06/+0.055, lead_lag −0.138/−0.133).

### Decomposizione (integrity/DIAGNOSTICS-va1-vs-*.json)

- **localize**: confusion TP142 / FP6 / FN19 / TN33 → precision 0.959, recall 0.882,
  accuratezza sui negativi 0.846. **IoU medio sui 161 positivi 0.094** (matched 0.107),
  acc@IoU0.1 0.199, @0.3 0.143, @0.5 0.081. Confronto: m30 IoU positivi 0.0017, acc@0.3 0.000.
  La localizzazione fine non è più zero: è bassa, ma è la prima volta che il modello piazza
  finestre sovrapposte al gold su una fetta misurabile dei casi. Resta il problema aperto che
  motiva v2 (posizioni fp32) e il Livello 2.
- **all_channels**: accuratezza sui 23 gold vuoti 0.913 + **F1 medio sui 77 non vuoti 0.067**
  (m30: 0.13 / 0.000; full1: 0.870 / 0.000). Il credito non è più solo calibrazione dei vuoti,
  ma il grosso resta lì: 0.2613 contro il null 0.230.
- **measure_magnitude**: 0.578, 6.3× il miglior TSLM pubblicato (0.092), CI che non tocca
  quello di m30 dal basso solo di poco (0.529 vs 0.556): guadagno reale ma modesto.
- **lead_lag**: direction_correct 0.30 su 100 (≈ caso, 1/3), lag_acc@5% 0.19. Predizioni
  spalmate (independent 43 / lead 37 / lag 20) contro gold 35/33/32. m30 predice
  independent 67 volte (direction_correct 0.33, anch'esso al caso) e incassa il credito dei
  35 gold "independent". **Nessun run impara lead_lag**; la differenza 0.166 vs 0.304 è un
  artefatto dello scorer (il null "sempre independent" vale 0.35), non una regressione
  di capacità. Da riportare così.

## Cosa cambia nelle letture

1. Il riferimento del Livello 1 è **va1** (copertura certificata, provenienza pinnata,
   selezione come il paper). m30 e full1 restano nel report come storia e come controllo.
2. Il claim "MMag ≫ TSLM pubblicati + detection di presenza precisa a 131k" regge e si
   rafforza (0.578; precision 0.959 / recall 0.882).
3. Il claim nuovo, prudente: "con copertura multicanale completa la localizzazione fine
   smette di essere zero (IoU positivi 0.094) e all_channels supera il null e OpenTSLM" —
   con i numeri di decomposizione accanto, mai il solo primary.
4. lead_lag: dichiarare esplicitamente che è al livello del caso per tutti i modelli SFT
   di questa classe.

## Artefatti

`integrity/artifacts/va1/`: `evalq/` (5 val + FINAL + BEST_EPOCH), `output_logs/` (JSON e
JSONL per-esempio di ogni eval-only, id/categoria/gold/predizione/parse_status/metriche),
`RUN_MANIFEST.json`, `config.yaml`, `logs/` (coverage, worker fail-closed, tail dell'eval
finale, listing del run dir). Checkpoint (`epoch_1..5_end.pt`, ~3.7 MB l'uno) sul nodo in
`~/timerlm/sft/tslm/results/anomaly_xl_precise/anomaly_xl_wavelet_va1/checkpoints/`.

## Addendum 10/09/2026 — identità del checkpoint e riproduzione esatta

- Il numero di riferimento (0,2744) viene da **`epoch_5_end.pt`** (sha256
  `cd7238ae81da1afca9f225a99e9af4261be829798a39985a427d88a098bf2531`, 41.276.034 B, epoch 5,
  val_loss 0,3478), non da `best_model.pt` (selezione per val loss). Hash di tutti i 47 checkpoint
  in `integrity/artifacts/va1/checkpoints_SHA256SUMS.txt`; copia fuori nodo verificata 47/47.
- Caricamento esatto (`l3/tools/check_va1_exact_load.py`, CPU): 42 tensori nel checkpoint = 42
  tensori allenabili del modello costruito come `--eval-only` (LoRA 32, encoder 6, proiettore 4),
  3.435.520 parametri, nessuna chiave mancante/in più, forme uguali, tutto finito
  (`integrity/artifacts/va1/exact_load_ep5.json`).
- Riproduzione del test 800 con lo stesso comando del worker (`TSLM_EVAL_TAG=repro20260910`,
  A40, 26 min): primary **0,2743539445041715**, loss 0,35340173219068677, 800/800, 0 errori —
  identici a tutte le cifre al FINAL del 03/09; per categoria identici (localize 0,2410,
  classify 0,0642, MMag 0,5783, all_channels 0,2613, lead_lag 0,1664); confronto riga per riga
  dei due JSONL: 0 righe diverse su 800 per ogni campo, testo generato incluso. La generazione
  greedy in bf16 su questo stack è deterministica. JSON: `integrity/artifacts/va1/evalq/va1_repro20260910_test.json`.
