# RISULTATI L3 — GRPO multimodale su va1 (punto 5: `va1 SFT` vs `va1 + RL`)

Stato: 11/09/2026, run principale e test conclusi; repliche L4 fermate (§6).
Artefatti: `integrity/artifacts/l3_longrun_20260910/` (commit `a802658`); smoke in
`integrity/artifacts/l3_smoke_20260910/`. Protocollo e regole dichiarate prima della run:
`l3/PROTOCOL-L3.md`; verdetto sullo smoke: `l3/review/CODEX-SMOKE-VERDICT.md`.

## 1. Cosa si è misurato

Pipeline `serie → WaveletTSEncoder va1 → proiettore → embedding → Qwen3.5-4B (LoRA) → politica RL`:
i coefficienti wavelet entrano nel modello come embedding (percettore del Livello 1), non come
testo. GRPO single-turn in puro HF/PyTorch sulla stessa architettura di va1; il kit `l2/`
(Qwen3.5-2B testuale via vLLM) non è coinvolto. Nessun tool, nessun ciclo agentico (punto 4
non iniziato).

| voce | valore |
|---|---|
| checkpoint iniziale = reference KL | va1 `epoch_5_end.pt`, sha256 `cd7238ae…2531` (riproduzione esatta del test 0,2744 verificata il 10/09) |
| parametri allenabili | solo i 32 tensori LoRA (1,84 M); encoder e proiettore congelati |
| reward | metrica primaria ufficiale della categoria (`extract_answer`/`evaluate_answer`); non parsabile → 0 (mai successo: 4800/4800 parsabili) |
| loss | policy gradient con ratio clippato (0,2), vantaggi normalizzati nel gruppo G8, KL k3 token-level sui token di risposta β 0,01, AdamW lr 1e-5, decay 0, clip 1,0 |
| campionamento | T 0,7, top_p 0,95, `max_new_tokens` 500 (= eval va1) |
| passo | 2 prompt × G8 = 16 rollout sequenziali, un aggiornamento per passo |
| dati RL | solo train split (4488 righe, sha `b8571d88…`); pool di 40 prompt (8 per categoria) filtrati una volta su 138 candidati / 1104 rollout tenendo quelli con varianza di reward > 0 (seed 20260911) |
| validation | probe80 ogni 25 passi (solo monitor); val520 intera greedy a 100/200/300/400; selezione = max primary su val520; stop dopo 2 val520 senza +0,005 |
| test | 800 righe pinnate (HF `df363a46`), **una sola lettura** del candidato selezionato, bootstrap appaiato 10.000 (seed 20260911), riferimento = JSONL va1 del 03/09 |
| stack | A40-24Q vGPU, venv L1 (torch 2.6.0+cu124, transformers 5.3.0.dev0, peft 0.18.1), bf16 AMP, sdpa, grad ckpt |
| kit | repo `47847fa` (kit Codex `bdff739` + fix percorso parquet); protocollo congelato prima della run |

## 2. Smoke (10/09, gate a-d)

Identità con va1 esatta su 5 prompt di train (token, testo e log-prob identici); gate memoria
16ch × 131072 a 11.678 MiB; 20 passi × G4 senza errori; probe80 dopo 20 passi 0,311 (va1 0,283).
Verdetto Codex: PASS, con le correzioni poi applicate nel kit della run lunga (decay 0, KL k3
in fp32 con `expm1`, log per rollout, catena SHA delle eval, resume).

## 3. Run principale (seed 20260911)

| voce | valore |
|---|---|
| passi | 300 (early stop per regola), 4800 rollout, 4800/4800 parsabili |
| gruppi informativi (varianza di reward > 0) | 508/600; passi con gradiente di policy 299/300 |
| reward medio per blocchi di 50 passi | 0,298 → 0,339 → 0,373 → 0,392 → 0,405 → 0,430 |
| KL verso va1 per blocchi di 50 passi | 0,0017 → 0,0018 → 0,0033 → 0,0055 → 0,0050 → 0,0080 |
| max \|log π − log π_old\| pre-update | 0,0 (esatto) |
| picco memoria | 11.888 MiB; 56.334 s di passi (15,6 h) + eval |
| val520 | step 100 **0,3327**, 200 0,3237, 300 0,3222 (va1 ep5: 0,3105) → selezione congelata allo step 100, sha `2a927aa0…` |
| probe80 (solo monitor) | 0,269 / 0,282 / 0,282 / 0,291 / 0,278 / 0,273 / 0,304 / 0,299 / 0,292 / 0,277 / 0,293 / 0,286 |

Val520 per categoria (RL step 100 | va1 ep5): localize 0,240 | 0,244; classify 0,091 | 0,090;
MMag 0,582 | 0,578; all_channels 0,392 | 0,358; lead_lag 0,358 | 0,238.

## 4. Test 800 (una lettura), confronto appaiato

| categoria | n | va1 | va1 + RL (step 100) | Δ | CI 95% |
|---|---:|---:|---:|---:|---|
| **primary** | 800 | 0,2744 | **0,2869** | **+0,0125** | [+0,0003, +0,0253] |
| classify_with_evidence | 200 | 0,0642 | 0,0740 | +0,0098 | [−0,0069, +0,0274] |
| lead_lag_with_magnitude | 100 | 0,1664 | 0,2688 | **+0,1024** | [+0,0378, +0,1715] |
| localize | 200 | 0,2410 | 0,2327 | −0,0083 | [−0,0290, +0,0137] |
| localize_all_channels | 100 | 0,2613 | 0,2473 | −0,0140 | [−0,0453, +0,0140] |
| measure_magnitude | 200 | 0,5783 | 0,5828 | +0,0045 | [−0,0144, +0,0234] |

Catena di integrità: sha del checkpoint composto identico prima e dopo l'eval
(`2a927aa0…`), 800/800 righe, 0 errori, appaiamento esatto per `(id, category, ground_truth)`.

## 5. Lettura (onesta)

**Il guadagno sul test è interamente l'artefatto di lead_lag.** Distribuzione delle
predizioni sui 100 esempi lead_lag del test (gold: 35 independent / 33 lead / 32 lag):

| | va1 | va1 + RL |
|---|---|---|
| predetti "independent" | 43 | **76** |
| direzione corretta | 30 | 29 |
| "independent" corretti (credito 1,0 senza lag) | 14 | 26 |
| primary lead_lag | 0,166 | 0,269 |

Lo scorer ufficiale assegna 1,0 a un "independent" corretto senza chiedere il lag, mentre per
lead/lag richiede direzione giusta e lag entro il 5%. GRPO con reward = metrica ufficiale ha
imparato il default sicuro: la direzione non è migliorata (29 vs 30), sono aumentati solo i
casi in cui rispondere "independent" paga. Sulla validation lo stesso: 65/75 "independent"
allo step 100 contro 40/75 di va1. È lo stesso meccanismo per cui i TSLM pubblicati segnano
0,350 e m30 0,340.

Nelle categorie che motivano il Livello 3, localizzazione fine e multicanale, la policy RL è
leggermente sotto va1, entro il rumore. Classify e MMag ferme. Quindi: **GRPO single-turn su
reward ufficiale, 100 passi × 2 prompt × G8 dalla policy va1, non migliora la percezione né la
localizzazione; sfrutta un difetto dello scorer.** Il +0,0125 complessivo (CI che sfiora lo
zero) non va presentato come miglioramento.

Cosa si è imparato di utile:
- il loop RL multimodale funziona (identità esatta a init, rollout con embedding, KL, resume,
  catena SHA) e costa ~190 s/passo su A40 con 16 rollout;
- il segnale di reward è sparso: senza il filtro del pool il 60% dei gruppi G4 era degenere;
  con G8 e pool filtrato 508/600 gruppi informativi;
- la KL cresce lentamente (0,002 → 0,008 in 300 passi) e la val520 cala dopo lo step 100:
  la policy si allontana da va1 senza guadagnare;
- **la reward va disegnata perché non premi il default**: per lead_lag, ad esempio, credito
  pieno solo con direzione e lag corretti anche quando la risposta è "independent" (o metrica
  di direzione separata), altrimenti qualunque RL ripete questo risultato.

## 6. Repliche (fermate per ordine di Jacopo, 11/09 16:50Z)

Le due repliche (seed 20260912, 20260913) sono state interrotte durante la prima, dopo 17
passi (`L4/STOPPED_BY_ORDER.txt`): il braccio senza zoom è solo il controllo e la
dispersione fra seed non cambia la lettura. Nessun test sulle repliche. GPU liberata per il
punto 4 (`l3/review/brief_l4_zoom_codex.md`).

## 7. Budget e deviazioni

- Ore A40: smoke ~0,4 h; L0 0,16 h; L1 0,6 h; L2 15,6 h di passi + ~2,5 h di eval; L3 0,6 h;
  repliche ~15 h l'una (in corso). Costo a 0,86 $/h: ~17 $ fino al test, ~45 $ con le repliche.
- Deviazioni dichiarate: T 0,7 fissa senza pilot (decisione Jacopo), decay 0 in RL, pool
  filtrato train-only, selezione su val520 con la metrica ufficiale (che include l'artefatto),
  early stop a 300, repliche senza test.
- Codice: `l3/grpo.py`, `l3/node_l3_longrun.sh`, `l3/paired_eval_l3.py`, config
  `l3/configs/l3_longrun.yaml`; test CPU 21/21. Da ripristinare a fine catena il test-guard
  su `--split test` limitato alla fase L3.
