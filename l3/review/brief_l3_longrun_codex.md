# Brief per Codex — L3 punto 5: kit della run lunga (BOZZA, in attesa del GO di Jacopo)

Stato: bozza del 10/09/2026 (Claude) scritta sul verdetto `CODEX-SMOKE-VERDICT.md` (commit
`54e2f24`). NON lanciare Codex su questo brief prima del GO: le scelte marcate **[DECISIONE]**
sono di Jacopo. Codex scrive il codice, Claude controlla il diff, esegue test e gate sul nodo.

## 1. Punto di partenza

Smoke `fa1e721` PASS a-d (artefatti `integrity/artifacts/l3_smoke_20260910/`). Restano fissi:
architettura, percettore, scorer, reward ufficiale (non parsabile → 0), `top_p 0.95`, LR LoRA
`1e-5`, clip 1.0, KL k3 β 0.01 sui token di risposta, reference = va1 `epoch_5_end.pt`
(`cd7238ae…`), solo i 32 tensori LoRA allenabili, prompt SOLO dal train split, test 800 una
volta sola alla fine.

## 2. Le 7 correzioni del verdetto (obbligatorie, nell'ordine)

1. **Modalità long-run** in `l3/grpo.py` + config `l3/configs/l3_longrun.yaml`: `prompts_per_step
   = 2` elaborati in sequenza (feature rilasciate fra un prompt e l'altro), `G = 8`
   sequenziali, normalizzazione dei vantaggi **per gruppo**, loss = media delle due loss di
   prompt; `steps ≤ 400`; il config dello smoke resta invariato e i suoi test continuano a
   passare. Nuovo gate `c0` long-run: 1 update su DUE prompt 16ch × 131072 con G8 → picco < 22.528
   MiB (misurato; niente batch parallelo dei due prompt).
2. **Checkpoint periodici e resume**: ogni 25 step salvataggio atomico (LoRA-only + composto)
   con SHA, `optimizer_state`, step, cursore dei prompt e stati RNG (torch/cuda/python); comando
   `--resume <ckpt>` che dimostri continuità (test: rollout non duplicati né saltati, contatori
   cumulativi coerenti).
3. **KL stabile vicino a zero**: `expm1(delta.float()) - delta.float()`, controllo finitezza e
   non-negatività entro tolleranza, test su delta piccoli.
4. **`weight_decay: 0`** esplicito nella sezione `rl` (va1 deve essere punto fisso a vantaggio
   nullo); test che un update a vantaggi zero e KL zero non muova i pesi.
5. **Log per rollout**: reward primario/ottimizzato, parsabilità, token e hash del completamento;
   per step `informative_groups`, `policy_informative_steps`; norme separate del gradiente
   policy e KL (o motivare se non praticabile). `positive_grad_steps` non è più un criterio.
6. **Catena SHA dell'eval**: SHA del checkpoint composto calcolato prima e dopo ogni eval
   (probe80/val520) e scritto nell'output JSON; confronto contro il summary del run.
7. **`audit/paired_eval.py` generalizzato** (nuovo modulo in `l3/`, senza rompere il test v3):
   due bracci `reference`/`candidate`, split `test`, 800 identità ordinate `(id, category,
   ground_truth)`, finitezza/copertura, bootstrap appaiato 10k con seed, CI95 su primary e
   categorie.

## 3. Sequenza sul nodo (script `l3/node_l3_longrun.sh`, fail-closed, tmux, marker DONE/FAILED)

| fase | contenuto | **[DECISIONE]** |
|---|---|---|
| P0 pilot temperatura (train only) | 50 prompt stratificati, G8, seed fissi, T 0.7 vs 0.9 (800 rollout, ~1.6-2 h); regola: 0.9 solo se +10 punti di gruppi informativi e ≥98% parsabili | fare il pilot o fissare T 0.7 |
| P1 pool filtrato (train only) | fino a 100 prompt candidati G8 (≤800 rollout), tenuti quelli con `reward_std > 0`, ≥8 per categoria, bilanciati per lunghezza; lista+seed+SHA nel manifest; tetto 1600 rollout di calibrazione in tutto | sì/no al filtro (alternativa: pool casuale stratificato, più gruppi degeneri) |
| P2 run principale | ≤400 step × 2 prompt × G8 = 6400 rollout, ckpt ogni 25, probe80 ogni 25 (solo monitor), val520 greedy strict a 100/200/300/400; selezione = max primary su val520 (a parità entro 0.002 lo step precedente); stop a 400 o dopo 2 val520 senza +0.005; hard stop su NaN/rollout persi/hash/picco ≥ 22.528/parsabilità <95% o gruppi informativi <50% per 2 finestre da 50 | max step (400) e seed |
| P3 repliche | 2 seed aggiuntivi con la stessa regola, senza ritoccare nulla; servono alla dispersione fra seed, non alla selezione | farle (≈ +26-31 h) o no |
| P4 test | 800 righe greedy del solo candidato selezionato, una volta; confronto appaiato con il JSONL va1 già archiviato (`integrity/artifacts/va1/output_logs/`), bootstrap 10k | — |

Budget stimato (verdetto): 19-22 h A40 per una run fino al test (≈ $17-19 a $0.86/h);
49-58 h con le repliche (≈ $42-50). Ogni fase scrive JSON con contatori; nessuna fase parte
senza il marker OK della precedente.

## 4. Deliverable

`l3/grpo.py` (long-run + resume + log), `l3/configs/l3_longrun.yaml`, `l3/node_l3_longrun.sh`,
`l3/paired_eval_l3.py`, test in `l3/tests/` (tutti i punti del §2 coperti; `py -m pytest
l3/tests -q` locale), `PROTOCOL-L3.md` aggiornato (regola di selezione e stop dichiarate PRIMA
della run, deviazioni elencate).
