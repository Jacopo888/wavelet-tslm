# Brief per Codex — L3 punto 5: kit della run lunga (GO di Jacopo, 10/09/2026)

Stato: versione FINALE (Claude, 10/09) sul verdetto `CODEX-SMOKE-VERDICT.md` (commit `54e2f24`).
Decisioni di Jacopo: **niente pilot di temperatura (T = 0,7 fissa)**, **filtro del pool sì**,
**run principale + 2 repliche con seed dichiarati**, GO sul disegno di Codex. Codex scrive il
codice, Claude controlla il diff, esegue test locali, spedisce sul nodo e lancia la catena.
Nessuna domanda aperta: dove serve una scelta minore, Codex decide e la documenta in
`PROTOCOL-L3.md` (sezione "Decisioni run lunga").

## 1. Punto di partenza e invarianti

Smoke `fa1e721` PASS a-d (artefatti `integrity/artifacts/l3_smoke_20260910/`). Restano fissi:
architettura, percettore, scorer, reward ufficiale (non parsabile → 0, `format_penalty 0`),
`temperature 0.7`, `top_p 0.95`, LR LoRA `1e-5`, clip 1.0, KL k3 β 0.01 sui soli token di
risposta, reference = va1 `epoch_5_end.pt` (sha `cd7238ae81da1afca9f225a99e9af4261be829798a39985a427d88a098bf2531`),
solo i 32 tensori LoRA allenabili (encoder/proiettore congelati), prompt SOLO dal train split
(4488 righe, sha `b8571d88…`), validation solo per monitor/selezione, **test 800 una volta sola**
alla fine sul solo candidato selezionato. Vincoli nodo: A40-24Q, venv L1 `~/venv`, batch 1,
`TSLM_GRAD_CKPT=1`, bf16 AMP, sdpa, niente `expandable_segments`, niente multiprocesso.
`kaggle/patch/` resta intatto; il kit smoke (`l3_smoke.yaml`, `node_l3_smoke.sh`) resta
funzionante e i suoi test continuano a passare.

## 2. Le 7 correzioni del verdetto (obbligatorie)

1. **Modalità long-run** in `l3/grpo.py` + `l3/configs/l3_longrun.yaml` (eredita
   `wavelet_precise_va1.yaml`): `prompts_per_step = 2` elaborati in sequenza (feature del
   primo rilasciate prima del secondo; MAI batch parallelo dei due prompt), `G = 8` sequenziali,
   vantaggi normalizzati **per gruppo**, loss = media delle due loss di prompt, un solo
   `optimizer.step` per update; `steps ≤ 400`. `validate_l3_config` distingue i due contratti
   (smoke / longrun) senza indebolire quello dello smoke.
2. **Checkpoint periodici e resume**: ogni 25 step salvataggio atomico (LoRA-only + composto a
   42 chiavi) con SHA nel manifest, `optimizer_state`, step, cursore del pool, stati RNG
   (torch CPU/CUDA, python); `--resume <run_dir>` riprende dall'ultimo checkpoint valido e
   dimostra continuità (test CPU: rollout non duplicati né saltati, contatori cumulativi
   coerenti, stesso seed → stessa sequenza di prompt).
3. **KL stabile vicino a zero**: `expm1(delta.float()) - delta.float()`, controllo finitezza e
   non-negatività entro tolleranza (≥ −1e-7), test su delta piccoli.
4. **`weight_decay: 0`** esplicito nella sezione `rl` (va1 deve essere punto fisso a vantaggio
   nullo); test: un update a vantaggi zero e KL zero non muove i pesi.
5. **Log per rollout** (`rollouts.jsonl`): step, prompt, indice train, categoria, reward primario
   e di ottimizzazione, parsabilità, token di risposta, sha del testo; per step
   `informative_groups`, `policy_informative_steps`, e norme separate del gradiente policy e
   KL (due backward separati sono ammessi se il costo è accettabile; altrimenti motivare).
   `positive_grad_steps` non è più un criterio di gate.
6. **Catena SHA dell'eval**: SHA del checkpoint composto calcolato prima e dopo ogni eval
   (probe80/val520/test) e scritto nel JSON di esito; confronto contro il manifest del run.
7. **`l3/paired_eval_l3.py`** (nuovo; `v3-recurrent/audit/paired_eval.py` resta intatto):
   due bracci `reference`/`candidate` da due JSONL per-riga, split `test` (o `val`), identità
   ordinate `(id, category, ground_truth)` con copertura 800/800 (o 520/520), finitezza,
   bootstrap appaiato 10.000 con seed dichiarato, CI95 su primary e per categoria, output JSON
   + tabella markdown. Test CPU su JSONL sintetici.

## 3. Catena sul nodo — `l3/node_l3_longrun.sh` (fail-closed, tmux, marker per fase)

Ogni fase scrive `<fase>/summary.json` + marker `OK`/`FAILED`; la fase successiva parte solo
con l'`OK` della precedente; una cartella di fase esistente = errore (mai riusata); log per
fase; `RUN_MANIFEST.json` con sha del checkpoint iniziale, del train parquet, del kit
(`KIT_COMMIT.txt`) e del config risolto. GPU libera pretesa all'avvio. Riga di stato per
step (categoria, reward, informative, KL, picco, s/step) nel log per il monitor di sessione.

| fase | contenuto | budget |
|---|---|---|
| **L0 gate memoria long-run** | 1 update su DUE prompt 16ch × 131072 del train (indici 3804 e 3930) con G8 sequenziale: picco < 22.528 MiB, valori finiti, contatori completi | ~7 min |
| **L1 pool filtrato (train only, T 0,7)** | candidati stratificati per categoria e lunghezza (seed `20260911`), G8 ciascuno, tenuti quelli con `reward_std > 0`; obiettivo ≥ 8 prompt ammessi per categoria, bilanciati per lunghezza; **tetto 1600 rollout** (200 candidati): se il tetto non basta per una categoria, la fase FALLISCE (marker + motivo) e non si prosegue. Lista, seed, sha nel manifest; il pool è congelato prima del training; nessun rejection sampling online | ≤ 3,2-3,9 h |
| **L2 run principale** (seed `20260911`) | ≤ 400 step × 2 prompt × G8 = 6400 rollout, prompt dal pool in ordine ciclico deterministico dal seed; checkpoint ogni 25; probe80 ogni 25 step **solo monitor** (`TSLM_EVAL_STRICT=0`, config probe80); **val520 intera greedy strict** a 100/200/300/400; selezione = max primary su val520 (a parità entro 0.002 lo step precedente; va1 ep5 sulla val intera = 0,3105 è il riferimento); stop a 400 o dopo 2 val520 consecutive senza +0.005 sul best; hard stop su NaN/Inf, rollout persi, sha/invarianti LoRA errati, picco ≥ 22.528 MiB, parsabilità mobile < 95% o gruppi informativi < 50% per 2 finestre consecutive da 50 gruppi | 13-16 h + ~2,1 h val |
| **L3 test del candidato** | UNA sola eval greedy del candidato selezionato sul test 800 pinnato (`--split test` è ammesso SOLO qui, dentro questo script, dopo `L2/OK` e con il candidato scritto in `L2/selection.json`); confronto appaiato con il JSONL va1 già archiviato sul nodo (`results/…/output_logs/evalonly_test_epoch_5_end_epoch_0.jsonl`, 800 righe) tramite `paired_eval_l3.py`; nessuna nuova selezione | ~0,6 h |
| **L4 repliche** (seed `20260912`, `20260913`) | stessa regola, stesso pool, nessun ritocco; ciascuna con val520 ai milestone e `selection.json`; **senza test**; servono alla dispersione fra seed (riportare primary val520 al best di ogni seed) | 2 × (13-16 + 2,1) h |

Aggiornare `l3/tests/test_l3_kit.py` in modo che il guard sul test split ammetta `--split test`
solo in `node_l3_longrun.sh` nella fase L3 (o equivalente esplicito), non altrove.

Budget totale stimato: 50-60 h A40 (~$45-52 a $0,86/h). Il driver stampa a ogni fase le ore
trascorse e la stima residua.

## 4. Deliverable

`l3/grpo.py` (long-run, resume, log per rollout, KL stabile, decay 0, sha eval),
`l3/configs/l3_longrun.yaml`, `l3/node_l3_longrun.sh`, `l3/paired_eval_l3.py`, test in
`l3/tests/` (tutti i punti del §2 coperti; `py -m pytest l3/tests -q` deve passare in locale),
`PROTOCOL-L3.md` aggiornato (regole di selezione/stop dichiarate PRIMA della run, budget,
deviazioni: T 0,7 senza pilot, decay 0, pool filtrato, 2 repliche senza test).
