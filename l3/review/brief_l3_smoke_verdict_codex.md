# Brief per Codex — verdetto sullo smoke L3 (punto 3) e proposta del budget per il punto 5

**Solo lettura**: nessuna modifica a file di codice o config. Scrivere il verdetto in
`l3/review/CODEX-SMOKE-VERDICT.md` (nuovo file, unico output ammesso).

## Cosa è successo

Kit `fa1e721` (rev. 1 + fix1), smoke `l3/node_l3_smoke.sh` eseguito sulla A40-24Q il 10/09
(19:43-20:05 UTC) nella venv L1. Artefatti in `integrity/artifacts/l3_smoke_20260910/`:
`stage_{a,b,c0,c,d}.json`, `stages.json`, `SMOKE_OK`, `c0/` e `c/` (`RUN_MANIFEST.json`,
`steps.jsonl` per step, `summary.json`), `stage_d_probe80.jsonl` (80 righe di val),
`checkpoints_SHA256SUMS.txt`, log degli stadi. Riferimento va1 ep5 sugli stessi 80:
`integrity/artifacts/v3tied/probes/va1_ep5_valprobe80.json` (primary 0,283: localize 0,151,
classify 0,142, MMag 0,543, all_ch 0,212, lead_lag 0,124).

Lettura di Claude (da verificare, non da assumere):

- b: 5/5 identità ESATTA (token, testo, max diff log-prob 0.0, ratio esatto 1).
- c0: 16ch × 131072, G=4: 4/4 parsabili, reward 0 per tutti (gruppo a varianza zero), picco
  11.678 MiB (limite 22.528), 92 s.
- c: 20 step, 80/80 rollout parsabili, `max_abs_log_ratio` 0.0 in ogni step (uguaglianza
  esatta fra forward no_grad e con grad), KL media 1,9e-4, grad_norm max 0,995,
  `positive_grad_steps` 19/20, **12/20 gruppi a varianza zero** (reward identici nei 4
  campioni: quasi sempre tutti 0, in 3 casi tutti 1), picco 11.705 MiB, 32 s/step medi
  (12-19 s mono/bicanale, 84-101 s 16ch), 638 s totali. Per categoria (reward medio sui 16
  rollout): classify 0,0, lead_lag 0,116, localize 0,312, all_ch 0,521, MMag 0,501.
- d: probe80 dopo 20 step: primary 0,311 (localize 0,175, classify 0,165, MMag 0,567,
  all_ch 0,333, lead_lag 0,046), 80/80, 0 errori.

## Domande a cui rispondere

1. **Il gate è passato davvero?** Controllare i JSON (non il riassunto sopra): conteggi,
   finitezza, coerenza fra `steps.jsonl` e `summary.json`, hash del checkpoint iniziale nel
   manifest, che lo stadio d abbia valutato il checkpoint composto giusto (sha in
   `checkpoints_SHA256SUMS.txt` vs `stage_d.json`).
2. **`positive_grad_steps` 19/20 con 12 gruppi a varianza zero**: spiegare da dove viene il
   gradiente quando i vantaggi sono tutti zero (atteso: solo dal termine KL k3, che è ≈0 al
   passo 1 e cresce) e se questo è coerente con i valori di `grad_norm` osservati.
3. **Segnale utilizzabile**: con G=4 e T=0,7, 12/20 gruppi non danno gradiente di policy. Per
   la run del punto 5 proporre — con numeri — come aumentare la frazione di gruppi informativi
   (G più alto, temperatura, filtro dei prompt a reward degenere, più prompt per step) SENZA
   cambiare reward, scorer o architettura, e stimare il costo in s/step dai tempi misurati
   (mono/bicanale ≈ 15-19 s per 4 rollout; 16ch ≈ 85-100 s).
4. **Il +0,028 dello stadio d** (0,311 vs 0,283 sugli stessi 80) va letto come rumore o come
   segnale? Dire cosa servirebbe per distinguerli (bootstrap appaiato sugli 80, run ripetute,
   val intera) e NON trarre conclusioni dal test.
5. **Proposta di budget per il punto 5** (una tabella): step, prompt per step, G, rollout
   totali, ore A40 stimate, regola di selezione del checkpoint sulla validation (val intera
   520 con generazioni? probe80? ogni quanti step?), criterio di stop, cosa dichiarare come
   deviazione. Vincoli: 24 GB (picco misurato 11,7 GB → c'è margine: dire se G=8 o 2 prompt/step
   ci stanno), solo LoRA allenabile, reference va1, test 800 una volta sola alla fine con
   bootstrap appaiato 10k (`v3-recurrent/audit/paired_eval.py`).
6. Qualunque difetto visto nei JSON o nel protocollo che debba essere corretto PRIMA della run
   lunga (elenco puntato, con file e riga).

Formato del verdetto: sezione "Gate" (PASS/FAIL con prove), sezione "Anomalie", sezione
"Proposta punto 5" (tabella), sezione "Correzioni prima della run" (o "nessuna").
