# wavelet-tslm — proposta PhD ETH (tokenizzazione multirisoluzione per TSLM)

Cartella di lavoro dedicata (isolata dalla vecchia proposta ADMIT, ora in `../admit-legacy/`; le cartelle `../airway-agent/` e `../infra/` sono dell'altro agente — non toccare).

## Mappa

- `timerlm/` — clone di OpenTSLM/TimeRLM + AnomalyXL rigenerato (seed 42) + innesti wavelet (tool v0, skill v1, docs/PRIME_RL_PIN.md per il Livello 2)
- `kaggle/patch/` — **la fonte di verità del codice**: train.py patchato (bf16 AMP, resume, budget step/tempo, epoch-checkpoints, eval-only, COPERTURA PER CATEGORIA con abort) + encoder wavelet (haar/db4, use_coarse) + tutti i config YAML
- `kaggle/` — kernel s1-s4 della catena Kaggle + risultati (RISULTATI-SESSIONI-WAVELET.md)
- `modal/` — app Modal (modal_train.py: train_session + select_and_eval), risultati run full1 e m30 (RISULTATI-FULL-RUN.md, RISULTATI-M30-UFFICIALE.md)
- `vultr/` — kit nodo bare metal (setup driver+fabricmanager, env, train multi-GPU, eval worker, status)
- `v3-recurrent/` — **variante v3, separata dalla v1**: percettore wavelet ricorrente (looped layer sulla scala), 3 ablazioni a protocollo va1 identico, watcher GPU e kit nodo (README nella cartella)
- `colab/` — kit Colab (piano C, declassato)
- `lvl0/` — pilot Livello 0 (agente wavelet-zoom vs code)
- `demo/` — video demo della tokenizzazione

## Regola d'oro (nata dall'errore m30 del 02/09)

**Mai dichiarare un training completo senza contatori**: ogni epoca stampa `[Coverage] trained/failed per categoria`, e con `TSLM_REQUIRE_FULL_COVERAGE=1` un solo campione fallito abortisce il run. Checkpointing attivo (`TSLM_GRAD_CKPT=1`) SEMPRE: le celle a 16 canali (4.096 token percettivi) senza checkpointing vanno in OOM irrecuperabile ad allocator saturo.

## Stato modelli

- **Livello 3, punto 4 — zoom (12/09): il meccanismo funziona, la scelta della finestra no.** Stesso modello (va1 + 1 epoca SFT a due turni) sulla val520: senza zoom 0,301; con finestra gold **stretta** (~1.024 campioni) **0,524** (localize **0,633** vs 0,244 di va1, all_channels **0,666** vs 0,358: livello dell'agente multi-turno TimeRLM, 0,656/0,551); con la stessa finestra gold ma **larga** 6.900 solo 0,323; con la finestra scelta dalla policy 0,287 (gate FALLITO, catena fermata). La risoluzione utile del percettore è ~4 campioni per token. Dettagli: `l3/RISULTATI-L4-ZOOM.md`.

- **Livello 3, punto 5 (11/09): `va1 + GRPO` vs `va1`** — test 800 una lettura: 0,2869 vs 0,2744, Δ appaiato +0,0125 CI[+0,0003, +0,0253], **interamente da lead_lag** (+0,102: la policy risponde "independent" 76/100 vs 43, direzione corretta 29 vs 30); localize −0,008 e all_channels −0,014 (ns). RL single-turn su reward ufficiale non migliora la localizzazione e sfrutta il default dello scorer. Dettagli e lettura: `l3/RISULTATI-L3.md`; artefatti `integrity/artifacts/l3_longrun_20260910/`. Repliche (2 seed, senza test) in corso.

- **Variante v3 (residuo su v1, ablazione 1, 08/09)**: test 800 primary **0,2832** vs va1 0,2744 (appaiato +0,009, ns). `context_scale` è rimasto a zero per 5 epoche: la run del commit `1b0f9fa` è una **replica di va1**, non una misura del contesto. Il deadlock è ora rotto nei config con `context_scale_init: 0.1` e gruppo ricorrente a lr `1e-3` (il percorso v1 resta a `1e-4`); ablazione 1 da ripetere. Dettagli in `v3-recurrent/RISULTATI-V3TIED.md` e `v3-recurrent/review/CODEX-REPORT.md` §12.
- **Pilot v3 context (09/09)**: l'harness valuta ora, sullo stesso checkpoint e sulle stesse
  righe di validation, contesto on/off/roll sui probe80, full-val on/off e confronti appaiati
  bootstrap 10k; il launcher post-hoc completa senza sovrascrivere i pilot già conclusi.
  Dettagli in `v3-recurrent/review/CODEX-REPORT.md` §15.
- **Riferimento Livello 1: va1 epoca 5** (re-run a copertura totale su Vultr A40, 03/09) — test 800 hfpin: primary **0.2744**, localize 0.241 (IoU positivi 0.094), MMag 0.578, all_channels 0.261, classify 0.064, lead_lag 0.166 (al caso). Dettagli in `vultr/RISULTATI-VA1.md`, artefatti in `integrity/artifacts/va1/`, checkpoint sul nodo.
- Storici (controllo): m30 epoca 4 (0.2156 hfpin; ha perso tutte le celle 16ch), full1 post-top-up (0.2417 hfpin).
- Prossimo: Livello 2 (GRPO prime-rl al pin, kit in `l2/`, modello Qwen3.5-2B su A40-24Q) — la A40 resta al lato wavelet (ordine 03/09).
