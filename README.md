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

- **Variante v3 (residuo su v1, ablazione 1, 08/09)**: test 800 primary **0,2832** vs va1 0,2744, differenza appaiata non significativa; localize peggiora (0,208 vs 0,241, significativo), lead_lag sale solo per la moda "independent". Il contesto ricorrente non aiuta. Dettagli in `v3-recurrent/RISULTATI-V3TIED.md`, artefatti in `integrity/artifacts/v3tied/`.
- **Riferimento Livello 1: va1 epoca 5** (re-run a copertura totale su Vultr A40, 03/09) — test 800 hfpin: primary **0.2744**, localize 0.241 (IoU positivi 0.094), MMag 0.578, all_channels 0.261, classify 0.064, lead_lag 0.166 (al caso). Dettagli in `vultr/RISULTATI-VA1.md`, artefatti in `integrity/artifacts/va1/`, checkpoint sul nodo.
- Storici (controllo): m30 epoca 4 (0.2156 hfpin; ha perso tutte le celle 16ch), full1 post-top-up (0.2417 hfpin).
- Prossimo: Livello 2 (GRPO prime-rl al pin, kit in `l2/`, modello Qwen3.5-2B su A40-24Q) — la A40 resta al lato wavelet (ordine 03/09).
