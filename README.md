# wavelet-tslm — proposta PhD ETH (tokenizzazione multirisoluzione per TSLM)

Cartella di lavoro dedicata (isolata dalla vecchia proposta ADMIT, ora in `../admit-legacy/`; le cartelle `../airway-agent/` e `../infra/` sono dell'altro agente — non toccare).

## Mappa

- `timerlm/` — clone di OpenTSLM/TimeRLM + AnomalyXL rigenerato (seed 42) + innesti wavelet (tool v0, skill v1, docs/PRIME_RL_PIN.md per il Livello 2)
- `kaggle/patch/` — **la fonte di verità del codice**: train.py patchato (bf16 AMP, resume, budget step/tempo, epoch-checkpoints, eval-only, COPERTURA PER CATEGORIA con abort) + encoder wavelet (haar/db4, use_coarse) + tutti i config YAML
- `kaggle/` — kernel s1-s4 della catena Kaggle + risultati (RISULTATI-SESSIONI-WAVELET.md)
- `modal/` — app Modal (modal_train.py: train_session + select_and_eval), risultati run full1 e m30 (RISULTATI-FULL-RUN.md, RISULTATI-M30-UFFICIALE.md)
- `vultr/` — kit nodo bare metal (setup driver+fabricmanager, env, train multi-GPU, eval worker, status)
- `colab/` — kit Colab (piano C, declassato)
- `lvl0/` — pilot Livello 0 (agente wavelet-zoom vs code)
- `demo/` — video demo della tokenizzazione

## Regola d'oro (nata dall'errore m30 del 02/09)

**Mai dichiarare un training completo senza contatori**: ogni epoca stampa `[Coverage] trained/failed per categoria`, e con `TSLM_REQUIRE_FULL_COVERAGE=1` un solo campione fallito abortisce il run. Checkpointing attivo (`TSLM_GRAD_CKPT=1`) SEMPRE: le celle a 16 canali (4.096 token percettivi) senza checkpointing vanno in OOM irrecuperabile ad allocator saturo.

## Stato modelli

- Best ufficiale (mono/bi-canale): m30 epoca 4 — volume Modal `wavelet-vol`, sessions/m30 (test: primary 0.211, localize 0.181)
- Evidenza multicanale: full1 epoca 5 (all_channels 0.180, curriculum accidentale)
- Da fare al prossimo GPU Vultr: re-run con copertura garantita (ordine Jacopo 02/09)
