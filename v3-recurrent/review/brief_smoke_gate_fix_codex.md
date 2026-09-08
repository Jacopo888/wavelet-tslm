# Brief per Codex — gate dello smoke sulla scala del contesto: criterio sbagliato (fix rapido)

Repo: `C:\Users\jacop\Proposta ETH FINALE\wavelet-tslm`. Non committare, niente rete. Modifica
direttamente `v3-recurrent/node/node_smoke_v3.sh` (e, se serve, `CODEX-REPORT.md` §12).

## Fatto (smoke A40, 08/09 10:44 UTC, commit `5bb1014`)
- `v3untied`: PASS, `context_scale_after_4_steps` = 0,10244.
- `v3tied`: training ok, `[V3Grad]` a ogni passo con gradiente dei blocchi non nullo e scala
  0,100098 → 0,099097 → 0,099644 → 0,100024 (dopo 4 step 0,10014); poi il check sul checkpoint
  fallisce: `AssertionError: ('context_scale moved less than 1e-3 after four training steps', 0.1, 0.10014118999242783)`
  (stage `v3tied_checkpoint`, riga 407 dello script, blocco PYCKPT).

Il criterio "valore finale a distanza ≥ 1e-3 da 0,1" è fragile: con lr 1e-3 e 4 passi la scala
oscilla e può tornare vicino all'init pur ricevendo gradiente. Il deadlock che vogliamo escludere
è "gradiente dei blocchi ≈ 0 e scala ferma", non "scala finale vicina all'init".

## Cosa fare
1. Sostituisci il criterio con uno robusto e documentato, ad es. TUTTI i seguenti dal log
   `[V3Grad]` del training di smoke (già emesso con `TSLM_V3_GRAD_LOG=1`): (a) `blocks_grad_norm`
   > 0 a ogni step e `blocks_relative` ≥ 1e-3; (b) `max_t |context_scale_t − init| ≥ 1e-4`
   (oppure la somma dei |Δ| per passo ≥ 1e-4); (c) `context_scale_grad` non nullo in almeno
   uno step. Registra nel JSON di ciascuna variante `context_scale_trajectory`,
   `context_scale_max_abs_move`, `blocks_grad_norm_min`, `blocks_relative_min`.
2. Verifica che con la vecchia run zero-init il criterio fallirebbe (grad blocchi 0, mosse 0):
   basta un controllo logico nel report, niente run.
3. `bash -n`, `git diff --check`. Nient'altro.
