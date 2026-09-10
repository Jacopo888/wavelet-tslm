# Brief per Codex — L3 punto 3, correzioni dopo la revisione del kit (rev. 1, commit del 10/09)

Claude ha rivisto `l3/grpo.py`, `l3/node_l3_smoke.sh`, `l3/configs/l3_smoke.yaml`,
`l3/tests/test_l3_kit.py`, `l3/PROTOCOL-L3.md` e verificato sul nodo che il train parquet ha
l'hash del config (`b8571d88…`) e che i 22 indici hanno id/categoria/lunghezza/canali attesi
(22/22 OK). Test CPU 10/10 in locale. Il disegno è accettato. Tre correzioni mirate prima di
spendere GPU; non riaprire le altre scelte. Codex modifica i file, Claude controlla il diff.

## 1. Generazione senza KV-cache (prestazioni, `grpo_step`)

`grpo_step` chiama `_gradient_checkpoint_train_mode(model)` (→ `model.train()`) PRIMA di
`generate_completion`. `ChatTSModel.generate` (`src/models/chatts.py:469`) non chiama `eval()`:
usa `torch.inference_mode()` ma `self.training` resta True. Con `TSLM_GRAD_CKPT=1` il decoder
Qwen passa dal ramo `if self.gradient_checkpointing and self.training`
(`transformers/modeling_layers.py:60` nella venv del nodo) e la generazione va senza cache:
ogni token rigenera l'intera sequenza (4096 token percettivi + prompt sulle celle 16ch), cioè
decine di forward completi per rollout invece di uno.

Correzione: generare con il modello in `eval()` (cache attiva; nessun dropout), poi tornare a
`_gradient_checkpoint_train_mode` per i tre forward teacher-forced (old, reference, corrente),
che devono restare nella STESSA modalità fra loro (l'uguaglianza old == corrente dipende da
questo). Documentare al §Reference/old policy del PROTOCOL. `_init_check` è già in `eval()`.

## 2. Gate di uguaglianza esatta del ratio (robustezza, `grpo_step` + smoke c0/c)

`grpo_step` alza `RuntimeError("ratio is not exactly one …")` se `torch.equal(ratio, 1)` fallisce
su un solo completamento; lo smoke pretende `ratio_all_exact_one` e `ratio_mean == 1.0`.
Old log-prob (`no_grad`) e log-prob corrente (con grad, checkpointing attivo) possono passare
da kernel SDPA diversi sullo stesso input: differenze all'ultimo bit sono possibili e non sono
una deriva di policy (con un solo aggiornamento per batch la policy È la old policy).

Correzione: il gate fatale diventa `max |log ratio| ≤ 1e-3` per completamento (valore loggato
per step come `max_abs_log_ratio`); `ratio_all_exact_one` resta come flag riportato, non
fatale. Stesso criterio nello smoke (c0, c) e nello stadio b (`max_logprob_diff ≤ 1e-3`,
`logprob_equal_reference` riportato). Aggiornare test e PROTOCOL (motivazione sopra).

## 3. Stadio d con `TSLM_EVAL_STRICT=1` sugli 80/520 (bug bloccante, `node_l3_smoke.sh`)

Lo script esporta `TSLM_EVAL_STRICT=1` per tutto lo smoke. In modalità strict `train.py
--eval-only` abortisce quando il numero di campioni valutati non copre lo split (80/520 con
`max_samples: 800` del config probe80) e NON scrive `--out-json` (lezione v3, commit
`b063ac8`; i probe sul nodo — `~/node_valprobe_waiter.sh`, `~/node_va1_probe.sh` — usano
`TSLM_EVAL_STRICT=0`). Lo stadio d fallirebbe sempre.

Correzione: `TSLM_EVAL_STRICT=0` SOLO per il comando dello stadio d (strict resta 1 altrove);
in cambio il check Python dello stadio d pretende `samples_evaluated == 80`, `eval_errors == 0`
e `split == "val"`, e legge anche il JSONL per contare 80 righe. Documentare nel PROTOCOL
(strict solo sulle val intere).

## Note accettate senza modifica

- Stadio b = confronto interno (percorso eval bf16 vs wrapper L3 con master fp32) invece del
  `--eval-only` di train.py (che non offre il train split): accettato, come da PROTOCOL.
- Checkpoint LoRA-only + artefatto composto a 42 chiavi per l'eval: accettato.
- Indici fissi e aspettative: verificati sul nodo.

## Ordine

1. Modifiche ai file + test CPU (`py -m pytest l3/tests -q` deve passare).
2. Aggiornare `PROTOCOL-L3.md` nei tre punti.
3. Nessun nuovo file fuori da `l3/`; `kaggle/patch/` resta intatto.
