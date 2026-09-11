# Brief per Codex — punto 4, correzioni dopo la revisione del kit zoom (rev. 1)

Claude ha letto `l3/zoom.py`, `l3/zoom_sft.py`, `l3/node_l3_zoom.sh`, i tre config e il diff di
`l3/grpo.py`; test CPU 34/34. Il disegno è accettato. Due correzioni mirate prima del nodo;
non riaprire altro. Codex modifica i file, Claude controlla il diff.

## 1. dtype del tensore della finestra di zoom (`l3/zoom.py`, `prepare_zoom_batch`)

`zoom_tensor` viene creato con `dtype = next(model.llm_model.parameters()).dtype` (bf16 sul
nodo), mentre i canali originali arrivano da `prepare_batch` nel loro dtype (float32) e
l'encoder li normalizza (MAD) in quella precisione. Con una serie di ampiezza ~1e3 e anomalie
di ampiezza relativa ~1e-2, la MAD in bf16 (8 bit di mantissa) rovina proprio il segnale che lo
zoom deve leggere. Correzione: usare lo stesso dtype e device di `original["time_series"]`
(`torch.as_tensor(values, dtype=original["time_series"].dtype, device=…)`), e un test CPU che
verifichi che la finestra passa all'encoder nello stesso dtype dei canali originali.

## 2. Percorsi di default dei JSONL di controllo (`l3/node_l3_zoom.sh`)

`VA1_TEST_JSONL` e `RL_TEST_JSONL` puntano a `$ROOT/integrity/artifacts/...`, che esiste solo
nel repo locale, non sul nodo: il controllo degli input richiesti fallirebbe all'avvio.
Sul nodo i due file sono:

- va1: `$ROOT/results/anomaly_xl_precise/anomaly_xl_wavelet_va1/output_logs/evalonly_test_epoch_5_end_epoch_0.jsonl`
- va1+RL step 100: `$ROOT/results/anomaly_xl_precise/anomaly_xl_wavelet_va1/output_logs/evalonly_test_step_000100_composed_l3_L3_test800_epoch_0.jsonl`

Correzione: questi come default (gli override `L3_VA1_TEST_JSONL`/`L3_RL_TEST_JSONL` restano),
più un controllo che entrambi abbiano 800 righe prima di partire (fallire chiuso).

## Note accettate senza modifica

- `no_zoom` in Z0 passa da `model.generate` come va1: deve riprodurre 0,3105 sulla val520
  (verifica implicita del percorso di eval); Claude lo controllerà.
- Log-prob su due segmenti con forward separati; gate `max|log ratio| ≤ 1e-3`; pool 200
  ricampionato senza filtro; lead_lag esclusa; `initial_checkpoint_sha256: null` per Z2
  sigillato dal driver.
