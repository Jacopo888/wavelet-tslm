# FREEZE — stato congelato pre-integrity-fix (02/09/2026)

Scopo: fotografia verificabile dello stato del progetto PRIMA dei fix di research
integrity (ordine di Jacopo, 02/09 sera, a valle della review esterna GPT 5.6).
Nessun training lanciato; nessuna modifica all'architettura del modello.

## 1. Codice

- Root repo: creato oggi (`git init -b main`); questo commit è la baseline.
- Sottorepo `timerlm/` (escluso dal root repo via .gitignore, tracciato qui):
  - upstream: https://github.com/OpenTSLM/TimeRLM
  - HEAD: `31fcd847b7cb37a1f1e6859b1dca7973ed0eae74`
  - 18 modifiche locali (6 modified + 12 untracked) = gli innesti wavelet.
    I file sorgente autorevoli degli innesti vivono in `kaggle/patch/` (committati
    qui); le copie sotto timerlm sono overlay deployati dai launcher.
    Lista completa:
    ```
     M anomalyXL/scripts/classical_baseline.py
     M environments/timeseries_qa/rlm.py
     M rlm-harness/src/rlm/skills/__init__.py
     M sft/tslm/src/datasets/anomaly_xl/qa_dataset.py
     M sft/tslm/src/models/qwen_backbone.py
     M sft/tslm/train.py
    ?? docs/PRIME_RL_PIN.md
    ?? environments/timeseries_qa/test_wavelet_pyramid_standalone.py
    ?? environments/timeseries_qa/wavelet_pyramid.py
    ?? rlm-harness/src/rlm/skills/wavelet_zoom.py
    ?? sft/tslm/configs/chatts_precise_budget.yaml
    ?? sft/tslm/configs/flamingo_precise_smoke.yaml
    ?? sft/tslm/configs/wavelet_precise.yaml
    ?? sft/tslm/configs/wavelet_precise_budget.yaml
    ?? sft/tslm/configs/wavelet_precise_smoke.yaml
    ?? sft/tslm/src/models/test_wavelet_encoder_standalone.py
    ?? sft/tslm/src/models/wavelet.py
    ?? sft/tslm/src/models/wavelet_encoder.py
    ```

## 2. Dati locali — hash e VERDETTO LEAKAGE

Hash sha256 + row count di tutti i 12 file in `timerlm/anomalyXL/datasets/`:
vedi `integrity/DATA-HASHES-LOCAL.json` (committato).

**LEAKAGE CONFERMATO (audit 02/09, verifica per ID):**

| File locale | Righe | Overlap col test |
|---|---:|---:|
| anomalyxl-precise/train.parquet | 723 | 723/723 (100%) |
| anomalyxl-precise/val.parquet | 77 | 77/77 (100%) |
| anomalyxl-train/anomalyxl-precise/data.parquet | 800 | 800/800 |
| anomalyxl-train/anomalyxl-precise/train.parquet | 723 | 723/723 |
| anomalyxl-train/anomalyxl-precise/val.parquet | 77 | 77/77 |
| anomalyxl-train/anomalyxl-coarse/data.parquet | 248 | 248/248 |
| anomalyxl-train/anomalyxl-coarse/train.parquet | 231 | 231/231 |
| anomalyxl-train/anomalyxl-coarse/val.parquet | 17 | 17/17 |

Origine: REPRODUCE.md Option B upstream (`split_anomalyxl_train_val.py` spezza il
TEST in train/val). **Su questa macchina non esiste alcun train pulito.** File
PULITI (i gold test legittimi): `anomalyxl-precise/data.parquet` (800),
`anomalyxl-coarse/data.parquet` (248), + manifest.json.

**I run principali NON sono toccati dal leak**: m30/full1/Kaggle hanno eseguito
`fetch_data.py` nel container → train HF `nz00shuuuu/AnomalyXL` (4.488 righe;
22.440 = 4.488×5 step in m30 lo conferma). I file contaminati sono spostati in
quarantena (vedi `_QUARANTINE_LEAKED/`), non cancellati.

## 3. Artefatti dei run (volume Modal `wavelet-vol`) — copie auditabili

Scaricati oggi in `integrity/artifacts/` (committati):

- `m30/test_ep4_outputs.json` — eval test 800 righe del checkpoint epoch_4_end
  (il modello "ufficiale"), CON output per-esempio (gold, prediction raw, metriche).
- `m30/test_endrun_outputs.json` — test di fine run (checkpoint finale ep5).
- `m30/select/ep{1..5}_val.json` + `best_ep4_test.json` — selezione upstream-style.
- `full1/test_outputs.json` — test di fine run full1, CON output per-esempio.
- `m30/config.yaml`, `full1/config.yaml`, `summary.json` di entrambi.

Inventario checkpoint sul volume (non scaricati, ~GB):
- `sessions/m30/.../checkpoints/`: `epoch_{1,2,3,4,5}_end.pt` + `best_model.pt`
  (= best per val loss, ep2) + `latest_checkpoint.pt` + intra-epoch batch-500..4000.
- `sessions/full1/.../checkpoints/`: `best_model.pt`, `latest_checkpoint.pt`,
  `checkpoint_epoch_6_batch_1988.pt` (fine top-up) + intra-epoch ep1-6.
  NB full1 NON ha epoch_*_end (TSLM_KEEP_EPOCH_CKPTS nato dopo).
- Log completi: `sessions/{m30,full1}/train.log` (audit coverage m30: 2.770
  "[CkptFallback] ... FAILED", 0 "recovered").

## 4. Risultati dichiarati al freeze (con caveat già noti)

- m30 ep4 (test 800): primary 0.211 — localize 0.181, classify 0.004, mmag 0.510,
  all_ch 0.030, lead_lag 0.264. ZERO training multicanale (2.770 celle 16ch perse).
- full1 (test 800): primary 0.238 — all_ch 0.180 (esposizione 16ch tardiva, non
  riproducibile a disegno).
- Baseline nulle del benchmark (verificate sul gold): localize sempre-false 0.195;
  all_channels sempre-vuoto 0.230; composizione test 200/200/200/100/100.

## 5. Registro difetti noti al freeze (verificati sul codice, da fixare DOPO questo commit)

1. `train.py` finally del CkptFallback disabilita il gradient checkpointing
   incondizionatamente (degrada i run con GRAD_CKPT=1 globale dopo il primo OOM).
2. `modal_train.py` env: TSLM_GRAD_CKPT="0", nessun TSLM_REQUIRE_FULL_COVERAGE.
3. `validate()`: eccezioni di generazione/loss silenziosamente ignorate; nessun
   ID per esempio; nessun check di completezza.
4. `fetch_data.py`: nessun pin di revisione HF, nessun manifest, nessun assert
   di disgiunzione.
5. Scorer ufficiale (NON verrà toccato, per confrontabilità): `bool("false")=True`
   su `present` (scoring.py:199); eventi malformati scartati e non penalizzati
   (scoring.py:315-320).
6. Encoder bf16 sulle posizioni (aliasing ottave alte a L=131k) — NON si fixa in
   questa fase: sarà la variante `v2` separata, confrontata a parità di protocollo.
7. Padding zero non mascherato nel batch (latente: tutti i run usano batch_size 1).
