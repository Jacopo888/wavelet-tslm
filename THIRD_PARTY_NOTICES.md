# Codice di terzi in questa repo

Alcuni file sono copie modificate di progetti open source. Le loro licenze restano valide per quei file; i testi completi sono in `third_party_licenses/`.

| Progetto | Licenza | File in questa repo | Revisione di partenza |
|---|---|---|---|
| [OpenTSLM/TimeRLM](https://github.com/OpenTSLM/TimeRLM) | MIT, © 2026 The TimeRLM Authors (`third_party_licenses/TimeRLM-MIT.txt`) | `kaggle/patch/` (`train.py`, `qa_dataset.py`, `qwen_backbone.py`, `fetch_data.py`, `wavelet.py`, `wavelet_encoder.py`, derivati da `sft/tslm/`); `l2/overlay/timerlm/environments/`, `l2/overlay/timerlm/scripts/`, `l2/overlay/timerlm/docs/` | `31fcd847` |
| rlm-harness (in TimeRLM) | MIT, © 2026 Prime Intellect (`third_party_licenses/rlm-harness-MIT.txt`) | `l2/overlay/timerlm/rlm-harness/` | `31fcd847` di TimeRLM |
| [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) | Apache 2.0 (`third_party_licenses/prime-rl-Apache-2.0.txt`) | `l2/overlay/prime-rl/src/prime_rl/entrypoints/rl.py`, `l2/overlay/prime-rl/src/prime_rl/trainer/models/layers/lora/multi_linear.py`, `l2/overlay/prime-rl/src/prime_rl/trainer/utils.py` | `da53d4c0` |

- I file sotto `l2/overlay/` sono sovrapposizioni: sostituiscono i file omonimi dei progetti originali e sono stati **modificati** rispetto a quelli. I tre file di prime-rl lo dichiarano anche in testa, come chiede la licenza Apache 2.0.
- I file di `kaggle/patch/` partono da quelli di TimeRLM e sono stati modificati (precisione bf16, ripresa, budget, checkpoint, conteggi per categoria, encoder wavelet).
- I dati AnomalyXL non sono nella repo: si rigenerano col generatore di TimeRLM.
- Il resto del codice è di questa repo.
