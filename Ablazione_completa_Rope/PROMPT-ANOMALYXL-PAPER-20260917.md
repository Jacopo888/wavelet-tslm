# Il prompt esatto di AnomalyXL/OpenTSLM, come lo costruiscono per il training

Estratto il 17/09/2026 dal codice di riferimento, non dal testo del paper. Sorgenti:

- `timerlm/sft/tslm/src/datasets/anomaly_xl/scoring.py:27` — `PRECISE_SYSTEM_PROMPT`
- `timerlm/sft/tslm/src/datasets/anomaly_xl/qa_dataset.py:114-139` — pre-prompt, post-prompt, etichette dei canali
- `timerlm/sft/tslm/src/datasets/qa_base.py:190-200` — `PromptWithAnswer`, EOS appeso alla risposta
- `timerlm/sft/tslm/src/models/chatts.py:324-392` — `_prepare_batch_raw`, l'assemblaggio finale della stringa
- `timerlm/sft/tslm/src/models/wavelet.py:24` — `WaveletTSModel(ChatTSModel)`, cioe' `va1` usa esattamente questo percorso

`kaggle/patch/qa_dataset.py` e' **byte-identico** all'originale: la nostra patch non ha toccato il prompt.

## Il template

```
{PRECISE_SYSTEM_PROMPT}
The following is channel {nome_canale_1}: {<ts_pad> ripetuto N volte}
The following is channel {nome_canale_2}: {<ts_pad> ripetuto N volte}
...
Question: {question} {answer}{EOS}
```

Dettagli che contano e che il template da solo non mostra:

- **Nessun chat template.** La stringa viene tokenizzata cosi' com'e', `tokenizer(text_inputs, padding="longest")`. Niente `<|im_start|>system`, niente turni, niente `role`. (`chatts.py:400-406`)
- **La risposta e' incollata alla domanda da un singolo spazio**, non da un a capo: `full_text = prompt + " " + item["answer"]` (`chatts.py:366`).
- **`N` non dipende dalla lunghezza.** Per l'encoder wavelet `get_num_output_patches()` restituisce `self.num_tokens` costante (`wavelet_encoder.py:85-86`), che in `wavelet_precise_va1.yaml` vale **256**. Quindi 256 token per canale sia a L=16.384 sia a L=131.072.
- **EOS** e' `text_tokenizer.eos_token` (`chatts.py:505`, `train.py:755`), per Qwen3.5-4B `<|im_end|>`.
- Il separatore fra i blocchi e' `\n`; pre-prompt e post-prompt sono `.strip()`-ati (`qa_base.py:196-198`).

## Il system prompt, verbatim

```
You analyze long-context time series and report grounded evidence. Your final answer MUST be a single JSON object on its own line, with no commentary, no markdown fences, no prose.
```

Variante `coarse`, non usata da noi: «You are an expert at analyzing time series. You are given one or more channels of time-series data followed by a question. Respond with only the answer and nothing else.»

## L'etichetta dei canali, verbatim

```python
TextTimeSeriesPrompt(f"The following is channel {name}:", arr)
```

Un blocco per canale, etichettato col nome del canale (`ch_00`, `ch_01`, …; `A_ch_00`/`B_ch_00` per lead-lag). Nessuna statistica.

## Le uniche cifre che il benchmark regala

**Solo `L`, e per il multicanale anche `C`**, dentro la `question` del dataset:

- `(Here L = 16384.)`
- `(Here L = 32768, C = 16.)`

Nient'altro. Niente mediana, scala, minimo, massimo, conteggio dei validi, numero di token.

## Le cinque domande, verbatim dal parquet gold

**localize**
```
Does the time series contain an anomaly, and if so where? Respond with exactly one JSON object on a single line, no other text:
{"present": true|false, "start": <int>, "end": <int>}
If "present" is false, "start" and "end" are ignored. If true, report the inclusive start and exclusive end integer sample indices of the anomaly. (Here L = 16384.)
```

**classify_with_evidence**
```
Identify the anomaly in the time series and its window. Respond with exactly one JSON object on a single line, no other text:
{"kind": <label>, "start": <int>, "end": <int>}
`kind` must be one of: "Level Shift", "Transient Spike", "Change in Seasonality", "Change in Variance", "Change in Trend". `start` is inclusive, `end` is exclusive. Each kind is defined as follows: a `Level Shift` is a sustained step in mean — values jump and remain shifted across the window; a `Transient Spike` is a brief impulse — one or a few samples diverge sharply, then return to baseline; a `Change in Variance` is a sustained increase (or decrease) in noise amplitude over a window with the mean unchanged; a `Change in Seasonality` is a sustained change in the periodic component's amplitude, frequency, or phase across a window; a `Change in Trend` is a slope change starting at the window — values drift up or down and remain on the new path afterwards. (Here L = 16384.)
```

**measure_magnitude**
```
An anomaly is present in the time series. Report its peak deviation in units of the channel's baseline (pre-anomaly) standard deviation (σ). Respond with exactly one JSON object on a single line, no other text:
{"magnitude_sigma": <float>}
Use a positive value (the deviation magnitude, not its sign). (Here L = 16384.)
```

**localize_all_channels**
```
The time series has multiple channels. Find every anomaly across all channels. Respond with exactly one JSON object on a single line, no other text:
{"anomalies": [{"channel": "<name>", "start": <int>, "end": <int>}, ...]}
Report an empty list `"anomalies": []` if no anomalies are present. `channel` must match a channel name from the input verbatim. `start` is inclusive, `end` is exclusive. (Here L = 32768, C = 16.)
```

**lead_lag_with_magnitude**
```
Two single-channel sub-series are shown, with channels prefixed `A_` and `B_`. Both contain an anomaly (or independent anomalies). Is A a leading or lagging indicator of B, and by how many samples? Respond with exactly one JSON object on a single line, no other text:
{"direction": "lead"|"lag"|"independent", "lag_samples": <int>}
`lag_samples` is the non-negative magnitude (in sample indices) of the offset between A's and B's anomaly start; it is ignored in the independent case. Two anomalies are considered independent when they occupy unrelated windows with no causal alignment; by construction, coupled lead/lag offsets never exceed L/4 samples. (Here L = 32768.)
```

## Esempio completo, istanziato

```
You analyze long-context time series and report grounded evidence. Your final answer MUST be a single JSON object on its own line, with no commentary, no markdown fences, no prose.
The following is channel ch_00: <ts_pad><ts_pad>...<ts_pad>          <- 256 volte
Question: Does the time series contain an anomaly, and if so where? Respond with exactly one JSON object on a single line, no other text:
{"present": true|false, "start": <int>, "end": <int>}
If "present" is false, "start" and "end" are ignored. If true, report the inclusive start and exclusive end integer sample indices of the anomaly. (Here L = 16384.) localize:{"_L": 16384, "end": 13717, "present": true, "start": 13632}<|im_end|>
```

Multicanale, sedici canali, stessa struttura ripetuta:

```
You analyze long-context time series and report grounded evidence. ...
The following is channel ch_00: <ts_pad>*256
The following is channel ch_01: <ts_pad>*256
...
The following is channel ch_15: <ts_pad>*256
Question: The time series has multiple channels. Find every anomaly across all channels. ... (Here L = 32768, C = 16.) localize_all_channels:{"_L": 32768, "anomalies": [{"channel": "ch_10", "end": 11967, "start": 11897}, ...]}<|im_end|>
```

## Confronto con il nostro header

Quello che `ConversationBuilder.global_pieces` stampa in `va1r` e che **non esiste nel riferimento**:

```
Channel ch_00: median 0.0123, scale 1.4567, min -3.21, max 45.78, valid 16384/16384, 256 tokens.
```

piu' la riga della griglia di lettura e la riga `Global view, N signal tokens ordered by time:`.

Due differenze strutturali, oltre alle statistiche:

1. **Il riferimento etichetta ogni canale e mette i token di quel canale subito dopo l'etichetta.** Noi mettiamo tutti i canali in **un unico blocco ordinato per tempo**, e l'identita' di canale vive solo nel `channel_embedding`. Per `localize_all_channels` il riferimento consegna al modello la corrispondenza nome→token nel testo; noi gliela facciamo dedurre.
2. **Il riferimento non dice mai quanti token ci sono ne' come mapparli sugli indici campione.** L'unico ancoraggio numerico e' `L` nella domanda.

### Cosa deve essere `va1rz` per essere confrontabile

Header grezzo = **solo** l'etichetta per canale, senza nessuna statistica. `L` e `C` arrivano gia' dalla domanda del benchmark e vanno lasciati li'. Non «header ridotto»: header assente.
