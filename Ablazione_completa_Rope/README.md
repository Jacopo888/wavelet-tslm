# Wavelet-RoPE Agent — prototipo completo (WR-Agent-4B-Wildcard-v1)

**Stato 15/09/2026, 04:10:** design v0.3 + decisioni Astra P0/P1/P2 e **scala** (`DECISIONI-ASTRA-20260915.md`);
backend MI300X = FlexAttention con Q/K zero-paddate a 512. La prima run P1 (`p1-main`, ora "P1-a") è risultata cieca al
contenuto: soft token 30× più grandi degli embedding testuali e quasi identici fra loro per colonne lineari non clippate,
stem non normalizzato e proiettore GELU-terminale (diagnosi e decisioni nella sezione "scala" del documento Astra).
Correzioni applicate (clip ±16, floor di banda, LayerNorm stem/fusione, gain identificatori, proiettore a due lineari con
gain calibrato agli embedding, veto del gate per categorie a zero; 125 test verdi, smoke P0 esteso PASS) e **P1-b in corso**
(run `p1-b`, albero `/workspace/code_p2`) con sentinella del gate percettivo r20 al passo 400 (`droplet/gate_early.sh`).
P1-a chiusa come risultato negativo documentato (`RISULTATI-P1A-20260915.md`: gate r120 FAIL). **P1-b fermata alla
sentinella del passo 400** (`RISULTATI-P1B-20260915.md`): la scala è corretta ma il proiettore converge di nuovo a una
direzione comune e il contenuto peggiora le risposte. **P1-c** (decisioni Astra, sezione "sentinella"): segmento di
"lettura" programmatica supervisionato in `<think>` (`src/wavelet_rope/reading.py`, peso 0,5) generato dal modello anche
in validazione e gate, RMS per token fissato senza parametri, penalità di decorrelazione λ = 0,01, lr 2e-5 / LoRA 1e-4
(`configs/p1c_mi300x.toml` + `prototype.spec.p1c.json`). **P1-c fermata alla sentinella 400** (`RISULTATI-P1C-20260915.md`):
CE identiche con token veri/mescolati/azzerati; la diagnostica causale mostra che la LoRA è stata inattiva in tutte le run e
che il segnale entra nell'attenzione senza arrivare ai logit. **P1-d** (decisioni Astra, sezione "P1-c e P1-d"): fase A con
LoRA congelata, frontend 1e-4, lettura compatta a indici di bin (`top ch:b,b,b; peak ch:b`, griglia dichiarata nel prompt),
uscita solo se la CE sui campi dei bin con token veri batte i controlli di 0,10 nats (`tools/reading_probe.py`); fase B con
LoRA da `--init-from` (`configs/p1dA_mi300x.toml`, `p1dB_mi300x.toml`). **La fase A è FAIL** (`RISULTATI-P1D-20260915.md`):
i token valgono 0,73 nats come presenza e zero come contenuto, e le rappresentazioni non diventano più decodificabili.
**P1-e in corso** (decisioni Astra, sezione dedicata): supervisione diretta per token con due teste lineari sui soft token
che predicono posizione del bin, energia di Haar e picco (`src/wavelet_rope/token_supervision.py`, bersagli standardizzati
con statistiche fisse del train da `tools/token_supervision_stats.py`); prima parte di 200 passi senza cross-entropy
(`configs/p1e1_mi300x.toml`, `ce_weight = 0`, il backbone non viene toccato), poi 300 passi con lettura e sostituzione dei
token al 25 % (`configs/p1e2_mi300x.toml`). Gate della prima parte: errore di posizione entro un bin e recupero dei primi
tre bin di energia almeno 0,70 su registrazioni mai addestrate (`tools/token_supervision_eval.py`).
Codice P2 (teacher di traiettorie, runner agentico, selettore Haar cieco) consegnato e verificato dallo smoke P2 sul nodo.

Documenti di riferimento, in ordine di autorità:

- [DESIGN.md](DESIGN.md) — architettura, contratti, training, criteri (v0.3, 13/09).
- [prototype.spec.json](prototype.spec.json) — valori numerici fissati (non è una config eseguibile).
- [DECISIONI-ASTRA-20260914.md](DECISIONI-ASTRA-20260914.md) — decisioni P0 vincolanti (package, backend, missing, epsilon, dati, storage).
- [RISULTATI-GATE-BACKEND-20260914.md](RISULTATI-GATE-BACKEND-20260914.md) — gate dei kernel e del backbone reale sulla MI300X.
- [DECISIONI-ASTRA-20260915.md](DECISIONI-ASTRA-20260915.md) — conferme e richieste di Astra su compilazione, equivalenza, config P1, crop teacher, gate percettivo prima di P2.
- [ABLATION.md](ABLATION.md) — confronti futuri, non opzioni del modello.

## Mappa del codice (`src/wavelet_rope/`)

| modulo | contenuto | stato |
|---|---|---|
| `signal_contract.py` | episodio, normalizzazione robusta (mediana/MAD, epsilon relativo), viste globali/locali con coordinate assolute | fatto (Singer) |
| `pyramid.py` | Haar fissa FP32, validità stretta, supporti e ancore temporali | fatto (Singer) |
| `deep_branches.py`, `tokenizer.py` | rami profondi A/D/raw + adapter locali, quote 256/512, salienza, fusione, testa ausiliaria | in revisione (agente) |
| `perceptual_rope.py` | mixer 2 blocchi con RoPE a due assi (ordinale, tempo fisico), memoria delle viste, proiettore 512→2560 | fatto, test verdi |
| `qwen_temporal_rope.py` | wildcard §7.4: fattorizzazione esatta Q*/K* (352/384), curriculum λ, FlexAttention pad512 / SDPA / MATH, cache | fatto, test verdi + gate GPU |
| `backbone.py` | Qwen3.5-4B al pin offline, torre visiva rimossa, wrap per istanza degli 8 layer, LoRA q/k/v/o risolti | fatto, gate GPU |
| `templates.py` | conversazioni direct/tool, segmenti (reasoning/action/final), placeholder `<|image_pad|>`, budget | fatto |
| `model.py` | encode_episode (memoria viste), sequenza, loss per segmento + BCE occupazione, generazione con cache | fatto, da smoke |
| `data.py`, `scoring.py` | parquet AnomalyXL lazy, ZoomAction/parser, ModeAssigner, crop teacher, target occupazione; scorer ufficiale vendorizzato | scorer fatto; data in scrittura (agente) |
| `config.py`, `manifest.py`, `checkpoint.py`, `run_guard.py` | config fail-fast legata alla famiglia GPU, fingerprint, checkpoint atomici, lease | fatto (Jason) |

Altri: `training/train.py` (SFT P1, resume, coverage), `training/p0_smoke.py` (smoke end-to-end con backbone
reale), `configs/p1_mi300x.toml` (valori risolti proposti ad Astra), `tools/backend_gate.py`,
`tools/wildcard_gate.py`, `tools/backbone_profile.py`, `tools/fetch_*.py`, `droplet/` (bootstrap, smoke,
lancio P1), `pbs/` e `tools/submit_run.py` (corsia Polimi, in riserva).

## Come si esegue (droplet MI300X)

1. Sync: `rsync` della cartella in `/root/wavelet-rope/code/` (vedi `droplet/README.md`).
2. `bash droplet/bootstrap.sh` (idempotente: scratch, container `wrope`, pin, asset verificati, fla-core).
3. `bash droplet/run_smoke.sh` (test → p0_smoke → mini-training). Fallisce al primo errore.
4. Fase A: `CODE_DIR=/workspace/code_p2 CONFIG=configs/p1dA_mi300x.toml bash droplet/run_p1.sh p1-dA --max-optimizer-steps 500` + `bash droplet/probe_watch.sh p1-dA`; fase B: `... CONFIG=configs/p1dB_mi300x.toml bash droplet/run_p1.sh p1-dB --init-from /workspace/runs/p1-dA/checkpoints/<step> --skip-curriculum --val-every 400` (tmux; riprendere = stesso comando; `CODE_DIR` e `CONFIG` scelgono albero e config nel container; `--val-every` non ha default).
5. `CODE_DIR=... CONFIG=... bash droplet/gate_early.sh p1-c 400` (sentinella: gate r20 sul checkpoint 400) e, a fine run, `bash droplet/post_p1.sh p1-c` con le stesse variabili.

Nessun training va lanciato senza smoke verde. Il test 800 non viene usato per selezione o tuning.
