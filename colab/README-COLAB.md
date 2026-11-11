# Catena wavelet su Google Colab — runbook

Continuazione della catena Kaggle quando la quota GPU settimanale è esaurita. CLI ufficiale `colab` (v0.6.0) installata su WSL (`/home/jacop/.local/bin/colab`, auth oauth2 già attiva sull'account Google personale).

## Hardware verificato (sonda 2026-08-27)

- Account free tier: **solo T4 singola, 15.3GB VRAM** (L4/A100 rifiutate: "no quota or entitlement")
- RAM 13.6GB, disco /content 70GB, torch 2.11.0+cu128 preinstallato
- Limiti free tier dinamici (~3-4h GPU/giorno tipici, VM revocabile in ogni momento) → checkpoint ogni 300 step nel config c*, budget training conservativo in session_c*.json
- Keep-alive automatico della CLI (nessun tab browser richiesto); tetto 24h/sessione

## Differenza di ricetta vs Kaggle (documentare nei report!)

Kaggle 2×T4 = 30GB → fp32 nativo. Colab T4 singola = 15GB → **fp16 + AMP** (`TSLM_AMP=1` in train.py: autocast + GradScaler + master weights fp32 per i trainabili). Il fp16 puro dava NaN su Kaggle proprio per l'assenza di GradScaler; questa è la cura, non un workaround. Il resume weights-only fp32→master fp32 non perde precisione.

## Procedura sessione Colab (c1, poi c2, c3…)

Prerequisito: checkpoint più recente in locale. Dall'output Kaggle:
```bash
kaggle kernels output jacopodabundo/timerlm-wavelet-s3 -p "esperimenti/colab/incoming-s3"
```
(il best è in `results_out/.../checkpoints/best_model.pt`; se la sessione Kaggle è morta prima del best, usare il `checkpoint_epoch_*_batch_*.pt` più alto o `latest_checkpoint.pt`)

Poi da WSL (`wsl bash -lc "..."`), tutti i path Windows via `/mnt/c/...`:

```bash
# 1. tar dei file patch (contiene train.py con AMP + i config wavelet_precise_c*.yaml)
tar -czf /tmp/patch.tar.gz -C "<cartella-locale>/esperimenti/kaggle/patch" .

# 2. sessione GPU
colab new -s wavc1 --gpu T4

# 3. upload input (checkpoint ~100-200MB: qualche minuto)
colab upload -s wavc1 /tmp/patch.tar.gz /content/patch.tar.gz
colab upload -s wavc1 <path del best_model.pt> /content/prev_best_model.pt
colab upload -s wavc1 ".../esperimenti/colab/session_c1.json" /content/session.json

# 4. setup + lancio detached (~15-25 min sincroni: install, gold seed-42, shard HF)
colab exec -s wavc1 -f ".../esperimenti/colab/colab_setup_train.py"
# stampa "LAUNCHED pid=..." e ritorna: il training gira in nohup su /content/train.log

# 5. polling (ogni 15 min; STATE: RUNNING oppure STATE: EXIT:<code>)
colab exec -s wavc1 -f ".../esperimenti/colab/poll_train.py"

# 6. a STATE: EXIT:0 — impacchetta e scarica risultati+checkpoint, poi SEMPRE stop
colab exec -s wavc1 -f ".../esperimenti/colab/pack_results.py"
colab download -s wavc1 /content/results_out.tar.gz ".../esperimenti/colab/results_c1.tar.gz"
colab stop -s wavc1
```

**`colab stop` è obbligatorio a fine sessione** (una VM idle brucia quota). Se la VM viene revocata a metà run: i checkpoint sotto results/ sono persi con la VM (il disco è effimero!) — per run lunghi conviene un poll che ogni ora scarichi `latest_checkpoint.pt` in locale come assicurazione:
```bash
colab download -s wavc1 /content/timerlm/sft/tslm/results/anomaly_xl_precise/anomaly_xl_wavelet_precise_c1/checkpoints/latest_checkpoint.pt backup_latest_c1.pt
```

## Sessione successiva (c2…)

1. Copiare `session_c1.json` → `session_c2.json`: subset_seed +1 (c1=1044, c2=1045…), budget a piacere.
2. Copiare `patch/wavelet_precise_c1.yaml` → `wavelet_precise_c2.yaml` aggiornando `name`/`run_name` (max_samples SEMPRE 2000: fissa le stesse 200 righe di val/test di s2/s3).
3. `prev_best_model.pt` = best della sessione precedente (dal tar scaricato).
4. Stessa procedura con `-s wavc2`.

## Rientro su Kaggle (sabato, quota rigenerata)

Il checkpoint Colab deve tornare raggiungibile dai kernel: creare/aggiornare un dataset privato Kaggle col best più recente:
```bash
# una tantum: cartella con dataset-metadata.json (id jacopodabundo/timerlm-wavelet-ckpt) + best_model.pt
kaggle datasets create -p <cartella>   # poi: kaggle datasets version -p <cartella> -m "ckpt cN"
```
e nella sessione Kaggle s4: aggiungere `"jacopodabundo/timerlm-wavelet-ckpt"` ai `dataset_sources` — `find_prev_checkpoint` lo trova (il path contiene "wavelet" e il file si chiama best_model.pt).

## Note operative CLI (imparate/verificate)

- La CLI su WSL aveva `jupyter_kernel_client` 1.0.2 (API incompatibile): **pinnata 0.15.0** nel venv `/home/jacop/.venvs/google-colab-cli` il 2026-08-27. Se un update la rompe di nuovo: stesso fix.
- `colab drivemount`, `colab auth`, `repl`, `console` sono interattivi: NON usarli da agente/script.
- Lo stato kernel persiste fra `colab exec` successivi; `colab stop`/`restart-kernel` lo azzera.
- Quoting: mai comandi python inline attraverso PowerShell→wsl→bash; sempre `colab exec -f file.py`.
- `colab sessions` verifica auth + sessioni attive; `colab whoami` mostra scope e scadenza token.
