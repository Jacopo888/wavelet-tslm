# v3-recurrent — percettore wavelet ricorrente (looped layer sulla scala)

Variante **separata** dell'encoder del Livello 1. La v1 (`kaggle/patch/wavelet_encoder.py`,
riferimento va1) non viene toccata: la v3 vive in questa cartella, viene sovrapposta sul nodo
come file nuovi (`src/models/wavelet_encoder_v3.py`, `src/models/wavelet_v3.py`) e si seleziona
con `model.architecture: wavelet_v3`. Le sole modifiche fuori cartella sono in `kaggle/patch/train.py`: la registrazione condizionale
della classe (la chiave `wavelet` resta identica) e il log `[StepTiming]` protetto da
`TSLM_STEP_TIMING=1`, usato soltanto dallo smoke end-to-end.

## Idea in una riga

In v1 i 256 token che entrano in Qwen sono *scelti* (banda grossolana + top-k globale dei
coefficienti di dettaglio per magnitudine MAD-normalizzata) e inseriti direttamente nella
sequenza. In v3 i 256 token sono *latenti* che **leggono la piramide un livello per iterazione,
dal grossolano al fine (A_L, D_L, …, D_1), con lo stesso blocco a pesi condivisi**: un layer
ricorrente il cui asse di ricorsione è la scala, non la profondità. I coefficienti vengono
consultati, mai inseriti: la sequenza di Qwen è identica a v1 (256 token per canale) qualunque
sia la profondità.

## Architettura (`patch/wavelet_encoder_v3.py`, classe `RecurrentWaveletTSEncoder`)

1. **Analisi identica a v1**: normalizzazione robusta (mediana/MAD), piramide Haar
   (`haar_pyramid` importata dal file v1), `levels = ceil(log2(L/128))` limitato a 10; con
   L = 131k sono 10 livelli di dettaglio + banda grossolana = 11 iterazioni.
2. **Token per livello, featurizzazione identica a v1**: valore (clamp ±20) + posizione di Fourier
   (16 frequenze) → `value_proj`, più `band_embed(livello)`. La banda grossolana entra intera
   (128 blocchi); ogni livello di dettaglio espone al massimo `level_topm = 1024` coefficienti,
   i più grandi per lo stesso punteggio |d|/σ di v1 (D_1 di una serie da 131k ha 65k coefficienti:
   il lettore ne vede 1024, non zero e non tutti).
3. **Il loop**: 256 latenti appresi; a ogni iterazione `z ← z + step_embed(t)`, poi un blocco
   pre-LN *cross-attention (z → token del livello) · self-attention fra latenti · MLP*. Con
   `tied: true` il blocco è uno solo per tutte le iterazioni (blocco 4,2 M; encoder completo
   4,63 M parametri); con `tied: false` ce n'è uno per iterazione (encoder completo 46,68 M).
4. **Gate per livello** (`level_gate: true`): uno scalare sigmoid per (serie, iterazione) calcolato
   da [media dei latenti, media dei token del livello, statistiche del livello: frazione sopra
   2σ e 3σ, log del massimo, indice], che moltiplica la lettura. Inizializzato quasi aperto
   (0,88). È il modo *morbido* di ignorare un livello inutile; i valori restano in
   `last_gates` e il launcher emette `[V3Gates]` ogni 100 chiamate: sono la prima
   evidenza empirica su "quali livelli usa".
5. **Profondità** (`depth_mode`): `full` = tutte le iterazioni; `sdepth` = in training si tronca
   a una profondità casuale per esempio (stile Huginn), così a inferenza ogni profondità è
   valida e la curva metrica-vs-profondità si ottiene gratis; `noise_gate` = una serie non
   legge un livello di dettaglio se il suo coefficiente massimo normalizzato è sotto la soglia
   universale σ·√(2 ln n) (Donoho–Johnstone: il massimo atteso di n coefficienti di puro rumore).
   In eval `TSLM_V3_DEPTH=r` tronca a r iterazioni per qualsiasi modello. `sdepth` estrae una
   sola profondità per chiamata (quindi condivisa dai canali dell'esempio con batch 1) dal RNG
   globale torch: il seed del trainer la rende riproducibile, ma la sequenza RNG diverge da `full`.
6. Uscita: LayerNorm sui latenti → (B, 256, 512) → proiettore MLP di v1 → Qwen.

Costo: cross-attention 256 × ≤1024 per livello × ≤11 livelli × canali. Con
`TSLM_GRAD_CKPT=1` ogni iterazione ricomputa embedding del livello, gate e blocco: non restano
vive le grandi attivazioni `(16,1024,512)` di tutti i livelli; restano soprattutto i confini
ricorrenti `(16,256,512)`. Test CPU: `py v3-recurrent/patch/test_encoder_v3.py` (18 test:
forme, budget, override, sdepth riproducibile, noise-gate, tied/untied, gradienti, bf16, serie
costanti/corte, fail-fast non-finiti, parità di featurizzazione v1, checkpoint slim e round-trip).

Perché i D_1 "troppo fini" non sono un problema di budget ma di rumore: non entrano in Qwen;
possono però sporcare i latenti se la cross-attention si diluisce su chiavi di rumore. Le tre
difese sono il top-m per livello (2), il gate (4) e l'arresto (5); l'ablazione 3 misura quanto
servono.

## Le tre ablazioni (protocollo va1 identico: seed 42, 4488 × 5 epoche, Qwen3.5-4B LoRA r16/α32, lr 1e-4, batch 1, bf16, grad-ckpt, coverage garantita, best-by-val-primary, test 800 hfpin)

| # | Run | Cosa cambia rispetto alla precedente | Domanda a cui risponde |
|---|-----|--------------------------------------|------------------------|
| 1 | `v3tied` vs **va1** (già fatto) | solo il modo in cui i coefficienti diventano 256 token: lettura ricorrente vs selezione dura | La ricorsione sulla scala batte la selezione a budget di token identico? Dove: IoU sui positivi (localize), F1 sui non-vuoti (all_channels), lead_lag |
| 2 | `v3untied` vs `v3tied` | un blocco per livello invece di uno condiviso (11× parametri, stesso calcolo) | Il guadagno viene dal *prior di condivisione tra scale* (la ricorrenza) o solo dalla profondità/attenzione? Se tied ≥ untied la ricorrenza è un vero bias induttivo, non un risparmio |
| 3 | `v3sdepth` vs `v3tied` + curva `TSLM_V3_DEPTH = 1,2,3,4,6,8,11` sulla val | training a profondità stocastica; a eval si varia quante scale il lettore consulta | Quando fermarsi: quanto vale ogni livello in più, i livelli fini aiutano o fanno male, e la profondità scelta è coerente con la larghezza dell'anomalia. Confronto con il gate appreso di (4) e con la regola `noise_gate` (valutabile sul modello `v3tied` senza ri-addestrare) |

Ordine di esecuzione: 1 → 2 → 3 (su A40-24Q ≈ 20 h di training + ≈ 4 h di eval per run, ≈ $21
per run; su MI300X 192 GB le tre run girano in parallelo). Ogni run è confrontata a va1 con il
test appaiato di `integrity/diagnostics.py` (bootstrap sulle 800 righe).

Rischi dichiarati: (i) il noise-gate è una regola sul massimo, quindi cieca alle anomalie di
varianza (che non producono coefficienti grandi ma piccoli): l'ablazione 3 lo misura per
categoria; (ii) `pos_dtype: input` replica v1 anche nella quantizzazione bf16 delle posizioni
(la correzione fp32 è la v2 e va tenuta separata); (iii) l'attenzione ha una fase di riscaldamento
in più rispetto a v1: se a epoca 1 la val è sotto va1 non è un segnale, conta il best-by-val.

## File

- `patch/wavelet_encoder_v3.py` — encoder; `patch/wavelet_v3.py` — `WaveletV3TSModel`
  (costruisce il modello v1 e sostituisce solo `ts_encoder`); `patch/test_encoder_v3.py` — test CPU.
- `configs/wavelet_precise_{v3tied,v3untied,v3sdepth}.yaml` — generati da `wavelet_precise_va1.yaml`,
  diff = solo blocco encoder e nomi; `configs/wavelet_precise_v3smoke_{untied,tied}.yaml` —
  config da un'epoca sui mini-parquet deterministici creati dallo smoke.
- `node/node_prep_v3.sh` (pacchetti + GPU positiva), `node/node_env_v3.sh` (env L1 invariato +
  overlay v3 + test CPU + smoke GPU checkpointed 16×131k con tetto 8 GiB +
  `RUN_MANIFEST_V3.json` con sha256 dell'overlay; `V3_ROCM=1` su MI300X e verifica
  che il venv usi davvero wheel ROCm 6.3), `node/node_smoke_v3.sh` (A40, untied poi tied:
  train.py reale, casi 16 canali/131072, memoria, timing, checkpoint ed eval-only fail-closed),
  `node/node_train_v3.sh <run>` (launcher va1 parametrizzato, una run per chiamata, sessioni tmux
  separate), `node/node_eval_worker_v3.sh` (fail-closed: coverage dal log, 5 val, best, test 800,
  poi curva di profondità sulla val).
- `ops/gpu_watch.py` + `ops/start_watch.ps1` — watcher/grab detached (Vultr A40-24 / L40S-48 in
  qualunque regione, DO MI300X in qualunque regione; stato in `%LOCALAPPDATA%\wavelet-v3\`);
  `ops/ship_v3.sh <ip>` — spedizione sul nodo da WSL.

## Sequenza sul nodo

```bash
# dal PC (WSL): bash ops/ship_v3.sh <ip>
bash ~/node_prep_v3.sh                       # PREP_OK
bash ~/node_env_v3.sh                        # V3_ENV_OK  (V3_ROCM=1 bash ~/node_env_v3.sh su MI300X)
bash ~/node_smoke_v3.sh                      # A40: deve terminare con SMOKE_OK
python3 -m json.tool ~/logs/v3smoke.ok        # entrambe le varianti pass=true, picco <= 23000 MiB
bash ~/node_train_v3.sh v3tied               # solo dopo il gate; A40 una run alla volta
cat ~/evalq/v3tied_BEST_EPOCH ~/evalq/v3tied_FINAL_test.json   # dopo FINAL_DONE in ~/logs/evw_v3tied.log
```

Stato: revisione Codex incorporata; suite CPU estesa a 18 test e smoke GPU obbligatorio nel
setup del nodo. Nessuna run GPU lanciata.
