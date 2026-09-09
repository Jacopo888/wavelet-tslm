# v3-recurrent — residuo ricorrente multi-scala sui token v1

Variante **separata** dell'encoder del Livello 1. La v1 (`kaggle/patch/wavelet_encoder.py`,
riferimento va1) non viene toccata: la v3 vive in questa cartella, viene sovrapposta sul nodo
come file nuovi (`src/models/wavelet_encoder_v3.py`, `src/models/wavelet_v3.py`) e si seleziona
con `model.architecture: wavelet_v3`. Le sole modifiche fuori cartella sono in
`kaggle/patch/train.py`: registrazione condizionale, diagnostiche, init comune, ordine dati e
arresto/checkpoint dei pilot. Ogni hook nuovo è protetto da env ed è inerte di default; la chiave
`wavelet` e i file v1 restano invariati.

## Stato corrente (09/09/2026)

L'audit ha mostrato che il lettore `full` storico collassa verso un modo quasi condiviso fra token.
L'implementazione corrente aggiunge, in quest'ordine:

- `reader_mode: cross_only|full`; `cross_only` accumula soltanto letture cross-attention e rimuove
  il modo medio condiviso, mentre `full` conserva self-attention e MLP;
- scale apprese per blocco `self_attn_scale_init` e `mlp_scale_init`; il default 1,0 conserva il
  forward `full` storico. Con init 0,1 il centraggio locale interpola verso il controllo
  `cross_only` (`local_center_strength=0,9`), senza cambiare il default;
- `context_scale` in un gruppo optimizer separato, con fallback al LR `context` quando la chiave
  dedicata manca; i pilot usano 1e-5;
- `token_gate: true` opzionale, condiviso fra token, inizializzato a identità: dopo LN,
  `g_i=2 sigmoid(Linear([LN_noaffine(base_i),h_i]))` e `out_i=base_i+a g_i h_i`;
- statistiche del solo contesto in `last_context_stats` e righe `[V3Ctx]`: RMS prima/dopo LN,
  diversità/coseno fra token, dipendenza dall'input, componenti cross/self/MLP e gate finale.

I tre config `v3pilot_{cross,fullsmall,gate}` sono i candidati attuali. I vecchi config
`v3tied/v3untied/v3sdepth` restano descrittivi della ricetta `full`, scale 1,0 e LR scala ereditato:
non sono l'ordine consigliato per i pilot dopo l'audit.

## Idea in una riga

V3 parte dagli stessi 256 token di v1 e somma soltanto un residuo contestuale letto
ricorrentemente dalla piramide. I config inizializzano la scala a 0,1 per dare gradiente ai
blocchi dal primo backward; disabilitando il residuo, `cross_only` e `full` restano entrambi
bit-per-bit identici a v1.

## Architettura (`patch/wavelet_encoder_v3.py`, classe `RecurrentWaveletTSEncoder`)

1. **Base v1 esatta**: `RecurrentWaveletTSEncoder` eredita `WaveletTSEncoder` e invoca il suo
   `_encode_channel` senza copiarne l'implementazione. Normalizzazione mediana/MAD, banda
   grossolana intera, top-k globale dei dettagli per |d|/σ, `value_proj`, `band_embed`, Fourier,
   `pad_token` e dtype delle posizioni sono quindi quelli di v1. `wavelet_v3.py` copia inoltre
   nel nuovo encoder i pesi del v1 appena costruito: con lo stesso seed il controllo va1 e v3
   hanno lo stesso identico punto di partenza. Posizione reale `p_i`, banda `b_i` e validità di
   ogni token vengono conservate; i soli slot di padding usano banda sentinella `-1`.
2. **Una sola piramide per canale**: la chiamata v1 registra la piramide che ha già calcolato e
   il loop riusa quei tensori. Non c'è una seconda trasformata Haar. Con L=131072 si hanno dieci
   livelli di dettaglio più la banda grossolana.
3. **Residuo ricorrente sulla scala**: a ogni iterazione le query sono i 256 token base più un
   piccolo `step_embed`. Leggono A_L, poi D_L … D_1; ogni dettaglio espone al massimo
   `level_topm=1024` coefficienti scelti per lo stesso |d|/σ. La cross-attention usa un prior
   Cauchy centrato sulla posizione reale `p_i` del token base (`locality_sigma=0.0078125`), non
   su `i/256`. In `cross_only` si accumulano solo queste letture; in `full` seguono self-attention
   e MLP, ciascuna moltiplicata dalla propria scala appresa. `tied: true` riusa un solo blocco,
   `tied: false` ne usa uno per iterazione.
4. **Gate per livello** (`level_gate: true`): uno scalare sigmoid per (serie, iterazione) calcolato
   da [media delle query base, media dei token del livello, statistiche del livello: frazione sopra
   2σ e 3σ, log del massimo, indice], che moltiplica la lettura. Inizializzato quasi aperto
   (0,88). È il modo *morbido* di ignorare un livello inutile; i valori restano in
   `last_gates` e il launcher emette `[V3Ctx]` ogni 100 chiamate: sono la prima
   evidenza empirica su "quali livelli usa".
5. **Profondità** (`depth_mode`): `full` = tutte le iterazioni; `sdepth` = in training si tronca
   a una profondità casuale per esempio (stile Huginn), così a inferenza ogni profondità è
   valida e la curva metrica-vs-profondità si ottiene gratis; `noise_gate` = una serie non
   legge un livello di dettaglio se il suo coefficiente massimo normalizzato è sotto la soglia
   universale σ·√(2 ln n) (Donoho–Johnstone: il massimo atteso di n coefficienti di puro rumore).
   In eval `TSLM_V3_DEPTH=r` tronca a r iterazioni per qualsiasi modello. `sdepth` estrae una
   sola profondità per chiamata (quindi condivisa dai canali dell'esempio con batch 1) dal RNG
   globale torch: il seed del trainer la rende riproducibile, ma la sequenza RNG diverge da `full`.
6. **Uscita residuale e fix del deadlock**: senza token gate,
   `out = token_v1 + context_scale · LN(contesto)`; con gate,
   `out_i = token_v1_i + context_scale · g_i · LN(contesto)_i`.
   La LayerNorm vede solo il contesto e non può rimescolare il token base. Il default della
   classe resta `context_scale_init: 0.0` per compatibilità e per il gate `v3 == v1`; i cinque
   config consegnati usano invece `0.1`. Infatti al primo step
   `∂L/∂θ_context = context_scale · ∂LN(z)/∂θ_context · ∂L/∂out`: con scala zero blocchi, gate,
   step embedding e norm non co-adattano, mentre lo scalare vede soltanto la correlazione con
   un ramo casuale. È il deadlock osservato nella run `1b0f9fa`: scala 0,0001/−0,0006/0,0004
   a epoca 1/3/5 e gate bias circa 1,98. Con init 0,1 il gradiente dei blocchi è non nullo dal
   primo backward. `WaveletV3TSModel.get_trainable_parameters()` separa `context` (lettore),
   `context_scale` e `context_token_gate` dal percorso v1. I checkpoint residual-on-v1 restano
   caricabili come pesi (scale mancanti migrano a 1 e il gate a identità), ma uno stato optimizer
   con topologia storica non viene caricato silenziosamente: occorre l'esplicito
   `TSLM_RESUME_WEIGHTS_ONLY=1`.

Costo: cross-attention 256 × ≤1024 per livello × ≤11 livelli × canali. La prior posizionale
usa una matrice target×source per ciascun batch/head. Con `TSLM_GRAD_CKPT=1` ogni iterazione
ricomputa embedding del livello, prior locale, gate e blocco: non restano
vive le grandi attivazioni `(16,1024,512)` di tutti i livelli; restano soprattutto i confini
ricorrenti `(16,256,512)`. Test CPU:
`py v3-recurrent/patch/test_encoder_v3.py` (30 test:
forme, budget, override, sdepth riproducibile, noise-gate, tied/untied, gradienti, bf16, serie
costanti/corte, fail-fast non-finiti, parità di featurizzazione v1, checkpoint slim e round-trip,
identità v1/v3 fp32 e bf16 anche su serie corte e 16×131072, una sola piramide, diversità,
witness zero/nonzero dopo 16 step sintetici, split del gruppo optimizer e ridge probe held-out). `token_diversity(enc, x)`
riporta coseno medio off-diagonal, std tra token/std totale e differenza relativa rispetto a
una serie di confronto (per default la stessa serie traslata temporalmente).
`position_decodability(enc, n=...)` genera spike, level shift e variance change fra 4096 e
32768 punti e risolve una ridge CPU senza Qwen su split held-out.

I D_1 non consumano token Qwen: possono soltanto modificare il residuo. Top-m, gate e arresto
controllano quanto rumore fine venga aggiunto alla base v1.

## Ablazioni storiche a cinque epoche (non sono i pilot correnti)

| # | Run | Cosa cambia rispetto alla precedente | Domanda a cui risponde |
|---|-----|--------------------------------------|------------------------|
| 1 | `v3tied` vs **va1** (da ripetere col fix) | selezione v1 + contesto ricorrente multi-scala vs selezione v1 sola; la run `1b0f9fa` a scala zero è risultata una replica di va1 e misura solo rumore run-to-run | Il contesto aggiuntivo migliora IoU sui positivi, F1 sui non-vuoti e lead_lag senza perdere la decodificabilità di v1? |
| 2 | `v3untied` vs `v3tied` | un blocco per livello invece di uno condiviso (11× parametri, stesso calcolo) | Il guadagno viene dal *prior di condivisione tra scale* (la ricorrenza) o solo dalla profondità/attenzione? Se tied ≥ untied la ricorrenza è un vero bias induttivo, non un risparmio |
| 3 | `v3sdepth` vs `v3tied` + curva `TSLM_V3_DEPTH = 1,2,3,4,6,8,11` sulla val | training a profondità stocastica; a eval si varia quante scale il lettore consulta | Quando fermarsi: quanto vale ogni livello in più, i livelli fini aiutano o fanno male, e la profondità scelta è coerente con la larghezza dell'anomalia. Confronto con il gate appreso di (4) e con la regola `noise_gate` (valutabile sul modello `v3tied` senza ri-addestrare) |

Il percettore a soli latenti dei commit `cfd5256` e `0d2484e` resta lavoro futuro, non un
checkpoint compatibile. Nel probe ridge a init aveva MAE posizione 0,112 e poi 0,105 contro
0,078 di v1 (1,45× e 1,36×) e presenza 0,82–0,85 contro 0,94. La run `cfd5256` dopo un'epoca
produceva finestre quasi costanti: primary 0,053 e localize IoU 0,0 su 80 esempi val, contro
primary 0,17 di va1 alla stessa epoca. Il nuovo residuo elimina quel deficit a init per
costruzione.

Questa sequenza precede l'audit del collasso ed è conservata soltanto come storia. Per i pilot
attuali usare `cross → fullsmall → gate`; nessun pilot legge il test.

Rischi dichiarati: (i) il noise-gate è una regola sul massimo, quindi cieca alle anomalie di
varianza (che non producono coefficienti grandi ma piccoli): l'ablazione 3 lo misura per
categoria; (ii) `pos_dtype: input` replica v1 anche nella quantizzazione bf16 delle posizioni
(la correzione fp32 è la v2 e va tenuta separata); (iii) il ridge probe è un gate sintetico e
non sostituisce IoU/localize: conta il best-by-val, ma una decodificabilità molto peggiore di v1
ora interrompe test e smoke prima di spendere una run completa.

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
  train.py reale, casi 16 canali/131072, memoria, timing, checkpoint ed eval-only fail-closed,
  diagnostica post-training sullo slim checkpoint con soglie coseno/input-dependence e ridge
  probe posizione/presenza/magnitudo rispetto a v1, con numeri nei JSON),
  `node/node_train_v3.sh <run>` (launcher va1 parametrizzato, una run per chiamata, sessioni tmux
  separate), `node/node_eval_worker_v3.sh` (fail-closed: coverage dal log, 5 val, best, test 800,
  poi curva di profondità sulla val).
- `node/node_pilot_v3.sh <nome> <config> <n_epochs>` — run dir univoca, scheduler fisso
  22440/673, init comune e ordine dati fissato; valuta soltanto val a step 500/2000 e fine epoca,
  con probe80 generation/teacher-forcing on/off più val intera on a fine epoca.
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
bash ~/node_pilot_v3.sh cross configs/wavelet_precise_v3pilot_cross.yaml 1
bash ~/node_pilot_v3.sh fullsmall configs/wavelet_precise_v3pilot_fullsmall.yaml 1
bash ~/node_pilot_v3.sh gate configs/wavelet_precise_v3pilot_gate.yaml 1
```

Stato: suite CPU 30/30; lo smoke richiede gradienti e spostamento misurabile dei pesi del lettore,
non uno spostamento minimo della scala lenta, e fallisce se la frazione variabile del contesto è
inferiore a 1e-3. La run `1b0f9fa` è archiviata come replica va1; nessuna nuova run GPU lanciata.
