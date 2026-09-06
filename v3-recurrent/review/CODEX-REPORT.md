# CODEX REPORT — v3 “percettore wavelet ricorrente”

Data revisione: 2026-09-06. Perimetro: `v3-recurrent/` e sole righe di registrazione
in `kaggle/patch/train.py`. Nessun commit, nessun accesso di rete.

## 0. Collasso dei 256 token: causa verificata e correzione

L'ipotesi di Jacopo è confermata dal codice. La bozza costruiva ogni query come
`z_in = z + step_embed.weight[step]`: `z` partiva da `N(0, 0,02)`, mentre
`nn.Embedding` inizializza `step_embed` a scala circa 1. La stessa componente di step era
quindi circa 50 volte più grande della differenza tra slot. La `LayerNorm` non rimuove una
traslazione vettoriale comune tra i 256 slot: normalizza le feature di ciascuno slot. Le query
normalizzate restavano perciò quasi uguali, la cross-attention produceva la stessa lettura e
self-attention/MLP conservavano la simmetria. Il gradiente poteva in teoria romperla, ma il
segnale differenziale sui latenti era troppo piccolo rispetto al percorso comune; i numeri
misurati (coseno 1,0000 e std tra token 0,0013–0,0053 su std totale circa 1) sono la firma
diretta di questo meccanismo.

L'ispezione ha mostrato due amplificatori ulteriori. Anche `band_embed`, creato con
`nn.Embedding`, partiva a scala circa 1 e copriva gran parte delle feature valore/posizione
dentro ciascun livello. Inoltre semplici latenti Fourier diversi non rendono automaticamente
locale una `MultiheadAttention` con proiezioni Q/K casuali. Infine, lasciare l'ancora fissa nel
residuo emesso aumentava la diversità apparente ma continuava a dominare la norma: due input
molto diversi cambiavano l'uscita di meno dell'1%. Per questo il fix non si limita a cambiare
una deviazione standard.

Ho scelto l'opzione Fourier (a), non latenti casuali `N(0,1)`: i casuali rompono la
permutazione ma non assegnano alcun significato temporale agli slot. Il fix applicato è:

- posizioni `(i+0,5)/N` con le stesse `pos_freqs` dei token, feature Fourier centrate,
  proiezione ortogonale a D e normalizzazione RMS 1 per inizializzare `latents`;
- `step_embed` e `band_embed` inizializzati a std 0,02;
- separazione fra ancora di query e stato ricorrente di contenuto, che parte da zero;
- bias locale heavy-tailed `-log(1 + (Δpos/(2/N))²)` nella cross-attention: parte locale ma
  lascia raggiungibili le chiavi lontane, così i logit appresi possono allargare la lettura;
- attenuazione del 90% della sola componente broadcast tra slot, conservandone il 10% come
  contesto globale, e residuo posizionale centrato prima della LayerNorm finale.

Forma `(B,N,D)`, `pos_dtype: input`, nomi/forme dei parametri, chiavi slim, config e contratto
pubblico restano invariati. I nuovi buffer `latent_positions` e `pos_freqs` sono entrambi
`persistent=False`. I vecchi checkpoint hanno chiavi compatibili, ma non devono essere usati
per inizializzare una nuova run: conterrebbero proprio latenti/embedding collassati o alla
vecchia scala.

`token_diversity(enc, x, comparison=None)` è ora la diagnostica condivisa da test e smoke.
Riporta coseno medio off-diagonal, std tra token, std totale, il loro rapporto e differenza
relativa tra due uscite; senza `comparison` trasla la serie di un terzo lungo l'asse temporale.
Lo smoke ricostruisce l'encoder dalle sole chiavi `ts_encoder.*` del checkpoint slim dopo i
quattro step reali, scrive questi numeri nei JSON e fallisce con coseno >0,9 o differenza
relativa ≤0,1. La vecchia bozza fallirebbe con largo margine (coseno 1,0000 e dipendenza
dall'input misurata 0,0008–0,0094).

## Esito e modifiche

La bozza è stata corretta nei punti che potevano compromettere una run non presidiata e nel
collasso rappresentazionale emerso dalla run completa:
checkpointing dell'intera iterazione dell'encoder (embedding incluso), controlli numerici
prima di `topk`, override di profondità fail-fast, una sola estrazione `sdepth` per chiamata,
diagnostica dei gate con una sola sincronizzazione GPU, worker fail-closed e preflight di
backend/spazio disco.

- `patch/wavelet_encoder_v3.py`: query Fourier, scale corrette, lettura locale, stato di
  contenuto separato, diagnostica riusabile, oltre a validazioni e checkpoint non-reentrant.
- `patch/test_encoder_v3.py`: 20 test, inclusi i quattro gate di regressione sul collasso a
  init e dopo quattro step sintetici, oltre a parità v1 bf16, round-trip/slim e checkpointing.
- `node/*.sh`: verifica CUDA/ROCm, smoke 16×131072 tied/untied, limite memoria,
  preflight disco e worker irrigidito.
- `configs/*.yaml` e `README.md`: corretta la documentazione; `training:` invariato.
- `kaggle/patch/train.py`: l'import opzionale ignora solo l'assenza del modulo v3;
  errori interni alla v3 ora abortiscono.

I file v1 `wavelet_encoder.py` e `wavelet.py` non hanno diff.

## 1. Checkpoint

Nei config consegnati `freeze_ts_encoder` è falso. In modalità LoRA il trainer congela
tutto, inietta gli adapter, poi il metodo ereditato `get_trainable_parameters()` prende
`self.ts_encoder.parameters()` dopo lo swap e li rimette allenabili. Il filtro slim include
quindi tutte le chiavi `ts_encoder.*`: `value_proj`, `band_embed`, `latents`,
`step_embed`, tutti i `blocks` (MHA/MLP/LayerNorm), `gate_head` e `ln_out`.

`pos_freqs` è `persistent=False` come in v1 e non entra in alcuno state_dict.
In `--eval-only` gli adapter vengono creati prima del load; `load_checkpoint` usa
`strict=False`, accetta le chiavi frozen mancanti e rende fatali le chiavi inattese.
Il nuovo test riproduce il filtro slim col prefisso `ts_encoder.` e verifica il round-trip.

Rischio residuo: il loader non rende fatale una chiave *trainable* mancante. Stessa overlay,
config e manifest SHA-256 mitigano il rischio sul nodo, ma un checkpoint manualmente
mismatched può lasciare un parametro inizializzato a caso. Non ho ampliato `train.py` oltre
la registrazione consentita dal mandato.

## 2. `from_config` e device

`WaveletTSModel.from_config` è un classmethod: chiamato dalla sottoclasse restituisce già un
`WaveletV3TSModel`. Backbone, tokenizer, proiettore, dtype e opzioni ChatTS restano nella
singola implementazione v1; viene sostituito solo l'encoder.

- Freeze: viene riapplicato al nuovo encoder.
- Gruppi/LR: sono calcolati dinamicamente dopo lo swap.
- `model.to(device)`: sposta anche v3 nel percorso normale.
- `TSLM_DEVICE_MAP`: il backbone può fare dispatch e `load_model` evita il blanket
  `.to()`; encoder/proiettore restano sul device locale come in v1. I run consegnati
  sono single-GPU e non impostano device-map.

Costruire e scartare la piccola v1 ha costo transitorio trascurabile e impedisce divergenze
dalla logica di inizializzazione di riferimento.

## 3. Numerica bf16

Il trainer usa master fp32 per i trainable e autocast bf16. Input, `pos_freqs` e posizioni
restano bf16 con `pos_dtype=input`, quindi la featurizzazione replica v1. MHA e LayerNorm
restano nel contesto autocast; lo smoke GPU obbligatorio copre forward/backward tied e untied.
Il bias di distanza viene calcolato in fp32 dalle posizioni e convertito al dtype dei token
prima di MHA; non cambia il contratto `pos_dtype` della featurizzazione.

Prima di `topk` ora sono richiesti finiti: normalizzazione robusta, piramide Haar e
`score=abs(d)/(1.4826*median(abs(d))+1e-8)`. Una serie costante produce `xn=0` e score
zero. Con L=64 e i default si hanno un livello, grossolana+D1. `torch.where` riceve tensori
materializzati dello stesso dtype; la view espansa iniziale non è modificata in-place.

## 4. Memoria/tempo A40

Un livello massimo vale `16×1024×512×2 byte = 16 MiB`; dieci dettagli più la grossolana
valgono circa 162 MiB per i soli embedding. Senza checkpoint si sommano value-MLP e blocchi.
Con `TSLM_GRAD_CKPT=1` l'intera iterazione è ricomputata: il principale stato salvato è il
confine `16×256×512×2 = 4 MiB` per step, circa 44 MiB a 11 step, più coefficienti e overhead.
La nuova matrice di bias del caso massimo occupa `16×8×256×1024×2 byte = 64 MiB` durante
una singola iterazione bf16; è costruita dentro la closure checkpointed e non resta viva per
tutti gli 11 livelli. Il tetto end-to-end dello smoke resta quindi il gate definitivo.

Parametri: tied ≈4,63 M; untied ≈46,68 M. Con master, gradiente e due momenti Adam,
untied aggiunge ≈0,70 GiB di stato residente (tied ≈0,07 GiB); incremento totale atteso,
con attivazioni/allocator, sotto circa 1–1,2 GiB. È plausibile rispetto a va1 con Qwen
checkpointed, ma non sostituisce una misura end-to-end. `node_env_v3.sh` misura entrambi
gli encoder sul caso 16×131072 e abortisce sopra 8 GiB. L'untied ha anche checkpoint
frequenti grandi: il launcher richiede 35 GiB liberi (15 per tied/sdepth).

## 5. `sdepth`

`torch.randint` usa il RNG globale torch, seminato dal trainer: la sequenza è riproducibile
a parità di ordine chiamate/ambiente. C'è una sola estrazione per forward pubblico; i canali
`(C,L)` dell'esempio batch 1 condividono la profondità. Anche `multi_variable=True` la
risolve una volta.

Sì, sdepth consuma RNG che `full` non consuma: la casualità successiva diverge. È accettabile
e documentato; il confronto è riproducibile per seed, non bit-identico nella sequenza RNG.

## 6. `noise_gate`

La regola è `max(abs(d)/sigma) > noise_tau*sqrt(2 ln n)`, equivalente alla soglia universale
`sigma*sqrt(2 ln n)` sul massimo non normalizzato. Il top-m contiene sempre il massimo.
Resta cieca a variazioni di varianza diffuse/moderate senza un picco oltre soglia.

Variante futura separata, non implementata: `noise_gate_stat: max|energy` con media
winsorizzata di `(abs(d)/sigma)^2` e soglia calibrata sul rumore. Deve essere una nuova
ablation/config, non una modifica silenziosa del default.

## 7. Gate appreso

Input: `[mean(query), mean(e), stats]`, dove `query = stato + ancora Fourier + step`. Il bias
+2 dà `sigmoid(2)=0,8808`; pesi zero, ma
derivata ≈0,105, quindi non parte saturo. Il rischio di chiusura dei fini esiste, soprattutto
in sdepth dove sono visti meno spesso. Non aggiungo regolarizzazione prima di evidenza:
forzare gate aperti confonderebbe l'ablazione. Il launcher scrive ogni 100 chiamate una riga
`[V3Gates]`. Se il collasso è precoce e stabile, valutare separatamente un piccolo prior
verso gate aperto.

## 8. Script nodo e ROCm

Restano i gate va1: preflight anti-leakage, Qwen pinnato, grad-ckpt, coverage totale,
cinque checkpoint, eval strict, best-by-val-primary e un solo test.

Il worker valida positivamente run/config/revisione, liveness, timeout, `EXIT:0`, insieme
esatto dei checkpoint, una coverage esatta per epoca, JSON e `primary` finita in [0,1].
Il gate coverage normalizza i carriage return di tqdm prima dei match esatti e il worker
rimuove all’avvio un eventuale marker `WORKER_FAILED` stantio, così da poter essere rieseguito.
Per v3sdepth la curva `1,2,3,4,6,8,11` usa la val e il checkpoint best; `FINAL_DONE` viene
emesso solo dopo lo sweep. Il test non seleziona la profondità.

`V3_ROCM=1` è strutturalmente plausibile: passa indice/pin ROCm 6.3 al `node_env.sh`,
PyTorch usa comunque `torch.cuda` e i launcher impostano CUDA/HIP/ROCR visible devices.
Non ho usato la rete per verificare la disponibilità attuale delle wheel. Il setup fallisce
prima del training se pin, `torch.version.hip`, bf16 o backward non passano.

## 9. Test

Comando obbligatorio:

```powershell
py v3-recurrent/patch/test_encoder_v3.py
```

I 20 test coprono round-trip strict, simulazione slim, prefisso `ts_encoder.*`,
assenza di `pos_freqs`, parità esatta v1/v3 su grossolana+D1 in bf16, equivalenza
`TSLM_V3_DEPTH=steps`/full, override invalido, costante L=64, non-finiti, sdepth e
gradienti con/senza checkpoint. I due nuovi test usano esattamente 256 token e verificano a
init e dopo quattro step di classificazione sintetica: coseno <0,5, rapporto std ≥0,3,
differenza tra serie >0,1, differenza tra anomalie in posizioni diverse >0,1 e rapporto tra
norma del gradiente di `latents` e di tutto `value_proj` >`1e-2`.

Verifiche eseguite:

- `py v3-recurrent/patch/test_encoder_v3.py`: **20/20 passati** su CPU;
- `git diff --check`: passato;
- `bash -n v3-recurrent/node/node_smoke_v3.sh`: passato;
- diff verificato: nessuna modifica a `kaggle/patch/wavelet_encoder.py` o `wavelet.py`.

Lo smoke A40 non è stato eseguito in locale: deve ancora validare memoria, SDPA bf16 e le
soglie sul checkpoint realmente prodotto dopo quattro esempi.

## Rischi residui e checklist Claude

Rischi: picco/throughput end-to-end A40 non ancora misurati; chiave trainable mancante non
fatale con strict=False; checkpoint untied grandi; wheel/SDPA ROCm da provare; noise-gate
sul massimo; possibile chiusura dei gate fini.

Prima del lancio Claude deve:

1. eseguire la suite e richiedere 20/20;
2. controllare la closure checkpointed, in particolare `_band` e `_step` legati come default;
3. richiedere `V3_BACKEND_OK`, due `V3_GPU_SMOKE` e `V3_ENV_OK`;
4. fare un forward/backward completo Qwen su una cella reale 16×131072;
5. proiettare il disco dopo il primo checkpoint untied;
6. aprire uno slim e verificare latents, step_embed, gate_head, blocks e ln_out;
7. verificare righe `[V3Gates]` e una coverage esatta per epoca;
8. su MI300X non procedere se versione ROCm, bf16 o SDPA falliscono.
