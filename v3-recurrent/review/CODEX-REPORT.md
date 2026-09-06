# CODEX REPORT — v3 “percettore wavelet ricorrente”

Data revisione: 2026-09-05. Perimetro: `v3-recurrent/` e sole righe di registrazione
in `kaggle/patch/train.py`. Nessun commit, nessun accesso di rete.

## Esito e modifiche

La bozza è stata corretta nei punti che potevano compromettere una run non presidiata:
checkpointing dell'intera iterazione dell'encoder (embedding incluso), controlli numerici
prima di `topk`, override di profondità fail-fast, una sola estrazione `sdepth` per chiamata,
diagnostica dei gate con una sola sincronizzazione GPU, worker fail-closed e preflight di
backend/spazio disco.

- `patch/wavelet_encoder_v3.py`: validazioni, calcolo dei soli livelli letti,
  checkpoint non-reentrant di featurizzazione+gate+blocco, profondità condivisa dalle
  variabili e log opzionale `[V3Gates]`.
- `patch/test_encoder_v3.py`: 18 test, inclusi parità v1 bf16, round-trip/slim,
  checkpointing, costanti/corte, non-finiti e riproducibilità sdepth.
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

Prima di `topk` ora sono richiesti finiti: normalizzazione robusta, piramide Haar e
`score=abs(d)/(1.4826*median(abs(d))+1e-8)`. Una serie costante produce `xn=0` e score
zero. Con L=64 e i default si hanno un livello, grossolana+D1. `torch.where` riceve tensori
materializzati dello stesso dtype; la view espansa iniziale non è modificata in-place.

## 4. Memoria/tempo A40

Un livello massimo vale `16×1024×512×2 byte = 16 MiB`; dieci dettagli più la grossolana
valgono circa 162 MiB per i soli embedding. Senza checkpoint si sommano value-MLP e blocchi.
Con `TSLM_GRAD_CKPT=1` l'intera iterazione è ricomputata: il principale stato salvato è il
confine `16×256×512×2 = 4 MiB` per step, circa 44 MiB a 11 step, più coefficienti e overhead.

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

Input: `[mean(z), mean(e), stats]`. Il bias +2 dà `sigmoid(2)=0,8808`; pesi zero, ma
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

I nuovi test coprono round-trip strict, simulazione slim, prefisso `ts_encoder.*`,
assenza di `pos_freqs`, parità esatta v1/v3 su grossolana+D1 in bf16, equivalenza
`TSLM_V3_DEPTH=steps`/full, override invalido, costante L=64, non-finiti, sdepth e
gradienti con/senza checkpoint.

Verifiche eseguite:

- `PY_COMPILE_OK 3` con Python 3.14;
- `bash -n` su `node/*.sh` e `ops/*.sh`;
- parsing PyYAML: dataset, backbone, projector, extra_kwargs e training uguali a va1;
- sezioni `training:` byte-identiche a va1;
- diff Git: v1 intatta.

Limite dell'ambiente: la suite torch non è stata eseguita qui. Il terminale gestito fallisce
con `CreateProcessWithLogonW failed: 2`; il Python 3.14+torch del profilo utente è visibile
nel registro ma il sandbox risponde `Accesso negato`. L'interprete 3.14 accessibile ha
permesso il compile ma non contiene torch. Non dichiaro quindi un falso “18/18”: il comando
`py` sopra è il primo gate obbligatorio per Claude o per il nodo.

## Rischi residui e checklist Claude

Rischi: picco/throughput end-to-end A40 non ancora misurati; chiave trainable mancante non
fatale con strict=False; checkpoint untied grandi; wheel/SDPA ROCm da provare; noise-gate
sul massimo; possibile chiusura dei gate fini.

Prima del lancio Claude deve:

1. eseguire la suite e richiedere 18/18;
2. controllare la closure checkpointed, in particolare `_band` e `_step` legati come default;
3. richiedere `V3_BACKEND_OK`, due `V3_GPU_SMOKE` e `V3_ENV_OK`;
4. fare un forward/backward completo Qwen su una cella reale 16×131072;
5. proiettare il disco dopo il primo checkpoint untied;
6. aprire uno slim e verificare latents, step_embed, gate_head, blocks e ln_out;
7. verificare righe `[V3Gates]` e una coverage esatta per epoca;
8. su MI300X non procedere se versione ROCm, bf16 o SDPA falliscono.
