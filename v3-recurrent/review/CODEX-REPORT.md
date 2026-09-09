# CODEX REPORT — v3 “percettore wavelet ricorrente”

Data revisione: 2026-09-07. Perimetro: `v3-recurrent/` e sole righe di registrazione
in `kaggle/patch/train.py`. Nessun commit, nessun accesso di rete.

## 0. Collasso dei 256 token: causa verificata e correzione

Nota storica: questa sezione descrive il fix confluito in `cfd5256`; la nuova evidenza e il
ridisegno che lo sostituisce sono nella sezione 10.

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
- `patch/test_encoder_v3.py`: 21 test, inclusi i gate di regressione sul collasso e il probe a
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

I 21 test coprono round-trip strict, simulazione slim, prefisso `ts_encoder.*`,
assenza di `pos_freqs`, parità esatta v1/v3 su grossolana+D1 in bf16, equivalenza
`TSLM_V3_DEPTH=steps`/full, override invalido, costante L=64, non-finiti, sdepth e
gradienti con/senza checkpoint. I due nuovi test usano esattamente 256 token e verificano a
init e dopo quattro step di classificazione sintetica: coseno <0,5, rapporto std ≥0,3,
differenza tra serie >0,1, differenza tra anomalie in posizioni diverse >0,1, ridge probe e rapporto tra
norma del gradiente di `latents` e di tutto `value_proj` >`1e-2`.

Verifiche eseguite:

- `py v3-recurrent/patch/test_encoder_v3.py`: **21/21 passati** su CPU;
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

1. eseguire la suite e richiedere 21/21;
2. controllare la closure checkpointed, in particolare `_band` e `_step` legati come default;
3. richiedere `V3_BACKEND_OK`, due `V3_GPU_SMOKE` e `V3_ENV_OK`;
4. fare un forward/backward completo Qwen su una cella reale 16×131072;
5. proiettare il disco dopo il primo checkpoint untied;
6. aprire uno slim e verificare latents, step_embed, gate_head, blocks e ln_out;
7. verificare righe `[V3Gates]` e una coverage esatta per epoca;
8. su MI300X non procedere se versione ROCm, bf16 o SDPA falliscono.

## 10. 07/09 — la posizione fluisce, ma il contenuto non era decodificabile

### Diagnosi sul codice e sui risultati di epoca 1

I nuovi dati separano due proprietà che il precedente gate `token_diversity` confondeva:
gli slot possono essere diversi e reagire nella regione corretta, senza che un decoder a valle
riesca a leggere facilmente *che cosa* è cambiato. È precisamente il caso di `cfd5256`.

In v1 il top-k fa già una parte importante della decodifica. Un coefficiente grande viene
inserito direttamente fra i 256 token e il token è
`MLP([coefficiente, Fourier(posizione)]) + band_embed`. Valore e posizione percorrono lo stesso
MLP, entrambi alla scala nativa del token; il proiettore e Qwen ricevono quindi una evidenza
selezionata, non devono prima estrarla da un piccolo residuo.

In `cfd5256`, invece, lo stato letto dalla piramide veniva trasformato in
`LN(z - 0,9 mean_slot(z) + 0,85 anchor)`. L'ancora Fourier centrata ha RMS circa uno ed è fissa;
il contenuto dipendente dall'input osservato era solo il 3–4% dopo la LayerNorm. La LayerNorm
finale, applicata alla somma, non separa i due percorsi: normalizza ogni slot e rende media/RMS
quasi costanti. Qwen deve perciò imparare contemporaneamente la mappa slot→posizione e una
direzione di lettura ad alto guadagno per il residuo. Il proiettore MLP e LoRA r16 possono farlo,
ma i risultati invarianti fra step 1000 e fine epoca mostrano che non lo fanno nel budget dato.
Questo spiega anche perché la loss teacher-forced (0,3966 contro 0,3963 della run collassata)
non discrimina: predire le risposte-moda è facile senza usare l'evidenza numerica.

Il gate non è la causa primaria. A init i pesi sono zero e il bias +2 dà esattamente
`sigmoid(2)=0,880797`: attenua ogni lettura del 12%, non la chiude e non contiene alcuna
selettività di posizione. Anche la prior di località è utile ma non sufficiente: con
`sigma=2/256=0,0078125` il bias Cauchy `-log(1+(delta/sigma)^2)` concentra l'attenzione, però il
valore locale attraversava ancora V e `out_proj` casuali, poi self-attention e MLP. Su D=512,
D1 di una serie sintetica da 4096, ho misurato RMS token di livello 0,1466, uscita della
cross-attention 0,2959 (0,2606 dopo il gate) e lettura locale 0,1082 (0,0953 dopo il gate):
`out_proj` non schiaccia la *norma*, ma ruota/miscela l'evidenza e non offre un percorso identità.

### Ridisegno scelto

Ho mantenuto query, prior, gate, top-m e loop, ma ho separato esplicitamente base locale,
contesto ricorrente e posizione:

1. la prior locale produce anche `local_read = softmax(position_bias) @ level_tokens`, senza
   proiezioni apprese; questa lettura viene gate-ata e accumulata in `direct`;
2. il blocco esistente continua a calcolare `z` con cross-attention, self-attention e MLP;
3. `context = z - direct` isola la correzione appresa; il suo scalare `context_scale` è appreso
   e parte dalla media per profondità `1/(max_levels+1)`, non da una costante calibrata;
4. l'uscita è
   `content_scale * LN((direct + context_scale*context) - mean_slot) + centered_anchor`, con
   `content_scale` appreso e inizializzato a uno.

La LayerNorm ora agisce **prima** dell'ancora: contenuto e posizione hanno RMS circa uno a init
e nessuno dei due deve essere recuperato come residuo del 3%. Il test azzera completamente
`cross.out_proj` e richiede ancora differenza relativa fra anomalie in posizioni diverse >0,1:
è la verifica fail-closed che il coefficiente locale non dipenda dal guadagno favorevole di
quella proiezione. Sul caso deterministico da 256 slot il nuovo output ha differenza relativa
0,266 e rapporto std-tra-slot/std-totale 0,982.

Il highway aggiunge un prodotto valori `(B,256,M)@(B,M,D)` per livello. Non replica la matrice
per le teste e viene ricomputato con activation checkpointing, ma aumenta il calcolo
dell'encoder; questo va misurato nello smoke A40. Contratto, 256 token, `pos_dtype: input`,
tied/untied, sdepth, noise_gate, `TSLM_V3_DEPTH` e `token_diversity` sono invariati. I due nuovi
scalari entrano correttamente nello slim checkpoint. Un checkpoint `cfd5256` non è caricabile
strict nel nuovo encoder perché non contiene `content_scale/context_scale`: è intenzionale,
non va usato per riprendere la nuova architettura.

### Probe offline CPU

`position_decodability(encoder, n=...)` genera coppie indipendenti rumore/anomalia, con
lunghezze 4096/8192/32768, posizione casuale e tre tipi (spike, level shift, variance change).
Il probe non confronta la coppia: ogni serie viene codificata da sola. Usa come feature il
massimo per canale sui token e l'RMS per slot, poi risolve ridge chiusa sul train e misura un
held-out fisso (ID multipli di quattro). Non standardizza le feature sul segnale: la scala
assoluta del contenuto resta parte della decodificabilità. Riporta MAE posizione, accuracy di
presenza bilanciata e MAE della magnitudo normalizzata.

Numeri deterministici del test principale (D=256, 256 token, `n=288`, ridge 3):

| encoder a init | MAE posizione | rapporto/v1 | presence accuracy | MAE magnitudo |
|---|---:|---:|---:|---:|
| v1 | 0,07750 | 1,000 | 0,9375 | 0,13815 |
| v3 `cfd5256` | 0,11236 | 1,450 | 0,8194 | 0,17543 |
| v3 nuovo | 0,10542 | 1,360 | 0,8472 | 0,19003 |

Per tipo, la MAE posizione del nuovo v3 è 0,1523 spike, 0,0762 level shift e 0,0877 variance
change (v1: 0,1016/0,0575/0,0734). Il gate di regressione è volutamente 1,40×, più severo del
requisito 1,50×: `cfd5256` fallisce, il nuovo encoder passa. La magnitudo nuova è leggermente
peggiore del vecchio v3 pur restando sotto 0,20; è un rischio reale da seguire, non un risultato
da nascondere.

Dopo quattro step sintetici del test compatto (D=64, max_levels=2, `n=72`) il v3 ha MAE
posizione 0,2660 contro 0,1924 di v1, rapporto 1,382; presence 0,750 contro 0,694 e magnitudo
0,2009 contro 0,1912. Quindi il criterio ≤1,5× sopravvive ai primi aggiornamenti, anche se il
piccolo probe ha varianza maggiore del test principale.

Per lo smoke sul checkpoint reale il probe ricostruisce v1 e v3 a D=512/topm=1024 e usa
`n=36` per limitare il costo CPU. A init locale il risultato è v1 0,16444 contro v3 0,14706
(rapporto 0,894), presence 0,944/0,778 e magnitudo 0,190/0,149. Lo smoke salva tutti i numeri
nel JSON e fallisce se il checkpoint supera 1,5× la MAE posizione di v1, scende sotto 0,55 di
presence o supera 0,25 di MAE magnitudo.

### Rischi residui

- È un probe sintetico lineare, non sostituisce localize/IoU di Qwen; serve come gate economico
  contro una rappresentazione difficile da leggere, non come stima della metrica finale.
- Lo split D=512 dello smoke è piccolo (`n=36`) e quindi meno preciso del gate CPU principale.
- Il highway costa un secondo value read per livello; memoria e tempo end-to-end A40 restano da
  confermare con `node_smoke_v3.sh`.
- La magnitudo D=256 peggiora da 0,175 a 0,190 rispetto a `cfd5256`; la run GPU deve verificare
  che il miglioramento di localizzazione non compri un peggioramento di `measure_magnitude`.
- Il probe dopo quattro step è volutamente compatto e misura stabilità precoce, non convergenza.

## 11. 07/09 — v3 “residuo su v1”: il punto di partenza non è più un'ipotesi

### Decisione architetturale

I due tentativi latent-only non hanno chiuso il problema strutturale. Il commit `cfd5256`
partiva a MAE posizione 0,11236 contro 0,07750 di v1 (1,45×) e presenza 0,819 contro 0,938;
il content highway di `0d2484e` portava la MAE a 0,10542 (1,36×) e la presenza a 0,847, ma
lasciava ancora a Qwen il compito di imparare una rappresentazione più difficile di va1. La
run `cfd5256` dopo un'epoca conferma che non era solo un difetto del probe: su 80 esempi val
produceva finestre quasi costanti, primary 0,053 e localize IoU 0,0, mentre va1 alla stessa
epoca era circa 0,17 primary. Una terza run da circa 23 ore sullo stesso punto di partenza non
era giustificata.

La nuova invariante è più forte di una buona inizializzazione statistica:

`out(x, context_scale=0) = WaveletTSEncoder(x)` **bit per bit**.

`RecurrentWaveletTSEncoder` ora eredita direttamente `WaveletTSEncoder`. La sequenza base è
calcolata eseguendo il bytecode del suo `_encode_channel`, con gli stessi parametri e nello
stesso ordine: normalizzazione robusta, piramide, banda grossolana, top-k globale, clamp,
Fourier nel dtype di input, `value_proj`, `band_embed` e `pad_token`. Non esiste una seconda
implementazione della selezione v1 da mantenere sincronizzata. `wavelet_v3.py`, dopo aver
costruito il modello v1, copia nel nuovo encoder proprio lo state dict di quell'encoder: anche
nel percorso Qwen reale il token base è ciò che va1 avrebbe passato allo stesso proiettore.

Per riusare la piramide senza modificare `kaggle/patch/wavelet_encoder.py`, la chiamata privata
v1 viene eseguita con una copia locale dei suoi globals nella quale la sola funzione DWT è un
wrapper di registrazione. Il modulo v1 globale non viene monkey-patchato; forward concorrenti
non condividono stato. Il wrapper esegue la DWT una volta e restituisce gli stessi tensori sia
alla selezione base sia allo schedule ricorrente. Un test strumentato conta esattamente una
chiamata Haar per canale.

Il loop non possiede più latenti di uscita. Le query sono `token_v1 + step_embed[t]`; la prior
di località è centrata sulla posizione normalizzata realmente selezionata per quel token. La
banda e la posizione della selezione globale vengono ricostruite dagli stessi dettagli; per
gli slot reali coincidono con v1, mentre gli eventuali pad hanno banda sentinella `-1`, mask di
validità falsa e una posizione deterministica usata solo dalla query. Il residuo parte da zero,
legge A_L e poi D_L…D_1 con cross-attention, self-attention e MLP, e viene normalizzato da una
LayerNorm propria. Il confine è soltanto:

`out = token_v1 + context_scale * context_norm(residuo)`.

Nessuna LayerNorm o proiezione tocca il percorso v1. `context_scale_init: 0.0` è il default;
`0.1` resta disponibile come ablation esplicita, non come default nascosto. In eval/no-grad a
scala esattamente zero il loop viene saltato perché non può cambiare l'uscita; in training viene
sempre costruito, così il primo backward muove `context_scale` e dai backward successivi il
gradiente arriva ai blocchi.

### Misure CPU riproducibili

La suite aggiornata termina `23/23 passed` in 23,5 s sulla macchina locale. L'uguaglianza usa
`torch.equal`, non una tolleranza, e passa in fp32 e bf16 su L=7, L=4096 e sul caso richiesto
16×131072. Il massimo errore osservato è 0,0 in tutti i casi. Passano inoltre tied/untied,
`TSLM_V3_DEPTH`, sdepth, noise_gate, checkpoint slim e round-trip, activation checkpointing,
non-finiti, bf16, diversità e assenza di assunzioni CUDA/ROCm.

Il ridge probe deterministico compatto (D=64, N=256, n=72) dà valori identici a init:

| encoder | MAE posizione | presence | MAE magnitudo |
|---|---:|---:|---:|
| v1 | 0,18019845 | 0,94444442 | 0,15701361 |
| v3 residuo, scala 0 | 0,18019845 | 0,94444442 | 0,15701361 |

Anche le tre MAE per tipo coincidono: spike 0,16608138, level shift 0,14354114, variance
change 0,23097281. L'identità del probe è una conseguenza verificata dell'identità dei token,
non un nuovo compromesso di soglia.

Dopo quattro passi sintetici a lr 1e-4, i gradienti di `context_scale` sono 0,1682, 0,1670,
0,1998 e 0,2358; lo scalare passa da 0 a -0,00040079. Come previsto dalla derivata del prodotto,
la somma dei gradienti assoluti sui blocchi è zero al primo passo e poi 0,0464, 0,0912 e
0,1343. Il probe post-update dà MAE posizione 0,19249903 contro 0,19240350 del controllo v1,
rapporto 1,00050: molto sotto il gate richiesto 1,1×. Presenza resta 0,69444 in entrambi e la
MAE magnitudo è 0,19130 contro 0,19119.

`TSLM_V3_DEPTH=1` ora significa una sola lettura contestuale, la banda grossolana. Non riduce
né tronca la selezione base: con scala zero tutte le profondità restano esattamente v1; con
scala non nulla il test distingue depth 1 da full e verifica che il base completo sia ancora
presente.

### Checkpoint e smoke

Lo state dict cambia deliberatamente: spariscono `latents`, `ln_out` e `content_scale`, entrano
il percorso v1 completo e `context_norm`. `_load_from_state_dict` riconosce le chiavi dei due
encoder latent-only e produce un errore esplicito anche con `strict=False`; quei checkpoint non
possono essere ripresi accidentalmente.

Lo smoke A40 esegue prima del training un controllo CPU fp32/bf16, tied e untied, su serie
normali e corte e richiede `V3_INIT_EQUALS_V1`. Dopo i quattro passi apre lo slim checkpoint,
richiede le chiavi v1, blocchi, gate, step embedding, context norm e scala, fallisce se la scala
è ancora zero e salva `context_scale_after_4_steps` nel JSON di ciascuna variante. Lo smoke GPU
non è stato eseguito localmente; resta il gate successivo sul nodo.

### Rischio residuo corretto da misurare

L'identità a init elimina il rischio già osservato di partire con meno informazione di va1, ma
non dimostra che il contesto migliori il task. Ora però l'ablazione 1 è interpretabile: confronta
“selezione v1 + contesto multi-scala” con “selezione v1 sola” dallo stesso punto. Il rischio
principale diventa che `context_scale` resti vicino a zero o impari un residuo inutile; gate,
curve di profondità, valore dello scalare e metriche appaiate rendono questo esito direttamente
osservabile senza confonderlo con una tokenizzazione iniziale peggiore.

## 12. 08/09 — rottura misurabile del deadlock dello zero-init

### Diagnosi verificata sul codice e sulla run

Il confine effettivo in `RecurrentWaveletTSEncoder._encode_channel` è
`base_tokens + context_scale * context_norm(z)`. Per qualunque parametro `θ` di blocchi, gate,
step embedding o context norm:

`∂L/∂θ = context_scale · (∂L/∂out) · ∂LN(z)/∂θ`.

Con scala zero questo gradiente è esattamente zero al primo step. Lo scalare vede invece
`∂L/∂context_scale = <∂L/∂out, LN(z)>`; prima che il ramo sia allenato è una correlazione con
feature casuali, non un segnale che possa co-adattare il ramo. I checkpoint della run
`1b0f9fa` (0,0001 a epoca 1, −0,0006 a epoca 3, 0,0004 a epoca 5, gate bias circa 1,98)
confermano che dopo 22.440 step il sistema è rimasto nel vicinato della soluzione nulla.
Quella run è quindi una replica di va1 con rumore run-to-run: il risultato 0,2832 contro
0,2744 non misura il valore del contesto.

### Scelta implementata

Ho scelto **init non nullo più learning rate differenziale**:

- i cinque config `v3{tied,untied,sdepth,smoke_tied,smoke_untied}` usano
  `context_scale_init: 0.1`;
- `WaveletV3TSModel.get_trainable_parameters()` sottrae per identità dal gruppo encoder i soli
  `blocks`, `gate_head`, `step_embed`, `context_norm` e `context_scale`, e li espone come gruppo
  `context`;
- `training.learning_rate.context: 1e-3`, mentre tutti i parametri va1 (`value_proj`,
  `band_embed`, `pad_token`) e il resto del training conservano `default: 1e-4`.

Il trainer già costruiva AdamW leggendo `training.learning_rate[group_name]`, quindi non è
servita alcuna modifica del percorso optimizer. Il test con una base fittizia verifica che i
gruppi non abbiano duplicati, che `context` contenga esattamente i parametri richiesti e che
encoder va1 e projector restino nei gruppi originali. Il 10× serve sia a co-adattare un ramo
nuovo rispetto al percorso va1 già informativo, sia a rendere osservabile in quattro step lo
spostamento minimo richiesto dallo smoke. Non ho scelto softplus/exp: cambierebbe la chiave o
la semantica del parametro nei checkpoint. Non ho scelto un warmup imposto dal forward:
introdurrebbe stato/step nel modello e renderebbe train/eval dipendenti dall'harness.

Al primo backward il gradiente atteso dei blocchi non è più zero ma
`0,1 · (∂L/∂out) · ∂LN(z)/∂θ`. Nel witness deterministico misurato, la sua norma è 0,0180925,
pari a **0,246994×** la norma del gradiente di `value_proj`, molto sopra il gate 1e-3.

### Test CPU e numeri

`py v3-recurrent/patch/test_encoder_v3.py` passa **25/25** sulla macchina locale.
Il nuovo test usa otto token tutti consumati dalla banda grossolana v1 e due impulsi +/− che
si cancellano in quella banda: la posizione precoce/tardiva è leggibile soltanto dai livelli
di dettaglio del ramo contestuale. Un witness ortogonale rende nullo per costruzione il
fortuito gradiente dello scalare casuale e documenta il prodotto parametrico senza una soglia
statistica fragile. Dopo K=16 step AdamW sul gruppo context:

| init scala | max norma grad blocchi | grad scala max | spostamento scala |
|---:|---:|---:|---:|
| 0,0 | 0 | 0 | 0 |
| 0,1 | 0,0180925 | 0,0629449 | **0,0141876** |

Il gate di identità usa ancora `context_scale_init=0.0` esplicito e passa con `torch.equal` in
fp32 e bf16, incluse serie corte e 16×131072. Con l'init di run 0,1, il ridge probe a init dà:

| encoder | MAE posizione | presence | MAE magnitudo |
|---|---:|---:|---:|
| v1 | 0,18019845 | 0,94444442 | 0,15701361 |
| v3, scala 0,1 | 0,17688306 | 0,94444442 | 0,15650079 |

Il rapporto v3/v1 è **0,981603** sulla MAE posizione e **0,996734** sulla MAE magnitudo;
la presenza è invariata. Tutti i gate di decodificabilità sono quindi entro 1,1× v1.

### Smoke e compatibilità

`node_smoke_v3.sh` conserva `V3_INIT_EQUALS_V1`: costruisce un encoder separato con scala zero
forzata, quindi il controllo non è indebolito dal nuovo config. Durante il vero training imposta
`TSLM_V3_GRAD_LOG=1`; `train.py` emette, solo sotto questo env, una riga `[V3Grad]` dopo
l'unscale e prima del clipping con norma dei blocchi, norma di `value_proj`, rapporto, scala e
gradiente della scala. Lo smoke richiede quattro righe, una per step, con norma blocchi >0 e
rapporto blocchi/`value_proj` almeno 1e-3 in ciascuna; richiede inoltre almeno un gradiente non
nullo della scala e `max_t abs(context_scale_t - context_scale_init) >= 1e-4`. Il massimo lungo
la traiettoria misura l'attivazione dell'ottimizzazione senza fallire se aggiornamenti oscillanti
riportano il checkpoint finale vicino a 0,1. Traiettoria, massimo spostamento e minimi delle due
diagnostiche dei blocchi sono salvati nel JSON di ciascuna variante; lo spostamento finale resta
una sola diagnostica. Il witness zero-init già riportato sopra (norma gradiente blocchi 0 e
spostamento scala 0) fallirebbe logicamente sia il primo sia il secondo requisito, quindi il
nuovo gate continua a escludere il deadlock osservato. Lo smoke A40 resta da rieseguire sul nodo.

La parametrizzazione e tutte le chiavi dello state dict sono immutate: `context_scale` resta
un parametro scalare con lo stesso nome. I checkpoint della run `1b0f9fa` continuano quindi a
caricarsi per eval o per recupero dei pesi; il loro valore quasi zero non viene sovrascritto
dall'init del nuovo config. Un resume con lo stato AdamW storico non è il percorso previsto
per la nuova ablazione, perché il nuovo gruppo `context` cambia intenzionalmente la topologia
dell'optimizer; la replica parte da init e protocollo puliti.

## 13. 09/09 — PLAN-CONTEXT-COLLAPSE, parte ingegneristica

### Lettore e conservazione locale

`model.encoder.reader_mode` accetta `full` (default storico) e `cross_only`. Il secondo accumula
soltanto l'uscita gated della cross-attention: self-attention e MLP non vengono eseguite e le loro
componenti diagnostiche sono esattamente zero. Ogni `_LoopBlock` possiede inoltre gli scalari
appresi `self_attn_scale` e `mlp_scale`; in untied sono quindi distinti per livello. I default
`self_attn_scale_init=1.0` e `mlp_scale_init=1.0` preservano il calcolo `full` precedente.

L'audit aveva mostrato che anche la lettura cross contiene un grande modo condiviso fra token.
Ho scelto di sottrarre quel modo nel controllo `cross_only`. Perché `full` a scale piccole sia una
reintroduzione continua delle trasformazioni, il contesto passato alla LN usa
`z_local = z - c mean_token(z)`, con `c=1` in `cross_only` e
`c=1-min(1,max(|self_scale_init|,|mlp_scale_init|))` in `full`. Quindi `full/1.0` ha `c=0` ed è
invariato; `full/0.1` ha `c=0.9`. `c` è una scelta costruttiva fissata dal config iniziale, non un
parametro né un ulteriore percorso di gradiente. Le scale apprese moltiplicano soltanto le uscite
self-attention/MLP come richiesto.

`last_context_stats` contiene una lista per iterazione e un record finale. Le statistiche sono
detached e campionate deterministicamente (primi 4 elementi, 64 token e 64 feature): RMS di z
prima della LN, RMS di h, `RMS(h-mean_token(h))/RMS(h)`, coseno medio off-diagonal, differenza fra
serie nel batch, differenza per traslazione dei token e RMS cross/self/MLP. Il probe esplicito
`context_input_dependence()` esegue, fuori dal normale forward, serie diversa e traslazione vera
dell'input. `TSLM_V3_GATE_LOG_EVERY` emette la riga compatta `[V3Ctx]`; nel trainer
`TSLM_V3_GRAD_LOG=1` più `TSLM_V3_CTX_LOG_EVERY=N` emette gradienti e stats ogni N step. Default 0:
nessun nuovo costo/log nel training normale.

Numeri deterministici CPU a init (D=64, K=256, tre livelli, seed 314/315):

| lettore | frac. variabile h | coseno token | RMS z | dip. serie diversa | dip. traslazione input |
|---|---:|---:|---:|---:|---:|
| `cross_only` | **0,931603** | 0,118358 | 0,063540 | 0,677816 | 0,831370 |
| `full`, scale 1,0 | 0,055383 | 0,996884 | 1,140591 | 0,040345 | 0,048574 |
| `full`, scale 0,1 | **0,708851** | 0,489558 | 0,082526 | 0,522685 | 0,633636 |

Il caso 0,1 dista 0,222752 dal controllo `cross_only` sulla frazione variabile, contro 0,876219
del `full` storico. Il test impone `cross_only >= 0,3`, entrambe le dipendenze >0,1 e distanza
`full-small/cross_only <0,25`.

Il witness held-out usa otto token tutti occupati dalla banda grossolana. Impulsi adiacenti +5/-5
si cancellano nella banda v1; la loro posizione early/late è presente nei dettagli intermedi letti
dal contesto. Dopo K=32 step su lettore+probe lineare, la BCE held-out è 0,000657 contro 0,693389
del probe sulla sola base (accuracy contesto 1,0); forzando off lo stesso modello dà BCE 0,694933.
Nel controllo negativo a bassa frequenza, già leggibile dalla base, accuracy on=off=1,0.

### Uscita, optimizer e checkpoint

Con `token_gate: true`, una sola `Linear(1024,1)` condivisa calcola
`g_i=2 sigmoid(w^T[LN_noaffine(base_i),h_i]+b)`. Peso e bias sono zero: g=1 e l'uscita coincide
bit-per-bit con la v3 senza gate. `_merge_context(..., token_gate_override=...)` rende verificabile
che g=0 su uno slot restituisca esattamente il token base. Media, min/max, p10/p50/p90 e frazioni
sature sotto 0,05/sopra 1,95 sono nel record finale.

I gruppi sono ora `context` (lettore), `context_scale` (solo lo scalare) e, se presente,
`context_token_gate`. Se `training.learning_rate.context_scale` manca, il trainer usa il LR di
`context`, quindi i config storici non cambiano. I pilot usano rispettivamente 1e-3, 1e-5 e 1e-4.
Ogni checkpoint nuovo salva nomi e cardinalità dei gruppi. Un resume con optimizer privo di questi
metadati o con topologia diversa fallisce indicando `TSLM_RESUME_WEIGHTS_ONLY=1`; perciò gli stati
Adam dei checkpoint `1b0f9fa`/`b2d6808` non possono essere accettati silenziosamente. Come soli
pesi, checkpoint residual-on-v1 precedenti migrano le nuove scale a 1 e il token gate a identità.

### Smoke e pilot val-only

Lo smoke usa `cross_only`, scale 0,1, token gate e LR scala 1e-5. Conserva i gate su gradienti
finiti del lettore (`blocks_grad_norm>0`, `blocks_relative>=1e-3`) e ora campiona direttamente lo
spostamento dei pesi dei blocchi fra init e ciascuno dei quattro update: l'ultimo deve essere >0.
Non esiste più una soglia sul movimento di `context_scale`. Il JSON include `context_stats` e
`context_input_dependence`; fallisce se la frazione variabile finale è <1e-3.

`node_pilot_v3.sh <pilot_name> <config.yaml> <n_epochs>` crea
`results/anomaly_xl_precise/pilot_<name>_<UTC timestamp>`, senza cancellare altre run e senza
invocare `node_train_v3.sh`/worker. Usa `TSLM_MAX_TRAIN_STEPS=n*4488`, arresto esterno al confine
d'epoca, `TSLM_SCHED_TOTAL_STEPS=22440` (warmup 673), checkpoint esatti 500/2000, seed 42,
`TSLM_DATA_ORDER_SEED=42` e `TSLM_SUBSET_SEED=42`. Non imposta `TSLM_REQUIRE_FULL_COVERAGE`, ma
verifica dal log una e una sola coverage 4488/0/4488 per epoca.

Al primo pilot il trainer salva in `~/pilots/init_common.pt` tutti i parametri allenabili comuni
(encoder v1, proiettore, LoRA e lettore); i successivi fanno copia strict per nome/shape. Le sole
chiavi aggiuntive ammesse sono quelle del token gate e devono essere ancora zero/identità. Il
primo pilot deve quindi essere uno senza token gate. L'ordine del DataLoader usa un generator
dedicato e non dipende dal consumo RNG dei moduli opzionali.

A step 500, 2000 e a ogni fine epoca, un worker parallelo valuta solo validation: generazione
on/off sui medesimi 80 esempi (`max_samples=800`, rapporto 0,1) con primary per categoria,
teacher forcing on/off tramite `audit/20260908-ctx01/forward_audit.py` e diagnostiche del contesto.
A fine epoca aggiunge generation on sull'intera val da 520 esempi. Manifest, config effettivi e
hash SHA-256 restano nella run dir. Il test split non compare in alcun comando del pilot.

Ordine consigliato per stanotte, circa 5 ore per pilot A40 (una epoca ciascuno):

```bash
bash ~/node_pilot_v3.sh cross configs/wavelet_precise_v3pilot_cross.yaml 1
bash ~/node_pilot_v3.sh fullsmall configs/wavelet_precise_v3pilot_fullsmall.yaml 1
bash ~/node_pilot_v3.sh gate configs/wavelet_precise_v3pilot_gate.yaml 1
```

Eseguire prima `bash ~/node_smoke_v3.sh`. Non rimuovere `~/pilots/init_common.pt` tra i tre pilot;
rimuoverlo soltanto per iniziare deliberatamente una nuova famiglia di inizializzazione comune.

## 14. 09/09 — decomposizione `full | local | global` del contesto

### Motivazione e semantica

Il pilot `pilot_cross_20260909T014929Z` ha mostrato che il centraggio unico fra tutti i token
non impedisce il collasso in due cluster: a step 2000 i grossolani hanno coseno 0,99 e frazione
variabile 0,08, mentre i dettagli hanno coseno 0,75 e frazione variabile 0,50. Il lettore può
quindi usare un modo comune per banda, ridondante con `band_embed`, senza conservare informazione
locale per token.

`RecurrentWaveletTSEncoder` accetta ora `context_mode: full | local | global`, con default
`full`. La decomposizione avviene dopo il centraggio storico governato da `reader_mode` e prima
della `context_norm`:

- `full` passa direttamente lo stesso tensore usato prima della modifica, quindi il default non
  aggiunge operazioni aritmetiche al forward e resta bit-per-bit invariato;
- `local` sottrae, per ogni serie e per ogni valore esatto di `base_band`, la media sugli slot
  validi della banda;
- `global` ribroadcasta sugli stessi slot soltanto quella media.

Ho scelto la **banda esatta**, anziché il solo split grossolana/dettaglio, perché è la condizione
più forte contro il canale spurio ed è allineata alla granularità di `band_embed`. Il padding
non contribuisce ad alcuna media. Per conservare l'identità algebrica sull'intero tensore, la
sua componente resta nel ramo `local`; gli invarianti semantici sono definiti sui token validi.
Su questi, la media per banda di `local` è zero, la varianza intrabanda di `global` è zero e
`full = local + global` (la stessa uguaglianza vale sull'intero tensore con la convenzione sul
padding). La LayerNorm resta successiva e distinta in tutte e tre le modalità. `reader_mode`,
scale self-attention/MLP, level gate, token gate, `context_scale`, inizializzazione, state dict e
gruppi optimizer non cambiano. Con `context_scale=0`, tutte le modalità restano v3 ≡ v1 bit per
bit.

### Diagnostiche e witness

`_context_snapshot` non usa più `[:4,:64,:64]`: RMS, frazione variabile e coseno sono calcolati
su tutti i token validi e tutte le dimensioni. Le chiavi canoniche sono
`token_variable_fraction_{total,coarse,detail}`,
`mean_token_cosine_{total,coarse,detail}` e
`input_series_relative_{total,coarse,detail}`; gli alias storici senza suffisso indicano il
totale. Le stesse chiavi sono presenti in `last_context_stats`, nel JSON del forward audit e
dello smoke, e nella riga testuale `[V3Ctx]`. Anche le statistiche del token gate usano tutti gli
slot validi.

I test costruttivi verificano separatamente che: (1) `local` abbia media intrabanda nulla,
(2) `global` sia costante entro banda, (3) le componenti ricostruiscano `full`, (4) un segnale
zero-sum di posizione sia decodificabile da `local` ma non da `global`, e (5) presenza e
magnitudo condivise siano recuperabili da `global` e assenti da `local`. Un ulteriore caso con
il solo elemento `[slot=255, feature=511]` non nullo impedisce regressioni al vecchio
campionamento dei primi 64 slot/feature.

### Config pilot e consegna

Sono aggiunti `wavelet_precise_v3pilot_crosslocal.yaml` e
`wavelet_precise_v3pilot_crossglobal.yaml`. Entrambi mantengono dal controllo `pilot_cross`
`reader_mode: cross_only`, `context_scale_init: 0.1`, scale self/MLP 0,1, token gate disattivo e
LR `default/context/context_scale = 1e-4/1e-3/1e-5`; cambia soltanto `context_mode`. Non serve
modificare `ship_v3.sh`, che spedisce già `v3-recurrent/configs/*.yaml`.

Verifiche locali finali: `py v3-recurrent/patch/test_encoder_v3.py` passa 33/33; il logging
`[V3Ctx]` è stato esercitato con tutte le nuove chiavi; `py_compile` sui file Python modificati,
`bash -n` su tutti gli script node v3 e su `ops/ship_v3.sh`, e `git diff --check` passano.
