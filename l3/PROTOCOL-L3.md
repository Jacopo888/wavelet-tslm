# Protocollo L3 — GRPO multimodale va1

Stato: protocollo smoke e run lunga, congelato prima della run il 10/09/2026.
Il gate smoke precede la catena lunga e qualunque accesso al test pinnato.

## Scopo dello smoke

Lo smoke verifica che un unico `WaveletTSModel` Qwen3.5-4B, inizializzato dal
checkpoint va1 `epoch_5_end.pt`, possa campionare e ricevere un update GRPO in puro
HF/PyTorch mantenendo i coefficienti wavelet nel percorso embedding. I rollout RL
usano esclusivamente il train split. La validation compare soltanto nel gate finale
probe80; il test non è un input dello smoke.

## Decisioni di disegno

### Reference e old policy

La reference è una copia CPU, clonata subito dopo il caricamento di va1, dei 32
tensori il cui nome contiene `lora_`. `LoraReference` controlla nomi, shape e dtype;
il suo context manager clona temporaneamente lo stato LoRA corrente, copia la
reference iniziale sul device, esegue il forward senza gradiente e ripristina lo
stato corrente. Il round-trip è verificato con `torch.equal`. Non viene mai
disattivata la LoRA: farlo cambierebbe la reference da va1 al backbone base. Non si
mantiene una seconda copia del 4B in GPU, che non entrerebbe nel budget A40-24Q.

La old policy è invece lo stato che ha prodotto ciascun rollout. La generazione
avviene con il modello in `eval()`: così il decoder usa la KV-cache anche quando
`TSLM_GRAD_CKPT=1` ed elimina il dropout. Subito dopo la generazione si ripristina
`_gradient_checkpoint_train_mode`; prima di qualsiasi `optimizer.step`, si calcolano
in `no_grad` i log-prob teacher-forced dei token campionati e se ne conserva una
copia. Anche i forward reference e corrente restano nella medesima modalità train,
con tutti i moduli `Dropout` posti in eval. Policy e old policy sono quindi oggetti
concettualmente distinti anche se, con una sola epoca interna, hanno gli stessi pesi.

I percorsi senza gradiente e con gradiente possono selezionare kernel SDPA diversi
sullo stesso input e differire agli ultimi bit. Il gate fatale richiede perciò, per
ogni completamento, `max |logpi - logpi_old| <= 1e-3`; ogni step registra il massimo
come `max_abs_log_ratio`. `ratio_all_exact_one` resta un flag diagnostico riportato,
non un gate. Conservare la reference come old policy è scartato perché dal secondo
step confonderebbe il vincolo KL verso va1 con il denominatore PPO.

### Loss

Per il completamento `i` e il token di risposta `t`:

```
r_it = exp(logpi_it - logpi_old_it)
L_pg_i = -mean_t min(r_it A_i, clip(r_it, 1-epsilon, 1+epsilon) A_i)
d_it = logpi_ref_it - logpi_it
KL_k3_i = mean_t(exp(d_it) - d_it - 1)
L = mean_i(L_pg_i + beta KL_k3_i)
```

I reward ufficiali dei `G=4` completamenti sono normalizzati nel gruppo con
deviazione standard di popolazione ed epsilon. Se la varianza è zero, tutti i
vantaggi sono esattamente zero e il gruppo è contato. Prima si media sui token di
ogni risposta, poi sui completamenti: una risposta lunga non pesa più di una corta.
La KL k3 riguarda soltanto le posizioni causali che predicono i token di risposta;
prompt e placeholder non entrano nella media. Un forward per completamento mantiene
la dimensione batch a uno e i gradienti sono accumulati dividendo ciascuna loss per
`G`.

Lo smoke usa AdamW, nessuno scheduler, clip globale 1.0 e LR LoRA `1e-5`. È un
decimo del LR SFT (`1e-4`): su appena 20 update riduce il rischio che reward di
gruppi piccoli e altamente discreti distruggano la policy, ma resta abbastanza alto
da rendere misurabile almeno un gradiente e una deviazione dalla reference. Il LR
non viene adattato in base agli esiti dello smoke.

### Memoria

Il modello base bf16 occupa circa 8 GB; i 1.835.008 parametri LoRA master fp32,
gradienti e due momenti Adam restano nell'ordine delle decine di MB. La reference
LoRA vive su CPU (circa 7 MiB fp32). Con la cella massima ci sono 4096 token
percettivi (16×256), il testo e al massimo 500 token di risposta. SDPA evita la
matrice di attenzione quadratica materializzata e gradient checkpointing evita di
conservare tutte le attivazioni di layer. I quattro completamenti sono generati e
valutati in sequenza; non esiste una dimensione GPU `G`.

Questa è una stima, non il gate. Lo stadio c0 azzera le statistiche CUDA e misura
`torch.cuda.max_memory_allocated()` sull'indice train 3804, verificato dai metadata
come `localize_all_channels`, 16 canali, lunghezza 131072. Il gate è strettamente
minore di 22 GiB (`22528 MiB`) e richiede reward e KL finiti. Sono scartati una
seconda reference 4B, rollout paralleli, attention eager, vLLM e allocator VMM.

### Reward e parsing

Il testo decodificato passa prima da `dataset.extract_answer`. È parsabile soltanto
se ha il prefisso della categoria del sample e se `parse_structured_pred` ufficiale
trova l'oggetto con le chiavi richieste per quella categoria. In tal caso il reward
è esclusivamente `dataset.evaluate_answer(...)[dataset.primary_metric]`. Categoria
sbagliata, JSON invalido o chiavi mancanti producono reward primario zero e
incrementano `unparsable`; ogni rollout finisce quindi in uno e un solo contatore.

`format_penalty` vale zero nello smoke. Se verrà abilitata, è registrata come
addendo separato negativo per gli output non parsabili e può entrare soltanto nel
reward di ottimizzazione; non modifica mai `reward_mean`, le metriche ufficiali o
lo scorer. Non si usa alcuna penalità di lunghezza.

### Prompt fissi

Gli indici sono stati scelti una volta dai metadata del train parquet pinnato
(4488 righe, SHA-256 `b8571d886772a55613c2e8851a774e6a561e19c8272227f5abb70e2d76e70b7d`).
Con `random.Random(20260910)`, per ciascuna categoria ordinata lessicograficamente,
si è estratto un elemento da ciascun quartile della sua lista di indici; gli estratti
sono poi intercalati per quartile, così nessuna categoria occupa quattro update
consecutivi. Il risultato è salvato, non rigenerato, nel YAML:

```
348, 4030, 46, 3484, 617,
1251, 4162, 925, 3658, 1654,
2000, 4277, 1768, 3753, 2382,
2897, 4353, 2599, 3930, 3174
```

La copertura è quattro prompt per ciascuna delle cinque categorie. Per i task
monocanale compaiono 16384, 32768, 65536 e 131072; per 16 canali e lead/lag
compaiono 32768 e 131072. I cinque indici di init, uno per categoria, sono
`[46, 348, 617, 3484, 4030]`. Il runner controlla nuovamente ID, categoria,
lunghezza e numero di canali contro le aspettative nel config prima di usare una
riga. Qualunque drift del dataset fallisce chiuso.

`train.py --eval-only` non offre il train split. Per non estenderlo e non passare
dalla validation, lo stadio b usa due chiamate greedy allo stesso percorso
`prepare_batch(training=False) -> model.generate` sul modello caricato con gli
stessi `load_model`, `configure_training_mode` e `load_checkpoint` di va1. Prima si
genera il baseline nel percorso eval bf16; poi i soli master LoRA sono promossi a
fp32, il checkpoint viene ricaricato senza perdita e la seconda generazione passa
dal wrapper rollout L3. Token e testo devono essere identici. Nello stesso stadio
policy/reference log-prob devono avere differenza massima non superiore a `1e-3`.
`logprob_equal_reference` e `ratio_all_exact_one` sono comunque riportati come flag
diagnostici, senza essere fatali.

### Stato salvato

Solo i 32 tensori LoRA ricevono gradienti e costituiscono il checkpoint RL primario.
Encoder e proiettore sono congelati. Poiché `train.py --eval-only` accetta un solo
checkpoint, il runner produce anche un artefatto derivato slim composto con le
stesse 42 chiavi adattabili di va1 (32 LoRA aggiornate più 6 encoder e 4 projector
immutate), mai i pesi del backbone 4B. Senza le dieci chiavi immutate una eval da
inizializzazione base perderebbe il percettore va1. Il manifest e il summary
distinguono `checkpoint` LoRA-only da `eval_composed_checkpoint`; il secondo non è
un checkpoint RL selezionabile.

### Fuori perimetro

Non sono coperti il ciclo agentico/tool del punto 4, ricorrenza v3, il Livello 2
testuale, lo sblocco di encoder/proiettore o più epoche PPO interne. Queste
estensioni richiedono un nuovo protocollo e un nuovo gate.

## Gate smoke a–d

Lo script `node_l3_smoke.sh` crea una directory nuova e marker fail-closed.

1. **a — CPU:** tutta `l3/tests` passa nella venv L1.
2. **b — identità:** sui cinque prompt train fissi, greedy va1 e wrapper L3 hanno
   token identici; la differenza massima policy/reference è al più `1e-3`, mentre
   l'uguaglianza esatta resta riportata.
3. **c0 — memoria:** un update, quattro rollout sequenziali, sulla cella
   16×131072; picco sotto 22528 MiB, valori finiti e
   `max_abs_log_ratio <= 1e-3`.
4. **c — update:** 20 prompt train fissi, 80 rollout contati, nessun NaN/Inf,
   `max_abs_log_ratio <= 1e-3` a ogni step, almeno un gradiente strettamente positivo
   e tutti i contatori obbligatori presenti.
5. **d — non distruzione:** eval greedy del checkpoint composto sui probe80 fissi
   della validation; primary almeno 0.10. Solo questo comando usa
   `TSLM_EVAL_STRICT=0`, perché il probe copre 80 righe delle 520 della validation;
   in cambio il gate richiede `split == "val"`, `samples_evaluated == 80`,
   `eval_errors == 0` e conta esattamente 80 righe nel JSONL. La modalità strict
   resta attiva per le validation intere. È il solo uso della validation nello
   smoke e non seleziona un checkpoint.

`SMOKE_OK` è scritto solo dopo tutti e quattro i gate; altrimenti resta `FAILED` con
stadio e motivo. Un path di run esistente causa errore e non viene ripulito o
riutilizzato automaticamente.

## Decisioni run lunga

Queste regole sono dichiarate prima della run e non vengono adattate dopo aver
osservato validation o test.

### Contratto di ottimizzazione

La temperatura resta fissa a `0.7`: per decisione di Jacopo non si esegue il pilot
di temperatura. Restano fissi `top_p=0.95`, LR LoRA `1e-5`, clip globale `1.0`,
`format_penalty=0` e KL k3 con beta `0.01`. Il k3 è calcolato in fp32 come
`expm1(delta) - delta`, deve essere finito e non può scendere sotto `-1e-7`; i
residui negativi entro tolleranza sono portati a zero.

Ogni optimizer step contiene due prompt elaborati uno dopo l'altro. Per ogni prompt
si generano otto completamenti in sequenza, si normalizzano i vantaggi soltanto
all'interno del suo gruppo G8 e si media prima sui token, poi sui completamenti e
infine sui due prompt. Le feature del primo prompt vengono eliminate prima di
preparare il secondo; i prompt non sono mai una dimensione batch. Si effettua un
solo `optimizer.step` per i due gruppi. Le norme dei gradienti policy e KL sono
calcolate con due backward separati sullo stesso forward, poi i gradienti sono combinati come
`policy + 0.01 * KL` e sottoposti al clip globale.

`rl.weight_decay=0` è una deviazione intenzionale dal decay SFT 0.01: se vantaggi e
KL sono nulli, va1 deve restare un punto fisso. Sono allenabili esclusivamente i 32
tensori LoRA; encoder e proiettore restano congelati. `positive_grad_steps` rimane,
se presente negli artefatti smoke, una diagnostica e non è un gate della run lunga.

### Pool train-only

Il pool è filtrato una volta, prima del training, sul solo train pinnato. Con seed
`20260911` gli indici sono mescolati entro strati `(categoria, lunghezza)` e visitati
round-robin. Ogni candidato riceve G8 a T 0.7; è ammissibile soltanto se
`reward_std > 0`. Il tetto è 200 candidati, cioè 1600 rollout. Dal risultato si
scelgono esattamente otto prompt per ciascuna delle cinque categorie, ciclando le
lunghezze disponibili per bilanciarle. Se una categoria non raggiunge otto prompt,
L1 fallisce. `pool.json`, seed, lista ordinata, SHA del contenuto e SHA del file sono
congelati nei manifest. Durante il training non esiste rejection sampling online.

L'ordine di training è una singola permutazione del pool prodotta dal seed della
run e ripetuta ciclicamente. La principale usa `20260911`; le repliche usano
`20260912` e `20260913`. Il cursore conta prompt consumati, non step.

### Checkpoint e ripresa

Si salva allo step zero e ogni 25 step, oltre all'eventuale confine richiesto dal
driver. Ogni punto contiene un checkpoint LoRA-only a 32 tensori e il composto slim
a 42 chiavi. Entrambi sono scritti con rename atomico e registrati con SHA-256 nel
manifest; il LoRA-only contiene inoltre stato AdamW, step, cursore del pool, contatori
cumulativi e RNG Python, torch CPU e CUDA. Il resume seleziona l'ultimo record di cui
entrambi i file e gli SHA sono validi, tronca i JSONL dopo quello step e riparte dal
cursore salvato: rollout parziali non vengono né duplicati né saltati.

`rollouts.jsonl` registra per ogni completamento step, ordinale del prompt, indice
train, ID, categoria, reward primario e di ottimizzazione, parsabilità, token di
risposta e SHA-256 del testo. `steps.jsonl` registra i due prompt, gruppi informativi,
step con segnale policy, norme policy/KL, KL, memoria e tempo. Le due norme sono
ottenute con due `backward` immediati sullo stesso forward (`retain_graph` soltanto
per il primo): è il percorso compatibile con il gradient checkpointing re-entrant;
il costo è accettato per rendere il segnale verificabile senza un secondo modello.

### Fasi, selezione e stop

- L0 esegue un update P2/G8 sui due prompt peggiori 16×131072, indici 3804 e 3930;
  richiede 16 rollout contati, finitezza e picco strettamente sotto 22528 MiB.
- L1 costruisce il pool congelato con il gate appena descritto.
- L2 esegue al massimo 400 step della run principale. Probe80 greedy ogni 25 step è
  solo monitor con `TSLM_EVAL_STRICT=0`; val520 greedy completa e strict avviene a
  100, 200, 300 e 400.
- L3 valuta una sola volta sul test 800 il candidato L2 già congelato e lo confronta
  col JSONL va1 archiviato. Il test non seleziona né modifica alcun checkpoint.
- L4 ripete la stessa procedura con i due seed dichiarati, senza test e senza ritocchi.

La selezione L2 e delle repliche usa il massimo primary val520. Se più checkpoint
sono entro 0.002 dal massimo osservato, vince il primo temporalmente. Il training si
ferma a 400 o dopo due val520 consecutive che non migliorano il massimo precedente
di almeno 0.005. Il riferimento descrittivo va1 ep5 sulla validation intera è
0.3105, ma non cambia la regola.

Sono hard stop: NaN/Inf, rollout persi, SHA o insieme delle chiavi LoRA errati,
picco `>=22528 MiB`, oppure per due finestre non sovrapposte consecutive da 50 gruppi
parsabilità sotto 95% o gruppi informativi sotto 50%. Ogni fase ha directory nuova,
log, `summary.json` e marker esclusivo `OK`/`FAILED`; la successiva richiede l'`OK`.
Il driver richiede una A40 libera, venv L1, processo singolo, bf16 AMP, SDPA,
gradient checkpointing e allocator nativo.

### Budget e valutazioni

Il budget dichiarato è 50–60 ore A40, circa 45–52 USD a 0.86 USD/ora: L0 circa 7
minuti, L1 3.2–3.9 ore, ogni seed 13–16 ore di training più circa 2.1 ore di
validation, e il test finale circa 0.6 ore. Il driver stampa ore trascorse e stima
residua a ogni fase.

Prima e dopo ogni probe80, val520 o test si calcola lo SHA del checkpoint composto.
Il JSON sigillato dell'eval contiene entrambi gli SHA e il confronto positivo con
il record del manifest. Il confronto finale usa `l3/paired_eval_l3.py`, non modifica
`v3-recurrent/audit/paired_eval.py`, impone copertura 800/800 (520/520 su validation),
identità ordinate `(id, category, ground_truth)`, valori finiti e bootstrap appaiato
10.000 con seed dichiarato, complessivo e per categoria.

## Confronto del punto 5

Dopo lo smoke, ogni candidato RL lungo viene selezionato soltanto sulla validation
con una regola dichiarata prima della run. Il confronto finale usa una sola volta lo
stesso test pinnato di 800 esempi per:

- va1 SFT `epoch_5_end.pt`;
- va1 + RL, inizializzato dallo stesso `epoch_5_end.pt`.

Le due condizioni usano stesso `eval_max_new_tokens`, decodifica greedy e stack
bf16/SDPA pinnato. Il confronto è appaiato per `(id, category, ground_truth)` e usa
10.000 bootstrap con seed dichiarato tramite `l3/paired_eval_l3.py`, lasciando
`v3-recurrent/audit/paired_eval.py` intatto. Si riportano primary medio,
differenza appaiata con CI 95%, risultati per categoria e conteggio esatto 800/800.

Ogni report dichiara budget RL in optimizer step, prompt, rollout totali e ore A40,
checkpoint iniziale e finale con SHA-256, config risolto, criterio di selezione e
ogni deviazione dal protocollo. Nessun risultato parziale o non appaiato viene
presentato come confronto finale.

## Punto 4 — azione zoom (protocollo congelato prima della run)

Questa sezione definisce il confronto a quattro bracci va1, va1+RL,
va1+zoom (`vz1`) e va1+zoom+RL. Lo zoom non cambia l'architettura: è un blocco
percettivo aggiuntivo elaborato dallo stesso encoder e proiettore congelati.

### Decisioni di disegno

Il testo aggiunto, identico in SFT, RL ed eval, è:

```
Optional action: before answering, you may output exactly zoom:{"channel":"<name>","start":s,"end":e}. Use one listed channel and absolute sample coordinates [start,end). Otherwise output the final answer now.
```

L'azione deve occupare l'intero output del primo turno e avere esattamente le
chiavi `channel`, `start`, `end`; valori booleani, coordinate non intere, chiavi
extra e canali non elencati sono invalidi. Il blocco inserito dall'ambiente ha
sempre l'intestazione:

```
Zoom on channel <name>, samples [s, e) of L:
```

Nel secondo turno il post-prompt riporta, dopo domanda e istruzione, la riga
`Assistant action: <azione canonica>` e chiede `Now output the final answer in
absolute coordinates.`. Le coordinate restano quindi riferite alla serie
originale, mai alla finestra.

Una richiesta con intersezione non vuota viene clampata a `[0,L)`. Se dopo il
clamp è più corta di 1024 campioni, viene estesa simmetricamente (con compensazione
al bordo) fino a `min(1024,L)`; la lunghezza massima è `L`. Intervallo invertito o
senza intersezione, canale ignoto o azione non parsabile producono nessun blocco e
incrementano `zoom_invalid`. Un output finale al primo turno non è invalido e
termina l'episodio senza zoom. È ammesso un solo zoom.

L'oracle usa l'intervallo gold pertinente con margine del 25% della sua
durata su ciascun lato. Per `localize_all_channels`, dove un episodio può leggere
un solo canale, Z0 sceglie l'evento gold più lungo (pareggio: ordine dei canali,
poi start); Z1 sceglie uniformemente fra gli eventi gold ordinati usando RNG
deterministico per sample. Per categorie senza intervallo gold utilizzabile si usa
la finestra euristica. In Z1 il 70% delle traiettorie usa una finestra gold con
margine casuale uniforme fra 10% e 50% per lato e il 30% usa l'euristica.

L'euristica costruisce la piramide Haar di `wavelet_pyramid.py`, divide la serie in
64 bucket e somma, per livello e bucket, l'energia detail normalizzata per
l'energia mediana robusta del livello. Il primo candidato è il massimo globale su
canale e bucket (pareggi: ordine canale e tempo); attorno al bucket viene estratta
una finestra di lunghezza `max(1024, ceil(L/8))`, limitata a `L`. È un singolo scan
NumPy deterministico, senza accesso al gold.

`prepare_batch` viene usato per costruire testo, placeholder, attention mask e
label. Poiché il suo stacking riempirebbe la finestra corta con zeri fino alla
lunghezza dei canali originali, il wrapper L3 codifica separatamente il tensore
dei canali originali e il tensore `(1, e-s)` dello zoom tramite
`_encode_and_project`, concatena le feature nello stesso ordine dei placeholder e
usa `_merge_features`. Questo dettaglio è necessario perché le 256 feature zoom
abbiano posizioni relative e normalizzazione MAD sulla vera finestra, non su una
finestra zero-padded a `L`; non introduce parametri né modifica `kaggle/patch/`.

I log-prob RL sono calcolati con un forward teacher-forced separato per ciascun
segmento emesso: azione e, se lo zoom è valido, risposta. Per un episodio senza
azione esiste il solo segmento di risposta. Il valore PPO e la KL sono mediati su
tutti e soli i token emessi, concatenando i due vettori; header zoom, action-history
e istruzioni dell'ambiente non sono target. Due forward evitano di materializzare
la traiettoria multimodale più lunga in un unico grafo, mantengono batch uno e
permettono il gate `old == current` sul prompt realmente visto da ogni segmento.

Il pool Z2 non usa filtro di informatività: quel filtro richiederebbe 1600 rollout
aggiuntivi a ogni ricampionamento e selezionerebbe prompt in funzione degli output
della policy. A step 0 e 100 si campionano invece 200 prompt train-only, 50 per
ciascuna delle quattro categorie ammesse, bilanciati round-robin anche per
lunghezza. Il seed del round è `20260911 + round`; i 200 prompt coprono esattamente
100 step P2. `lead_lag_with_magnitude` è esclusa dal pool e dalla reward RL, ma
resta nelle eval.

Z1 costa circa 2 secondi per ciascuno dei 4488 optimizer step già misurati, più il
secondo forward e l'eval: la previsione prudente è 6 ore + 0,5 ore. Z0 richiede tre
letture di probe80 e val520 (circa 1,5 ore); Z2 200 step G8 P2 più quattro val520
(circa 14 + 2 ore); Z3 due letture test800 (circa 1,2 ore). Totale dichiarato:
circa 25 ore A40, circa 22 USD.

Non sono coperti più zoom per episodio, encoder o proiettore allenabili, né
`lead_lag_with_magnitude` nel pool RL. `vz1` è un nuovo SFT e non deve essere
identico a va1; il gate di identità esatta vale soltanto per l'inizializzazione Z2
rispetto a `vz1`.

### Fasi e accesso ai dati

Z0 esegue no-zoom, oracle e heuristic su probe80 e val520. Z1 addestra una sola
epoca completa e richiede coverage esatta per categoria; il suo gate usa val520
con zoom scelto dalla policy. Z2 usa G8, P2, 200 step, checkpoint ogni 25 e val520
agli step 50/100/150/200; seleziona il massimo primary (pareggio entro 0.002: primo)
e si ferma dopo due validation senza miglioramento di almeno 0.005. Z3 è l'unica
fase autorizzata a usare `--split test` e confronta entrambi i bracci zoom con i
due JSONL di controllo tramite bootstrap appaiato 10.000.

Ogni fase richiede il marker `OK` della precedente, usa una directory nuova e
scrive `summary.json`, log e marker terminale `OK` oppure `FAILED`. Il worker deve
essere lanciato in tmux con redirezione esplicita su file: il self-detach senza
redirezione perde l'output operativo.
