# Protocollo L3 — GRPO multimodale va1

Stato: protocollo dello smoke, 10/09/2026. Il gate descritto qui precede qualunque
run lunga e qualunque accesso al test pinnato.

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
testuale, lo sblocco di encoder/proiettore, più epoche PPO interne o una run RL
lunga. Queste estensioni richiedono un nuovo protocollo e un nuovo gate.

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

## Confronto del punto 5

Dopo lo smoke, ogni candidato RL lungo viene selezionato soltanto sulla validation
con una regola dichiarata prima della run. Il confronto finale usa una sola volta lo
stesso test pinnato di 800 esempi per:

- va1 SFT `epoch_5_end.pt`;
- va1 + RL, inizializzato dallo stesso `epoch_5_end.pt`.

Le due condizioni usano stesso `eval_max_new_tokens`, decodifica greedy e stack
bf16/SDPA pinnato. Il confronto è appaiato per `(id, category, ground_truth)` e usa
10.000 bootstrap con seed dichiarato, adattando
`v3-recurrent/audit/paired_eval.py` alla coppia SFT/RL. Si riportano primary medio,
differenza appaiata con CI 95%, risultati per categoria e conteggio esatto 800/800.

Ogni report dichiara budget RL in optimizer step, prompt, rollout totali e ore A40,
checkpoint iniziale e finale con SHA-256, config risolto, criterio di selezione e
ogni deviazione dal protocollo. Nessun risultato parziale o non appaiato viene
presentato come confronto finale.
