# Piano: rendere misurabile e selettivo il contesto v3

Piano iniziale: 8 settembre 2026; aggiornato il 9 settembre dopo il completamento dell'audit. Stato: analisi del codice, piano sperimentale e audit GPU a pesi fissi sul nodo; nessuna modifica al modello o ai launcher e nessun nuovo training.

**Aggiornamento dopo la verifica sul nodo.** Il [report dell'audit](AUDIT-CONTEXT-20260908.md) aggiorna la priorità di questo piano: il contesto è già quasi uniforme al passo 500, quando la scala è ancora 0,10354. A fine epoca 1 la componente variabile fra token vale circa lo 0,156% della RMS del contesto; self-attention e soprattutto MLP dominano le letture nel loop. La scala risale inoltre da 0,00244 a 0,01264 nel checkpoint successivo. Prima di adottare LR lento e gate finale, verificare un lettore che conservi l'informazione, usando cross-attention-only come controllo e reintroducendo le trasformazioni ricorrenti con contributi controllati. Le sezioni seguenti conservano il protocollo iniziale e le ablazioni utili, ma questo aggiornamento prevale sul loro ordine di esecuzione. Scala, geometria e confronti appaiati su 80 esempi sono ora misurati; verifica sulla val completa e nuovi training restano da fare.

Anche il confronto sulle risposte generate è concluso: primary 0,16410 acceso contro 0,16071 spento, delta +0,00340 con IC95 [-0,00855; +0,01522]. IoU sui 18 positivi localize e F1 sui 9 all_channels non vuoti sono zero in entrambe le condizioni. Il campione non mostra un recupero di localizzazione dovuto al contesto attuale.

Obiettivo: verificare se il contesto multi-scala aggiunge informazione utile a v1 e, se sì, permettere al modello di usarla selettivamente. Una scala diversa da zero non è un criterio di successo.

**1. Evidenza disponibile e correzioni alla diagnosi**

Ho letto encoder v1/v3, wrapper v3, proiettore e loss ereditati, configurazione tied, trainer, setup e launcher del nodo, worker di valutazione, smoke, test e storia delle revisioni. Nel piano iniziale i valori della nuova traiettoria venivano dal messaggio di Jacopo; gli artefatti `integrity/artifacts/v3tied/` documentano la precedente run a init zero. La nuova traiettoria e le ampiezze sono state successivamente verificate sul nodo: risultati e limiti del campionamento sono nel report collegato sopra.

Il setup attuale è: 256 token da 512 dimensioni per canale; base v1 con banda grossolana e top-k globale dei dettagli; una piramide Haar riutilizzata; letture A_L, D_L…D_1 fino a 11 iterazioni; al massimo 1024 coefficienti per livello; blocco condiviso nella variante tied; prior Cauchy centrata sulle posizioni dei coefficienti; Qwen3.5-4B con LoRA r16/alpha32. Encoder base, proiettore e LoRA sono allenati a 1e-4; tutto il nuovo gruppo context, inclusi scala e affine della LayerNorm, a 1e-3. Batch 1, clipping globale a 1, bf16 con parametri allenabili promossi a fp32, 4488 esempi per 5 epoche, selezione sulla validation da 520 esempi.

Quattro precisazioni cambiano le verifiche da fare:

- `context_norm = nn.LayerNorm(D)` ha gamma e beta apprendibili. Prima dell'affine, la normalizzazione dà RMS circa 1, non norma L2 pari a 1; dopo l'affine la norma può cambiare. Quindi `context_scale` non è l'unico controllo dell'ampiezza. La normalizzazione ostacola una codifica dell'utilità nella sola norma di z, ma non rende matematicamente identica l'ampiezza finale di ogni token per tutto il training. Vedere `patch/wavelet_encoder_v3.py:163` e [definizione PyTorch](https://docs.pytorch.org/docs/2.9/generated/torch.nn.LayerNorm.html).
- Il gate esistente è uno scalare per canale e livello, condiviso dai 256 token. Moltiplica solo la lettura cross-attention; self-attention e MLP continuano a trasformare z. Chiudere il gate non salta l'intera iterazione. I livelli si combinano attraverso operazioni non lineari: la LN finale non cancella esattamente tutti i loro effetti di ampiezza. Vedere `patch/wavelet_encoder_v3.py:64` e `:345`.
- Dopo l'encoder, `MLPProjector` applica un'altra LayerNorm alla somma base+contesto, poi Linear e GELU. Il 9,7% prima del proiettore non quantifica da solo la perturbazione vista da Qwen, né esclude un costo iniziale di adattamento. Vedere `timerlm/sft/tslm/src/models/chatts.py:30`.
- AdamW usa `delta_a = -lr * m_hat/(sqrt(v_hat)+eps) - lr*wd*a`, non un passo costante di 1e-3. Il fattore a nel gradiente dei blocchi è reale; la sua traduzione in aggiornamento dei pesi dipende anche dai momenti Adam, dal clipping e dalla storia del training. Misurare entrambi. Vedere `kaggle/patch/train.py:1090` e [algoritmo AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW).

Calcolo locale dal config, assumendo lo scheduler lineare senza override del launcher: 22440 step totali, 673 di warmup. Il prodotto dei soli fattori di decay `prod(1 - 0.01*lr_t)` porterebbe 0,1 a **0,096255** dopo 4488 step, riduzione **3,745%**; a **0,094426** dopo 6988 step, riduzione **5,574%**. Epoca 2, passo 2500 corrisponde a 6988 update complessivi con coverage piena. Questo esclude il decay come spiegazione quantitativa principale del calo riportato; non identifica ancora la causa della componente adattiva.

**2. Prima fase: interrogare i checkpoint esistenti**

Prima di modificare l'architettura, produrre un audit della nuova run con identificatore univoco: commit e hash dell'overlay effettivamente caricato, config, manifest dati/Qwen, coverage, stato optimizer/scheduler e timestamp. Conservare i checkpoint e i log. Non riutilizzare il launcher attuale sullo stesso nome: elimina la directory della run e lo stato del worker.

Usare tutti i checkpoint disponibili per la tabella dei parametri; per i forward costosi iniziare da init, step 500, 2000, 4488 e ultimo checkpoint disponibile vicino a 6988. Non assumere che il nome del batch sia il numero di optimizer update: verificarlo nei metadati e nei log.

Definire un insieme diagnostico fisso di 80 esempi della sola validation, stratificato per categoria, presenza/assenza, lunghezza e numero di canali, includendo serie lunghe e 16 canali. Salvare gli ID; usare gli stessi esempi in ogni confronto. Le analisi per sottogruppi piccoli sono esplorative e vanno confermate sulla val intera.

Per ogni checkpoint misurare:

| Misura | Perché serve |
|---|---|
| a; distribuzione di gamma; RMS di beta; prodotti a*gamma e a*beta | Individuare compensazioni o un secondo spegnimento nell'affine finale |
| RMS di z prima della LN, RMS di h=LN(z), diversità di h tra token e tra input | Separare crescita interna, omogeneizzazione e dipendenza dalla serie |
| `rho_i = RMS(a*h_i)/(RMS(base_i)+eps)` | Misurare quanto contesto entra davvero, con p10/p50/p90/p99 per canale e categoria |
| `RMS(P(base_i+a*h_i)-P(base_i))/(RMS(P(base_i))+eps)` e coseno | Quantificare il cambiamento effettivo dopo il proiettore P dello stesso checkpoint |
| Derivata della loss rispetto ad a, per esempio e categoria | Capire quali esempi favoriscono aumento o riduzione, senza inferirlo dai soli checkpoint |
| Gradiente prima/dopo clipping, fattore di clipping, update RMS e update/weight RMS per blocco | Capire se il lettore cambia effettivamente e quale gruppo domina il clipping |

I rapporti rispetto alla base si calcolano sui token validi: il padding può avere norma zero. Riportare separatamente gli slot di padding, che oggi ricevono anch'essi un residuo. Non salvare grandi attivazioni durante tutto il training: aggregare statistiche detached e campionare, per esempio, ogni 100 update.

Per Adam, salvare da nuovi pilot LR per gruppo, gradiente dello scalare, momenti con bias correction e delta effettivo prima/dopo ogni update campionato. Nei checkpoint storici i momenti consentono un audit parziale; checkpoint distanti 500 passi non ricostruiscono tutti i gradienti o gli update intermedi. Distinguere componente decay e componente adattiva.

Eseguire tre interventi a pesi fissi, in eval deterministica:

1. **Contesto normale contro a=0**, sullo stesso checkpoint e sulle stesse righe. Registrare `delta_NLL = NLL(on)-NLL(off)`: negativo significa vantaggio del contesto. Il modello con a=0 conserva base, proiettore e LoRA co-adattati della run v3: non equivale a una v1 allenata separatamente.
2. **Sweep di ampiezza** con `{0, 0.5*a_ckpt, a_ckpt, 2*a_ckpt, 0.1}`, eliminando duplicati. Fare prima teacher forcing; generare le risposte almeno per on/off nei checkpoint iniziale-intermedio e finale. Selezionare altre ampiezze sulla val soltanto. Un beneficio ottenuto rialzando a suggerisce soppressione eccessiva; un danno non prova che un ramo addestrato con una diversa scala sarebbe inutile.
3. **Contesto disallineato**, mantenendo base e domanda della serie originale: sostituire solo h con quello di un'altra serie di lunghezza/canali compatibili, oppure permutare h tra token validi. Preservare quanto possibile le norme. Se normale batte off ma non il controllo disallineato, il guadagno non è ancora attribuibile all'informazione della piramide. È un intervento potenzialmente fuori distribuzione: è evidenza complementare, non una prova isolata.

La NLL è quella dei token di risposta effettivamente supervisionati dal trainer; riportare media per esempio e per categoria, oltre all'aggregato. Valutare anche primary, IoU sui positivi, F1 sui non-vuoti e lead/lag sui casi dipendenti. Il solo miglioramento della sintassi della risposta o della moda `independent` non dimostra migliore percezione.

Output previsto: `context_audit_<run>.json` con provenance e statistiche, tabella on/off per esempio, grafico di scala/residuo effettivo/update lungo gli step. Sono deliverable della fase da implementare, non file già prodotti.

**3. Prima modifica: separare velocità del lettore e della scala**

Intervento minimo e compatibile con i pesi esistenti:

- In `patch/wavelet_v3.py`, mantenere `context` per blocchi, step embedding, gate di livello e context norm; spostare soltanto `context_scale` in `context_scale` come gruppo optimizer separato.
- Config diagnostico: `default: 1e-4`, `context: 1e-3`, `context_scale: 1e-5`. Init dello scalare ancora 0,1; stesso scheduler. 1e-5 è un valore di partenza da verificare, non un optimum dimostrato.
- Nel primo confronto mantenere il decay già in uso, così cambia solo il LR dello scalare. Ridurne il LR riduce anche il suo decay, già quantitativamente piccolo. Una successiva ricetta può mettere decay zero su scala e gate, ma questo non è il rimedio principale e richiede supporto esplicito per decay per gruppo nel trainer.
- Registrare la variazione di gamma/beta: rallentare a non impedisce che l'affine finale riduca il residuo.

Se a cambia poco ma il residuo effettivo si spegne comunque, fare un controllo separato con **a e affine della sola context norm fissati ai valori iniziali per la prima epoca**, mantenendo allenabile il lettore. Serve a testare se il ramo impara quando l'uscita non può ridurre globalmente la propria ampiezza. Rilasciare poi i parametri e vedere se l'utilità persiste. Non adottare una soglia minima permanente della scala come prova di successo.

Un congelamento va gestito dal trainer per optimizer step, registrato nel checkpoint, con gestione esplicita di momenti e scheduler allo sblocco. Non introdurre contatori di forward nell'encoder: activation checkpointing riesegue parti del forward durante backward. Non congelare il percorso v1 nel confronto principale: cambierebbe un'altra condizione sperimentale.

**4. Seconda modifica: consentire una scelta per token all'uscita**

La modifica architetturale prioritaria è un gate per token dopo `context_norm`, prima della somma e del proiettore ereditato:

```text
h_i = context_norm(z_i)
g_i = 2 * sigmoid(w^T [LN_no_affine(base_i), h_i] + b)
out_i = base_i + a * g_i * h_i
```

`w=0`, `b=0` inizializzano g a 1: con gli stessi pesi di lettore e scala, il forward iniziale coincide con la v3 attuale, senza dimezzare accidentalmente il contesto. Il gate può avvicinarsi a zero o salire fino a 2; il lettore riceve gradiente fin dal primo backward. Usare inizialmente una sola Linear, condivisa tra token e canali, senza una nuova MLP profonda.

Il gate è condizionato sul contenuto di ogni token e sul contesto letto. Non usa gold, categoria target o annotazioni di anomalia nel forward. Un eventuale confronto con le annotazioni serve soltanto all'analisi e non impone che i token utili coincidano con la finestra anomala.

Gruppo `context_token_gate` iniziale a 1e-4, con lo stesso scheduler. Conservare inizialmente context norm, gate di livello, prior, top-m, profondità e percorso v1: il confronto con la variante precedente isola l'aggiunta del gate. Per il primo confronto conservare anche il comportamento corrente del padding; un masking nuovo sarebbe un'ulteriore ablazione.

Monitorare distribuzione e saturazione di g, residuo effettivo e utilità on/off. Un gate tutto chiuso è una possibile risposta dell'ottimizzazione, non un guasto automaticamente da proibire. Non aggiungere subito penalità che premiano gate aperti, loss ausiliarie o dropout della base: favorirebbero uso forzato senza dimostrarne il vantaggio.

Tenere la LN del contesto nel primo candidato. Toglierla insieme all'aggiunta del gate cambierebbe sia la stabilità del loop sia la selettività. Se l'audit indica che la LN distrugge una norma informativa, confrontare in un secondo momento normalizzazione condivisa tra token o un'uscita non normalizzata, calibrando lo stesso residuo iniziale e controllando crescita con profondità e precisione bf16.

**5. Esperimenti in ordine e criteri di decisione**

| Run | Uscita | LR scala | Domanda |
|---|---|---:|---|
| A | v3 attuale, gate finale uniforme | 1e-3 | Controllo del fenomeno |
| B | Come A | 1e-5 | La scala si adatta troppo rapidamente rispetto al lettore? |
| C | B + gate per token dopo LN | 1e-5 | La selettività aggiunge utilità oltre al cambio optimizer? |
| D, solo se utile | A + lo stesso gate per token | 1e-3 | Serve davvero anche rallentare la scala, o basta il gate? |

Partire da B dopo l'audit; aggiungere C se B non risolve o se resta un danno eterogeneo tra esempi/token. D completa il confronto fattoriale quando serve attribuire l'interazione. A può essere riusata solo se manifest, inizializzazione, ordine dati e scheduler sono davvero confrontabili; altrimenti è riferimento descrittivo e va ripetuta per il confronto controllato.

Usare init pulita per l'esperimento principale, con pesi iniziali identici nelle parti comuni, inclusi LoRA e proiettore. Il solo `seed: 42` non garantisce questo dopo aver aggiunto moduli che consumano RNG: isolare il RNG della nuova inizializzazione o caricare lo stesso stato iniziale comune, e fissare indipendentemente l'ordine dei campioni. Riutilizzare anche i pesi iniziali del lettore tra A/B/C. Tied/full resta fisso.

Prima epoca completa per lo screening; controllo e candidato promettente devono arrivare almeno a due epoche, oltre la finestra di collasso riportata. I primi 500 passi verificano salute tecnica, non decidono il vincitore. Mantenere lo scheduler della ricetta da 22440 step: impostare semplicemente `num_epochs: 1` accorcerebbe warmup e decadimento, invalidando il pilot.

Valutazioni diagnostiche sugli stessi 80 esempi a step 500/2000/fine epoca; confronto determinante sulla validation intera. Usare differenze appaiate per esempio e intervalli di confidenza; per primary, che aggrega categorie, preservare quell'aggregazione nel bootstrap. Prima di attribuire il risultato all'architettura replicare candidato e controllo v1 con almeno due seed, preferibilmente tre, trattando separatamente variabilità di training e incertezza sugli esempi.

Criteri:

- **B aiuta e C non aggiunge vantaggio:** scegliere la modifica minima all'optimizer.
- **C supera B e a=0, e il contesto corretto supera il controllo disallineato:** evidenza a favore del contesto selettivo. Confermare su val intera e seed aggiuntivi.
- **La scala cala ma residuo effettivo e vantaggio on/off restano:** non chiamarlo collasso basandosi sullo scalare.
- **Residuo e update restano attivi, ma on non aiuta rispetto a off:** il problema non è più solo l'accesso al gradiente. Indagare il lettore e il contenuto, senza tenere aperto il ramo per definizione.
- **Tutti i candidati falliscono:** conclusione limitata a queste ricette e a questo protocollo. Non prova che una rappresentazione multi-scala sia inutile in generale.

Solo dopo questo passaggio investire nelle run a 5 epoche e nelle ablazioni tied/untied/sdepth. Il test da 800 esempi resta fuori dalle scelte di architettura, LR, profondità e soglie; usarlo quando ricetta e selezione best-by-val sono fissate.

**6. Verifiche tecniche necessarie prima dei pilot**

- Conservare identità v3/v1 con residuo disabilitato, singola piramide, budget token e parità fp32/bf16. Per il gate nuovo verificare che g=1 replichi l'uscita precedente e che g=0 su uno slot lasci quel token base esattamente intatto.
- Verificare gruppi optimizer senza parametri omessi o duplicati; testare LR effettivi dopo scheduler, eventuale congelamento/sblocco e round-trip del checkpoint.
- Estendere il witness sintetico: esempi in cui l'informazione target è assente dalla selezione v1 ma presente nella piramide, insieme a esempi dove il contesto non serve. Misurare miglioramento held-out e differenza on/off. Il witness attuale altera l'affine per costruire un caso ortogonale: dimostra un meccanismo di gradiente, non l'utilità end-to-end su AnomalyXL.
- Lo smoke Qwen di quattro step rimane un controllo di memoria, coverage, checkpoint/eval e gradienti. Il requisito attuale di spostamento minimo dello scalare va reso coerente con LR lento o congelamento: richiedere update finiti e misurabili del lettore, senza imporre che si muova proprio a. Non ricavare il LR di training dalla necessità di superare una soglia in quattro passi.
- Aggiungere un harness pilot con directory univoche, coverage esplicita, scheduler originale e sole valutazioni val. Il worker corrente pretende cinque epoche e lancia automaticamente il test: non è adatto ai pilot.
- Il trainer attuale riparte da `checkpoint['epoch'] + 1`: non è un resume esatto intra-epoca. Per continuazioni causali utilizzare confini d'epoca o implementare e testare ripristino di cursore dati, RNG e optimizer. Non basta caricare il checkpoint al passo 500.
- Separare compatibilità dei pesi e compatibilità optimizer: lo split dello scalare cambia i gruppi Adam; l'aggiunta del gate cambia le chiavi. Mantenere la modalità legacy e migrazioni esplicite con g=1, senza accettare silenziosamente stato optimizer incongruente.

File interessati dall'implementazione successiva: `patch/wavelet_encoder_v3.py`, `patch/wavelet_v3.py`, `patch/test_encoder_v3.py`, nuovi config diagnostici, hook opt-in in `kaggle/patch/train.py`, smoke e nuovo harness di audit/pilot. Aggiornare README e note smoke: alcune descrizioni sono storiche e non corrispondono alle chiavi e soglie correnti. La fonte della valutazione è il codice effettivamente spedito, verificato dal manifest.

**7. Se serve una terza fase architetturale**

Aprirla soltanto dopo i confronti precedenti:

- **Lettura poco informativa:** misurare copertura locale dei coefficienti scelti. A L=131072 il livello D1 conserva al massimo 1024/65536 coefficienti, circa 1,56%; il top-m globale può lasciare finestre senza evidenza locale. Non attribuire automaticamente un fallimento al gate.
- **Gate per livello:** confrontare separatamente il gating dell'intero incremento `z_new-z` col gating della sola cross-attention, usando interventi per livello. La media di un gate non è una misura causale dell'importanza della scala.
- **Ricorrenza:** le query cross-attention attuali sono sempre `base + step_embed`, senza z. Il residuo è ricorrente, la query di lettura non si aggiorna col contesto accumulato. Un'eventuale query dipendente da z è una nuova ablazione.
- **Geometria/obiettivo:** solo con evidenza specifica valutare correzione delle posizioni bf16, ampiezza del prior dipendente dal livello, pretraining del lettore o loss ausiliarie. Preservare confronti separati dalla domanda principale.

La proposta iniziale era **misurare l'utilità a pesi fissi, separare il LR della scala, poi provare il gate per token dopo la context norm**. L'audit ha aggiunto una priorità precedente alle modifiche di uscita: verificare che il lettore conservi struttura e dipendenza dalla serie, anziché amplificare una direzione quasi comune. La ricetta finale si sceglie sui miglioramenti della validation e sull'uso informativo del contesto, non sulla sopravvivenza dello scalare.
