# Ablazioni e design alternativi successivi

**Stato: backlog scientifico, non configurazioni da eseguire ora.**  
**Riferimento unico:** `WR-Agent-4B-Wildcard-v1`, definito in [DESIGN.md](DESIGN.md) e [prototype.spec.json](prototype.spec.json), versione 0.3.

Il modello principale comprende Haar, rami profondi A/D/raw, adapter zoom, mixer RoPE, wildcard nella full attention Qwen, CoT e zoom multicanale. Il training è SFT P1→P2→P3. `direct` e `agentic` sono modalità dello stesso checkpoint; non sono due architetture da sviluppare in parallelo.

Questo documento separa tre cose: diagnosi a pesi fissi, vere ablazioni con riaddestramento e possibili estensioni del sistema. **Prima si costruisce e si verifica il modello completo.** Non è necessario eseguire l'intero elenco per ottenere il primo risultato.

## 1. Priorità dopo il primo modello funzionante

| Priorità | Confronto | Domanda scientifica | Cosa mantenere controllato |
|---|---|---|---|
| 1 | Wildcard mista vs fase mista con testo a zero | Eliminare la fase testo–segnale aiuta davvero? | Stesse regole TS–TS e testo–testo; cambia soltanto la regola per le coppie miste |
| 2 | RoPE Qwen proposta vs RoPE Qwen nativa | Serve il meccanismo nel decoder oltre a quello del percettore? | Percettore, dati, traiettorie, LoRA, budget e training |
| 3 | CoT agentica vs agente senza segmenti CoT | Il ragionamento esplicito migliora decisioni e risposta? | Stesso accesso al tool, numero massimo di viste e criteri di scoring |
| 4 | Elaborazione temporale profonda vs elaborazione puntuale | È utile mescolare coefficienti prima della compressione? | DWT, feature in ingresso, tokenizzazione e numero di token |
| 5 | Rami A/D distinti vs pesi condivisi | Serve la specializzazione delle due statistiche? | Profondità, normalizzazione e token; riportare la differenza parametrica |
| 6 | Adapter locali vs trunk senza adapter locali | Quanto conta la specializzazione dello zoom? | Stessi crop e 512 token locali per canale |
| 7 | Ramo raw attivo vs sostituzione della sua quota con wavelet | Il percorso raw conserva informazione utile o domina il modello? | Budget totale identico, regola di riassegnazione fissata prima del training |

### A1 — Wildcard vs tempo zero: controllo corretto

Nel modello principale il termine T vale:

- TS–TS: `qT R(tau_j-tau_i) kT`;
- testo–testo: `qT R(p_j-p_i) kT`;
- coppie miste: `qT kT`.

Nel controllo A1 cambiano **soltanto le coppie miste**: si usa il termine con `t_testo=0`, `t_TS=tau`. Testo–testo conserva la RoPE nativa anche nel controllo. Confrontare direttamente l'intera versione 0.2 contro 0.3 cambierebbe anche la regola testo–testo e non isolerebbe l'effetto wildcard.

Diagnosi iniziale a pesi fissi: gap testuale crescente, traslazione dei timestamp a Q/K fissate e lettura degli stessi crop. Questo misura proprietà matematiche e sensibilità del checkpoint; **non** dimostra quale architettura imparerebbe meglio. Per la conclusione scientifica allenare i due bracci con protocollo comune, assegnando gli stessi esempi e semi alle traiettorie di training.

### A2 — RoPE nel decoder

Il controllo mantiene il nuovo mixer percettivo ma ripristina la RoPE posizionale nativa in Qwen per tutte le coppie. Serve a isolare il valore dell'intervento nel backbone. Il risultato va distinto dal confronto con va1, che cambia molte altre componenti.

Si possono confrontare anche gap fra osservazioni e lunghezze della serie mai viste nel train. I token aggiunti non devono introdurre nuova evidenza numerica. Non confondere un test con testo riempitivo con una dimostrazione che una CoT semanticamente migliore sia inutile.

### A3 — CoT

Entrambi i bracci restano agenti capaci di più zoom. Nel controllo si rimuovono i segmenti di ragionamento dal protocollo e dalle traiettorie, mantenendo azioni e risposte. Un agente senza CoT non è una baseline single-pass.

Riportare due letture: uguale accesso alle osservazioni e confronto a budget generato dichiarato. La CoT consuma token: assegnare arbitrariamente il suo risparmio a più zoom cambia due fattori. Disabilitare la CoT soltanto in inferenza su un checkpoint che l'ha appresa è una diagnosi di distribuzione, non l'ablazione di training definitiva.

### A4–A7 — Percettore

Per profondità e separazione A/D confrontare lettura a crop fissi oltre alla policy autonoma; altrimenti la qualità del selettore può nascondere quella del lettore. Un controllo a parametri approssimativamente equivalenti può essere informativo, ma non sostituisce il conteggio di operazioni e token realmente eseguiti.

Rimuovere un ramo soltanto dopo il training è un controllo di dipendenza del modello. Per stabilire se quel ramo fosse necessario occorre anche riaddestrare il controllo. Nella prova raw, mantenere 256/512 token riassegnando la quota ai riassunti wavelet secondo una regola preregistrata; una seconda lettura con slot mascherati misura invece la dipendenza dal ramo, non l'efficienza a budget uguale.

## 2. Estensioni che non entrano nel modello principale

| Estensione | Quando avrebbe senso | Perché è fuori dal primo modello |
|---|---|---|
| **db4 al posto di Haar** | Errori concentrati su oscillazioni o transitori con struttura di banda | Non garantisce miglioramento; richiede supporti e bordi corretti |
| **Selettore appreso dei token** | Il lettore è sano ma la selezione perde eventi deboli | Introduce una nuova policy di compressione e relativi gradienti/obiettivi |
| **Encoder locale completamente indipendente** | Gli adapter condivisi non bastano su morfologie locali | Cambia condivisione, parametri e fabbisogno di dati |
| **Rapporto RoPE diverso da 50/50 o rotazione di più dimensioni Qwen** | Il modello mostra un conflitto misurabile fra ordine linguistico e geometria TS | Aggiunge iperparametri e modifica più feature preaddestrate |
| **Memoria esterna con cross-attention dedicata dal testo** | La wildcard non basta a recuperare eventi con contesto lungo | Cambia il percorso di accesso alle osservazioni, non soltanto la RoPE |
| **REPL numerico, statistiche dedicate o immagini** | Servono operazioni precise non recuperate dagli embedding | Cambia l'ambiente e quindi il significato del confronto con TimeRLM |
| **GRPO dopo SFT** | Lettore percettivo verificato, ma scelta/arresto rimangono deboli | È un esperimento di post-training successivo, non una fase del prototipo fissato |

Non prioritaria la riproposizione del resampler ricorrente v3: l'audit ha già mostrato collasso del suo contesto. Per riprenderlo serve una motivazione nuova e verificabile. V3 resta un'ablazione del Livello 1.

Per GRPO conservare step 0, congelare prima il criterio di selezione, controllare localize/multicanale e distinguere reward ufficiale dai controlli su lead_lag. Log-probabilità e campionamento devono riferirsi alla stessa distribuzione; osservazioni e tool output non sono azioni da includere nella policy loss. Il confronto deve attribuire il guadagno al post-training, non a un cambio simultaneo di tool o percettore.

## 3. Regole comuni della futura campagna

1. Un fattore principale alla volta, con una domanda esplicita prima della run; nessuna griglia completa obbligatoria.
2. Stessi split e ID; nessun crop della stessa registrazione in split diversi. I crop di lettura sono fissati prima del confronto, indipendentemente dalla policy.
3. Dati e budget comparabili: riportare parametri, token, traiettorie e aggiornamenti. Quando non sono uguali, dichiarare quale domanda il confronto può ancora risolvere.
4. Metriche per categoria, errori di formato e sensibilità al contenuto; primary da solo non basta. Non chiamare la finestra gold un ceiling architetturale.
5. Differenze appaiate e bootstrap sugli ID. Più crop dello stesso ID non sono repliche indipendenti. Dopo un pilot promettente, repliche di training prima di rivendicare un effetto piccolo.
6. Checkpoint scelti con lo stesso criterio di validation; test escluso dalla scelta del design. L'uso ripetuto della validation storica richiede una valutazione finale tenuta separata.
7. Nessun risultato è previsto in anticipo. In particolare, db4, wildcard o CoT non autorizzano a promettere un primary specifico.

I confronti di sistema con va1, W1, TSLM diretti e TimeRLM restano nel design principale perché valutano il modello completo. Questa pagina serve soltanto a pianificare come attribuirne successivamente i risultati alle componenti.
