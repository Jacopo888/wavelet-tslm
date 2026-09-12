**Valutazione della repository e della continuazione proposta da Claude — 12 settembre 2026**

Verdetto: continuare il Livello 3 su va1 è ragionevole, ma non avviare ancora la catena da circa 30 ore proposta nella chat. Prima serve un breve esperimento che separi l'informazione del segnale dall'informazione delle coordinate oracle, seguito da un pilot della scelta della finestra. Le modifiche all'SFT sono in parte corrette; due zoom e RL sono estensioni da giustificare con quei risultati.

**1. Perimetro e gerarchia del progetto**

Ho esaminato la sessione Claude `456b55dd-2879-4efe-87da-2905622d2d00`, compresa la continuazione finale del 12/09 alle 07:14 UTC; la mappa della repository, protocolli e risultati dei diversi livelli; il codice del percettore, zoom, SFT, parti del runner GRPO e driver; i risultati per esempio della validation. Il ricalcolo è locale su CPU. Non ho lanciato training, interrogato il nodo o generato nuove predizioni sul test.

La copia locale `timerlm/` è vuota: non è stato possibile rivalidare l'intero runtime upstream o caricare i checkpoint del nodo. Gli esiti delle run GPU sono quelli degli artefatti archiviati; non certifico lo stato attuale della GPU. Alcuni diagnostici hanno localmente solo il riepilogo, non il JSONL: in particolare vz1 senza zoom e oracle largo. Per questi non invento intervalli di confidenza appaiati.

La gerarchia corretta è:

| Ambito | Ruolo e stato documentato | Conseguenza |
|---|---|---|
| Livello 0 | Pilot su 10 episodi: wavelet 7,4 comandi medi, code 14,8; IoU 0,881 contro 0,909 | Evidenza esplorativa di efficienza di navigazione, non vittoria definitiva di accuratezza |
| Livello 1 | Percettore va1, 256 token per canale; 5 epoche a copertura completa, test primary 0,2744 | Baseline da conservare; localizzazione positiva ancora debole |
| v3-recurrent | Ablazione architetturale del Livello 1 | Non è il Livello 3 e non è un prerequisito dello zoom |
| Livello 2 | Agente testuale con REPL e skill wavelet, kit prime-rl/2B | Nei materiali ispezionati non c'è un confronto finale completo dei due bracci; non attribuirgli un risultato consolidato |
| Livello 3, controllo senza zoom | va1 + GRPO single-turn: test 0,2869 | Il guadagno aggregato viene quasi integralmente da lead_lag; non dimostra migliore localizzazione |
| Livello 3, punto 4 | Zoom come blocco percettivo aggiuntivo, SFT Z1 completata e gate fallito | È il lavoro attuale; Z2/Z3 della catena zoom non sono stati eseguiti negli artefatti disponibili |

Le sigle L0–L4 nel driver della run lunga sono stadi operativi. `RISULTATI-L4-ZOOM.md` riguarda il punto 4 del Livello 3, non un nuovo livello concettuale.

Su v3: la run iniziale con contributo residuo quasi nullo non misura un beneficio del contesto ricorrente. L'audit successivo con scala iniziale non nulla documenta anche perdita di struttura del contesto; i pilot locali/globali non stabiliscono un vantaggio. Queste conclusioni restano nel ramo di ablazione L1. Non suggerisco ora di riaprire v3 o cambiare RoPE per risolvere il problema del controllo dello zoom.

La separazione fra dati pinnati, run storiche incomparabili e va1 a copertura completa è un punto forte dell'infrastruttura. Le affermazioni sui vecchi ChatTS in OOM vanno mantenute come risultati di fattibilità su quell'hardware, non come confronti di accuratezza fra modelli entrambi addestrati. I riferimenti ai migliori TSLM pubblicati in documenti precedenti sono inoltre da aggiornare prima dell'uso nella proposta.

**2. Cosa confermano i risultati dello zoom**

Tutti i numeri seguenti sono sulla validation 520, non sul test 800.

| Condizione | Primary | Localize | All channels | Classify | Lead-lag |
|---|---:|---:|---:|---:|---:|
| va1 | 0,3105 | 0,2438 | 0,3583 | 0,0902 | 0,2381 |
| vz1 senza zoom | 0,3011 | 0,2200 | 0,3269 | 0,0431 | 0,3475 |
| vz1, azione scelta dalla policy | 0,2869 | 0,2050 | 0,3148 | 0,0337 | 0,2972 |
| vz1, oracle largo | 0,3234 | 0,2890 | 0,4000 | 0,0911 | 0,2547 |
| vz1, oracle stretto | 0,5244 | 0,6333 | 0,6657 | 0,4539 | 0,2547 |

Z1 ha realmente completato 4488 episodi, 8976 segmenti, con 2741 azioni euristiche e 1747 gold. La policy produce azioni valide e risposte non vuote, ma zooma sempre. L'interfaccia è stata appresa; la strategia non supera il riferimento.

L'oracle stretto batte nettamente il largo nella configurazione misurata. È una buona ragione per indagare lo zoom. Non è ancora una dimostrazione che il solo contenuto percettivo della finestra causi il guadagno, né un upper bound matematico della policy ottimale.

La dicitura «~1024 campioni» riguarda le finestre sui casi con intervallo gold. La modalità oracle ripiega sull'euristica per gli altri: la larghezza media su tutti i 520 esempi è 4500,7 campioni. Analogamente, l'esperimento largo usa target 6144 sui casi con intervallo; 6901 è la media aggregata con i fallback. Le condizioni vanno riportate per strato.

**3. L'oracle comunica quasi esattamente il centro dell'anomalia**

`oracle_zoom` seleziona un evento gold, costruisce una finestra simmetrica e la espande al minimo 1024. `zoom_header` e la cronologia dell'azione mostrano al modello canale, start ed end. Di conseguenza il centro dell'header coincide con il centro gold entro mezzo campione in 98/102 positivi localize e 108/110 classify; le eccezioni sono principalmente finestre ai bordi.

Ho calcolato un controllo puramente geometrico: predire un intervallo fisso centrato nell'header, senza leggere alcun valore della serie.

| Controllo sui soli 102 positivi localize | IoU medio |
|---|---:|
| Centro dell'header + ampiezza fissa 64 | 0,4602 |
| Centro dell'header + ampiezza fissa 128 | 0,5970 |
| Centro dell'header + ampiezza fissa 256 | 0,5093 |
| Risposta di vz1 con vero zoom oracle sugli stessi positivi | 0,5549 |

Questo controllo è post-hoc e usa informazione privilegiata: non è una baseline applicabile autonomamente, né un nuovo score ufficiale su tutta la categoria. Dimostra però che le coordinate sono già estremamente informative. Non prova che vz1 ignori i coefficienti; mostra che il risultato corrente non identifica il loro contributo.

Inoltre il jitter SFT attuale varia simmetricamente il margine, non il centro. Sugli eventi brevi la successiva espansione a 1024 elimina quasi tutta la variazione di ampiezza. È una scorciatoia presente anche nelle tracce di training, oltre al problema del 61% euristico.

Il controllo prioritario è mantenere esattamente gli stessi originali, header, azione, formato e numero di token, sostituendo soltanto le feature della finestra con quelle di una finestra diversa di pari lunghezza. Affiancare eventualmente feature azzerate come controllo più fuori distribuzione. Una caduta rispetto al crop corretto misurerebbe il contributo del contenuto. Ripetere con il gold in posizioni casuali interne alla finestra, anziché sempre al centro. `oracle_text` significa formato testuale dell'intestazione più embedding: non è già un controllo header-only.

**4. Il bersaglio non è risolto: il 70% nasconde errori di centraggio**

Ho riprodotto la definizione dello script storico: «copertura» significa almeno il 90% di un evento nella finestra. Non significa centro accurato, né copertura di tutti gli eventi. Il vecchio `gold_intervals` scarta il canale nel multicanale, sovrastimando lì la copertura utile.

| Categoria, soli esempi positivi | n | Copertura temporale riportata | Copertura anche sul canale giusto | Stesso centro, finestra 1024 | Stesso centro, finestra 2048 |
|---|---:|---:|---:|---:|---:|
| Localize | 102 | 70,6% | 70,6% | 16,7% | 33,3% |
| Classify | 110 | 71,8% | 71,8% | 27,3% | 43,6% |
| All channels | 48 | 68,8% | 62,5% | 6,25% | 14,6% |

L'errore mediano di centro è 1778,5 campioni su localize e 1550,25 su classify. Sul multicanale, fra i casi con un evento nel canale selezionato, è 4373,5. Restringere la finestra senza migliorare il centraggio non basta. Questi numeri non predicono ciò che imparerà un nuovo SFT: confutano solo la tesi che la policy attuale abbia già risolto il puntamento fine.

Due zoom potrebbero aiutare, ma il primo crop deve effettivamente permettere di scegliere meglio il secondo. Il fatto che contenga un evento non dimostra questa capacità. Nel multicanale serve anche poter visitare un altro canale: due zoom sempre annidati nello stesso canale non coprono gli eventi distribuiti su canali differenti. Un recupero deve inoltre poter uscire dalla prima finestra quando questa è sbagliata.

**5. Tre interpretazioni da ridimensionare**

«Risoluzione utile = 4 campioni/token» non è dimostrata. L'encoder produce medie grossolane più dettagli top-k a scale diverse, non 256 patch uniformi. A lunghezza 1024 ha 128 medie di blocchi da 8 campioni e 128 dettagli selezionati; a 6144 ha 96 medie di blocchi da 64 e 160 dettagli. Il rapporto L/256 è un rapporto di compressione medio, non il supporto temporale di ciascun token. Crop, MAD locale, bande selezionate, posizioni e formato cambiano insieme. Occorre uno sweep controllato, non una soglia estratta da due condizioni.

«L'SFT non ha degradato il modello» è troppo forte. Senza zoom vz1 perde circa metà dello score classify, scende su localize, all_channels e magnitudo; lead_lag sale e maschera la regressione nell'aggregato. Il confronto policy-vs-va1 ricalcolato sui JSONL dà delta classify −0,0566, CI95 descrittivo [−0,0999; −0,0136], e all_channels −0,0435, CI [−0,0908; −0,0046]. Sono bootstrap post-hoc senza correzione per confronti multipli, non una prova confermativa o una misura della varianza fra training. Per vz1 senza zoom manca qui il JSONL necessario al confronto appaiato.

«Siamo al livello di TimeRLM» non è un confronto valido: oracle sulla validation contro agente autonomo sul test, oltre al differente backbone. I valori 0,656/0,551 citati da Claude appartengono a TimeRLM-text **GPT-5.5 depth-1**, non al Qwen3.5-4B. Per Qwen3.5-4B RL la tabella riporta 0,538 localize e 0,244 all_channels. La fonte distingue chiaramente i 200/100 esempi test dai conteggi della nostra validation. [TimeRLM, tabella 1](https://arxiv.org/html/2608.03391v1#S4.T1).

**6. Valutazione puntuale della proposta finale di Claude**

| Proposta | Giudizio | Correzione necessaria |
|---|---|---|
| Finestre strette 1024–2048 | Direzione sensata | Prima controlli sul contenuto; jitter reale del centro e ampiezza dichiarata, evitando gold sempre centrato |
| Rispondere senza zoom su magnitudo, lead_lag e assenza | Sensato come curriculum iniziale | Imparare il ramo diretto, mantenere capacità originali e misurare falsi negativi; l'assenza è una label disponibile nel train, non un'informazione nota in inferenza |
| Due zoom grossolano→fine | Ipotesi plausibile, non ancora validata | Pilot chiuso in cui la seconda azione dipende dalla prima osservazione; consentire salto di regione/canale; non aggiungerlo insieme a tutte le altre modifiche |
| 20% crop sbagliato + risposta gold | Non sufficiente per insegnare recupero | Serve una traiettoria di correzione o una risposta sostenibile dagli originali; altrimenti si insegna a rispondere ignorando il crop |
| GRPO e test a quattro bracci | Corretto dopo i gate | Metriche di localizzazione, mantenimento del candidato SFT fra i selezionabili, confronto diretto SFT-zoom vs RL-zoom e controllo a pesi identici |

Il 61,07% euristico è reale e viene dal fallback. Non dimostra però che nelle categorie dotate di intervallo l'euristica fosse al 61%, né identifica da solo la causa del fallimento. Il contratto attuale forza un'azione per ogni esempio e non insegna alcun episodio di risposta diretta; questo è un problema distinto e più evidente.

Il calo su lead_lag con crop monocanale non dimostra che lo zoom tolga fisicamente il contesto: il codice conserva entrambi i canali originali e aggiunge il blocco. Sono possibili interferenza e cambio del comportamento di risposta. Escluderlo dal training dello zoom è una scelta operativa ragionevole; non una conclusione generale sull'inutilità di zoom sincronizzati su due canali.

**7. Continuazione concreta che consiglio**

1. **Diagnostica sul checkpoint vz1 già esistente.** Usare un sottoinsieme stratificato e fissato prima dell'esperimento, comprendente categorie, lunghezze, positivi/negativi e multicanale. Evitare le prime 80 righe, che non coprono tutte le categorie. Confrontare crop oracle corretto, crop sostituito a header identico, oracle con posizione interna randomizzata e, con uguale larghezza, proposte senza gold. Registrare IoU sui positivi, negativi corretti, F1 sugli esempi multicanale non vuoti e copertura per canale. Questo è il gate prima di un nuovo training lungo.
2. **Pilot Z1-bis con una sola azione opzionale.** Partire nuovamente da va1 per un confronto pulito. Generare tracce train-only con posizioni interne variabili, ramo diretto e quota di replay del formato originale. Usare sorgenti e percentuali esplicite per categoria, evitando fallback silenziosi. Salvare checkpoint intermedi con una regola di selezione fissata. Non riaprire contemporaneamente v3, encoder o RoPE. Misurare nel formato operativo anche la risposta diretta senza crop: l'attuale eval no_zoom usa il vecchio prompt, quindi non certifica il ramo diretto del nuovo protocollo.
3. **Se il pilot non centra bene, separare proposta e lettura.** Confrontare proposte della policy e di un selettore wavelet deterministico senza gold, a uguale budget. Il selettore può restituire solo coordinate all'ambiente, lasciando il blocco osservato esclusivamente in embedding, coerentemente con il vincolo della chat. Se il selettore funziona e la policy no, il problema è il controllo. Provare il secondo zoom solo dopo aver misurato che una rilettura grossolana migliora la proposta fine. Annotare chiaramente l'eventuale assistenza del selettore esterno.
4. **GRPO soltanto con un comportamento utile da affinare.** Verificare diversità delle azioni e varianza di reward fra finestre, non soltanto fra risposte. Conservare lo scorer ufficiale, riportando separatamente le metriche diagnostiche. Non selezionare per il solo aggregato che lead_lag può gonfiare. Congelare un criterio focalizzato su localize/all_channels con vincoli di non regressione delle altre categorie. Includere step 0 = checkpoint SFT: se RL peggiora, il risultato corretto è mantenere SFT.
5. **Confronto finale.** Conservare i quattro bracci già previsti, aggiungendo il confronto diretto zoom-SFT vs zoom-RL e il controllo dello stesso checkpoint con/senza informazione del crop. Riportare costo, token, azioni e accuratezza: i quattro bracci da soli confondono anche il maggiore SFT del ramo zoom. Le letture test di va1 e del controllo RL sono già storiche; non chiamare il test intero «mai visto». Dopo il congelamento del disegno, un ulteriore set indipendente con nuovi seed di generazione renderebbe più forte la conferma.

Le circa 30 ore non sono ancora un budget validato per un episodio a tre generazioni: l'attuale kit supporta una sola azione. Bisogna estendere storico delle osservazioni, maschere dei log-prob e checkpoint del formato, e rimisurare memoria e tempo. La prossima spesa sensata è la diagnostica a pesi fissi, non tutta la catena SFT+RL+test.

La formulazione scientifica sostenuta oggi è: **«Il crop guidato da un oracle migliora molto i risultati rispetto alla lettura globale; restano da isolare il contributo del contenuto percettivo e la capacità di selezionare autonomamente una finestra utile.»** È una domanda di ricerca concreta per proseguire L3. Il claim di una soglia percettiva già misurata o di equivalenza con TimeRLM è prematuro.

**8. Artefatti e riproducibilità di questa revisione**

- Script CPU: `l3/review/audit_zoom_20260912.py`.
- Output: `l3/review/AUDIT-ZOOM-20260912.json`, con hash dei tre JSONL, verifica dell'appaiamento di 520 ID/categorie/gold, bootstrap 10000 repliche e copertura corretta per canale.
- Esecuzione: `py l3/review/audit_zoom_20260912.py` dalla repository; non richiede checkpoint, scorer upstream o GPU.
- Ho anche rieseguito `lvl0/score.py`; i numeri del pilot qui riportati vengono dal ricalcolo.
- I file operativi del progetto e le modifiche L2 preesistenti non sono stati modificati. Non è stata rieseguita la suite completa del runtime GPU.

Note documentali da sistemare in un successivo intervento: README ancora con «repliche in corso» e «prossimo L2»; riferimenti L4/punto4 facilmente ambigui; conclusioni troppo forti in `RISULTATI-L4-ZOOM.md`; confronto con TimeRLM attribuito al backbone sbagliato. Questa revisione resta separata dai risultati storici per renderne esplicita la provenienza.
