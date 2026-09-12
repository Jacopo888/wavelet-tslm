**Revisione del gate crop e decisione sullo Z1-bis — 12/09/2026**

Ho verificato il commit `c28007f`, `diag_crop_content.py` e le 832 righe del gate (208 ID × 4 condizioni). Le schermate di Claude sono materiale da valutare, non un'autorizzazione ad avviare training. Nessuna run è stata lanciata.

**Verdetto.** Ritirare lo score oracle come dimostrazione della qualità percettiva è corretto. Non è invece corretto concludere che i token relativi rendano necessario modificare l'encoder o eliminare le coordinate. Consiglio un pilot Z1-bis con encoder invariato, coordinate veritiere e posizione interna realmente variabile; prima va corretto il confronto della baseline.

**1. Cosa dimostra il gate**

| Confronto sui medesimi 208 ID | Delta primary | Bootstrap appaiato 95%, 10000 repliche |
|---|---:|---|
| Crop vero − sostituito | +0,0063 | [−0,0154; +0,0302] |
| Crop vero − azzerato | −0,0028 | [−0,0264; +0,0216] |
| Crop vero centrato − vero fuori centro | +0,3566 | [+0,3111; +0,4029] |

Sono confronti descrittivi post-hoc, senza correzione per confronti multipli. Nessuna soglia di equivalenza è stata fissata: assenza di un vantaggio statisticamente distinguibile non è prova di effetto esattamente nullo.

La lettura supportata è: **«Nel checkpoint vz1 e su questo gate, il crop corretto non mostra un beneficio medio rilevabile rispetto a contenuti sostituiti o azzerati; la prestazione dipende fortemente dalla relazione fra header e posizione gold.»**

«Il modello non legge il contenuto» è troppo assoluto. Le predizioni cambiano in 170/208 esempi con swap e 178/208 con zero; il punteggio cambia rispettivamente in 116 e 128. Il contenuto influenza il comportamento, ma il beneficio netto misurato è vicino a zero. Questo è compatibile con una forte scorciatoia del centro, senza identificare ogni uso interno degli embedding.

Il gate conserva gli originali: i controlli non sono una valutazione senza alcun segnale. Il significato causale è il contributo aggiuntivo del blocco zoom condizionato agli originali e all'header. I veri header sono identici fra true/swap/zero; offcenter cambia legittimamente sia l'header sia il crop.

**2. Il confronto 0,123 contro 0,301 è scorretto**

0,1226 riguarda 208 positivi (80 localize, 80 classify, 48 multicanale); 0,3011 riguarda il checkpoint vz1 su tutte le 520 righe, inclusi negativi, magnitudo e lead_lag. Non si può dire che il primo risultato sia peggiore del modello senza zoom.

Come controllo disponibile, ho filtrato il JSONL va1 senza zoom sugli stessi ID:

| Sugli stessi ID del gate | va1 senza zoom | vz1 con oracle fuori centro |
|---|---:|---:|
| Totale 208 | 0,0612 | 0,1226 |
| Localize, 80 | 0,0694 | 0,1086 |
| Classify, 80 | 0,0673 | 0,0902 |
| Multicanale, 48 | 0,0375 | 0,2000 |

Il delta complessivo è +0,0614, CI95 descrittivo [+0,0237; +0,0996]. **Non dimostra un beneficio del contenuto**: cambia anche il checkpoint e il crop è ancora scelto usando il gold. Mostra perché la baseline deve essere appaiata. Occorre recuperare o calcolare vz1 senza zoom sugli stessi 208 ID; 0,301 non è una soglia valida per questo gate.

Il conteggio iniziale 282 del summary è il sottoinsieme selezionato; 74 righe senza intervallo sono saltate e non entrano negli score delle condizioni. Per il nuovo pilot, tenere distinti il gate percettivo sui positivi e la valutazione operativa completa, che include anche negativi e ramo diretto.

**3. Non c'è il bivio architetturale indicato da Claude**

Una posizione relativa nel crop può essere convertita in assoluta tramite un header veritiero:

`t_assoluto = start_crop + u_relativo × (end_crop − start_crop)`.

Le coordinate della finestra sono metadati necessari alla registrazione temporale, non necessariamente una fuga della risposta. La fuga nasce dalla correlazione artificiale «centro del crop = centro gold», non dalla presenza dell'origine temporale. Eliminare tale correlazione permette di conservare encoder, header e formato delle risposte globali.

La frase «posizioni relative perché normalizza nel blocco» va distinta: la normalizzazione MAD riguarda i valori; le posizioni temporali sono normalizzate separatamente dalla lunghezza del blocco. Sono scelte diverse, nessuna delle quali impone da sola di nascondere l'header.

La scelta minima consigliata è **va1 + encoder congelato + header veritiero + gold in posizione interna variabile + risposta assoluta**. Non servono v2, RoPE temporale o modifiche a v3 per verificare questa ipotesi. Non è una garanzia che un solo SFT basti a far apprendere la trasformazione, ma evita di confondere insieme più modifiche.

Una risposta in coordinate locali con conversione deterministica dell'ambiente resta una buona ablazione successiva, utile per separare lettura del segnale e aritmetica. Richiede però un contratto esplicito: nel multicanale la risposta globale può includere eventi letti dagli originali, quindi non si può aggiungere indiscriminatamente l'offset del crop a tutti gli intervalli. Inoltre togliere le coordinate dal solo header non le nasconde se rimangono nel testo dell'azione precedente. Per una prova crop-only va isolata tutta la vista locale. Non la introdurrei adesso come modifica obbligatoria del protocollo principale.

**4. Cosa mettere nel brief Z1-bis**

1. Pilot limitato, con ripartenza da va1, encoder/proiettore congelati, una sola azione opzionale. Separare il gate di lettura con crop forzato dal gate della policy che seleziona il crop senza gold.
2. Sugli esempi positivi, scegliere ampiezza e posizione interna con regole fissate prima del run. Estrarre start tra quelli che contengono l'evento completo, entro i bordi della serie. Verificare a posteriori la distribuzione della posizione interna, anche per lunghezza e categoria; mantenere alcuni casi centrati senza renderli dominanti.
3. Includere risposte dirette e replay del formato originale. Niente «crop vuoto → risposta globale gold» come presunto recupero: quando la vista è insufficiente, la traiettoria deve mostrare come usare gli originali o compiere una correzione.
4. Dopo il pilot, valutare lo stesso checkpoint sui medesimi ID nelle condizioni **fuori centro vero / fuori centro swap / fuori centro zero / risposta senza crop**. True/swap/zero devono avere header identici dentro questo nuovo confronto. Il solo miglioramento su un oracle fuori centro non dimostra ancora utilità dei coefficienti: canale e regione sono comunque informazioni privilegiate.
5. Criterio percettivo: guadagno appaiato del crop corretto rispetto ai controlli a header identico, con una soglia pratica e budget campionario fissati prima del run; metriche per categoria, non soltanto media aggregata. Criterio operativo separato: policy autonoma sulla stessa validation e nello stesso formato, rispetto ai riferimenti appaiati, con controllo dei negativi e del ramo diretto. Nessuna soglia 0,301 riciclata sul sottoinsieme dei positivi.
6. Non concatenare automaticamente due zoom, RL e test. Se il pilot fallisce il gate percettivo, indagare rappresentazione, interfaccia o training; se passa quello percettivo ma fallisce quello operativo, lavorare sul selettore della finestra.

Il gate appena eseguito è utile perché restringe la domanda sperimentale. Non rende obbligatoria una run completa di Z1-bis né dimostra che l'architettura wavelet non possa leggere un crop. Giustifica un pilot mirato che insegni e misuri la lettura senza il centro oracle come risposta implicita.

Ricalcolo riproducibile: `py l3/review/audit_crop_gate_20260912.py`; output `l3/review/AUDIT-CROP-GATE-20260912.json`. Il codice operativo e i report storici non sono stati modificati.
