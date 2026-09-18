# va1r (16/09/2026): l'header risponde da solo a una categoria su cinque

Risultati della run `va1r`, il cui disegno è in [PIANO-VA1R-20260916.md](PIANO-VA1R-20260916.md). Documento in
aggiornamento: la run è in corso.

## Epoca 1: il modello impara il formato, non il segnale

Validazione rapida su 104 righe stratificate al passo 4488, con **zero formati invalidi** su 104.

| categoria | primary |
|---|---:|
| measure_magnitude | **0,7546** |
| localize_all_channels | 0,2667 |
| localize | 0,0000 |
| classify_with_evidence | 0,0000 |
| lead_lag_with_magnitude | 0,0000 |
| **complessivo** | **0,2271** |

Il formato è imparato perfettamente alla prima epoca. Il punteggio complessivo, però, viene quasi tutto da una sola
categoria.

## Il confronto con va1, sulle stesse identiche righe

Gli artefatti `integrity/artifacts/l3_riferimenti/` conservano le predizioni di `va1` riga per riga sulle 520
registrazioni di validazione. Ho quindi potuto fare il confronto che Astra chiedeva: **le stesse 104 righe**, abbinate
per identificativo, con bootstrap appaiato (`tools/compare_with_va1.py`).

| categoria | n | `va1` | `va1r` ep. 1 | Δ | CI95 |
|---|---:|---:|---:|---:|---|
| classify_with_evidence | 22 | 0,0820 | 0,0000 | −0,0820 | [−0,171; −0,005] |
| lead_lag_with_magnitude | 15 | 0,2854 | 0,0000 | −0,2854 | [−0,513; −0,085] |
| localize | 26 | 0,2725 | 0,0000 | −0,2725 | [−0,462; −0,115] |
| localize_all_channels | 15 | 0,2667 | 0,2667 | **+0,0000** | **[0,0000; 0,0000]** |
| measure_magnitude | 26 | 0,7533 | 0,7546 | +0,0013 | [−0,138; +0,137] |
| **complessivo** | 104 | 0,3534 | 0,2271 | −0,1263 | [−0,197; −0,060] |

La prima lettura che avevo dato — «`va1r` eguaglia o supera `va1`» — era sbagliata, perché confrontava 104 righe di
validazione con 800 di test. Alla pari, dopo una epoca `va1r` è sotto di 0,126 complessivi.

Ma la riga che conta è `localize_all_channels`: **differenza esattamente zero con intervallo di ampiezza zero**, cioè i
due modelli ottengono lo stesso punteggio su *ogni singola riga*. Non è una media che coincide: è lo stesso
comportamento.

## Il confronto giusto: epoca 1 contro epoca 1

Il confronto appaiato qui sopra mette `va1r` dopo **una** epoca contro `va1` dopo **cinque**. `vultr/RISULTATI-VA1.md`
conserva però la validazione di `va1` epoca per epoca, sulle stesse 520 righe:

| epoca | primary | localize | classify | magnitudo | multicanale | lead_lag |
|---|---:|---:|---:|---:|---:|---:|
| **1** | **0,1703** | **0,000** | **0,000** | 0,378 | 0,167 | 0,360 |
| 2 | 0,2257 | 0,197 | 0,023 | 0,418 | 0,153 | 0,308 |
| 3 | 0,2482 | 0,201 | 0,013 | 0,483 | 0,304 | 0,213 |
| 4 | 0,2925 | 0,232 | 0,048 | 0,557 | 0,355 | 0,237 |
| 5 | 0,3105 | 0,244 | 0,090 | 0,578 | 0,358 | 0,238 |

Alla stessa altezza dell'addestramento i numeri stanno così, e vanno letti come indicativi e non
come una dichiarazione di vantaggio, perché le due colonne hanno popolazioni diverse:

| | `va1` ep. 1 (520 righe) | `va1r` ep. 1 (104 righe) |
|---|---:|---:|
| primary | 0,1703 | **0,2271** |
| localize | 0,000 | 0,000 |
| classify | 0,000 | 0,000 |
| magnitudo | 0,378 | **0,7546** |
| multicanale | 0,167 | **0,2667** |
| lead_lag | **0,360** | 0,000 |

Lo zero di `va1r` su `localize` e `classify` **non è un'anomalia: è esattamente quello che faceva `va1`** allo stesso
punto. Quelle due categorie in `va1` compaiono fra la prima e la seconda epoca, e da lì salgono fino in fondo. Il
confronto resta **val104 contro val520**, e su questo Astra è stata esplicita: non si dichiara che `va1r`
sia avanti. Quello che si può dire è che gli zeri coincidono e che la differenza sulla magnitudo è troppo
grande per essere campionamento.

Ne segue una previsione verificabile, e a breve: **se `va1r` segue la traiettoria di `va1`, la validazione della
seconda epoca deve staccare `localize` da zero.** È il primo vero test della run, e arriva senza spendere nulla.

Resta il buco su `lead_lag`, dove `va1` faceva 0,360 alla prima epoca e `va1r` fa zero: è l'unica categoria in cui la
nuova architettura parte indietro, e va tenuta d'occhio.

## Che cosa misurava davvero il punteggio di va1

Quella coincidenza chiedeva una spiegazione, e la spiegazione ribalta la lettura di tutto il progetto. Scomponendo il
punteggio di `va1` sulle 520 righe di validazione fra registrazioni **con** e **senza** anomalia:

| categoria | metrica | totale | con anomalia | senza anomalia | quota da assenza |
|---|---:|---:|---:|---:|
| localize | IoU o vuoto corretto | 0,2438 | **0,0606** | 0,8667 | **80,8 %** |
| localize_all_channels | F1 sui canali | 0,3583 | **0,0375** | 1,0000 | **93,0 %** |
| classify_with_evidence | tipo × IoU | 0,0902 | 0,0902 | — | 0 % |

Sulle 15 righe multicanale del sottoinsieme da 104 i punteggi sono binari, zero o uno, e valgono uno **solo** quando la
risposta giusta è la lista vuota — ed è per questo che i due modelli coincidono riga per riga. **Quella osservazione
vale però solo su quelle 15 righe**, come Astra ha giustamente obiettato: sulle 520 complete `va1` supera lo zero su
**4 delle 48 righe multicanale con anomalia** (0,333, 0,400, 0,400, 0,667), e in quei casi i canali indicati sono
quelli giusti con finestre vicine a quelle vere. Non «mai», quindi: molto di rado. Su `localize`, quattro quinti del
punteggio di `va1` è dire «non c'è niente» quando non c'è niente.

**La capacità percettiva vera di `va1` sul compito diretto è 0,061 di IoU su localize e 0,038 di F1 sul multicanale.**
Sommata
a `measure_magnitude`, che si risponde dall'header, e a `classify` che vale 0,09, resta ben poco. Il riferimento che
inseguiamo da giorni non è mai stato una dimostrazione di percezione nella vista globale: la dimostrazione, +0,1349,
è arrivata da Z1-bis, cioè **nella vista ritagliata**.

Va aggiunto, sempre seguendo Astra, che riconoscere l'assenza non è di per sé una scorciatoia: è parte del compito. Quello che i numeri dicono è che nella vista globale il punteggio è dominato da quella componente, non che sia illegittima. Per questo il tasso di falsi positivi sui negativi va riportato accanto alla prestazione sui positivi.

Quanto a `va1r`, il divario su `localize` ha una causa precisa e rimediabile: all'epoca 1 **non dice mai «assente»**,
zero volte su 26 righe di cui 8 senza anomalia. Sul multicanale ha già imparato a rispondere con la lista vuota, sei
volte su quindici, ed è esattamente per questo che lì eguaglia `va1`. Non gli manca la percezione che `va1` aveva: gli
manca ancora la scorciatoia che `va1` aveva.

## La scoperta: l'header contiene la risposta di `measure_magnitude`

Il prompt dichiara, per ogni canale, `median`, `scale`, `min`, `max` (`ConversationBuilder.global_pieces`).
`measure_magnitude` chiede `magnitude_sigma`, cioè l'ampiezza dell'anomalia in unità di deviazione robusta.

Su 120 righe di validazione, la formula **scarto massimo dalla mediana diviso la scala robusta**, che si calcola
interamente dai numeri stampati nell'header, riproduce il valore atteso con **rapporto mediano 1,158** e Pearson 0,539
in scala logaritmica. Le prime stime contro i valori veri: 37,4 contro 35,97; 4,5 contro 1,97; 73,3 contro 59,6;
26,4 contro 21,3.

**Quella categoria si risponde senza leggere un solo token di segnale.** E dà 0,7546 di primary, cioè da sola quasi
tutto lo 0,2271 complessivo.

### Quanto in alto arriva la scorciatoia

La stessa statistica dell'header non spiega tutto il resto, e vale la pena essere precisi su quanto spiega.

Su `localize`, che chiede se l'anomalia è presente e dove, la statistica dell'header separa i presenti dagli assenti
con **AUC 0,76**: informativa, non decisiva. E la soglia migliore dà accuratezza 0,7955 contro un tasso di base di
0,7727, cioè **due punti percentuali sopra il classificatore banale che risponde sempre «presente»**. La posizione
della finestra, che è il grosso del punteggio di quella categoria, nell'header non c'è.

Quindi la scorciatoia è **decisiva per `measure_magnitude` e marginale per `localize`**. Le tre categorie del gate
restano una misura di percezione legittima; il primary complessivo no.

## Conferma dal probe

Probe sulle cinque condizioni al checkpoint 4488, quattro righe, solo come validazione dello strumento:

| condizione | CE finale | primary |
|---|---:|---:|
| real | 0,4901 | 0,2334 |
| empty | 0,4926 | **0,2364** |
| substituted | 0,4987 | 0,2335 |
| oracle | 0,4943 | 0,2340 |
| permuted | 0,4925 | 0,2334 |

Dipendenza dal contenuto, primary vero meno sostituito: **−0,0000034**. Azzerare i token non peggiora il punteggio: lo
migliora di un soffio. Il modello, a questo punto dell'addestramento, ignora completamente il segnale e risponde con
quello che trova nel testo.

Quattro righe non dimostrano nulla di statistico, ma l'uguaglianza del primary a tre decimali su cinque condizioni
diverse non è rumore: è una risposta che non dipende dall'ingresso.

### Che cosa risponde davvero il modello

Le predizioni al passo 4488 sono strutturalmente perfette e posizionalmente arbitrarie:

```
localize:{"_L": 32768,  "end": 11061,  "present": true, "start": 10914}     primary 0,00
localize:{"_L": 131072, "end": 102161, "present": true, "start": 102010}    primary 0,00
classify_with_evidence:{"kind": "Change in Variance", "start": 99800, "end": 100000}   primary 0,00
localize_all_channels:{"anomalies": []}                                     primary 1,00
measure_magnitude:{"magnitude_sigma": 2.011114}                             primary 0,81
lead_lag_with_magnitude:{"direction": "lag", "lag_samples": 10000}          primary 0,86
```

Le finestre hanno larghezza quasi costante — 147 e 151 campioni — e vengono piazzate da qualche parte. Il modello ha
imparato la **forma** della risposta, non dove guardare. Lo zero su `localize` non è un problema di formato: è che la
posizione è un'ipotesi a priori. E l'1,00 su `localize_all_channels` viene da una riga senza anomalie, dove la lista
vuota è la risposta giusta: di nuovo la rilevazione dell'assenza.

Il gate su sei righe conferma il quadro: sulle tre categorie percettive la differenza vero-sostituito è **esattamente
+0,0000** su tutte e quattro le righe utili, e azzerare i token **migliora** il primary complessivo di 0,030. Allo
stato attuale i soft token sono rumore per il backbone.

### L'ago nel pagliaio

Sulle 909 righe `localize` del train con anomalia presente, la finestra vera ha larghezza mediana **137 campioni**,
quartili 76 e 195, massimo 256. Il modello emette 147 e 151: **la larghezza l'ha imparata**, gli manca solo dove
metterla.

Il rapporto fra larghezza e lunghezza della registrazione ha mediana **0,0026**.

**Correzione di Astra, verificata.** Avevo scritto «quattro token su 4.096»: è sbagliato. I 4.096 sono solo il tetto
`max_global_ts_tokens`; la vista globale produce **256 token per canale**, cioè 64 di approssimazione, 64 di copertura
dei dettagli, 64 di dettaglio saliente e 64 di grezzo, e la selezione non è uniforme ma per magnitudine robusta di
banda su 32 bin di salienza.

Il conto corretto è più duro sul ramo di copertura e più incoraggiante sul resto. I 64 token di approssimazione
coprono la registrazione uniformemente: su 131.072 campioni ciascuno ne vale 2.048, quindi un'anomalia da 137 campioni
è **un quindicesimo di un singolo token**, sotto la sua risoluzione. Ma il ramo saliente esiste proprio per questo: i
suoi 64 token vengono scelti dove la banda ha magnitudine anomala, cioè in linea di principio sull'anomalia.

Quindi la domanda non è se il budget di token basti in astratto, ma **se la selezione saliente cada davvero sulla
finestra vera**. È una domanda empirica e verificabile, e resta la cosa più informativa da misurare: se i token
salienti non coprono l'anomalia il modello non può localizzarla, e se la coprono il problema è a valle.

Resta plausibile che nella vista globale la localizzazione sia intrinsecamente difficile — è coerente con il fatto
che il 68 % dello 0,241 di `va1` su localize fosse rilevazione dell'assenza, e con il motivo per cui lo zoom di
Z1-bis è stato inventato — ma il conto dei token da solo **non lo dimostra**, come Astra ha giustamente osservato.

### La selezione saliente cade sull'anomalia: misurato

La domanda lasciata aperta è verificabile senza checkpoint, perché la selezione non è appresa e la piramide di Haar è
una trasformata fissa. `tools/salience_coverage.py`, su 100 righe `localize` con anomalia presente e larghezza mediana
144,5 campioni:

| famiglia | colpisce la finestra | IoU del token migliore (mediana) | copertura |
|---|---:|---:|---:|
| A (approssimazione) | 1,000 | 0,131 | 1,000 |
| D copertura | 1,000 | 0,131 | 1,000 |
| **D saliente** | **0,930** | **0,213** | 0,558 |
| grezzo | 1,000 | 0,131 | 1,000 |

Le famiglie uniformi colpiscono sempre, ovviamente, perché piastrellano l'intera registrazione: il loro IoU 0,131 è
semplicemente il rapporto fra la larghezza dell'anomalia e quella di un bin. La famiglia saliente invece **sceglie**, e
nel **93 % delle righe** almeno un suo token si sovrappone alla finestra vera, con un IoU del token migliore di 0,213,
sensibilmente più alto di qualunque token uniforme.

**L'informazione è presente fra i candidati.** Il frontend, senza alcun addestramento, produce 64 token salienti
fra i quali almeno uno copre l'anomalia in nove casi su dieci. Il collo di bottiglia non sembra quindi la
sensibilità percettiva del tokenizer.

Due precisazioni di Astra, entrambe giuste, che limitano ciò che il numero autorizza a dire. Il **93 % è copertura
fra candidati, non accuratezza della finestra effettivamente scelta**: dice che l'informazione è fra le opzioni, non
che il modello sappia individuarla. E lo **0,213 è il massimo su 64 token**, quindi né il valore che un lettore reale
otterrebbe, né il tetto di un lettore che combinasse più token raffinandone i confini — potrebbe essere più alto.

Resta però il confronto di ordine di grandezza: `va1` su tutta la categoria faceva 0,241, di cui il 68 % era
rilevazione dell'assenza. Anche letta con tutte le cautele, la copertura dei candidati salienti è nella stessa
zona del meglio che questo benchmark abbia prodotto in vista globale, e questo rende la scelta della finestra —
non la sua disponibilità — il problema da attaccare.

## Epoca 2: la previsione si avvera, ma non nel modo che conta

La previsione era che `localize` dovesse staccarsi da zero, come in `va1` fra la prima e la seconda epoca. Si è
avverata quasi alla cifra: **0,000 → 0,1923**, contro lo 0,000 → 0,197 di `va1`. Il primary complessivo sale da
0,2271 a 0,2642 e il divario da `va1` finale si stringe da −0,126 a −0,089.

Poi si guarda la scomposizione che Astra ha imposto, e il quadro cambia:

| categoria | totale | **sui positivi** | sui negativi | specificità | falsi «vuoto» |
|---|---:|---:|---:|---:|---:|
| localize | 0,1923 | **0,0000** | 0,6250 | 0,625 | 0,000 |
| localize_all_channels | 0,2000 | **0,0000** | 0,7500 | 0,750 | 0,000 |
| classify_with_evidence | 0,0137 | 0,0137 | — | — | — |
| measure_magnitude | 0,7377 | 0,7377 | — | — | — |

**Tutto il salto è rilevazione dell'assenza.** Alla prima epoca il modello non diceva mai «assente», zero volte su
otto righe vuote; alla seconda lo dice cinque volte su otto. L'IoU sulle righe **con** anomalia resta esattamente
zero, come alla prima epoca.

Una nota a favore, però: quella rilevazione è **selettiva**, non cieca. Falsi «vuoto» sulle righe con anomalia:
**0 su 18**. Il modello non ha imparato a rispondere «non c'è niente» per default, ha imparato a distinguere — e per
distinguere deve leggere il segnale. È la stessa abilità vera ma facile che `va1` aveva.

Restano tre epoche. Il bersaglio da battere è l'IoU sui positivi di `va1`, che in validazione vale **0,0606**. Se
`va1r` arriva a fine corsa con i positivi ancora a zero, la conclusione non sarà «la ricetta non si replica» — il
composito la sta replicando — ma «la ricetta replica ciò che `va1` sapeva fare, cioè rilevare e non puntare».

`tools/compare_with_va1.py` ora stampa questa scomposizione accanto al confronto appaiato, a ogni epoca.

## Epoca 3: il composito raggiunge va1, il puntamento resta a zero

| | epoca 1 | epoca 2 | epoca 3 | epoca 4 | `va1` finale |
|---|---:|---:|---:|---:|---:|
| primary | 0,2271 | 0,2642 | 0,3042 | **0,3128** | 0,3534 |
| localize | 0,000 | 0,1923 | 0,2692 | **0,2692** | 0,2725 |
| **localize sui positivi** | **0,000** | **0,000** | **0,000** | **0,000** | 0,0606 |
| specificità | 0/8 | 5/8 | 7/8 | **7/8** | — |
| multicanale | 0,2667 | 0,2000 | 0,2667 | 0,2667 | 0,2667 |
| multicanale sui positivi | 0,000 | 0,000 | 0,000 | **0,000** | 0,0375 |

`localize` ha raggiunto il valore finale di `va1` sulle stesse righe — 0,2692 contro 0,2725, differenza −0,0032 con
intervallo [−0,0097; 0,0000] — e il divario complessivo si è ridotto a −0,049. Sul multicanale la differenza è di
nuovo **esattamente zero con intervallo di ampiezza zero**.

**Ma la curva di apprendimento di `localize` è, riga per riga, la curva della specificità**: 0 su 8, poi 5 su 8, poi
7 su 8. L'IoU sulle righe con anomalia è rimasto **0,0000 per tre epoche intere**, cioè per 13.464 aggiornamenti.
Compare anche il primo errore nell'altra direzione: un falso «assente» su 18 positivi, prima erano zero.

Questo isola finalmente una **differenza vera fra `va1` e `va1r`**, che il composito nascondeva. `va1` chiudeva con
0,0606 di IoU sui positivi e 0,0375 di F1 multicanale: poco, ma non nulla. `va1r`, al punto in cui ha già eguagliato
il composito di `va1`, sta a zero su entrambi. Non è quindi vero che sta replicando `va1`: sta replicando **la metà
facile** di `va1`, e con più efficienza, mentre la metà difficile non è ancora comparsa.

All'epoca 4 il quadro non cambia: il primary sale a 0,3128 e la magnitudo a 0,7717, che supera `va1` di
0,0184, il divario complessivo scende a −0,041 con intervallo che ora attraversa lo zero — ma i positivi
restano a 0,0000 per la quarta epoca consecutiva, su entrambe le categorie.

Resta una epoca perché compaia. Se non compare, la lettura non è «la ricetta non si replica» ma «questa
architettura, con questa ricetta, impara a rilevare meglio di `va1` e a puntare per niente» — ed è l'argomento più
forte possibile per il braccio con lo zoom, che è precisamente ciò che in `va1` aveva fatto salire l'IoU sui positivi
da 0,094 a 0,248.

## Il gate sui 208 positivi all'epoca 3: nessuna dipendenza dal contenuto

Pannello dei 208 identificativi di Z1-bis — 80 localize, 80 classify, 48 multicanale, tutti con anomalia presente —
tre condizioni necessarie, 624 generazioni, 37 minuti. Soglia dichiarata prima: Δ ≥ +0,05 con CI95 appaiato sopra lo
zero **contro entrambi** i controlli.

| condizione | primary | cross-entropy |
|---|---:|---:|
| token veri | 0,0019 | 1,1080 |
| token azzerati | **0,0072** | 1,1083 |
| token di un'altra registrazione | 0,0057 | 1,1076 |

| differenza appaiata | valore | CI95 |
|---|---:|---|
| primary vero − sostituito | **−0,0037** | [−0,0100; +0,0001] |
| primary vero − azzerato | **−0,0052** | [−0,0151; +0,0018] |
| CE sostituito − vero | −0,0003 | [−0,0015; +0,0009] |

**Verdetto: FAIL**, e non di misura. L'effetto non è piccolo e positivo: è **leggermente negativo**. Azzerare i soft
token rende il modello marginalmente *migliore* di quanto lo renda dargli quelli veri, su tutte e tre le categorie.
Per categoria, con i token veri: localize 0,0000, multicanale 0,0000, classify 0,0050.

Il risultato è coerente con la scomposizione della validazione e la rende definitiva su un pannello quattro volte più
grande. Su registrazioni che contengono davvero un'anomalia, `va1r` alla terza epoca vale praticamente zero, e la sua
risposta **non cambia** se il segnale viene sostituito o azzerato. Per il compito, i soft token sono rumore.

Va detto con precisione che cosa questo dimostra e che cosa no. Dimostra che a questo punto dell'addestramento non
esiste dipendenza dal contenuto misurabile con il disegno pre-registrato. Non dimostra che due epoche in più non
possano produrla — anche se tre epoche di zero esatto la rendono improbabile — e non dice nulla sul comportamento in
vista ritagliata, che è la domanda diversa che Astra ha tenuto separata.

## Esito della run: replica riuscita sul composito, percezione mai comparsa

Cinque epoche, 22.440 aggiornamenti, completate alle 19:53 UTC del 16/09 senza interruzioni oltre l'incidente del
checkpoint alla prima epoca. Validazione finale su 104 righe, confronto appaiato con `va1` sulle stesse:

| categoria | n | `va1` | `va1r` | Δ | CI95 |
|---|---:|---:|---:|---:|---|
| measure_magnitude | 26 | 0,7533 | **0,7582** | +0,0049 | [−0,130; +0,136] |
| localize_all_channels | 15 | 0,2667 | 0,2667 | **+0,0000** | [0,0000; 0,0000] |
| localize | 26 | 0,2725 | 0,2692 | −0,0032 | [−0,0097; 0,0000] |
| lead_lag_with_magnitude | 15 | 0,2854 | 0,1454 | −0,1401 | [−0,388; +0,120] |
| classify_with_evidence | 22 | 0,0820 | 0,0130 | −0,0690 | [−0,164; +0,008] |
| **complessivo** | 104 | 0,3534 | **0,3190** | −0,0344 | **[−0,089; +0,018]** |

**Il confronto sul composito è inconcludente, non una dimostrazione di equivalenza.** Il divario è −0,034 con
intervallo [−0,089; +0,018]: attraversa lo zero, ma resta compatibile con una perdita rilevante. Astra è stata
esplicita su questo punto e la formulazione precedente era troppo forte. Sulla magnitudo lo supera, sul multicanale lo
eguaglia riga per riga, su localize gli arriva a tre millesimi.

**La localizzazione utile non è emersa sul pannello valutato.** Non «tutta la percezione resta zero»: `classify`
vale 0,0130 e la rilevazione dell'assenza è un'abilità vera, non una scorciatoia per definizione. La scomposizione
finale:

| categoria | totale | **sui positivi** | sui negativi | specificità | falsi «vuoto» |
|---|---:|---:|---:|---:|---:|
| localize | 0,2692 | **0,0000** | 0,8750 | 0,875 | 0,000 |
| localize_all_channels | 0,2667 | **0,0000** | 1,0000 | 1,000 | 0,000 |

Traiettoria completa della componente difficile, epoca per epoca:

| | ep. 1 | ep. 2 | ep. 3 | ep. 4 | ep. 5 | `va1` |
|---|---:|---:|---:|---:|---:|---:|
| primary | 0,2271 | 0,2642 | 0,3042 | 0,3128 | **0,3190** | 0,3534 |
| localize | 0,000 | 0,1923 | 0,2692 | 0,2692 | **0,2692** | 0,2725 |
| **localize sui positivi** | 0,000 | 0,000 | 0,000 | 0,000 | **0,000** | 0,0606 |
| specificità | 0/8 | 5/8 | 7/8 | 7/8 | **7/8** | — |

Cinque epoche, 22.440 aggiornamenti, e l'IoU sulle righe con anomalia è rimasto **esattamente zero** dall'inizio alla
fine, su entrambe le categorie di localizzazione.

**Correzione a una conclusione precedente.** Avevo scritto che `va1` chiudeva con 0,0606 di IoU sui positivi contro lo
zero di `va1r`: quel numero viene dalle 520 righe, mentre lo zero di `va1r` viene dalle 104. Sul confronto appaiato,
**stesse righe e soli positivi**, la differenza è molto più piccola:

| categoria | n positivi | `va1` | `va1r` | Δ |
|---|---:|---:|---:|---:|
| localize | 18 | 0,0047 | 0,0000 | −0,0047 |
| localize_all_channels | 11 | 0,0000 | 0,0000 | +0,0000 |
| classify_with_evidence | 22 | 0,0820 | 0,0130 | −0,0690 |
| **tre categorie** | 51 | 0,0370 | 0,0056 | −0,0314 |

Su queste 104 righe **anche `va1` è sostanzialmente a zero sul puntamento**: il divario misurabile sui positivi è
quasi tutto `classify`. Per una affermazione solida sul puntamento serve la validazione sulle 520, che è in corso.

## Che cosa conclude questa run

1. **L'architettura nuova impara il compito**, e la parte non percettiva la impara meglio di `va1`: magnitudo sopra,
   multicanale identico, specificità 7/8 senza un solo falso «assente», zero formati invalidi su 104 in tutte e cinque
   le validazioni.
2. **La dissociazione è netta e riproducibile**: tutto ciò che non richiede il segnale sale, tutto ciò che lo richiede
   resta a zero. Non è un fallimento generico della ricetta: è un fallimento localizzato in un punto preciso.
3. **`final = 0` non era la causa.** Questa run aveva la risposta finale a peso pieno, nessuna lettura dei bin, nessuna
   testa ausiliaria, e il risultato sui positivi è lo stesso zero delle run di P1. La diagnosi del verdetto di P1 va
   quindi corretta: il problema non stava nella scelta diagnostica di azzerare il finale.
4. **L'informazione è disponibile e non viene usata.** I token salienti coprono l'anomalia nel 93 % dei casi; il gate
   sui 208 all'epoca 3 mostra che azzerare i token non peggiora la risposta. Fra disponibilità e uso c'è il salto che
   questa architettura, con questa ricetta, non compie.

## Gate finale sui 208 positivi, checkpoint dell'epoca 5

| condizione | localize | classify | multicanale | CE |
|---|---:|---:|---:|---:|
| token veri | 0,0114 | 0,0036 | 0,0000 | 1,1330 |
| token azzerati | **0,0171** | **0,0114** | 0,0000 | 1,1347 |
| token di un'altra registrazione | 0,0064 | 0,0037 | 0,0000 | 1,1338 |

| differenza appaiata | valore | CI95 |
|---|---:|---|
| primary vero − sostituito | +0,0019 | [−0,0029; +0,0080] |
| primary vero − azzerato | −0,0052 | [−0,0125; +0,0001] |
| CE sostituito − vero | +0,0008 | [−0,0020; +0,0041] |
| CE azzerato − vero | +0,0017 | [−0,0026; +0,0067] |

**Verdetto: FAIL**, contro la soglia di +0,05 dichiarata prima.

Rispetto all'epoca 3 una quantità cambia segno, e va registrata senza gonfiarla — la prima versione di questo
paragrafo lo faceva, e Astra l'ha corretta. All'epoca 3 il confronto vero-contro-sostituito era **negativo**
(−0,0037); all'epoca 5 è **+0,0019**, cioè **26 volte sotto** la soglia di +0,05, con limite superiore dell'intervallo
a +0,008 che resta molto al di sotto dell'effetto richiesto. E non è vero che tutte le quantità puntano nella
direzione attesa: **il primary con i token azzerati resta migliore di quello con i token veri**, −0,0052. Tre
indicatori su quattro sono coerenti, uno no, e nessuno è significativo. Non è evidenza di dipendenza dal contenuto.

Resta il fatto che i token azzerati non peggiorano mai il primary — su localize e classify lo migliorano — e che sul
multicanale tutte e tre le condizioni valgono zero esatto.

## Il mini-test di memorizzazione: disegno, e due versioni scartate

Astra ha chiesto, prima di qualunque altra run lunga, il controllo piu' diretto: far memorizzare al sistema completo
ventiquattro segnali semplici con **header identici** e anomalie in posizioni diverse, sulla stessa GPU e nella stessa
precisione della run vera, con il controllo vero-contro-sostituito. `tools/memorization_probe.py`.

Il disegno ha una proprieta' che lo rende decisivo: **se i prompt sono identici, memorizzare e' gia' la prova di
lettura**, perche' il segnale e' l'unica cosa che distingue un esempio dall'altro. E un sistema che non riesce a
fittare ventiquattro esempi visti venticinque volte ciascuno ha un problema che nessuna quantita' di dati risolve.

Rendere i prompt identici e' pero' meno banale di quanto sembri, e le prime due versioni erano sbagliate.

1. **Fondo casuale indipendente per ogni esempio**: 24 prefissi distinti su 24. Mediana, scala, minimo e massimo
   cambiano con il rumore, quindi il modello poteva riconoscere l'esempio dal testo e richiamare la risposta senza
   guardare il segnale. Questa versione ha girato e dava 2 esempi risolti su 24 con delta +0,0417: **il risultato va
   scartato**, non perche' sbagliato in se', ma perche' non misurava quello che doveva.
2. **Fondo condiviso, picco inserito in posizioni diverse**: ancora 19 prefissi distinti su 24. Il mio ragionamento
   era che l'insieme dei valori si conservasse; e' falso, perche' il picco si somma a campioni di rumore diversi e le
   statistiche si separano lo stesso.
3. **Rotazione circolare di un unico segnale**: corretta. Ogni esempio e' `roll(base, start_i)`, quindi tutti
   contengono **esattamente lo stesso insieme di valori** in ordine diverso: mediana, scala robusta, minimo, massimo e
   conteggio dei validi coincidono alla cifra. Verificato dal tool, che ora **si ferma con errore** se i prefissi non
   sono identici invece di limitarsi ad avvisare.

Parametri: 24 registrazioni da 131.072 campioni, canale singolo, picco di Hann largo 137 campioni e ampiezza 8 sigma,
posizioni distribuite sull'intervallo utile, header ridotto, nessun blocco di ragionamento, 600 passi a lr 1e-4 su
frontend, mixer, proiettore e LoRA. Valutazione ogni 100 passi su tutti e 24 gli esempi, con e senza sostituzione del
segnale (donatore = esempio successivo, header e risposta del ricevente).

I due bracci partono dalla **stessa inizializzazione** — il seme e' fissato prima di costruire il modello — e
differiscono solo per `tokenizer.meta_proj_fp32`, che fa girare la proiezione dei metadati fuori dall'autocast bf16.

### Esito: nessuno dei due bracci memorizza, e la correzione numerica non cambia nulla

| passo | base, risolti | base, delta | fp32, risolti | fp32, delta |
|---:|---:|---:|---:|---:|
| 100 | 0/24 | +0,0000 | 0/24 | +0,0000 |
| 200 | 0/24 | +0,0000 | 0/24 | +0,0000 |
| 300 | 1/24 | +0,0000 | 0/24 | +0,0000 |
| 400 | 1/24 | +0,0000 | 0/24 | −0,0417 |
| 500 | 1/24 | +0,0000 | 1/24 | +0,0000 |
| **600** | **1/24** | **+0,0000** | **1/24** | **+0,0000** |

Entrambi **NOT MEMORISED**, con lo stesso identico punteggio finale. La loss scende da 0,30 a 0,084 nel braccio base
e da 0,37 a 0,090 nel braccio fp32, quindi qualcosa viene fittato — la forma della risposta, presumibilmente una
risposta pressoche' costante — ma non la posizione. E l'unico esempio «risolto» prende lo stesso punteggio quando gli
si mostra il segnale di un altro: non e' letto nemmeno quello.

**La perdita di precisione in bf16 non e' l'ostacolo.** E' reale e misurata — a campione 65.536 servono 257 campioni
perche' il valore cambi, a 131.000 ne servono 585, contro anomalie larghe 137 — ma correggerla non sposta il
risultato di un millesimo. Questo chiude l'ipotesi piu' concreta che avevamo.

Il quadro che ne esce, messo accanto a quello che sapevamo gia':

1. **L'informazione entra nei soft token.** Le due teste lineari di P1-e decodificavano la posizione con errore
   1,174 bin su registrazioni mai viste, con Pearson 0,767 sull'energia e 0,852 sul picco.
2. **Il sistema completo non lega 24 segnali a 24 risposte**, con prompt identici alla cifra, un picco di 8 sigma e
   venticinque passaggi per esempio.

La rottura sta quindi **fra i soft token e la risposta**, non prima. Resta da separare se l'informazione si perda
attraversando il backbone o se arrivi e non venga usata: sono due diagnosi diverse con rimedi diversi, e il controllo
che le distingue e' un probe sugli **embedding finali**, non sui soft token — quello sull'ingresso lo abbiamo gia'
fatto ed e' positivo.

**L'obiezione sul budget e' stata chiusa.** Una terza run con **2.000 passi e lr 3e-4** — 3,3 volte gli
aggiornamenti e il triplo del passo — finisce nello stesso punto: 1 esempio su 24, delta 0, NOT MEMORISED, loss 0,089.
Tre run indipendenti, due precisioni della proiezione dei metadati, budget da uno a tre: stesso esito.

Un'avvertenza sul disegno, imparata qui. Con 24 esempi ogni punteggio e' un multiplo di 1/24 = 0,0417, quindi **un
solo esempio che oscilla sposta l'intero numero**. Al passo 500 della prima run e al passo 1000 della terza sono
comparsi delta di esattamente 0,0417, che sembravano l'inizio di qualcosa e sono rientrati alla valutazione
successiva. Vanno letti solo i plateau su piu' valutazioni, mai un punto singolo; la nota e' ora nel docstring dello
strumento.

### Un guasto hardware, e perche' non tocca i risultati

Alle 22:37 UTC del 16/09 il nodo ha prodotto:

```
Memory access fault by GPU node-1 on address (nil). Reason: DRAM ECC failure.
HW Exception by GPU node-1 reason: GPU Hang
```

La run di memorizzazione lunga e' morta al passo ~500 e il server tmux e' caduto. Il guasto e' **hardware**, non un
errore del codice: un fallimento ECC della DRAM della MI300X. Il heartbeat lo ha intercettato entro quindici minuti,
che e' la ragione per cui esiste.

Due verifiche. La GPU si e' ripresa da sola quando il processo e' morto: matmul corretta, 190,9 GB liberi su 191,7.
E l'evento e' **posteriore** a tutti i risultati riportati qui — `va1r` aveva chiuso alle 19:53, la validazione sulle
520, i due gate sui 208 e i due bracci di memorizzazione da 600 passi erano tutti gia' completati e gia' esportati con
hash verificati contro il nodo. Il log del crash e' conservato come `mem_long_ecc_crash.log`.

## Il bilancio: va1r non legge nulla dai soft token

Tre misure indipendenti, tutte con lo stesso disegno — stessa riga, stessa domanda, stesso header, si cambiano solo i
soft token — danno la stessa risposta.

| capacita' | token veri | token azzerati | token sostituiti | fonte reale |
|---|---:|---:|---:|---|
| posizione (IoU sui positivi) | 0,0089 | — | — | nessuna: non appresa |
| tipo dell'anomalia | 0,713 | **0,700** | 0,725 | header |
| presenza (specificita') | 0,767 | **0,800** | 0,767 | header |
| magnitudo | 0,767 | — | — | header (rapporto 1,16) |

**Il modello ottiene tutto quello che ottiene dalle quattro statistiche stampate nell'header**, mediana, scala, minimo
e massimo. Sono sufficienti per la magnitudo con rapporto mediano 1,16, per il tipo — spike, salto di livello, cambio
di varianza e cambio di trend hanno firme diverse in quelle quattro cifre — e per la presenza, dove la stessa
statistica separa presenti e assenti con AUC 0,76, che e' quasi esattamente la specificita' osservata.

L'unica cosa che l'header non contiene e' **dove** sta l'anomalia, ed e' l'unica cosa che il modello non fa.

Vale la pena essere espliciti su quanto e' forte questa inferenza. La parte diretta — che i soft token non
contribuiscono — e' **misurata**, non dedotta: azzerarli o sostituirli non cambia nessuna delle tre capacita', su
pannelli da 80, 132 e 208 righe. Che la fonte sia l'header e' invece un'inferenza, sostenuta dal fatto che quelle
statistiche bastano numericamente per tutte e tre e che non resta altra sorgente nel prompt.

### Conseguenza sul benchmark, non solo sul modello

Se le tre capacita' facili si rispondono dall'header, allora il primary composito di **qualunque** modello su questo
compito in vista globale misura soprattutto l'uso di quelle quattro cifre. Lo 0,3105 di `va1` e lo 0,3174 di `va1r`
sono in gran parte lo stesso esercizio. La componente percettiva vera e' l'IoU sui positivi: 0,0606 per `va1`,
0,0089 per `va1r`, e 0,248 per il braccio con lo zoom — che e' l'unico punto del progetto in cui qualcosa abbia
davvero letto il segnale.

## Il probe sugli embedding finali: la posizione non arriva dove serve

`tools/embedding_probe.py`. Per ogni registrazione si costruisce il prompt esattamente come all'inferenza, si fa un
forward e si prende lo stato nascosto **dell'ultima posizione del prompt**, cioe' quella da cui il modello scrive il
primo token della risposta. Una ridge con intercetta non penalizzata mappa quello stato sul centro normalizzato
dell'anomalia, adattata su registrazioni **disgiunte** da quelle valutate, con un bersaglio mescolato come null.

| layer | pearson | null mescolato | R2 |
|---|---:|---:|---:|
| 0 | +0,000 | +0,000 | −0,003 |
| 8 | −0,114 | −0,138 | −0,850 |
| 16 | −0,006 | −0,043 | −0,628 |
| 24 | +0,080 | −0,156 | −0,411 |
| 31 | +0,036 | −0,171 | −0,397 |
| finale | +0,073 | −0,162 | −0,351 |

260 registrazioni, 182 per l'adattamento e 78 per la valutazione; errore standard atteso di una correlazione 0,115.
Il massimo in valore assoluto e' −0,114, **un errore standard**, con R2 negativo a ogni profondita' e null di ampiezza
uguale o maggiore. **La posizione non e' linearmente disponibile nel punto in cui va scritta.**

**Una prima versione di questo probe mi aveva ingannato.** Con le sole 102 righe `localize` positive la valutazione
cadeva su 31 registrazioni, dove l'errore standard e' 0,19: il risultato era +0,282 al layer 31 e −0,274 al layer 8,
e la lettura naturale sarebbe stata «la posizione arriva». Allargando il campione a tutte le categorie che portano una
finestra il presunto segnale e' sceso a +0,080. Lo strumento ora stampa l'intervallo di confidenza di Fisher accanto a
ogni correlazione, perche' un valore puntuale su un campione piccolo invita esattamente a quell'errore.

**Limiti dichiarati, e uno e' piu' serio di quanto avessi scritto.** Astra ha osservato che **l'ultima posizione del
prompt precede l'inizio della risposta, non il campo delle coordinate**: il modello scrive
`localize:{"_L": 16384, "end": ...` e le cifre arrivano molti token dopo, quindi potrebbe calcolare la posizione nel
momento in cui la scrive e non prima. Questo probe ha sondato il posto sbagliato, e un risultato nullo li' **non
stabilisce** che la posizione sia indisponibile quando serve. Vale anche il limite generale: un probe lineare negativo
non dimostra assenza, l'informazione potrebbe essere codificata in modo non lineare.

## Il quadro completo della notte

Quattro misure indipendenti, in ordine lungo il percorso del segnale:

1. **L'informazione entra — ma questo NON e' stabilito per `va1r`.** Le due teste lineari di P1-e decodificavano la
   posizione con errore 1,174 bin, ma su **un altro checkpoint** e con bersaglio **la posizione dei token sulla griglia
   di lettura**, non quella dell'anomalia. Astra ha segnalato che il risultato non si trasferisce, e ha ragione: per
   `va1r` non abbiamo mai misurato se l'informazione sia nei suoi soft token.
2. **Il modello non la usa.** Azzerare o sostituire i soft token non cambia posizione, tipo, presenza ne'
   cross-entropy, su pannelli da 80, 132 e 208 righe.
3. **Il sistema non riesce a legare 24 segnali a 24 risposte** con prompt identici alla cifra, in tre run
   indipendenti, con budget da uno a tre e due precisioni della proiezione dei metadati.
4. **La posizione non e' linearmente leggibile** nello stato nascosto da cui la risposta viene scritta, a nessuna
   delle sei profondita' sondate.

**Il punto di rottura non e' ancora localizzato**, ed e' un errore averlo scritto. Quello che i quattro risultati
stabiliscono e' piu' stretto: il modello non usa i soft token, non riesce a memorizzare, e ne' la precisione numerica
ne' il budget spiegano il fallimento. Dove esattamente l'informazione si perda — o se sia mai stata presente nei soft
token di `va1r` — resta aperto.

Il controllo successivo, fissato da Astra: **4-8 segnali corti, stesso header, RoPE nativa contro wildcard dalla stessa
inizializzazione**, con un controllo positivo del compito. Segnali corti perche' il fallimento su 131.072 campioni non
esclude che a lunghezze trattabili il sistema impari.

## Il confronto RoPE su segnali corti: il sistema legge, e la wildcard aiuta

Controllo chiesto da Astra il 17/09: 8 segnali corti, stesso header, RoPE nativa contro wildcard dalla stessa
inizializzazione, con un controllo positivo del compito. Lunghezza **4.096 campioni** invece di 131.072, cosi'
l'anomalia occupa il 3,3 % della serie invece dello 0,1 %. 600 passi, lr 1e-4, prompt verificati identici.

| passo | wildcard λ=1: veri / sostituiti | risolti | nativa λ=0 | risolti | oracolo |
|---:|---:|---:|---:|---:|---:|
| 100 | 0,125 / 0,125 | 1/8 | 0,103 / 0,125 | 1/8 | 7/8 |
| **200** | **0,750 / 0,000** | **6/8** | 0,125 / 0,125 | 1/8 | **8/8** |
| 300 | 0,625 / 0,000 | 5/8 | 0,149 / 0,275 | 1/8 | 8/8 |
| 400 | 0,500 / 0,000 | 4/8 | 0,125 / 0,125 | 1/8 | 8/8 |
| 600 | 0,125 / 0,250 | 1/8 | 0,375 / 0,125 | 3/8 | 8/8 |

Tre conclusioni, e la prima costringe a ritrattare quanto scritto sopra.

1. **Il ciclo di addestramento funziona.** Il controllo positivo — la finestra vera fornita come testo nel turno utente,
   cosi' che il compito non richieda il segnale — arriva a **8/8 con loss 0,0014** dal passo 200. Ottimizzatore,
   gradienti, LoRA, costruzione della sequenza e scorer sono verificati end-to-end.
2. **A 4.096 campioni il sistema LEGGE il segnale.** Al passo 200 il braccio wildcard risolve sei esempi su otto con
   il braccio sostituito **esattamente a zero**: le risposte dipendono interamente dal contenuto. Sei contro zero su
   otto non e' rumore di campionamento. **Quindi «il sistema non sa leggere» era una conclusione sbagliata**: e' vera
   a 131.072 campioni, non in generale.
3. **La wildcard non e' il collo di bottiglia, e' un vantaggio**: 6/8 contro 1/8 della RoPE nativa al picco.

Resta pero' un secondo problema, separato e nuovo: **la soluzione si degrada**. Da 6/8 a 5/8, 4/8, 1/8 in
quattrocento passi, in modo monotono, mentre la loss resta nello stesso intervallo. Il modello trova la lettura e poi
ne esce.

Il seguito immediato e' uno **sweep di lunghezze** — 4.096, 16.384, 65.536, 131.072, due semi ciascuna perche' con
otto esempi ogni punto vale 0,125 — per localizzare dove la lettura si rompe.

## Lo sweep di lunghezze: non e' la scala, ed e' il risultato che riorienta tutto

Otto run, quattro lunghezze, due semi ciascuna, λ=1, 8 esempi, 400 passi, prompt identici alla cifra.

| lunghezza | seme 17 | seme 29 |
|---:|---|---|
| 4.096 | 0,250 / delta +0,125 | **0,875 / delta +0,875 — MEMORISED** |
| 16.384 | **1,000 / delta +1,000 — MEMORISED** | **1,000 / delta +1,000 — MEMORISED** |
| 65.536 | 0,375 / delta +0,375 | 0,250 / delta +0,125 |
| **131.072** | **1,000 / delta +1,000 — MEMORISED** | **0,875 / delta +0,875 — MEMORISED** |

**Alla lunghezza piena, quella dove tutte e undici le run vere hanno fallito, entrambi i semi memorizzano**: otto su
otto e sette su otto, con il braccio sostituito a zero. Le risposte dipendono **interamente** dal contenuto del
segnale.

E il profilo non e' monotono: 4.096 oscilla fra i semi, 16.384 e' perfetto, 65.536 e' debole, 131.072 e' di nuovo
perfetto. Con otto esempi la varianza fra semi e' grande, ma cinque punti su otto memorizzano e i due piu' lunghi sono
fra questi. **L'ipotesi della scala, che avevo formulato mezz'ora prima sulla base del solo punto a 4.096, e' falsa.**

### Che cosa resta in piedi, e che cosa cade

Cadono, nell'ordine in cui le avevo proposte: che il sistema non sappia leggere; che il problema sia la risoluzione
dei token a serie lunghe; che sia la geometria wildcard; che sia la precisione bf16 dei metadati; che sia il budget di
addestramento; che sia il ciclo di ottimizzazione, verificato a 8/8 dal controllo positivo.

Resta una sola differenza fra il caso che funziona e quello che non funziona: **otto esempi sintetici con prompt
identici contro 4.488 esempi reali in cui l'header offre una scorciatoia**. Nel compito vero il modello puo' abbassare
la loss usando mediana, scala, minimo e massimo — che bastano per magnitudo, tipo e presenza — senza mai guardare i
soft token. Nel test sintetico quella strada non esiste, perche' il prompt e' identico in tutti gli esempi, e allora
legge.

La lettura piu' semplice compatibile con tutte le misure e': **il modello non e' incapace di leggere, sceglie la
strada piu' economica.** Va verificata, non e' dimostrata — ma indica un esperimento preciso invece che un altro giro
di run lunghe.

### Il seguito che ne discende

Due bracci, gia' predisposti, che attaccano la scorciatoia invece della percezione:

1. **`va1rh`, header ridotto** (`configs/va1rh_mi300x.toml`, gia' pronto): via minimo e massimo, restano mediana e
   scala. Astra lo aveva approvato come braccio successivo e serve anche alla domanda metodologica sul benchmark,
   perche' e' il confronto header completo contro ridotto a parita' di tutto il resto.
2. **Sostituzione con la risposta del donatore**: si mostra il segnale di un'altra registrazione, si tiene l'header
   del ricevente, e il bersaglio diventa la risposta **del donatore**. Cosi' la scorciatoia non e' solo inutile: e'
   attivamente sbagliata. L'infrastruttura esiste in `substitution.py` e oggi usa la lettura del donatore.

## Conseguenza sul gate

**Il primary complessivo non è una misura di percezione.** Il gate di dipendenza dal contenuto va calcolato solo sulle
categorie che l'header non può risolvere: `localize`, `classify_with_evidence`, `localize_all_channels`. Non è una
scelta ad hoc — sono esattamente le tre categorie su cui Z1-bis aveva dichiarato il suo gate, prima di misurare
+0,1349.

`tools/oracle_text_probe.py` ora riporta tre blocchi: le medie per condizione e per categoria, la dipendenza dal
contenuto complessiva, e la stessa dipendenza ristretta alle tre categorie percettive, che è la quantità confrontabile
con Z1-bis e l'unica usata come criterio.

## Un incidente, e la sua correzione

La run è morta alle 02:14 al passo 4488, fine della prima epoca, con `CheckpointError: checkpoint already exists`: il
salvataggio periodico ogni 1122 passi e quello di fine passata ricadono sullo stesso passo, perché 4488 è un multiplo
di 1122. Cinque ore di droplet perse, circa 13 $. Il trainer ora rimuove il checkpoint periodico superato e lo
riscrive con la contabilità di passata aggiornata; il contratto è fissato da un test di regressione in
`tests/test_checkpoint.py`. Ripresa dal passo 4488 senza perdita di lavoro — l'impronta dell'esperimento copre solo
`src/`, quindi la correzione al trainer non l'ha invalidata.

## Che cosa resta da vedere

Quattro epoche. La domanda è se `localize` e `classify_with_evidence` si stacchino da zero: sono le categorie che
richiedono davvero di leggere il segnale, e finché restano a zero il modello non sta usando la percezione, qualunque
cosa dica il punteggio complessivo.
