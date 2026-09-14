# RISULTATI selettori — dove guarda la policy e quanto vale guardare altrove (13/09/2026)

Misure a pesi congelati sul solo checkpoint `vz1bis_epoch_1_end.pt`
(sha `924110a5…`), nessun gradiente, nessun accesso al test 800. Disegno e soglie
dichiarati prima in `l3/PROTOCOL-L3.md` §«Selettori della finestra»; kit `947eaa3`.
Ordine dei punti da `CODEX-GATE-FOLLOWUP` e dal piano relayato il 13/09.

Righe: le 240 della validation 520 in cui la policy autonoma ha emesso un'azione,
di cui 235 con evento gold. Le altre 280 rispondono direttamente e non entrano.

## 1. Perché non l'IoU

L'IoU media fra finestra scelta e anomalia è 0,051, e ieri l'avevo riportata come
misura del puntamento: è sbagliato. Una finestra da 2.000 campioni che contiene per
intero un evento da 34 ha IoU 0,017 pur avendo fatto centro. La metrica dichiarata
è il **contenimento**: l'evento cade per intero nella finestra e sul canale scelto.

## 2. Geometria a parità di budget (fase CPU, 35 s, picco 1,4 GiB)

Tutte le condizioni sulle stesse 235 righe. Il selettore parametrizzato riproduce
esattamente `l3.zoom.heuristic_zoom` alle sue impostazioni native su 240/240 righe
(verifica fail-closed).

| condizione | ampiezza media | contenimento | canale corretto |
|---|---:|---:|---:|
| finestra dell'oracle alla stessa ampiezza | 1.947 | 100,0 % | 100,0 % |
| **policy** | 1.998 | **55,3 %** | **99,6 %** |
| wavelet libero, stessa ampiezza | 1.988 | 68,9 % | 90,2 % |
| **wavelet nel canale della policy** | 1.988 | **77,0 %** | 99,6 % |
| wavelet al suo budget nativo (L/8) | 7.986 | 80,0 % | 90,2 % |
| policy, finestra ×2 | 3.901 | 79,6 % | 99,6 % |
| policy, finestra ×4 | 7.499 | 87,2 % | 99,6 % |
| policy, finestra ×8 | 14.124 | 92,8 % | 99,6 % |

Il criterio dichiarato prima del run — il selettore wavelet è una strada se supera
la policy di almeno 10 punti a parità di ampiezza — è **superato**: +13,6 punti nella
versione libera, +21,7 in quella che eredita il canale dalla policy.

Contenimento per lunghezza della serie, dove la differenza diventa strutturale:

| L | policy | wavelet nel canale | policy ×2 | policy ×4 |
|---|---:|---:|---:|---:|
| 16.384 | **80,4 %** | 73,9 % | 93,5 % | 93,5 % |
| 32.768 | 67,1 % | 77,1 % | 92,9 % | 95,7 % |
| 65.536 | 50,0 % | 75,0 % | 77,3 % | 86,4 % |
| 131.072 | 32,0 % | **80,0 %** | 60,0 % | 76,0 % |

Sulle serie corte la policy è già migliore del selettore; su quelle lunghe crolla
al 32 % mentre il selettore resta all'80 % **senza allargare la finestra**. La causa
è nel disegno delle tracce Z1-bis: le ampiezze {1024, 2048, 4096} sono state
estratte indipendentemente da L, quindi a 131.072 campioni insegnano a coprire meno
del 2 % della serie.

Tre fatti collegati, misurati sulle stesse righe:

- quando l'evento è contenuto la posizione interna è centrata (media 0,506,
  mediana 0,520): la policy non sbaglia il centramento, sbaglia la **scala**;
- `primary` condizionato: **0,2126** con evento contenuto contro **0,0219** senza;
- dei 104 mancati, il 59 % cade entro mezza ampiezza dal bordo e il 70 % entro una:
  sono quasi-centri. Solo l'11 % è oltre dieci ampiezze, su serie lunghe.

Il canale non è il problema: 40 su 41 righe multicanale corrette (97,6 %); il
selettore wavelet libero invece sbaglia canale nel 9,8 % dei casi, ed è tutto il suo
svantaggio rispetto alla versione che eredita il canale dalla policy.

## 3. Limite di questa fase

È geometria, non punteggio. Il crop viene sempre codificato in 256 token per canale,
quindi allargare aumenta il contenimento ma riduce la risoluzione: il contenimento
al 92,8 % di una finestra da 14.124 campioni non implica un punteggio migliore. Le
due misure GPU che decidono — contenuto sulle finestre autonome con la scala della
finestra, e punteggio dei selettori a parità di budget — sono nelle sezioni
seguenti.

## 4. Il contenuto sulle finestre scelte dalla policy (GPU, 1,1 h)

Stesse 235 righe, azioni della policy **congelate**: stesso checkpoint, stessa
azione, stesso header, stesso numero di token. Cambia solo il contenuto del crop.

| condizione | primary |
|---|---:|
| **crop vero** | **0,1278** |
| crop azzerato | 0,0341 |
| crop sostituito | 0,0241 |

Δ appaiato vero − miglior controllo = **+0,0937**, CI95 **[+0,0648; +0,1234]**, contro
la soglia +0,03 dichiarata prima. **Superata.** Il beneficio percettivo misurato con
le finestre dell'oracle esiste anche sulle finestre che la policy sceglie da sé.

La sotto-ipotesi registrata prima del run è confermata in modo netto: il vantaggio
del contenuto vale **+0,1627** sulle righe in cui la finestra contiene l'evento e
**+0,0082** dove lo manca. Il contenuto conta esattamente quando c'è qualcosa da
leggere: è il legame fra percezione e puntamento che mancava.

## 5. Allargare la finestra non paga

Stesso centro scelto dalla policy, ampiezza moltiplicata, header veritiero sulle
nuove coordinate.

| ampiezza | contenimento | primary | Δ vs ampiezza scelta |
|---|---:|---:|---|
| scelta dalla policy (~1.990) | 55,3 % | 0,1278 | — |
| ×2 (~3.900) | 79,6 % | 0,1341 | +0,0063 [−0,0269; +0,0400] |
| ×4 (~7.500) | 87,2 % | 0,0615 | **−0,0663** [−0,0986; −0,0339] |

Il contenimento sale di 32 punti, il punteggio no. A ×4 **peggiora in modo
significativo**. Il crop resta 256 token: allargare compra contenimento e paga in
risoluzione, e a partire da ×4 il secondo effetto domina. La strada «finestre più
grandi» è chiusa da questa misura; quella giusta è **puntare meglio alla stessa
ampiezza**.

## 6. Punteggio dei selettori a parità di budget — il risultato centrale

Stesse righe, stessa ampiezza, header veritiero, contenuto vero: cambia solo il
criterio con cui la finestra è scelta.

| braccio | contenimento | primary | Δ appaiato vs policy |
|---|---:|---:|---|
| policy (finestra emessa dal modello) | 55,3 % | 0,1278 | — |
| wavelet libero | 68,9 % | 0,2188 | **+0,0910** [+0,0536; +0,1309] |
| **wavelet nel canale della policy** | 77,0 % | **0,2340** | **+0,1062** [+0,0647; +0,1479] |
| finestra dell'oracle | 100 % | 0,2327 | +0,1049 [+0,0628; +0,1458] |

**Il selettore deterministico pareggia la finestra dell'oracle.** A parità di token e di ampiezza,
una regola di energia Haar nel canale che la policy sceglie vale quanto sapere dove
sta l'anomalia: 0,2340 contro 0,2327, e quasi il doppio dello 0,1278 della policy.

Il margine misurato qui sta dunque nella **scelta della finestra**, e una regola non
addestrata lo copre già. Attenzione a non leggerlo come «il lettore non ha margine»:
la finestra dell'oracle è una strategia di scelta, non un limite superiore del
lettore, e pareggiarla non dice nulla sul margine percettivo residuo.

## 7. Decisione presa dalla catena

Tabella dichiarata prima in `PROTOCOL-L3.md`, valutata alle 00:45Z: contenuto
+0,0937 (soglia +0,03) ✓, finestre migliori +0,1049 (soglia +0,05) ✓, miglior
selettore +0,1062 (soglia +0,03) ✓ → ramo **`selector_sft`**: insegnare alla policy
a proporre quelle coordinate. `DECISION.json` conserva misure e soglie.

Piano delle tracce W1 verificato prima della GPU (49,8 s, picco 2.655 MiB): su 2.104
righe con azione, **l'88,7 % riceve una finestra proposta dall'energia Haar**
(rank 1 nell'81 %, rank 2-4 nel 7,7 %) e solo l'11,3 % ricade sul meccanismo Z1-bis.
Posizione interna dell'evento: media 0,50-0,54, deviazione 0,23-0,27, casi centrati
1,4-4,5 % — la scorciatoia del centro resta impossibile.

## 8. Ramo `selector_sft` (W1): esito

Run `/root/l3/night_20260913T005001Z`, catena 7,6 h, artefatti in
`integrity/artifacts/l3_w1_20260913/`. SFT di 1 epoca da `va1`: 4.488 passi, 7,13 h,
loss media 0,3637, picco 12.019 MiB, solo LoRA (32 tensori), 236.641 token
supervisionati, rotte 46,9 % zoom / 38,1 % diretta / 15,0 % replay.
Checkpoint `vw1_epoch_1_end.pt`, sha
`8221cd4aba0a9101374ec551ba31f239020586bb6fcc1257a407abdee748ed10`
(copia fuori dal nodo in `~/l3_offnode/w1_20260913/`).

### Gate operativo su val520: superato, tutti i controlli

`primary` complessivo **0,3551** (`vz1bis` 0,3426; `va1` 0,3105). Azione nel 45,2 %
degli esempi, **0 azioni invalide**, **0 risposte vuote**, falsi negativi sui positivi
10,4 %, falsi positivi sui negativi **3,7 %** (era 9,3 %). Rotte apprese come da
disegno: classify 100 %, localize 62,9 %, multicanale 58,3 %, magnitudo e lead_lag
0 %.

Confronto appaiato (bootstrap 10k):

| categoria | W1 | Δ vs va1 | Δ vs vz1bis |
|---|---:|---|---|
| classify_with_evidence | 0,1497 | **+0,0594** [+0,0135; +0,1071] | +0,0015 [−0,0426; +0,0456] |
| localize | 0,2977 | **+0,0539** [+0,0123; +0,0960] | +0,0383 [−0,0038; +0,0824] |
| localize_all_channels | 0,4194 | **+0,0611** [+0,0042; +0,1245] | +0,0250 [−0,0250; +0,0759] |
| measure_magnitude | 0,5825 | +0,0047 [−0,0188; +0,0279] | +0,0008 [−0,0205; +0,0221] |
| lead_lag_with_magnitude | 0,2983 | +0,0602 [−0,0131; +0,1383] | −0,0083 [−0,0883; +0,0713] |

**Il limite residuo di Z1-bis è chiuso**: localize e multicanale, che contro `va1`
avevano intervalli che attraversavano lo zero (+0,016 e +0,036), ora battono `va1`
con intervalli che escludono lo zero (+0,054 e +0,061). Contro `vz1bis` il
miglioramento va nella stessa direzione ma non è individualmente significativo.

### Geometria delle nuove finestre: un terzo del divario recuperato

Stesse 235 righe con azione, stessa metrica di prima:

| | contenimento | canale | ampiezza |
|---|---:|---:|---:|
| policy Z1-bis (ieri) | 55,3 % | 99,6 % | 1.998 |
| **policy W1** | **60,5 %** | **100,0 %** | 1.908 |
| wavelet nel canale della policy W1 | 78,1 % | 100 % | 1.906 |
| oracle | 100 % | 100 % | 1.873 |

Per banda di lunghezza, W1 contro Z1-bis: 74,5 % contro 80,4 % a L=16k, 71,0 %
contro 67,1 % a 32k, **59,1 % contro 50,0 %** a 65k, **42,5 % contro 32,0 %** a 131k.
Il guadagno è dove serviva, sulle serie lunghe; sulle corte cede qualcosa.

Sostenuto: *insegnare le coordinate del selettore migliora sia il puntamento
(+5,2 punti di contenimento, +10,5 sulle serie da 131k) sia il risultato finale, e
rende significativi contro `va1` proprio i due compiti che restavano incerti.*

Non sostenuto: che l'imitazione sia completa. La stessa regola Haar, applicata alle
righe della nuova policy, trova ancora l'evento nel 78,1 % dei casi contro il 60,5 %
della policy: **il modello ha recuperato circa un terzo del divario**, non tutto. Il
margine che resta è misurato e ha un obiettivo esplicito.

### Prossimo passo naturale

GRPO sulla scelta della finestra partendo da `vw1`, con step 0 fra i candidati: ora
esiste un comportamento utile da affinare e un tetto noto (78,1 % di contenimento,
0,234 di primary sulle righe con azione). In alternativa, o in aggiunta, il selettore
deterministico può entrare nel ciclo come proposta esterna — resta anche una baseline
onesta da riportare nel confronto. Il test 800 non è stato toccato.

## 9. Quanto margine resta dopo W1 — con una ritrattazione (13/09)

Ripetuto su `vw1` il confronto dei selettori, sulle sue 233 righe con azione, e poi
sulle 231 righe zoomate da entrambi i checkpoint. Controllo di convenzione superato:
su `vz1bis` la finestra forzata dà 0,1278 e l'episodio reale 0,1274 sulle stesse
righe, quindi i due riferimenti sono intercambiabili.

| sulle 231 righe comuni | vz1bis | vw1 | Δ appaiato |
|---|---:|---:|---|
| policy (episodio reale) | 0,1296 | 0,1470 | +0,0174 [−0,0146; +0,0505] |
| finestra dell'oracle, ampiezza della policy | 0,2364 | 0,1892 | −0,0472 [−0,0906; −0,0037] |
| wavelet nel canale della policy | 0,2380 | 0,1849 | — |
| divario oracle − policy | +0,1068 [+0,067; +0,147] | +0,0422 [+0,006; +0,078] | — |

### Ritrattazione: quel −0,047 non isola il lettore

Segnalato da Codex e **verificato**: `oracle_at_width` costruisce la finestra usando
**l'ampiezza scelta da ciascuna policy**. Fra i due bracci cambiano quindi insieme il
checkpoint e l'input: su 231 righe comuni, **116 hanno coordinate e ampiezza
diverse**. Restringendo alle **115 righe con canale e coordinate identiche**:

| sottoinsieme | n | Δ oracle vw1 − vz1bis |
|---|---:|---|
| finestre identiche | 115 | −0,0517 **[−0,1073; +0,0032]** |
| classify | 64 | −0,0180 [−0,0846; +0,0440] |
| localize | 36 | −0,0313 [−0,1218; +0,0547] |
| multicanale | 15 | −0,2444 [−0,4556; −0,0444] |

A finestre identiche **l'effetto aggregato non esclude lo zero**. Ritiro quindi
l'affermazione «il tetto del lettore è sceso»: non è isolata, e con il vincolo
corretto non è sostenuta. Resta un solo segnale, sul multicanale con n=15, che va
trattato come indizio da verificare, non come risultato.

### Altre due correzioni alla mia lettura

- «Il selettore esterno ha esaurito il vantaggio» era troppo forte: Δ +0,0376 con
  CI [−0,0037; +0,0788] significa **incerto**, ed è compatibile anche con un beneficio
  rilevante. Il selettore resta un candidato operativo, non qualcosa da scartare per
  un intervallo che sfiora lo zero.
- La finestra dell'oracle **non è un tetto assoluto**: è una particolare strategia di
  scelta (centrata sull'evento più lungo, ampiezza imposta). Pareggiarla non dimostra
  che lettore e rappresentazione non abbiano margine. Nel documento ho sostituito
  «tetto» con «finestra dell'oracle» dove indicava un limite superiore.

Quello che resta sostenuto: sulle righe con azione il divario fra la finestra
dell'oracle e la policy si è ridotto da +0,107 a +0,042, e la parte di questa
riduzione attribuibile al lettore **non è ancora separata** da quella attribuibile
alle finestre.

### Misura a finestre identiche: esito inconcludente (13/09, 09:50Z)

`vw1` valutato sugli **stessi 208 crop fuori centro** del gate percettivo di Z1-bis,
con lo stesso config e lo stesso seed. Finestre identiche **verificate riga per riga**
su tutti i 208 ID: hash dell'header, canale, coordinate, ampiezza e ground truth.
Criteri dichiarati prima in `PROTOCOL-L3.md` §«Separazione lettore/finestre», con
soglia di rilevanza δ = 0,05.

| confronto | valore | CI95 |
|---|---:|---|
| **Δ_lettura** = W1 vero − Z1-bis vero (stessi crop) | **−0,0176** | **[−0,0522; +0,0161]** |
| **Δ_contenuto** = W1 vero − miglior controllo | **+0,1115** | [+0,0790; +0,1456] |

Condizioni W1: crop vero 0,1818, sostituito 0,0702, azzerato 0,0475.
Per categoria, lettura: classify −0,0324 [−0,0849; +0,0176], localize −0,0025
[−0,0469; +0,0422], multicanale −0,0181 [−0,1181; +0,0778].

**Esito: inconcludente sulla lettura, positivo sul contenuto.** L'intervallo di
Δ_lettura è compatibile sia con un calo rilevante (−0,05) sia con nessun calo, quindi
per la regola dichiarata **non parte alcun training lungo**. La stima resta negativa:
non si dichiara equivalenza, si dichiara incertezza. Contemporaneamente W1 **ricava
ancora un beneficio netto dal contenuto** del crop, +0,112 con intervallo ben sopra
lo zero, in tutte e tre le categorie.

Due sottoprodotti utili:

- l'indizio sul multicanale (−0,244 su n=15) **non si riproduce** a finestre
  identiche: qui è −0,018 [−0,118; +0,078]. Era un artefatto delle finestre diverse,
  e va abbandonato.
- il gate percettivo di W1 passerebbe di nuovo con margine (+0,112 contro la soglia
  +0,05): la percezione del crop non è stata compromessa dall'addestramento W1.

### Ampliamento eseguito: la regressione è reale, piccola, e localizzata (12:25Z)

Due crop aggiuntivi per ID su **entrambi** i checkpoint (`--seed-offset` 1 e 2): le
coordinate dipendono solo da config, seed e offset, quindi restano appaiate fra i
bracci. 208 ID × 3 crop = 624 valutazioni per braccio; differenze mediate **dentro
l'ID** e bootstrap **sugli ID**, perché i crop dello stesso esempio sono correlati.
Finestre identiche verificate su tutti gli ID di tutte e tre le coppie.

| confronto | 1 crop per ID | **3 crop per ID** |
|---|---|---|
| Δ_lettura | −0,0176 [−0,0522; +0,0161] | **−0,0299 [−0,0511; −0,0083]** |
| Δ_contenuto | +0,1115 [+0,0790; +0,1456] | **+0,1053 [+0,0798; +0,1319]** |

Condizioni W1 sui tre crop: vero 0,1742, sostituito 0,0632, azzerato 0,0566.

**Ora il calo esclude lo zero, ma resta sotto la soglia di rilevanza.** La stima è
−0,030 contro un δ dichiarato di 0,05, e l'intervallo mancava di escludere −0,05 per
un millesimo (estremo inferiore −0,0511). Formalmente **nessuna diramazione è
soddisfatta**: non è un calo rilevante e convincente (la stima è metà della soglia),
e non si può dichiarare che un calo di 0,05 sia escluso. Quindi, per la regola
dichiarata, **nessun training lungo parte**.

Il valore di questa misura sta però nella risoluzione per categoria, che è nuova:

| categoria | n | Δ_lettura | Δ_contenuto |
|---|---:|---|---|
| localize | 80 | **−0,0005** [−0,0308; +0,0318] | +0,1433 [+0,0987; +0,1892] |
| classify_with_evidence | 80 | −0,0300 [−0,0634; +0,0022] | +0,0916 [+0,0586; +0,1265] |
| **localize_all_channels** | 48 | **−0,0789** [−0,1296; −0,0289] | +0,0648 [+0,0076; +0,1255] |

- **Su localize non si rileva alcun calo** (−0,0005 con incertezza di circa ±0,03):
  «nessun calo rilevato», non «nessun costo» — l'intervallo ammette ancora
  variazioni fino a tre centesimi in entrambe le direzioni.
- Su classify il calo è piccolo e l'intervallo sfiora lo zero.
- **Sul multicanale il calo è il più forte** (−0,079 con CI che esclude lo zero): la
  stima supera la soglia di rilevanza, ma l'intervallo [−0,130; −0,029] **non
  dimostra** che il calo la superi con certezza. È il segnale più solido che abbiamo,
  non una misura conclusiva della sua entità.

Correzione di una mia correzione: con un solo crop per ID avevo **ritirato**
l'indizio sul multicanale (−0,018 [−0,118; +0,078]). A potenza tripla l'indizio
riemerge in forma più piccola ma statisticamente netta. Il −0,244 su n=15 del
confronto confuso resta non riprodotto nella grandezza; il fenomeno però esiste.

Resta pieno il beneficio del contenuto in tutte le categorie (+0,105 in aggregato,
+0,065 anche sul multicanale): W1 legge ancora il crop, non lo ignora.

### Situazione per la decisione

Sostenuto: W1 migliora il puntamento e il risultato operativo; legge gli stessi crop
in modo **leggermente** peggiore, di 0,030 in aggregato (CI esclude lo zero, la
soglia di rilevanza no); il segnale più forte è sul **multicanale** (−0,079, stima
oltre la soglia ma intervallo che non ne prova il superamento), mentre su localize
**non si rileva** alcun calo; il beneficio dal contenuto resta ampio.

Non sostenuto: che il calo aggregato sia irrilevante (l'intervallo tocca −0,051) né
che sia rilevante (la stima è −0,030). Su questo punto la misura, a questa potenza,
resta al confine.

Le opzioni, con i numeri accanto, restano da decidere: recupero del lettore mirato al
multicanale con replay di crop percettivi (è l'unica categoria dove il calo supera la
soglia); pilot GRPO da `vw1` con il multicanale sotto osservazione nel criterio di
selezione; o un ulteriore ampliamento se si vuole chiudere il confine aggregato.
Config GRPO pronta e non lanciata. Test 800 non toccato. GPU libera.

## 10. W2 — il recupero è riuscito, e ha superato il punto di partenza (14/09)

Run `/root/l3/w2_20260913T133901Z`, artefatti in `integrity/artifacts/l3_w2_20260913/`.
Intervento unico: metà delle righe multicanale con azione (176, quota a conteggio
esatto) insegna un **crop percettivo** — finestra che contiene il gold con start
uniforme — invece delle coordinate del selettore. Tutto il resto invariato, solo
LoRA da `vw1`, 1 epoca: 4.488 passi, 7,11 h, loss 0,3441, picco 12.018 MiB.
Checkpoint `vw2_epoch_1_end.pt`, sha
`9f48ca99490b1dd8b8fb9b8a075495e2be38dfb7ea2da7df51dfe6a55e629964`, copia in
`~/l3_offnode/w2_20260913/`.

### Obiettivo 1 — lettura, sui 208 crop fissi × 3 offset a finestre identiche

| confronto | valore | CI95 |
|---|---:|---|
| multicanale, W2 − W1 | **+0,0653** | [+0,0148; +0,1213] |
| classify, W2 − W1 | +0,0578 | [+0,0256; +0,0914] |
| localize, W2 − W1 | +0,0773 | [+0,0411; +0,1134] |
| **aggregato, W2 − Z1-bis** | **+0,0371** | [+0,0140; +0,0598] |
| contenuto, W2 vero − miglior controllo | +0,1834 | [+0,1530; +0,2143] |

Il criterio dichiarato chiedeva che il multicanale migliorasse su W1 e che
l'aggregato contro Z1-bis non fosse più compatibile con un calo rilevante. **Il
risultato va oltre**: non solo il −0,030 di W1 è recuperato, l'aggregato è ora
**positivo** con intervallo che esclude lo zero, cioè W2 legge gli stessi crop
**meglio** del riferimento pre-W1. Il guadagno è in tutte e tre le categorie, non solo
in quella trattata, e il beneficio dal contenuto sale a +0,183.

### Obiettivo 2 — puntamento e operativo

`primary` su val520 **0,3747** (W1 0,3551; Z1-bis 0,3426; `va1` 0,3105), contenimento
**63,9 %** (W1 60,5 %), gate operativo superato, 0 azioni invalide, 0 risposte vuote,
falsi negativi 10,4 %, falsi positivi 3,7 %. Rotte: classify 100 %, localize 62 %,
multicanale 60 %, magnitudo e lead_lag 0 %.

Appaiato contro `va1`: classify **+0,0935**, localize **+0,1092**, multicanale
**+0,0667**, magnitudo +0,005, lead_lag +0,043. Contro `W1`: localize **+0,0553**,
classify +0,0341, multicanale +0,0056, lead_lag −0,0169.

Contenimento per banda: 84,8 % a L=16k, 74,3 % a 32k, 54,5 % a 65k, **46,6 % a 131k**
(W1: 74,5 / 71,0 / 59,1 / 42,5) — migliora agli estremi, cede un po' a 65k.

### Verdetto e conseguenza

`w2_becomes_candidate`: entrambi gli obiettivi soddisfatti, quindi **W2 sostituisce
W1 come candidato** e il pilot GRPO parte da lui, come dichiarato prima del run.

Due letture da tenere separate. Sostenuto: *aggiungere crop percettivi alle tracce di
selezione recupera la lettura e migliora anche il puntamento e il risultato
operativo*. Non sostenuto: che l'intervento fosse specifico del multicanale — il
guadagno di lettura è ampio su tutte le categorie, quindi il meccanismo non è «ho
curato il multicanale» ma «il replay percettivo fa bene in generale». Il multicanale
resta la banda più debole in geometria (46,6 % a 131k).

## 11. Pilot GRPO da W2: il rinforzo sulla scelta della finestra non muove la policy (14/09)

Run `/root/l3/grpo_20260914T012904Z`, 18,0 h in tutto (guards 15 min, 150 passi di
training ≈ 14,7 h, tre validazioni ≈ 1 h, verifica della lettura ≈ 1,7 h), artefatti
in `integrity/artifacts/l3_grpo_20260914/`. Partenza `vw2` (verdetto
`w2_becomes_candidate`), disegno e criteri dichiarati prima dell'esito di W2 in
`PROTOCOL-L3.md` §«Pilot GRPO dopo W2». Guards superati: identità della policy
iniziale verificata, gate di memoria a 13.095 MiB con 16 rollout su 16 parsabili.

Un passo costa in media **359 s** (mediana 187, massimo 659): gli episodi zoom su
serie multicanale da 131k con 8 rollout sono lenti. La stima iniziale di 6 ore era
sbagliata di più del doppio.

### Le tre validazioni su val520, appaiate contro lo step 0

| step | primary | multicanale | Δ vs step 0 | selezionabile |
|---:|---:|---:|---|---|
| 0 (`vw2`) | 0,3747 | 0,4250 | — | sì |
| 50 | 0,3799 | 0,4560 | +0,0052 [−0,0120; +0,0222] | sì |
| 100 | 0,3743 | 0,4347 | −0,0004 [−0,0193; +0,0178] | no |
| 150 | 0,3741 | 0,4194 | −0,0005 [−0,0205; +0,0194] | no |

Tre validazioni su tre **indistinguibili dal punto di partenza**; per categoria
nessun intervallo esclude lo zero. Comportamento intatto per tutto il pilot:
235-237 azioni, 0 invalide, 0 vuote, KL 0,0005-0,0014. Arresto per stagnazione allo
step 150 (stale = 2). Selezione congelata dalla regola dichiarata sullo **step 50**,
che vince per 0,005: rumore, ma la regola è la regola e non si cambia dopo.

### Verifica della lettura sul checkpoint scelto (208 crop × 3 offset, finestre identiche)

| confronto | valore | CI95 |
|---|---:|---|
| multicanale, step 50 − `vw2` | +0,0106 | [−0,0282; +0,0507] |
| aggregato, step 50 − `vw2` | +0,0027 | [−0,0097; +0,0155] |
| aggregato, step 50 − Z1-bis | +0,0398 | [+0,0180; +0,0618] |
| contenuto, vero − miglior controllo | +0,1778 | [+0,1486; +0,2069] |

**Accettato**: nessun peggioramento della lettura. Lo step 50 legge come `vw2`, e
come `vw2` legge meglio di Z1-bis. Checkpoint `vw2_grpo_step50.pt`, sha
`cebefffb2ca40114d9b4875b745e74c4d10a2a05e9007b8638ef82e3959b97b8`, copia in
`~/l3_offnode/grpo_20260914/`.

### Cosa dice il pilot

Sostenuto: *150 passi di GRPO sul reward ufficiale, G=8, a partire da una policy che
già zooma bene, non producono alcun cambiamento misurabile né in operativo né in
lettura*. La policy non peggiora — il KL resta minuscolo — ma non impara nulla che
il reward riesca a premiare oltre quello che l'SFT le ha già insegnato.

Non sostenuto: che il rinforzo sulla finestra sia inutile in generale. Il pilot è
breve, il reward è quello grezzo del benchmark (denso solo su localize), la
temperatura è fissa: un reward modellato sul contenimento, o più passi, potrebbero
muovere qualcosa. Quello che il pilot esclude è che a **questo** budget e con
**questo** reward ci sia un guadagno facile da prendere.

Letto insieme al resto della giornata, il quadro è coerente: allargare la finestra
non paga (§5), il replay percettivo paga su tutte le categorie (§10), il rinforzo
sulla finestra non aggiunge (§11). Il margine che resta è **percettivo**, e la leva
che ha funzionato è nelle **tracce**, non nel reward.

### Candidato e prossimo passo

Il candidato formale è lo step 50; in pratica è `vw2` con rumore addosso, e per ogni
uso successivo la provenienza più semplice è `vw2` (sha `9f48ca99…`). La scelta fra i
due non cambia nulla di misurabile.

Il passo naturale ora è il **test 800, una sola volta, a disegno congelato**, con
confronto appaiato a più bracci: `va1` (senza zoom), `vz1bis`, `vw2`, e il candidato
GRPO se si vuole chiudere anche quel confronto. È una decisione di Jacopo, non della
catena. Prima, se si vuole un'ulteriore garanzia sul risultato principale, un secondo
seed di W2 (~9,5 h) verificherebbe che il +0,037 di lettura e lo 0,3747 operativo non
dipendano dall'estrazione delle tracce.
