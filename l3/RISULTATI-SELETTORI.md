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
| oracle alla stessa ampiezza (limite superiore) | 1.947 | 100,0 % | 100,0 % |
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
| oracle alla stessa ampiezza | 100 % | 0,2327 | +0,1049 [+0,0628; +0,1458] |

**Il selettore deterministico raggiunge l'oracle.** A parità di token e di ampiezza,
una regola di energia Haar nel canale che la policy sceglie vale quanto sapere dove
sta l'anomalia: 0,2340 contro 0,2327, e quasi il doppio dello 0,1278 della policy.

Il margine residuo del sistema non è dunque nel lettore né nella rappresentazione: è
**tutto nella scelta della finestra**, e una regola non addestrata lo copre già.

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
