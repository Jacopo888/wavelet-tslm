# va1rh (17/09/2026): togliere la scorciatoia dell'header

Braccio approvato da Astra il 16/09 (opzione C) e lanciato il 17/09 dopo lo sweep di lunghezze. Identico a `va1r`
salvo una sola modifica testuale: **il header globale non stampa più minimo e massimo**, mentre mediana e scala
restano, perché sono la normalizzazione in cui i soft token sono espressi. Due epoche come pilota, 8.976
aggiornamenti, partenza pulita, `configs/va1rh_mi300x.toml` e `prototype.spec.va1rh.json`.

## Perché questo braccio

Il 16/09 avevo misurato che `va1r` non usa i soft token: azzerarli o sostituirli non cambia posizione, tipo, presenza
né cross-entropy. E che le quattro statistiche stampate nell'header bastano numericamente per magnitudo, tipo e
presenza. Astra aveva però posto la condizione giusta prima di attribuire il risultato a quelle quattro cifre: serve
**il confronto header completo contro ridotto, con tutto il resto fissato**. Questo braccio è quel confronto.

Lo sweep di lunghezze del 17/09 ha poi aggiunto il motivo per cui l'esperimento è interessante e non solo doveroso:
con 8 esempi e prompt identici alla cifra, l'architettura **legge il segnale a qualunque lunghezza**, 131.072
compresi. Quindi l'ipotesi sotto test non è «il modello non sa leggere» ma «il modello sceglie la strada più
economica, e nel compito vero quella strada è l'header».

## Epoca 1

| categoria | `va1r` ep. 1 | `va1rh` ep. 1 |
|---|---:|---:|
| **measure_magnitude** | **0,7546** | **0,0469** |
| localize_all_channels | 0,2667 | 0,2667 |
| lead_lag_with_magnitude | 0,0000 | 0,0346 |
| localize | 0,0000 | 0,0000 |
| classify_with_evidence | 0,0000 | 0,0000 |
| **complessivo** | 0,2271 | **0,0552** |

**La magnitudo crolla di sedici volte**, da 0,7546 a 0,0469, per la sola rimozione di due numeri dal prompt. È la
prova diretta che quella categoria veniva risposta dall'header e non dal segnale, ed è il controllo che Astra
richiedeva prima di poter affermare qualcosa sul benchmark.

**Ma rimuovere la scorciatoia non ha prodotto lettura**: `localize` resta a 0,0000. Il modello non ha ripiegato sul
segnale, ha perso i punti che prendeva gratis.

### La scomposizione, che il punteggio nasconde

| categoria | totale | sui positivi | sui negativi | specificità | falsi «vuoto» |
|---|---:|---:|---:|---:|---:|
| localize | 0,0000 | 0,0000 | 0,0000 | 0,000 | 0,000 |
| localize_all_channels | 0,2667 | 0,0000 | 1,0000 | 1,000 | **0,909** |

Su multicanale la specificità è 1,000, ma il tasso di falsi «vuoto» sui positivi è **0,909**: non sta rilevando
l'assenza, sta **rispondendo «nessuna anomalia» quasi sempre**, dieci volte su undici anche quando l'anomalia c'è, e
incassa i quattro negativi per costruzione. `va1r` alla stessa epoca sbagliava così due volte su undici, quindi
discriminava davvero.

Su `localize` il comportamento è speculare: **non dice mai «assente»**, zero volte su otto negativi, e sbaglia ogni
finestra. Due default opposti nelle due categorie: è il profilo di un modello che ha perso l'appiglio dell'header e
non ne ha ancora trovato un altro. `va1r` alla prima epoca era nella stessa condizione su `localize` e ha imparato a
dire «assente» nella seconda.

## Gate dichiarati prima della run

IoU sui positivi di `localize`; F1 sui positivi del multicanale; accuratezza del tipo e IoU dell'evidenza riportate
**separatamente**; sensibilità e specificità della presenza separate; e il gate vero/sostituito/azzerato sui 208
positivi di Z1-bis. **Mai il primary composito**, che su questo benchmark in vista globale misura l'uso dell'header.


## La provenienza dell'header: la scorciatoia e' nostra

Verificato il 17/09 su richiesta di Jacopo, che sospettava venisse dal benchmark.

Il dataset AnomalyXL fornisce soltanto `id`, `channel_names`, `n_channels`, `category`, `length`, `question`,
`answer` e le annotazioni gold. La `question` e' quella del benchmark e non contiene statistiche.

Il prompt di **`va1`** (`kaggle/patch/qa_dataset.py`) e': il system prompt del benchmark, poi per ogni canale la
stringa `"The following is channel {nome}:"` seguita dagli embedding, poi `"Question: {domanda}"`. **Nessuna
statistica**: niente mediana, scala, minimo, massimo, nessuna riga con la lunghezza.

Il prompt di **TimeRLM** e' agentico a esecuzione di codice — «a specialized time-series agent that uses code to solve
tasks» — quindi il modello ispeziona la serie scrivendo Python e non riceve alcun header testuale.

La stringa `median {` compare in **un solo punto dell'intero repository**: `src/wavelet_rope/templates.py`, cioe'
`ConversationBuilder.global_pieces`. **L'header con le quattro statistiche e' stato introdotto da noi nella linea
Ablazione_completa_Rope**, e non esisteva ne' nel benchmark ne' in `va1`.

### Tre conseguenze

1. **La mail a ETH cambia di segno.** Non si puo' scrivere che il loro training set sporcava la lettura: la
   scorciatoia l'abbiamo costruita noi, e solo in questa linea. Resta valido il risultato metodologico — mettere
   statistiche riassuntive nel prompt consente un buon composito senza usare gli embedding — ma e' un avvertimento
   sul disegno dei prompt per TSLM, non una critica al benchmark.
2. **Il confronto `va1` contro `va1r` era confondato.** `va1r` disponeva di un vantaggio d'ingresso che `va1` non
   aveva: il +0,1892 sulla magnitudo veniva da due numeri regalati nel prompt. E `va1`, **senza header**, faceva
   0,578 su quella categoria e 0,0606 di IoU sui positivi, cioe' leggeva davvero il segnale.
3. **Togliere l'header non e' un'ablazione in piu': e' ripristinare la condizione d'ingresso di `va1`**, l'unica
   configurazione del progetto in cui qualcosa abbia letto il segnale in vista globale. Undici run hanno cercato
   perche' il modello non leggesse, con davanti una scorciatoia che il riferimento non aveva.
