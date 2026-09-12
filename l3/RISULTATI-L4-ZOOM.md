# RISULTATI Livello 3, punto 4 (zoom) — stato al 12/09/2026

Stato: Z0 e Z1 eseguite; **Z1 fermata dal suo gate** (fail-closed), Z2/Z3 mai eseguite.
Artefatti: `integrity/artifacts/l3_zoom_20260911/`. Protocollo: `l3/PROTOCOL-L3.md` §punto 4.
Kit `7350c92`, run `/root/l3/zoom_20260911c`.

**Revisione indipendente di Codex (12/09): `l3/review/CODEX-REVIEW-ZOOM-20260912.md`, ricalcoli
in `l3/review/AUDIT-ZOOM-20260912.json`.** Questo documento è stato riscritto dopo quella
revisione: la prima versione (commit `8a79d48`) conteneva tre affermazioni troppo forti, qui
corrette e segnalate come tali.

## 1. Formulazione sostenuta oggi

> Il crop guidato da un oracle migliora molto i risultati rispetto alla lettura globale;
> restano da isolare il contributo del contenuto percettivo e la capacità di selezionare
> autonomamente una finestra utile.

Non è sostenuto: che il guadagno venga dimostrabilmente dai coefficienti della finestra, che
esista una soglia percettiva misurata, che il modello sia al livello dell'agente TimeRLM.

## 2. Numeri misurati (validation 520, greedy, stesso checkpoint `vz1` salvo l'ultima riga)

| condizione | primary | localize | all_channels | classify | lead_lag |
|---|---:|---:|---:|---:|---:|
| `va1` (riferimento L1) | 0,3105 | 0,2438 | 0,3583 | 0,0902 | 0,2381 |
| `vz1` senza zoom (prompt va1 originale) | 0,3011 | 0,2200 | 0,3269 | 0,0431 | 0,3475 |
| `vz1`, finestra scelta dalla policy | 0,2869 | 0,2050 | 0,3148 | 0,0337 | 0,2972 |
| `vz1`, oracle **largo** (target 6144 sui casi con gold) | 0,3234 | 0,2890 | 0,4000 | 0,0911 | 0,2547 |
| `vz1`, oracle **stretto** (gold ±25 %) | **0,5244** | **0,6333** | **0,6657** | **0,4539** | 0,2547 |

Ampiezze: «~1024 campioni» vale sulle righe **con** intervallo gold; la modalità oracle ripiega
sull'euristica sulle altre, quindi la media su tutti i 520 è 4.500,7 campioni (oracle stretto) e
6.901 (oracle largo). Le condizioni vanno sempre riportate per strato, non in aggregato.

## 3. Tre correzioni alla prima lettura (Codex §3, §4, §5)

**(a) L'oracle comunica quasi esattamente il centro dell'anomalia.** L'intestazione del blocco
mostra canale, start ed end, e il centro coincide col centro gold entro mezzo campione in 98/102
positivi localize e 108/110 classify. Un controllo puramente geometrico *post-hoc* (predire un
intervallo fisso centrato nell'header, senza leggere il segnale) ottiene IoU 0,597 con ampiezza
128, contro 0,555 della risposta vera di `vz1` con lo stesso crop. Non prova che il modello
ignori i coefficienti, ma **impedisce di attribuire loro il guadagno**: serve il confronto fra
crop vero e crop sostituito a header, coordinate e numero di token identici (gate in corso).

**(b) Il bersaglio NON è risolto: la copertura del 70 % è dovuta all'ampiezza.** Restringendo a
1024 campioni le finestre attuali conservandone il centro, la copertura crolla:

| categoria (soli positivi) | n | copertura attuale (canale incluso) | a 1024 | a 2048 |
|---|---:|---:|---:|---:|
| localize | 102 | 70,6 % | **16,7 %** | 33,3 % |
| classify | 110 | 71,8 % | **27,3 %** | 43,6 % |
| all_channels | 48 | 62,5 % | **6,25 %** | 14,6 % |

Errore mediano di centro: 1.778 campioni su localize, 1.550 su classify, 4.373 su all_channels.
Il mio script `analyze_self_window.py` ignorava il canale nel multicanale (riportava 68,8 %
invece di 62,5 %): corretto in questo commit. Quindi **va imparato anche il centraggio**, non
solo l'ampiezza.

**(c) Affermazioni ritirate.**
- «Risoluzione utile ≈ 4 campioni per token»: `L/256` è un **rapporto di compressione medio**,
  non il supporto temporale dei token. L'encoder produce medie grossolane (128 blocchi) più
  dettagli top-k a scale diverse, e fra le due condizioni cambiano insieme crop, MAD locale,
  bande selezionate e posizioni. Servirebbe uno sweep controllato di ampiezza.
- «L'SFT non ha degradato il modello»: senza zoom `vz1` perde metà dello score classify
  (0,043 contro 0,090) e scende su localize, all_channels e magnitudo; sale su lead_lag, che
  maschera il resto nell'aggregato. Delta appaiati policy contro va1: classify −0,0566
  CI95 descrittivo [−0,0999; −0,0136], all_channels −0,0435 [−0,0908; −0,0046] (bootstrap
  post-hoc, senza correzione per confronti multipli).
- «Al livello di TimeRLM»: confronto non valido, oracle su validation contro agente autonomo sul
  test. Inoltre 0,656/0,551 sono di **TimeRLM-text GPT-5.5 depth-1**; per **Qwen3.5-4B RL** la
  tabella riporta 0,538 su localize e 0,244 su all_channels.

## 4. Z0 e Z1: cosa è successo

**Z0** (controlli su va1, senza addestramento). val520: senza zoom **0,3105**, identico a va1 a
quattro cifre, quindi il percorso di eval del kit è validato; oracle silenzioso 0,268, euristica
silenziosa 0,263. Diagnosi preliminare (`l3/tools/diag_zoom_empty.py`): va1 emette EOS immediato
appena si aggiunge una riga di testo dopo la domanda, anche solo la frase che descrive l'azione e
senza alcun blocco. Il formato va insegnato, non si valuta a freddo.

**Z1** (SFT del formato a due turni, solo LoRA, encoder e proiettore congelati). 4.488 episodi,
8.976 segmenti, 9,8 h, loss media 0,3096, copertura completa per categoria, picco 12.020 MiB,
checkpoint `vz1_epoch_1_end.pt` (sha256 `79571951…4dd2`). Alla valutazione: azioni sempre valide,
zoom_rate 1,0, **0 risposte vuote** (l'interfaccia è stata appresa), primary 0,2869 < 0,3105 →
gate FALLITO.

Difetti misurati nelle tracce, tutti da correggere prima di un nuovo training:
1. `heuristic_rate` reale 0,61 contro 0,30 previsto: `choose_supervised_action` ripiega sulla
   finestra euristica larga ogni volta che il gold non contiene un intervallo (measure_magnitude,
   lead_lag, casi senza anomalia).
2. **Nessuna traccia insegna la risposta diretta**: il contratto forza un'azione su ogni esempio.
3. Il jitter varia il **margine** simmetricamente, non il centro: dopo l'espansione al minimo di
   1024 il gold resta quasi sempre centrato. È una scorciatoia presente anche nel training.
4. Il calo su lead_lag con crop mono-canale non dimostra che lo zoom togliga contesto (i canali
   originali restano nel prompt): è interferenza o cambio di comportamento. Escluderlo dal ramo
   zoom resta una scelta operativa, non una conclusione.

## 5. Continuazione (piano di Codex, adottato)

1. **Gate a pesi fissi sul `vz1` esistente** (in corso, `l3/tools/diag_crop_content.py`):
   sottoinsieme stratificato fissato prima dell'esperimento (seed 20260912, 282 righe, categorie
   × positivi/negativi × banda di lunghezza, lead_lag escluso), quattro condizioni a header,
   coordinate, formato e numero di token identici: crop vero, **crop sostituito** (contenuto di
   un'altra finestra dello stesso canale, priva di eventi), crop azzerato, crop vero con gold in
   **posizione interna casuale**. Metriche: primary, IoU sui positivi, risposte vuote.
2. **Pilot Z1-bis con una sola azione opzionale**, ripartendo da va1: finestre strette, jitter
   vero del centro, ramo di risposta diretta, quota di replay del formato originale, percentuali
   esplicite per categoria e nessun fallback silenzioso; misurare il ramo diretto **nel nuovo
   formato** (l'attuale eval senza zoom usa il vecchio prompt e non lo certifica).
3. **Due zoom solo se il pilot dimostra** che la prima rilettura migliora la proposta fine; nel
   multicanale la seconda azione deve poter cambiare canale, e il recupero deve poter uscire
   dalla finestra sbagliata.
4. **GRPO solo con un comportamento utile da affinare**, con step 0 = checkpoint SFT fra i
   selezionabili, criterio di selezione centrato su localize/all_channels e vincoli di non
   regressione, non sul solo aggregato che lead_lag può gonfiare.
5. **Test finale** a quattro bracci più il confronto diretto zoom-SFT contro zoom-RL e il
   controllo dello stesso checkpoint con e senza informazione del crop; riportare costo, token e
   azioni. Le letture del test di va1 e del controllo RL sono già storiche.

Le ~30 h stimate in chat non sono un budget validato: il kit attuale supporta **una sola** azione
per episodio; due zoom richiedono storico delle osservazioni, maschere dei log-prob e checkpoint
del formato nuovi, quindi memoria e tempo da rimisurare.

## 6. Costi

Z0 1,1 h; Z1 9,8 h + 0,5 h di eval; diagnosi 1,4 h; gate in corso ~1 h; due lanci falliti
(import mancante, EOS) ~1 h. Totale punto 4 ≈ 15 h A40 ≈ 13 $.
