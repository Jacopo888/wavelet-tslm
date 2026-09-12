# RISULTATI Livello 3, punto 4 (zoom) — stato al 12/09/2026

Stato: Z0 e Z1 eseguite; **Z1 fermata dal suo gate** (fail-closed), Z2/Z3 mai eseguite.
Artefatti: `integrity/artifacts/l3_zoom_20260911/`. Protocollo: `l3/PROTOCOL-L3.md` §punto 4.
Kit `7350c92`, run `/root/l3/zoom_20260911c`.

**Revisione indipendente di Codex (12/09): `l3/review/CODEX-REVIEW-ZOOM-20260912.md`, ricalcoli
in `l3/review/AUDIT-ZOOM-20260912.json`.** Questo documento è stato riscritto dopo quella
revisione: la prima versione (commit `8a79d48`) conteneva tre affermazioni troppo forti, qui
corrette e segnalate come tali.

## 1. Formulazione sostenuta oggi

> Nel checkpoint `vz1` e su questo gate, il crop corretto **non mostra un beneficio medio
> rilevabile** rispetto a contenuti sostituiti o azzerati; la prestazione dipende fortemente
> dalla relazione fra header e posizione del gold. Il modello addestrato in Z1 sfrutta la
> scorciatoia «rispondi al centro delle coordinate che ti vengono date», che era la soluzione
> ottimale delle tracce insegnate. Resta aperta la domanda se una rilettura a piena risoluzione
> porti informazione utile quando quella scorciatoia è resa impossibile.

Non è sostenuto: che lo zoom migliori la percezione, che esista una soglia percettiva misurata,
che il modello sia al livello dell'agente TimeRLM.

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

## 2-bis. GATE A PESI FISSI (12/09, esito): il guadagno viene dalle COORDINATE, non dal contenuto

Condizioni a header, coordinate, formato e numero di token **identici per costruzione**; cambia
solo il tensore iniettato nell'encoder. Sottoinsieme stratificato fissato prima dell'esperimento
(seed 20260912): 208 righe **con intervallo gold** (80 localize, 80 classify, 48 all_channels;
measure_magnitude, lead_lag e i negativi non hanno un gold utilizzabile e restano fuori da questo
gate). Strumento: `l3/tools/diag_crop_content.py`; artefatti `integrity/artifacts/l3_zoom_20260911/Z1/gate/`.

| condizione | contenuto della finestra | header | primary | Δ appaiato vs vero |
|---|---|---|---:|---:|
| **vero** | crop reale, gold centrato | vero | 0,4792 | — |
| **sostituito** | altra finestra dello stesso canale, senza eventi | identico al vero | 0,4729 | +0,0063 (sd 0,168) |
| **azzerato** | tutti zeri | identico al vero | 0,4820 | −0,0028 (sd 0,177) |
| **fuori centro** | crop reale, gold in posizione interna casuale | vero | **0,1226** | **+0,3566** (sd 0,336) |

**Conclusione sostenuta** (formulazione di Codex, follow-up §1): *nel checkpoint `vz1` e su
questo gate, il crop corretto non mostra un beneficio medio rilevabile rispetto a contenuti
sostituiti o azzerati; la prestazione dipende fortemente dalla relazione fra header e posizione
del gold.* Bootstrap appaiato 10k: vero − sostituito +0,0063 CI95 [−0,0154; +0,0298];
vero − azzerato −0,0028 [−0,0269; +0,0216]; vero centrato − vero fuori centro +0,3566
[+0,3111; +0,4029].

Non è sostenuto «il modello non legge il contenuto»: sostituendo il crop cambiano 170 predizioni
su 208 e azzerandolo 178 (il punteggio cambia in 116 e 128 casi). Il contenuto influenza il
comportamento, ma **il beneficio netto misurato è vicino a zero**, e nessuna soglia di equivalenza
era stata fissata prima.

**Baseline appaiata corretta** (mio errore, segnalato da Codex §2: 0,123 e 0,301 erano popolazioni
diverse, 208 positivi contro 520 righe intere). Sugli **stessi 208 ID**, strumento
`l3/tools/subset_baseline.py`:

| serie, stessi 208 ID | tutti | localize | classify | all_channels |
|---|---:|---:|---:|---:|
| `va1` senza zoom | 0,0612 | 0,0694 | 0,0673 | 0,0375 |
| `vz1` senza zoom | **0,0264** | 0,0188 | 0,0306 | 0,0319 |
| `vz1`, finestra scelta dalla policy | 0,0134 | 0,0110 | 0,0156 | 0,0139 |
| gate, crop vero fuori centro | 0,1226 | 0,1086 | 0,0902 | 0,2000 |
| gate, crop vero centrato | 0,4792 | 0,5417 | 0,4050 | 0,4986 |

Quindi il confronto giusto è: fuori centro − `vz1` senza zoom = **+0,0963** CI95
[+0,0610; +0,1316]. Non dimostra un beneficio del contenuto (l'oracle fornisce ancora canale e
regione, informazione privilegiata), ma **smentisce la mia affermazione che fuori centro fosse
peggio del non zoomare**. Da notare anche: `vz1` senza zoom − `va1` senza zoom = −0,0349
[−0,0599; −0,0108] su questi positivi, cioè l'SFT ha davvero peggiorato il ramo senza crop.

Quindi lo 0,5244 della riga «oracle stretto» **non è una misura della percezione**: è la fuga di
informazione dalle coordinate dell'oracle, esattamente il rischio segnalato da Codex. Anche il
tetto «0,633 su localize» va letto così. Per la proposta non resta alcuna evidenza che la
rilettura a piena risoluzione porti informazione utile: resta una domanda aperta.

Origine della scorciatoia: nel blocco zoom le posizioni temporali dei token sono normalizzate
sulla lunghezza del blocco (la normalizzazione MAD riguarda invece i valori), quindi l'header
fornisce l'origine temporale necessaria alla registrazione. **Non serve però cambiare
architettura né togliere le coordinate** (Codex follow-up §3): la conversione
`t_assoluto = start_crop + u_relativo × ampiezza` è disponibile con un header veritiero. La fuga
nasce dalla correlazione artificiale «centro del crop = centro gold» introdotta dalle tracce di
Z1, non dalla presenza delle coordinate. Rimuovere quella correlazione conserva encoder, header
e formato delle risposte assolute; la risposta in coordinate locali resta un'ablazione
successiva, utile a separare lettura del segnale e aritmetica.

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
