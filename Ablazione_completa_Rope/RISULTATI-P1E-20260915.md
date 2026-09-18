# P1-e (15/09/2026): supervisione diretta per token, la prima informazione leggibile del progetto

Disegno deciso da Astra (sezioni "Decisioni Astra su P1-e" e "Decisioni Astra dopo la fase 1" di
[DECISIONI-ASTRA-20260915.md](DECISIONI-ASTRA-20260915.md)) dopo il fallimento di P1-d fase A: due teste lineari sui soft
token dopo la normalizzazione RMS, con ingresso `soft / 0,013`, che predicono per ciascun token la posizione del suo bin
sulla griglia di lettura, l'energia di Haar `log1p(E)` del bin e il picco `log1p` dello scarto robusto massimo nel bin.
Bersagli standardizzati con statistiche fisse del train (512 registrazioni: posizione 0,5 ± 0,289; energia 7,878 ± 2,195;
picco 1,273 ± 1,673), pesi 1 / 1 / 0,5, riduzione annidata token → bin → canale → vista → registrazione, decorrelazione
0,01, LoRA congelata, frontend 1e-4. Implementazione in `src/wavelet_rope/token_supervision.py`; statistiche da
`tools/token_supervision_stats.py`; gate da `tools/token_supervision_eval.py` (soglie di Astra: errore di posizione ≤ 1 bin
e recupero dei primi tre bin ≥ 0,70 su registrazioni mai addestrate, viste globali a 64 bin, caso 16 bin e 0,047).

## Fase 1 (`p1-e1`): 200 passi di sola supervisione, senza cross-entropy

Con `ce_weight = 0` il backbone non viene nemmeno costruito nel percorso di addestramento: 2,1 s per esempio contro 2,6.

| passo | errore posizione (bin) | recupero top-3 | Pearson energia | Pearson picco |
|------:|----------------------:|---------------:|----------------:|--------------:|
| 100   | 4,16 | 0,361 | 0,519 | 0,602 |
| 200   | 2,37 | 0,378 | 0,672 | 0,698 |

FAIL sulle soglie, ma è la prima volta nel progetto che i soft token contengono informazione leggibile che **generalizza
a registrazioni nuove**: nei probe lineari su P1-d la salienza era esattamente al livello del caso, a inizializzazione
come dopo l'addestramento.

## Estensione (`p1-e1b`): 600 passi con il termine di ordinamento

Ripresa dal checkpoint 200 conservando i momenti di Adam (`--resume-foreign` + `--extend-steps`, perché l'impronta
dell'esperimento include l'hash del sorgente), warmup 18, picco 1e-4, coseno, più la perdita a coppie tipo RankNet con
peso 0,2 decisa da Astra.

| passo | errore posizione | recupero top-3 | Pearson energia |
|------:|-----------------:|---------------:|----------------:|
| 300   | 2,31 | 0,378 | 0,643 |
| 400   | 3,70 | 0,406 | 0,728 |

Posizione ed energia si scambiano capacità. La perdita di ordinamento resta attorno a 0,64 contro il valore 0,693 che
corrisponde a nessuna discriminazione: i negativi scelti come "i sedici con punteggio predetto più alto" sono un bersaglio
mobile. Astra ha fermato l'estensione al passo 400 e messo il peso del termine a zero per il seguito, osservando che quel
valore, misurato sui negativi più difficili e continuamente riselezionati, non dimostra assenza di discriminazione globale.

## Fase 2 (`p1-e2`): lettura più sostituzione, la configurazione migliore

Dal checkpoint 200, 300 passi con LoRA congelata, cross-entropy della lettura a peso 1 e finale a peso 0, sostituzione dei
token al 25 % (realizzata esattamente: 177 esempi su 707), bersaglio della lettura preso dal segnale effettivamente
mostrato e nessun bersaglio finale per gli esempi sostituiti.

| passo | errore posizione | recupero top-3 | Pearson energia | vantaggio del LM | bin generati: veri / sostituiti |
|------:|-----------------:|---------------:|----------------:|-----------------:|--------------------------------:|
| 0     | 2,37 | 0,378 | 0,672 | | |
| 100   | 6,06 | 0,372 | 0,581 | +0,016 CI [−0,002; +0,039] | 0,014 / 0,028 |
| 200   | 1,54 | 0,350 | 0,584 | +0,023 CI [−0,005; +0,057] | 0,066 / 0,035 |
| 300   | 1,21 | 0,400 | 0,783 | | |

Due fatti importanti. Primo: la cross-entropy attraverso il modello linguistico **degrada** la leggibilità nei primi cento
passi, da 2,37 a 6,06 bin, poi la combinazione con la supervisione per token e la sostituzione la porta a 1,21, il valore
migliore del progetto, vicino alla soglia di un bin. Secondo: al passo 200 i token veri battono per la prima volta il
controllo nella generazione, 0,066 contro 0,035, e il vantaggio in cross-entropy cresce fra un controllo e l'altro.

Il gate del modello linguistico resta FAIL (soglia 0,10 nats con intervallo di confidenza positivo), ma la direzione è
coerente su tutti gli indicatori, cosa mai vista nelle run precedenti.

## Lettura complessiva

I vincoli sono due e distinti. Il primo, la leggibilità dei token, è stato risolto dalla supervisione diretta. Il secondo,
la capacità del backbone di usare quella codifica, resta aperto: con la LoRA congelata l'unica parte adattabile è il
frontend, a cui si chiede di produrre token già comprensibili a un modello congelato. La sostituzione al 25 % è lo
strumento che neutralizza le distribuzioni a priori senza dover congelare la LoRA.

## Seguito (P1-f)

Astra ha ordinato la variante V1: dal checkpoint 200 pre-ordinamento, frontend interamente congelato e in modalità di
valutazione, solo LoRA a 1e-4 con warmup 20 e poi costante, 300 passi, lettura 1 / finale 0, sostituzione 25 %. In
parallelo il controllo geometrico a parità di tutto il resto, con il λ della RoPE wildcard forzato a 0 contro 1. Un
fallimento di V1 non basterebbe ad attribuire il problema all'interfaccia: dimostrerebbe soltanto mancato apprendimento
con quella LoRA, quella ottimizzazione e quella durata.

Artefatti: `integrity/artifacts/rope_p1e1_20260915/`, `rope_p1e1b_20260915/`, `rope_p1e2_20260915/`.
