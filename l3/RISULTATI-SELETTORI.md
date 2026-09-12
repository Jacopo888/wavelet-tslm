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
