# RISULTATI Z1-bis — il crop viene letto: entrambi i gate superati (12/09/2026)

Run `/root/l3/z1bis_20260912`, kit `7b4e604`, artefatti in `integrity/artifacts/l3_z1bis_20260912/`.
Disegno e soglie dichiarati **prima** del run: `l3/PROTOCOL-L3.md` §Z1-bis, config
`l3/configs/l3_zoom_z1bis.yaml`, brief `l3/review/brief_z1bis_codex.md` (piano di Codex,
revisioni `CODEX-REVIEW-ZOOM-20260912.md` e `CODEX-GATE-FOLLOWUP-20260912.md`).

Checkpoint prodotto: `vz1bis_epoch_1_end.pt`, sha256
`924110a5a344dc802c8eee646320f0c7035572c69e1125c714a3db737e8fd23f` (solo sul nodo).

## 1. Il risultato

Il gate precedente aveva stabilito che, con il gold sempre centrato nella finestra, il modello
usava soltanto le coordinate dell'header: azzerare o sostituire il crop non cambiava il punteggio.
Rimossa quella correlazione dalle tracce (start campionato uniformemente fra tutti quelli che
contengono l'evento intero), **il contenuto del crop diventa necessario**.

### Gate percettivo — 208 positivi, finestre fuori centro, header identici fra le tre condizioni

| condizione | contenuto | primary |
|---|---|---:|
| **fuori centro, crop vero** | reale | **0,1994** |
| fuori centro, crop sostituito | altra finestra senza eventi | 0,0645 |
| fuori centro, crop azzerato | zeri | 0,0626 |
| risposta diretta, formato nuovo | nessun crop | 0,0000 (vedi nota) |

Δ appaiato crop vero − miglior controllo = **+0,1349**, CI95 bootstrap 10k
**[+0,0954; +0,1748]**. Soglia dichiarata prima: ≥ +0,05 con CI che esclude lo zero. **Superata.**

Per categoria (vero | miglior controllo | Δ, CI95):

| categoria | n | vero | controllo | Δ | CI95 |
|---|---:|---:|---:|---:|---|
| localize | 80 | 0,2133 | 0,0489 | **+0,1644** | [+0,1011; +0,2285] |
| classify_with_evidence | 80 | 0,1826 | 0,0777 | **+0,1049** | [+0,0501; +0,1647] |
| localize_all_channels | 48 | 0,2042 | 0,0722 | **+0,1319** | [+0,0410; +0,2306] |

Identità degli header verificata su tutti i 208 ID; 186 episodi su 208 hanno usato l'azione.

**Nota sulla quarta riga** (correzione di Codex, 13/09): lo 0,0000 della risposta diretta
*non* è un riferimento «senza zoom». In quella condizione il modello riceve il formato con
azione opzionale e in 186 casi su 208 emette un'azione invece della risposta finale, quindi
lo zero misura la rotta scelta, non un'incapacità di rispondere senza crop. Il riferimento
senza zoom appaiato sugli stessi ID resta `va1` 0,0612 e `vz1` 0,0264
(`l3/tools/subset_baseline.py`), non questa riga.

### Gate operativo — policy autonoma, validation intera 520

`primary` complessivo **0,3426** (va1 0,3105; `vz1` del primo tentativo 0,3011), zoom nel 46,2 %
degli esempi, **0 azioni invalide**, **0 risposte vuote**, ampiezza media della finestra 1.990
campioni.

Confronto appaiato (bootstrap 10k) contro `va1` e contro il primo `vz1`:

| categoria | Z1-bis | va1 | Δ vs va1 | Δ vs vz1 |
|---|---:|---:|---:|---:|
| classify_with_evidence | 0,1482 | 0,0902 | **+0,0579** [+0,0094; +0,1064] | **+0,1051** [+0,0605; +0,1521] |
| localize | 0,2594 | 0,2438 | +0,0156 [−0,0206; +0,0503] | **+0,0394** [+0,0029; +0,0746] |
| localize_all_channels | 0,3944 | 0,3583 | +0,0361 [−0,0106; +0,0843] | **+0,0676** [+0,0093; +0,1292] |
| measure_magnitude | 0,5817 | 0,5778 | +0,0040 [−0,0214; +0,0280] | +0,0229 [−0,0133; +0,0594] |
| lead_lag_with_magnitude | 0,3067 | 0,2381 | +0,0685 [−0,0186; +0,1608] | −0,0408 [−0,0861; −0,0053] |

Tutti i controlli del gate operativo superati: tasso di azione entro 20-60 %, azioni invalide
≤ 2 %, risposte vuote ≤ 2 %, falsi negativi sui positivi 9,6 % (soglia 15 %), falsi positivi sui
negativi 9,3 % (soglia 15 %), Δ per categoria contro va1 non negativi entro le tolleranze.

**Il modello ha imparato quando zoomare**: classify sempre (100 %), localize 67 %,
multicanale 57 %, mentre measure_magnitude e lead_lag rispondono sempre direttamente (0 %),
esattamente il disegno delle tracce.

## 2. Cosa è sostenuto e cosa no

Sostenuto: *con finestre in cui l'anomalia non sta al centro, la rilettura a piena risoluzione
porta un vantaggio misurabile e appaiato di circa +0,13 di primary sui positivi, e la policy
addestrata su queste tracce supera il riferimento SFT sulla validation intera, scegliendo da sé
quando zoomare.*

Non sostenuto: che il vantaggio valga anche quando la finestra è scelta senza informazione
privilegiata — nel gate percettivo canale e regione vengono ancora dall'oracle. Il gate operativo
mostra che la policy autonoma **migliora** rispetto ai due riferimenti, ma il guadagno su localize
contro va1 (+0,016) ha un intervallo che attraversa lo zero: il puntamento autonomo resta il
limite. IoU media fra finestra scelta e anomalia 0,051 su finestre da ~2.000 campioni.

### Dove guarda la policy (misura del 13/09, dal JSONL del gate operativo)

L'IoU 0,051 non dice che il puntamento sia casuale: una finestra da 2.000 campioni che
contiene per intero un evento da 34 non puo' avere IoU alta. La metrica corretta e' il
**contenimento**, misurato con `l3/tools/analyze_policy_windows.py` sulle 240 righe in cui
la policy ha agito (235 con gold):

| categoria | n con gold | evento contenuto | canale corretto |
|---|---:|---:|---:|
| classify_with_evidence | 110 | 57 (52 %) | - |
| localize | 84 | 55 (65 %) | - |
| localize_all_channels | 41 | 18 (44 %) | 40/41 (98 %) |
| **totale** | **235** | **130 (55 %)** | **40/41** |

Quando l'evento e' contenuto la posizione interna e' centrata (media 0,506, mediana 0,520):
la policy non sbaglia il centramento, sbaglia la *scala*. Due misure lo mostrano:

- `primary` condizionato: **0,2126** quando l'evento e' contenuto, **0,0219** quando non lo e'.
  Tutto il divario residuo su localize e multicanale vive nel 45 % di finestre che mancano l'evento.
- il contenimento crolla con la lunghezza, mentre l'ampiezza scelta resta ferma attorno a 2.000
  campioni: 80 % a L=16.384 (ampiezza/L 9,8 %), 67 % a 32.768, 50 % a 65.536, **32 % a 131.072**
  (ampiezza/L 1,7 %); il `primary` segue esattamente (0,200 / 0,164 / 0,095 / 0,068).
- fra i 104 mancati, il 59 % cade entro **mezza** ampiezza dal bordo e il 70 % entro una: sono
  quasi-centri, non tentativi ciechi. Solo l'11 % e' oltre dieci ampiezze, e sono serie lunghe.

La causa e' nel disegno delle tracce Z1-bis: le ampiezze {1024, 2048, 4096} sono state estratte
indipendentemente da L, quindi su una serie da 131.072 campioni insegnano a coprire meno del 2 %
della serie. Non e' un difetto del selettore appreso: e' il budget.

Nessun risultato sul test 800: la catena non lo tocca, come da protocollo.

## 3. Dettagli del run

- Tracce: 4.488 righe pianificate in 6 s con 504 MiB (il primo planner materializzava le serie e
  aveva fatto cadere il nodo; corretto in `7b4e604` con lettura per colonne via PyArrow).
  Sorgenti: classify 971 zoom / 171 replay; localize 769 / 183 diretta / 168 replay;
  all_channels 364 / 107 / 83; measure_magnitude 953 diretta / 168 replay; lead_lag 468 / 83.
  Posizione interna dell'anomalia: media 0,50-0,53, deviazione 0,26-0,27, casi centrati 1-2 %.
  Pseudo-recupero: 0.
- SFT: 1 epoca, 4.488 passi, 7,2 h, loss media 0,3638, picco 12.019 MiB, solo LoRA (encoder e
  proiettore congelati).
- Gate percettivo 0,6 h; gate operativo 0,4 h. Catena totale 8,2 h (~7 $).

## 4. Prossimi passi (da decidere)

1. **Selettore della finestra**: il limite è scegliere dove guardare senza gold. Due strade già
   discusse: un selettore wavelet deterministico che proponga solo coordinate (confronto a parità
   di budget con la policy), oppure un secondo zoom grossolano→fine, da validare prima come
   ipotesi.
2. **GRPO** sulla scelta della finestra, ora che esiste un comportamento utile da affinare, con
   step 0 = questo checkpoint fra i selezionabili e criterio centrato su localize/all_channels.
3. **Test 800** una sola volta, a disegno congelato, con confronto appaiato a quattro bracci.
