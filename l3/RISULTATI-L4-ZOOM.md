# RISULTATI L4 (punto 4) — lo zoom funziona, la scelta della finestra no

Stato: 12/09/2026. Z0 e Z1 eseguite; **Z1 fermata dal suo gate** (fail-closed) e catena
interrotta prima di Z2/Z3. Artefatti: `integrity/artifacts/l3_zoom_20260911/`.
Protocollo: `l3/PROTOCOL-L3.md` §punto 4. Kit: `7350c92` (run `/root/l3/zoom_20260911c`).

## 1. Il risultato principale

Tutte le righe sono lo **stesso modello** `vz1` (va1 + un'epoca di SFT sul formato a due
turni, solo LoRA, encoder e proiettore congelati) valutato sulla **validation intera 520**
con decodifica greedy; l'ultima riga è il riferimento SFT `va1`.

| finestra di zoom | ampiezza media | primary | localize | all_channels | classify | MMag | lead_lag |
|---|---:|---:|---:|---:|---:|---:|---:|
| nessuna | — | 0,3011 | 0,220 | 0,327 | 0,043 | 0,559 | 0,348 |
| scelta dalla policy | 6.150-10.100 | 0,2869 | 0,205 | 0,315 | 0,034 | 0,561 | 0,297 |
| **gold, larga** | 6.901 | 0,3234 | 0,289 | 0,400 | 0,091 | 0,550 | 0,255 |
| **gold, stretta (±25 %)** | ~1.024 | **0,5244** | **0,633** | **0,666** | **0,454** | 0,550 | 0,255 |
| `va1` (riferimento) | — | 0,3105 | 0,244 | 0,358 | 0,090 | 0,578 | 0,238 |

Letture, tutte misurate e non inferite:

1. **Il percettore wavelet legge davvero la rilettura a piena risoluzione.** Con una finestra
   giusta e stretta la localizzazione passa da 0,220 a **0,633** e il multicanale da 0,327 a
   **0,666**. Per riferimento, l'agente multi-turno di TimeRLM (Qwen3.5 + REPL, codice numpy)
   fa 0,656 su localize e 0,551 su all_channels: **un solo passaggio percettivo con una
   finestra giusta arriva al livello dell'agente**.
2. **Conta la risoluzione, non il contenimento.** Centrata sul gold ma larga 6.901 campioni la
   stessa finestra dà solo 0,289 su localize (+0,07 sul niente), contro 0,633 a 1.024
   campioni. Il blocco percettivo ha 256 token: a 1.024 campioni sono ~4 campioni per token,
   a 6.900 ne sono ~27 e un'anomalia di ~100 campioni occupa 4 token, invisibile. La
   **risoluzione utile del percettore è ~4 campioni per token**.
3. **L'SFT non ha degradato il modello**: `vz1` senza zoom fa 0,3011 contro 0,3105 di va1.
4. **La policy sbaglia l'ampiezza, non il bersaglio.** Le finestre che chiede contengono
   l'anomalia nel 69-72 % dei casi (`integrity/artifacts/l3_zoom_20260911`, analisi
   `analyze_self_window.py`), ma sono 6-10 volte troppo larghe: 0,2869, sotto il gate.
5. **Difetto misurato nelle tracce di Z1**: `heuristic_rate` reale 0,61 contro 0,30 previsto.
   `choose_supervised_action` ripiega sulla finestra euristica (larga, `max(1024, L/8)`) ogni
   volta che il gold non contiene un intervallo: measure_magnitude (1.121 righe), lead_lag
   (551) e i casi senza anomalia. Il modello ha quindi imparato soprattutto finestre larghe.
6. **Lo zoom mono-canale non serve a lead_lag** (0,255 contro 0,348 senza zoom): la domanda
   confronta due canali e una finestra su uno dei due toglie contesto. Da escludere anche
   nell'SFT, non solo nel pool RL.

## 2. Z0 (controlli su va1, senza addestramento)

val520: senza zoom **0,3105** (identico a va1 a quattro cifre: il percorso di eval del kit è
validato), oracle silenzioso 0,268, euristica silenziosa 0,263. Su un modello che non ha mai
visto un blocco in più il blocco peggiora, come atteso; `flag_z1_required` alzato.

Diagnosi preliminare (`l3/tools/diag_zoom_empty.py`): va1 emette EOS immediato appena si
aggiunge una riga di testo dopo la domanda, anche solo la frase che descrive l'azione e senza
alcun blocco. Il formato a due turni va quindi insegnato (Z1), non si può valutare a freddo.

## 3. Z1 (SFT del formato a due turni)

4.488 passi in 9,8 h, loss media 0,3096, copertura completa per categoria, picco 12.020 MiB,
checkpoint `vz1_epoch_1_end.pt` (sha256 `79571951…4dd2`). Alla valutazione con zoom scelto
dalla policy: zoom_rate 1,0 (zooma sempre), 0 azioni invalide, **0 risposte vuote** (il
formato è stato appreso), primary 0,2869 < 0,3105 → **gate FALLITO**, catena fermata.

## 4. Conseguenze per il disegno (proposta)

Il tetto misurato (0,524 complessivo, 0,633 su localize) dice che vale la pena rifare Z1 con
tracce corrette, non abbandonare. Modifiche necessarie, tutte nelle tracce e nel budget di
azioni, nessuna nell'architettura:

- finestre **strette** (target ~1.024-2.048 campioni, cioè ~4-8 campioni per token) ogni volta
  che il gold ha un intervallo, con jitter sul centro e non sull'ampiezza;
- **insegnare a non zoomare** dove lo zoom non serve: measure_magnitude, lead_lag e i casi
  senza anomalia rispondono direttamente;
- **due zoom per episodio** (grossolano poi fine): la finestra larga della policy contiene
  l'anomalia nel 70 % dei casi, una finestra stretta sulla prima risposta di va1 solo nel
  37-48 %; un primo zoom larga-ma-contenente seguito da uno stretto dentro di essa è il
  percorso che unisce i due numeri;
- una quota di tracce (~20 %) con finestra **fuori bersaglio** e risposta comunque corretta,
  per insegnare il recupero quando la finestra manca l'anomalia.

Poi Z2 (GRPO) ottimizza la scelta della finestra con la reward ufficiale, e Z3 chiude il
confronto a quattro bracci sul test in una sola lettura.

Budget stimato del rifacimento: Z1-bis ~12 h, Z2 ~15 h, Z3 ~1,5 h ≈ 30 h A40 (~26 $).

## 5. Costi già spesi (punto 4)

Z0 1,1 h; Z1 9,8 h + 0,5 h di eval; diagnosi 1,4 h; due lanci falliti (import, EOS) ~1 h.
Totale ≈ 14 h A40 ≈ 12 $.
