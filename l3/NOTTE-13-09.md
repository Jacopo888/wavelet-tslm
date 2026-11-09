# Catena notturna del 13/09/2026 — cosa gira e cosa guardare al risveglio

Due sessioni tmux sul nodo, avviate alle 23:17 e 23:49 UTC del 12/09.

| sessione | cosa fa | run root |
|---|---|---|
| `l3-selectors-20260912T231746Z` | gate del contenuto sulle finestre autonome + scala ×2/×4 | `/root/l3/selectors_20260912T231746Z` |
| `l3-night-20260912T234945Z` | punteggio dei selettori → decisione (conclusa 00:45Z) | `/root/l3/night_20260912T234945Z` |
| `l3-night-20260913T005001Z` | **ramo `selector_sft` in corso** | `/root/l3/night_20260913T005001Z` |

La seconda attende che la prima liberi la GPU, quindi non si sovrappongono.

## Ordine delle fasi

1. `preflight` — SHA di `vz1bis` e `va1`, 28 test CPU. **Fatto.**
2. `wait_content` — attesa del gate del contenuto (max 75 min).
3. `scores` — `score_selectors.py`: tre bracci a parità di ampiezza (wavelet libero,
   wavelet nel canale della policy, oracle come tetto) appaiati contro il braccio
   della policy. ~40 min.
4. `decision` — `decide_branch.py` applica la tabella dichiarata in
   `PROTOCOL-L3.md` §«Catena notturna condizionale» e scrive `DECISION.json`.
5. Un solo ramo, secondo la decisione.

## I tre rami

**`selector_sft`** (atteso se un selettore batte la policy di ≥ +0,03 con CI oltre
lo zero). Insegna alla policy le coordinate dell'energia Haar invece di quelle del
gold: candidati = i primi 4 intervalli per energia nel canale dell'evento, start
campionato fra le finestre che contengono il *centro dell'intervallo* — non
l'evento, così la posizione interna resta variabile e la scorciatoia del centro non
si ricrea. Si tiene il primo candidato che contiene l'evento intero; se nessuno
serve, ricaduta sul meccanismo Z1-bis marcata `gold_fallback`. **Il gold filtra solo
fra candidati calcolabili in inferenza, mai le coordinate.** Poi SFT di 1 epoca
partendo da `va1` (~7,2 h), gate operativo contro `va1` e `vz1bis`, e geometria
delle nuove finestre. Fallisce chiusa prima della GPU se le proposte wavelet sono
utili in meno del 45 % delle righe.

**`grpo`** (se il puntamento conta ma nessun selettore batte la policy). Da
`vz1bis`, **step 0 fra i candidati** col suo 0,3426 su val520, chunk da 25 fino a
150, validazione ogni 50, arresto per stagnazione. Se nessuno step supera lo step 0,
il candidato resta Z1-bis.

**`report_only`** (se il contenuto non serve sulle finestre autonome, o se finestre
migliori non migliorano le risposte). Nessuna run lunga: il limite sarebbe il
lettore, e quell'esperimento — encoder e proiettore sbloccati a learning rate basso
con replay — non si improvvisa al buio. In quel caso la GPU resta libera dalle 02:30
circa, ed è la scelta giusta: meglio ferma che spesa su un'ipotesi già smentita.

## Al risveglio, in ordine

```bash
ssh root@<ip-del-nodo> 'cat /root/l3/night_20260913T005001Z/BRANCH.txt; ls /root/l3/night_20260913T005001Z/*/OK'
```

**Corso effettivo**: la decisione è stata presa alle 00:45Z (ramo `selector_sft`,
misure in `/root/l3/DECISION_20260913.json`). Il primo tentativo di pianificare le
tracce è fallito sulla quota delle ampiezze — allocate a conteggi esatti in Z1-bis,
estratte in modo indipendente da me — senza consumare GPU; corretto, la catena è
stata rilanciata alle 00:50Z riusando la decisione già presa. I percorsi qui sotto
si riferiscono alla run `night_20260913T005001Z`.

1. `night_20260913T005001Z/DECISION.json` — il ramo scelto e **tutte** le misure che lo hanno
   deciso, con le soglie registrate accanto.
2. `night_20260912T234945Z/scores/scores/summary.json` — i tre bracci per categoria e per banda.
3. `selectors_.../selfwindow/content/summary.json` — contenuto vero contro
   sostituito e azzerato sulle finestre autonome, più la scala ×2/×4 e la
   stratificazione contenute/mancate.
4. Ramo scelto, `selector_sft`: `traces_w1/plan/summary.json` (88,7 % di proposte wavelet utili), `w1_sft/summary.json` (loss, passi, picco),
   `w1_operational/gate.json` (gate contro va1 e vz1bis) e
   `w1_geometry/geometry/summary.json` — **il numero da battere è 55,3 % di
   contenimento**.
5. Se il ramo è `grpo`: `wgrpo/selection.json` — `selected_step: 0` significa che
   nessuno step ha superato Z1-bis, ed è un risultato, non un errore.

## Limiti dichiarati

Scadenza 10 h dall'avvio: alla scadenza nessuna fase nuova parte, il nodo resta
accesso e libero. Nessun accesso al test 800. Nessuna modifica dell'architettura,
encoder e proiettore restano congelati in entrambi i rami. Costo stimato della
notte: 8-10 h a 0,86 $/h.
