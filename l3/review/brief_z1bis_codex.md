# Brief per Codex — Z1-bis: pilot SFT con una sola azione opzionale e gate percettivo

Stato: 12/09/2026. Scritto seguendo `CODEX-REVIEW-ZOOM-20260912.md` §7.2 e
`CODEX-GATE-FOLLOWUP-20260912.md` §4, adottati per ordine di Jacopo. Codex scrive il codice,
Claude controlla il diff, esegue e riporta. **Nessun training parte prima del diff rivisto.**

## 1. Perché

Il gate a pesi fissi ha stabilito che il guadagno del crop oracle non è attribuibile al suo
contenuto: con header identico, sostituire o azzerare la finestra non cambia il punteggio
(Δ +0,006 CI[−0,015;+0,030] e −0,003 [−0,027;+0,022] su 208 positivi), mentre spostare il gold
fuori dal centro lo fa crollare (−0,357). Le tracce di Z1 avevano il gold quasi sempre al centro:
«rispondi al centro delle coordinate ricevute» era la soluzione ottimale del compito insegnato.

Z1-bis serve a rispondere a **una sola domanda**: *quando la scorciatoia del centro è resa
impossibile, il contenuto della finestra porta un vantaggio misurabile?*

Baseline appaiate già disponibili sugli stessi 208 ID positivi (`l3/tools/subset_baseline.py`):
`va1` senza zoom 0,0612 · `vz1` senza zoom 0,0264 · policy 0,0134 · oracle fuori centro 0,1226 ·
oracle centrato 0,4792.

## 2. Disegno (scelta minima, invariante architetturale)

`va1` → encoder e proiettore **congelati** → header **veritiero** (canale, start, end) → gold in
**posizione interna variabile** → risposta in **coordinate assolute**. Nessuna v2, nessuna RoPE
temporale, nessun tocco a v3 o a `kaggle/patch/`. Una sola azione opzionale per episodio (il kit
attuale supporta una sola azione: non introdurre il secondo zoom qui).

La risposta in coordinate locali con conversione deterministica dell'ambiente è un'ablazione
**successiva**, non parte di questo pilot; se in futuro la si prova, va isolata tutta la vista
locale (anche il testo dell'azione precedente contiene le coordinate) e nel multicanale non si
può applicare l'offset del crop agli eventi letti negli altri canali.

## 3. Tracce (regole fissate PRIMA del run, nessun fallback silenzioso)

1. **Positivi con intervallo gold** (localize presente, classify con anomalia, all_channels):
   ampiezza della finestra estratta da un insieme dichiarato (proposta: {1024, 2048, 4096}
   con pesi dichiarati), e **start campionato uniformemente fra quelli che contengono l'evento
   intero** e stanno nei bordi della serie. Così la posizione interna del gold è variabile per
   costruzione. Mantenere una quota di casi centrati **non dominante** (proposta: ≤ 15 %).
   Registrare nel manifest la distribuzione della posizione interna (istogramma per categoria e
   per banda di lunghezza) e verificarla a posteriori: se la media della posizione interna cade
   fuori da [0,35; 0,65] o la deviazione è sotto 0,15, il generatore ha un difetto e il run
   fallisce chiuso.
2. **Ramo diretto**: measure_magnitude, lead_lag e i casi **senza** anomalia rispondono senza
   azione. Nota: l'assenza è una label del train, non un'informazione disponibile in inferenza;
   quindi il modello deve imparare a scegliere il ramo diretto, e va misurato quanto sbaglia
   (falsi negativi) nella valutazione operativa.
3. **Replay del formato originale**: una quota dichiarata di esempi nel prompt va1 originale
   (senza la frase dell'azione), per non perdere il ramo senza crop: nel pilot precedente `vz1`
   senza zoom era scesa a 0,0264 contro 0,0612 di va1 sui positivi, Δ −0,0349 [−0,060;−0,011].
4. **Niente pseudo-recupero**: vietato «crop inutile → risposta globale gold». Se la vista locale
   non basta, la traiettoria deve mostrare l'uso dei canali originali (risposta diretta) o una
   correzione esplicita; altrimenti si insegna a ignorare il crop.
5. Percentuali per categoria **esplicite** nel config e verificate nel summary (come la coverage
   di L1): un contatore per sorgente, per categoria, e il fallimento chiuso se una quota
   dichiarata non è rispettata entro ±2 punti.

## 4. Gate percettivo (dopo il pilot, a pesi fissi, stessi ID)

Riusare `l3/tools/diag_crop_content.py` estendendolo: valutare **lo stesso checkpoint** sugli
**stessi 208 ID** del gate attuale, nelle quattro condizioni

1. **fuori centro vero** (header veritiero, gold in posizione interna casuale, crop reale);
2. **fuori centro sostituito** (header *identico* a 1, contenuto di un'altra finestra senza eventi);
3. **fuori centro azzerato** (header *identico* a 1, contenuto azzerato);
4. **senza crop**, nel formato nuovo (ramo diretto), non col vecchio prompt.

Criterio percettivo, dichiarato ora: **Δ appaiato (1) − max((2),(3)) ≥ +0,05 con CI95 bootstrap
10k che esclude lo zero**, riportato anche per categoria (localize, classify, all_channels) e non
solo in aggregato. Header identici fra 1, 2 e 3 per costruzione. Anche superandolo, canale e
regione dell'oracle restano informazione privilegiata: il gate misura il contributo del
contenuto **dato** un crop utile, non l'utilità autonoma.

## 5. Gate operativo (separato)

Policy autonoma, stesso formato, **val520 intera**: primary per categoria contro i riferimenti
appaiati (`va1` senza zoom, `vz1` precedente), più tasso di azione, validità, risposte vuote,
falsi negativi sui negativi e uso del ramo diretto. **Nessuna soglia 0,301 riciclata sul
sottoinsieme dei positivi**; le soglie di questo gate vanno dichiarate nel config prima del run.

## 6. Deliverable

`l3/zoom_sft.py` (o un modulo nuovo `l3/zoom_traces.py` + patch minima), `l3/configs/l3_zoom_z1bis.yaml`
(quote per categoria, ampiezze, quota centrata, quota replay, seed), estensione di
`l3/tools/diag_crop_content.py` con le condizioni «fuori centro» a header identico,
`l3/node_l3_z1bis.sh` (tracce → SFT 1 epoca → gate percettivo → gate operativo, fail-closed,
tmux con redirezione su file, `PYTHONPATH` alla radice), test CPU (distribuzione della posizione
interna, quote per categoria, header identici fra le condizioni, divieto del pseudo-recupero),
`PROTOCOL-L3.md` §Z1-bis con criteri e soglie **dichiarati prima**.

Budget stimato: generazione tracce e SFT 1 epoca ~10 h, gate percettivo ~0,8 h, gate operativo
~0,5 h → ~11-12 h A40 (~10 $). Il nodo è a 0,86 $/h; crediti residui stimati ~130 $.

## 7. Fuori perimetro

Due zoom, GRPO, test 800, risposte relative, v2/RoPE, sblocco di encoder o proiettore. Si
decidono dopo i due gate.
