# Pre-allineamento in linguaggio naturale, prima della SFT

Disegno, non ancora lanciato. Deciso con Fable il 18/09 mentre girava `va1ethpos`; da eseguire **solo dopo** il
suo esito, perché se `va1ethpos` passa questo diventa un miglioramento e non più una diagnosi.

## Perché, e perché non è P1-d

I modelli visione-linguaggio che funzionano usano due fasi. LLaVA congela il modello linguistico e allena **solo
il proiettore** su 558.000 coppie facili, con l'unico scopo di portare le feature visive nello spazio di embedding
del LM; solo dopo fa instruction tuning. Noi la prima fase non l'abbiamo mai fatta nella sua forma vera: P1
allena frontend, mixer, proiettore e LoRA **insieme**, direttamente sul compito difficile, con 4.488 esempi.

La fase A di P1-d sembra averlo già provato — LoRA congelata, frontend a 1e-4 — e ha dato FAIL. Ma il bersaglio
erano **indici di bin**, cioè un formato che un LM congelato non ha mai visto in vita sua. Chiedere a un modello
congelato di emettere `bin 37` non misura se il proiettore sappia parlare la sua lingua: misura se sappia
inventarne una nuova senza poter cambiare i propri pesi. È il contrario di quello che fa l'allineamento.

**La differenza è tutta nel bersaglio.** Qui il bersaglio è **linguaggio naturale**, cioè esattamente ciò che il
LM congelato sa già produrre.

## Disegno

**Fase A — allineamento.** Backbone congelato **e LoRA congelata**. Allenabili solo frontend, mixer e proiettore.
Bersaglio: una descrizione del segnale generata dal teacher dalle annotazioni gold, in prosa. Per esempio:

    Nel canale ch_00 c'e' un picco intorno al campione 12300, ampiezza circa 8 sigma sopra la linea di base.
    Il resto della registrazione e' regolare.

La cross-entropy sta solo su quella descrizione. Nessun JSON, nessun indice di bin, nessuna coordinata da
emettere in formato rigido.

**Fase B — SFT.** La ricetta di `va1eth` così com'è, partendo dai pesi percettivi della fase A.

## Perché dovrebbe funzionare dove P1-dA ha fallito

Un LM congelato ha un solo modo di ridurre la perdita su «picco intorno al campione 12300»: **far sì che i soft
token spingano le sue probabilità verso quelle parole e quelle cifre**. Non può adattarsi lui, quindi tutta la
pressione del gradiente va sul proiettore. È la definizione di allineamento, ed è il motivo per cui LLaVA congela
il LM in fase 1 invece di allenare tutto insieme.

Il bersaglio in prosa ha anche un vantaggio che gli indici non avevano: contiene le **cifre** della posizione in
forma scritta, cioè esattamente la conversione che il modello poi deve saper fare nel compito vero.

## Cosa serve costruire

1. **Il teacher delle descrizioni.** Deterministico, dalle annotazioni gold: canale, posizione, ampiezza in
   sigma, tipo se disponibile. Con variazioni di forma per non insegnare una sola frase. Va scritto e testato
   sulle 4.488 righe del train, con un controllo che nessuna descrizione contenga informazione che il segnale
   non porta.
2. **Il modo di training `alignment`** in `build_sequence`: prompt ETH verbatim, risposta = descrizione, e il
   congelamento di LoRA oltre al backbone.
3. **Il criterio**, dichiarato prima: alla fine della fase A, il **gate 208** deve mostrare vero > sostituito con
   intervallo sopra lo zero **sulla descrizione**. Se il gate resta piatto anche qui, il proiettore non ha
   imparato a parlare al LM nemmeno con la sua lingua e il bersaglio non c'entra.

## Costo

Una passata di allineamento sui 4.488 = 4.488 aggiornamenti ≈ 3 ore ≈ **6 $** a 1,99 $/h. Più il gate, 1,2 $.
La fase B è la SFT normale, 13 $. Totale circa **20 $**.

## Ordine rispetto a `va1ethpos`

Se `va1ethpos` **passa** — IoU sui positivi sopra zero con gate non piatto — la leva della posizione copiabile è
quella giusta e il pre-allineamento diventa un miglioramento da valutare dopo, non una priorità.

Se `va1ethpos` **fallisce** con gate piatto, allora sono state escluse tutte e quattro le spiegazioni
(prompt, selezione percettiva, geometria dell'attenzione, codifica della posizione) e resta questa: **il
proiettore non ha mai imparato a parlare al modello linguistico**, perché non gliel'abbiamo mai chiesto da solo.
