# Brief per Astra — 17/09/2026, sera

Rispondi conciso e numerato.

## 1. L'header delle statistiche era nostro, non del benchmark

Verificato nel codice di riferimento, non nel testo del paper. Il prompt di AnomalyXL/OpenTSLM
(`qa_dataset.py:114-139`, `chatts.py:324-366`, e `wavelet.py` conferma che `va1` passa di lì) è:

```
{system prompt del benchmark}
The following is channel {nome}: {256 token}
Question: {domanda} {risposta}{EOS}
```

Nessuna statistica. Le uniche cifre che il benchmark regala sono `L`, e `C` sul multicanale, dentro la `question`
stessa: `(Here L = 16384.)`. E **nessun chat template**: la stringa è piatta, tokenizzata così com'è; Qwen3.5 non
aggiunge niente (nessun BOS, `add_special_tokens` è un no-op), quindi l'unico token speciale dell'intera sequenza è
l'EOS che chiude la risposta. La patch `va1` è byte-identica all'originale.

Conseguenza: **il confronto `va1` contro `va1r` era confondato**. `va1r` aveva mediana, scala, min, max, conteggio
dei validi, numero di token e griglia di lettura che `va1` non ha mai avuto. E `va1`, senza niente di tutto questo,
faceva 0,578 su magnitudo e 0,0606 di IoU sui positivi di localize.

## 2. `va1rh` — header ridotto (via min e max), 2 epoche, tutto il resto fissato

| | ep. 1 | ep. 2 |
|---|---:|---:|
| measure_magnitude | 0,0469 (da 0,7546 di `va1r`) | 0,0471 |
| localize | 0,0000 | 0,0000 |
| classify_with_evidence | 0,0000 | 0,0000 |
| localize_all_channels | 0,2667 | 0,2667 |
| lead_lag_with_magnitude | 0,0346 | **0,3333** |
| **primary** | 0,0552 | 0,0983 |

`lead_lag` a 0,3333 batte lo 0,1454 di `va1r` a cinque epoche e lo 0,1664 di `va1` sul test 800.

**Gate sui 208 positivi, epoca 2** — vero / azzerato / sostituito:

| | CE | primary |
|---|---:|---:|
| token veri | 1,12738 | 0,00181 |
| token azzerati | 1,12657 | 0,00000 |
| token di un'altra registrazione | 1,12680 | 0,00111 |

Piatto. **Togliere la scorciatoia ha fatto perdere i punti gratis senza produrre lettura.**

## 3. Massa di attenzione sui soft token

24 righe, ultima posizione del prompt, pesi ricalcolati dalle stesse Q\*/K\* della wildcard (i tre backend sono fusi e
non li restituiscono), media sugli otto layer avvolti.

| checkpoint | `va1r` segnale | `va1rh` segnale |
|---|---:|---:|
| 4488 (ep. 1) | 0,0105 | **0,0163** |
| 8976 (ep. 2) | 0,0206 | 0,0219 |
| 13464 (ep. 3) | **0,0447** | — |
| 17952 (ep. 4) | 0,0299 | — |
| 22440 (ep. 5) | 0,0262 | — |

Tre letture:

1. Togliere min e max **alza** l'attenzione sul segnale del 56 % alla prima epoca, ma il vantaggio è quasi sparito
   alla seconda.
2. **In `va1r` l'attenzione sul segnale non cala in modo monotono**: sale fino all'epoca 3 e poi scende. L'ipotesi
   «le prime epoche sporcano l'attention e la comprimono progressivamente» non è confermata in quella forma.
3. La quota header **non è confrontabile fra bracci**: il confine lo stimo con un rapporto sui caratteri e l'header
   di `va1rh` è più corto. La quota segnale sì, usa la maschera `is_signal` esatta.

## 4. `va1r` a 5 epoche, per il confronto

Primary 0,319. Ma la scomposizione: `localize` 0,2692 totale, **0,0000 sui positivi**, 0,8750 sui negativi;
multicanale 0,2667 totale, **0,0000 sui positivi**, 1,0000 sui negativi. Il punteggio è rilevazione dell'assenza —
selettiva, non cieca (zero falsi «vuoto» su 18 positivi), ma non è localizzazione. Gate a epoca 3: FAIL, vero meno
sostituito **−0,0037** con CI95 [−0,0100; +0,0001].

## 5. `va1eth` — lanciato alle 18:28, in corso

Il prompt del benchmark verbatim, **forma compresa**: stringa piatta, nessun chat template, nessun turno, nessuna
statistica, un blocco `The following is channel X:` per canale con i suoi token subito dopo.

La quota coincide già: 4 famiglie × 64 = **256 token per canale**, esattamente come il riferimento; 16 canali = 4096,
cioè il `max_global_ts_tokens` già dichiarato.

Verificato su registrazioni reali prima di lanciare: i blocchi **partizionano** i soft token (ognuno in un blocco
solo), i token del blocco `slot` sono esattamente quelli con `channel_slot == slot` confrontati sugli embedding
scritti in sequenza, `tau` segue la stessa selezione, il prompt di generazione è **prefisso byte-esatto** della
sequenza di training, e la supervisione copre esattamente `" {risposta}<|im_end|>"`. 168 test verdi.

2 passate, 8.976 aggiornamenti, ~6,3 ore, per essere confrontabile con `va1rh`.

## Domande

1. **P1 ora è piatto, P2 reintrodurrebbe il chat template che P1 non ha mai visto.** Va bene così — Qwen il template
   ce l'ha dal suo post-training e P2 ha 13.464 aggiornamenti per assorbirlo — o vuoi che anche la parte diretta di
   P2 resti piatta?
2. **Con gate piatto in `va1rh`, qual è la tua previsione per `va1eth`, e quale osservazione ti farebbe cambiare
   diagnosi?** In particolare: la corrispondenza nome-canale → token, che finora non davamo, è una variabile nuova
   solo per il multicanale o pensi che cambi anche il singolo canale?
3. **Test 800**: lo leggiamo su `va1eth` a fine run — una volta sola, come da protocollo L3 — o restiamo su val520
   più gate208? Serve per stare nella stessa tabella dello 0,2744 di `va1`.
4. **Budget**: restano circa 20 $, la run ne costa circa 17. Se `va1eth` all'epoca 1 ha gate piatto come `va1rh`,
   continuiamo fino all'epoca 2 o fermiamo e spendiamo il resto altrove?
