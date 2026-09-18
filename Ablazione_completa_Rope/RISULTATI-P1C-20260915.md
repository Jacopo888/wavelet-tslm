# P1-c (run `p1-c`, 15/09/2026): lettura supervisionata, fermata alla sentinella del passo 400

Run P1 da zero con le decisioni di Astra della sezione "sentinella" di [DECISIONI-ASTRA-20260915.md](DECISIONI-ASTRA-20260915.md):
segmento di lettura programmatica supervisionato in `<think>` (peso 0,5; `src/wavelet_rope/reading.py`) e generato dal
modello in validazione e gate; RMS per token fissato a 0,013 senza parametri; penalità di decorrelazione λ = 0,01;
lr frontend/proiettore 2e-5 e LoRA 1e-4 (`configs/p1c_mi300x.toml`, `prototype.spec.p1c.json`). Smoke esteso PASS
(`integrity/artifacts/rope_p0_smoke_p1c_20260915/`). Lanciata alle 06:00 locali, fermata alle 07:35 al passo ~470.

## Training (primi 400 passi)

CE della lettura 1,92 → 0,30 e del finale 2,03 → 0,40 già entro i primi 100 passi; decorrelazione 0,43 → 0,46 (coseno ≈ 0,68,
sotto l'allarme 0,90); RMS dei soft token fisso a 0,01303; 2,6 s/esempio; letture di ~200 caratteri.

## Sentinella al checkpoint 400 (gate r20, 60 crop su 20 registrazioni)

**FAIL con CI inconcludente.** TRUE 0,099 / SUBSTITUTED 0,114 / ZEROED 0,085, Δ −0,015, CI95 [−0,057; +0,016].
Per categoria (TRUE / SUBST / ZERO): localize 0,030 / 0,011 / 0,0003 (n 24); classify 0,011 / 0,004 / 0,003 (21);
16ch 0,333 / 0,433 / 0,333 (15, collasso di categoria). 0 % invalidi.

Validazione rapida (104 righe, lettura generata dal modello, 768 token finali): primary 0,256 (P1-a/b 0,2175), localize 0,
classify 0, 16ch 0,07, measure 0,79, lead_lag 0,33; 0 invalidi; nessuna chiusura forzata; letture di ~107 token.
Le letture generate sono quasi identiche fra registrazioni della stessa lunghezza (per L = 16384: "energia in ch_00
[15872,16128), [15616,15872), [15360,15616)") e la risposta finale copia quei bin (start ≈ 15800). Il teacher non è
sbilanciato: i bin nominati nelle letture di training sono distribuiti uniformemente sui decili, come gli inizi degli
eventi gold; il prior "ultimi bin" è del modello.

## Diagnostica decisiva (teacher forcing, 24 righe di validazione, checkpoint 400)

| condizione dei soft token | CE lettura | CE finale |
|---------------------------|-----------:|----------:|
| veri                      | 0,1702     | 0,3105    |
| mescolati                 | 0,1706     | 0,3109    |
| azzerati                  | 0,1701     | 0,3123    |

Il backbone ignora completamente i soft token, sia per la lettura sia per il finale: la lettura (≈100 token, ~10 numerici)
viene predetta da template, header e prior. Con la LoRA libera a lr alto l'LLM ha imparato a rendere invisibili token
inizialmente rumorosi prima che il frontend, più lento, li rendesse informativi; l'obiettivo denso non basta se il testo
di lettura è quasi tutto boilerplate. In P1-b (frontend 1e-4, LoRA 2e-5) i token cambiavano invece le risposte, in peggio.

## Diagnostica causale (Astra D3, checkpoint 400, 24 righe val)

Contributo del segnale all'uscita dell'attenzione (ingresso di `o_proj`, dopo il gate) alle posizioni che predicono i campi
dei bin: 13–26 % per layer (3: 0,17, 7: 0,26, 11: 0,21, 15: 0,21, 19: 0,17, 23: 0,16, 27: 0,14, 31: 0,13). |Δlogit| medio fra
token veri e azzerati alle stesse posizioni 0,26. Il canale meccanico testo→segnale quindi esiste: il segnale entra
nell'attenzione delle otto layer e muove i logit, ma non nella direzione che riduce la CE dei bin (vantaggio dei token veri
−0,0008). Il frontend non ha imparato a codificare la posizione in modo decodificabile dal backbone; la fase A di P1-d
(frontend 1e-4, LoRA congelata, lettura compatta a peso 1,0) è il test diretto di questa capacità. Log:
`diag_lora_causal_step400.log`.

**Errata (trovato alle 08:45).** Il confronto LoRA attiva/disattivata contenuto in quel log NON è valido: entrambe le
condizioni venivano eseguite dentro un unico contesto `torch.autocast`, che memorizza il cast bf16 dei pesi **fp32**. I pesi
LoRA sono fp32 (`backbone.py` li converte esplicitamente), quindi il secondo forward riusava il cast vecchio e la differenza
risultava esattamente nulla. Riprodotto in isolamento: stesso contesto |Δhidden| = 0,000000; un contesto per forward oppure
`cache_enabled=False` |Δhidden| = 0,770. Anche la prima lettura delle norme LoRA era fuorviante: per elemento |B| ≈ 0,002,
ma la norma di Frobenius di ΔW = 2·B·A è 0,37–0,80 per modulo, cioè circa il 3 % del peso base. **La LoRA quindi si muove ed
è attiva nel forward.** Le condizioni sui soft token (veri / mescolati / azzerati) restano valide: agiscono su un tensore
calcolato, non su un peso in cache.

**Misura corretta** (12 righe val, un contesto autocast per forward), CE sui campi dei bin:

| condizione        | token veri | token azzerati |
|-------------------|-----------:|---------------:|
| LoRA attiva       | 0,2967     | 0,2952         |
| LoRA azzerata     | 0,6322     | 0,6167         |

La LoRA vale +0,336 nats (dimezza la CE dei bin): ha imparato molto, ma **senza i token**, cioè la distribuzione a priori
degli indici di bin. I token restano ininfluenti in entrambi i regimi (−0,0015). Questo conferma la diagnosi e giustifica
il disegno della fase A di P1-d: congelando la LoRA, l'unica via per abbassare la CE della lettura passa per i soft token,
e il controllo SUBSTITUTED (token reali di un'altra registrazione) separa il contenuto dal semplice bias dei token.

Audit del repo su questo schema (Codex Luna, effort max): l'unico punto affetto era la diagnostica stessa, ora corretta con
un contesto autocast per forward e un test di regressione (`tests/test_autocast_weight_cache.py`). Training, smoke,
validazione ufficiale, gate percettivo e probe non sono toccati: in tutti gli aggiornamenti dei pesi avvengono fuori dal
contesto e ogni forward lo riapre. Le conclusioni sperimentali precedenti restano quindi valide.

## Decisione

P1-c fermata per la regola D3 (Δ ≤ 0 con prior fisse). Proposta sottoposta ad Astra: P1-d in due fasi (A: LoRA congelata,
frontend 1e-4, lettura compatta e quasi solo numerica, uscita dalla fase solo se la CE con token veri batte quella con
token azzerati di almeno 0,10; B: LoRA 2e-5 con il mix P1 completo e la sentinella al passo 400).
_Risposta di Astra e seguito: vedi la sezione corrispondente di DECISIONI-ASTRA-20260915.md._

Artefatti: `integrity/artifacts/rope_p1c_20260915/` (checkpoint 400, gate r20, validazione rapida, log di training, manifesto).
