# P1-a (run `p1-main`, 15/09/2026): risultato negativo e diagnosi della scala dei soft token

Prima run P1 completa del prototipo WR-Agent-4B-Wildcard-v1 sul droplet MI300X (config `configs/p1_mi300x.toml`,
4488 esempi, 1122 passi ottimizzatore, ~2,7 s/esempio, lanciata alle 01:00 locali). Fine training 04:25 locali circa.
La run è servita a scoprire un difetto di scala del frontend che rendeva il backbone cieco al contenuto dei soft token.
Le decisioni correttive di Astra sono nella sezione "scala" di [DECISIONI-ASTRA-20260915.md](DECISIONI-ASTRA-20260915.md);
la run corretta è `p1-b` (albero `/workspace/code_p2`).

## Metriche intermedie (validazione rapida, 104 righe stratificate, 768 token)

| passo | primary | measure | lead_lag | localize | classify | 16ch | invalidi |
|------:|--------:|--------:|---------:|---------:|---------:|-----:|---------:|
| 400   | 0,2175  | 0,69    | 0,27     | 0        | 0,03     | 0    | 0        |
| 800   | 0,2596  | 0,85    | 0,26     | 0        | 0        | 0    | 0        |

Le coordinate predette per localize/16ch sono addensate su una posizione fissa (prior), non dipendono dal segnale;
`measure_magnitude` è probabilmente inferibile dalle statistiche dell'header (mediana, scala, min, max), non dai token.

Gate percettivo ridotto (pannello congelato r20, 60 crop) sul checkpoint 400: TRUE 0,082 / SUBSTITUTED 0,076 / ZEROED 0,080,
Δ −0,02 con CI95 [−0,11; +0,04] → FAIL (nessun beneficio del contenuto).

## Diagnosi (checkpoint 800 e inizializzazione, misure su righe reali)

1. **Backbone insensibile all'ordine dei soft token**: |Δlogit| ultimo token 0,30 con token mescolati contro 1,59 con token
   azzerati. I soft token portano un vettore comune, non informazione posizionale/contenutistica leggibile.
2. **Scala dei soft token**: norma 17 (init) → 25 (passo 800) contro 0,66–0,73 degli embedding testuali Qwen (RMS 0,0130
   per dimensione): 30× più grandi. Coseno medio a coppie fra soft token 0,58 (init) → 0,72–0,86 (passo 800): cresce durante
   il training, cioè i token convergono verso una direzione comune.
3. **Origine nel frontend** (60 righe di validazione, init): norma dei descrittori in uscita dalla fusione mediana 61, media
   244, massimo 6.049 (embedding categoriali 22, meta_proj 46). I quattro blocchi da 64 dell'ingresso della fusione valgono
   10.576 / 325 / 12.142 / 17 (proprio / A / D-cov / raw) sulla riga peggiore. Il mixer pre-LN viene scavalcato dal residuo
   (z ≈ H0, coseno invariato) e la LayerNorm d'ingresso del proiettore riporta tutto a scala fissa lasciando la direzione
   comune.
4. **Cause primarie**: (a) colonne lineari dello stem non clippate: DESIGN §6 r.141 presuppone un clipping duro accanto ad
   `asinh`; A |z| p99 95 max 118; per D `z_band = z / s_band` con `s_band = max(1,4826·MAD, 1e-6·max|V|)`: su segmenti
   piatti o quantizzati la MAD di banda è ≈ 0 e `z_band` arriva a 33.410 (D1–D3); (b) stem Linear+SiLU senza normalizzazione,
   con blocchi residui pre-LN a incrementi limitati: la scala dello stem attraversa intatta pooling e fusione; (c) proiettore
   LN→Linear→GELU con uscita a media positiva (componente comune).

## Correzioni (Astra 15/09, applicate in P1-b)

Clip ±16 delle colonne lineari (asinh sui valori non clippati), floor della scala di banda 1e-3 in unità normalizzate,
LayerNorm(64) dopo lo stem, LayerNorm(512) sull'uscita della fusione prima degli identificatori, `meta_proj` con gain fisso
0,25 e embedding categoriali inizializzati a std 0,5, proiettore LN→Linear→GELU→Linear(bias 0) con gain fisso `out_scale`
calibrato su 32 righe reali del train all'RMS degli embedding testuali (0,01303; fattore 0,0645), `soft_rms` registrato per
esempio, veto del gate per categorie principali a zero. Misure a init dopo le correzioni: descrittori 33–35 di norma
(stabili fra righe), rami a norma 8, il mixer modifica i token del 30 %, soft token RMS 0,0133 e norma 0,69.
Smoke P0 esteso PASS (`integrity/artifacts/rope_p0_smoke_scale_20260915/`): i gradienti della sola CE raggiungono tutto
lo stack percettivo (aux head a zero), scala dei soft token entro il target.

## Validazione finale (fine passata, checkpoint 1122, tutte le 520 righe, 768 token, 22,9 min)

| primary | localize | classify | measure | 16ch | lead_lag | invalidi |
|--------:|---------:|---------:|--------:|-----:|---------:|---------:|
| 0,276   | 0,0018   | 0,0034   | 0,732   | 0,278| 0,36     | 0        |

Le categorie che richiedono di leggere il segnale restano a zero; `measure_magnitude` e la parte di `localize_all_channels`
e `lead_lag` sopra lo zero sono compatibili con prior e statistiche dell'header. Training plan completato alle 04:42 locali.
Artefatti: `integrity/artifacts/rope_p1a_20260915/` (log di training, validazioni, manifesti, checkpoint 1122).
Report della run (`REPORT.json`): 4488 esempi, loss 2,31 → 0,40, 2,4 s/esempio; crop del teacher effettivamente visti per categoria (positivi interi / parziali / negativi): classify 327/84/160, localize 281/54/225, 16ch 117/28/132, lead_lag 169/34/72, measure 311/91/158.

## Gate percettivo §12.2, pannello congelato r120 (360 crop su 120 registrazioni), checkpoint 1122

**Verdetto: FAIL.** TRUE − miglior controllo = −0,0133, CI95 [−0,0270; −0,0008] (bootstrap appaiato per registrazione,
10.000 ricampionamenti); 0 % formati invalidi in tutte le condizioni.

| condizione  | primary media | localize | classify | 16ch  |
|-------------|--------------:|---------:|---------:|------:|
| TRUE        | 0,0902        | 0,041    | 0,023    | 0,279 |
| SUBSTITUTED | 0,0941        | 0,044    | 0,027    | 0,285 |
| ZEROED      | 0,0896        | 0,038    | 0,020    | 0,286 |

Il contenuto vero del crop non porta alcun beneficio rispetto a un crop sostituito o azzerato (anzi, leggermente peggio):
il modello risponde da prior e header. Il gate è stato eseguito con l'albero di P1-a (senza il nuovo veto per categorie a
zero, che qui non cambierebbe l'esito). Artefatti: `integrity/artifacts/rope_p1a_20260915/gate_r120/` e `REPORT.json`.
