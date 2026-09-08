# Risultati ablazione 1 — `v3tied` (residuo su v1, blocco condiviso, profondità piena) vs va1

Run: A40-24Q Vultr `wavelet-l1`, 07/09 03:21 → 08/09 03:01 UTC (5 epoche, ≈4,7 h l'una), eval
fail-closed del worker fino a 08/09 ~09:30 UTC. Protocollo identico a va1 (seed 42, 4488 × 5,
LoRA r16/α32, lr 1e-4, batch 1, bf16, grad-ckpt, coverage 4488/0 in ogni epoca, best-by-val
primary → epoca 5 in entrambe, test 800 hfpin, eval_errors 0). Encoder: commit `1b0f9fa`
(a init v3 ≡ v1; il loop aggiunge `context_scale·LN(contesto)`). Artefatti:
`integrity/artifacts/v3tied/`, confronto appaiato `integrity/DIAGNOSTICS-v3tied-vs-va1.json`.

## Test 800 (appaiato, v3tied − va1)

| categoria | v3tied | va1 | Δ | significativo (bootstrap 10k) |
|---|---|---|---|---|
| **primary** | **0,2832** [0,247; 0,320] | 0,2744 | +0,009 | no |
| localize | 0,2076 | 0,2410 | −0,034 | **sì** |
| localize_all_channels | 0,2323 | 0,2613 | −0,029 | no |
| classify_with_evidence | 0,0524 | 0,0642 | −0,012 | no |
| measure_magnitude | 0,5929 | 0,5783 | +0,015 | no |
| lead_lag_with_magnitude | 0,3275 | 0,1664 | +0,161 | **sì** |

Scomposizioni:
- localize: IoU sui 161 positivi **0,047** (va1 0,094); acc@IoU0,3 0,068 (0,143); presenza recall
  0,92 (0,88), precisione 0,97 (0,96); accuratezza sui 39 negativi 0,87 (0,85).
- all_channels: accuratezza sui 23 vuoti 0,87 (0,91); F1 sui 77 non vuoti **0,042** (0,067).
- lead_lag: predizioni 67 independent / 26 lead / 7 lag (gold 35/33/32); direzione corretta 44/100
  di cui 31 "independent"; lead/lag con lag entro il 5% solo 6. È la **moda "independent"**
  (il null "sempre independent" vale 0,35), non una capacità nuova; va1 la evitava (43/37/20,
  direzione 30/100).

## Val intera (520), per epoca

| epoca | v3tied | va1 |
|---|---|---|
| 1 | 0,135 | 0,170 |
| 2 | 0,193 | 0,226 |
| 3 | 0,194 | 0,248 |
| 4 | 0,262 | 0,293 |
| 5 | **0,315** | 0,311 |

Sondaggi appaiati sugli stessi 80 esempi di val (in parallelo al training): ep1 0,165 vs 0,124,
ep2 0,172 vs 0,161, ep3 0,151 vs 0,205, ep5 0,287 vs 0,283.

## Lettura (corretta dopo l'ispezione dei checkpoint)

`ts_encoder.context_scale` è rimasto a zero per tutta la run: 0,0001 (ep1), −0,0006 (ep3),
0,0004 (ep5). **Il contesto ricorrente non è mai entrato nei token**: il modello valutato è, di
fatto, va1 replicato (stesso seed, grafo di calcolo diverso, bf16). Le differenze sul test
(+0,009 complessivo, −0,034 su localize, lead_lag caduto sulla moda "independent") sono quindi
**rumore run-to-run del protocollo va1**, non un effetto dell'architettura. Due conseguenze utili:
(i) effetti sotto ≈0,03 per categoria non sono risolvibili con una singola run, e il bootstrap
appaiato sul test non cattura la varianza di training (dichiara "significativa" una differenza
fra due repliche); (ii) il collasso di lead_lag sulla moda può avvenire o no a parità di modello.

Perché la scala non si è mossa: con `context_scale_init = 0` il ramo di contesto riceve gradiente
proporzionale alla scala (≈0) e non si allena, mentre la scala riceve gradiente proporzionale alla
correlazione fra un contesto *non allenato* e la direzione utile (≈0 in media): un deadlock. Lo
schema "ReZero" funziona solo quando il ramo si allena insieme alla scala; qui, con lr 1e-4 e
4488 esempi, la scala ha oscillato intorno a zero (negativa all'epoca 3).

L'ablazione 1 resta quindi **aperta**: per misurare davvero il contesto serve
`context_scale_init` piccolo ma non nullo (es. 0,1: a init v3 = v1 + 10% di contesto, probe di
decodificabilità ancora ≈ v1), oppure un learning rate separato per il ramo. Il config lo
supporta già; costo di una run ≈ $23 / 27 h. La curva di profondità sul checkpoint attuale è
stata interrotta perché priva di senso (contesto inattivo).

## Storia della variante (tre lanci)

1. `cfd5256`-precedente (percettore a soli latenti): 256 token identici a init e a epoca 5
   (coseno 1,0000); val ep1 0,026. Archiviata sul nodo (`~/archive/v3tied_collapsed_init_20260906`).
2. `cfd5256` (fix del collasso, latenti Fourier + prior di località): posizione presente nei token
   ma non decodificabile da Qwen entro un'epoca (sondaggio val ep1 0,053, localize IoU 0);
   probe lineare 1,36-1,45× l'errore di v1. Fermata durante ep2
   (`~/archive/v3tied_latent_cfd5256_partial_20260907`).
3. `1b0f9fa` (residuo su v1, questa run): completa; risultati sopra.

## Cosa resta aperto (decisione)

- Ripetere l'ablazione 1 con `context_scale_init: 0.1` (unica modifica di config): è l'unico
  modo per misurare il contesto ricorrente. Con lo zero-init anche `v3untied` e `v3sdepth`
  resterebbero a scala zero: vanno lanciate solo dopo, con lo stesso init.
- In alternativa chiudere la variante come "non misurabile con zero-init" e rimettere la GPU sul
  Livello 2. Il valore già ottenuto: una replica di va1 che quantifica il rumore run-to-run del
  protocollo (±0,03 per categoria, moda lead_lag instabile).
