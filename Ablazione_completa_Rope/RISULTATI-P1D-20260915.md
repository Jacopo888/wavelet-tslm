# P1-d fase A (run `p1-dA`, 15/09/2026): lettura compatta con LoRA congelata, FAIL su tutti i criteri

Fase A del disegno deciso da Astra (sezione "P1-c e P1-d" di [DECISIONI-ASTRA-20260915.md](DECISIONI-ASTRA-20260915.md)):
LoRA congelata con `requires_grad=False`, frontend/mixer/proiettore a 1e-4, lettura compatta a indici di bin supervisionata
in `<think>` con peso 1,0 e risposta finale 0,5, occupancy ausiliaria 0, RMS per token 0,013, decorrelazione λ 0,01,
orizzonte di 500 passi con warmup 3 % e coseno sulla fase. Con la LoRA ferma l'unico componente che apprende è il
frontend, quindi ogni calo della cross-entropy deve passare per i soft token. Lanciata alle 07:58 locali, conclusa alle
09:24 (500 passi, 2,6 s/esempio).

## Training

La cross-entropy della lettura scende da 2,95 a 0,71 e quella della risposta finale da 2,12 a 0,59. La decorrelazione
(media dei coseni al quadrato entro vista) sale da 0,44 a 0,78, cioè coseno medio ≈ 0,88: i token restano quasi collineari.
L'RMS per token resta fisso a 0,01303 per costruzione.

## Probe di uscita (24 righe di validazione fisse, CE sui campi informativi dei bin)

| passo | TRUE | SUBSTITUTED | ZEROED | vantaggio TRUE su entrambi | CI95 |
|------:|-----:|------------:|-------:|---------------------------:|------|
| 100   | 1,978 | 1,979 | 2,031 | −0,015 | [−0,041; +0,010] |
| 200   | 1,364 | 1,383 | 2,035 | +0,019 | [−0,017; +0,059] |
| 300   | 1,324 | 1,327 | 2,031 | +0,003 | [−0,013; +0,020] |
| 400   | 1,301 | 1,300 | 2,035 | −0,001 | [−0,025; +0,024] |
| 500   | 1,304 | 1,313 | 2,031 | +0,009 | [−0,014; +0,033] |

Accuratezza dei campi dei bin nelle letture GENERATE al passo 500: TRUE 0,021, SUBSTITUTED 0,035, ZEROED 0,000. I token
veri fanno peggio di quelli di un'altra registrazione. **Verdetto FAIL** su tutti i criteri fissati prima della run
(soglia 0,10 nats, CI95 positivo contro entrambi i controlli, accuratezza generata superiore ai controlli).

Lettura: i soft token valgono circa 0,73 nats come semplice presenza (azzerarli costa molto) e zero come contenuto
(TRUE ≡ SUBSTITUTED). Il frontend ha imparato uno spostamento comune degli embedding, cioè prompt tuning, non una lettura.

## Probe lineari sulle rappresentazioni congelate

80 registrazioni di validazione, train e validazione disgiunti per registrazione, ridge con λ scelto su uno split interno;
si riporta la correlazione di Pearson perché l'R² su registrazioni nuove è negativo per forte sovradattamento del probe
(R² sul train 0,999). La prevalenza della classe positiva è 0,051; il caso della accuratezza bilanciata è 0,5.

| rappresentazione | posizione r (init) | posizione r (passo 500) | top-3 (init) | top-3 (passo 500) |
|---|---:|---:|---:|---:|
| H0 descrittori | 0,558 | 0,585 | 0,559 | 0,519 |
| Z uscita mixer | 0,463 | 0,500 | 0,561 | 0,527 |
| SOFT token | 0,535 | 0,463 | 0,531 | 0,538 |

Cinquecento passi con il solo frontend in apprendimento non rendono le rappresentazioni più decodificabili: la posizione
resta debolmente estraibile (arriva da `meta_proj`, che la inietta esplicitamente) e la salienza resta al caso.

## Nota di metodo: la trappola dell'autocast

Il confronto LoRA attiva/disattivata riportato la notte precedente era viziato. `torch.autocast` memorizza il cast bf16 dei
pesi in fp32 per tutta la durata del contesto, e i pesi LoRA sono in fp32: mutarli fra due forward nello stesso contesto non
ha alcun effetto. Riprodotto in isolamento: |Δhidden| 0,000000 nello stesso contesto contro 0,770 con un contesto per
forward. Misura corretta sul checkpoint 400 di P1-c: la LoRA vale 0,336 nats, ottenuti imparando la distribuzione a priori
dei bin. Un audit dell'intero pacchetto ha trovato affetta solo la diagnostica stessa: training, smoke, validazione, gate e
probe aggiornano i pesi fuori dal contesto e riaprono l'autocast a ogni forward. Regressione in
`tests/test_autocast_weight_cache.py`.

## Seguito

P1-e (decisioni Astra, sezione dedicata): si riparte dal checkpoint 500 di questa fase con 200 passi di sola supervisione
per token senza cross-entropy, poi 300 passi con lettura e sostituzione dei token al 25 %. La fase B di P1-d e P2 restano
ferme.

Artefatti: `integrity/artifacts/rope_p1dA_20260915/` (cinque probe di uscita, due probe lineari, log di training completo,
manifesto, checkpoint 500).
