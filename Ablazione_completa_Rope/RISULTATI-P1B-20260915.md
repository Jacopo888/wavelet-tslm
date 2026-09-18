# P1-b (run `p1-b`, 15/09/2026): correzioni di scala applicate, fermata alla sentinella del passo 400

Run P1 da zero con tutte le correzioni di scala di Astra (sezione "scala" di
[DECISIONI-ASTRA-20260915.md](DECISIONI-ASTRA-20260915.md)): clip ±16, floor di banda 1e-3, LayerNorm dopo stem e
fusione, gain dei metadati 0,25 ed embedding a std 0,5, proiettore a due lineari con gain fisso calibrato su 32 righe
reali del train (fattore 0,0645, RMS 0,01303), veto del gate per categorie a zero. Lanciata alle 04:06 locali dall'albero
`/workspace/code_p2`, fermata alle 06:05 al passo ~540 per decisione di Astra (D1 della sezione "sentinella").

## Cosa è cambiato rispetto a P1-a

- A init i descrittori hanno norma 33–35 stabile fra registrazioni (P1-a: mediana 61, massimo 6.049), i rami norma 8,
  il mixer modifica i token del 30 % (P1-a: scavalcato), i soft token hanno RMS 0,013 e norma 0,69 come gli embedding.
- Le loss di training sono uguali o inferiori a P1-a allo stesso punto (localize a crop fisso 0,46 contro 1,30 dopo 400 esempi).
- Smoke esteso PASS: i gradienti della sola CE raggiungono tutto lo stack percettivo; scala entro il target.

## Sentinella al checkpoint 400 (gate r20, 60 crop su 20 registrazioni)

**FAIL, ma in modo diverso da P1-a.** TRUE 0,059 / SUBSTITUTED 0,118 / ZEROED 0,082, Δ −0,071 con CI95 [−0,152; −0,011].
Per categoria (TRUE / SUBST / ZERO): localize 0,055 / 0,064 / 0,031; classify 0,039 / 0,072 / 0,036; 16ch 0,093 / 0,267 / 0,227
(collasso di categoria segnalato per 16ch). Il contenuto vero ora cambia le risposte, in peggio: il backbone è diventato
sensibile ai token, senza aver imparato a leggerli.

Validazione rapida (104 righe, 768 token) sul checkpoint 400: primary 0,2174 (P1-a allo stesso passo 0,2175), localize /
classify / 16ch a zero, measure 0,68, lead_lag 0,33, 0 invalidi; coordinate predette addensate attorno al 30 % della
lunghezza (prior posizionale). Criterio di abort di Astra (prior fisse + nessun beneficio del contenuto) soddisfatto.

## Diagnosi al checkpoint 400 (24 righe di validazione)

| grandezza          | init            | checkpoint 400   |
|--------------------|-----------------|------------------|
| descrittori H0     | norma 34, cos 0,62 | norma 35,5, cos 0,58 |
| mixer z            | norma 35, cos 0,64 | norma 46, cos 0,72   |
| soft token         | norma 0,69, RMS 0,013, cos 0,63 | **norma 2,0, RMS 0,040, cos 0,92 (0,94 entro vista)** |

Deriva dell'RMS dei soft token: 0,013 → 0,030 (passo 100) → 0,035 (200) → 0,041 (400). Con gain fisso e pesi liberi il
proiettore impara una mappa quasi di rango 1 con norma crescente: la componente comune riemerge dall'ottimizzazione, non
più dalla scala del frontend. Lettura: con lr frontend 1e-4 e LoRA 2e-5 la soluzione facile per la CE finale (sparsa) è un
blocco di token quasi identici che agisce da bias appreso.

## Decisione (Astra, sezione "sentinella")

P1-b fermata; P1-c da zero con: segmento di "lettura" supervisionato dentro `<think>` (peso 0,5) generato dal modello anche
in validazione e gate; RMS per token fissato a 0,013 senza parametri; penalità di decorrelazione λ = 0,01 sulla media dei
coseni al quadrato entro vista; lr frontend/proiettore 2e-5 e LoRA 1e-4; soglie della sentinella fissate a priori
(proseguire solo con Δ ≥ +0,02 e CI95 > 0; coseno entro vista ≤ 0,80 obiettivo, > 0,90 allarme).

Artefatti: `integrity/artifacts/rope_p1b_20260915/` (checkpoint 400 con pesi allenabili, log di training, gate r20,
validazione rapida, calibrazione del proiettore, manifesto).
