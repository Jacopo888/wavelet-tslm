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

## Lettura

Il contesto ricorrente multi-scala (blocco condiviso, tutti i livelli letti) **non aiuta**: la
metrica complessiva è pari (differenza non significativa), la localizzazione fine peggiora in modo
significativo (IoU sui positivi dimezzato) e il "guadagno" su lead_lag è il collasso sulla moda
"independent". La percezione v1 (selezione dura dei coefficienti) resta il riferimento.

Ipotesi sul perché: il residuo, con `context_scale` libero e LayerNorm sul contesto, perturba i
token base (valore + posizione) su cui Qwen localizza; il modello impara a usare il contesto per
statistiche globali (presenza, moda lead_lag) e perde precisione posizionale.

## Storia della variante (tre lanci)

1. `cfd5256`-precedente (percettore a soli latenti): 256 token identici a init e a epoca 5
   (coseno 1,0000); val ep1 0,026. Archiviata sul nodo (`~/archive/v3tied_collapsed_init_20260906`).
2. `cfd5256` (fix del collasso, latenti Fourier + prior di località): posizione presente nei token
   ma non decodificabile da Qwen entro un'epoca (sondaggio val ep1 0,053, localize IoU 0);
   probe lineare 1,36-1,45× l'errore di v1. Fermata durante ep2
   (`~/archive/v3tied_latent_cfd5256_partial_20260907`).
3. `1b0f9fa` (residuo su v1, questa run): completa; risultati sopra.

## Cosa resta aperto (decisione)

- Ablazione 2 (`v3untied`) e 3 (`v3sdepth` + curva di profondità): ≈$23 e 27 h ciascuna sulla
  A40; con il risultato dell'ablazione 1 misurerebbero varianti di un contesto che oggi non aiuta.
- Curva di profondità sul checkpoint `v3tied` ep5 (`TSLM_V3_DEPTH` = 1, 3, 6, 11 sulla val, in
  corso l'08/09): dice se leggere meno livelli riduce il danno.
- Alternative: chiudere qui la variante v3 come risultato negativo documentato (percezione v1 +
  contesto ricorsivo non migliora AnomalyXL a parità di budget) e rimettere la GPU sul Livello 2.
