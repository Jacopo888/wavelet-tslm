# Run ufficiale wavelet — risultati (Modal A100-40GB, bf16)

Ricetta del paper: full train set (4.488) × 5 epoche, batch 1, LoRA r16/α32, LR 1e-4 linear+warmup 3%, selezione best per val loss. Eval sul gold test **completo (800 righe, seed 42)** — confrontabile riga-per-riga con la Table 1.

## Storia del run (per onestà del report)

- Epoche 1-3: complete MA con ~554 batch/epoca in OOM silenzioso = **quasi tutta la porzione a 16 canali esclusa** (run senza gradient checkpointing).
- Epoca 4: troncata al 61% da crash VRAM (ripresa da checkpoint con stato optimizer).
- Epoca 5: **completa e pulita** (grad ckpt ON, 0 batch persi) — prima epoca con le celle multicanale. Val loss 0.5144 → **0.3741** (−27%!): il salto è l'effetto 16ch + coda LR.
- Top-up finale: +1.988 step su dati misti per completare i 22.440 step ufficiali (schedule LR fino in fondo) → **eval A** definitiva.
- Caveat: celle 16ch viste ~1,4 volte contro le 5 della ricetta pulita → all_channels sottostimato.

## ESITO FINALE (2026-08-29)

Top-up completato: **22.440/22.440 step ufficiali**, schedule LR percorsa fino in fondo. Selezione best-by-val: epoca 5 (val 0.3741) batte lo stato post-top-up (val 0.3787) → **il modello ufficiale della run è il checkpoint di epoca 5, e la tabella sotto — già calcolata su quel checkpoint — è l'eval definitiva (eval A)**. Il `train.py` upstream è crashato su una print cosmetica dopo il training del top-up (bug corretto nel patch, `best_val_metric` None dopo resume senza miglioramento); nessun impatto sui risultati.

## Eval definitiva — modello best (epoca 5), 800/800 righe scorate, all_channels 100/100 ✓

Test loss 0.383 · **test primary 0.238**

| Task (precise) | **Wavelet (nostra)** | Miglior TSLM SFT (Table 1) | Classical | TimeRLM (agente) |
|---|---|---|---|---|
| localize | **0.158** | 0.003-0.008 (~20-50×) | 0.160 (≈ pari) | 0.656 |
| classify_with_evidence | 0.011 | 0.000 | 0.022 | 0.745 |
| measure_magnitude | **0.522** | 0.092 (5,7×) | 0.685 | 0.554 (≈ raggiunto) |
| localize_all_channels | **0.180** | 0.230 OpenTSLM (ChatTS: OOM) | 0.549 | 0.551 |
| lead_lag_with_magnitude | **0.340** | 0.350 (≈ pari) | 0.108 (3,1×) | 0.978 |

(Colonne Toto della Table 1 da riverificare sul PDF prima di citarle — nota di fetch.)

## Lettura

1. **localize 0.158 è il titolo**: i TSLM SFT pubblicati stanno a 0.003-0.008 — non localizzano affatto. La variante wavelet è il primo percettore SFT di questa classe che localizza, alla pari del baseline classico. La tesi ("la rappresentazione multirisoluzione conserva il *dove* che i latenti compressi buttano") ha il suo numero.
2. **measure_magnitude 0.522**: 5,7× il miglior TSLM, quasi al livello dell'agente TimeRLM (0.554) che usa GPT+codice.
3. **lead_lag 0.340** pareggia il miglior TSLM e triplica il classico.
4. **all_channels 0.180** poco sotto OpenTSLM (0.230) NONOSTANTE l'esposizione 16ch a ~1,4 passate (vs 5); ChatTS pubblicato lì segna OOM. Con esposizione piena c'è margine evidente.
5. classify resta terreno dell'agente (tutti i percettori ~0): coerente col disegno Livello 1 vs Livello 2.

## Audit riproducibilità benchmark (2026-08-29)

**Gold set rigenerato vs shard HF rilasciato** (confronto riga-per-riga, md5 su valori arrotondati a 1e-6): 765/800 righe bit-identiche — TUTTE le righe di localize, classify, measure_magnitude e all_channels. Le 35 diverse sono esattamente le 35 righe "independent" di lead_lag: domanda, answer gold, coupling_kind/lag identici; cambia solo la realizzazione del segnale (21 righe: stesso seed, finestre di iniezione altrove; 14 righe: seed derivato e kind diversi, e HF ha `n_channels_a/b=None` dove il codice pubblicato scrive `1/1`). Causa: lo shard HF (caricato una volta, 30 lug) è stato generato con una versione interna del generatore precedente allo squash "initial commit" del repo (2 soli commit pubblici) → il codice pubblicato non ha mai riprodotto quelle 35 righe. Impatto: l'errore-lag si misura solo sulle 65 righe accoppiate (bit-identiche); per le independent lo scorer valuta solo la direzione (gold identico) → lead_lag 0.340 resta confrontabile; le altre 4 categorie sono sul benchmark esatto.

**Stabilità del classical baseline** (teste HistGradientBoosting senza `random_state`, ES con split casuale — due run sugli stessi identici feature cache): localize 0.200→0.198, classify 0.024→0.020, all_ch 0.550→0.548 (stabili); MMag 0.916→0.770 (±0.15!), lead_lag 0.132→0.177 (±0.05); primary 0.370→0.338. Vs Table 1 (0.160/0.022/0.685/0.549/0.108): localize ~0.20 nostro è un gap sistematico (non rumore), MMag/lead_lag pubblicati rientrano nella banda stocastica. In tabella si cita il classical pubblicato; i re-run vanno in footnote di trasparenza.

## Costi

Run completo (incl. falso avvio fp16 su L4, crash e ripresa): ~$25-35 su Modal, ~$30 coperti dai crediti Starter.

## Deviazioni dalla ricetta pubblicata (lista completa)

bf16 AMP (loro classe A100); selezione best per val LOSS (niente generazioni in val); 1 GPU; sottoesposizione celle 16ch epoche 1-4 (OOM, documentata sopra); epoca 4 al 61% + top-up di pareggio step (22.440 esatti, coda schedule inclusa); i 1.988 step di top-up pescano da una passata rimescolata extra.
