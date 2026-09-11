# Brief per Codex — Livello 3, punto 4: azione di zoom nella policy multimodale va1

Stato: 11/09/2026 (Claude), ordine di Jacopo "ferma le repliche e scrivi brief punto 4".
Codex scrive il codice, Claude controlla il diff, esegue test e gate sul nodo. Prima del
codice, Codex compila la sezione **Decisioni di disegno** (§7) e la lascia nel PR.

## 1. Cosa vogliamo e cosa abbiamo già

Domanda: **una rilettura a risoluzione piena di una finestra (zoom) migliora la localizzazione
fine e multicanale della policy multimodale, oltre a quanto fa l'RL da solo?**

Abbiamo già (test 800, una lettura ciascuno, `l3/RISULTATI-L3.md`):

| braccio | primary | localize | all_ch | lead_lag | nota |
|---|---:|---:|---:|---:|---|
| va1 (SFT, `epoch_5_end.pt`) | 0,2744 | 0,241 | 0,261 | 0,166 | riferimento L1 |
| va1 + RL senza zoom (step 100, `2a927aa0…`) | 0,2869 | 0,233 | 0,247 | 0,269 | **controllo**: il +0,10 di lead_lag è collasso su "independent" (76/100), direzione 29 vs 30 |

Il confronto del punto 4 è quindi a quattro bracci: va1 · va1+RL · **va1+zoom (solo SFT di
riscaldamento)** · **va1+zoom+RL**. Se lo zoom serve, deve vedersi su localize e all_ch contro
ENTRAMBI i controlli, non solo contro va1.

## 2. Il meccanismo (nessuna modifica all'architettura)

Il modello va1 accetta già più blocchi di serie: `prepare_batch` costruisce per ogni
`TextTimeSeriesPrompt` un'intestazione testuale + 256 `<ts_pad>`, l'encoder produce 256 token
per blocco qualunque sia la lunghezza (`WaveletTSEncoder.forward: (B, L) → (B, 256, d)`,
posizioni relative al blocco, normalizzazione MAD per blocco) e `_merge_features` sostituisce i
placeholder in ordine. **Lo zoom è un blocco in più**: un `TextTimeSeriesPrompt` con
intestazione del tipo `Zoom on channel <name>, samples [s, e) of L:` e la finestra grezza
`x[s:e]` del canale, codificata dallo STESSO encoder+proiettore va1 (pesi congelati). Nessuna
modifica a `kaggle/patch/`; tutto vive in `l3/zoom.py` usando l'interfaccia pubblica
(`prepare_batch`, `generate`, `_encode_and_project`, `_merge_features`).

Episodio a due turni:
1. **turno 1**: prompt va1 (serie + domanda, come oggi) + una frase nel prompt che descrive
   l'azione disponibile → il modello emette O la risposta finale (JSON, come oggi) O l'azione
   `zoom:{"channel": "<name>", "start": s, "end": e}` (JSON stretto, coordinate assolute);
2. **turno 2** (solo se azione valida): il prompt viene ricostruito con il blocco zoom in più
   (dopo i canali originali, prima della domanda) e con l'azione riportata come turno assistant;
   il modello emette la risposta finale in coordinate assolute.

Regole: al massimo **uno** zoom per episodio; finestra clampata a `[0, L)`, lunghezza minima
1024 campioni (sotto, l'encoder non ha livelli utili) e massima L; azione non parsabile o fuori
range → nessuno zoom, si passa direttamente alla risposta (contato in `zoom_invalid`).
Costo: +256 token percettivi + ~30 di testo + una seconda generazione.

Per l'RL i log-prob si calcolano su **due segmenti** (azione | risposta), entrambi sotto la
policy, con la maschera sui soli token emessi dal modello (mai sul blocco zoom e sulle
intestazioni inserite dall'ambiente); reward = metrica ufficiale della risposta finale.

## 3. Reward e pool (correzioni dal punto 5)

- **lead_lag ESCLUSA dal pool RL** (viene solo valutata): con la reward ufficiale la policy
  collassa su "independent" (scorciatoia dello scorer). Nessuna reward inventata: le altre 4
  categorie usano la metrica ufficiale; non parsabile → 0.
- **Pool ampio e ricampionato**: 200 prompt di train stratificati per categoria (50 × 4) e
  lunghezza, **ricampionati ogni 100 passi** con seed dichiarato (mai gli stessi 40 per 300
  passi: nel punto 5 la policy li ha memorizzati, reward su di essi 0,30 → 0,43 con val520 in
  calo dopo lo step 100). Filtro sull'informatività: opzionale, motivato al §7.
- Metriche per passo come nel kit long-run, più: `zoom_rate` (episodi con azione valida),
  `zoom_invalid`, `zoom_window_len` (media, per categoria), `zoom_iou_with_gold` (quanto la
  finestra scelta copre l'anomalia gold: diagnostica, MAI nella reward).

## 4. Stadi e gate (script `l3/node_l3_zoom.sh`, fail-closed, tmux con log su file, marker per fase)

| stadio | contenuto | gate | budget A40 |
|---|---|---|---|
| **Z0 — meccanismo, solo eval** | va1 con zoom **forzato** (nessuna scelta della policy): (a) *oracle* = finestra gold ±25 % di margine (tetto del meccanismo), (b) *euristica* = primo candidato della piramide (`l2/overlay/timerlm/environments/timeseries_qa/wavelet_pyramid.py`, scan di energia, numpy), (c) senza zoom = va1. Eval greedy su probe80 e **val520** per tutti e tre | riportare; se (a) non supera va1 su localize/all_ch il modello non legge il blocco fuori distribuzione → Z1 è comunque necessario, flag nel report | ~1,5 h |
| **Z1 — SFT di riscaldamento** (`vz1`) | LoRA da va1 (encoder/proiettore congelati), 1 epoca sul train (4488) con blocco zoom e traiettoria supervisionata: turno 1 = azione (finestra gold ± jitter casuale 10-50 %, nel 30 % dei casi finestra euristica), turno 2 = risposta gold; loss sui token di azione e risposta; stessa ricetta va1 (bf16, ckpt attivo, lr 1e-4, batch 1); coverage per categoria obbligatoria | val520 con zoom scelto dalla policy: primary ≥ va1 (0,3105) e localize+all_ch non peggiori; `zoom_rate` e `zoom_invalid` riportati | ~6 h + 0,5 h |
| **Z2 — GRPO da vz1 con zoom scelto dalla policy** | G8, 2 prompt/passo, 200 passi, pool §3, reference KL = vz1, val520 a 50/100/150/200, selezione max val520, stop dopo 2 val senza +0,005, checkpoint ogni 25 con resume (riusare `l3/grpo.py`) | come long-run; hard stop invariati | ~14 h + 2 h |
| **Z3 — test, una lettura** | `vz1` e `vz1+RL` (candidato) sul test 800; confronto appaiato 10k contro va1 e contro va1+RL (JSONL già archiviati); lead_lag riportata anche come accuratezza di direzione | — | ~1,2 h |

Totale ≈ 25 h A40 (~$22). Nessuna fase parte senza il marker OK della precedente; nessun
`--split test` fuori da Z3.

## 5. Vincoli

Come nel punto 5: `kaggle/patch/` intatto; A40-24Q, venv L1, batch 1, grad ckpt, bf16, sdpa;
niente `expandable_segments`; contatori prima delle conclusioni; cartelle mai riusate; test CPU
(`py -m pytest l3/tests -q`) per: costruzione del blocco zoom (posizioni assolute, clamp, min
len), parser dell'azione (valido / non valido / fuori range), maschera dei due segmenti,
finestra oracle/euristica, campionamento del pool con ricampionamento, esclusione di lead_lag,
identità a init (vz1 senza azione == comportamento va1 sui prompt senza zoom? NO: vz1 è un
nuovo SFT; l'identità esatta vale per Z2 contro vz1). Riportare la lezione ops: il worker va
lanciato in tmux con redirezione su file (il self-detach perde l'output).

## 6. Deliverable

`l3/zoom.py` (blocco zoom, parser azione, oracle/euristica, episodio a due turni per eval e RL),
`l3/zoom_sft.py` (costruzione delle traiettorie supervisionate e training vz1 riusando il più
possibile `train.py`; se serve un loop proprio, stessa ricetta e stessi contatori di coverage),
estensione di `l3/grpo.py` (episodio a due segmenti, pool ricampionato, categorie escluse),
`l3/configs/l3_zoom_{z0,z1,z2}.yaml`, `l3/node_l3_zoom.sh`, test, `PROTOCOL-L3.md` §zoom
(regole dichiarate prima della run).

## 7. Decisioni di disegno (Codex)

- Testo esatto dell'istruzione sull'azione nel prompt e dell'intestazione del blocco zoom
  (breve, coordinate assolute; deve restare identico fra SFT, RL ed eval).
- Finestra: min/max, clamp, jitter in Z1, quota euristica in Z1, come si sceglie il canale nei
  casi multicanale (per all_ch: un canale per episodio, quale?).
- Euristica: adattamento di `wavelet_pyramid.py` (scan energia) alla scelta di UNA finestra.
- Log-prob su due segmenti: un forward per segmento o un forward sull'intera traiettoria
  ricostruita? Motivare (memoria, esattezza old == corrente).
- Pool: dimensione, ricampionamento, filtro sì/no.
- Stime di tempo per fase dai tempi misurati (Z1: ~2 s/step SFT su va1 → 4488 esempi + eval).
- Cosa NON è coperto: più zoom per episodio, encoder/proiettore allenabili, lead_lag nel RL.
