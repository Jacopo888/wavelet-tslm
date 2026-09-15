# Inventario dei checkpoint fuori dal nodo (15/09/2026)

Confronto per SHA-256 fra i 118 file `.pt` presenti sul nodo `wavelet-l1` e le copie
locali, fatto prima della distruzione dell'istanza. Le copie stanno in
`~/l3_offnode/` (129 MB) e `~/va1_offnode/` (1,9 GB) dentro WSL, ciascuna con
`SHA256SUMS.txt`.

## Tenuti: ogni checkpoint di un braccio documentato

| file | sha256 (16) | cosa è | dove è documentato |
|---|---|---|---|
| `va1_offnode/.../epoch_5_end.pt` | `cd7238ae81da1afc` | L1, SFT singolo passaggio, test 0,2744 | `vultr/RISULTATI-VA1.md` |
| `bracci_documentati/va1_rl_step100_composed.pt` | `2a927aa0812b4198` | punto 5, va1 + GRPO single-turn, test 0,2869 | `l3/RISULTATI-L3.md` |
| `bracci_documentati/vz1_epoch_1_end.pt` | `7957195172d90b76` | Z1, primo tentativo di zoom, gate non superato | `l3/RISULTATI-L4-ZOOM.md` |
| `z1bis_20260912/vz1bis_epoch_1_end.pt` | `924110a5a344dc80` | Z1-bis, test **0,3343** | `l3/RISULTATI-Z1BIS.md` |
| `w1_20260913/vw1_epoch_1_end.pt` | `8221cd4aba0a9101` | W1, coordinate del selettore | `l3/RISULTATI-SELETTORI.md` §8 |
| `w2_20260913/vw2_epoch_1_end.pt` | `9f48ca99490b1dd8` | **W2, arm di punta, test 0,3521** | §10, §12 |
| `w3_20260914/vw3_epoch_1_end.pt` | `77a3de78ed33ddf1` | W3, test **0,3739** | §12 |
| `grpo_20260914/vw2_grpo_step50.pt` | `cebefffb2ca40114` | pilot GRPO, step selezionato | §11 |
| `bracci_documentati/pilot_cross_epoch_1_end.pt` | `f699dc2c9178319e` | pilot 09/09 cross | `v3-recurrent/README.md` |
| `bracci_documentati/pilot_crosslocal_epoch_1_end.pt` | `411bb9ba073e4aea` | pilot 09/09 cross-local | `v3-recurrent/README.md` |

Più i 47 checkpoint dell'addestramento `va1` (copia del 05/09 verificata identica).

## Lasciati sul nodo, deliberatamente: 60 checkpoint distinti

- **31** step intermedi del long run del punto 5 (step 0, 1, 25…300 in `lora` e
  `composed`): conserviamo lo step 100, l'unico selezionato e portato al test;
- **17** step intermedi del pilot GRPO (0, 1, 25, 75, 100, 125, 150): tutti misurati
  come indistinguibili dallo step 0, §11;
- **8** checkpoint intermedi dei pilot 09/09 (`best_model`, `batch_500/2000/4488`):
  conserviamo l'`epoch_1_end` di ciascuno;
- **4** checkpoint dello smoke RL di 20 passi.

Nessuno di questi è riferimento di un risultato riportato. Ricrearli richiederebbe
ripetere il training, ma nessuna conclusione documentata dipende da loro.

## Altri dati estratti (nel repo, non in WSL)

`integrity/artifacts/l3_reading_rows/` — le righe dei 208 crop fissi per ogni braccio
e ogni offset (13 file), unico modo di rifare confronti di lettura a finestre
identiche senza ricalcolare. `integrity/artifacts/l3_riferimenti/` — `va1` su test
(800 righe, primary 0,274354) e val (520, 0,310475), più le righe che definiscono i
208 ID, con hash d'ordine verificato uguale al dichiarato `34ccb284…` e categorie
80/80/48. `integrity/artifacts/l3_trace_plans/` — i quattro piani delle tracce
(z1bis, w1, w2, w3), 4.488 righe ciascuno. Più gli artefatti di ogni run in
`integrity/artifacts/l3_*`.

## Cosa resta solo sul nodo

Il checkout `~/timerlm` con il venv e il parquet del train (2,05 GB, rigenerabile dal
codice rilasciato con seed 42) e i log completi dei worker, di cui sono archiviati i
`worker.log` delle run recenti. Lo setup si ricrea con gli script in `vultr/`.
