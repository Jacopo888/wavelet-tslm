# Research Integrity Pass — report (02/09/2026)

Ordine di Jacopo: integrity pass completo, nessun training, nessuna modifica
all'architettura (fp32-positions/NMS/DB4 rimandate a una variante `v2`
separata a protocollo identico). Baseline congelata in commit `ad2fb78`,
fix in commit `4601c62`.

## 1. Congelamento (fatto, commit ad2fb78)

- Repo git creato nella root `wavelet-tslm/`; timerlm tracciato per SHA
  (`31fcd847`) + lista dei 18 overlay in [FREEZE.md](FREEZE.md).
- Hash sha256 + row count dei 12 file dati locali: `DATA-HASHES-LOCAL.json`.
- Artefatti auditabili scaricati dal volume Modal in `artifacts/`
  (output per-esempio dei test m30-ep4, m30-endrun, full1 + config + selezione).
- Inventario checkpoint del volume documentato (m30: 5 epoch-end + best;
  full1: latest + intra-epoch).

## 2. Quarantena e preflight (fatto)

- 8 file contaminati spostati (NON cancellati) in
  `timerlm/anomalyXL/datasets/_QUARANTINE_LEAKED/` con README: train/val
  locali erano al 100% sottoinsiemi del test — precise (723/77/800) E
  coarse (231/17/248). Origine: REPRODUCE.md Option B upstream.
- `kaggle/patch/preflight_data.py`: gate che rifiuta qualunque root dati
  con split non disgiunti, conteggi errati o manifest fuori pin. Agganciato
  a node_env.sh, node_train.sh e modal_train.py (ogni percorso, cache inclusa).
  **Indurito il 02/09 sera su rilievo della review esterna**: in modalità
  training (default) FALLISCE anche su train/val/manifest ASSENTI e
  riverifica righe+sha256 di ogni entry del manifest contro i file su disco
  (una cache accettata sulla sola stringa di revisione può essere corrotta);
  `--allow-test-only` per root di sola valutazione. In train.py aggiunta la
  cintura "righe attese": a fine epoca ok+failed deve eguagliare le righe
  del dataset (un dataloader che droppa in silenzio non passa più).

## 3. Fix di codice (fatto, commit 4601c62)

1. **finally del CkptFallback** ora ripristina lo stato GLOBALE del gradient
   checkpointing (prima: disable incondizionato → un solo OOM degradava
   tutti i batch successivi dei run GRAD_CKPT=1).
2. **Fail-fast di copertura**: con `TSLM_REQUIRE_FULL_COVERAGE=1` il run
   muore al PRIMO campione perso (loss invalida, fallback fallito, errore
   generico), non a fine epoca. Obbligatorio in tutti i launcher di ricerca
   (Modal e Vultr).
3. **Evaluator**: JSONL per-esempio (id, categoria, gold, prediction raw,
   parse_status, tutte le metriche); gli errori di loss/generazione non sono
   più inghiottiti ma registrati come righe a punteggio 0; con
   `TSLM_EVAL_STRICT=1` (default in --eval-only) un solo esempio mancante
   abortisce l'eval. `TSLM_EVAL_TAG` evita di sovrascrivere log storici.
4. **Dati deterministici**: `fetch_data.py` del patch pinna la revisione HF
   `df363a46` (dataset immutato dal 30/07/2026), scrive MANIFEST.json
   (sha256+righe), asserisce disgiunzione e conteggi attesi (4488/520/800).
   modal_train.py si fida della cache del volume solo se il manifest
   combacia col pin.
5. **Scorer ufficiale INTATTO** (comparabilità). La parte strict vive in
   `diagnostics.py`, separata.

## 4. Provenienza — catena chiusa (PROVENANCE-VERIFY.json)

Digest canonici di contenuto (id / gold answers / fingerprint serie), HF
pinnato vs cache volume usate dai run vs test locale generato:

- **Train e val di m30/full1 ≡ HF pinnato** su tutti e tre i digest, e
  **disgiunti dal test ufficiale (overlap 0)** → i run principali sono
  certificati senza leakage. Il benchmark HF stesso è pulito (0/0/0).
- **Il gold degli eval storici ha id e risposte identiche all'ufficiale, ma
  serie di input DIVERSE** (rigenerazione seed-42 = altra realizzazione del
  rumore; le risposte dipendono dalla spec d'iniezione, non dal rumore).
  → Gli eval storici sono su una *variante* degli input del test. La
  ri-valutazione sui byte ufficiali è in corso (sezione 7).

## 5. Diagnostica sui per-esempio esistenti — LA QUESTIONE LOCALIZE È CHIUSA

`DIAGNOSTICS-m30ep4-vs-full1.json` (validator strict + decomposizioni +
bootstrap CI 95%, seed fisso). Sintesi:

**m30 ep4, localize (200 righe, null sempre-assente = 0.195):**
- Confusion presence: TP 131, FP 3, FN 30, TN 36 → recall 0.814,
  **precision 0.978**, accuratezza negativi 0.923, accuratezza totale 0.835
  (sopra ENTRAMBI i null: sempre-no 0.195, sempre-sì 0.805).
- **IoU sui 161 positivi gold: 0.0017** (matched: 0.0021; acc@IoU0.3: 0.000).

→ Il modello **rileva** l'anomalia (segnale percettivo reale, non banale:
non è né sempre-sì né sempre-no, e sbaglia pochissimo sui negativi) ma
**non sa piazzare la finestra** (IoU ≈ 0). Il primary 0.181 è quasi tutto
credito di presenza sui negativi (36/200 = 0.180). La review aveva ragione
sul claim "localizza"; la decomposizione mostra però che il segnale di
detection è genuino. Nota meccanicistica: è ESATTAMENTE il fenotipo atteso
dal bug bf16-posizioni (l'energia dell'anomalia è visibile nei token, la
posizione fine è rumore) → ipotesi falsificabile per la variante `v2`.

**full1 (input identici, paired su 800):**
- localize: stesso fenotipo (recall 0.839, precision 0.944, IoU pos 0.0033).
- **all_channels 0.180 SGONFIATO**: 18/23 di accuratezza sui gold vuoti +
  **F1=0.000 sui 77 gold non vuoti**. Il "vantaggio curriculum" del run
  sporco era solo calibrazione a rispondere "nessuna anomalia" — NON
  localizzazione multicanale. La narrativa va corretta anche per full1.
- lead_lag: 0.264 (m30) e 0.340 (full1), entrambi ≤ null 0.350.
- **measure_magnitude regge**: 0.510 CI [0.463, 0.556] (m30) e 0.522
  CI [0.471, 0.572] (full1) — il risultato forte, stabile nei due run.
- Paired (m30−full1): full1 meglio overall (−0.027, CI [−0.044, −0.010]);
  m30 meglio su localize (+0.024 SIG), full1 su all_channels (+0.150 SIG)
  e lead_lag (−0.076 SIG).
- Validator strict: m30 → 55 predizioni senza JSON + 5 `anomalies` non-lista;
  full1 → ZERO violazioni. Nessun caso `present:"false"` stringa nei due run.

## 6. Test (17/17 PASS, `tests/run_tests.py`)

Coprono ciascun fix, inclusi due test sul CASO PATOLOGICO REALE (lezione
m30: i meccanismi di sicurezza si validano nella condizione vera): il leak
da 77 righe rilevato sui file di quarantena effettivi; la root dati reale
ora passa pulita. I percorsi torch-bound di train.py sono compilati
(py_compile) e la loro logica è estratta in eval_integrity.py testato.

## 7. Ri-valutazione sul test UFFICIALE pinnato — FATTA (numeri certificati)

Job Modal `reeval` (A100-40GB, evaluator nuovo, 800/800, eval_errors=0,
JSONL per-esempio in artifacts/):

| Categoria | m30 ep4 (hfpin) | m30 ep4 (storico) | full1 best ep5 (hfpin) | full1 best ep5 (storico) | full1 post-top-up (hfpin) |
|---|---:|---:|---:|---:|---:|
| primary | **0.2156** | 0.2106 | **0.2356** | 0.2376 | **0.2417** |
| localize | 0.1814 | 0.1814 (id.) | 0.1621 | 0.1576 | 0.1755 |
| classify | 0.0037 | 0.0037 (id.) | 0.0106 | 0.0105 | 0.0089 |
| measure_magnitude | 0.5102 | 0.5102 (id.) | 0.5229 | 0.5220 | 0.5326 |
| all_channels | 0.0300 | 0.0300 (id.) | 0.1900 | 0.1800 | 0.2000 |
| lead_lag | 0.3043 | 0.2643 | 0.3035 | 0.3403 | 0.2993 |

Decomposizione full1-best sugli input ufficiali: localize conf
TP135/FP7/FN26/TN32 (prec 0.951, rec 0.839), IoU positivi 0.0026;
all_channels 0.190 = empty-acc 0.826 con F1 0.000 sui non vuoti; MMag
0.5229 CI [0.472, 0.572]; zero violazioni strict. Stesso fenotipo.
Le piccole differenze full1-best storico↔hfpin fuori da lead_lag derivano
dal percorso (eval in-run 29/08 vs eval-only da checkpoint ricaricato);
d'ora in poi fanno fede SOLO gli eval-only hfpin.

Letture:
- **Scoperta di provenienza, ora PROVATA sui contenuti** (recepito il rilievo
  della review esterna: metriche identiche ≠ byte identici): fingerprint
  per-categoria delle serie (id, n_canali, somma float64 a 6 cifre) su
  generato-locale vs HF pinnato → **4/5 categorie con digest IDENTICI,
  differisce SOLO lead_lag_with_magnitude (100 righe)** — vedi
  `PROOF-PERCATEGORY-DIGESTS.txt`. Coerente coi punteggi m30 identici fuori
  da lead_lag. (Fingerprint di somma, non hash bit-a-bit dei float: la
  probabilità di collisione su 700 righe è trascurabile ma va detto.)
  lead_lag resta comunque ≈ null (0.350) in ogni versione.
- **Nota terminologica** (review): "eval_errors=0" = zero errori di
  loss/generazione della pipeline; m30 ha comunque 60 predizioni
  parse_failed, correttamente conteggiate come punteggio 0 nel primary.
- Il confronto best-ep5 (0.2356) vs post-top-up (0.2417) è ora A PARITÀ DI
  INPUT (entrambi hfpin) — legittimo dopo il terzo eval, come richiesto.
- **La decomposizione regge sugli input ufficiali** (DIAGNOSTICS-hfpin.json):
  m30 localize confusion identica (TP131/FP3/FN30/TN36, precision 0.978),
  IoU positivi 0.0017; full1 confusion TP131/FP4/FN30/TN35, IoU positivi
  0.0006; all_channels di full1 = 0.200, tutto da empty-accuracy 0.870 con
  F1 0.000 sui non vuoti. Conclusioni input-robuste.
- Parse status (JSONL): m30 740/800 parsed, 60 parse_failed; full1 800/800.
- NB: il primo re-eval full1 ha usato `latest_checkpoint.pt` (post-top-up),
  non il canonico `best_model.pt` (ep5) del run storico — re-eval del
  best_model lanciato per completezza; il post-top-up risulta comunque
  MIGLIORE (0.2417), candidato a diventare il full1 di riferimento.

## 8. Implicazioni per la proposta

- Claim difendibile: tokenizzazione multirisoluzione a budget controllato
  nella classe Chronos+4B → **measure_magnitude 5.5× il pubblicato** e
  **detection di presenza precisa a 131k** (0.978 precision), con costo
  memoria/lunghezza controllato. La localizzazione fine è il problema
  APERTO che motiva v2 (posizioni fp32) e il Livello 2 (zoom agentico).
- Da smettere di citare: "primo TSLM che localizza" (falso: IoU≈0);
  "curriculum accidentale di full1 su all_channels" (era calibrazione dei
  vuoti); Toto va incluso nei confronti (con nota sulla classe 32B vs 4B).
- Bug upstream per il report agli autori: ora 7 (4 già noti + Option B
  leak, bool("false"), eventi malformati scartati).
