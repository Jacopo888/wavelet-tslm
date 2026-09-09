# Brief per Codex — tre verifiche aggiuntive per i pilot del contesto (harness + post-hoc)

Repo: `C:\Users\jacop\Proposta ETH FINALE\wavelet-tslm`. Non committare, niente rete. Modifica
direttamente i file; io eseguo test, ship e run. Report in `CODEX-REPORT.md` §15.
NON toccare il comportamento di training; solo valutazione/diagnostica. Il pilot
`pilot_crosslocal_20260909T071641Z` è IN CORSO sul nodo (fine ≈ 12:40 UTC): spedirò dopo.

## Richieste di Jacopo (testuali)
1. **Val intera acceso/spento sullo stesso checkpoint, appaiata con intervalli di confidenza.**
   Oggi lo harness produce sui 520 solo "acceso". Servono `epoch{N}_full_val_on` E
   `epoch{N}_full_val_off` (stesso checkpoint, stesse righe) e un JSON di confronto appaiato
   (bootstrap 10k seed fisso, primary aggregato per categoria come nella metrica ufficiale, IC95
   per primary e per categoria; riusa/estendi `integrity/diagnostics.py` se comodo — legge JSONL
   per esempio — o `audit/.../summarize.py`).
2. **Struttura dopo LayerNorm, per banda esatta, sui dati reali.** Il centraggio locale avviene
   prima della `context_norm`: dopo la LN la media per banda non è necessariamente nulla.
   Misurare sugli 80 esempi reali (e a fine epoca sulla val intera, se economico) le statistiche
   di h = context_norm(z_centrato) raggruppate per `base_band` ESATTA (0, 1, …, L; padding
   escluso): frazione variabile, coseno medio, RMS della media di gruppo / RMS di h, dipendenza
   dall'input; nel JSON del teacher forcing e nel `[V3Ctx]`. Ricorda nel report che diversità
   alta non equivale a informazione utile.
3. **Contesto corretto contro contesto disallineato**, oltre ad acceso/spento: (a) contesto
   permutato fra token validi (roll/permutazione dentro lo stesso esempio, norme preservate);
   (b) contesto di un'altra serie compatibile (stessa lunghezza/canali) — se possibile senza
   rifare il forward completo, altrimenti (a) basta. Sia in teacher forcing (NLL) sia in
   generazione sugli 80 (primary per categoria). Serve un hook env-gated di sola valutazione,
   es. `TSLM_V3_CONTEXT_EVAL=off|roll_tokens|shuffle_series` (inerte di default; `off` deve
   coincidere con l'attuale `TSLM_V3_CONTEXT_OFF=1`), applicato nell'encoder v3 al momento di
   sommare il contesto ai token base. Test: roll preserva le norme per token e non cambia la
   base; con contesto nullo (`context_scale`=0) tutte le condizioni coincidono.

## Consegna
- `node_pilot_v3.sh`: a fine epoca esegue on/off/roll in generazione (80) e TF (80), val intera
  on E off (+ roll se il tempo lo consente: ~25 min l'una) e scrive
  `pilot_eval/epoch{N}_paired.json` (on−off, on−roll, off−roll con IC95) — il confronto appaiato
  è ora l'esito principale del pilot; `PILOT_OK` solo se tutto c'è.
- `node_pilot_posthoc.sh <run_dir> <epoch>`: esegue le stesse valutazioni su un pilot GIÀ concluso
  (checkpoint `epoch_{N}_end.pt` + `pilot_train.yaml`/`pilot_probe80.yaml` nella run dir; init
  comune non serve), scrivendo negli stessi file `pilot_eval/…` senza sovrascrivere quelli
  esistenti (aggiunge i mancanti: `full_val_off`, `roll`, `paired`, stats per banda). Lo userò
  sui pilot `pilot_cross_20260909T014929Z` e `pilot_crosslocal_20260909T071641Z`.
- Test CPU (`py`) verdi, `bash -n`, `git diff --check`; README/report aggiornati.
