# Brief per Codex — deadlock dello zero-init del contesto: scegli e implementa il fix, poi ripetiamo l'ablazione 1

Repo: `C:\Users\jacop\Proposta ETH FINALE\wavelet-tslm`. Non committare, niente rete. Modifica
direttamente i file; io eseguo `py v3-recurrent/patch/test_encoder_v3.py`, smoke sul nodo, run.
Analisi nel report (Jacopo la legge).

## Fatto misurato (run `v3tied` residuo-su-v1, commit `1b0f9fa`, 5 epoche complete, A40)
- `ts_encoder.context_scale` nei checkpoint: ep1 0,0001, ep3 −0,0006, ep5 0,0004; `gate_head.bias`
  fermo a ~1,98. Il contesto non è mai entrato nei token: il modello finale è va1 (test 800:
  0,2832 vs va1 0,2744, differenze = rumore run-to-run; dettagli in `v3-recurrent/RISULTATI-V3TIED.md`).
- Meccanismo (verificalo sul codice): uscita = `base + context_scale·LN(z)`; con scala 0 i blocchi,
  il gate e `context_norm` ricevono gradiente ∝ scala (≈0) e non si allenano; la scala riceve
  gradiente = ⟨∂L/∂out, LN(z)⟩ con z prodotto da un ramo NON allenato → ≈0 in media, oscilla
  intorno a zero. Deadlock stile ReZero senza co-adattamento; lr 1e-4, 22.440 step.

## Cosa fare
1. **Rompere il deadlock in modo misurabile**, scegliendo e motivando fra:
   a. `context_scale_init` piccolo non nullo (es. 0,1) nei tre config di run + config di smoke;
   b. in aggiunta, un moltiplicatore di learning rate per i parametri del ramo di contesto
      (blocks, gate_head, step_embed, context_norm, context_scale) SE il trainer lo permette via
      config senza toccare `train.py` (controlla `training.learning_rate` in `kaggle/patch/train.py`
      / `src/utils/config.py` / `chatts.py` `get_param_groups`: esistono gruppi `encoder`/`projector`?
      se serve una modifica a `train.py`, deve essere env-gated e inerte di default, documentata);
   c. altro (es. scala parametrizzata come `softplus`/`exp` con init −2, o warmup lineare della
      scala nei primi N step) se lo ritieni più robusto. Motiva la scelta con il gradiente atteso
      sui blocchi al primo step.
2. **Test**: (a) `test_init_matches_v1_*` resta valido con `context_scale_init=0` esplicito;
   (b) nuovo test: con il config di run (init scelto), dopo K step sintetici su un obiettivo che
   richiede il contesto (es. la posizione dell'anomalia letta da un livello che il top-k di v1 NON
   include), la norma del gradiente sui blocchi è > 1e-3 relativa a `value_proj` E la scala si
   muove di più di 1e-2 in valore assoluto; con init 0 lo stesso test deve mostrare il deadlock
   (scala ferma, gradiente sui blocchi ≈0) — serve come documentazione del bug;
   (c) decodificabilità a init con l'init scelto ≤ 1,1× v1.
3. **Smoke** (`node_smoke_v3.sh`): mantieni `V3_INIT_EQUALS_V1` (eseguito con scala 0 forzata)
   e pretendi che dopo i 4 step la scala si sia mossa di ≥ 1e-3 dal valore iniziale e che il
   gradiente sui blocchi sia stato non nullo (log o JSON).
4. Config: `configs/wavelet_precise_v3{tied,untied,sdepth,smoke_tied,smoke_untied}.yaml`
   aggiornati; sezione `training:` identica a va1 salvo, se scegli (b), il gruppo lr.
5. `README.md` della cartella (architettura + storia: la run `1b0f9fa` è una replica di va1) e
   `CODEX-REPORT.md` sezione 12: analisi, scelta, numeri dei test.

Vincoli: v1 intatta; contratto/ablazioni invariati; niente dipendenze nuove; i checkpoint della run
`1b0f9fa` devono restare caricabili (stesse chiavi) se cambi solo l'init, oppure rifiutati con
errore chiaro se cambi la parametrizzazione.
