# Brief per Codex — fix del worker `v3-recurrent/node/node_eval_worker_v3.sh` (falso negativo sulla coverage)

Repo: `C:\Users\jacop\Proposta ETH FINALE\wavelet-tslm`. Non committare, niente rete. Modifica
direttamente i file; io spedisco e rilancio. URGENTE: la GPU è ferma.

## Cosa è successo (06/09 15:57 UTC, nodo A40)
Training `v3tied` concluso: `~/logs/v3tied.done` = `EXIT:0`, 5 checkpoint `epoch_1..5_end.pt`,
e nel log `~/logs/v3tied.log` ci sono le 5 righe
`[Coverage] epoch N: trained 4488, failed 0, expected 4488`. Il worker però ha scritto:

```
training v3tied concluso con EXIT:0, parte la selezione
WORKER_FAILED: coverage epoca 1: trovate 0 righe esatte, attesa 1
```

Causa: il worker usa `grep -Ec "^[[:space:]]*\[Coverage\] epoch ${ep}: trained 4488, failed 0, expected 4488$"`
sul log grezzo. Nel log la riga di coverage NON inizia a inizio riga: train.py la stampa dopo la
barra tqdm, sulla stessa riga separata da `\r` (carriage return), es. (dallo smoke):

```
Epoch 1 [Train]: 100%|██████████| 4/4 [00:22<00:00,  4.02s/it, loss=1.9173, lr=0.00e+00]                                    [Coverage] epoch 1: trained 4, failed 0, expected 4
```

Quindi `^` non matcha (e possono esserci `\r`/spazi finali). Il test di verifica dello smoke
(`node_smoke_v3.sh`) faceva già `train_text.replace("\r", "\n")` prima di cercare la riga: il
worker no. Anche `vultr/node_eval_worker_single.sh` (va1) non faceva questo check: è nuovo.

## Cosa fare
1. Nel worker, normalizza il log prima dei grep: `tr '\r' '\n' < ~/logs/${RUN}.log | grep -Ec '^[[:space:]]*\[Coverage\] epoch N: trained 4488, failed 0, expected 4488[[:space:]]*$'`
   (sia il check per epoca sia il conteggio totale). Mantieni il check esatto (1 riga per epoca, 5 totali).
2. Rendi il worker RIESEGUIBILE dopo un `WORKER_FAILED`: all'avvio rimuovi un eventuale
   `~/evalq/${RUN}_WORKER_FAILED` stantio (oggi lo rimuove solo il launcher, che però azzera anche
   la run: NON va usato). Le val json già presenti vanno riusate come già fa.
3. Controlla che nessun altro grep del worker o di `node_train_v3.sh` legga righe che possono
   trovarsi dopo `\r` (es. `Training Complete`, `[SkipFinalTest]`); se sì, stessa normalizzazione.
4. Aggiorna `v3-recurrent/review/CODEX-REPORT.md` (o `CODEX-SMOKE-NOTES.md`) con una riga sul fix.

Nessun altro cambiamento. Comando che eseguirò sul nodo dopo il tuo fix (il training NON va rifatto):

```bash
tmux new-session -d -s evw_v3tied "env RUN=v3tied V3_GPU=0 TSLM_QWEN_REVISION=<rev> bash ~/node_eval_worker_v3.sh > ~/logs/evw_v3tied_2.log 2>&1"
```
