# Corsia DigitalOcean MI300X (ROCm)

Nodo: 1× AMD Instinct MI300X (192 GB), 1,99 $/h, credito AMD 200 $ con scadenza 4/10/2026.
**Fatturato anche da spento**: distruggere il droplet quando non si lavora; il bootstrap ricostruisce tutto in circa 5 minuti.

- SSH dal PC: alias `do-mi300x` in `~/.ssh/config` di Windows (chiave `vultr_airway`).
- Sync del codice (da WSL): `rsync -az --exclude .git/ --exclude __pycache__/ --exclude .pytest_cache/ -e 'ssh -i ~/.ssh/vultr_wavelet' '<cartella-locale>/wavelet-tslm/Ablazione_completa_Rope/' root@<ip>:/root/wavelet-rope/code/`
- Sync verso il nodo: escludere sempre `.venv-luna/` (ambiente Python locale per gli agenti di codifica, ~2 GB) oltre a `.git/`, `__pycache__/`, `.pytest_cache/`, `integrity/artifacts/`.
- Bootstrap: `bash /root/wavelet-rope/code/droplet/bootstrap.sh` (idempotente: scratch, container `wrope`, pin L1, asset verificati, manifest ambiente).
- Comandi nel container: `docker exec -w /workspace/code wrope python ...`; `/workspace` = `/root/wavelet-rope`, `/scratch` = `/mnt/scratch`.
- Gate del backend: `docker exec -w /workspace/code wrope python tools/backend_gate.py --backend FLEX --pad-qk-to 512 --report /workspace/runs/p0-backend-gate/<label>.json` (esito del 14/09 in `RISULTATI-GATE-BACKEND-20260914.md`).
- Training: `[CODE_DIR=/workspace/code_p2] [CONFIG=configs/p1c_mi300x.toml] bash droplet/run_p1.sh <run-id> --val-every 400` (tmux `p1-<run-id>`, log `/root/wavelet-rope/logs/train_<run-id>.log`); sentinella `bash droplet/gate_early.sh <run-id> 400`; catena finale `bash droplet/post_p1.sh <run-id>`; dopo un reboot `bash droplet/recover.sh <run-id>`.
- Il container `rocm` originale dell'immagine DO è un Jupyter esposto sulla porta 8888 pubblica: non serve al progetto.
- Checkpoint e report vanno copiati fuori (`scp do-mi300x:/root/wavelet-rope/runs/... integrity/artifacts/...`) prima di distruggere il droplet. Lo scratch è volatile.
