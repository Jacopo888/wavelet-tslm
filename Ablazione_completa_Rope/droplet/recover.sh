#!/usr/bin/env bash
# Recupero dopo reboot del droplet: rimonta lo scratch, riavvia il container, riprende il training P1 dal checkpoint.
# Uso (da locale): ssh do-mi300x 'bash /root/wavelet-rope/code/droplet/recover.sh p1-main'
set -Eeuo pipefail
RUN="${1:-p1-main}"
mountpoint -q /mnt/scratch || mount /dev/disk/by-label/DOSCRATCH /mnt/scratch
docker start wrope >/dev/null 2>&1 || true
docker exec wrope python -c "import torch, transformers, fla; assert torch.cuda.is_available(); print('container OK', torch.__version__, transformers.__version__)"
if [ -f /root/wavelet-rope/runs/$RUN/DONE ]; then echo "run $RUN already DONE"; exit 0; fi
# clear a stale lease left by the killed process (the reboot ended it)
rm -rf /root/wavelet-rope/runs/$RUN/state/ACTIVE.lock
cd /root/wavelet-rope/code && bash droplet/run_p1.sh $RUN --resume auto --val-every 400 --val-subset 104 --val-full-at-pass-end 1 --max-new-tokens 400
