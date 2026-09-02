#!/bin/bash
# Fase 2 (post-eventuale-reboot): verifica GPU/FM prima del "NODO PRONTO".
set -e
systemctl start nvidia-fabricmanager 2>/dev/null || true
sleep 3
echo "=== driver/GPU ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
N=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$N" = "8" ] || { echo "ERRORE: viste $N GPU, attese 8"; exit 1; }
echo "=== fabric manager ==="
systemctl is-active nvidia-fabricmanager
echo "=== topologia ==="
nvidia-smi topo -m | head -12
echo "=== alloc CUDA su GPU 0 e 7 (via venv, se gia' creato, altrimenti skip) ==="
if [ -x ~/venv/bin/python ]; then
  CUDA_VISIBLE_DEVICES=0 ~/venv/bin/python -c "import torch; x=torch.zeros(8, device='cuda'); print('gpu0 ok', (x+1).sum().item())"
  CUDA_VISIBLE_DEVICES=7 ~/venv/bin/python -c "import torch; x=torch.zeros(8, device='cuda'); print('gpu7 ok', (x+1).sum().item())"
fi
echo "VERIFY_OK"
