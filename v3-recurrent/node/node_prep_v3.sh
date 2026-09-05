#!/bin/bash
# Preparazione minima di un nodo NUOVO per le run v3 (da root, appena su).
# Vultr slice vGPU (os 2284 Ubuntu 24.04): driver GRID GIA' installato -> solo
# pacchetti base + verifica positiva della GPU. Droplet MI300X (immagine
# rocmpytorchjupyt): ROCm gia' presente -> stessa cosa con rocm-smi.
# NON e' vultr/node_setup.sh (quello installa driver+fabricmanager per il
# bare metal 8xA100 e qui romperebbe il driver vGPU).
set -ex
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q build-essential python3-venv python3-pip python3-dev git tmux htop rsync curl
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  apt-get install -y -q nvtop || true
elif command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showproductname --showmeminfo vram | head -20
else
  echo "NESSUNA GPU visibile (ne' nvidia-smi ne' rocm-smi)"; exit 1
fi
df -h / | tail -1
free -g | head -2
echo "PREP_OK"
