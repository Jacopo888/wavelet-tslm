#!/bin/bash
# Fase 1 sul nodo 8xA100 (da root, appena su): base di sistema + driver + FM.
# Dopo il "NODO PRONTO" apt e' CONGELATO (patto col peer).
set -ex

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y build-essential python3-venv python3-pip git tmux htop \
    nvtop rsync curl

# Driver NVIDIA server + fabric manager (nodo SXM: senza FM CUDA non parte).
apt-get install -y nvidia-driver-570-server nvidia-utils-570-server \
    nvidia-fabricmanager-570 || {
  echo "570 non disponibile, provo ubuntu-drivers"; \
  apt-get install -y ubuntu-drivers-common; \
  ubuntu-drivers install --gpgpu; \
  V=$(ls /usr/lib/modules/$(uname -r)/kernel/nvidia-* -d 2>/dev/null | grep -oE '[0-9]+' | head -1); \
  apt-get install -y nvidia-fabricmanager-$V; }

systemctl enable nvidia-fabricmanager
# Il driver appena installato di solito richiede reboot per il modulo giusto.
if ! nvidia-smi >/dev/null 2>&1; then
  echo "REBOOT_NEEDED"
else
  systemctl start nvidia-fabricmanager
  echo "SETUP_OK_NO_REBOOT"
fi
