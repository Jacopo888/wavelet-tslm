#!/bin/bash
# Spedisce sul nodo tutto cio' che serve alle run v3 (da WSL, chiave wavelet):
#   ~/patch/  = kaggle/patch/* (v1, train.py registrato) + v3-recurrent/patch/*
#               + v3-recurrent/configs/*.yaml
#   ~/        = vultr/node_env.sh, v3-recurrent/node/*.sh
# Uso (WSL): bash ship_v3.sh <ip>            -> poi sul nodo:
#   bash ~/node_prep_v3.sh      (pacchetti base + GPU positiva; driver gia' nell'immagine)
#   bash ~/node_env_v3.sh       (V3_ROCM=1 sul droplet MI300X)  -> V3_ENV_OK
#   bash ~/node_train_v3.sh v3tied   (poi v3untied, v3sdepth: in sequenza su A40,
#                                    in parallelo con V3_GPU=0 su MI300X)
set -euo pipefail
IP="${1:?uso: ship_v3.sh <ip>}"
KEY=~/.ssh/vultr_wavelet
REPO="/mnt/c/Users/jacop/Proposta ETH FINALE/wavelet-tslm"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -i $KEY root@$IP"
SCP="scp -o StrictHostKeyChecking=no -o BatchMode=yes -i $KEY"

$SSH "mkdir -p ~/patch ~/logs ~/evalq"
$SCP "$REPO"/kaggle/patch/*.py "$REPO"/kaggle/patch/*.yaml root@$IP:~/patch/
$SCP "$REPO"/v3-recurrent/patch/*.py "$REPO"/v3-recurrent/configs/*.yaml root@$IP:~/patch/
$SCP "$REPO"/vultr/node_env.sh "$REPO"/v3-recurrent/node/*.sh root@$IP:~/
# CRLF -> LF (i file possono arrivare da Windows) e permessi
$SSH "sed -i 's/\r\$//' ~/*.sh ~/patch/*.py ~/patch/*.yaml && chmod +x ~/*.sh && ls ~/patch | wc -l && md5sum ~/patch/wavelet_encoder.py ~/patch/wavelet_encoder_v3.py ~/patch/train.py"
echo "--- md5 locali (devono coincidere)"
md5sum "$REPO"/kaggle/patch/wavelet_encoder.py "$REPO"/v3-recurrent/patch/wavelet_encoder_v3.py "$REPO"/kaggle/patch/train.py
echo "SHIP_OK $IP"
