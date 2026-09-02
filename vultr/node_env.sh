#!/bin/bash
# Fase 3: ambiente python + repo + dati (richiede ~/patch/ gia' scp'ato).
set -ex

python3 -m venv ~/venv
~/venv/bin/pip install -q --upgrade pip
~/venv/bin/pip install -q torch numpy pandas pyarrow scipy psutil
~/venv/bin/pip install -q "transformers @ git+https://github.com/huggingface/transformers@5c1c72b" \
    peft==0.18.1 einops einops-exts datasets pyyaml wandb open_clip_torch huggingface-hub
~/venv/bin/pip install -q --no-deps chronos-forecasting==2.2.2 open-flamingo

git clone --depth 1 https://github.com/OpenTSLM/TimeRLM ~/timerlm
# overlay
cp ~/patch/wavelet_encoder.py ~/timerlm/sft/tslm/src/models/wavelet_encoder.py
cp ~/patch/wavelet.py         ~/timerlm/sft/tslm/src/models/wavelet.py
cp ~/patch/qwen_backbone.py   ~/timerlm/sft/tslm/src/models/qwen_backbone.py
cp ~/patch/qa_dataset.py      ~/timerlm/sft/tslm/src/datasets/anomaly_xl/qa_dataset.py
cp ~/patch/train.py           ~/timerlm/sft/tslm/train.py
cp ~/patch/*.yaml             ~/timerlm/sft/tslm/configs/

~/venv/bin/pip install -q -e ~/timerlm/anomalyXL

cd ~/timerlm/anomalyXL
~/venv/bin/python scripts/generate_anomalyxl_sweep.py --out-root datasets --precise-only --seed 42
cd ~/timerlm/sft/tslm
~/venv/bin/python fetch_data.py

mkdir -p ~/logs ~/evalq
echo "ENV_OK"
