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
cp ~/patch/eval_integrity.py  ~/timerlm/sft/tslm/eval_integrity.py
cp ~/patch/fetch_data.py      ~/timerlm/sft/tslm/fetch_data.py
cp ~/patch/preflight_data.py  ~/timerlm/sft/tslm/preflight_data.py
cp ~/patch/*.yaml             ~/timerlm/sft/tslm/configs/

~/venv/bin/pip install -q -e ~/timerlm/anomalyXL

# Dati: SOLO dal repo HF alla revisione pinnata (train+val+gold test).
# Il fetch_data.py del patch scrive MANIFEST.json e ABORTISCE se gli split
# non sono disgiunti o i conteggi non tornano (lezione leakage 02/09:
# mai piu' generatore locale + split del test come dati di training).
cd ~/timerlm/sft/tslm
~/venv/bin/python fetch_data.py --variant precise
~/venv/bin/python preflight_data.py ~/timerlm/anomalyXL/datasets --variant precise

mkdir -p ~/logs ~/evalq
echo "ENV_OK"
