#!/bin/bash
# Fase 3: ambiente python + repo + dati (richiede ~/patch/ gia' scp'ato).
set -ex

# Pin dell'upstream: il checkpoint slim ha senso solo insieme a QUESTO
# commit + overlay patch (review 02/09: resume cross-node non ambiguo).
TIMERLM_COMMIT=31fcd847b7cb37a1f1e6859b1dca7973ed0eae74

python3 -m venv ~/venv
~/venv/bin/pip install -q --upgrade pip

# TRAPPOLA vGPU Vultr (02/09): il driver GRID 550.x espone CUDA 12.4, ma il
# wheel torch di default e' cu13x -> torch.cuda.is_available()=False, tutto
# su CPU e crash dtype al primo batch. E il pip successivo (open_clip_torch
# -> torchvision senza pin -> torch 2.14/CUDA13) puo' SOVRASCRIVERE il torch
# giusto (review 02/09, bloccante #2). Quindi: constraints file passato a
# OGNI pip install + assert positivo finale.
: "${TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu124}"
: "${TORCH_PIN:=2.6.0+cu124}"
: "${TORCHVISION_PIN:=0.21.0+cu124}"
cat > ~/pip-constraints.txt <<EOF
torch==${TORCH_PIN}
torchvision==${TORCHVISION_PIN}
EOF
~/venv/bin/pip install -q -c ~/pip-constraints.txt \
    torch torchvision --index-url "$TORCH_INDEX_URL"
~/venv/bin/pip install -q -c ~/pip-constraints.txt numpy pandas pyarrow scipy psutil
~/venv/bin/pip install -q -c ~/pip-constraints.txt \
    "transformers @ git+https://github.com/huggingface/transformers@5c1c72b" \
    peft==0.18.1 einops einops-exts datasets pyyaml wandb open_clip_torch huggingface-hub
~/venv/bin/pip install -q --no-deps chronos-forecasting==2.2.2 open-flamingo

# Verifica POSITIVA post-install (non "nessun errore" ma "la GPU c'e'"):
~/venv/bin/python - <<EOF
import torch, torchvision
assert torch.__version__ == "${TORCH_PIN}", f"torch {torch.__version__} != ${TORCH_PIN}"
assert torchvision.__version__.startswith("${TORCHVISION_PIN%%+*}"), \
    f"torchvision {torchvision.__version__}"
assert torch.cuda.is_available(), "CUDA NON disponibile dopo l'install"
x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
assert float((x @ x).abs().sum()) > 0
print("POST-INSTALL OK:", torch.__version__, torchvision.__version__,
      torch.cuda.get_device_name(0))
EOF

git clone https://github.com/OpenTSLM/TimeRLM ~/timerlm
git -C ~/timerlm checkout -q "$TIMERLM_COMMIT"
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

~/venv/bin/pip install -q -c ~/pip-constraints.txt -e ~/timerlm/anomalyXL

# Dati: SOLO dal repo HF alla revisione pinnata (train+val+gold test).
# Il fetch_data.py del patch scrive MANIFEST.json e ABORTISCE se gli split
# non sono disgiunti o i conteggi non tornano (lezione leakage 02/09).
cd ~/timerlm/sft/tslm
~/venv/bin/python fetch_data.py --variant precise
~/venv/bin/python preflight_data.py ~/timerlm/anomalyXL/datasets --variant precise

# RUN_MANIFEST: tutto cio' che serve per rendere i checkpoint slim
# ricostruibili altrove (review 02/09, serio #6).
~/venv/bin/python - <<EOF
import json, subprocess, torch, torchvision
from huggingface_hub import HfApi
man = {
    "timerlm_commit": subprocess.run(
        ["git", "-C", "$HOME/timerlm", "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "qwen_model_id": "Qwen/Qwen3.5-4B",
    "qwen_revision": HfApi().model_info("Qwen/Qwen3.5-4B").sha,
    "data_manifest": json.load(open(
        "$HOME/timerlm/anomalyXL/datasets/anomalyxl-train/MANIFEST.json")),
}
json.dump(man, open("$HOME/RUN_MANIFEST.json", "w"), indent=1)
print("RUN_MANIFEST:", man["timerlm_commit"][:12], man["qwen_revision"][:12])
EOF

mkdir -p ~/logs ~/evalq
echo "ENV_OK"
