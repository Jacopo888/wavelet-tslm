#!/bin/bash
# Ambiente v3 = ambiente Livello 1 (~/node_env.sh del kit vultr/, INVARIATO)
# + overlay v3 (2 file modello, train.py con la registrazione, 3 config).
# Prerequisiti (li mette ops/ship_v3.sh): ~/patch/ = kaggle/patch/* +
# v3-recurrent/patch/* + v3-recurrent/configs/*; ~/node_env.sh; ~/node_*_v3.sh.
# ROCm (droplet MI300X): V3_ROCM=1 -> stesso node_env.sh con i pin torch rocm6.3
# al posto di cu124 (node_env.sh e' gia' parametrizzato: TORCH_INDEX_URL/PIN).
set -ex

if [ "${V3_ROCM:-0}" = "1" ]; then
  export V3_EXPECT_BACKEND=rocm
  export TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.3
  export TORCH_PIN=2.6.0+rocm6.3
  export TORCHVISION_PIN=0.21.0+rocm6.3
  command -v rocm-smi && rocm-smi --showproductname | head -8 || true
  apt-get install -y -q python3-venv python3-dev tmux git > /dev/null
else
  export V3_EXPECT_BACKEND=cuda
fi

# Env L1 (venv pinnato, TimeRLM al commit, patch v1, dati HF pinnati con
# preflight anti-leakage, RUN_MANIFEST.json). Idempotente: salta se gia' fatto.
if [ ! -f ~/RUN_MANIFEST.json ] || [ ! -d ~/venv ] || [ ! -d ~/timerlm ]; then
  bash ~/node_env.sh
fi

# Also validate an already-existing base environment: V3_ROCM=1 must never
# silently reuse CUDA wheels (or vice versa).
~/venv/bin/python - <<'EOF'
import os
import torch
expected = os.environ["V3_EXPECT_BACKEND"]
assert torch.cuda.is_available(), "PyTorch non vede la GPU"
if expected == "rocm":
    assert torch.version.hip, f"atteso ROCm, torch={torch.__version__}"
    assert "+rocm6.3" in torch.__version__, torch.__version__
else:
    assert torch.version.cuda and not torch.version.hip, (
        f"atteso CUDA, torch={torch.__version__}"
    )
print("V3_BACKEND_OK", expected, torch.__version__,
      torch.cuda.get_device_name(0))
EOF

cp ~/patch/wavelet_encoder_v3.py ~/timerlm/sft/tslm/src/models/wavelet_encoder_v3.py
cp ~/patch/wavelet_v3.py         ~/timerlm/sft/tslm/src/models/wavelet_v3.py
cp ~/patch/train.py              ~/timerlm/sft/tslm/train.py   # + registrazione wavelet_v3 (v1 identica)
cp ~/patch/wavelet_precise_v3*.yaml ~/timerlm/sft/tslm/configs/

# 1) suite CPU dell'encoder — stesso file che gira nel repo
cd ~/patch && ~/venv/bin/python test_encoder_v3.py

# 2) registrazione in train.py: v1 e v3 entrambe presenti
cd ~/timerlm/sft/tslm && ~/venv/bin/python - <<'EOF'
import sys
sys.argv = ["train.py"]
import train
keys = sorted(train.MODEL_CLASSES)
assert "wavelet" in keys and "wavelet_v3" in keys, keys
print("MODEL_CLASSES", keys)
EOF

# 3) GPU positiva + smoke dell'encoder v3 in bf16 sul caso peggiore
#    (16 canali x 131072 campioni, forward+backward): tempo e picco memoria.
TSLM_GRAD_CKPT=1 ~/venv/bin/python - <<'EOF'
import gc, sys, time, torch
sys.path.insert(0, "/root/timerlm/sft/tslm")
from src.models.wavelet_encoder_v3 import RecurrentWaveletTSEncoder
assert torch.cuda.is_available(), "GPU assente"
print("device:", torch.cuda.get_device_name(0), "torch", torch.__version__)
for tied in (True, False):
    enc = RecurrentWaveletTSEncoder(tied=tied).cuda().to(torch.bfloat16).train()
    x = torch.randn(16, 131072, device="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
    y = enc(x); y.float().pow(2).mean().backward(); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    assert y.shape == (16, 256, 512) and torch.isfinite(y.float()).all()
    assert peak < 8 * 2**30, f"encoder peak inatteso: {peak / 2**30:.2f} GiB"
    print(f"V3_GPU_SMOKE tied={tied} steps={enc.last_steps} t={time.time()-t:.2f}s "
          f"peak={peak/2**20:.0f}MiB gates={[round(g,2) for g in enc.last_gates]}")
    del y, x, enc
    gc.collect()
    torch.cuda.empty_cache()
EOF

# 4) manifest v3 = manifest L1 + sha256 dei file dell'overlay (parita' con va1
#    verificabile: i file v1 devono avere lo stesso sha del repo).
~/venv/bin/python - <<'EOF'
import hashlib, json, os, pathlib
root = pathlib.Path(os.path.expanduser("~/timerlm/sft/tslm"))
man = json.load(open(os.path.expanduser("~/RUN_MANIFEST.json")))
files = ["src/models/wavelet_encoder.py", "src/models/wavelet.py",
         "src/models/wavelet_encoder_v3.py", "src/models/wavelet_v3.py", "train.py"]
man["v3_overlay_sha256"] = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files}
man["v3_rocm"] = os.environ.get("V3_ROCM", "0") == "1"
json.dump(man, open(os.path.expanduser("~/RUN_MANIFEST_V3.json"), "w"), indent=1)
for f, h in man["v3_overlay_sha256"].items():
    print(f"  {h[:16]}  {f}")
print("RUN_MANIFEST_V3 ok")
EOF

mkdir -p ~/logs ~/evalq
echo "V3_ENV_OK"
