#!/usr/bin/env bash
# Bootstrap idempotente del droplet DigitalOcean MI300X per WR-Agent-4B-Wildcard-v1 (corsia ROCm).
# Eseguire come root sul droplet DOPO aver sincronizzato il codice in /root/wavelet-rope/code:
#   bash /root/wavelet-rope/code/droplet/bootstrap.sh
# Immagine DO attesa: "PyTorch 2.6.0 / ROCm 6.4" (torch vive nel container Docker `rocm:latest`, non sull'host).
set -Eeuo pipefail
log() { printf '[bootstrap %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# 1. scratch NVMe (5 TB, etichetta DOSCRATCH) su /mnt/scratch, persistente in fstab
if ! mountpoint -q /mnt/scratch; then
  mkdir -p /mnt/scratch
  mount /dev/disk/by-label/DOSCRATCH /mnt/scratch
fi
UUID=$(blkid -s UUID -o value /dev/disk/by-label/DOSCRATCH)
grep -q "$UUID" /etc/fstab || echo "UUID=$UUID /mnt/scratch ext4 defaults,nofail,noatime 0 2" >> /etc/fstab
mkdir -p /root/wavelet-rope/{code,runs,logs} /mnt/scratch/{hf,cache,tmp,pip,torch,triton,datasets}
log "scratch: $(df -h /mnt/scratch | tail -1)"

# 2. container di lavoro dalla stessa immagine del Jupyter fornito da DO (torch 2.6.0+git684f6f2, ROCm 6.4.1)
docker image inspect rocm:latest >/dev/null
if ! docker inspect wrope >/dev/null 2>&1; then
  docker run -d --name wrope --restart no \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --ipc=host --shm-size=32g --network host --security-opt seccomp=unconfined \
    -v /root/wavelet-rope:/workspace -v /mnt/scratch:/scratch \
    -e HF_HOME=/scratch/hf -e HF_HUB_CACHE=/scratch/hf/hub -e XDG_CACHE_HOME=/scratch/cache \
    -e TMPDIR=/scratch/tmp -e PIP_CACHE_DIR=/scratch/pip -e TORCH_HOME=/scratch/torch -e TRITON_CACHE_DIR=/scratch/triton \
    -e HF_HUB_ENABLE_HF_TRANSFER=0 -e TOKENIZERS_PARALLELISM=false \
    -w /workspace rocm:latest sleep infinity >/dev/null
fi
docker start wrope >/dev/null 2>&1 || true
docker exec wrope python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'hip', torch.version.hip, torch.cuda.get_device_name(0))"

# 3. stack L1 (stessi pin di vultr/node_env.sh e integrity/artifacts/va1/RUN_MANIFEST.json), torch/torchvision/numpy vincolati a quelli dell'immagine
docker exec wrope bash -lc 'python - > /workspace/logs/pip_constraints.txt <<EOF
import torch, torchvision, numpy
print(f"torch=={torch.__version__}"); print(f"torchvision=={torchvision.__version__}"); print(f"numpy=={numpy.__version__}")
EOF
pip install -q -c /workspace/logs/pip_constraints.txt \
  "transformers @ git+https://github.com/huggingface/transformers@5c1c72b" peft==0.18.1 \
  einops einops-exts datasets pyyaml huggingface-hub safetensors pandas pyarrow scipy psutil pytest
pip install -q --no-deps flash-linear-attention==0.5.2 fla-core==0.5.2
python -c "import transformers, peft, torch; assert torch.cuda.is_available(); print(\"stack\", transformers.__version__, peft.__version__, torch.__version__)"
python -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule; print(\"fla-core 0.5.2 kernels importable\")"'

# 4. asset verificati per hash (mai per nome): AnomalyXL al pin df363a46, Qwen3.5-4B al pin 851bf6e8
docker exec -w /workspace/code wrope python tools/fetch_anomalyxl.py | tail -1
docker exec -w /workspace/code wrope python tools/fetch_qwen_snapshot.py | tail -1

# 5. manifest dell'ambiente
docker exec wrope python - > /workspace/logs/env_manifest.json <<'EOF'
import json, subprocess, sys, torch, transformers, peft, triton
props = torch.cuda.get_device_properties(0)
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
print(json.dumps({"torch": torch.__version__, "hip": torch.version.hip, "device": torch.cuda.get_device_name(0),
                  "gcn_arch": getattr(props, "gcnArchName", None), "vram_gib": round(props.total_memory / 2**30, 1),
                  "python": sys.version.split()[0], "transformers": transformers.__version__, "peft": peft.__version__,
                  "triton": getattr(triton, "__version__", "3.2.0 (source install)"), "pip_freeze": freeze.splitlines()}, indent=2))
EOF
log "BOOTSTRAP_OK"
