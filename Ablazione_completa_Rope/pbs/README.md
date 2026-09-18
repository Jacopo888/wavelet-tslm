# Polimi P0 backend gate

Remote root: `/work/u10970284/wavelet-tslm-rope-ablation`. The `code/`,
`logs/`, `runs/`, `cache/`, and `tmp/` subdirectories keep every artifact for
this project inside that root.

From WSL, preview the source-only synchronization with:

```bash
rsync -azn --delete-delay --itemize-changes \
  --exclude '.git/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
  -e 'ssh.exe -o BatchMode=yes' \
  '/mnt/c/Users/jacop/Proposta ETH FINALE/wavelet-tslm/Ablazione_completa_Rope/' \
  'polimi:/work/u10970284/wavelet-tslm-rope-ablation/code/'
```

Remove `-n` only after reviewing the itemized diff. This sync target is only
`code/`; it cannot delete remote logs, run state, caches, checkpoints, or
results.

On the login node, plan a P0 submission without calling `qsub`:

```bash
cd /work/u10970284/wavelet-tslm-rope-ablation/code
python3 tools/submit_run.py \
  --run-id p0-efficient-attention-v1 \
  --gate-kind kernel \
  --command '<explicit P0 backend-gate command>' \
  --scratch-root '<verified compute-node scratch path or pbs-tmpdir>' \
  --scratch-lifetime-hours '<verified lifetime>' \
  --environment-mode readonly \
  --environment-source '<read-only CUDA/PyTorch environment>'
```

The command is deliberately mandatory because the Qwen temporal-RoPE backend
gate does not yet have an agreed entrypoint. Once that entrypoint exists, add
`--submit`. The helper records an immutable request, derives a stable PBS job
name, detects an active duplicate with `qselect`, and refuses an ambiguous
retry unless `--retry-failed` is explicit.

The PBS request is fixed to queue `gpu`, one GPU, four CPUs, 16 GiB host RAM,
and 20 minutes. On the assigned node it rejects `/home`, `/work`, memory-backed
storage, and the node root filesystem as scratch; probes write access, mount,
user/group quotas, free bytes, inode count, and declared lifetime. Required
free space is `max(25 GiB, 2*S_snapshot + S_env + S_data + S_output + 5 GiB)`.
All Hugging Face, Torch, XDG, and temporary caches then point to the verified
job scratch. Hugging Face and Transformers are forced offline.

`kernel` does not accept model weights. `full-smoke` additionally requires a
complete, readable, shared, read-only snapshot whose directory name is the
exact pinned revision; its byte inventory comes from the files rather than the
model name. The eventual loader must pass `local_files_only=True` explicitly.
The gate itself must assert A40 execution and compare the Q*/K* 352/384,
V256 `EFFICIENT_ATTENTION` path against the reference before training is
allowed.
