"""Read-only snapshot of the archived context-0.1 run; writes audit JSON only."""
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch

torch.set_num_threads(2)
archive = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
run = archive / 'run'
manifest = json.loads((run / 'RUN_MANIFEST.json').read_text())
root = Path('/root/timerlm/sft/tslm')
provenance = {}
for filename, expected in manifest['v3_overlay_sha256'].items():
    p = root / filename
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    provenance[filename] = dict(expected=expected, actual=actual, matches=actual == expected)

def stats(x):
    x = x.detach().float()
    return dict(mean=x.mean().item(), rms=x.square().mean().sqrt().item(),
                min=x.min().item(), max=x.max().item())

rows = []
paths = list((run / 'checkpoints').glob('checkpoint_epoch_*.pt'))
paths += [run / 'checkpoints' / 'epoch_1_end.pt']
for path in paths:
    c = torch.load(path, map_location='cpu', weights_only=False)
    s = c['model_state']
    a = s['ts_encoder.context_scale'].item()
    opt = c['optimizer_state']
    sched = c['scheduler_state']
    scalar_states = []
    groups = []
    for gi, g in enumerate(opt['param_groups']):
        groups.append({k: v for k, v in g.items() if k != 'params'} | dict(count=len(g['params'])))
        for pid in g['params']:
            st = opt['state'].get(pid, {})
            if 'exp_avg' in st and st['exp_avg'].ndim == 0:
                t = int(st['step'])
                b1, b2 = g['betas']
                mh = st['exp_avg'].item() / (1 - b1 ** t)
                vh = st['exp_avg_sq'].item() / (1 - b2 ** t)
                direction = mh / (math.sqrt(vh) + g['eps'])
                scalar_states.append(dict(group_index=gi, parameter_id=pid, step=t,
                    exp_avg=st['exp_avg'].item(), exp_avg_sq=st['exp_avg_sq'].item(),
                    bias_corrected_direction=direction,
                    direction_times_next_lr=-g['lr'] * direction,
                    decay_at_next_lr=-g['lr'] * g['weight_decay'] * a))
    gamma = s['ts_encoder.context_norm.weight']
    beta = s['ts_encoder.context_norm.bias']
    row = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        epoch=c['epoch'], batch_idx=c['batch_idx'], timestamp=c['timestamp'],
        optimizer_step=scalar_states[0]['step'], scheduler=sched,
        scale=a, gamma=stats(gamma), beta=stats(beta),
        scale_gamma=stats(a * gamma), scale_beta=stats(a * beta),
        gate_weight=stats(s['ts_encoder.gate_head.weight']),
        gate_bias=stats(s['ts_encoder.gate_head.bias']),
        optimizer_groups=groups, scalar_states=scalar_states,
        trainable_tensors=len(s), val_loss_field=c['val_loss'])
    rows.append(row)
    print(json.dumps({k: row[k] for k in ('file', 'optimizer_step', 'scale', 'gamma', 'beta', 'scalar_states')}), flush=True)
rows.sort(key=lambda r: r['optimizer_step'])
log = (archive / 'logs' / 'v3tied.log').read_text(errors='replace').replace('\r', '\n')
coverage = [line.strip() for line in log.splitlines() if '[Coverage]' in line]
setup = [line.strip() for line in log.splitlines() if any(x in line for x in
    ('Training mode:', 'Trainable:', 'Warmup steps:', 'Total steps:', '[AMP]', 'context_scale=', 'Epoch 1/5:', 'Epoch 2/5:'))]
gate_lines = [line.strip() for line in log.splitlines() if '[V3Gates]' in line]
report = dict(archive=str(archive), archive_note=(archive/'README.txt').read_text(),
    torch_actual=torch.__version__, manifest=manifest, code_provenance=provenance,
    coverage=coverage, setup=setup, first_gates=gate_lines[:3], last_gates=gate_lines[-3:],
    checkpoint_rows=rows,
    existing_val_probe=json.loads((archive/'evalq'/'v3tied_ep1_valprobe80.json').read_text()))
(out / 'snapshot.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ('archive_note', 'code_provenance', 'coverage', 'setup', 'first_gates', 'last_gates', 'existing_val_probe')}, indent=2))
