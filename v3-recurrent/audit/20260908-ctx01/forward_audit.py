"""Paired validation interventions on archived v3 weights, with no optimizer steps.

Only val.parquet is read. Cached encoder outputs are checked against native
forward. Model files and checkpoints are never changed; JSONL contains results.
"""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--archive')
ap.add_argument('--out')
ap.add_argument('--checkpoints', nargs='+')
ap.add_argument('--pilot-config')
ap.add_argument('--checkpoint')
ap.add_argument('--out-json')
ap.add_argument('--n', type=int, default=80)
ap.add_argument('--precision', choices=['masters', 'eval'], default='masters')
ap.add_argument('--generate', action='store_true')
ap.add_argument('--modes', nargs='+', default=['on', 'off', 'scale03', 'scale10', 'roll'])
args = ap.parse_args()

if args.pilot_config:
    if not args.checkpoint or not args.out_json:
        ap.error('--pilot-config requires --checkpoint and --out-json')
    os.environ.setdefault('TSLM_SAVE_TRAINABLE_ONLY', '1')
    root = Path.cwd()
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    import train

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    config = train.ExperimentConfig.from_yaml(args.pilot_config)
    if config.runtime.max_samples != 800:
        raise ValueError('pilot audit config must set runtime.max_samples: 800')
    model = train.load_model(config, 'cuda')
    train.configure_training_mode(model, config)
    if os.environ.get('TSLM_AMP') == '1':
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.dtype in (torch.bfloat16, torch.float16):
                parameter.data = parameter.data.float()
    eos = model.get_eos_token()
    _, val_loader, _, _, _, _ = train.create_dataloaders(config, eos_token=eos)
    if len(val_loader.dataset) != args.n:
        raise ValueError(f'pilot audit expected {args.n} val rows, got {len(val_loader.dataset)}')
    train.load_checkpoint(args.checkpoint, model)
    model.eval()
    rows = []
    started = time.time()
    with torch.no_grad():
        for index, batch in enumerate(val_loader):
            prepared = model.prepare_batch(batch)
            os.environ.pop('TSLM_V3_CONTEXT_OFF', None)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                on_loss = float(model(**prepared)['loss'])
            stats = json.loads(json.dumps(model.ts_encoder.last_context_stats))
            os.environ['TSLM_V3_CONTEXT_OFF'] = '1'
            with torch.autocast('cuda', dtype=torch.bfloat16):
                off_loss = float(model(**prepared)['loss'])
            os.environ.pop('TSLM_V3_CONTEXT_OFF', None)
            sample = batch[0]
            rows.append({
                'index': index,
                'id': sample.get('id'),
                'category': sample.get('category'),
                'teacher_forcing_on': on_loss,
                'teacher_forcing_off': off_loss,
                'delta_on_minus_off': on_loss - off_loss,
                'context_stats': stats,
            })
            if (index + 1) % 10 == 0:
                print('PILOT_FORWARD_PROGRESS', index + 1, '/', args.n, flush=True)
    final_keys = ('z_rms', 'h_rms', 'token_variable_fraction', 'mean_token_cosine',
                  'gate_mean', 'gate_saturation_low', 'gate_saturation_high')
    aggregate = {}
    for key in final_keys:
        values = [row['context_stats'].get('final', {}).get(key) for row in rows]
        values = [value for value in values if isinstance(value, (int, float)) and np.isfinite(value)]
        aggregate[key] = float(np.mean(values)) if values else None
    result = {
        'mode': 'pilot_val_teacher_forcing',
        'config': str(args.pilot_config),
        'checkpoint': str(args.checkpoint),
        'n': len(rows),
        'split': 'val',
        'teacher_forcing_on_mean': float(np.mean([row['teacher_forcing_on'] for row in rows])),
        'teacher_forcing_off_mean': float(np.mean([row['teacher_forcing_off'] for row in rows])),
        'delta_on_minus_off_mean': float(np.mean([row['delta_on_minus_off'] for row in rows])),
        'context_final_mean': aggregate,
        'rows': rows,
        'elapsed_s': time.time() - started,
    }
    destination = Path(args.out_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('PILOT_FORWARD_AUDIT', json.dumps({
        key: value for key, value in result.items() if key != 'rows'
    }, separators=(',', ':')), flush=True)
    raise SystemExit(0)

if not args.archive or not args.out or not args.checkpoints:
    ap.error('archive mode requires --archive, --out, and --checkpoints')
archive, out = Path(args.archive), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
manifest = json.loads((archive/'run/RUN_MANIFEST.json').read_text())
os.environ.update(TSLM_TORCH_DTYPE='bfloat16', TSLM_AMP='1', TSLM_AMP_DTYPE='bfloat16',
    TSLM_ATTN_IMPL='sdpa', TSLM_GRAD_CKPT='1', TSLM_QWEN_REVISION=manifest['qwen_revision'],
    HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
    TSLM_V3_GATE_LOG_EVERY='0', CUDA_VISIBLE_DEVICES='0')
root = Path('/root/timerlm/sft/tslm')
os.chdir(root)
sys.path.insert(0, str(root))
import numpy as np
import torch
from datasets import Dataset
import train
from src.models.wavelet_encoder import WaveletTSEncoder

torch.set_num_threads(4)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
config = train.ExperimentConfig.from_yaml(archive/'run/config.yaml')
model = train.load_model(config, 'cuda')
groups = train.configure_training_mode(model, config)
trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
if args.precision == 'masters':
    for p in model.parameters():
        if p.requires_grad and p.dtype in (torch.bfloat16, torch.float16):
            p.data = p.data.float()
model.eval()
enc = model.ts_encoder
native_forward = enc.forward
captured = {}
def capture_norm(module, inputs, output):
    captured['z'] = inputs[0].detach()
    captured['h'] = output.detach()
hook = enc.context_norm.register_forward_hook(capture_norm)

# Reuse the exact archived 80-example probe IDs, or audit the entire val set.
valpath = Path('/root/timerlm')/config.dataset.extra_kwargs['train_dir']/'val.parquet'
raw = Dataset.from_parquet(str(valpath))
formatter = train.AnomalyXLDataset.__new__(train.AnomalyXLDataset)
formatter.variant = 'precise'
formatter.EOS_TOKEN = model.get_eos_token()
probe_path = archive/'valprobe/output_logs/evalonly_val_epoch_1_end_valprobe_epoch_0.jsonl'
probe = [json.loads(line) for line in probe_path.read_text().splitlines()]
ids = list(raw['id'])
index = {value: i for i, value in enumerate(ids)}
selected = ([index[row['id']] for row in probe][:args.n]
            if args.n <= len(probe) else list(range(len(raw))))
assert len(selected) == args.n, (len(selected), args.n)
selection = [dict(index=i, id=ids[i], category=raw[i]['category'],
                  length=raw[i]['length'], n_channels=raw[i]['n_channels']) for i in selected]
(out/'selection.json').write_text(json.dumps(selection, indent=2))
print('AUDIT_SELECTION', json.dumps(dict(n=len(selected), categories=dict(Counter(x['category'] for x in selection)),
    precision=args.precision, modes=args.modes, generation=args.generate)), flush=True)

def rms(x):
    return x.float().square().mean().sqrt().item()

def quantiles(x):
    x = x.detach().float().flatten()
    return dict(zip(['p10','p50','p90','p99'], torch.quantile(x, x.new_tensor([.1,.5,.9,.99])).cpu().tolist()))

def features_for(base, h, z, a):
    return base + torch.as_tensor(a, dtype=enc.context_scale.dtype, device=base.device).to(z.dtype) * h

def read_features(x):
    enc.forward = native_forward
    captured.clear()
    y = enc(x)
    h, z = captured['h'], captured['z']
    valid = enc.last_base_valid.clone()
    gates = list(enc.last_gates)
    base = WaveletTSEncoder._encode_channel(enc, x)
    rebuilt = features_for(base, h, z, enc.context_scale.detach())
    assert torch.equal(y, rebuilt), 'Cached native feature reconstruction mismatch'
    return base, h, z, y, valid, gates

def intervention(mode, base, h, z, y, valid):
    if mode == 'on': return y
    if mode == 'off': return base
    if mode == 'scale03': return features_for(base, h, z, .03)
    if mode == 'scale10': return features_for(base, h, z, .1)
    if mode == 'roll':
        # Permutation preserves the multiset of context vectors within channel.
        rolled = h.clone()
        for c in range(h.shape[0]):
            v = torch.where(valid[c])[0]
            rolled[c, v] = h[c, v].roll(max(1,len(v)//3), dims=0)
        return features_for(base, rolled, z, enc.context_scale.detach())
    raise ValueError(mode)

started = time.time()
for checkpoint in args.checkpoints:
    if checkpoint != 'init':
        ck = torch.load(archive/'run/checkpoints'/checkpoint, map_location='cpu', weights_only=False)
        state = ck['model_state']
        assert set(state) == trainable_names, (set(state)-trainable_names, trainable_names-set(state))
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected
        assert not (set(missing) & trainable_names)
        del state, ck
    elif checkpoint != args.checkpoints[0]:
        raise ValueError('init must be first')
    model.eval()
    tag = checkpoint.replace('.pt','') + '_' + args.precision
    filename = out/(tag + ('_generation' if args.generate else '_forward') + '.jsonl')
    if filename.exists():
        raise FileExistsError(filename)
    print('AUDIT_CHECKPOINT', checkpoint, float(enc.context_scale.detach()), flush=True)
    with filename.open('w') as f, torch.no_grad():
        for ni, ix in enumerate(selected):
            sample = formatter._format_sample(raw[ix])
            batch = model.prepare_batch([sample])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                base, h, z, y, valid, gates = read_features(batch['time_series'])
                pbase = model.ts_projector(base)
                py = model.ts_projector(y)
                norm_base = base.float().square().mean(-1).sqrt()
                rho = (y.float()-base.float()).square().mean(-1).sqrt() / norm_base.clamp_min(1e-12)
                postrho = (py.float()-pbase.float()).square().mean(-1).sqrt() / pbase.float().square().mean(-1).sqrt().clamp_min(1e-12)
                hnorm = torch.nn.functional.normalize(h.float(), dim=-1)
                k = h.shape[1]
                offcos = ((hnorm.sum(dim=1).square().sum(dim=-1)-k)/(k*(k-1))).mean().item()
                means = h.float().mean(dim=1, keepdim=True)
                context_std_ratio = rms(h.float()-means)/max(rms(h),1e-12)
                record = dict(checkpoint=checkpoint, precision=args.precision, index=ix, id=sample['id'],
                    category=sample['category'], length=sample['length'], n_channels=sample['n_channels'],
                    scale=float(enc.context_scale), base_rms=rms(base), raw_context_rms=rms(z),
                    context_rms=rms(h), residual_rms=rms(y.float()-base.float()),
                    residual_ratio=quantiles(rho[valid]), projected_residual_ratio=quantiles(postrho[valid]),
                    projected_relative_l2=rms(py.float()-pbase.float())/max(rms(pbase),1e-12),
                    context_offdiag_cosine=offcos, context_between_token_ratio=context_std_ratio,
                    gates=gates, dtypes=dict(base=str(base.dtype), z=str(z.dtype), h=str(h.dtype), output=str(y.dtype)),
                    n_supervised=int((batch['labels'][...,1:] != -100).sum()), conditions={})
                for mode in args.modes:
                    replacement = intervention(mode, base, h, z, y, valid)
                    enc.forward = lambda x, replacement=replacement: replacement
                    result = model(**batch)
                    loss = float(result['loss'])
                    assert np.isfinite(loss), (checkpoint, ni, mode)
                    entry = dict(nll=loss)
                    if args.generate:
                        genbatch = model.prepare_batch([sample], training=False)
                        assert torch.equal(genbatch['time_series'], batch['time_series'])
                        genbatch.pop('labels')
                        token_ids = model.generate(**genbatch, max_new_tokens=config.training.eval_max_new_tokens,
                                                   do_sample=False)
                        text = model.tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
                        entry.update(prediction=text, scores=formatter.evaluate_answer(text, sample))
                    record['conditions'][mode] = entry
                    del result
                if ni == 0:
                    enc.forward = native_forward
                    native_loss = float(model(**batch)['loss'])
                    assert abs(native_loss-record['conditions']['on']['nll']) < 1e-7, (native_loss, record['conditions']['on']['nll'])
                    record['native_forward_verified'] = True
                enc.forward = native_forward
            f.write(json.dumps(record) + '\n')
            f.flush()
            if (ni+1) % 10 == 0 or ni == 0:
                print('AUDIT_PROGRESS', json.dumps(dict(checkpoint=checkpoint, completed=ni+1, total=len(selected),
                    elapsed_s=round(time.time()-started,1), last_nll={m: e['nll'] for m,e in record['conditions'].items()},
                    context_rms=record['context_rms'], raw_context_rms=record['raw_context_rms'],
                    projected_relative=record['projected_relative_l2'])), flush=True)
            del batch, base, h, z, y, pbase, py, replacement
        gc.collect()
    print('AUDIT_CHECKPOINT_DONE', str(filename), flush=True)
hook.remove()
print('AUDIT_DONE', round(time.time()-started,1), 'seconds', flush=True)
