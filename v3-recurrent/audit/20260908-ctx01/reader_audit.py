"""Encoder-only interventions to locate homogenization of recurrent context."""
import argparse
import json
import os
from pathlib import Path
import sys

ap=argparse.ArgumentParser()
ap.add_argument('--archive', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--checkpoint', default='epoch_1_end.pt')
args=ap.parse_args()
archive, out=Path(args.archive), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
os.environ['TSLM_V3_GATE_LOG_EVERY']='0'
sys.path.insert(0,'/root/timerlm/sft/tslm')
import torch
from datasets import Dataset
from src.models.wavelet_encoder_v3 import RecurrentWaveletTSEncoder

torch.set_num_threads(2)
ck=torch.load(archive/'run/checkpoints'/args.checkpoint, map_location='cpu', weights_only=False)
cfg=dict(ck['config']['model']['encoder'])
cfg.pop('type',None)
enc=RecurrentWaveletTSEncoder(**cfg).to(device='cuda',dtype=torch.bfloat16)
for p in enc.parameters(): p.data=p.data.float()
enc.load_state_dict({k.removeprefix('ts_encoder.'):v for k,v in ck['model_state'].items() if k.startswith('ts_encoder.')},strict=True)
enc.eval()
del ck
raw=Dataset.from_parquet('/root/timerlm/anomalyXL/datasets/anomalyxl-train/anomalyxl-precise/val.parquet')
meta=raw.select_columns(['id','category','length','n_channels'])[:]
selected=[]
for cat in sorted(set(meta['category'])):
    candidates=[i for i,c in enumerate(meta['category']) if c==cat]
    candidates.sort(key=lambda i:(meta['length'][i],meta['id'][i]))
    selected.extend([candidates[0],candidates[-1]])

captured={}
trace=[]
layer_values={}
def rms(x): return x.detach().float().square().mean().sqrt().item()
def norm_hook(module,inputs,output):
    captured['z']=inputs[0].detach()
    captured['h']=output.detach()
def layer_hook(label):
    def hook(module, inputs, output):
        value=output[0] if isinstance(output,tuple) else output
        layer_values[label]=rms(value)
    return hook
def block_hook(module,inputs,output):
    trace.append(dict(before_rms=rms(inputs[0]),after_rms=rms(output),
                      increment_rms=rms(output-inputs[0]), **layer_values))
enc.context_norm.register_forward_hook(norm_hook)
enc.blocks[0].cross.register_forward_hook(layer_hook('cross_rms'))
enc.blocks[0].self_attn.register_forward_hook(layer_hook('self_rms'))
enc.blocks[0].mlp.register_forward_hook(layer_hook('mlp_rms'))
enc.blocks[0].register_forward_hook(block_hook)
def zero_tensor(module, inputs, output): return torch.zeros_like(output)
def zero_attn(module, inputs, output): return (torch.zeros_like(output[0]),output[1])
original_tokens=enc._tokens
original_block_forward=enc.blocks[0].forward

def run(x,depth=None,condition='native'):
    temporary=[]
    if condition in ('no_mlp','cross_only'): temporary.append(enc.blocks[0].mlp.register_forward_hook(zero_tensor))
    if condition in ('no_self','cross_only'): temporary.append(enc.blocks[0].self_attn.register_forward_hook(zero_attn))
    if condition=='zero_source_values': enc._tokens=lambda v,p,b: original_tokens(torch.zeros_like(v),p,b)
    if condition=='fp32_state':
        enc.blocks[0].forward=lambda z,q,e,g,p: original_block_forward(z.float(),q,e,g,p)
    trace.clear()
    captured.clear()
    try:
        enc._encode_channel(x, depth_override=depth)
        h,z=captured['h'].clone(),captured['z'].clone()
        mean=h.float().mean(dim=1,keepdim=True)
        unit=torch.nn.functional.normalize(h.float(),dim=-1)
        n=h.shape[1]
        sigma=z.float().var(dim=-1,unbiased=False).add(enc.context_norm.eps).sqrt()
        result=dict(condition=condition,depth=enc.last_depth,raw_rms=rms(z),context_rms=rms(h),
            token_variable_fraction=rms(h.float()-mean)/max(rms(h),1e-12),
            offdiag_cosine=float(((unit.sum(1).square().sum(-1)-n)/(n*(n-1))).mean()),
            mean_feature_std=float(sigma.mean()),
            scale_over_feature_std_median=float((enc.context_scale.abs()/sigma).median()),
            trace=list(trace),gates=list(enc.last_gates),mean_context=mean.mean(0).flatten().cpu().tolist())
        return h,result
    finally:
        for handle in temporary: handle.remove()
        enc._tokens=original_tokens
        enc.blocks[0].forward=original_block_forward

rows=[]
with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
    for ix in selected:
        sample=raw[ix]
        x=torch.tensor(sample['channel_values'],device='cuda',dtype=torch.bfloat16)
        h,base_result=run(x)
        result=dict(index=ix,id=sample['id'],category=sample['category'],length=sample['length'],n_channels=sample['n_channels'],native=base_result,interventions=[])
        shifted,shifted_result=run(x.roll(max(1,x.shape[-1]//3),dims=-1))
        result['shifted_input_context_relative_difference']=rms(shifted.float()-h.float())/rms(h)
        for condition in ('no_mlp','no_self','cross_only','zero_source_values','fp32_state'):
            other,record=run(x,condition=condition)
            record['relative_difference_from_native']=rms(other.float()-h.float())/rms(h)
            result['interventions'].append(record)
        result['depth_sweep']=[]
        maxdepth=base_result['depth']
        for depth in sorted({min(d,maxdepth) for d in (1,2,4,6,8,11)}):
            other,record=run(x,depth=depth)
            result['depth_sweep'].append(record)
        rows.append(result)
        print('READER_SAMPLE',json.dumps({k:result[k] for k in ('id','length','n_channels','shifted_input_context_relative_difference')}),flush=True)
tag=args.checkpoint.removesuffix('.pt')
(out/(tag+'_reader.json')).write_text(json.dumps(dict(checkpoint=args.checkpoint,scale=float(enc.context_scale),samples=rows),indent=2))
print('READER_DONE',tag,flush=True)
