"""Net parameter movement between saved checkpoints (not per-step update norms)."""
import json
from pathlib import Path
import sys
import torch

torch.set_num_threads(2)
archive, out = Path(sys.argv[1]), Path(sys.argv[2])
snapshot = json.loads((out/'snapshot.json').read_text())
families = dict(blocks='ts_encoder.blocks.', level_gate='ts_encoder.gate_head.',
                step_embed='ts_encoder.step_embed.', base_value='ts_encoder.value_proj.',
                base_band='ts_encoder.band_embed.', norm='ts_encoder.context_norm.',
                projector='ts_projector.')
previous = None
rows = []
for row in snapshot['checkpoint_rows']:
    ck = torch.load(archive/'run/checkpoints'/row['file'], map_location='cpu', weights_only=False)
    state = ck['model_state']
    if previous is not None:
        item = dict(from_step=previous_step, to_step=row['optimizer_step'], families={})
        for family, prefix in families.items():
            names = [name for name in state if name.startswith(prefix)]
            n = sum(state[name].numel() for name in names)
            before = sum(previous[name].float().square().sum().item() for name in names)
            now = sum(state[name].float().square().sum().item() for name in names)
            delta = sum((state[name].float()-previous[name].float()).square().sum().item() for name in names)
            item['families'][family] = dict(weight_rms=(now/n)**.5,
                net_delta_rms=(delta/n)**.5, net_delta_over_previous_norm=(delta/max(before,1e-30))**.5)
        rows.append(item)
    previous, previous_step = state, row['optimizer_step']
    del ck
(out/'parameter_deltas.json').write_text(json.dumps(rows, indent=2))
for row in rows:
    print(json.dumps(dict(from_step=row['from_step'], to_step=row['to_step'], blocks=row['families']['blocks'])))
