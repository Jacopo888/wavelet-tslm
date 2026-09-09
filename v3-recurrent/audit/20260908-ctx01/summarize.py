"""Summarize completed paired audits; bootstrap describes validation-example uncertainty."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
expected = int(sys.argv[2]) if len(sys.argv) > 2 else 80
rng = np.random.default_rng(20260908)

def paired(values):
    a = np.asarray(values, dtype=float)
    boots = a[rng.integers(0, len(a), size=(10000, len(a)))].mean(1)
    return dict(mean=float(a.mean()), ci95=np.quantile(boots,[.025,.975]).tolist(),
                fraction_negative=float((a<0).mean()), fraction_positive=float((a>0).mean()))

summary = {}
for file in sorted(root.rglob('*.jsonl')):
    rows = [json.loads(s) for s in file.read_text().splitlines() if s.strip()]
    if len(rows) != expected or not rows or 'conditions' not in rows[0]:
        continue
    modes = list(rows[0]['conditions'])
    assert all(list(r['conditions']) == modes for r in rows)
    assert len({r['id'] for r in rows}) == len(rows)
    result = dict(n=len(rows), categories=dict(Counter(r['category'] for r in rows)),
        scale=rows[0]['scale'], precision=rows[0]['precision'],
        mean_nll={m:float(np.mean([r['conditions'][m]['nll'] for r in rows])) for m in modes},
        differences_vs_off={}, mean_geometry={}, by_category={})
    for mode in modes:
        if mode != 'off' and 'off' in modes:
            result['differences_vs_off'][mode] = paired([r['conditions'][mode]['nll']-r['conditions']['off']['nll'] for r in rows])
    for name in ('base_rms','raw_context_rms','context_rms','residual_rms','projected_relative_l2',
                 'context_offdiag_cosine','context_between_token_ratio'):
        a = [r[name] for r in rows]
        result['mean_geometry'][name] = dict(mean=float(np.mean(a)), median=float(np.median(a)), min=min(a), max=max(a))
    result['mean_geometry']['median_token_residual_ratio'] = float(np.median([r['residual_ratio']['p50'] for r in rows]))
    result['mean_geometry']['median_token_projected_ratio'] = float(np.median([r['projected_residual_ratio']['p50'] for r in rows]))
    if 'roll' in modes:
        result['on_minus_roll_nll'] = paired([r['conditions']['on']['nll']-r['conditions']['roll']['nll'] for r in rows])
    for category in result['categories']:
        subset = [r for r in rows if r['category']==category]
        result['by_category'][category] = dict(n=len(subset), mean_nll={m:float(np.mean([r['conditions'][m]['nll'] for r in subset])) for m in modes})
    if 'scores' in rows[0]['conditions']['on']:
        result['primary'] = {m:float(np.mean([r['conditions'][m]['scores']['primary'] for r in rows])) for m in modes}
        result['primary_on_minus_off'] = paired([r['conditions']['on']['scores']['primary']-r['conditions']['off']['scores']['primary'] for r in rows])
        result['identical_predictions'] = sum(r['conditions']['on']['prediction']==r['conditions']['off']['prediction'] for r in rows)
        for category in result['categories']:
            subset = [r for r in rows if r['category']==category]
            result['by_category'][category]['primary'] = {m:float(np.mean([r['conditions'][m]['scores']['primary'] for r in subset])) for m in modes}
    summary[str(file.relative_to(root))] = result
(root/'summary.json').write_text(json.dumps(summary, indent=2))
for name, result in summary.items():
    print(name)
    print(json.dumps(result, indent=2))
