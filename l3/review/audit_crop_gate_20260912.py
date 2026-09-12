"""CPU-only paired audit of existing crop gate rows, without model execution."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'integrity/artifacts/l3_zoom_20260911'

def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def main():
    gate_path = ART / 'Z1/gate/rows.jsonl'
    baseline_path = ART / 'Z0/val520_no_zoom.jsonl'
    rows = read(gate_path)
    conditions = {c: {r['id']: r for r in rows if r['condition'] == c}
                  for c in ('true', 'swap', 'zero', 'offcenter')}
    ids = sorted(conditions['true'])
    assert len(ids) == 208 and len(rows) == 832
    for condition in conditions.values():
        assert set(condition) == set(ids)
        assert all(condition[i]['ground_truth'] == conditions['true'][i]['ground_truth'] for i in ids)
    baseline = {r['id']: r for r in read(baseline_path)}
    assert all(baseline[i]['ground_truth'] == conditions['true'][i]['ground_truth'] for i in ids)
    rng = np.random.default_rng(20260912)

    def paired(a, b, selected):
        delta = np.array([a[i]['primary'] - b[i]['primary'] for i in selected])
        boot = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(axis=1)
        return {'n': len(selected), 'delta': float(delta.mean()),
                'ci95': np.quantile(boot, [.025, .975]).tolist()}

    result = {'scope': 'Descriptive post-hoc bootstrap; no equivalence margin or multiple-testing correction.',
              'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (gate_path, baseline_path)}, 'comparisons': {}, 'aligned_va1': {}}
    for name in ('swap', 'zero', 'offcenter'):
        result['comparisons']['true_minus_' + name] = {
            **paired(conditions['true'], conditions[name], ids),
            'changed_prediction': sum(conditions['true'][i]['prediction'] != conditions[name][i]['prediction'] for i in ids),
            'changed_primary': sum(conditions['true'][i]['primary'] != conditions[name][i]['primary'] for i in ids)}
        if name in ('swap', 'zero'):
            assert all(conditions['true'][i]['zoom'] == conditions[name][i]['zoom'] for i in ids)
    for category in ('ALL', 'localize', 'classify_with_evidence', 'localize_all_channels'):
        selected = [i for i in ids if category == 'ALL' or conditions['true'][i]['category'] == category]
        result['aligned_va1'][category] = {
            'va1': float(np.mean([baseline[i]['primary'] for i in selected])),
            'vz1_offcenter': float(np.mean([conditions['offcenter'][i]['primary'] for i in selected])),
            'vz1_offcenter_minus_va1': paired(conditions['offcenter'], baseline, selected)}
    result['limitation'] = 'va1 is a different checkpoint. This does not replace the missing same-subset vz1 no-zoom control; oracle offcenter still uses gold information.'
    output = Path(__file__).with_name('AUDIT-CROP-GATE-20260912.json')
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
