"""Read-only CPU audit of saved validation predictions; no model or test access."""
from pathlib import Path
import collections
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'integrity/artifacts/l3_zoom_20260911'

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def payload(text):
    return json.JSONDecoder().raw_decode(text[text.index('{'):])[0]

def events(row):
    gold = payload(row['ground_truth'])
    if row['category'] == 'localize_all_channels':
        return gold.get('anomalies', [])
    if row['category'] in ('localize', 'classify_with_evidence') and gold.get('start') is not None and gold.get('end') is not None:
        if gold['end'] > gold['start']:
            return [dict(channel='ch_00', start=gold['start'], end=gold['end'])]
    return []

def covers(start, end, event):
    return max(0, min(end, event['end'])-max(start, event['start'])) >= .9*(event['end']-event['start'])

def centered_window(mid, width, length):
    start = max(0, min(length-width, int(mid-width/2)))
    return start, start+width

def bootstrap(values):
    arr = np.asarray(values)
    rng = np.random.default_rng(20260912)
    means = arr[rng.integers(len(arr), size=(10000, len(arr)))].mean(axis=1)
    return dict(n=len(arr), delta=float(arr.mean()), ci95=np.quantile(means,[.025,.975]).tolist())

def main():
    paths = {'va1': ART/'Z0/val520_no_zoom.jsonl',
             'policy': ART/'Z1/val520_policy.jsonl',
             'oracle': ART/'Z1/diag/val520_vz1_oracle_text.jsonl'}
    data = {key: rows(path) for key,path in paths.items()}
    assert all(len(value)==520 for value in data.values())
    for candidate in ('policy','oracle'):
        assert [(r['id'],r['category'],r['ground_truth']) for r in data[candidate]] == [(r['id'],r['category'],r['ground_truth']) for r in data['va1']]
    result = {'scope':'Existing val520 only; intervals are descriptive post-hoc, not confirmatory.',
              'input_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},
              'paired':{},'windows':{},'oracle_center_only_proxy':{},'saved_summaries':{},'oracle_positive_scores':{}}
    for relative in ('Z0/val520_no_zoom.json','Z1/val520_policy.json',
                     'Z1/diag/val520_vz1_no_zoom.json','Z1/diag/val520_vz1_oracle_text.json',
                     'Z1/diag/val520_vz1_oracle_w6144.json'):
        summary=json.loads((ART/relative).read_text(encoding='utf-8'))
        result['saved_summaries'][relative]={'primary':summary['primary'],
            'per_category':{k:v['primary'] for k,v in summary['per_category'].items()}}
    for candidate in ('policy','oracle'):
        for category in ['ALL']+sorted({r['category'] for r in data['va1']}):
            values=[b['primary']-a['primary'] for a,b in zip(data['va1'],data[candidate]) if category=='ALL' or a['category']==category]
            result['paired'][candidate+'_minus_va1/'+category]=bootstrap(values)
    stats=collections.defaultdict(list)
    for row, oracle in zip(data['policy'],data['oracle']):
        ev=events(row)
        if not ev:
            continue
        z=row['zoom']; length=payload(row['ground_truth'])['_L']
        same=[e for e in ev if e['channel']==z['channel']]
        mid=(z['start']+z['end'])/2
        entry={'width':z['window_len'],'time_only_coverage':float(any(covers(z['start'],z['end'],e) for e in ev)),
               'channel_aware_coverage':float(any(covers(z['start'],z['end'],e) for e in same)),
               'correct_channel':float(bool(same)),
               'nearest_center_error':min((abs(mid-(e['start']+e['end'])/2) for e in same),default=float('nan')),
               'oracle_width':oracle['zoom']['window_len']}
        for width in (1024,2048):
            start,end=centered_window(mid,width,length)
            entry['recenter_same_policy_'+str(width)]=float(any(covers(start,end,e) for e in same))
        stats[row['category']].append(entry)
    for category, entries in stats.items():
        result['windows'][category]={'n_positive':len(entries)}
        for key in entries[0]:
            values=np.asarray([e[key] for e in entries])
            result['windows'][category][key]=float(np.nanmean(values))
            if key=='nearest_center_error':
                result['windows'][category]['median_center_error']=float(np.nanmedian(values))
    for category in ('localize','classify_with_evidence','localize_all_channels'):
        positive=[r for r in data['oracle'] if r['category']==category and events(r)]
        result['oracle_positive_scores'][category]={'n_positive':len(positive),
            'primary':float(np.mean([r['primary'] for r in positive]))}
    # Demonstrate how informative the oracle header is, without reading signal values.
    # This is NOT an official end-to-end score and uses an explicitly fixed width.
    for category in ('localize','classify_with_evidence'):
        selected=[r for r in data['oracle'] if r['category']==category and events(r)]
        for width in (64,128,256):
            scores=[]; errors=[]
            for row in selected:
                e=events(row)[0]; z=row['zoom']; mid=(z['start']+z['end'])/2
                start,end=centered_window(mid,width,payload(row['ground_truth'])['_L'])
                inter=max(0,min(end,e['end'])-max(start,e['start']))
                scores.append(inter/(end-start+e['end']-e['start']-inter))
                errors.append(abs(mid-(e['start']+e['end'])/2))
            result['oracle_center_only_proxy'][category+'/'+str(width)]={'n_positive':len(selected),
                'mean_iou':float(np.mean(scores)),'max_header_center_error':float(max(errors)),
                'header_center_within_half_sample':sum(e<=.5 for e in errors)}
    out=Path(__file__).with_name('AUDIT-ZOOM-20260912.json')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
