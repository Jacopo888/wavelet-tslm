"""Paired deltas between gate conditions on the same rows (CPU, node or local)."""
import json
import statistics
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "/root/l3/zoom_20260911c/Z1/gate/rows.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
score = {(r["id"], r["condition"]): r["primary"] for r in rows}
cats = {r["id"]: r["category"] for r in rows}
ids = sorted({r["id"] for r in rows if r["condition"] == "true"})
print(f"rows={len(rows)} ids={len(ids)}")
for other in ("swap", "zero", "offcenter"):
    pairs = [(i, score[(i, "true")] - score[(i, other)]) for i in ids if (i, other) in score]
    d = [x for _, x in pairs]
    print(f"true - {other:9} n={len(d):3} mean={sum(d)/len(d):+.4f} sd={statistics.pstdev(d):.4f} "
          f"|d|>0.05: {sum(1 for x in d if abs(x) > 0.05)}")
    per = defaultdict(list)
    for i, x in pairs:
        per[cats[i]].append(x)
    for cat in sorted(per):
        v = per[cat]
        print(f"    {cat:24} n={len(v):3} mean={sum(v)/len(v):+.4f}")
