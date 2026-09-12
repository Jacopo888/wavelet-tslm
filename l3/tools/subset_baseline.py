"""Baseline appaiata mancante (Codex follow-up §2): vz1 SENZA zoom sugli stessi 208 ID del gate.

Solo CPU: filtra i JSONL per riga già esistenti. Nessuna GPU.
Uso: subset_baseline.py [run_root]
"""
import json
import random
import statistics
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/l3/zoom_20260911c")
GATE = R / "Z1" / "gate" / "rows.jsonl"
SOURCES = {
    "va1_no_zoom": R / "Z0" / "val520_no_zoom.jsonl",
    "vz1_no_zoom": R / "Z1" / "diag" / "val520_vz1_no_zoom.jsonl",
    "vz1_policy": R / "Z1" / "val520_policy.jsonl",
    "vz1_oracle_narrow": R / "Z1" / "diag" / "val520_vz1_oracle_text.jsonl",
}

gate_rows = [json.loads(l) for l in GATE.open(encoding="utf-8") if l.strip()]
ids = sorted({r["id"] for r in gate_rows if r["condition"] == "true"})
cats = {r["id"]: r["category"] for r in gate_rows}
gate = {}
for r in gate_rows:
    gate.setdefault(r["condition"], {})[r["id"]] = r["primary"]
print(f"gate ids={len(ids)} conditions={sorted(gate)}")

series = {}
for name, path in SOURCES.items():
    if not path.is_file():
        print(f"MISSING {name}: {path}")
        continue
    rows = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["id"]] = float(row["primary"])
    hit = [i for i in ids if i in rows]
    series[name] = {i: rows[i] for i in hit}
    print(f"{name:20} rows={len(rows):4} on gate subset={len(hit)}")
for name, values in gate.items():
    series[f"gate_{name}"] = {i: values[i] for i in ids if i in values}


def mean(d, subset=None):
    v = [x for i, x in d.items() if subset is None or cats[i] == subset]
    return sum(v) / len(v) if v else float("nan")


def paired(a, b, resamples=10000, seed=20260912):
    common = sorted(set(a) & set(b))
    d = [a[i] - b[i] for i in common]
    if not d:
        return None
    rng = random.Random(seed)
    boots = []
    for _ in range(resamples):
        boots.append(statistics.fmean(rng.choices(d, k=len(d))))
    boots.sort()
    lo, hi = boots[int(0.025 * resamples)], boots[int(0.975 * resamples) - 1]
    return len(d), statistics.fmean(d), lo, hi


print(f"\n{'serie':26} {'n':>4} {'tutti':>8} {'localize':>9} {'classify':>9} {'all_ch':>8}")
for name in sorted(series):
    d = series[name]
    print(f"{name:26} {len(d):4} {mean(d):8.4f} {mean(d,'localize'):9.4f} "
          f"{mean(d,'classify_with_evidence'):9.4f} {mean(d,'localize_all_channels'):8.4f}")

print("\nconfronti appaiati (bootstrap 10k, seed 20260912):")
for a, b in (("gate_true", "gate_swap"), ("gate_true", "gate_zero"),
             ("gate_offcenter", "vz1_no_zoom"), ("gate_true", "vz1_no_zoom"),
             ("vz1_no_zoom", "va1_no_zoom"), ("gate_offcenter", "va1_no_zoom")):
    if a not in series or b not in series:
        continue
    out = paired(series[a], series[b])
    if out:
        n, m, lo, hi = out
        print(f"  {a:16} - {b:16} n={n:3} Δ={m:+.4f} CI95=[{lo:+.4f}; {hi:+.4f}]")
