"""Confronto onesto dei tetti: vz1bis e vw1 sulle righe che ENTRAMBI hanno zoomato."""
import json
import random
import statistics
import sys

OLD, NEW, OLD_CONTENT = sys.argv[1], sys.argv[2], sys.argv[3]


def load(path, key="selector"):
    out = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        out.setdefault(str(row[key]), {})[str(row["id"])] = row
    return out


old = load(OLD)
new = load(NEW)
content = load(OLD_CONTENT, key="condition")

shared = sorted(set(old["oracle_matched"]) & set(new["oracle_matched"]))
print(f"righe zoomate da vz1bis: {len(old['oracle_matched'])}  da vw1: "
      f"{len(new['oracle_matched'])}  in comune: {len(shared)}")


def boot(values, seed=20260913, resamples=10000):
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values)))
                   for _ in range(resamples))
    return (statistics.fmean(values), means[int(0.025 * resamples)],
            means[min(resamples - 1, int(0.975 * resamples))])


def mean(rows, ids, field="primary"):
    return statistics.fmean(float(rows[i][field]) for i in ids)


print("\nsulle righe in comune:")
print(f"  policy vz1bis (episodio)   {mean(old['oracle_matched'], shared, 'policy_primary'):.4f}")
print(f"  policy vw1 (episodio)      {mean(new['oracle_matched'], shared, 'policy_primary'):.4f}")
for arm in ("oracle_matched", "wavelet_in_policy_channel"):
    print(f"  {arm:26} vz1bis {mean(old[arm], shared):.4f}  "
          f"vw1 {mean(new[arm], shared):.4f}")

diff = [float(new["oracle_matched"][i]["primary"]) - float(old["oracle_matched"][i]["primary"])
        for i in shared]
m, lo, hi = boot(diff)
print(f"\ntetto oracle, vw1 - vz1bis, appaiato: {m:+.4f} CI[{lo:+.4f}; {hi:+.4f}] n={len(shared)}")

pol = [float(new["oracle_matched"][i]["policy_primary"])
       - float(old["oracle_matched"][i]["policy_primary"]) for i in shared]
m, lo, hi = boot(pol, seed=7)
print(f"policy, vw1 - vz1bis, appaiato:      {m:+.4f} CI[{lo:+.4f}; {hi:+.4f}]")

gap_old = [float(old["oracle_matched"][i]["primary"])
           - float(old["oracle_matched"][i]["policy_primary"]) for i in shared]
gap_new = [float(new["oracle_matched"][i]["primary"])
           - float(new["oracle_matched"][i]["policy_primary"]) for i in shared]
for label, values in (("divario oracle-policy vz1bis", gap_old),
                      ("divario oracle-policy vw1", gap_new)):
    m, lo, hi = boot(values, seed=11)
    print(f"{label}: {m:+.4f} CI[{lo:+.4f}; {hi:+.4f}]")

# Controllo del riferimento: la condizione self_true del gate del contenuto
# (finestra forzata) contro l'episodio reale, sulle stesse righe di vz1bis.
forced = content.get("self_true", {})
both = sorted(set(forced) & set(old["oracle_matched"]))
if both:
    print(f"\ncontrollo convenzione su vz1bis (n={len(both)}): "
          f"finestra forzata {mean(forced, both):.4f} vs episodio reale "
          f"{mean(old['oracle_matched'], both, 'policy_primary'):.4f}")
