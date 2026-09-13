"""Tetto con finestra perfetta, per categoria, sulle righe zoomate da entrambi."""
import json
import random
import statistics
import sys
from collections import defaultdict


def load(path):
    out = defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            out[str(row["selector"])][str(row["id"])] = row
    return out


old = load(sys.argv[1])
new = load(sys.argv[2])
shared = sorted(set(old["oracle_matched"]) & set(new["oracle_matched"]))


def boot(values, seed=3, resamples=10000):
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values)))
                   for _ in range(resamples))
    return (statistics.fmean(values), means[int(0.025 * resamples)],
            means[min(resamples - 1, int(0.975 * resamples))])


header = "{:26} {:>4} {:>10} {:>10} {:>26} {:>10} {:>10}".format(
    "categoria", "n", "tetto vz1", "tetto vw1", "delta tetto", "pol vz1", "pol vw1")
print(header)
for category in sorted({old["oracle_matched"][i]["category"] for i in shared}):
    ids = [i for i in shared if old["oracle_matched"][i]["category"] == category]
    o1 = statistics.fmean(float(old["oracle_matched"][i]["primary"]) for i in ids)
    o2 = statistics.fmean(float(new["oracle_matched"][i]["primary"]) for i in ids)
    p1 = statistics.fmean(float(old["oracle_matched"][i]["policy_primary"]) for i in ids)
    p2 = statistics.fmean(float(new["oracle_matched"][i]["policy_primary"]) for i in ids)
    delta, low, high = boot(
        [float(new["oracle_matched"][i]["primary"])
         - float(old["oracle_matched"][i]["primary"]) for i in ids], seed=len(category))
    print("{:26} {:4} {:10.4f} {:10.4f}  {:+.4f} [{:+.4f};{:+.4f}] {:10.4f} {:10.4f}".format(
        category, len(ids), o1, o2, delta, low, high, p1, p2))
