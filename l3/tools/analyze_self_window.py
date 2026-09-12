"""Would 'zoom on your own first guess' land on the anomaly?

NOTE (12/09): channel-aware coverage after the Codex review; the first version ignored the
channel in the multichannel category and overstated its coverage (68.8% instead of 62.5%).

CPU-only, on the node: reads va1's val520 predictions (Z0 no_zoom JSONL) and the policy
windows chosen by vz1, then measures, per category, how often a window built around va1's
own predicted interval contains the gold anomaly.
"""
import json
import re
import sys
from pathlib import Path

R = Path("/root/l3/zoom_20260911c")
VA1 = R / "Z0" / "val520_no_zoom.jsonl"
VZ1 = R / "Z1" / "val520_policy.jsonl"
MIN_W = 1024


def payload(text):
    text = text or ""
    i = text.find("{")
    if i < 0:
        return {}
    depth, instr, esc = 0, False, False
    for j, ch in enumerate(text[i:], i):
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
        elif ch == '"':
            instr = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i:j + 1])
                except Exception:
                    return {}
    return {}


def gold_channels(row):
    """Channel of each gold interval, aligned with gold_intervals(row)."""
    p = payload(row["ground_truth"])
    if row["category"] == "localize_all_channels":
        return [str(a["channel"]) for a in p.get("anomalies", [])
                if isinstance(a, dict) and isinstance(a.get("start"), int)]
    return ["ch_00"] * len(gold_intervals(row))


def gold_intervals(row):
    p = payload(row["ground_truth"])
    cat = row["category"]
    if cat == "localize_all_channels":
        return [(int(a["start"]), int(a["end"])) for a in p.get("anomalies", [])
                if isinstance(a, dict) and isinstance(a.get("start"), int)]
    if cat in {"localize", "classify_with_evidence"}:
        if isinstance(p.get("start"), int) and isinstance(p.get("end"), int):
            return [(int(p["start"]), int(p["end"]))]
    return []


def pred_interval(row):
    p = payload(row["prediction"])
    if isinstance(p.get("start"), int) and isinstance(p.get("end"), int) and p["start"] < p["end"]:
        return int(p["start"]), int(p["end"])
    anomalies = p.get("anomalies")
    if isinstance(anomalies, list):
        for a in anomalies:
            if isinstance(a, dict) and isinstance(a.get("start"), int) and isinstance(a.get("end"), int):
                return int(a["start"]), int(a["end"])
    return None


def window(center_lo, center_hi, length, factor, min_w=MIN_W):
    width = max(min_w, int((center_hi - center_lo) * factor))
    mid = (center_lo + center_hi) // 2
    s, e = mid - width // 2, mid + (width - width // 2)
    s, e = max(0, s), min(length, e)
    if e - s < min(min_w, length):
        if s == 0:
            e = min(length, min_w)
        else:
            s = max(0, e - min_w)
    return s, e


def covers(win, gold, frac=0.9):
    inter = max(0, min(win[1], gold[1]) - max(win[0], gold[0]))
    return inter >= frac * (gold[1] - gold[0])


va1 = [json.loads(l) for l in VA1.open(encoding="utf-8") if l.strip()]
vz1 = {json.loads(l)["id"]: json.loads(l) for l in VZ1.open(encoding="utf-8") if l.strip()}
print(f"rows va1={len(va1)} vz1={len(vz1)}")

stats = {}
for row in va1:
    cat = row["category"]
    golds = gold_intervals(row)
    if not golds:
        continue
    length = int(payload(row["ground_truth"]).get("_L", 0)) or int(row.get("length", 0))
    s = stats.setdefault(cat, {"n": 0, "has_pred": 0, "self3": 0, "self6": 0,
                               "policy_cov": 0, "policy_w": 0, "self3_w": 0})
    s["n"] += 1
    pi = pred_interval(row)
    if pi:
        s["has_pred"] += 1
        for factor, key in ((3.0, "self3"), (6.0, "self6")):
            win = window(pi[0], pi[1], length, factor)
            if any(covers(win, g) for g in golds):
                s[key] += 1
            if key == "self3":
                s["self3_w"] += win[1] - win[0]
    z = vz1.get(row["id"], {}).get("zoom")
    if z:
        win = (int(z["start"]), int(z["end"]))
        s["policy_w"] += win[1] - win[0]
        # Channel-aware (fix 12/09, Codex review): a window only covers events of ITS channel.
        same = [g for g, ch in zip(golds, gold_channels(row)) if ch == str(z["channel"])]
        if any(covers(win, g) for g in same):
            s["policy_cov"] += 1

print(f"{'category':26} {'n':>4} {'pred%':>6} {'self3%':>7} {'self6%':>7} {'policy%':>8} "
      f"{'self3_w':>8} {'policy_w':>9}")
for cat, s in sorted(stats.items()):
    n = max(1, s["n"])
    hp = max(1, s["has_pred"])
    print(f"{cat:26} {s['n']:4d} {100*s['has_pred']/n:6.1f} {100*s['self3']/n:7.1f} "
          f"{100*s['self6']/n:7.1f} {100*s['policy_cov']/n:8.1f} "
          f"{s['self3_w']/hp:8.0f} {s['policy_w']/n:9.0f}")
