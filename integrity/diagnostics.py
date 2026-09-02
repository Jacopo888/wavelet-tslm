#!/usr/bin/env python
"""Diagnostic scorer/validator for AnomalyXL eval outputs — SEPARATE from the
official scorer (which stays untouched for comparability).

Reads per-example eval artifacts (the JSON files with an "outputs" list, or
the new per-example JSONL) and reports what the official aggregate hides:

  * strict schema validation of predictions (e.g. "present": "false" — a
    string — which the official scorer reads as TRUE via bool());
  * localize decomposed: presence confusion matrix, accuracy on the 39
    negatives, IoU restricted to the 161 gold-positives, acc@IoU bands;
  * all_channels decomposed: empty-case accuracy vs F1 on non-empty golds;
  * null-baseline comparison (always-"absent" scores .195 on localize,
    .230 on all_channels by test composition);
  * per-category primary with 95% bootstrap CI (seeded, deterministic);
  * optional paired comparison of two runs on the same test.

Usage:
  python diagnostics.py runA.json [--name-a m30ep4] [--b runB.json --name-b full1]
"""

from __future__ import annotations

import argparse
import json
import random
import sys

NULL_BASELINES = {"localize": 0.195, "localize_all_channels": 0.230}
BOOT_N = 10000
BOOT_SEED = 20260902


# --------------------------------------------------------------------------
# loading

def load_entries(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["outputs"] if isinstance(d, dict) else d


def gold_payload(entry: dict):
    """Parse the 'category:{json}' gold envelope."""
    gt = entry.get("ground_truth", "")
    cat, _, payload = gt.partition(":")
    try:
        return cat, json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return cat, None


# --------------------------------------------------------------------------
# strict prediction parsing

def first_json_object(text: str):
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def strict_violations(category: str, pred: dict | None) -> list[str]:
    """Schema-strict issues the official scorer tolerates or misreads."""
    if pred is None:
        return ["no_json_object"]
    v: list[str] = []
    if category == "localize":
        p = pred.get("present")
        if not isinstance(p, bool):
            v.append(f"present_not_bool:{type(p).__name__}")
        if (p is True) or (not isinstance(p, bool) and p):
            if not _is_int(pred.get("start")):
                v.append("start_not_int")
            if not _is_int(pred.get("end")):
                v.append("end_not_int")
    elif category == "classify_with_evidence":
        if not isinstance(pred.get("kind"), str):
            v.append("kind_not_str")
    elif category == "measure_magnitude":
        if not _is_num(pred.get("magnitude_sigma")):
            v.append("magnitude_not_num")
    elif category == "localize_all_channels":
        ev = pred.get("anomalies")
        if not isinstance(ev, list):
            v.append("anomalies_not_list")
        else:
            bad = sum(1 for e in ev
                      if not (isinstance(e, dict) and "channel" in e
                              and _is_int(e.get("start")) and _is_int(e.get("end"))))
            if bad:
                v.append(f"malformed_events:{bad}")
    elif category == "lead_lag_with_magnitude":
        if not isinstance(pred.get("direction"), str):
            v.append("direction_not_str")
        if not _is_int(pred.get("lag_samples")):
            v.append("lag_not_int")
    return v


# --------------------------------------------------------------------------
# decompositions

def iou(a0, a1, b0, b1) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def localize_report(entries: list[dict]) -> dict:
    conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    pos_ious: list[float] = []          # over ALL gold-positives (miss = 0)
    matched_ious: list[float] = []      # only when both said present
    for e in entries:
        _, gold = gold_payload(e)
        if gold is None:
            continue
        gp = bool(gold.get("present"))
        pred = first_json_object(e.get("predicted_answer")
                                 or e.get("prediction_raw") or "")
        # official leniency on purpose here: this mirrors what was scored
        pp = bool(pred.get("present", False)) if pred else False
        conf["tp" if gp and pp else "fp" if pp else "fn" if gp else "tn"] += 1
        if gp:
            i = 0.0
            if (pred and pp and _is_num(pred.get("start"))
                    and _is_num(pred.get("end"))):
                i = iou(int(gold["start"]), int(gold["end"]),
                        int(pred["start"]), int(pred["end"]))
                matched_ious.append(i)
            pos_ious.append(i)
    n_pos, n_neg = conf["tp"] + conf["fn"], conf["fp"] + conf["tn"]
    return {
        "confusion": conf,
        "n_gold_positive": n_pos,
        "n_gold_negative": n_neg,
        "negatives_accuracy": conf["tn"] / n_neg if n_neg else None,
        "positives_mean_iou": (sum(pos_ious) / len(pos_ious)) if pos_ious else None,
        "matched_mean_iou": (sum(matched_ious) / len(matched_ious))
                            if matched_ious else None,
        "n_matched": len(matched_ious),
        "positives_acc_at_iou": {
            t: sum(1 for i in pos_ious if i >= t) / len(pos_ious)
            for t in (0.1, 0.3, 0.5)} if pos_ious else None,
        "presence_recall": conf["tp"] / n_pos if n_pos else None,
        "presence_precision": (conf["tp"] / (conf["tp"] + conf["fp"])
                               if (conf["tp"] + conf["fp"]) else None),
    }


def all_channels_report(entries: list[dict]) -> dict:
    empty_ok = empty_n = 0
    nonempty_primaries: list[float] = []
    for e in entries:
        _, gold = gold_payload(e)
        if gold is None:
            continue
        if len(gold.get("anomalies") or []) == 0:
            empty_n += 1
            if float(e.get("primary", 0.0)) >= 1.0 - 1e-9:
                empty_ok += 1
        else:
            nonempty_primaries.append(float(e.get("primary", 0.0)))
    return {
        "n_empty_gold": empty_n,
        "empty_accuracy": empty_ok / empty_n if empty_n else None,
        "n_nonempty_gold": len(nonempty_primaries),
        "nonempty_mean_f1": (sum(nonempty_primaries) / len(nonempty_primaries))
                            if nonempty_primaries else None,
    }


# --------------------------------------------------------------------------
# bootstrap

def boot_ci(values: list[float], n=BOOT_N, seed=BOOT_SEED):
    if not values:
        return None
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(n))
    return {"mean": sum(values) / k,
            "ci95": [means[int(0.025 * n)], means[int(0.975 * n)]]}


def paired_boot(diffs: list[float], n=BOOT_N, seed=BOOT_SEED):
    if not diffs:
        return None
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=k)) / k for _ in range(n))
    lo, hi = means[int(0.025 * n)], means[int(0.975 * n)]
    return {"mean_diff": sum(diffs) / k, "ci95": [lo, hi],
            "significant": lo > 0 or hi < 0}


# --------------------------------------------------------------------------

def analyze(entries: list[dict], name: str) -> dict:
    by_cat: dict[str, list[dict]] = {}
    for e in entries:
        cat = e.get("category") or gold_payload(e)[0]
        by_cat.setdefault(cat, []).append(e)

    out = {"name": name, "n": len(entries), "categories": {}}
    strict_counts: dict[str, int] = {}
    for cat, rows in sorted(by_cat.items()):
        primaries = [float(r.get("primary", 0.0)) for r in rows]
        cat_rep = {"n": len(rows), "primary": boot_ci(primaries),
                   "null_baseline": NULL_BASELINES.get(cat)}
        viols: dict[str, int] = {}
        for r in rows:
            pred = first_json_object(r.get("predicted_answer")
                                     or r.get("prediction_raw") or "")
            for v in strict_violations(cat, pred):
                viols[v] = viols.get(v, 0) + 1
                strict_counts[v] = strict_counts.get(v, 0) + 1
        cat_rep["strict_violations"] = viols
        if cat == "localize":
            cat_rep["decomposition"] = localize_report(rows)
        elif cat == "localize_all_channels":
            cat_rep["decomposition"] = all_channels_report(rows)
        out["categories"][cat] = cat_rep
    out["strict_violations_total"] = strict_counts
    out["primary_overall"] = boot_ci(
        [float(e.get("primary", 0.0)) for e in entries])
    return out


def compare(a: list[dict], b: list[dict], name_a: str, name_b: str) -> dict:
    key = lambda e: (e.get("category") or gold_payload(e)[0],
                     e.get("ground_truth", ""))
    bmap: dict = {}
    for e in b:
        bmap.setdefault(key(e), []).append(e)
    diffs, by_cat = [], {}
    for e in a:
        matches = bmap.get(key(e))
        if not matches:
            continue
        other = matches.pop(0)
        d = float(e.get("primary", 0.0)) - float(other.get("primary", 0.0))
        diffs.append(d)
        by_cat.setdefault(key(e)[0], []).append(d)
    return {"paired_n": len(diffs),
            "overall": paired_boot(diffs),
            "per_category": {c: paired_boot(v) for c, v in sorted(by_cat.items())},
            "direction": f"{name_a} minus {name_b}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("--name-a", default="runA")
    ap.add_argument("--b")
    ap.add_argument("--name-b", default="runB")
    ap.add_argument("--out")
    args = ap.parse_args()

    a = load_entries(args.run_a)
    report = {"a": analyze(a, args.name_a)}
    if args.b:
        b = load_entries(args.b)
        report["b"] = analyze(b, args.name_b)
        report["paired"] = compare(a, b, args.name_a, args.name_b)
    txt = json.dumps(report, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"saved -> {args.out}")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
