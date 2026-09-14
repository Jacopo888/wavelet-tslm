#!/usr/bin/env python3
"""Verdetto di W2 secondo i criteri dichiarati in PROTOCOL-L3.md §W2.

Due obiettivi separati, entrambi necessari perché W2 sostituisca W1:
  1. lettura recuperata sui crop identici — Δ(W2−W1) sul multicanale > 0 con CI>0
     **e** Δ(W2−Z1bis) aggregato ≥ −0,03 con estremo inferiore del CI > −0,05;
  2. puntamento e operativo preservati — primary ≥ 0,3451, contenimento ≥ 57,5 %,
     gate operativo superato.

Solo CPU: legge i JSON già prodotti. Non decide nulla che non sia dichiarato.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Mapping

MULTICHANNEL = "localize_all_channels"


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vs-w1", required=True, type=Path)
    parser.add_argument("--vs-z1bis", required=True, type=Path)
    parser.add_argument("--policy-json", required=True, type=Path)
    parser.add_argument("--policy-rows", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-primary", type=float, default=0.3451)
    parser.add_argument("--min-contained-rate", type=float, default=0.575)
    parser.add_argument("--min-aggregate-reading", type=float, default=-0.03)
    parser.add_argument("--reading-ci-floor", type=float, default=-0.05)
    parser.add_argument("--mode", choices=("w2", "w3"), default="w2",
                        help="w2: il multicanale deve migliorare sulla partenza; "
                             "w3: nessun calo rilevante in aggregato sulla partenza")
    parser.add_argument("--start-primary", type=float, default=None,
                        help="w3: primary val520 della partenza, per l'arm di punta")
    parser.add_argument("--min-vs-z1bis", type=float, default=0.03,
                        help="w3: guadagno minimo di lettura aggregato contro Z1-bis")
    args = parser.parse_args()

    vs_w1 = load(args.vs_w1)
    vs_z1bis = load(args.vs_z1bis)
    geometry = load(args.geometry)
    gate = load(args.gate) if args.gate and args.gate.exists() else None

    multichannel = vs_w1["per_category"][MULTICHANNEL]["reading"]
    aggregate = vs_z1bis["reading"]
    rows = [json.loads(line) for line in args.policy_rows.open(encoding="utf-8")
            if line.strip()]
    primary = statistics.fmean(float(row["primary"]) for row in rows)
    contained = float(geometry["overall"]["policy"]["contained_rate"])

    if args.mode == "w2":
        reading_checks = {
            "multichannel_improves_over_w1": (multichannel["mean"] > 0.0
                                              and multichannel["ci95"][0] > 0.0),
            "aggregate_not_a_relevant_decline": (
                aggregate["mean"] >= args.min_aggregate_reading
                and aggregate["ci95"][0] > args.reading_ci_floor),
        }
    else:
        # W3: la partenza e' gia' un lettore buono; si chiede di non peggiorarlo in
        # aggregato (nessun calo rilevante) e di conservare il vantaggio su Z1-bis.
        start_aggregate = vs_w1["reading"]
        reading_checks = {
            "aggregate_no_relevant_decline_vs_start": (
                start_aggregate["mean"] >= args.min_aggregate_reading
                and start_aggregate["ci95"][0] > args.reading_ci_floor),
            "aggregate_gain_vs_z1bis_kept": (
                aggregate["mean"] >= float(args.min_vs_z1bis)
                and aggregate["ci95"][0] > 0.0),
        }
    operational_checks = {
        "primary_at_least_threshold": primary >= args.min_primary,
        "contained_rate_at_least_threshold": contained >= args.min_contained_rate,
        "operational_gate_passed": bool(gate and gate.get("gate_passed")),
    }
    reading_ok = all(reading_checks.values())
    operational_ok = all(operational_checks.values())
    headline = None
    if args.mode == "w3" and args.start_primary is not None:
        # Arm di punta per il test: il nuovo checkpoint solo se passa i gate E non
        # e' sotto la partenza in operativo; altrimenti la partenza.
        headline = ("new" if reading_ok and operational_ok
                    and primary >= float(args.start_primary) else "start")
    if reading_ok and operational_ok:
        verdict = "w2_becomes_candidate"
        note = ("entrambi gli obiettivi soddisfatti: W2 diventa il candidato e il "
                "pilot GRPO parte da lui")
    elif reading_ok:
        verdict = "keep_w1_reading_only"
        note = ("lettura recuperata ma operativo peggiorato: si tiene W1, il recupero "
                "resta informativo")
    else:
        verdict = "keep_w1_grpo_with_constraint"
        note = ("lettura non recuperata: si tiene W1 e il pilot GRPO resta possibile "
                "con vincolo esplicito sulla lettura multicanale")

    report = {
        "format": "l3_w2_verdict_v1",
        "mode": args.mode,
        "verdict": verdict,
        "headline_arm": headline,
        "note": note,
        "reading": {
            "multichannel_vs_w1": multichannel,
            "aggregate_vs_z1bis": aggregate,
            "per_category_vs_w1": {name: block["reading"] for name, block
                                   in vs_w1["per_category"].items()},
            "content_vs_z1bis_reference": vs_z1bis.get("content"),
            "checks": reading_checks,
        },
        "operational": {
            "primary": primary,
            "contained_rate": contained,
            "gate_passed": bool(gate and gate.get("gate_passed")),
            "gate_failed_checks": ([name for name, value in gate["checks"].items()
                                    if not value] if gate else None),
            "checks": operational_checks,
        },
        "criteria": {
            "min_primary": args.min_primary,
            "min_contained_rate": args.min_contained_rate,
            "min_aggregate_reading": args.min_aggregate_reading,
            "reading_ci_floor": args.reading_ci_floor,
            "declared_in": "l3/PROTOCOL-L3.md, W2 recupero mirato",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"lettura multicanale W2-W1: {multichannel['mean']:+.4f} "
          f"CI[{multichannel['ci95'][0]:+.4f}; {multichannel['ci95'][1]:+.4f}]")
    print(f"lettura aggregata W2-Z1bis: {aggregate['mean']:+.4f} "
          f"CI[{aggregate['ci95'][0]:+.4f}; {aggregate['ci95'][1]:+.4f}]")
    print(f"val520 primary {primary:.4f} | contenimento {100 * contained:.1f} % | "
          f"gate {'superato' if report['operational']['gate_passed'] else 'non superato'}")
    print(f"L3_W2_VERDICT={verdict}\n  {note}")
    if headline is not None:
        print(f"L3_HEADLINE_ARM={headline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
