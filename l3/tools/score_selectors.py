#!/usr/bin/env python3
"""Confronto dei selettori a punteggio, a pesi congelati e budget identico.

La fase CPU (`compare_selectors.py`) misura *dove* guardano i selettori. Qui si
misura *quanto valgono*: stesso checkpoint, stesse righe, stessa ampiezza di
finestra, header veritiero sulle coordinate effettive, contenuto vero. Cambia solo
il criterio con cui la finestra viene scelta.

Bracci disponibili (`--selectors`):
  wavelet_matched            energia Haar su tutti i canali, ampiezza della policy
  wavelet_in_policy_channel  canale scelto dalla policy, posizione dell'energia Haar
  oracle_matched             finestra centrata sul gold, stessa ampiezza (limite superiore)

Il braccio della policy non viene rigenerato: si appaia contro la condizione
`self_true` del gate del contenuto, che usa la stessa meccanica di turno 2 sulla
finestra emessa dalla policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from l3.grpo import _atomic_json_dump, _write_jsonl, load_l3_config
from l3.paired_eval_l3 import bootstrap_mean
from l3.tools.compare_selectors import (
    band_of,
    channel_details,
    geometry,
    oracle_at_width,
    wavelet_zoom_at_width,
    wavelet_zoom_in_channel,
)
from l3.tools.diag_crop_content import _encode_original, run_crop
from l3.zoom import ZoomAction, _load_eval_runtime, build_zoom_block, gold_events, series_length

SELECTORS = ("wavelet_matched", "wavelet_in_policy_channel", "oracle_matched")


def build_action(name: str, sample: Mapping[str, Any], policy: ZoomAction,
                 details: Sequence[Mapping[int, Any]]) -> ZoomAction | None:
    width = policy.window_len
    if name == "wavelet_matched":
        return wavelet_zoom_at_width(sample, width, details=details)
    if name == "wavelet_in_policy_channel":
        return wavelet_zoom_in_channel(sample, policy.channel, width, details=details)
    if name == "oracle_matched":
        return oracle_at_width(sample, width)
    raise ValueError(f"unknown selector: {name}")


def baseline_scores(path: Path | None, condition: str) -> dict[str, float]:
    if path is None:
        return {}
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return {str(r["id"]): float(r["primary"]) for r in rows
            if str(r["condition"]) == condition}


def section(arm: Mapping[str, float], baseline: Mapping[str, float], *, seed: int,
            resamples: int) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(arm), "mean": statistics.fmean(arm.values()) if arm else None}
    shared = sorted(set(arm) & set(baseline))
    if shared:
        out["paired"] = {
            "n": len(shared),
            "arm_mean": statistics.fmean([arm[i] for i in shared]),
            "baseline_mean": statistics.fmean([baseline[i] for i in shared]),
            "delta_vs_policy": bootstrap_mean(
                [arm[i] - baseline[i] for i in shared], seed=seed, resamples=resamples),
        }
    return out


def report(rows: Sequence[Mapping[str, Any]], baseline: Mapping[str, float], *,
           selectors: Sequence[str], seed: int, resamples: int, config_path: Path,
           checkpoint: Path, policy_rows: Path,
           baseline_rows: Path | None) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for ordinal, name in enumerate(selectors):
        picked = {str(r["id"]): float(r["primary"]) for r in rows if r["selector"] == name}
        entry = section(picked, baseline, seed=seed + ordinal, resamples=resamples)
        states = Counter(str(r["geometry"]["state"]) for r in rows if r["selector"] == name)
        scored = sum(states.values())
        entry["contained_rate"] = states["contained"] / scored if scored else None
        entry["window_states"] = dict(states)
        by_state = {
            state: [float(r["primary"]) for r in rows
                    if r["selector"] == name and r["geometry"]["state"] == state]
            for state in states
        }
        entry["mean_by_state"] = {state: statistics.fmean(values)
                                  for state, values in by_state.items() if values}
        entry["per_category"] = {}
        for index, category in enumerate(sorted({str(r["category"]) for r in rows})):
            subset = {str(r["id"]): float(r["primary"]) for r in rows
                      if r["selector"] == name and str(r["category"]) == category}
            entry["per_category"][category] = section(
                subset, baseline, seed=seed + 100 + 10 * ordinal + index, resamples=resamples)
        entry["per_band"] = {}
        for index, band in enumerate(sorted({int(r["band"]) for r in rows})):
            subset = {str(r["id"]): float(r["primary"]) for r in rows
                      if r["selector"] == name and int(r["band"]) == band}
            entry["per_band"][str(band)] = section(
                subset, baseline, seed=seed + 300 + 10 * ordinal + index, resamples=resamples)
        arms[name] = entry
    return {
        "status": "DONE",
        "format": "l3_selector_scores_v1",
        "selectors": list(selectors),
        "rows_per_selector": {name: arms[name]["n"] for name in selectors},
        "baseline": {
            "condition": "self_true",
            "rows": len(baseline),
            "mean": statistics.fmean(baseline.values()) if baseline else None,
            "source": str(baseline_rows.resolve()) if baseline_rows else None,
        },
        "arms": arms,
        "criterion": {"kind": "descriptive, paired bootstrap; no pass/fail",
                      "bootstrap_resamples": resamples, "seed": seed},
        "inputs": {"config": str(config_path.resolve()),
                   "checkpoint": str(checkpoint.resolve()),
                   "policy_rows": str(policy_rows.resolve())},
    }


def evaluate(args: argparse.Namespace) -> int:
    out = args.out_dir.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"selector-score output already exists: {out}")
    out.mkdir(parents=True)
    try:
        config_path = args.config.expanduser().resolve()
        checkpoint = args.checkpoint.expanduser().resolve()
        policy_path = args.policy_rows.expanduser().resolve()
        baseline_path = (args.baseline_rows.expanduser().resolve()
                         if args.baseline_rows else None)
        selectors = [name.strip() for name in args.selectors.split(",") if name.strip()]
        unknown = [name for name in selectors if name not in SELECTORS]
        if unknown:
            raise ValueError(f"unknown selectors: {unknown}")
        load_l3_config(config_path)
        baseline = baseline_scores(baseline_path, "self_true")

        policy_rows = [json.loads(line) for line in policy_path.open(encoding="utf-8")
                       if line.strip()]
        acted = [row for row in policy_rows if row.get("zoom")]
        if not acted:
            raise ValueError("no acted policy rows")

        model, dataset, _ = _load_eval_runtime(str(config_path), str(checkpoint), "val")
        model.eval()
        by_id: dict[str, int] = {}
        for index in range(len(dataset)):
            sample_id = str(dataset[index].get("id", ""))
            if sample_id in by_id:
                raise ValueError(f"duplicate validation ID: {sample_id}")
            by_id[sample_id] = index

        written: list[dict[str, Any]] = []
        with (out / "rows.jsonl").open("x", encoding="utf-8") as handle:
            for ordinal, policy_row in enumerate(acted):
                sample_id = str(policy_row["id"])
                if sample_id not in by_id:
                    raise ValueError(f"policy ID missing from validation: {sample_id}")
                sample = dataset[by_id[sample_id]]
                if not gold_events(sample):
                    continue
                length = series_length(sample)
                zoom = policy_row["zoom"]
                policy_action = ZoomAction(str(zoom["channel"]), int(zoom["start"]),
                                           int(zoom["end"]), length)
                details = channel_details(sample)
                original_features = _encode_original(model, sample)
                for name in selectors:
                    action = build_action(name, sample, policy_action, details)
                    if action is None:
                        raise ValueError(f"{name} produced no window on {sample_id}")
                    _, values = build_zoom_block(sample, action)
                    prediction = run_crop(model, sample, action, values,
                                          original_features, args.max_tokens)
                    result = dataset.evaluate_answer(prediction, sample)
                    primary = float(result[dataset.primary_metric])
                    if not math.isfinite(primary):
                        raise ValueError("non-finite selector score")
                    row = {
                        "index": by_id[sample_id], "id": sample_id,
                        "category": str(sample["category"]), "length": length,
                        "band": band_of(length),
                        "ground_truth": dataset.get_ground_truth(sample),
                        "selector": name, "prediction": prediction, "primary": primary,
                        "policy_primary": float(policy_row["primary"]),
                        "zoom": {"channel": action.channel, "start": action.start,
                                 "end": action.end, "window_len": action.window_len},
                        "policy_zoom": {"channel": policy_action.channel,
                                        "start": policy_action.start,
                                        "end": policy_action.end},
                        "geometry": geometry(sample, action),
                    }
                    _write_jsonl(handle, row)
                    written.append(row)
                del original_features, details
                print(f"SELECTOR_SCORES {ordinal + 1}/{len(acted)}", flush=True)

        summary = report(written, baseline, selectors=selectors, seed=int(args.seed),
                         resamples=int(args.bootstrap), config_path=config_path,
                         checkpoint=checkpoint, policy_rows=policy_path,
                         baseline_rows=baseline_path)
        _atomic_json_dump(out / "summary.json", summary)
        base = summary["baseline"]["mean"]
        print(f"\npolicy (self_true) media={base}")
        for name in selectors:
            arm = summary["arms"][name]
            line = f"{name:28} media={arm['mean']:.4f} contenuto={100.0 * (arm['contained_rate'] or 0):.1f}%"
            if "paired" in arm:
                delta = arm["paired"]["delta_vs_policy"]
                line += (f" delta={delta['mean']:+.4f} "
                         f"CI=[{delta['ci95'][0]:+.4f}; {delta['ci95'][1]:+.4f}] "
                         f"n={arm['paired']['n']}")
            print(line)
        _atomic_json_dump(out / "DONE", {"status": "DONE", "rows": len(written)})
        print("L3_SELECTOR_SCORES_OK")
        return 0
    except Exception as exc:
        failed = out / "FAILED"
        if not failed.exists():
            _atomic_json_dump(failed, {"status": "FAILED", "error_type": type(exc).__name__,
                                       "reason": str(exc)})
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--policy-rows", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--baseline-rows", type=Path,
                        help="rows.jsonl del gate del contenuto (condizione self_true)")
    parser.add_argument("--selectors", default=",".join(SELECTORS))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--max-tokens", type=int, default=500)
    return parser


if __name__ == "__main__":
    raise SystemExit(evaluate(build_parser().parse_args()))
