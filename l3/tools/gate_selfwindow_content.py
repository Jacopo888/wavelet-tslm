#!/usr/bin/env python3
"""Controllo del contenuto sulle finestre scelte dalla policy (Codex punto 2).

Il gate percettivo di Z1-bis ha usato finestre dell'oracle: canale e regione erano
informazione privilegiata. Questo strumento ripete lo stesso controllo *sulle azioni
che la policy ha emesso da sola*, congelate: stesso checkpoint, stessa azione, stesso
header, stesso numero di token; cambia soltanto il contenuto del crop.

Sotto-ipotesi registrata prima del run: il vantaggio del contenuto vero deve
concentrarsi sulle righe in cui la finestra *contiene* l'evento gold; dove la finestra
manca l'evento, il contenuto non puo' aiutare.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from l3.grpo import _atomic_json_dump, _write_jsonl, load_l3_config
from l3.paired_eval_l3 import bootstrap_mean
from l3.tools.compare_selectors import geometry
from l3.tools.diag_crop_content import (
    _encode_original,
    crop_header_fingerprints,
    replacement_values,
    run_crop,
)
from l3.zoom import (
    ZoomAction,
    _load_eval_runtime,
    build_zoom_block,
    gold_events,
    series_length,
)

CONDITIONS = ("self_true", "self_swap", "self_zero")


def paired_section(rows: Sequence[Mapping[str, Any]], seed: int, resamples: int,
                   subset: Sequence[str] | None = None) -> dict[str, Any]:
    """Confronto appaiato fra contenuto vero e miglior controllo sugli stessi ID."""
    selected = [r for r in rows if subset is None or r["id"] in set(subset)]
    by_condition = {
        condition: {r["id"]: float(r["primary"]) for r in selected
                    if r["condition"] == condition}
        for condition in CONDITIONS
    }
    ids = sorted(set.intersection(*(set(v) for v in by_condition.values()))
                 ) if by_condition else []
    if not ids:
        return {"n": 0}
    counts = {c: len(v) for c, v in by_condition.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"unpaired content rows: {counts}")
    means = {c: statistics.fmean([v[i] for i in ids]) for c, v in by_condition.items()}
    control = max(("self_swap", "self_zero"), key=lambda name: means[name])
    differences = [by_condition["self_true"][i] - by_condition[control][i] for i in ids]
    return {
        "n": len(ids),
        "condition_means": means,
        "best_control": control,
        "true_minus_best_control": bootstrap_mean(differences, seed=seed, resamples=resamples),
    }


def report(rows: Sequence[Mapping[str, Any]], *, seed: int, resamples: int,
           min_delta: float, config_path: Path, checkpoint: Path,
           policy_rows: Path) -> dict[str, Any]:
    true_rows = [r for r in rows if r["condition"] == "self_true"]
    contained = [r["id"] for r in true_rows if r["geometry"]["state"] == "contained"]
    missed = [r["id"] for r in true_rows if r["geometry"]["state"] != "contained"]
    overall = paired_section(rows, seed, resamples)
    sections = {
        "contained": paired_section(rows, seed + 1, resamples, contained),
        "not_contained": paired_section(rows, seed + 2, resamples, missed),
    }
    for ordinal, category in enumerate(sorted({r["category"] for r in true_rows})):
        ids = [r["id"] for r in true_rows if r["category"] == category]
        sections[f"category:{category}"] = paired_section(
            rows, seed + 10 + ordinal, resamples, ids)
    for ordinal, band in enumerate(sorted({r["band"] for r in true_rows})):
        ids = [r["id"] for r in true_rows if r["band"] == band]
        sections[f"band:{band}"] = paired_section(rows, seed + 40 + ordinal, resamples, ids)

    delta = overall["true_minus_best_control"]
    checks = {
        "effect_at_least_threshold": delta["mean"] >= min_delta,
        "ci95_excludes_zero": delta["ci95"][0] > 0.0,
    }
    sub = {
        "contained_delta": sections["contained"].get("true_minus_best_control", {}).get("mean"),
        "not_contained_delta": sections["not_contained"].get(
            "true_minus_best_control", {}).get("mean"),
    }
    if sub["contained_delta"] is not None and sub["not_contained_delta"] is not None:
        sub["contained_exceeds_not_contained"] = (
            sub["contained_delta"] > sub["not_contained_delta"])
    counts = Counter(str(r["condition"]) for r in rows)
    if len(set(counts.values())) != 1:
        raise ValueError(f"condition coverage mismatch: {dict(counts)}")
    states = Counter(str(r["geometry"]["state"]) for r in true_rows)
    return {
        "status": "DONE" if all(checks.values()) else "FAILED",
        "format": "l3_selfwindow_content_gate_v1",
        "gate_passed": all(checks.values()),
        "checks": checks,
        "criterion": {"min_true_minus_best_control": min_delta,
                      "bootstrap_resamples": resamples, "seed": seed},
        "registered_subhypothesis": sub,
        "overall": overall,
        "sections": sections,
        "window_states": dict(states),
        "condition_counts": dict(counts),
        "action_text_source": "canonical ZoomAction.text() reconstruction",
        "inputs": {"config": str(config_path.resolve()),
                   "checkpoint": str(checkpoint.resolve()),
                   "policy_rows": str(policy_rows.resolve())},
    }


def evaluate(args: argparse.Namespace) -> int:
    out = args.out_dir.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"self-window gate output already exists: {out}")
    out.mkdir(parents=True)
    try:
        config_path = args.config.expanduser().resolve()
        checkpoint = args.checkpoint.expanduser().resolve()
        policy_path = args.policy_rows.expanduser().resolve()
        load_l3_config(config_path)
        policy = [json.loads(line) for line in policy_path.open(encoding="utf-8")
                  if line.strip()]
        acted = [row for row in policy if row.get("zoom")]
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
        skipped: dict[str, int] = defaultdict(int)
        rows_path = out / "rows.jsonl"
        with rows_path.open("x", encoding="utf-8") as handle:
            for ordinal, policy_row in enumerate(acted):
                sample_id = str(policy_row["id"])
                if sample_id not in by_id:
                    raise ValueError(f"policy ID missing from validation: {sample_id}")
                sample = dataset[by_id[sample_id]]
                if str(sample["category"]) != str(policy_row["category"]):
                    raise ValueError(f"category drift on {sample_id}")
                if not gold_events(sample):
                    skipped["negative_row"] += 1
                    continue
                length = series_length(sample)
                zoom = policy_row["zoom"]
                action = ZoomAction(str(zoom["channel"]), int(zoom["start"]),
                                    int(zoom["end"]), length)
                geo = geometry(sample, action)
                header_hashes = crop_header_fingerprints(action)
                if len(set(header_hashes.values())) != 1:
                    raise RuntimeError("crop headers are not identical")
                header_sha = next(iter(header_hashes.values()))
                rng = random.Random(int(args.seed) + ordinal * 104729)
                _, true_values = build_zoom_block(sample, action)
                try:
                    swap_values, replacement_start = replacement_values(sample, action, rng)
                except ValueError:
                    skipped["no_event_free_replacement"] += 1
                    continue
                zero_values = np.zeros(action.window_len, dtype=np.float32)
                original_features = _encode_original(model, sample)
                predictions = {
                    "self_true": run_crop(model, sample, action, true_values,
                                          original_features, args.max_tokens),
                    "self_swap": run_crop(model, sample, action, swap_values,
                                          original_features, args.max_tokens),
                    "self_zero": run_crop(model, sample, action, zero_values,
                                          original_features, args.max_tokens),
                }
                for condition in CONDITIONS:
                    prediction = predictions[condition]
                    result = dataset.evaluate_answer(prediction, sample)
                    primary = float(result[dataset.primary_metric])
                    if not math.isfinite(primary):
                        raise ValueError("non-finite content score")
                    row = {
                        "index": by_id[sample_id], "id": sample_id,
                        "category": str(sample["category"]), "length": length,
                        "band": int(length),
                        "ground_truth": dataset.get_ground_truth(sample),
                        "condition": condition, "prediction": prediction,
                        "primary": primary,
                        "policy_primary": float(policy_row["primary"]),
                        "zoom": {"channel": action.channel, "start": action.start,
                                 "end": action.end, "window_len": action.window_len},
                        "geometry": geo, "header_sha256": header_sha,
                        "replacement_start": (replacement_start
                                              if condition == "self_swap" else None),
                    }
                    _write_jsonl(handle, row)
                    written.append(row)
                del original_features
                print(f"SELFWINDOW {ordinal + 1}/{len(acted)}", flush=True)

        for sample_id in {row["id"] for row in written}:
            group = [row for row in written if row["id"] == sample_id]
            if len(group) != len(CONDITIONS) or len({r["header_sha256"] for r in group}) != 1:
                raise RuntimeError(f"header identity failed for {sample_id}")
        summary = report(written, seed=int(args.seed), resamples=int(args.bootstrap),
                         min_delta=float(args.min_delta), config_path=config_path,
                         checkpoint=checkpoint, policy_rows=policy_path)
        summary["skipped"] = dict(skipped)
        _atomic_json_dump(out / "summary.json", summary)
        delta = summary["overall"]["true_minus_best_control"]
        print(f"delta={delta['mean']:+.6f} CI95=[{delta['ci95'][0]:+.6f}; "
              f"{delta['ci95'][1]:+.6f}] n={summary['overall']['n']}")
        print(f"contenute: {summary['registered_subhypothesis']['contained_delta']}")
        print(f"non contenute: {summary['registered_subhypothesis']['not_contained_delta']}")
        if not summary["gate_passed"]:
            raise RuntimeError("self-window content criterion failed")
        _atomic_json_dump(out / "DONE", summary)
        print("L3_SELFWINDOW_OK")
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
    parser.add_argument("--min-delta", type=float, default=0.03)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--max-tokens", type=int, default=500)
    return parser


if __name__ == "__main__":
    raise SystemExit(evaluate(build_parser().parse_args()))
