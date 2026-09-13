#!/usr/bin/env python3
"""Fail-closed Z1-bis perceptual gate on the fixed 208 positive validation IDs.

The three crop conditions use the same checkpoint, sample, action text, truthful
header, and token layout.  Only the values encoded for the added crop differ.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from l3.grpo import _atomic_json_dump, _autocast_for, _write_jsonl, load_l3_config
from l3.paired_eval_l3 import bootstrap_mean
from l3.zoom import (
    _generate,
    _load_eval_runtime,
    _model_inputs,
    build_turn2_sample,
    build_zoom_block,
    channel_names,
    gold_events,
    parse_zoom_decision,
    series_length,
    with_action_instruction,
    zoom_header,
)
from l3.zoom_traces import random_gold_action


CONDITIONS = (
    "offcenter_true",
    "offcenter_swap",
    "offcenter_zero",
    "direct_new_format",
)


def crop_header_fingerprints(action: Any) -> dict[str, str]:
    """Fingerprint the complete textual coordinate disclosure for crop controls."""
    digest = hashlib.sha256(
        (action.text() + "\n" + zoom_header(action)).encode("utf-8")
    ).hexdigest()
    return {condition: digest for condition in CONDITIONS[:3]}


def _json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def fixed_reference_rows(path: Path, gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in _json_lines(path) if row.get("condition") == "true"]
    expected = int(gate["expected_ids"])
    if len(rows) != expected or len({str(row.get("id", "")) for row in rows}) != expected:
        raise ValueError(f"reference gate must contain {expected} unique true-condition IDs")
    ids = [str(row["id"]) for row in rows]
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()
    if digest != gate["expected_id_order_sha256"]:
        raise ValueError(f"reference ID/order SHA mismatch: {digest}")
    counts = Counter(str(row["category"]) for row in rows)
    expected_counts = {str(key): int(value) for key, value in gate["expected_categories"].items()}
    if dict(counts) != expected_counts:
        raise ValueError(f"reference category coverage mismatch: {dict(counts)}")
    return rows


def _overlaps(start: int, end: int, event: Mapping[str, Any]) -> bool:
    return min(end, int(event["end"])) > max(start, int(event["start"]))


def replacement_values(
    sample: Mapping[str, Any], action: Any, rng: random.Random,
) -> tuple[np.ndarray, int]:
    """Choose another same-channel, same-width window containing no gold event."""
    names = channel_names(sample)
    values = np.asarray(sample["time_series"][names.index(action.channel)], dtype=np.float32)
    width = action.window_len
    events = [event for event in gold_events(sample) if str(event["channel"]) == action.channel]
    maximum = values.size - width
    if maximum < 0:
        raise ValueError("replacement width exceeds channel length")
    candidates = {0, maximum}
    candidates.update(rng.randint(0, maximum) for _ in range(min(512, maximum + 1)))
    clean = [start for start in candidates
             if start != action.start
             and all(not _overlaps(start, start + width, event) for event in events)]
    if not clean:
        # Deterministic complete fallback search is diagnostic selection, not a
        # pseudo-recovery training trajectory.  It still fails if none exists.
        clean = [start for start in range(maximum + 1)
                 if start != action.start
                 and all(not _overlaps(start, start + width, event) for event in events)]
    if not clean:
        raise ValueError("no event-free replacement window exists")
    start = max(clean, key=lambda candidate: (abs(candidate - action.start), -candidate))
    return values[start:start + width].copy(), start


def _encode_original(model: Any, sample: Mapping[str, Any]) -> torch.Tensor:
    original = model.prepare_batch([sample], training=False)
    series = original["time_series"]
    with torch.no_grad(), _autocast_for(model):
        return model._encode_and_project(series).detach()


def run_crop(
    model: Any, sample: Mapping[str, Any], action: Any, values: np.ndarray,
    original_features: torch.Tensor, max_tokens: int,
) -> str:
    turn2 = build_turn2_sample(sample, action)
    prepared = model.prepare_batch([turn2], training=False)
    prepared.pop("labels", None)
    tensor = torch.as_tensor(
        np.asarray(values, dtype=np.float32), dtype=torch.float32,
        device=original_features.device,
    ).unsqueeze(0)
    with torch.no_grad(), _autocast_for(model):
        crop_features = model._encode_and_project(tensor).detach()
    prepared["_l3_ts_features"] = torch.cat((original_features, crop_features), dim=0)
    tokenizer = getattr(model, "text_tokenizer", None)
    if tokenizer is not None:
        pad_id = tokenizer(model.TS_PAD_TOKEN, add_special_tokens=False)["input_ids"][0]
        placeholders = int((prepared["input_ids"] == pad_id).sum().item())
        if placeholders != int(prepared["_l3_ts_features"].shape[0]):
            raise RuntimeError("perceptual feature/placeholder mismatch")
    ids = _generate(model, prepared, sample=False, max_new_tokens=max_tokens)
    return model.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()


def run_direct(model: Any, sample: Mapping[str, Any], max_tokens: int) -> tuple[str, str]:
    prompt = _model_inputs(model, with_action_instruction(sample))
    ids = _generate(model, prompt, sample=False, max_new_tokens=max_tokens)
    text = model.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()
    decision = parse_zoom_decision(text, channel_names(sample), series_length(sample))
    return text, decision.kind


def _paired_section(
    rows: Sequence[Mapping[str, Any]], category: str | None, seed: int, resamples: int,
) -> dict[str, Any]:
    selected = [row for row in rows if category is None or row["category"] == category]
    by_condition = {
        condition: [float(row["primary"]) for row in selected if row["condition"] == condition]
        for condition in CONDITIONS
    }
    counts = {condition: len(values) for condition, values in by_condition.items()}
    if len(set(counts.values())) != 1 or not next(iter(counts.values()), 0):
        raise ValueError(f"unpaired perceptual rows: {counts}")
    controls = ("offcenter_swap", "offcenter_zero")
    control = max(controls, key=lambda name: sum(by_condition[name]) / len(by_condition[name]))
    true = by_condition["offcenter_true"]
    differences = [a - b for a, b in zip(true, by_condition[control])]
    return {
        "n": len(true),
        "condition_means": {
            name: sum(values) / len(values) for name, values in by_condition.items()
        },
        "best_control": control,
        "true_minus_best_control": bootstrap_mean(
            differences, seed=seed, resamples=resamples,
        ),
    }


def perceptual_report(
    rows: Sequence[Mapping[str, Any]], gate: Mapping[str, Any], *, config_path: Path,
    checkpoint: Path, reference_rows: Path, seed_offset: int = 0,
) -> dict[str, Any]:
    seed = int(gate["seed"])
    resamples = int(gate["bootstrap_resamples"])
    overall = _paired_section(rows, None, seed, resamples)
    categories = {
        category: _paired_section(rows, category, seed + ordinal + 1, resamples)
        for ordinal, category in enumerate(sorted(gate["expected_categories"]))
    }
    delta = overall["true_minus_best_control"]
    threshold = float(gate["min_true_minus_best_control"])
    checks = {
        "effect_at_least_threshold": delta["mean"] >= threshold,
        "ci95_excludes_zero": delta["ci95"][0] > 0.0,
    }
    if not bool(gate.get("require_ci_low_above_zero", True)):
        checks["ci95_excludes_zero"] = True
    condition_counts = Counter(str(row["condition"]) for row in rows)
    expected = int(gate["expected_ids"])
    if condition_counts != Counter({condition: expected for condition in CONDITIONS}):
        raise ValueError(f"perceptual condition coverage mismatch: {dict(condition_counts)}")
    direct_decisions = Counter(
        str(row["direct_decision"]) for row in rows
        if row["condition"] == "direct_new_format"
    )
    return {
        "status": "DONE" if all(checks.values()) else "FAILED",
        "format": "l3_z1bis_perceptual_gate_v1",
        "gate_passed": all(checks.values()), "checks": checks,
        "criterion": {
            "min_true_minus_best_control": threshold,
            "bootstrap_resamples": resamples, "seed": seed,
        },
        "seed_offset": seed_offset,
        "overall": overall, "per_category": categories,
        "condition_counts": dict(condition_counts),
        "header_identity_verified_ids": expected,
        "direct_decisions": dict(direct_decisions),
        "inputs": {
            "config": str(config_path.resolve()), "checkpoint": str(checkpoint.resolve()),
            "reference_rows": str(reference_rows.resolve()),
        },
    }


def evaluate(args: argparse.Namespace) -> int:
    out = args.out_dir.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"perceptual-gate output already exists: {out}")
    out.mkdir(parents=True)
    try:
        config_path = args.config.expanduser().resolve()
        checkpoint = args.checkpoint.expanduser().resolve()
        reference_path = args.reference_rows.expanduser().resolve()
        config = load_l3_config(config_path)
        sft = config["zoom_sft"]
        gate = config["perceptual_gate"]
        references = fixed_reference_rows(reference_path, gate)
        model, dataset, _ = _load_eval_runtime(str(config_path), str(checkpoint), "val")
        model.eval()
        by_id: dict[str, int] = {}
        for index in range(len(dataset)):
            sample_id = str(dataset[index].get("id", ""))
            if sample_id in by_id:
                raise ValueError(f"duplicate validation ID: {sample_id}")
            by_id[sample_id] = index
        missing = [str(row["id"]) for row in references if str(row["id"]) not in by_id]
        if missing:
            raise ValueError(f"fixed perceptual IDs missing from validation: {missing[:3]}")

        written: list[dict[str, Any]] = []
        rows_path = out / "rows.jsonl"
        with rows_path.open("x", encoding="utf-8") as handle:
            for ordinal, reference in enumerate(references):
                sample = dataset[by_id[str(reference["id"])]]
                if str(sample["category"]) != str(reference["category"]):
                    raise ValueError("fixed-ID category drift")
                if (reference.get("ground_truth") is not None
                        and str(reference["ground_truth"]) != str(dataset.get_ground_truth(sample))):
                    raise ValueError("fixed-ID ground-truth drift")
                if not gold_events(sample):
                    raise ValueError("perceptual gate contains a non-positive row")
                # `--seed-offset` estrae un crop DIVERSO per lo stesso ID, a parita di
                # tutto il resto: serve ad ampliare la misura appaiata (piu' crop
                # per ID) quando un singolo crop lascia l'intervallo inconcludente.
                rng = random.Random(int(gate["seed"]) + int(args.seed_offset)
                                    + ordinal * 104729)
                action, event, position = random_gold_action(sample, sft, rng)
                radius = float(sft["position"]["centered_radius"])
                if abs(position - 0.5) <= radius:
                    # The gate is explicitly off-centre. Draw uniformly again from
                    # the valid starts until outside the declared centred band.
                    for _ in range(1000):
                        action, event, position = random_gold_action(sample, sft, rng)
                        if abs(position - 0.5) > radius:
                            break
                    else:
                        raise ValueError("could not construct an off-centre crop")
                header_hashes = crop_header_fingerprints(action)
                if len(set(header_hashes.values())) != 1:
                    raise RuntimeError("crop headers are not identical")
                header_sha = header_hashes["offcenter_true"]
                _, true_values = build_zoom_block(sample, action)
                swap_values, replacement_start = replacement_values(sample, action, rng)
                zero_values = np.zeros(action.window_len, dtype=np.float32)
                original_features = _encode_original(model, sample)
                predictions = {
                    "offcenter_true": run_crop(
                        model, sample, action, true_values, original_features, args.max_tokens,
                    ),
                    "offcenter_swap": run_crop(
                        model, sample, action, swap_values, original_features, args.max_tokens,
                    ),
                    "offcenter_zero": run_crop(
                        model, sample, action, zero_values, original_features, args.max_tokens,
                    ),
                }
                direct_text, direct_kind = run_direct(model, sample, args.max_tokens)
                predictions["direct_new_format"] = direct_text
                for condition in CONDITIONS:
                    prediction = predictions[condition]
                    result = dataset.evaluate_answer(prediction, sample)
                    primary = float(result[dataset.primary_metric])
                    if not math.isfinite(primary):
                        raise ValueError("non-finite perceptual score")
                    row = {
                        "index": by_id[str(reference["id"])], "id": str(reference["id"]),
                        "category": str(sample["category"]),
                        "ground_truth": dataset.get_ground_truth(sample),
                        "condition": condition, "prediction": prediction, "primary": primary,
                        "zoom": None if condition == "direct_new_format" else {
                            "channel": action.channel, "start": action.start, "end": action.end,
                            "window_len": action.window_len, "gold_position": position,
                            "gold_start": int(event["start"]), "gold_end": int(event["end"]),
                        },
                        "header_sha256": None if condition == "direct_new_format" else header_sha,
                        "seed_offset": int(args.seed_offset),
                        "replacement_start": replacement_start if condition == "offcenter_swap" else None,
                        "direct_decision": direct_kind if condition == "direct_new_format" else None,
                    }
                    _write_jsonl(handle, row)
                    written.append(row)
                del original_features
                print(f"Z1BIS_PERCEPTUAL {ordinal + 1}/{len(references)}", flush=True)
        for sample_id in {row["id"] for row in written}:
            crop_rows = [row for row in written
                         if row["id"] == sample_id and row["condition"] != "direct_new_format"]
            if len(crop_rows) != 3 or len({row["header_sha256"] for row in crop_rows}) != 1:
                raise RuntimeError(f"header identity failed for {sample_id}")
        report = perceptual_report(
            written, gate, config_path=config_path, checkpoint=checkpoint,
            reference_rows=reference_path, seed_offset=int(args.seed_offset),
        )
        _atomic_json_dump(out / "summary.json", report)
        if not report["gate_passed"]:
            raise RuntimeError("Z1-bis perceptual criterion failed")
        _atomic_json_dump(out / "DONE", report)
        print(f"L3_Z1BIS_PERCEPTUAL_OK delta={report['overall']['true_minus_best_control']['mean']:+.6f}")
        return 0
    except Exception as exc:
        failed = out / "FAILED"
        if not failed.exists():
            _atomic_json_dump(failed, {
                "status": "FAILED", "error_type": type(exc).__name__, "reason": str(exc),
            })
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference-rows", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="sposta l'estrazione del crop mantenendo tutto il resto")
    return parser


if __name__ == "__main__":
    raise SystemExit(evaluate(build_parser().parse_args()))
