#!/usr/bin/env python3
"""Oracle-text probe and content-dependence gate for a frozen checkpoint.

Five conditions share one decoding budget and one stratified validation subset:

  real         real soft tokens, no oracle text
  empty        soft tokens zeroed, no oracle text
  substituted  soft tokens of a compatible donor recording, no oracle text
  oracle       soft tokens zeroed, the TRUE reading supplied as ``Reading:`` text in the user turn
  permuted     soft tokens zeroed, the reading of another recording supplied the same way

When the checkpoint's format has a reading segment, each condition first GENERATES its own reading inside
``<think>`` (Astra 16/09: no teacher reading is ever placed in the assistant turn, otherwise every condition
would already hold the bin information as text) and those reasoning ids are then replayed verbatim -- closure
included -- in front of the gold final answer.  A checkpoint trained without a reading segment, such as va1r,
keeps its own format and answers directly; ``--thinking`` follows the spec unless overridden.  In both cases
the cross-entropy is measured only on supervised final-answer tokens that contain a digit.

Two questions are answered at once.  The oracle question, pre-registered by Astra: does the backbone use bin
information when it arrives as text?  Verdict POSITIVE only when both paired CI95 of ``CE_control - CE_oracle``
are strictly positive and the oracle primary beats both zero-token controls; a CI crossing zero is
INCONCLUSIVE, not NEGATIVE.  The content question, comparable with the +0.135 of Z1-bis: does the primary drop
when the real tokens are replaced by a donor's real tokens?  That block is descriptive and does not gate.

Usage:
  python tools/oracle_text_probe.py --config <toml> --checkpoint <dir> --out <json> [--rows 60]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safetensors.torch import load_file  # noqa: E402

from wavelet_rope.backbone import load_backbone  # noqa: E402
from wavelet_rope.checkpoint import inspect_foreign_checkpoint  # noqa: E402
from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import AnomalyXLSplit, stratified_subset  # noqa: E402
from wavelet_rope.model import WRAgentModel  # noqa: E402
from wavelet_rope.scoring import REQUIRED_KEYS, parse_structured_pred, score_precise  # noqa: E402
from wavelet_rope.templates import IGNORE_INDEX, SEG_FINAL, PreTokenized  # noqa: E402

CONDITIONS = ("real", "empty", "substituted", "oracle", "permuted")
# Categories whose answer cannot be produced from the header alone.  The prompt header states median, scale,
# min and max per channel, and "max minus median over scale" reproduces measure_magnitude's magnitude_sigma
# with a median ratio of 1.16 on 120 validation rows -- so the overall primary is not a perception measure.
# These are the three categories Z1-bis gated on.
SIGNAL_DEPENDENT = ("localize", "classify_with_evidence", "localize_all_channels")
ZEROED_CONDITIONS = frozenset({"empty", "oracle", "permuted"})
BOOTSTRAP_RESAMPLES = 10_000
_DIGIT = re.compile(r"\d")


def derangement_offset(n: int, seed: int) -> int:
    """Choose the fixed cyclic shift used for the permuted oracle text."""

    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("a derangement needs at least two rows")
    return random.Random(int(seed)).randrange(1, n)


def derangement_indices(n: int, seed: int) -> list[int]:
    """Return the deterministic map ``i -> (i + k) % n``."""

    k = derangement_offset(n, seed)
    return [(index + k) % n for index in range(n)]


def final_field_positions(sequence, tokenizer) -> torch.Tensor:
    """Return supervised final-answer label positions whose token contains a digit."""

    ids = sequence.input_ids.tolist()
    labels = sequence.labels.tolist()
    segments = sequence.segment.tolist()
    positions = [
        index
        for index in range(1, len(ids))
        if labels[index] != IGNORE_INDEX
        and segments[index] == SEG_FINAL
        and _DIGIT.search(tokenizer.decode([ids[index]]))
    ]
    return torch.tensor(positions, dtype=torch.long)


def final_field_logit_positions(sequence, tokenizer) -> torch.Tensor:
    """Return causal-logit positions for :func:`final_field_positions`."""

    return final_field_positions(sequence, tokenizer) - 1


def paired_bootstrap(values: Sequence[float], seed: int, n: int = BOOTSTRAP_RESAMPLES) -> dict:
    """Bootstrap the mean of already paired per-recording differences."""

    if n <= 0:
        raise ValueError("the number of bootstrap resamples must be positive")
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "ci95_lower": None, "ci95_upper": None, "n": 0, "resamples": int(n)}
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, arr.size, size=(int(n), arr.size))
    means = arr[indices].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
        "n": int(arr.size),
        "resamples": int(n),
    }


def _fmt_stat(stat: dict) -> str:
    if stat.get("mean") is None:
        return "n/a"
    return f"{stat['mean']:+.4f} CI95 [{stat['ci95_lower']:+.4f}; {stat['ci95_upper']:+.4f}] n={stat['n']}"


def _has_anomaly(meta) -> bool:
    """True when the gold answer describes a real anomaly (the Z1-bis panel was 208 positives)."""

    payload = meta.answer.split(":", 1)[-1]
    try:
        gold = json.loads(payload)
    except (ValueError, TypeError):
        return True
    if "present" in gold:
        return bool(gold["present"])
    if "anomalies" in gold:
        return bool(gold["anomalies"])
    if "kind" in gold:
        return str(gold["kind"]).strip().lower() not in {"", "no anomaly", "none"}
    return True


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(np.asarray(list(values), dtype=np.float64))) if values else None


def _to_device(episode, device: torch.device):
    return type(episode)(
        episode.values.to(device),
        episode.valid.to(device),
        episode.channel_ids,
        episode.sample_times.to(device),
        episode.task,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="complete checkpoint directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=60)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-reasoning-tokens", type=int, default=384)
    parser.add_argument("--categories", default="all",
                        help="all | signal (the three categories the header cannot answer) | comma-separated list")
    parser.add_argument("--ids-file", type=Path, default=None,
                        help="one recording id per line: the exact panel to evaluate, e.g. the 208 of Z1-bis in "
                             "integrity/artifacts/l3_riferimenti/gate208_ids.txt")
    parser.add_argument("--positives-only", action="store_true",
                        help="keep only rows whose gold answer carries an anomaly, as the Z1-bis panel did")
    parser.add_argument("--conditions", default="all",
                        help="all | content (real,empty,substituted) | comma-separated subset")
    parser.add_argument("--thinking", choices=("auto", "on", "off"), default="auto",
                        help="auto follows the checkpoint's reading_segment: a model trained without a "
                             "think block must not be asked to generate one")
    args = parser.parse_args()
    if args.rows < 2:
        parser.error("--rows must select at least two recordings")
    if args.max_new_tokens <= 0 or args.max_reasoning_tokens <= 0:
        parser.error("the decoding budgets must be positive")
    started = time.time()

    cfg = load_runtime_config(args.config, validate_paths=False)
    raw, spec = cfg.as_dict(), dict(cfg.spec)
    device = torch.device("cuda:0")
    runtime = raw["runtime"]
    backbone = load_backbone(
        raw["backbone"]["snapshot"]["root"],
        backend=runtime["attention_backend"],
        pad_to=int(runtime["attention_qk_pad_to"]),
        device=device,
        gradient_checkpointing=False,
    )
    training = spec["training"]
    model = WRAgentModel(backbone, spec, segment_weights=dict(training["segment_loss_weights"]), aux_weight=0.0)
    model.frontend.to(device)
    model.mixer.to(device)
    model.projector.to(device)
    ref = inspect_foreign_checkpoint(args.checkpoint)
    model.load_trainable_state_dict(load_file(str(ref.path / "model.safetensors")))
    backbone.ctx.lam = float(ref.trainer_state["wildcard_lambda"])
    model.eval()
    backbone.text.eval()
    if args.conditions == "all":
        conditions = CONDITIONS
    elif args.conditions == "content":
        conditions = ("real", "empty", "substituted")
    else:
        conditions = tuple(name.strip() for name in args.conditions.split(",") if name.strip())
        unknown = [name for name in conditions if name not in CONDITIONS]
        if unknown:
            parser.error(f"unknown conditions: {', '.join(unknown)}")
    thinking = bool(model.reading_segment) if args.thinking == "auto" else (args.thinking == "on")
    print(f"thinking={thinking} (spec reading_segment={bool(model.reading_segment)})", flush=True)

    dataset_root = Path(raw["dataset"]["root"])
    validation = AnomalyXLSplit(
        dataset_root / raw["dataset"]["validation_relative_path"],
        "validation",
        int(raw["dataset"]["validation_rows"]),
    )
    metas = validation.metas()
    if args.categories == "signal":
        wanted = set(SIGNAL_DEPENDENT)
    elif args.categories == "all":
        wanted = None
    else:
        wanted = {name.strip() for name in args.categories.split(",") if name.strip()}
    pool = [meta for meta in metas if wanted is None or meta.category in wanted]
    if args.positives_only:
        pool = [meta for meta in pool if _has_anomaly(meta)]
    if args.ids_file is not None:
        panel = {line.strip() for line in args.ids_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        pool = [meta for meta in pool if meta.id in panel]
        found = {meta.id for meta in pool}
        if missing := sorted(panel - found):
            print(f"warning: {len(missing)} of {len(panel)} panel ids are not in this split "
                  f"(first: {missing[0]})", flush=True)
        rows = [meta.row_index for meta in pool]  # the panel IS the sample; no further subsetting
    else:
        rows = stratified_subset(pool, args.rows, args.seed)
    print(f"panel: {len(rows)} rows drawn from {len(pool)} candidates "
          f"(categories={args.categories}, positives_only={args.positives_only}, "
          f"ids_file={args.ids_file})", flush=True)
    if len(rows) < 2:
        raise RuntimeError("the selected validation subset has fewer than two recordings")
    by_row = {meta.row_index: meta for meta in metas}

    readings = {row: model.reading_text(validation.episode(row), []) for row in rows}
    permutation = derangement_indices(len(rows), args.seed)
    permuted_row = {row: rows[permutation[index]] for index, row in enumerate(rows)}

    shapes = {}
    for meta in metas:
        episode = validation.episode(meta.row_index)
        shapes[meta.row_index] = (int(episode.values.shape[-1]), int(episode.values.shape[0]))

    def token_donor_for(row: int) -> int | None:
        """Deterministic compatible donor: same length and channel count, different recording."""

        target = shapes[row]
        pool = sorted(other for other, shape in shapes.items() if shape == target and other != row)
        return pool[row % len(pool)] if pool else None

    # The projector is patched rather than the encoder so that zeroing costs one forward, not one encode.
    original_projector_forward = model.projector.forward
    zero_state = {"enabled": False}

    def projector_forward(z):
        projected = original_projector_forward(z)
        return torch.zeros_like(projected) if zero_state["enabled"] else projected

    model.projector.forward = projector_forward
    records: list[dict] = []
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for number, row in enumerate(rows, start=1):
                meta = by_row[row]
                true_episode = _to_device(validation.episode(row), device)
                true_header = model.encode_episode(true_episode, [], gold=meta.gold_intervals).views[0].header
                token_donor = token_donor_for(row)
                donor_episode = None
                if token_donor is not None:
                    donor = validation.episode(token_donor)
                    donor_episode = type(true_episode)(
                        donor.values.to(device), true_episode.valid, true_episode.channel_ids,
                        true_episode.sample_times, true_episode.task,
                    )
                text_donor = permuted_row[row]
                oracle_by_condition = {
                    "real": None,
                    "empty": None,
                    "substituted": None,
                    "oracle": readings[row],
                    "permuted": readings[text_donor],
                }
                record = {
                    "recording_id": meta.id, "category": meta.category, "row": int(row),
                    "text_donor_row": int(text_donor),
                    "token_donor_row": None if token_donor is None else int(token_donor),
                    "true_reading": readings[row],
                    "ce": {}, "primary": {}, "invalid_format": {}, "prediction": {},
                    "generated_reading": {}, "generation": {}, "final_field_tokens": {},
                }
                for condition in conditions:
                    if condition == "substituted" and donor_episode is None:
                        for key in ("ce", "primary", "invalid_format"):
                            record[key][condition] = None
                        continue
                    zero_state["enabled"] = condition in ZEROED_CONDITIONS
                    oracle_text = oracle_by_condition[condition]
                    episode = donor_episode if condition == "substituted" else true_episode
                    # the donor must not leak through the header: the receiver's header is kept verbatim
                    overrides = {0: true_header} if condition == "substituted" else None

                    generated = model.generate(
                        meta, episode, mode="direct", max_new_tokens=args.max_new_tokens,
                        max_reasoning_tokens=args.max_reasoning_tokens, thinking=thinking,
                        header_overrides=overrides, oracle_text=oracle_text,
                    )
                    prediction = generated.get("text") or ""
                    record["primary"][condition] = float(score_precise(prediction, meta.answer)["primary"])
                    record["invalid_format"][condition] = parse_structured_pred(
                        prediction, REQUIRED_KEYS[meta.category]) is None
                    record["prediction"][condition] = prediction[:500]
                    record["generated_reading"][condition] = (generated.get("reading") or "")[:300]
                    record["generation"][condition] = {
                        "prompt_tokens": int(generated.get("prompt_length", 0)),
                        "reasoning_tokens": int(generated.get("reasoning_tokens", 0)),
                        "replayed_ids": len(generated.get("reasoning_ids", [])),
                        "final_tokens": int(generated.get("final_tokens", 0)),
                        "finished": bool(generated.get("finished", False)),
                        "forced_closure": bool(generated.get("forced_closure", False)),
                    }

                    # replay that reading verbatim in front of the gold final answer; a checkpoint trained
                    # without a think block keeps its own format, with no reasoning turn at all
                    encoding = model.encode_episode(episode, [], gold=meta.gold_intervals,
                                                    header_overrides=overrides)
                    sequence = model.build_sequence(
                        meta, encoding, mode="direct", actions=[], answer=meta.answer,
                        supervise_action=False,
                        reading=PreTokenized(generated["reasoning_ids"]) if thinking else None,
                        header_overrides=overrides, oracle_text=oracle_text,
                    )
                    target_positions = final_field_positions(sequence, backbone.tokenizer)
                    record["final_field_tokens"][condition] = int(target_positions.numel())
                    record["ce"][condition] = None
                    if target_positions.numel():
                        embeds, is_signal, tau = model._inputs(sequence, encoding)
                        backbone.ctx.set_tokens(is_signal, tau)
                        hidden = backbone.text(inputs_embeds=embeds, use_cache=False).last_hidden_state[0]
                        logits = backbone.lm_head(hidden[(target_positions - 1).to(device)]).float()
                        record["ce"][condition] = float(
                            F.cross_entropy(logits, sequence.labels[target_positions].to(device)))
                zero_state["enabled"] = False
                records.append(record)
                print(f"[{number:2d}/{len(rows)}] {meta.category:24s} " + " ".join(
                    f"{condition[:4]} ce {record['ce'].get(condition) if record['ce'].get(condition) is not None else float('nan'):.3f}"
                    f"/p {record['primary'].get(condition) if record['primary'].get(condition) is not None else float('nan'):.2f}"
                    for condition in conditions), flush=True)
    finally:
        zero_state["enabled"] = False
        model.projector.forward = original_projector_forward

    def paired(condition_a: str, condition_b: str, key: str, categories=None) -> list[float]:
        """Per-recording ``a - b`` over the rows where both conditions produced a value."""

        out = []
        for record in records:
            if categories is not None and record["category"] not in categories:
                continue
            left, right = record[key].get(condition_a), record[key].get(condition_b)
            if left is not None and right is not None:
                out.append(float(left) - float(right))
        return out

    def by_category(key: str) -> dict:
        """Per-category condition means, so that a header-derivable category cannot carry the verdict."""

        categories = sorted({record["category"] for record in records})
        return {category: {condition: _mean([record[key][condition] for record in records
                                             if record["category"] == category
                                             and record[key].get(condition) is not None])
                           for condition in conditions}
                for category in categories}

    ce_mean = {c: _mean([r["ce"][c] for r in records if r["ce"].get(c) is not None]) for c in conditions}
    primary_mean = {c: _mean([r["primary"][c] for r in records if r["primary"].get(c) is not None])
                    for c in conditions}
    invalid_rate = {c: _mean([float(r["invalid_format"][c]) for r in records
                              if r["invalid_format"].get(c) is not None]) for c in conditions}

    # --- the oracle question (pre-registered by Astra, gates the verdict) ---
    oracle_bootstrap = {
        "permuted_minus_oracle": paired_bootstrap(paired("permuted", "oracle", "ce"), args.seed),
        "empty_minus_oracle": paired_bootstrap(paired("empty", "oracle", "ce"), args.seed),
    }
    criteria = {
        "ci95_lower_gt_zero_permuted_minus_oracle": (
            (oracle_bootstrap["permuted_minus_oracle"]["ci95_lower"] or 0.0) > 0.0),
        "ci95_lower_gt_zero_empty_minus_oracle": (
            (oracle_bootstrap["empty_minus_oracle"]["ci95_lower"] or 0.0) > 0.0),
        "primary_oracle_gt_permuted": (primary_mean.get("oracle") or 0.0) > (primary_mean.get("permuted") or 0.0),
        "primary_oracle_gt_empty": (primary_mean.get("oracle") or 0.0) > (primary_mean.get("empty") or 0.0),
    }
    if not {"oracle", "permuted", "empty"} <= set(conditions):
        verdict = "NOT_RUN"  # the oracle arm was not part of this panel
    elif all(criteria.values()):
        verdict = "POSITIVE"
    elif ((oracle_bootstrap["permuted_minus_oracle"]["ci95_upper"] or 0.0) < 0.0
          or (oracle_bootstrap["empty_minus_oracle"]["ci95_upper"] or 0.0) < 0.0):
        verdict = "NEGATIVE"
    else:
        verdict = "INCONCLUSIVE"  # Astra 16/09: a CI crossing zero is not a negative result

    # --- the content question (descriptive; the quantity Z1-bis measured as +0.1349) ---
    content = {
        "primary_real_minus_substituted": paired_bootstrap(paired("real", "substituted", "primary"), args.seed),
        "primary_real_minus_empty": paired_bootstrap(paired("real", "empty", "primary"), args.seed),
        "ce_substituted_minus_real": paired_bootstrap(paired("substituted", "real", "ce"), args.seed),
        "ce_empty_minus_real": paired_bootstrap(paired("empty", "real", "ce"), args.seed),
        "z1bis_reference": {"primary_delta": 0.1349, "ci95": [0.0954, 0.1748], "threshold": 0.05},
    }
    # the gate quantity: the same difference restricted to the categories the header cannot answer
    signal_only = {
        "categories": list(SIGNAL_DEPENDENT),
        "primary_real_minus_substituted": paired_bootstrap(
            paired("real", "substituted", "primary", SIGNAL_DEPENDENT), args.seed),
        "primary_real_minus_empty": paired_bootstrap(
            paired("real", "empty", "primary", SIGNAL_DEPENDENT), args.seed),
        "ce_substituted_minus_real": paired_bootstrap(
            paired("substituted", "real", "ce", SIGNAL_DEPENDENT), args.seed),
    }
    def clears(stat: dict) -> bool:
        return bool(stat["mean"] is not None and stat["mean"] >= 0.05 and (stat["ci95_lower"] or 0.0) > 0.0)

    # Astra 16/09: replicating the Z1-bis gate means clearing the threshold against BOTH controls, not only the donor.
    signal_only["clears_vs_substituted"] = clears(signal_only["primary_real_minus_substituted"])
    signal_only["clears_vs_empty"] = clears(signal_only["primary_real_minus_empty"])
    signal_only["meets_z1bis_threshold"] = (
        signal_only["clears_vs_substituted"] and signal_only["clears_vs_empty"])

    report = {
        "checkpoint": str(ref.path), "optimizer_step": ref.optimizer_step, "rows": len(records),
        "seed": args.seed, "conditions": list(conditions),
        "budgets": {"max_new_tokens": args.max_new_tokens,
                    "max_reasoning_tokens": args.max_reasoning_tokens if thinking else 0,
                    "thinking": thinking},
        "ce_mean": ce_mean, "primary_mean": primary_mean, "invalid_rate": invalid_rate,
        "oracle_bootstrap": oracle_bootstrap, "criteria": criteria, "verdict": verdict,
        "content_dependence": content, "content_dependence_signal_only": signal_only,
        "primary_by_category": by_category("primary"), "ce_by_category": by_category("ce"),
        "seconds": round(time.time() - started, 1), "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"CONTENT_GATE step {ref.optimizer_step} on {SIGNAL_DEPENDENT}: "
          f"primary real-substituted {_fmt_stat(signal_only['primary_real_minus_substituted'])} | "
          f"threshold +0.05 with CI above zero -> "
          f"{'PASS' if signal_only['meets_z1bis_threshold'] else 'FAIL'}", flush=True)
    print(f"ORACLE_PROBE step {ref.optimizer_step}: ce {ce_mean} | primary {primary_mean} | "
          f"criteria {criteria} | VERDICT {verdict} | content "
          f"{ {k: v.get('mean') for k, v in content.items() if isinstance(v, dict) and 'mean' in v} } "
          f"({report['seconds']}s)", flush=True)
    return 0 if (verdict == "POSITIVE" or signal_only["meets_z1bis_threshold"]) else 1


if __name__ == "__main__":
    sys.exit(main())
