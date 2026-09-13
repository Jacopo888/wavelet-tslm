#!/usr/bin/env python3
"""Pianificatore W1: tracce che insegnano alla policy le coordinate del selettore.

Ramo 2 del piano notturno. Differenza rispetto a Z1-bis: la finestra insegnata non
viene costruita attorno al gold, ma attorno a un **massimo di energia Haar**, che il
modello puo' stimare dalla sua vista globale. Il gold interviene solo come filtro:
decide quale candidato (tutti calcolabili in inferenza) vale la pena insegnare e
quali sono inutili.

Catena per ogni riga positiva del train:
  1. candidati = i primi `candidate_buckets` intervalli per energia Haar nel canale
     considerato, alla larghezza dichiarata;
  2. per ogni candidato, lo start viene campionato uniformemente fra quelli che
     contengono il **centro dell'intervallo** (non l'evento): la posizione interna
     resta variabile e la scorciatoia del centro non si ricrea;
  3. si tiene il primo candidato la cui finestra contiene l'evento gold intero;
  4. se nessun candidato e' utile, si ricade sul meccanismo Z1-bis (start uniforme
     fra quelli che contengono l'evento), registrando la riga come `gold_fallback`.

Il passo 4 evita di insegnare «zoom inutile poi risposta giusta», che
`validate_trace_plan` vieta, e mantiene la copertura a 4.488 righe. La frazione di
ricadute e' essa stessa una misura: dice quanto spesso l'energia Haar da' sola non
basta sul train.

Il manifest prodotto e' nel formato che `zoom_sft.py` consuma senza modifiche; gli
origini per riga vanno in `w1_origins.jsonl` accanto al piano.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from l3.grpo import _atomic_json_dump, load_l3_config, sha256_file
from l3.tools.compare_selectors import channel_details, wavelet_scores
from l3.zoom import ZoomAction, gold_events, series_length
from l3.zoom_traces import (
    ALL_CATEGORIES,
    DIRECT_CATEGORIES,
    TRAIN_TRACE_COLUMNS,
    Trace,
    _choose_replay_indices,
    _weighted_counts,
    _stable_rng,
    _train_parquet_path,
    compact_trace_sample,
    length_band,
    sample_containing_action,
    validate_trace_plan,
)

W1_COLUMNS = TRAIN_TRACE_COLUMNS + ("channel_values",)


def _w1_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    sft = config.get("zoom_sft", {})
    if sft.get("trace_contract") != "z1bis_v1":
        raise ValueError("zoom_sft.trace_contract must be z1bis_v1 (same plan format)")
    w1 = sft.get("w1")
    if not isinstance(w1, Mapping):
        raise ValueError("zoom_sft.w1 block is required for the selector-imitation plan")
    buckets = int(w1.get("candidate_buckets", 0))
    if not 1 <= buckets <= 16:
        raise ValueError("w1.candidate_buckets must be in [1, 16]")
    draws = int(w1.get("draws_per_candidate", 0))
    if not 1 <= draws <= 16:
        raise ValueError("w1.draws_per_candidate must be in [1, 16]")
    return w1


def assign_widths(samples: Sequence[Mapping[str, Any]],
                  provisional: Mapping[int, Mapping[str, Any]],
                  sft: Mapping[str, Any], seed: int) -> dict[int, int]:
    """Conteggi esatti per categoria, come Z1-bis: i pesi sono verificati a valle.

    `validate_trace_plan` confronta la distribuzione osservata delle ampiezze con i
    pesi dichiarati entro il 2%, quindi l'estrazione indipendente non basta.
    """
    choices = [(int(item["samples"]), float(item["weight"])) for item in sft["window_sizes"]]
    by_category: dict[str, list[int]] = defaultdict(list)
    for index in provisional:
        by_category[str(samples[index]["category"])].append(index)
    widths: dict[int, int] = {}
    for category, indices in sorted(by_category.items()):
        counts = _weighted_counts(len(indices), choices)
        bag = [width for width, _ in choices for _ in range(counts[width])]
        _stable_rng(seed, category, "widths").shuffle(bag)
        ordered = list(indices)
        _stable_rng(seed, category, "rows").shuffle(ordered)
        for index, width in zip(ordered, bag):
            event = provisional[index]
            span = int(event["end"]) - int(event["start"])
            if width < span or width > series_length(samples[index]):
                raise ValueError(
                    f"assigned window {width} cannot contain event at train index "
                    f"{index}; no fallback is permitted")
            widths[index] = width
    return widths


def sample_containing_point(length: int, point: int, width: int,
                            rng: random.Random) -> tuple[int, int]:
    """Start uniforme fra tutte le finestre in-bounds che contengono ``point``."""
    if not 0 <= point < length or not 0 < width <= length:
        raise ValueError("invalid point/width for the containing-window sampler")
    low = max(0, point - width + 1)
    high = min(point, length - width)
    if low > high:
        low = high = max(0, min(point, length - width))
    start = rng.randint(low, high)
    return start, start + width


def wavelet_candidate_centers(sample: Mapping[str, Any], channel_index: int, width: int,
                              count: int,
                              details: Sequence[Mapping[int, np.ndarray]] | None = None,
                              ) -> list[int]:
    """Centri degli intervalli a piu' alta energia Haar nel canale, in ordine."""
    length = series_length(sample)
    buckets = max(8, math.ceil(length / max(1, width)))
    scores, edges = wavelet_scores(sample, buckets, details)
    row = scores[channel_index]
    order = np.argsort(-row, kind="stable")[:count]
    return [(int(edges[bucket]) + int(edges[bucket + 1])) // 2 for bucket in order]


def plan_positive_row(sample: Mapping[str, Any], w1: Mapping[str, Any],
                      event: Mapping[str, Any], width: int, rng: random.Random,
                      details: Sequence[Mapping[int, np.ndarray]],
                      ) -> tuple[ZoomAction, float, str]:
    """Sceglie la finestra da insegnare; ritorna anche l'origine del candidato."""
    names = [str(name) for name in sample["channel_names"]]
    channel_index = names.index(str(event["channel"]))
    length = series_length(sample)
    draws = int(w1["draws_per_candidate"])
    centers = wavelet_candidate_centers(
        sample, channel_index, width, int(w1["candidate_buckets"]), details)
    start, end = int(event["start"]), int(event["end"])
    for rank, center in enumerate(centers, start=1):
        for _ in range(draws):
            window = sample_containing_point(length, center, width, rng)
            if window[0] <= start and end <= window[1]:
                action = ZoomAction(str(event["channel"]), window[0], window[1], length)
                position = ((start + end) / 2.0 - window[0]) / width
                return action, position, f"wavelet_rank_{rank}"
    action, position = sample_containing_action(sample, event, width, rng)
    return action, position, "gold_fallback"


def choose_event(sample: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
    """Evento gold da insegnare, scelto sui soli metadati (nessun valore serve)."""
    events = sorted(gold_events(sample), key=lambda item: (
        str(item["channel"]), int(item["start"]), int(item["end"])))
    if not events:
        raise ValueError("positive row without gold events")
    return dict(rng.choice(events))


def build_w1_routes(samples: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
                    seed: int) -> tuple[dict[int, Trace], dict[int, Mapping[str, Any]],
                                        dict[int, int], set[int]]:
    """Rotte, evento e ampiezza senza toccare i valori: replay, diretta, zoom."""
    sft = config["zoom_sft"]
    _w1_config(config)
    if not samples:
        raise ValueError("cannot plan an empty dataset")
    replay = _choose_replay_indices(samples, sft, seed)
    fixed: dict[int, Trace] = {}
    provisional: dict[int, Mapping[str, Any]] = {}
    for index, sample in enumerate(samples):
        category = str(sample.get("category", ""))
        if category not in ALL_CATEGORIES:
            raise ValueError(f"unexpected category at index {index}: {category}")
        common = {"index": index, "sample_id": str(sample.get("id", "")),
                  "category": category}
        if index in replay:
            fixed[index] = Trace(**common, source="replay")
        elif category in DIRECT_CATEGORIES or not gold_events(sample):
            fixed[index] = Trace(**common, source="direct")
        else:
            provisional[index] = choose_event(
                sample, _stable_rng(seed, sample.get("id", ""), index, "w1_event"))
    widths = assign_widths(samples, provisional, sft, seed)
    return fixed, provisional, widths, replay


def plan_zoom_row(sample: Mapping[str, Any], details: Sequence[Mapping[int, np.ndarray]],
                  config: Mapping[str, Any], seed: int, index: int,
                  event: Mapping[str, Any], width: int,
                  ) -> tuple[Trace, dict[str, Any]]:
    """Pianifica una singola riga con azione, usando la piramide appena calcolata."""
    sft = config["zoom_sft"]
    w1 = _w1_config(config)
    radius = float(sft["position"]["centered_radius"])
    rng = _stable_rng(seed, sample.get("id", ""), index, "w1")
    action, position, origin = plan_positive_row(sample, w1, event, width, rng, details)
    common = {"index": index, "sample_id": str(sample.get("id", "")),
              "category": str(sample["category"])}
    trace = Trace(
        **common, source="zoom", action=action, event=dict(event), position=position,
        length_band=length_band(series_length(sample), sft),
        centered=abs(position - 0.5) <= radius,
    )
    return trace, {**common, "origin": origin, "width": action.window_len,
                   "position": position}


def stream_train_rows(config: Mapping[str, Any]) -> Iterator[tuple[dict[str, Any], list[np.ndarray]]]:
    """Legge il train una riga alla volta, valori inclusi, senza materializzarlo."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required by the W1 planner") from exc
    train_path = _train_parquet_path(config)
    if not train_path.is_file():
        raise FileNotFoundError(f"missing train parquet: {train_path}")
    parquet = pq.ParquetFile(train_path)
    available = set(parquet.schema_arrow.names)
    missing = [column for column in W1_COLUMNS if column not in available]
    if missing:
        raise ValueError(f"train parquet missing required column: {missing[0]}")
    for batch in parquet.iter_batches(batch_size=1, columns=list(W1_COLUMNS)):
        meta = {
            "id": str(batch.column("id")[0].as_py()),
            "category": str(batch.column("category")[0].as_py()),
            "length": int(batch.column("length")[0].as_py()),
            "n_channels": int(batch.column("n_channels")[0].as_py()),
            "channel_names": [str(name) for name in
                              (batch.column("channel_names")[0].as_py() or [])],
            "answer": str(batch.column("answer")[0].as_py()),
        }
        length, names = meta["length"], meta["channel_names"]
        outer = batch.column("channel_values")[0].values
        try:
            channels = [np.asarray(outer[i].values.to_numpy(zero_copy_only=False),
                                   dtype=np.float64) for i in range(len(outer))]
        except (AttributeError, TypeError):
            channels = [np.asarray(values, dtype=np.float64)
                        for values in outer.to_pylist()]
        if len(channels) != len(names) or any(ch.size != length for ch in channels):
            raise ValueError(f"channel length mismatch on row {meta['id']}")
        if meta["n_channels"] != len(names):
            raise ValueError(f"n_channels mismatch on row {meta['id']}")
        yield meta, channels
        del channels, outer, batch


def write_w1_plan(out_dir: Path, config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Due passaggi: metadati per le rotte, poi valori in streaming per le azioni.

    Le piramidi non vengono mai accumulate: si calcolano sulla riga corrente e si
    scartano. E' la regola imparata quando il planner Z1-bis fece cadere il nodo.
    """
    if out_dir.exists():
        raise FileExistsError(f"trace output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    started = time.perf_counter()
    try:
        from l3.zoom_traces import _load_train_samples

        expected = int(config["rl"]["expected_train_rows"])
        samples, observed_sha = _load_train_samples(config)
        if len(samples) != expected:
            raise RuntimeError(f"expected {expected} train rows, found {len(samples)}")
        fixed, provisional, widths, replay = build_w1_routes(samples, config, seed)
        zoom_indices = set(provisional)
        print(f"W1_TRACES routes: zoom={len(zoom_indices)} replay={len(replay)} "
              f"direct={len(fixed) - len(replay)}", flush=True)

        planned: dict[int, Trace] = {}
        origins: dict[int, dict[str, Any]] = {}
        for index, (meta, channels) in enumerate(stream_train_rows(config)):
            if index not in zoom_indices:
                del channels
                continue
            compact = samples[index]
            if str(meta["id"]) != str(compact["id"]):
                raise RuntimeError(f"stream/metadata order mismatch at index {index}")
            full = {**meta, "time_series": channels}
            details = channel_details(full)
            trace, origin = plan_zoom_row(compact, details, config, seed, index,
                                          provisional[index], widths[index])
            planned[index] = trace
            origins[index] = origin
            del channels, full, details
            if len(planned) % 250 == 0:
                print(f"W1_TRACES planned {len(planned)}/{len(zoom_indices)}", flush=True)
        if set(planned) != zoom_indices:
            missing = sorted(zoom_indices - set(planned))[:3]
            raise RuntimeError(f"zoom rows left unplanned: {missing}")

        traces: list[Trace] = []
        origin_rows: list[dict[str, Any]] = []
        for index in range(expected):
            if index in planned:
                traces.append(planned[index])
                origin_rows.append(origins[index])
            else:
                trace = fixed[index]
                traces.append(trace)
                origin_rows.append({"index": index, "sample_id": trace.sample_id,
                                    "category": trace.category,
                                    "origin": trace.source})

        rows_path = out_dir / "traces.jsonl"
        with rows_path.open("x", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(json.dumps(trace.to_json(), sort_keys=True) + "\n")
        with (out_dir / "w1_origins.jsonl").open("x", encoding="utf-8") as handle:
            for row in origin_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        validated = validate_trace_plan(samples, traces, config)
        counts = Counter(str(row["origin"]) for row in origin_rows)
        zoom_rows = sum(value for key, value in counts.items()
                        if key.startswith("wavelet_rank_") or key == "gold_fallback")
        w1_block = {
            "origin_counts": dict(counts),
            "zoom_rows": zoom_rows,
            "wavelet_rate": (sum(value for key, value in counts.items()
                                 if key.startswith("wavelet_rank_")) / zoom_rows
                             if zoom_rows else None),
            "gold_fallback_rate": (counts.get("gold_fallback", 0) / zoom_rows
                                   if zoom_rows else None),
            "candidate_buckets": int(config["zoom_sft"]["w1"]["candidate_buckets"]),
            "draws_per_candidate": int(config["zoom_sft"]["w1"]["draws_per_candidate"]),
            "declared_window_sizes": [int(item["samples"])
                                      for item in config["zoom_sft"]["window_sizes"]],
        }
        summary = {
            "status": "DONE", "format": "l3_zoom_z1bis_trace_manifest_v1",
            "seed": seed, "train_sha256": observed_sha,
            "traces_sha256": sha256_file(rows_path),
            **validated, "w1": w1_block,
            "planner_elapsed_seconds": time.perf_counter() - started,
        }
        try:
            import resource
            summary["planner_peak_rss_mib"] = round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        except ImportError:
            pass
        declared = config["zoom_sft"]["w1"].get("min_wavelet_rate")
        if declared is not None and w1_block["wavelet_rate"] is not None:
            if w1_block["wavelet_rate"] < float(declared):
                raise RuntimeError(
                    "wavelet proposals useful on only "
                    f"{w1_block['wavelet_rate']:.4f} of zoom rows, below the "
                    f"declared {float(declared):.4f}")
        _atomic_json_dump(out_dir / "summary.json", summary)
        _atomic_json_dump(out_dir / "DONE", summary)
        return summary
    except Exception as exc:
        _atomic_json_dump(out_dir / "FAILED", {
            "status": "FAILED", "error_type": type(exc).__name__, "reason": str(exc),
        })
        raise

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    config = load_l3_config(str(args.config.expanduser().resolve()))
    summary = write_w1_plan(args.out_dir.expanduser().resolve(), config, int(args.seed))
    block = summary["w1"]
    print(f"L3_W1_TRACES_OK wavelet_rate={block['wavelet_rate']:.4f} "
          f"gold_fallback={block['gold_fallback_rate']:.4f} "
          f"elapsed_s={summary['planner_elapsed_seconds']:.1f} "
          f"peak_rss_mib={summary.get('planner_peak_rss_mib')}")
    print(json.dumps(block, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
