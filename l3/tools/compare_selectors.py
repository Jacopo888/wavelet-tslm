#!/usr/bin/env python3
"""Confronto dei selettori di finestra a pesi congelati - parte CPU (Codex punto 1).

Nessuna GPU, nessuna generazione: misura soltanto *dove* guardano i selettori sulle
stesse righe, allo stesso budget di campioni. La domanda e' se un selettore wavelet
deterministico punti meglio della policy autonoma a parita' di ampiezza.

Condizioni (tutte sulle righe in cui la policy ha agito, join per `id`):
  policy            finestra registrata dalla policy
  wavelet_matched   euristica Haar, stessa ampiezza della policy, griglia = L/ampiezza
  wavelet_native    euristica Haar con il suo budget nativo (max(1024, L/8))
  policy_wide_xK    centro della policy, ampiezza x K (quanto vale allargare)
  wavelet_wide_xK   centro dell'euristica, ampiezza x K
  oracle_matched    limite superiore: finestra della stessa ampiezza centrata sul gold

Metrica principale: *contenimento* dell'evento gold sul canale corretto. L'IoU fra una
finestra di ~2.000 campioni e un evento di decine di campioni non misura il puntamento.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import numpy as np

from l3.zoom import (
    MIN_ZOOM_SAMPLES,
    ZoomAction,
    channel_names,
    clamp_zoom_window,
    gold_events,
    heuristic_zoom,
    series_length,
)

WIDE_FACTORS = (2, 4, 8)
L_BANDS = (16384, 32768, 65536, 131072)


def _wavelet_pyramid_class() -> Any:
    try:
        from l2.overlay.timerlm.environments.timeseries_qa.wavelet_pyramid import WaveletPyramid
    except ImportError:
        from l3.wavelet_pyramid import WaveletPyramid
    return WaveletPyramid


def channel_details(sample: Mapping[str, Any]) -> list[dict[int, np.ndarray]]:
    """Dettagli Haar per canale, calcolati una volta sola per riga.

    La piramide non dipende dal numero di intervalli: calcolarla una volta per riga
    evita di rifarla per ogni budget confrontato (la fase e' quella che aveva fatto
    cadere il nodo).
    """
    WaveletPyramid = _wavelet_pyramid_class()
    return [WaveletPyramid(np.asarray(raw, dtype=np.float64)).details
            for raw in sample["time_series"]]


def wavelet_scores(sample: Mapping[str, Any], buckets: int,
                   details: Sequence[Mapping[int, np.ndarray]] | None = None,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Energia Haar normalizzata per livello, aggregata in `buckets` intervalli.

    Ritorna (punteggi [n_canali, buckets], bordi). Stessa aritmetica di
    `l3.zoom.heuristic_zoom`, con il numero di intervalli parametrizzato.
    """
    length = series_length(sample)
    bucket_count = min(max(1, buckets), length)
    edges = np.linspace(0, length, bucket_count + 1, dtype=int)
    if details is None:
        details = channel_details(sample)
    out = np.zeros((len(sample["time_series"]), bucket_count), dtype=np.float64)
    for channel_index, channel in enumerate(details):
        for level, detail in channel.items():
            energy = np.square(np.asarray(detail, dtype=np.float64))
            positive = energy[energy > 0]
            scale = float(np.median(positive)) if positive.size else 1.0
            scale = max(scale, np.finfo(np.float64).eps)
            centers = np.arange(energy.size) * (2 ** level) + (2 ** level) // 2
            ids = np.searchsorted(edges, centers, side="right") - 1
            valid = (ids >= 0) & (ids < bucket_count) & (centers < length)
            np.add.at(out[channel_index], ids[valid], energy[valid] / scale)
    return out, edges


def wavelet_zoom_at_width(sample: Mapping[str, Any], width: int,
                          buckets: int | None = None,
                          details: Sequence[Mapping[int, np.ndarray]] | None = None,
                          ) -> ZoomAction:
    """Euristica Haar con ampiezza imposta; la griglia segue il budget richiesto."""
    length = series_length(sample)
    names = channel_names(sample)
    target = min(length, max(MIN_ZOOM_SAMPLES, int(width)))
    if buckets is None:
        buckets = max(8, math.ceil(length / max(1, target)))
    scores, edges = wavelet_scores(sample, buckets, details)
    best: tuple[float, int, int] | None = None
    for channel_index in range(scores.shape[0]):
        bucket = int(np.argmax(scores[channel_index]))
        candidate = (float(scores[channel_index][bucket]), -channel_index, -bucket)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    channel_index, bucket = -best[1], -best[2]
    center = (int(edges[bucket]) + int(edges[bucket + 1])) // 2
    window = clamp_zoom_window(center - target // 2, center + (target - target // 2), length)
    if window is None:
        raise RuntimeError("wavelet selector produced an invalid window")
    return ZoomAction(names[channel_index], *window, length)


def recenter(action: ZoomAction, width: int, length: int) -> ZoomAction:
    """Stesso centro, ampiezza diversa: isola il costo di una finestra troppo stretta."""
    center = (action.start + action.end) // 2
    target = min(length, max(MIN_ZOOM_SAMPLES, int(width)))
    window = clamp_zoom_window(center - target // 2, center + (target - target // 2), length)
    if window is None:
        raise RuntimeError("recentring produced an invalid window")
    return ZoomAction(action.channel, *window, length)


def oracle_at_width(sample: Mapping[str, Any], width: int) -> ZoomAction | None:
    """Finestra centrata sull'evento piu' lungo, alla stessa ampiezza (limite superiore)."""
    events = gold_events(sample)
    if not events:
        return None
    length = series_length(sample)
    event = max(events, key=lambda e: (int(e["end"]) - int(e["start"]), -int(e["start"])))
    center = (int(event["start"]) + int(event["end"])) // 2
    target = min(length, max(MIN_ZOOM_SAMPLES, int(width)))
    window = clamp_zoom_window(center - target // 2, center + (target - target // 2), length)
    if window is None:
        return None
    return ZoomAction(str(event["channel"]), *window, length)


def geometry(sample: Mapping[str, Any], action: ZoomAction | None) -> dict[str, Any]:
    """Contenimento, sovrapposizione, canale e posizione interna rispetto al gold."""
    events = gold_events(sample)
    if action is None:
        return {"state": "no_action"}
    if not events:
        return {"state": "no_gold"}
    same = [e for e in events if str(e["channel"]) == action.channel]
    width = action.window_len
    out: dict[str, Any] = {
        "channel_hit": bool(same),
        "multichannel": len({str(e["channel"]) for e in events}) > 1,
        "width": width,
    }
    if not same:
        out["state"] = "wrong_channel"
        return out
    best = None
    for event in same:
        start, end = int(event["start"]), int(event["end"])
        inside = start >= action.start and end <= action.end
        overlap = max(0, min(end, action.end) - max(start, action.start))
        rank = (2 if inside else (1 if overlap > 0 else 0), overlap)
        if best is None or rank > best[0]:
            best = (rank, start, end, inside, overlap)
    _, start, end, inside, overlap = best
    center = (start + end) / 2.0
    out["state"] = "contained" if inside else ("partial" if overlap > 0 else "outside")
    out["position"] = (center - action.start) / width
    out["distance_widths"] = 0.0 if inside else max(
        action.start - end, start - action.end, 0) / width
    return out


def load_policy_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    acted = [row for row in rows if row.get("zoom")]
    if not acted:
        raise ValueError(f"no acted rows in {path}")
    return acted


def band_of(length: int) -> int:
    for band in L_BANDS:
        if length <= band:
            return band
    return L_BANDS[-1]


def summarize(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    states: dict[str, int] = defaultdict(int)
    positions: list[float] = []
    distances: list[float] = []
    widths: list[int] = []
    channel: list[bool] = []
    for record in records:
        geo = record["conditions"][key]
        states[geo.get("state", "unknown")] += 1
        if "position" in geo:
            positions.append(float(geo["position"]))
        if "distance_widths" in geo and geo["state"] != "contained":
            distances.append(float(geo["distance_widths"]))
        if "width" in geo:
            widths.append(int(geo["width"]))
        if "channel_hit" in geo:
            channel.append(bool(geo["channel_hit"]))
    total = sum(states.values())
    scored = total - states.get("no_gold", 0) - states.get("no_action", 0)
    out: dict[str, Any] = {
        "n": total,
        "scored": scored,
        "states": dict(states),
        "contained_rate": states["contained"] / scored if scored else float("nan"),
        "overlap_rate": ((states["contained"] + states["partial"]) / scored
                         if scored else float("nan")),
        "channel_hit_rate": (sum(channel) / len(channel)) if channel else float("nan"),
        "mean_width": statistics.fmean(widths) if widths else float("nan"),
    }
    if positions:
        out["position_mean"] = statistics.fmean(positions)
        out["position_median"] = statistics.median(positions)
    if distances:
        out["miss_distance_median_widths"] = statistics.median(distances)
    return out


def run(args: argparse.Namespace) -> int:
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists():
        raise FileExistsError(f"output already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    started = time.time()

    from l3.grpo import _atomic_json_dump, _runtime_imports, _write_jsonl, load_l3_config
    resolved = load_l3_config(str(args.config.expanduser().resolve()))
    ExperimentConfig = _runtime_imports(resolved)[0]
    from src.datasets.anomaly_xl.qa_dataset import AnomalyXLDataset

    experiment = ExperimentConfig.from_dict(
        {k: v for k, v in resolved.items() if not k.startswith("_")})
    kwargs = dict(experiment.dataset.extra_kwargs)
    # Il token EOS entra solo nel prompt testuale, che questo strumento non usa:
    # qui contano soltanto i valori delle serie e il gold.
    kwargs["EOS_TOKEN"] = args.eos_token
    dataset = AnomalyXLDataset(split="validation", **kwargs)

    by_id: dict[str, int] = {}
    for index in range(len(dataset)):
        sample_id = str(dataset[index].get("id", ""))
        if sample_id in by_id:
            raise ValueError(f"duplicate validation ID: {sample_id}")
        by_id[sample_id] = index

    policy_rows = load_policy_rows(args.policy_rows.expanduser().resolve())
    records: list[dict[str, Any]] = []
    rows_path = out_dir / "rows.jsonl"
    mirror_checked = 0
    with rows_path.open("x", encoding="utf-8") as handle:
        for ordinal, row in enumerate(policy_rows):
            sample_id = str(row["id"])
            if sample_id not in by_id:
                raise ValueError(f"policy ID missing from validation: {sample_id}")
            sample = dataset[by_id[sample_id]]
            if str(sample["category"]) != str(row["category"]):
                raise ValueError(f"category drift on {sample_id}")
            length = series_length(sample)
            zoom = row["zoom"]
            policy_action = ZoomAction(str(zoom["channel"]), int(zoom["start"]),
                                       int(zoom["end"]), length)
            width = policy_action.window_len
            details = channel_details(sample)
            wavelet_action = wavelet_zoom_at_width(sample, width, details=details)
            native = heuristic_zoom(sample)
            conditions = {
                "policy": geometry(sample, policy_action),
                "wavelet_matched": geometry(sample, wavelet_action),
                "wavelet_native": geometry(sample, native),
                "oracle_matched": geometry(sample, oracle_at_width(sample, width)),
            }
            for factor in WIDE_FACTORS:
                conditions[f"policy_wide_x{factor}"] = geometry(
                    sample, recenter(policy_action, width * factor, length))
                conditions[f"wavelet_wide_x{factor}"] = geometry(
                    sample, recenter(wavelet_action, width * factor, length))
            # Auto-verifica: alle impostazioni native il selettore parametrizzato deve
            # coincidere esattamente con `l3.zoom.heuristic_zoom` (64 intervalli, L/8).
            mirrored = wavelet_zoom_at_width(
                sample, max(MIN_ZOOM_SAMPLES, math.ceil(length / 8)), buckets=64,
                details=details)
            if (native.channel, native.start, native.end) != (
                    mirrored.channel, mirrored.start, mirrored.end):
                raise RuntimeError(f"selector mirror mismatch on {sample_id}")
            mirror_checked += 1
            record = {
                "id": sample_id, "index": by_id[sample_id],
                "category": str(sample["category"]), "length": length,
                "band": band_of(length), "policy_width": width,
                "primary": float(row["primary"]),
                "policy_action": {"channel": policy_action.channel,
                                  "start": policy_action.start, "end": policy_action.end},
                "wavelet_action": {"channel": wavelet_action.channel,
                                   "start": wavelet_action.start, "end": wavelet_action.end},
                "conditions": conditions,
            }
            _write_jsonl(handle, record)
            records.append(record)
            del sample
            if (ordinal + 1) % 20 == 0 or ordinal + 1 == len(policy_rows):
                print(f"SELECTORS {ordinal + 1}/{len(policy_rows)}", flush=True)

    keys = sorted(records[0]["conditions"])
    report = {
        "format": "l3_selector_geometry_v1",
        "inputs": {
            "config": str(args.config.expanduser().resolve()),
            "policy_rows": str(args.policy_rows.expanduser().resolve()),
        },
        "rows": len(records),
        "mirror_checked_rows": mirror_checked,
        "overall": {key: summarize(records, key) for key in keys},
        "per_category": {
            category: {key: summarize([r for r in records if r["category"] == category], key)
                       for key in keys}
            for category in sorted({r["category"] for r in records})
        },
        "per_band": {
            str(band): {key: summarize([r for r in records if r["band"] == band], key)
                        for key in keys}
            for band in sorted({r["band"] for r in records})
        },
        "runtime": {"seconds": round(time.time() - started, 1)},
    }
    try:
        import resource
        report["runtime"]["peak_rss_mib"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except ImportError:
        pass
    _atomic_json_dump(out_dir / "summary.json", report)

    print(f"\n{'condizione':22} {'n':>4} {'contenuto':>10} {'sovrapp':>9} "
          f"{'canale':>8} {'ampiezza':>9}")
    for key in keys:
        s = report["overall"][key]
        print(f"{key:22} {s['scored']:4} {100.0 * s['contained_rate']:9.1f}% "
              f"{100.0 * s['overlap_rate']:8.1f}% {100.0 * s['channel_hit_rate']:7.1f}% "
              f"{s['mean_width']:9.0f}")
    print("\ncontenimento per banda di lunghezza:")
    print(f"{'L':>8}  " + "  ".join(f"{k:>18}" for k in keys))
    for band in sorted({r["band"] for r in records}):
        cells = "  ".join(
            f"{100.0 * report['per_band'][str(band)][k]['contained_rate']:17.1f}%"
            for k in keys)
        print(f"{band:8}  {cells}")
    _atomic_json_dump(out_dir / "DONE", {"status": "DONE", "rows": len(records)})
    print("L3_SELECTORS_OK")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--policy-rows", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--eos-token", default="<|im_end|>")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
