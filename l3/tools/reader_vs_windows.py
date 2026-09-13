#!/usr/bin/env python3
"""Separa il cambiamento del lettore da quello delle finestre (criteri dichiarati).

Confronta due valutazioni del gate percettivo prodotte con lo **stesso** config e
lo **stesso** seed su checkpoint diversi: canale, coordinate, ampiezza e header sono
identici per costruzione, e qui si verifica riga per riga confrontando l'hash
dell'header fra i due run. Senza quella verifica il confronto non isola il lettore.

Due confronti distinti, come da `PROTOCOL-L3.md` §«Separazione lettore/finestre»:
  Δ_lettura   = nuovo crop vero − vecchio crop vero (stessi crop)
  Δ_contenuto = nuovo crop vero − miglior controllo del nuovo (sostituito/azzerato)

Con piu' crop per ID (`--pair vecchio.jsonl:nuovo.jsonl` ripetuto) le differenze
vengono prima **mediate dentro l'ID** e poi bootstrappate **sugli ID**: i crop dello
stesso esempio sono correlati, quindi l'unita' di ricampionamento resta l'ID.

Solo CPU: legge i JSONL già prodotti.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
from typing import Any, Mapping, Sequence

TRUE = "offcenter_true"
CONTROLS = ("offcenter_swap", "offcenter_zero")


def load(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out.setdefault(str(row["condition"]), {})[str(row["id"])] = row
    return out


def boot(values: Sequence[float], *, seed: int, resamples: int = 10000) -> dict[str, Any]:
    if not values:
        raise ValueError("bootstrap requires values")
    rng = random.Random(seed)
    count = len(values)
    means = sorted(sum(values[rng.randrange(count)] for _ in range(count)) / count
                   for _ in range(resamples))
    return {"n": count, "mean": statistics.fmean(values),
            "ci95": [means[int(0.025 * resamples)],
                     means[min(resamples - 1, int(0.975 * resamples))]]}


def check_identical_windows(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Fail-closed: stessi ID e stesso header su ogni riga, altrimenti niente confronto."""
    ids = sorted(set(old[TRUE]) & set(new[TRUE]))
    if not ids:
        raise ValueError("no shared IDs between the two evaluations")
    missing = sorted((set(old[TRUE]) | set(new[TRUE])) - set(ids))
    if missing:
        raise ValueError(f"ID sets differ: {missing[:5]}")
    for sample_id in ids:
        a, b = old[TRUE][sample_id], new[TRUE][sample_id]
        if a.get("header_sha256") != b.get("header_sha256"):
            raise ValueError(f"header differs on {sample_id}: the windows are not identical")
        za, zb = a.get("zoom") or {}, b.get("zoom") or {}
        keys = ("channel", "start", "end", "window_len")
        if tuple(za.get(k) for k in keys) != tuple(zb.get(k) for k in keys):
            raise ValueError(f"window differs on {sample_id}: {za} vs {zb}")
        if str(a.get("ground_truth")) != str(b.get("ground_truth")):
            raise ValueError(f"ground truth differs on {sample_id}")
    return ids


def paired(old: Mapping[str, Any], new: Mapping[str, Any], ids: Sequence[str], *,
           seed: int, resamples: int) -> dict[str, Any]:
    differences = [float(new[i]["primary"]) - float(old[i]["primary"]) for i in ids]
    out = boot(differences, seed=seed, resamples=resamples)
    out["old_mean"] = statistics.fmean(float(old[i]["primary"]) for i in ids)
    out["new_mean"] = statistics.fmean(float(new[i]["primary"]) for i in ids)
    return out


def content_section(new: Mapping[str, Mapping[str, Any]], ids: Sequence[str], *,
                    seed: int, resamples: int) -> dict[str, Any]:
    means = {name: statistics.fmean(float(new[name][i]["primary"]) for i in ids)
             for name in CONTROLS}
    control = max(CONTROLS, key=lambda name: means[name])
    differences = [float(new[TRUE][i]["primary"]) - float(new[control][i]["primary"])
                   for i in ids]
    out = boot(differences, seed=seed, resamples=resamples)
    out["condition_means"] = {TRUE: statistics.fmean(
        float(new[TRUE][i]["primary"]) for i in ids), **means}
    out["best_control"] = control
    return out


def decide(reading: Mapping[str, Any], content: Mapping[str, Any], *,
           delta: float, min_content: float) -> dict[str, Any]:
    low, high = reading["ci95"]
    relevant_decline = reading["mean"] <= -delta and high < 0.0
    excludes_decline = low > -delta
    content_ok = content["mean"] >= min_content and content["ci95"][0] > 0.0
    inconclusive = low <= -delta and high >= 0.0
    if relevant_decline:
        branch = "reader_recovery"
        reason = (f"calo rilevante e convincente: Δ_lettura {reading['mean']:+.4f} "
                  f"con CI95 alto {high:+.4f} < 0")
    elif excludes_decline and content_ok:
        branch = "grpo_pilot"
        reason = (f"un calo di {delta} e' escluso (CI basso {low:+.4f}) e il contenuto "
                  f"resta utile ({content['mean']:+.4f})")
    elif inconclusive:
        branch = "inconclusive_enlarge"
        reason = (f"CI95 [{low:+.4f}; {high:+.4f}] compatibile sia con un calo >= {delta} "
                  "sia con nessun calo: si amplia la misura")
    else:
        branch = "inconclusive_enlarge"
        reason = (f"nessuna diramazione soddisfatta (Δ_lettura {reading['mean']:+.4f} "
                  f"CI[{low:+.4f}; {high:+.4f}], contenuto {content['mean']:+.4f} "
                  f"CI[{content['ci95'][0]:+.4f}; {content['ci95'][1]:+.4f}])")
    return {"branch": branch, "reason": reason,
            "checks": {"relevant_decline": relevant_decline,
                       "excludes_relevant_decline": excludes_decline,
                       "content_still_useful": content_ok,
                       "inconclusive": inconclusive},
            "criteria": {"delta": delta, "min_content": min_content,
                         "declared_in": "l3/PROTOCOL-L3.md, Separazione lettore/finestre"}}


def collect(pairs: Sequence[tuple[Path, Path]]) -> tuple[list[str], dict[str, list[float]],
                                                        dict[str, list[float]],
                                                        dict[str, str], dict[str, Any]]:
    """Differenze per ID su tutti i crop, con verifica delle finestre su ogni coppia."""
    reading: dict[str, list[float]] = {}
    content: dict[str, list[float]] = {}
    category: dict[str, str] = {}
    conditions: dict[str, list[float]] = {TRUE: [], CONTROLS[0]: [], CONTROLS[1]: []}
    shared: set[str] | None = None
    for old_path, new_path in pairs:
        old, new = load(old_path), load(new_path)
        ids = check_identical_windows(old, new)
        shared = set(ids) if shared is None else (shared & set(ids))
        # Il controllo si sceglie come nel gate percettivo dichiarato: il migliore
        # in MEDIA, non per riga, altrimenti il confronto non e' comparabile col
        # +0,135 pubblicato su Z1-bis.
        control = max(CONTROLS, key=lambda name: statistics.fmean(
            float(new[name][i]["primary"]) for i in ids))
        for sample_id in ids:
            reading.setdefault(sample_id, []).append(
                float(new[TRUE][sample_id]["primary"])
                - float(old[TRUE][sample_id]["primary"]))
            content.setdefault(sample_id, []).append(
                float(new[TRUE][sample_id]["primary"])
                - float(new[control][sample_id]["primary"]))
            category[sample_id] = str(new[TRUE][sample_id]["category"])
            for name in conditions:
                conditions[name].append(float(new[name][sample_id]["primary"]))
    assert shared is not None
    ids = sorted(shared)
    means = {name: statistics.fmean(values) for name, values in conditions.items()}
    return ids, reading, content, category, means


def cluster_boot(per_id: Mapping[str, Sequence[float]], ids: Sequence[str], *,
                 seed: int, resamples: int) -> dict[str, Any]:
    """Media dentro l'ID, poi bootstrap sugli ID: i crop di un ID sono correlati."""
    values = [statistics.fmean(per_id[i]) for i in ids]
    out = boot(values, seed=seed, resamples=resamples)
    out["crops_per_id"] = statistics.fmean(len(per_id[i]) for i in ids)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", default=[],
                        help="vecchio.jsonl:nuovo.jsonl, ripetibile per piu' crop per ID")
    parser.add_argument("--old-rows", type=Path)
    parser.add_argument("--new-rows", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--min-content", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    pairs: list[tuple[Path, Path]] = []
    for item in args.pair:
        old_text, _, new_text = item.partition(":")
        if not new_text:
            raise ValueError(f"coppia malformata: {item}")
        pairs.append((Path(old_text).expanduser().resolve(),
                      Path(new_text).expanduser().resolve()))
    if args.old_rows and args.new_rows:
        pairs.append((args.old_rows.expanduser().resolve(),
                      args.new_rows.expanduser().resolve()))
    if not pairs:
        raise ValueError("serve almeno una coppia di JSONL")

    ids, reading_by_id, content_by_id, category, condition_means = collect(pairs)
    crops = statistics.fmean(len(reading_by_id[i]) for i in ids)
    print(f"finestre identiche verificate su {len(ids)} ID, "
          f"{crops:.1f} crop per ID ({len(pairs)} coppie)")

    reading = cluster_boot(reading_by_id, ids, seed=args.seed, resamples=args.bootstrap)
    content = cluster_boot(content_by_id, ids, seed=args.seed + 1,
                           resamples=args.bootstrap)
    content["condition_means"] = condition_means
    per_category: dict[str, Any] = {}
    for ordinal, name in enumerate(sorted(set(category.values()))):
        subset = [i for i in ids if category[i] == name]
        per_category[name] = {
            "reading": cluster_boot(reading_by_id, subset,
                                    seed=args.seed + 10 + ordinal,
                                    resamples=args.bootstrap),
            "content": cluster_boot(content_by_id, subset,
                                    seed=args.seed + 30 + ordinal,
                                    resamples=args.bootstrap),
        }
    decision = decide(reading, content, delta=args.delta, min_content=args.min_content)

    print(f"\nΔ_lettura   (W1 vero − Z1bis vero): {reading['mean']:+.4f} "
          f"CI[{reading['ci95'][0]:+.4f}; {reading['ci95'][1]:+.4f}]")
    print(f"Δ_contenuto (W1 vero − miglior controllo): {content['mean']:+.4f} "
          f"CI[{content['ci95'][0]:+.4f}; {content['ci95'][1]:+.4f}]")
    print("condizioni W1:", {k: round(v, 4) for k, v in condition_means.items()})
    print("\nper categoria:")
    for name, block in per_category.items():
        r, c = block["reading"], block["content"]
        print(f"  {name:26} n={r['n']:3} lettura {r['mean']:+.4f} "
              f"CI[{r['ci95'][0]:+.4f}; {r['ci95'][1]:+.4f}]  contenuto {c['mean']:+.4f} "
              f"CI[{c['ci95'][0]:+.4f}; {c['ci95'][1]:+.4f}]")
    print(f"\nRAMO: {decision['branch']}\n  motivo: {decision['reason']}")

    report = {"format": "l3_reader_vs_windows_v2", "ids": len(ids),
              "crops_per_id": crops, "pairs": [[str(a), str(b)] for a, b in pairs],
              "identical_windows_verified": True, "reading": reading,
              "content": content, "per_category": per_category, **decision}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
