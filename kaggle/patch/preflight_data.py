#!/usr/bin/env python
"""Preflight gate for AnomalyXL data roots — refuse to train on leaked splits.

Usage:
    python preflight_data.py <datasets_root> [--variant precise|coarse|both]

Checks, for each variant present under <root>:
  1. gold test exists (`anomalyxl-<v>/data.parquet`) with the expected rows;
  2. train/val (`anomalyxl-train/anomalyxl-<v>/{train,val}.parquet`), when
     present, are ID-DISJOINT from the gold test and from each other;
  3. when `anomalyxl-train/MANIFEST.json` exists, its revision matches the
     project pin (or TSLM_HF_REVISION).

Exit 0 = safe to train. Exit != 0 = do not train, message says why.

Born from the 02/09/2026 audit: the workstation carried train/val files that
were 100% subsets of the test (REPRODUCE.md Option B leak) — this gate makes
that state fatal instead of silent. Leaked files live on in
`_QUARANTINE_LEAKED/` (evidence, ignored by this check).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pyarrow.parquet as pq

PIN = os.environ.get(
    "TSLM_HF_REVISION", "df363a467f855e55aaad414b0bdd58590829df9a")
EXPECTED_TEST_ROWS = {"precise": 800, "coarse": 248}
# Training splits at the pinned revision. In training mode (default) their
# absence is FATAL: a root with only gold tests must never pass as
# "safe to train" (review finding, 02/09).
EXPECTED_SPLIT_ROWS = {"precise": {"train": 4488, "val": 520}}
EXPECTED_TEST_CATEGORIES = {
    "precise": {
        "localize": 200,
        "classify_with_evidence": 200,
        "measure_magnitude": 200,
        "localize_all_channels": 100,
        "lead_lag_with_magnitude": 100,
    },
}


def _ids(path: str) -> set:
    pf = pq.ParquetFile(path)
    if "id" not in pf.schema_arrow.names:
        return set()
    return set(pq.read_table(path, columns=["id"]).column("id").to_pylist())


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_variant(root: str, variant: str, problems: list,
                  require_training: bool = False) -> None:
    test_p = os.path.join(root, f"anomalyxl-{variant}", "data.parquet")
    if not os.path.isfile(test_p):
        problems.append(f"{variant}: gold test missing at {test_p}")
        return
    n_test = pq.ParquetFile(test_p).metadata.num_rows
    exp = EXPECTED_TEST_ROWS.get(variant)
    if exp is not None and n_test != exp:
        problems.append(f"{variant}: gold test has {n_test} rows, expected {exp}")
    exp_cats = EXPECTED_TEST_CATEGORIES.get(variant)
    if exp_cats:
        t = pq.read_table(test_p, columns=["category"])
        got = {}
        for c in t.column("category").to_pylist():
            got[c] = got.get(c, 0) + 1
        if got != exp_cats:
            problems.append(f"{variant}: test category counts {got} != expected {exp_cats}")

    test_ids = _ids(test_p)
    tr_dir = os.path.join(root, "anomalyxl-train", f"anomalyxl-{variant}")
    split_ids = {}
    for split in ("train", "val"):
        p = os.path.join(tr_dir, f"{split}.parquet")
        if os.path.isfile(p):
            split_ids[split] = _ids(p)
            exp_rows = EXPECTED_SPLIT_ROWS.get(variant, {}).get(split)
            if exp_rows is not None and len(split_ids[split]) != exp_rows:
                problems.append(
                    f"{variant}: {split} has {len(split_ids[split])} rows, "
                    f"expected {exp_rows} at pinned revision")
        elif require_training:
            problems.append(
                f"{variant}: {split}.parquet MISSING at {p} — a root without "
                f"training splits is NOT safe to train (run the patched "
                f"fetch_data.py; use --allow-test-only for eval-only roots)")
    for split, ids in split_ids.items():
        inter = ids & test_ids
        if inter:
            problems.append(
                f"{variant}: {split} shares {len(inter)}/{len(ids)} IDs with "
                f"the TEST — leaked split, quarantine it")
    if "train" in split_ids and "val" in split_ids:
        inter = split_ids["train"] & split_ids["val"]
        if inter:
            problems.append(f"{variant}: train and val share {len(inter)} IDs")
    # A stray data.parquet inside the train dir was the leak's signature.
    stray = os.path.join(tr_dir, "data.parquet")
    if os.path.isfile(stray) and (_ids(stray) & test_ids):
        problems.append(
            f"{variant}: {stray} contains TEST rows inside the train dir")

    print(f"[preflight] {variant}: test={n_test} rows, "
          f"train={len(split_ids.get('train', []))}, "
          f"val={len(split_ids.get('val', []))}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--variant", choices=["precise", "coarse", "both"],
                    default="precise")
    ap.add_argument("--allow-test-only", action="store_true",
                    help="eval-only roots: gold tests alone are acceptable "
                         "(training splits/manifest not required)")
    ap.add_argument("--skip-hash", action="store_true",
                    help="skip sha256 re-verification of manifest entries "
                         "(hashing the 2GB train shard takes ~10-20s)")
    args = ap.parse_args()
    require_training = not args.allow_test_only

    problems: list = []
    variants = (["precise", "coarse"] if args.variant == "both"
                else [args.variant])
    for v in variants:
        check_variant(args.root, v, problems,
                      require_training=require_training)

    manifest = os.path.join(args.root, "anomalyxl-train", "MANIFEST.json")
    if os.path.isfile(manifest):
        try:
            man = json.load(open(manifest))
        except Exception as e:
            man = None
            problems.append(f"MANIFEST unreadable: {e}")
        if man is not None:
            rev = man.get("revision")
            if rev != PIN:
                problems.append(f"MANIFEST revision {rev} != pinned {PIN}")
            else:
                print(f"[preflight] manifest pinned @ {PIN[:12]}")
            # Re-verify every manifest entry on disk: rows AND bytes. A
            # cache accepted on the revision string alone can still be a
            # corrupted or swapped file (review finding, 02/09).
            for key, ent in (man.get("files") or {}).items():
                # manifest paths are repo-root-relative ("anomalyXL/datasets/…");
                # our root IS <repo>/anomalyXL/datasets, so strip the prefix.
                rel = ent.get("path", "")
                marker = "anomalyXL/datasets/"
                rel = rel[rel.index(marker) + len(marker):] if marker in rel else rel
                p = os.path.join(args.root, rel)
                if not os.path.isfile(p):
                    problems.append(f"MANIFEST entry {key}: file missing at {p}")
                    continue
                rows = pq.ParquetFile(p).metadata.num_rows
                if rows != ent.get("rows"):
                    problems.append(
                        f"MANIFEST entry {key}: {rows} rows on disk vs "
                        f"{ent.get('rows')} in manifest")
                if not args.skip_hash:
                    h = _sha256(p)
                    if h != ent.get("sha256"):
                        problems.append(
                            f"MANIFEST entry {key}: sha256 mismatch "
                            f"(disk {h[:12]} vs manifest "
                            f"{str(ent.get('sha256'))[:12]})")
            if not problems:
                print("[preflight] manifest entries verified"
                      + ("" if args.skip_hash else " (rows+sha256)"))
    elif require_training:
        problems.append(
            "MANIFEST.json MISSING — unpinned data is not safe to train on "
            "(run the patched fetch_data.py)")
    else:
        print("[preflight] no MANIFEST.json (test-only root, allowed by flag)")

    if problems:
        for p in problems:
            print(f"[preflight] FAIL: {p}", file=sys.stderr)
        print(f"[preflight] ABORT: {len(problems)} problem(s) — DO NOT TRAIN",
              file=sys.stderr)
        return 1
    print("[preflight] OK: data root is safe to train on")
    return 0


if __name__ == "__main__":
    sys.exit(main())
