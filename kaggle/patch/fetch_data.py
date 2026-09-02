#!/usr/bin/env python
"""Materialize the AnomalyXL shards the TSLM configs read — PINNED revision.

Integrity-pass replacement for the upstream fetch_data.py (02/09/2026):

* every download is pinned to REVISION (override: TSLM_HF_REVISION) so the
  training data is a function of the code, not of when the run happened;
* a MANIFEST.json is written next to the shards with revision, per-file
  sha256 and row counts;
* the train/val/test splits are asserted ID-DISJOINT before returning —
  the local workspace was found carrying train/val files that were 100%
  subsets of the test (REPRODUCE.md Option B leak, see
  integrity/FREEZE.md); this assert makes that class of accident fatal.

Layout produced (relative to the repo root), same as upstream:

  anomalyXL/datasets/anomalyxl-train/anomalyxl-{precise,coarse}/{train,val}.parquet
  anomalyXL/datasets/anomalyxl-{precise,coarse}/data.parquet          (gold test)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO = "nz00shuuuu/AnomalyXL"
# Dataset last modified 2026-07-30: this revision covers every run of this
# project (m30, full1, Kaggle chain all fetched after that date).
REVISION = os.environ.get(
    "TSLM_HF_REVISION", "df363a467f855e55aaad414b0bdd58590829df9a")
PARTITIONS = {"precise": "Localize", "coarse": "MCQ"}
SPLITS = {"train": "train.parquet", "val": "val.parquet"}

# Expected row counts of the pinned revision; a mismatch means the download
# is not the benchmark we think it is.
EXPECTED_ROWS = {
    ("precise", "train"): 4488,
    ("precise", "val"): 520,
    ("precise", "test"): 800,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ids(path: Path) -> set:
    pf = pq.ParquetFile(path)
    if "id" not in pf.schema_arrow.names:
        return set()
    return set(pq.read_table(path, columns=["id"]).column("id").to_pylist())


def fetch(variant: str, manifest: dict) -> None:
    partition = PARTITIONS[variant]
    train_dir = REPO_ROOT / "anomalyXL/datasets/anomalyxl-train" / f"anomalyxl-{variant}"
    test_dir = REPO_ROOT / "anomalyXL/datasets" / f"anomalyxl-{variant}"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    local = {}
    for split, local_name in SPLITS.items():
        src = hf_hub_download(REPO, f"{partition}/{split}.parquet",
                              repo_type="dataset", revision=REVISION)
        shutil.copyfile(src, train_dir / local_name)
        local[split] = train_dir / local_name
        print(f"  {REPO}@{REVISION[:12]}:{partition}/{split:5s} -> {local[split]}")

    src = hf_hub_download(REPO, f"{partition}/test.parquet",
                          repo_type="dataset", revision=REVISION)
    shutil.copyfile(src, test_dir / "data.parquet")
    local["test"] = test_dir / "data.parquet"
    print(f"  {REPO}@{REVISION[:12]}:{partition}/{'test':5s} -> {local['test']}")

    # --- integrity gate -------------------------------------------------
    ids = {s: _ids(p) for s, p in local.items()}
    for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
        inter = ids[a] & ids[b]
        if inter:
            raise SystemExit(
                f"[DataIntegrity] ABORT: {variant} {a} and {b} share "
                f"{len(inter)} example IDs — splits are NOT disjoint")
    for split, path in local.items():
        rows = pq.ParquetFile(path).metadata.num_rows
        exp = EXPECTED_ROWS.get((variant, split))
        if exp is not None and rows != exp:
            raise SystemExit(
                f"[DataIntegrity] ABORT: {variant}/{split} has {rows} rows, "
                f"expected {exp} at pinned revision {REVISION[:12]}")
        manifest["files"][f"{variant}/{split}"] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "rows": rows,
            "sha256": _sha256(path),
        }
    print(f"  [DataIntegrity] {variant}: splits disjoint, counts OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["precise", "coarse", "both"],
                    default="both")
    args = ap.parse_args()
    variants = ["precise", "coarse"] if args.variant == "both" else [args.variant]
    manifest = {"repo": REPO, "revision": REVISION, "files": {}}
    for v in variants:
        fetch(v, manifest)
    out = REPO_ROOT / "anomalyXL/datasets/anomalyxl-train/MANIFEST.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"  manifest -> {out}")


if __name__ == "__main__":
    main()
