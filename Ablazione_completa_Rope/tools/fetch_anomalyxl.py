#!/usr/bin/env python3
"""Fetch the AnomalyXL parquet files at the pinned revision and verify them byte-for-byte
against integrity/artifacts/va1/RUN_MANIFEST.json (sha256 + row counts).

At revision df363a46 the dataset repo holds Localize/{train,val,test}.parquet and
MCQ/{train,val,test}.parquet; the "precise" variant used by every run of this project is the
Localize family (fetch_data.py of the va1 kit). Files are identified by sha256, never by name.
Writes MANIFEST_VERIFIED.json next to the files. Exit 1 if anything is missing or differs."""
import hashlib, json, os, sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

REPO, REV = "nz00shuuuu/AnomalyXL", "df363a467f855e55aaad414b0bdd58590829df9a"
WANT = {  # split -> (sha256, rows) from integrity/artifacts/va1/RUN_MANIFEST.json
    "train": ("b8571d886772a55613c2e8851a774e6a561e19c8272227f5abb70e2d76e70b7d", 4488),
    "validation": ("90b805b8f40e203b76873a9cf3a0f986e21700c9185c74acc33c44a183b1d977", 520),
    "test": ("7b37269d106f70ba7bee5feb504c00059bc880217a56a8279a2631c8bf5e1524", 800),
}
OUT = Path(os.environ.get("ANOMALYXL_DIR", f"/scratch/datasets/anomalyxl_{REV}"))
OUT.mkdir(parents=True, exist_ok=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


files = HfApi().list_repo_files(REPO, repo_type="dataset", revision=REV)
parquet = [f for f in files if f.endswith(".parquet")]
print(f"{len(files)} files at {REPO}@{REV[:12]}; parquet: {parquet}")
# Localize first (expected family), then everything else only if something is still missing
candidates = sorted(parquet, key=lambda f: (not f.startswith("Localize/"), f))
found, seen = {}, {}
for f in candidates:
    if all(k in found for k in WANT):
        break
    local = hf_hub_download(REPO, f, repo_type="dataset", revision=REV, local_dir=str(OUT))
    digest, size = sha256(local), os.path.getsize(local)
    rows = pq.ParquetFile(local).metadata.num_rows
    match = [k for k, (s, _) in WANT.items() if s == digest]
    seen[f] = {"bytes": size, "sha256": digest, "rows": rows, "matches": match}
    print(f"  {f}: {size} bytes rows={rows} sha256={digest[:16]}... -> {match or 'no match'}", flush=True)
    for k in match:
        found[k] = {"repo_path": f, "local_path": local, "sha256": digest, "bytes": size, "rows": rows,
                    "rows_ok": rows == WANT[k][1]}
missing = [k for k in WANT if k not in found]
bad_rows = [k for k, v in found.items() if not v["rows_ok"]]
manifest = {"repo": REPO, "revision": REV, "repo_type": "dataset", "files": found, "missing": missing,
            "row_count_mismatch": bad_rows, "downloaded": seen}
(OUT / "MANIFEST_VERIFIED.json").write_text(json.dumps(manifest, indent=2))
ok = not missing and not bad_rows
print("DATA_VERIFIED" if ok else f"DATA_FAIL missing={missing} bad_rows={bad_rows}")
sys.exit(0 if ok else 1)
