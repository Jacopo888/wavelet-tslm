#!/usr/bin/env python3
"""Download the pinned Qwen3.5-4B snapshot (weights shards + config + tokenizer + chat template only) into the
HF hub cache and verify every downloaded file against the sha256/size published by the Hub for that revision.
Writes /scratch/hf/qwen35_4b_<rev>_VERIFIED.json. Exit 1 on any mismatch."""
import fnmatch, hashlib, json, sys, time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

REPO, REV = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ALLOW = ["*.json", "*.txt", "*.jinja", "model.safetensors*", "tokenizer*"]  # no README/LICENSE/images


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


t0 = time.time()
path = snapshot_download(REPO, revision=REV, allow_patterns=ALLOW)
print(f"snapshot at {path} after {time.time() - t0:.0f}s", flush=True)
info = HfApi().model_info(REPO, revision=REV, files_metadata=True)
report = {"repo": REPO, "revision": REV, "snapshot_path": path, "allow_patterns": ALLOW, "files": {}, "ok": True}
total = 0
for s in info.siblings:
    local = Path(path) / s.rfilename
    wanted = any(fnmatch.fnmatch(s.rfilename, pat) for pat in ALLOW)
    if not local.exists():
        report["files"][s.rfilename] = {"status": "MISSING" if wanted else "SKIPPED_BY_PATTERN"}
        report["ok"] &= not wanted
        continue
    size = local.stat().st_size
    entry = {"bytes": size, "expected_bytes": s.size, "status": "OK"}
    lfs = s.lfs
    exp_sha = (getattr(lfs, "sha256", None) or (lfs.get("sha256") if isinstance(lfs, dict) else None)) if lfs else None
    if exp_sha:
        got = sha256(local)
        entry.update({"sha256": got, "expected_sha256": exp_sha})
        if got != exp_sha:
            entry["status"] = "SHA_MISMATCH"
            report["ok"] = False
    elif s.size is not None and size != s.size:
        entry["status"] = "SIZE_MISMATCH"
        report["ok"] = False
    total += size
    report["files"][s.rfilename] = entry
report["total_bytes"] = total
out = Path(f"/scratch/hf/qwen35_4b_{REV[:12]}_VERIFIED.json")
out.write_text(json.dumps(report, indent=2))
print(f"total {total / 2**30:.2f} GiB; report {out}")
print("QWEN_VERIFIED" if report["ok"] else "QWEN_VERIFY_FAILED")
sys.exit(0 if report["ok"] else 1)
