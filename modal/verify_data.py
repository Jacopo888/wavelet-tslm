"""Provenance verification for AnomalyXL — CPU-only, no GPU, no training.

Closes the data-provenance chain of the 02/09 integrity pass by comparing
CONTENT-canonical digests (parquet bytes differ across writers) of:

  1. the pinned official HF shards (Localize/{train,val,test} @ REVISION);
  2. the volume caches actually consumed by the m30/full1 runs
     (/vol/data/gold-precise, /vol/data/anomalyxl-train);
  3. (locally, --local mode) the seed-42 generated test on the workstation.

Also asserts train/val/test ID-disjointness on the real pinned data.

Run remote:  python -m modal run modal/verify_data.py
Run local:   python modal/verify_data.py --local <path/to/data.parquet>
"""

import json
import sys

REVISION = "df363a467f855e55aaad414b0bdd58590829df9a"
REPO_ID = "nz00shuuuu/AnomalyXL"


def canonical_digests(parquet_path: str) -> dict:
    """Order-independent content digests: ids, id+gold answers, and a
    float64-sum fingerprint of the series. Streams row batches and keeps
    the series values inside Arrow/NumPy (a 16ch x 131k row materialized
    as Python objects OOMs even at batch_size 64). The fingerprint is
    rounded to 6 significant digits so summation order can't flip it."""
    import hashlib

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    names = pf.schema_arrow.names
    cols = [c for c in ("id", "answer", "channel_values") if c in names]
    ids, ans, ser = [], [], []
    for rb in pf.iter_batches(batch_size=16, columns=cols):
        rb_ids = (rb.column("id").to_pylist() if "id" in cols
                  else [""] * rb.num_rows)
        rb_ans = rb.column("answer").to_pylist() if "answer" in cols else None
        cv = rb.column("channel_values") if "channel_values" in cols else None
        n_ch = pc.list_value_length(cv).to_pylist() if cv is not None else None
        for i in range(rb.num_rows):
            rid = str(rb_ids[i])
            ids.append(rid)
            if rb_ans is not None:
                ans.append(rid + "\t" + str(rb_ans[i]))
            if cv is not None:
                row = cv.slice(i, 1)
                flat = pc.list_flatten(pc.list_flatten(row))
                tot = pc.sum(pc.cast(flat, pa.float64()), min_count=0).as_py()
                ser.append(f"{rid}\t{n_ch[i]}\t{float(tot or 0.0):.6g}")

    def h(lines):
        m = hashlib.sha256()
        for line in sorted(lines):
            m.update(line.encode("utf-8"))
            m.update(b"\n")
        return m.hexdigest()

    return {
        "rows": len(ids),
        "ids_sha": h(ids),
        "answers_sha": h(ans) if ans else None,
        "series_sha": h(ser) if ser else None,
        "id_set": ids,
    }


def _strip(d):
    return {k: v for k, v in d.items() if k != "id_set"}


def _verify(report_ids=True):
    import os

    from huggingface_hub import hf_hub_download

    out = {"repo": REPO_ID, "revision": REVISION, "shards": {}}
    idsets = {}

    for split in ("test", "val", "train"):
        p = hf_hub_download(REPO_ID, f"Localize/{split}.parquet",
                            repo_type="dataset", revision=REVISION)
        d = canonical_digests(p)
        idsets[f"hf_{split}"] = set(d["id_set"])
        out["shards"][f"hf_Localize_{split}"] = _strip(d)
        print(f"hf {split}: rows={d['rows']} ids={d['ids_sha'][:16]} "
              f"answers={d['answers_sha'][:16]}", flush=True)

    for label, path in (
        ("vol_gold_test", "/vol/data/gold-precise/data.parquet"),
        ("vol_train", "/vol/data/anomalyxl-train/anomalyxl-precise/train.parquet"),
        ("vol_val", "/vol/data/anomalyxl-train/anomalyxl-precise/val.parquet"),
    ):
        if os.path.exists(path):
            d = canonical_digests(path)
            idsets[label] = set(d["id_set"])
            out["shards"][label] = _strip(d)
            print(f"{label}: rows={d['rows']} ids={d['ids_sha'][:16]} "
                  f"answers={d['answers_sha'][:16]}", flush=True)
        else:
            out["shards"][label] = "MISSING"
            print(f"{label}: MISSING at {path}", flush=True)

    # Disjointness on the pinned official data — the assert that should
    # have existed from day one.
    checks = {}
    checks["hf_train_x_test"] = len(idsets["hf_train"] & idsets["hf_test"])
    checks["hf_val_x_test"] = len(idsets["hf_val"] & idsets["hf_test"])
    checks["hf_train_x_val"] = len(idsets["hf_train"] & idsets["hf_val"])
    if "vol_gold_test" in idsets:
        checks["volgold_x_hftest_equal_ids"] = (
            idsets["vol_gold_test"] == idsets["hf_test"])
    if "vol_train" in idsets:
        checks["voltrain_x_hftest"] = len(idsets["vol_train"] & idsets["hf_test"])
    out["checks"] = checks
    print("CHECKS:", json.dumps(checks), flush=True)

    ok = (checks["hf_train_x_test"] == 0 and checks["hf_val_x_test"] == 0
          and checks["hf_train_x_val"] == 0)
    out["disjointness_ok"] = ok
    print("VERIFY_RESULT:", json.dumps({k: v for k, v in out.items()
                                        if k != "shards"}), flush=True)
    return out


try:
    import modal

    app = modal.App("wavelet-verify-data")
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("pyarrow", "numpy", "huggingface-hub"))
    vol = modal.Volume.from_name("wavelet-vol", create_if_missing=False)

    @app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=8192,
                  timeout=3600)
    def verify():
        return _verify()

    @app.local_entrypoint()
    def main():
        out = verify.remote()
        with open("integrity/PROVENANCE-VERIFY.json", "w") as f:
            json.dump(out, f, indent=1)
        print("saved -> integrity/PROVENANCE-VERIFY.json")
except ImportError:
    pass

if __name__ == "__main__" and "--local" in sys.argv:
    path = sys.argv[sys.argv.index("--local") + 1]
    d = canonical_digests(path)
    print(json.dumps(_strip(d), indent=1))
