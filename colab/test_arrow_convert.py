"""Test on-VM: small-batch parquet -> Arrow IPC stream + Dataset.from_file.

Validates the low-RAM loading path before patching qa_dataset.py with it.
"""

import os
import resource
import time

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset


def convert(path, batch_size=16):
    arrow_path = path + ".mmap.arrow"
    tmp = arrow_path + ".tmp"
    pf = pq.ParquetFile(path)
    t0 = time.time()
    with pa.OSFile(tmp, "wb") as sink:
        with pa.ipc.new_stream(sink, pf.schema_arrow) as writer:
            n = 0
            for batch in pf.iter_batches(batch_size=batch_size):
                writer.write_batch(batch)
                n += batch.num_rows
    os.replace(tmp, arrow_path)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"{os.path.basename(path)}: {n} rows in {time.time()-t0:.0f}s, "
          f"arrow {os.path.getsize(arrow_path)/1e9:.2f} GB, peak RSS {peak:.1f} GB")
    return arrow_path


base = "/content/timerlm/anomalyXL/datasets"
for rel in ("anomalyxl-train/anomalyxl-precise/val.parquet",
            "anomalyxl-train/anomalyxl-precise/train.parquet",
            "anomalyxl-precise/data.parquet"):
    p = os.path.join(base, rel)
    ap = convert(p)
    ds = Dataset.from_file(ap)
    row = ds[0]
    print(f"  from_file OK: len={len(ds)}, cols={sorted(row.keys())[:5]}...")
print("ALL CONVERSIONS OK")
