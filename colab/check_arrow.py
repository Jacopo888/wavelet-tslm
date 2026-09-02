import glob
import os
import subprocess

for p in sorted(glob.glob("/content/timerlm/anomalyXL/datasets/**/*.arrow*",
                          recursive=True)):
    print(p, f"{os.path.getsize(p)/1e9:.2f} GB")

print(subprocess.run(["free", "-m"], capture_output=True, text=True).stdout)

import pyarrow.parquet as pq
base = "/content/timerlm/anomalyXL/datasets"
for rel in ("anomalyxl-train/anomalyxl-precise/val.parquet",
            "anomalyxl-train/anomalyxl-precise/train.parquet",
            "anomalyxl-precise/data.parquet"):
    pf = pq.ParquetFile(os.path.join(base, rel))
    m = pf.metadata
    rg_rows = [m.row_group(i).num_rows for i in range(m.num_row_groups)]
    print(rel, "rows:", m.num_rows, "row_groups:", m.num_row_groups,
          "max_rg:", max(rg_rows), "size GB:",
          round(os.path.getsize(os.path.join(base, rel)) / 1e9, 2))
