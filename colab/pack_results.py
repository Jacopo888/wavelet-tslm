"""Pack a finished Colab session's results for download.

Run with:  colab exec -s <session> -f pack_results.py
Creates /content/results_out.tar.gz containing the whole results tree
(metrics, history, logs, checkpoints) - then fetch it locally with:
  colab download -s <session> /content/results_out.tar.gz <local path>
"""

import os
import subprocess

RESULTS = "/content/timerlm/sft/tslm/results"
OUT = "/content/results_out.tar.gz"

if not os.path.isdir(RESULTS):
    raise SystemExit(f"no results dir at {RESULTS}")

if os.path.exists(OUT):
    os.remove(OUT)
subprocess.run(["tar", "-czf", OUT, "-C", RESULTS, "."], check=True)
size = os.path.getsize(OUT) / 1e6
print(f"packed {OUT} ({size:.1f} MB)")
for root, _dirs, files in os.walk(RESULTS):
    for f in files:
        p = os.path.join(root, f)
        print(f"  {os.path.relpath(p, RESULTS)}  {os.path.getsize(p)/1e6:.1f} MB")
