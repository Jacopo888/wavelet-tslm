"""Detach the full setup+train pipeline on the Colab VM and return at once.

Run with:  colab exec -s <session> -f bootstrap_detached.py
Expects /content/colab_setup_train.py already uploaded (plus its inputs).
Setup log -> /content/setup.log, marker /content/setup_done.txt; training then
logs to /content/train.log with marker /content/train_done.txt.
"""

import os
import subprocess

for stale in ("setup_done.txt", "train_done.txt"):
    p = os.path.join("/content", stale)
    if os.path.exists(p):
        os.remove(p)

cmd = ("python /content/colab_setup_train.py > /content/setup.log 2>&1; "
       "echo EXIT:$? > /content/setup_done.txt")
proc = subprocess.Popen(["bash", "-c", cmd], start_new_session=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("SETUP LAUNCHED pid=", proc.pid)
