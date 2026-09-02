"""Status probe for a running Colab wavelet session.

Run with:  colab exec -s <session> -f poll_train.py
Prints: DONE marker (exit code) if training finished, else RUNNING, plus the
informative tail of /content/train.log (progress, checkpoints, errors).
"""

import os
import re

LOG = "/content/train.log"
DONE = "/content/train_done.txt"
SETUP_LOG = "/content/setup.log"
SETUP_DONE = "/content/setup_done.txt"

if os.path.exists(DONE):
    with open(DONE) as f:
        print("STATE:", f.read().strip())
elif os.path.exists(LOG):
    print("STATE: TRAINING")
elif os.path.exists(SETUP_DONE):
    with open(SETUP_DONE) as f:
        print("STATE: SETUP_" + f.read().strip())
elif os.path.exists(SETUP_LOG):
    print("STATE: SETUP")
    with open(SETUP_LOG, errors="replace") as f:
        for l in f.read().split("\n")[-6:]:
            print(l[:200])
else:
    print("STATE: NOT_STARTED")

if os.path.exists(LOG):
    with open(LOG, errors="replace") as f:
        text = f.read()
    # tqdm writes \r-separated updates; keep only the last of each burst
    lines = [seg for chunk in text.split("\n") for seg in [chunk.split("\r")[-1]]]
    interesting = [
        l for l in lines
        if re.search(r"Epoch|loss|Checkpoint|TimeBudget|Error|Traceback|"
                     r"Test|Per-category|primary|Resumed|Partial|AMP|"
                     r"Train subset seed|CUDA|Summary", l)
    ]
    tail = interesting[-25:] if interesting else lines[-10:]
    for l in tail:
        print(l[:250])
else:
    print("no train.log yet")
