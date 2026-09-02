import os
import subprocess

print("== free -m ==")
print(subprocess.run(["free", "-m"], capture_output=True, text=True).stdout)
print("== dmesg OOM ==")
d = subprocess.run(["bash", "-c", "dmesg 2>/dev/null | grep -i -E 'oom|killed process' | tail -5"],
                   capture_output=True, text=True)
print(d.stdout or "(dmesg empty/not permitted)")

for name in ("setup.log", "train.log"):
    p = f"/content/{name}"
    print(f"== tail {name} ==")
    if os.path.exists(p):
        with open(p, errors="replace") as f:
            lines = [seg for chunk in f.read().split("\n")
                     for seg in [chunk.split("\r")[-1]] if seg.strip()]
        for l in lines[-25:]:
            print(l[:220])
    else:
        print("(missing)")
