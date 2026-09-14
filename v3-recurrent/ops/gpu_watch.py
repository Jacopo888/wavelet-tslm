"""GPU stock watcher + grab for the v3 (recurrent perceiver) training.

Ordine Jacopo 05/09: watcher su Vultr (A40 24GB o altra slice utile del
nostro account) e su AMD Developer Cloud (MI300X, qualunque regione);
appena c'e' stock, PRENDERE la GPU (create) e fermarsi.

Gira su Windows (python 3.11), detached (ops/start_watch.ps1). Stato e log
in %LOCALAPPDATA%\\wavelet-v3\\ :
  gpu_watch.log   righe con timestamp UTC (TRIGGER / GRABBED / CAP_BLOCKED /
                  ERROR / HEARTBEAT); il Monitor di sessione fa tail di questo.
  winner.json     scritto SOLO dopo un create riuscito (lock: mai 2 GPU).
  gpu_watch.pid   pid del watcher.

Vultr: stock dal listino pubblico (GET /v2/plans?type=vcg, nessuna chiave);
il create passa da WSL (`vultr-cli` + chiave in ~/.vultr_api_key letta dal
processo, MAI stampata). DigitalOcean: doctl.exe autenticato ("My AMD Team").

Cap Vultr (lezione 02-03/09): il limite mensile e' la SOMMA dei canoni attivi
(wavelet-l1 = $625/mo gia' dentro); "maximum monthly fee limit" -> il piano
va in cooldown 10 min e si continua a guardare. La 16GB (vcg-a40-8c) e le
A16 sono ESCLUSE: il 4B con LoRA + 16 canali x 256 token non ci sta / troppo
lente (memoria 29/08). Priorita' a parita' di tick: Vultr (stack CUDA gia'
validato da va1), poi MI300X (ROCm da validare con smoke).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

POLL_S = int(os.environ.get("V3_WATCH_POLL_S", "15"))
HEARTBEAT_S = 1800
CAP_COOLDOWN_S = 600
ERR_COOLDOWN_S = 120

STATE_DIR = Path(os.environ.get("LOCALAPPDATA", ".")) / "wavelet-v3"
LOG = STATE_DIR / "gpu_watch.log"
WINNER = STATE_DIR / "winner.json"
PIDFILE = STATE_DIR / "gpu_watch.pid"

# ---- Vultr ---------------------------------------------------------------
VULTR_PLANS = [  # ordine di preferenza; tutti >=24GB, canone <= ~$2.3k/mo
    "vcg-a40-12c-60g-24vram",      # 24GB  $625/mo  $0.86/h  (stack va1)
    "vcg-l40s-16c-180g-48vram",    # 48GB  $1123/mo $1.54/h  (cap incerto)
    "vcg-l40s-32c-375g-96vram",    # 96GB  $2246/mo (cap: quasi certo NO)
]
VULTR_OS = "2284"                  # Ubuntu 24.04: driver vGPU gia' installato
VULTR_SSH_KEY = "aa986770-cd68-4dfc-8126-504b65185801"   # wavelet-agent
VULTR_LABEL = "wavelet-v3"
VULTR_PLANS_URL = "https://api.vultr.com/v2/plans?type=vcg&per_page=500"

# ---- DigitalOcean (AMD Developer Cloud) ----------------------------------
DOCTL = r"C:\Users\jacop\AppData\Local\Programs\doctl\doctl.exe"
DO_SIZE = "gpu-mi300x1-192gb"       # $2.59/h; crediti AMD validi SOLO MI300X
DO_IMAGE = "digitaloceanai-rocmpytorchjupyt"   # PyTorch 2.6 ROCm preinstallato
DO_NAME = "wavelet-v3-mi300x"
DO_SSH_KEY_NAMES = ("wavelet-agent", "airway-agent")


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(line: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{now()} {line}\n")
    print(f"{now()} {line}", flush=True)


def run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    except Exception as e:  # noqa: BLE001
        return 1, f"EXC {e!r}"


def wsl(cmd: str, timeout: int = 120) -> tuple[int, str]:
    # La chiave e' esportata DENTRO WSL dal processo; non transita nel log.
    return run(["wsl", "-e", "bash", "-lc",
                "export VULTR_API_KEY=$(cat ~/.vultr_api_key); " + cmd], timeout)


# ---- stock ---------------------------------------------------------------
def vultr_stock() -> dict[str, list[str]]:
    req = urllib.request.Request(VULTR_PLANS_URL, headers={"User-Agent": "wavelet-v3-watch"})
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.load(r)
    return {p["id"]: list(p.get("locations") or []) for p in d.get("plans", [])}


def do_stock() -> list[str]:
    rc, out = run([DOCTL, "compute", "size", "list", "-o", "json"], 25)
    if rc != 0:
        raise RuntimeError(f"doctl size list rc={rc}: {out[:200]}")
    for s in json.loads(out):
        if s.get("slug") == DO_SIZE:
            return list(s.get("regions") or [])
    return []


# ---- grab ----------------------------------------------------------------
def vultr_existing() -> str | None:
    rc, out = wsl("vultr-cli instance list")
    for line in out.splitlines():
        if f"\t{VULTR_LABEL}\t" in line or f" {VULTR_LABEL} " in line:
            return line
    return None


def grab_vultr(plan: str, region: str) -> tuple[str, str]:
    """-> ("GRABBED", ip) | ("CAP", msg) | ("ERROR", msg)"""
    if vultr_existing():
        return "ERROR", "istanza wavelet-v3 gia' presente: non creo doppioni"
    rc, out = wsl(
        f"vultr-cli instance create --region {region} --plan {plan} --os {VULTR_OS} "
        f"--label {VULTR_LABEL} --host {VULTR_LABEL} --ssh-keys {VULTR_SSH_KEY}", 180)
    low = out.lower()
    if "monthly fee limit" in low:
        return "CAP", out.strip()[:300]
    if rc != 0 or "error" in low or "failed" in low:
        return "ERROR", out.strip()[:300]
    log(f"CREATE_OK vultr {plan} {region}; attendo IP")
    for _ in range(40):
        time.sleep(15)
        row = vultr_existing() or ""
        cols = row.split("\t")
        cols = [c for c in cols if c != ""]
        if len(cols) >= 6:
            ip, status = cols[1].strip(), cols[4].strip()
            if status == "active" and ip and ip != "0.0.0.0":
                return "GRABBED", ip
    return "ERROR", "create riuscito ma nessun IP attivo entro 10 min (istanza ESISTE: controllare)"


def do_key_ids() -> str:
    rc, out = run([DOCTL, "compute", "ssh-key", "list", "--format", "ID,Name", "--no-header"], 60)
    ids = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in DO_SSH_KEY_NAMES:
            ids.append(parts[0])
    return ",".join(ids)


def grab_do(region: str) -> tuple[str, str]:
    rc, out = run([DOCTL, "compute", "droplet", "list", "--format", "Name,PublicIPv4", "--no-header"], 60)
    if any(l.split() and l.split()[0] == DO_NAME for l in out.splitlines()):
        return "ERROR", "droplet wavelet-v3-mi300x gia' presente: non creo doppioni"
    keys = do_key_ids()
    if not keys:
        return "ERROR", "nessuna chiave SSH wavelet-agent/airway-agent su DO"
    rc, out = run([DOCTL, "compute", "droplet", "create", DO_NAME, "--size", DO_SIZE,
                   "--image", DO_IMAGE, "--region", region, "--ssh-keys", keys,
                   "--wait", "--format", "ID,Name,PublicIPv4,Status"], 900)
    low = out.lower()
    if rc != 0 or "error" in low or "not available" in low or "capacity" in low:
        return "ERROR", out.strip()[:300]
    for _ in range(20):
        rc, out = run([DOCTL, "compute", "droplet", "list", "--format", "Name,PublicIPv4,Status", "--no-header"], 60)
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] == DO_NAME and parts[2] == "active" and parts[1] not in ("", "0.0.0.0"):
                return "GRABBED", parts[1]
        time.sleep(15)
    return "ERROR", "droplet creato ma senza IP attivo entro 5 min (ESISTE: controllare)"


def write_winner(provider: str, plan: str, region: str, ip: str) -> None:
    WINNER.write_text(json.dumps({
        "provider": provider, "plan": plan, "region": region, "ip": ip,
        "label": VULTR_LABEL if provider == "vultr" else DO_NAME,
        "grabbed_at": now(), "purpose": "wavelet-tslm v3 recurrent ablations",
    }, indent=1), encoding="utf-8")


# ---- main loop -----------------------------------------------------------
def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if WINNER.exists():
        log(f"EXIT winner.json gia' presente ({WINNER.read_text()[:120]!r}); niente da fare")
        return 0
    PIDFILE.write_text(str(os.getpid()))
    log(f"START poll={POLL_S}s vultr={VULTR_PLANS} do={DO_SIZE} pid={os.getpid()}")
    cooldown: dict[str, float] = {}
    last_hb = 0.0
    last_stock_sig = None
    while True:
        t = time.time()
        vs, ds = {}, []
        try:
            vs = vultr_stock()
        except Exception as e:  # noqa: BLE001
            log(f"WARN vultr stock: {e!r}")
        try:
            ds = do_stock()
        except Exception as e:  # noqa: BLE001
            log(f"WARN do stock: {e!r}")
        sig = json.dumps({"v": {p: vs.get(p, []) for p in VULTR_PLANS}, "d": ds}, sort_keys=True)
        if sig != last_stock_sig:
            log(f"STOCK {sig}")
            last_stock_sig = sig
        if t - last_hb > HEARTBEAT_S:
            log(f"HEARTBEAT ok {sig}")
            last_hb = t

        # Vultr prima (stack validato), poi DO.
        for plan in VULTR_PLANS:
            regs = vs.get(plan, [])
            if not regs or cooldown.get(plan, 0) > t:
                continue
            region = regs[0]
            log(f"TRIGGER vultr {plan} regioni={regs} -> create in {region}")
            st, msg = grab_vultr(plan, region)
            if st == "GRABBED":
                write_winner("vultr", plan, region, msg)
                log(f"GRABBED vultr {plan} {region} ip={msg}")
                return 0
            cooldown[plan] = t + (CAP_COOLDOWN_S if st == "CAP" else ERR_COOLDOWN_S)
            log(f"{'CAP_BLOCKED' if st == 'CAP' else 'ERROR'} vultr {plan} {region}: {msg}")
        if ds and cooldown.get("do", 0) <= t:
            region = ds[0]
            log(f"TRIGGER do {DO_SIZE} regioni={ds} -> create in {region}")
            st, msg = grab_do(region)
            if st == "GRABBED":
                write_winner("do", DO_SIZE, region, msg)
                log(f"GRABBED do {DO_SIZE} {region} ip={msg}")
                return 0
            cooldown["do"] = t + ERR_COOLDOWN_S
            log(f"ERROR do {DO_SIZE} {region}: {msg}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("EXIT interrupted")
    except Exception as e:  # noqa: BLE001
        log(f"EXIT crash {e!r}")
        raise
