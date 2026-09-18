#!/usr/bin/env bash
# Smoke P0 sul droplet, in ordine fail-closed: test unitari nel container -> p0_smoke (backbone reale, righe reali)
# -> mini-training (2 optimizer step, 8 esempi, 5 righe di validation generativa). Ogni passo scrive un log in
# /root/wavelet-rope/logs; il primo fallimento ferma tutto. Uso: bash droplet/run_smoke.sh
set -Eeuo pipefail
LOGS=/root/wavelet-rope/logs; mkdir -p "$LOGS" /root/wavelet-rope/runs
X() { docker exec -w /workspace/code -e FLA_TILELANG=0 wrope "$@"; }
echo "== [1/3] unit tests"; X python -m pytest -q tests 2>&1 | grep -vE "redis|warnings.warn|Triton version" | tee "$LOGS/smoke_pytest.log" | tail -3
grep -qE "passed" "$LOGS/smoke_pytest.log" && ! grep -qE "failed|error" "$LOGS/smoke_pytest.log" || { echo "unit tests FAILED"; exit 1; }
echo "== [2/3] p0_smoke (real backbone, real rows)"; X python training/p0_smoke.py --config configs/p1_mi300x.toml --report /workspace/runs/p0-smoke/report.json 2>&1 | grep -vE "redis|warnings.warn|Triton version|UserWarning|return F\." | tee "$LOGS/smoke_p0.log" | tail -12
grep -q "VERDICT PASS" "$LOGS/smoke_p0.log" || { echo "p0_smoke FAILED"; exit 1; }
echo "== [3/3] mini training (2 optimizer steps)"; X python training/train.py --config configs/p1_mi300x.toml --run-id p1-smoke --resume none --max-optimizer-steps 2 --limit-examples 8 --val-subset 5 --val-full-at-pass-end 0 --max-new-tokens 64 2>&1 | grep -vE "redis|warnings.warn|Triton version|UserWarning|return F\." | tee "$LOGS/smoke_train.log" | tail -15
grep -q "max optimizer steps reached" "$LOGS/smoke_train.log" || { echo "mini training FAILED"; exit 1; }
echo "SMOKE_OK"
