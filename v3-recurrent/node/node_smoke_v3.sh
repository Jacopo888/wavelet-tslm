#!/usr/bin/env bash
# End-to-end v3 gate for the Vultr A40-24Q. Runs untied first, then tied,
# through train.py, validates each slim checkpoint, and performs eval-only.
# No network, tmux, partial training budget, or full-coverage env is used.
set -Eeuo pipefail

ROOT="$HOME/timerlm/sft/tslm"
PY="$HOME/venv/bin/python"
LOG_DIR="$HOME/logs"
DATA_ROOT="$HOME/timerlm/anomalyXL/datasets"
SMOKE_DATA_DIR="$DATA_ROOT/anomalyxl-v3smoke"
SOURCE_DIR="$DATA_ROOT/anomalyxl-train/anomalyxl-precise"
RUN_BASE="$ROOT/results/anomaly_xl_precise"
RUN_DIRS=(
  "$RUN_BASE/anomaly_xl_wavelet_v3smoke_untied"
  "$RUN_BASE/anomaly_xl_wavelet_v3smoke_tied"
)
OK_FILE="$LOG_DIR/v3smoke.ok"
FAILED_FILE="$LOG_DIR/v3smoke.FAILED"
SUBSET_MANIFEST="$LOG_DIR/v3smoke.subset.json"
PEAK_LIMIT_MIB=23000
TOTAL_WALL_S=1800
TRAIN_ROWS=4
GPU="${V3_GPU:-0}"
START_WALL=$(date +%s)
CURRENT_STAGE="startup"
SAMPLER_PID=""

mkdir -p "$LOG_DIR"

stop_sampler() {
  if [ -n "$SAMPLER_PID" ]; then
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    SAMPLER_PID=""
  fi
}

cleanup_artifacts() {
  stop_sampler
  rm -rf -- "$SMOKE_DATA_DIR"
  rm -rf -- "${RUN_DIRS[@]}"
}

fail() {
  local reason="$*"
  set +e
  cleanup_artifacts
  rm -f "$OK_FILE"
  V3SMOKE_REASON="$reason" V3SMOKE_STAGE="$CURRENT_STAGE" \
    python3 - "$FAILED_FILE" <<'PYFAIL'
import json
import os
import sys
import time

doc = {
    "status": "FAILED",
    "stage": os.environ.get("V3SMOKE_STAGE", "unknown"),
    "reason": os.environ.get("V3SMOKE_REASON", "unknown"),
    "unix_time": int(time.time()),
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(doc, handle, indent=2)
    handle.write("\n")
print(json.dumps(doc))
PYFAIL
  echo "SMOKE_FAILED: $reason" >&2
  exit 1
}

on_err() {
  local rc=$?
  local line="$1"
  fail "unexpected command failure at line $line (exit $rc)"
}
trap 'on_err $LINENO' ERR
trap cleanup_artifacts EXIT

run_bounded() {
  local elapsed remaining
  elapsed=$(( $(date +%s) - START_WALL ))
  remaining=$(( TOTAL_WALL_S - elapsed ))
  if [ "$remaining" -le 0 ]; then
    fail "total wall limit of ${TOTAL_WALL_S}s already exhausted"
  fi
  timeout --signal=TERM --kill-after=30s "${remaining}s" "$@"
}

start_sampler() {
  local csv="$1"
  : > "$csv"
  nvidia-smi -i "$GPU" --query-gpu=memory.used \
    --format=csv,noheader,nounits -l 2 > "$csv" 2>/dev/null &
  SAMPLER_PID=$!
}

peak_of() {
  awk '
    $1 ~ /^[0-9]+$/ { seen=1; if ($1 > peak) peak=$1 }
    END { if (seen) print peak }
  ' "$1"
}

# Remove only dedicated stale smoke state. A stale run dir could auto-resume.
rm -f "$OK_FILE" "$FAILED_FILE" "$SUBSET_MANIFEST" \
  "$LOG_DIR"/v3smoke.v3*.json \
  "$LOG_DIR"/v3smoke.v3*.checkpoint.json \
  "$LOG_DIR"/v3smoke.v3*.eval.json \
  "$LOG_DIR"/v3smoke.v3*.train.log \
  "$LOG_DIR"/v3smoke.v3*.eval.log \
  "$LOG_DIR"/v3smoke.v3*.memory.csv
cleanup_artifacts

[ -x "$PY" ] || fail "missing executable $PY"
[ -f "$HOME/RUN_MANIFEST_V3.json" ] || fail "missing ~/RUN_MANIFEST_V3.json; run node_env_v3.sh"
[ -f "$ROOT/train.py" ] || fail "missing $ROOT/train.py"
grep -q 'TSLM_STEP_TIMING' "$ROOT/train.py" || \
  fail "deployed train.py lacks the smoke timing hook; rerun node_env_v3.sh after ship_v3.sh"
for cfg in wavelet_precise_v3smoke_untied.yaml wavelet_precise_v3smoke_tied.yaml; do
  [ -f "$ROOT/configs/$cfg" ] || \
    fail "missing configs/$cfg; rerun node_env_v3.sh after ship_v3.sh"
done

# Native allocator only: inherited expandable_segments is invalid on GRID vGPU.
unset PYTORCH_CUDA_ALLOC_CONF
unset TSLM_REQUIRE_FULL_COVERAGE TSLM_MAX_TRAIN_STEPS TSLM_MAX_TRAIN_SECONDS
unset TSLM_RESUME_FROM TSLM_RESUME_WEIGHTS_ONLY TSLM_V3_DEPTH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

CURRENT_STAGE="gpu_preflight"
GPU_NAME=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader | head -1)
GPU_TOTAL=$(nvidia-smi -i "$GPU" --query-gpu=memory.total \
  --format=csv,noheader,nounits | head -1 | tr -d ' ')
case "$GPU_NAME" in
  *A40*) ;;
  *) fail "expected an A40 device at index $GPU, got: $GPU_NAME" ;;
esac
case "$GPU_TOTAL" in
  ''|*[!0-9]*) fail "could not parse GPU memory total: $GPU_TOTAL" ;;
esac
[ "$GPU_TOTAL" -ge "$PEAK_LIMIT_MIB" ] || \
  fail "GPU has ${GPU_TOTAL} MiB, below smoke peak limit ${PEAK_LIMIT_MIB} MiB"
GPU_PROCS=$(nvidia-smi -i "$GPU" --query-compute-apps=pid \
  --format=csv,noheader,nounits 2>/dev/null || true)
if printf '%s\n' "$GPU_PROCS" | grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$'; then
  fail "GPU $GPU already has compute processes: $GPU_PROCS"
fi
echo "GPU_PREFLIGHT_OK name=$GPU_NAME total_mib=$GPU_TOTAL limit_mib=$PEAK_LIMIT_MIB"

CURRENT_STAGE="anti_leakage_preflight"
cd "$ROOT"
"$PY" preflight_data.py "$DATA_ROOT" --variant precise

CURRENT_STAGE="v1_identity_preflight"
"$PY" - <<'PYIDENTITY'
import torch

from src.models.wavelet_encoder import WaveletTSEncoder
from src.models.wavelet_encoder_v3 import RecurrentWaveletTSEncoder

common = dict(
    output_dim=32, num_tokens=256, max_levels=3, coarse_blocks=8,
    n_pos_freqs=4,
)
for tied in (True, False):
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(17)
        v1 = WaveletTSEncoder(**common).to(dtype).eval()
        torch.manual_seed(17)
        v3 = RecurrentWaveletTSEncoder(
            **common, level_topm=32, tied=tied, n_heads=4, mlp_ratio=2,
            context_scale_init=0.0, locality_sigma=0.0078125,
        ).to(dtype).eval()
        for length in (7, 256):
            torch.manual_seed(length)
            x = torch.randn(2, length).to(dtype)
            with torch.no_grad():
                expected = v1(x)
                actual = v3(x)
            assert torch.equal(actual, expected), (
                tied, dtype, length,
                (actual.float() - expected.float()).abs().max().item(),
            )
print("V3_INIT_EQUALS_V1")
PYIDENTITY

QWEN_REV=$("$PY" -c \
  "import json,os; print(json.load(open(os.path.expanduser('~/RUN_MANIFEST_V3.json')))['qwen_revision'])")
[ -n "$QWEN_REV" ] || fail "empty qwen_revision in RUN_MANIFEST_V3.json"

CURRENT_STAGE="build_deterministic_subset"
"$PY" - "$SOURCE_DIR" "$SMOKE_DATA_DIR" "$SUBSET_MANIFEST" "$TRAIN_ROWS" <<'PYSUBSET'
import json
import shutil
import sys
from pathlib import Path

from datasets import Dataset

source, out_dir, manifest_path, train_rows = sys.argv[1:5]
source = Path(source)
out_dir = Path(out_dir)
train_rows = int(train_rows)
assert train_rows >= 3
train_path = source / "train.parquet"
val_path = source / "val.parquet"
assert train_path.is_file(), train_path
assert val_path.is_file(), val_path

if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)

train = Dataset.from_parquet(str(train_path))
cols = {
    name: list(train[name])
    for name in ("id", "category", "length", "n_channels")
}
records = [
    {
        "source_index": i,
        "id": str(cols["id"][i]),
        "category": str(cols["category"][i]),
        "length": int(cols["length"][i]),
        "n_channels": int(cols["n_channels"][i]),
    }
    for i in range(len(train))
]
stable = lambda row: (row["id"], row["source_index"])
hard16_candidates = [
    row for row in records
    if row["category"] == "localize_all_channels" and row["n_channels"] == 16
]
assert hard16_candidates, "no localize_all_channels row with n_channels == 16"
hard16_max_length = max(row["length"] for row in hard16_candidates)
hard16 = min(
    (row for row in hard16_candidates if row["length"] == hard16_max_length),
    key=stable,
)
hard131 = (
    hard16
    if hard16["length"] == 131072
    else min((row for row in records if row["length"] == 131072), key=stable)
)

selected = []
for row in (hard16, hard131):
    if row["source_index"] not in selected:
        selected.append(row["source_index"])
fillers = sorted(
    records,
    key=lambda row: (
        row["length"], row["n_channels"], row["id"], row["source_index"]
    ),
)
for row in fillers:
    if row["source_index"] not in selected:
        selected.append(row["source_index"])
    if len(selected) == train_rows:
        break
assert len(selected) == train_rows, selected

row16 = train[hard16["source_index"]]
row131 = train[hard131["source_index"]]
assert row16["category"] == "localize_all_channels"
assert int(row16["n_channels"]) == 16
assert int(row16["length"]) == hard16_max_length
assert len(row16["channel_values"]) == 16
assert all(
    len(channel) == int(row16["length"])
    for channel in row16["channel_values"]
)
assert int(row131["length"]) == 131072
assert len(row131["channel_values"]) >= 1
assert all(len(channel) == 131072 for channel in row131["channel_values"])

train.select(selected).to_parquet(str(out_dir / "train.parquet"))

val = Dataset.from_parquet(str(val_path))
val_ids = list(val["id"])
val_lengths = list(val["length"])
val_channels = list(val["n_channels"])
val_index = min(
    range(len(val)),
    key=lambda i: (
        int(val_lengths[i]), int(val_channels[i]), str(val_ids[i]), i
    ),
)
val.select([val_index]).to_parquet(str(out_dir / "val.parquet"))

selected_set = set(selected)
selected_records = [
    row for row in records if row["source_index"] in selected_set
]
selected_records.sort(key=lambda row: selected.index(row["source_index"]))
doc = {
    "source_train": str(train_path),
    "source_val": str(val_path),
    "train_rows": selected_records,
    "val_row": {
        "source_index": val_index,
        "id": str(val_ids[val_index]),
        "length": int(val_lengths[val_index]),
        "n_channels": int(val_channels[val_index]),
    },
    "hard_16": hard16,
    "hard_131072": hard131,
    "checks": {
        "localize_all_channels_16_present": (
            row16["category"] == "localize_all_channels"
            and int(row16["n_channels"]) == 16
        ),
        "hard_16_has_max_available_length": (
            int(row16["length"]) == hard16_max_length
        ),
        "length_131072_present": int(row131["length"]) == 131072,
        "all_selected_rows_are_consumed": len(selected) == train_rows,
    },
}
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(doc, handle, indent=2)
    handle.write("\n")
print("SMOKE_SUBSET", json.dumps(doc, separators=(",", ":")))
PYSUBSET

[ -s "$SMOKE_DATA_DIR/train.parquet" ] || fail "smoke train.parquet is empty"
[ -s "$SMOKE_DATA_DIR/val.parquet" ] || fail "smoke val.parquet is empty"
[ -s "$SUBSET_MANIFEST" ] || fail "subset manifest is empty"

# node_train_v3.sh COMMON_ENV, except full-coverage is intentionally absent.
# Smoke-only additions are offline mode, per-step timing, gate logging, and
# post-unscale gradient diagnostics.
COMMON_ENV=(
  TSLM_TORCH_DTYPE=bfloat16
  TSLM_AMP=1
  TSLM_AMP_DTYPE=bfloat16
  TSLM_ATTN_IMPL=sdpa
  TSLM_GRAD_CKPT=1
  TSLM_CKPT_FALLBACK=1
  TSLM_EVAL_STRICT=1
  TSLM_SKIP_FINAL_TEST=1
  TSLM_QWEN_REVISION="$QWEN_REV"
  TSLM_SAVE_TRAINABLE_ONLY=1
  TSLM_SAVE_OPT_STATE=1
  TSLM_VAL_GEN=0
  TSLM_KEEP_EPOCH_CKPTS=1
  TSLM_V3_GATE_LOG_EVERY=1
  TSLM_V3_GRAD_LOG=1
  TSLM_STEP_TIMING=1
  HF_HUB_DISABLE_PROGRESS_BARS=1
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  HF_DATASETS_OFFLINE=1
  TOKENIZERS_PARALLELISM=false
  PYTHONUNBUFFERED=1
  CUDA_VISIBLE_DEVICES="$GPU"
  HIP_VISIBLE_DEVICES="$GPU"
  ROCR_VISIBLE_DEVICES="$GPU"
)

run_variant() {
  local variant="$1"
  local suffix cfg run_name run_dir train_log eval_log checkpoint
  local checkpoint_json eval_json result_json mem_csv
  local variant_start train_start train_wall eval_start eval_wall variant_wall
  local peak rc

  case "$variant" in
    v3untied) suffix="untied" ;;
    v3tied) suffix="tied" ;;
    *) fail "internal unknown variant: $variant" ;;
  esac
  cfg="configs/wavelet_precise_v3smoke_${suffix}.yaml"
  run_name="anomaly_xl_wavelet_v3smoke_${suffix}"
  run_dir="$RUN_BASE/$run_name"
  train_log="$LOG_DIR/v3smoke.${variant}.train.log"
  eval_log="$LOG_DIR/v3smoke.${variant}.eval.log"
  checkpoint_json="$LOG_DIR/v3smoke.${variant}.checkpoint.json"
  eval_json="$LOG_DIR/v3smoke.${variant}.eval.json"
  result_json="$LOG_DIR/v3smoke.${variant}.json"
  mem_csv="$LOG_DIR/v3smoke.${variant}.memory.csv"
  checkpoint="$run_dir/checkpoints/best_model.pt"

  CURRENT_STAGE="${variant}_train"
  rm -rf -- "$run_dir"
  rm -f "$train_log" "$eval_log" "$checkpoint_json" "$eval_json" \
    "$result_json" "$mem_csv"
  start_sampler "$mem_csv"
  variant_start=$(date +%s)
  train_start=$variant_start
  echo "SMOKE_TRAIN_START variant=$variant config=$cfg"
  if run_bounded env "${COMMON_ENV[@]}" \
      "$PY" train.py --config "$cfg" --no-wandb \
      > "$train_log" 2>&1; then
    :
  else
    rc=$?
    stop_sampler
    tail -80 "$train_log" >&2 || true
    fail "$variant train.py failed or timed out (exit $rc)"
  fi
  train_wall=$(( $(date +%s) - train_start ))
  [ -s "$checkpoint" ] || fail "$variant slim checkpoint missing: $checkpoint"

  CURRENT_STAGE="${variant}_checkpoint"
  "$PY" - "$checkpoint" "$variant" "$checkpoint_json" <<'PYCKPT'
import json
import sys

import torch
from src.models.wavelet_encoder import WaveletTSEncoder
from src.models.wavelet_encoder_v3 import (
    RecurrentWaveletTSEncoder,
    position_decodability,
    token_diversity,
)

path, variant, out = sys.argv[1:4]
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
state = checkpoint.get("model_state")
assert isinstance(state, dict) and state, "missing/empty model_state"
keys = sorted(state)
prefixes = {
    "v1_value_proj": "ts_encoder.value_proj.",
    "v1_band_embed": "ts_encoder.band_embed.",
    "v1_pad_token": "ts_encoder.pad_token",
    "step_embed": "ts_encoder.step_embed.",
    "gate_head": "ts_encoder.gate_head.",
    "blocks": "ts_encoder.blocks.",
    "context_norm": "ts_encoder.context_norm.",
    "context_scale": "ts_encoder.context_scale",
}
matches = {}
for group, prefix in prefixes.items():
    if group in ("v1_pad_token", "context_scale"):
        found = [key for key in keys if key == prefix]
    else:
        found = [key for key in keys if key.startswith(prefix)]
    assert found, f"checkpoint lacks required {group}: {prefix}"
    matches[group] = found
pos_freqs = [key for key in keys if "pos_freqs" in key]
assert not pos_freqs, f"pos_freqs must not be in slim checkpoint: {pos_freqs}"
tied = bool(checkpoint["config"]["model"]["encoder"]["tied"])
assert tied == (variant == "v3tied"), (variant, tied)

# Rebuild only the encoder and load only ts_encoder.* from the slim checkpoint.
# A spatially localized +30 sigma event is then rolled by token_diversity's
# default comparison, so an input-independent/global-mode encoder fails.
encoder_config = dict(checkpoint["config"]["model"]["encoder"])
encoder_config.pop("type", None)
encoder_config.pop("model_id", None)
encoder = RecurrentWaveletTSEncoder(**encoder_config).eval()
encoder_state = {
    key.removeprefix("ts_encoder."): value
    for key, value in state.items()
    if key.startswith("ts_encoder.")
}
missing, unexpected = encoder.load_state_dict(encoder_state, strict=True)
assert not missing and not unexpected, (missing, unexpected)
context_scale = float(encoder.context_scale.detach())
context_scale_initial = float(encoder_config["context_scale_init"])
context_scale_move = abs(context_scale - context_scale_initial)
assert context_scale_move >= 1.0e-3, (
    "context_scale moved less than 1e-3 after four training steps",
    context_scale_initial, context_scale,
)
torch.manual_seed(20260906)
probe = torch.randn(1, 4096)
probe[:, 896:1024] += 30.0
diversity = token_diversity(encoder, probe)
torch.manual_seed(0)
v1 = WaveletTSEncoder(**encoder_config).eval()
v1_decodability = position_decodability(v1, n=36, batch_pairs=6)
v3_decodability = position_decodability(encoder, n=36, batch_pairs=6)
position_ratio = (
    v3_decodability["position_mae"] / v1_decodability["position_mae"]
)
doc = {
    "checkpoint": path,
    "tensor_count": len(keys),
    "required_prefix_counts": {
        group: len(found) for group, found in matches.items()
    },
    "witness_keys": {
        group: found[0] for group, found in matches.items()
    },
    "pos_freqs_keys": pos_freqs,
    "tied": tied,
    "context_scale_initial": context_scale_initial,
    "context_scale_after_4_steps": context_scale,
    "context_scale_abs_move": context_scale_move,
    "token_diversity": diversity,
    "decodability": {
        "v1_init": v1_decodability,
        "v3_checkpoint": v3_decodability,
        "position_mae_ratio": position_ratio,
    },
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(doc, handle, indent=2)
    handle.write("\n")
print("CHECKPOINT_DIAGNOSTICS", json.dumps(doc, separators=(",", ":")))
assert diversity["mean_token_cosine"] <= 0.9, diversity
assert diversity["input_relative_difference"] > 0.1, diversity
assert position_ratio <= 1.5, doc["decodability"]
assert v3_decodability["presence_accuracy"] >= 0.55, v3_decodability
assert v3_decodability["magnitude_mae"] <= 0.25, v3_decodability
print("CHECKPOINT_OK")
PYCKPT

  CURRENT_STAGE="${variant}_eval_only"
  eval_start=$(date +%s)
  echo "SMOKE_EVAL_START variant=$variant checkpoint=$checkpoint"
  if run_bounded env "${COMMON_ENV[@]}" TSLM_EVAL_TAG=v3smoke \
      "$PY" train.py --config "$cfg" --no-wandb \
      --eval-only --checkpoint "$checkpoint" --split val \
      --out-json "$eval_json" > "$eval_log" 2>&1; then
    :
  else
    rc=$?
    stop_sampler
    tail -80 "$eval_log" >&2 || true
    fail "$variant eval-only failed or timed out (exit $rc)"
  fi
  eval_wall=$(( $(date +%s) - eval_start ))
  stop_sampler
  variant_wall=$(( $(date +%s) - variant_start ))

  [ -s "$mem_csv" ] || fail "$variant nvidia-smi sampler produced no rows"
  peak=$(peak_of "$mem_csv")
  case "$peak" in
    ''|*[!0-9]*) fail "$variant invalid sampled peak: $peak" ;;
  esac
  [ "$peak" -le "$PEAK_LIMIT_MIB" ] || \
    fail "$variant GPU peak ${peak} MiB exceeds ${PEAK_LIMIT_MIB} MiB"

  CURRENT_STAGE="${variant}_verify_logs"
  "$PY" - "$variant" "$train_log" "$eval_log" "$checkpoint_json" \
      "$eval_json" "$result_json" "$TRAIN_ROWS" "$peak" \
      "$PEAK_LIMIT_MIB" "$train_wall" "$eval_wall" "$variant_wall" <<'PYVERIFY'
import json
import re
import sys

(
    variant, train_log, eval_log, checkpoint_json, eval_json, out,
    expected_steps, peak, peak_limit, train_wall, eval_wall, variant_wall,
) = sys.argv[1:]
expected_steps = int(expected_steps)
train_text = open(
    train_log, encoding="utf-8", errors="replace"
).read().replace("\r", "\n")
eval_text = open(
    eval_log, encoding="utf-8", errors="replace"
).read().replace("\r", "\n")

timings = [
    (int(step), float(seconds))
    for step, seconds in re.findall(
        r"\[StepTiming\] step=(\d+) seconds=([0-9.]+)", train_text
    )
]
assert [step for step, _ in timings] == list(
    range(1, expected_steps + 1)
), timings
durations = [seconds for _, seconds in timings]
assert all(seconds > 0 for seconds in durations), durations
steady = durations[1:]
assert steady, "need at least one timing after the first step"

coverage_line = (
    f"[Coverage] epoch 1: trained {expected_steps}, failed 0, "
    f"expected {expected_steps}"
)
assert coverage_line in train_text, coverage_line
category_match = re.search(
    r"(?m)^    localize_all_channels: ok=(\d+) failed=0$", train_text
)
assert category_match and int(category_match.group(1)) >= 1
gate_lines = [
    line for line in train_text.splitlines() if "[V3Gates]" in line
]
assert gate_lines, "no [V3Gates] line in training output"
gradient_rows = [
    (int(step), float(blocks), float(value), float(relative))
    for step, blocks, value, relative in re.findall(
        r"\[V3Grad\] step=(\d+).*?blocks_grad_norm=([0-9.eE+\-]+) "
        r"value_proj_grad_norm=([0-9.eE+\-]+) "
        r"blocks_relative=([0-9.eE+\-]+)",
        train_text,
    )
]
assert [row[0] for row in gradient_rows] == list(range(expected_steps)), gradient_rows
assert all(row[1] > 0.0 for row in gradient_rows), gradient_rows
assert "Training Complete" in train_text
assert "[SkipFinalTest]" in train_text
assert "[CkptFallback]" not in train_text, \
    "unexpected checkpoint fallback in smoke"

checkpoint = json.load(open(checkpoint_json, encoding="utf-8"))
evaluation = json.load(open(eval_json, encoding="utf-8"))
assert "EVALONLY_RESULT:" in eval_text
assert "[EvalCoverage]" in eval_text
assert "1/1 examples accounted for" in eval_text
assert evaluation.get("split") == "val", evaluation
assert evaluation.get("num_batches") == 1, evaluation
assert evaluation.get("samples_evaluated") == 1, evaluation
assert evaluation.get("eval_errors") == 0, evaluation

doc = {
    "variant": variant,
    "steps": expected_steps,
    "step_seconds": durations,
    "seconds_per_step_after_first": sum(steady) / len(steady),
    "train_wall_s": int(train_wall),
    "eval_wall_s": int(eval_wall),
    "total_wall_s": int(variant_wall),
    "gpu_peak_mib": int(peak),
    "gpu_peak_limit_mib": int(peak_limit),
    "v3_gate_lines": len(gate_lines),
    "gradient_diagnostics": {
        "rows": len(gradient_rows),
        "min_blocks_grad_norm": min(row[1] for row in gradient_rows),
        "max_blocks_grad_norm": max(row[1] for row in gradient_rows),
        "max_blocks_relative_to_value_proj": max(row[3] for row in gradient_rows),
    },
    "coverage": {
        "trained": expected_steps,
        "failed": 0,
        "expected": expected_steps,
        "localize_all_channels_ok": int(category_match.group(1)),
    },
    "checkpoint_keys": checkpoint["witness_keys"],
    "checkpoint_prefix_counts": checkpoint["required_prefix_counts"],
    "checkpoint_tensor_count": checkpoint["tensor_count"],
    "checkpoint_has_pos_freqs": bool(checkpoint["pos_freqs_keys"]),
    "context_scale_initial": checkpoint["context_scale_initial"],
    "context_scale_after_4_steps": checkpoint["context_scale_after_4_steps"],
    "context_scale_abs_move": checkpoint["context_scale_abs_move"],
    "token_diversity": checkpoint["token_diversity"],
    "decodability": checkpoint["decodability"],
    "eval": {
        "EVALONLY_RESULT": True,
        "split": evaluation["split"],
        "num_batches": evaluation["num_batches"],
        "samples_evaluated": evaluation["samples_evaluated"],
        "eval_errors": evaluation["eval_errors"],
        "loss": evaluation.get("loss"),
    },
    "pass": True,
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(doc, handle, indent=2)
    handle.write("\n")
print("VARIANT_OK", json.dumps(doc, separators=(",", ":")))
PYVERIFY

  cat "$result_json"
  echo "SMOKE_VARIANT_OK variant=$variant steps=$TRAIN_ROWS peak_mib=$peak"
}

# Heaviest encoder first; tied must pass independently afterwards.
run_variant v3untied
run_variant v3tied

CURRENT_STAGE="finalize"
TOTAL_WALL=$(( $(date +%s) - START_WALL ))
[ "$TOTAL_WALL" -le "$TOTAL_WALL_S" ] || \
  fail "smoke took ${TOTAL_WALL}s, exceeding ${TOTAL_WALL_S}s"

# Required transient run dirs and mini-parquets are gone before success exists.
cleanup_artifacts
for run_dir in "${RUN_DIRS[@]}"; do
  [ ! -e "$run_dir" ] || fail "run dir cleanup failed: $run_dir"
done
[ ! -e "$SMOKE_DATA_DIR" ] || \
  fail "smoke data cleanup failed: $SMOKE_DATA_DIR"

"$PY" - "$OK_FILE" "$SUBSET_MANIFEST" \
    "$LOG_DIR/v3smoke.v3untied.json" "$LOG_DIR/v3smoke.v3tied.json" \
    "$TOTAL_WALL" "$TOTAL_WALL_S" <<'PYFINAL'
import json
import sys

out, subset_path, untied_path, tied_path, total_wall, wall_limit = sys.argv[1:]
subset = json.load(open(subset_path, encoding="utf-8"))
variants = [
    json.load(open(untied_path, encoding="utf-8")),
    json.load(open(tied_path, encoding="utf-8")),
]
assert [item["variant"] for item in variants] == ["v3untied", "v3tied"]
assert all(item.get("pass") for item in variants)
assert subset["checks"]["localize_all_channels_16_present"]
assert subset["checks"]["hard_16_has_max_available_length"]
assert subset["checks"]["length_131072_present"]
doc = {
    "status": "OK",
    "variants": variants,
    "subset": subset,
    "total_wall_s": int(total_wall),
    "total_wall_limit_s": int(wall_limit),
    "run_dirs_removed": True,
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(doc, handle, indent=2)
    handle.write("\n")
print(json.dumps(doc, separators=(",", ":")))
PYFINAL

rm -f "$FAILED_FILE"
cat "$OK_FILE"
echo "SMOKE_OK"
