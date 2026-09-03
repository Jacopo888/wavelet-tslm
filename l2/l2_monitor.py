#!/usr/bin/env python3
"""L2 coverage monitor: one ``[L2-Coverage]`` line per orchestrator step, from the artefacts
prime-rl writes at the pin (no GPU, no prime-rl import needed):

  <output_dir>/orchestrator_metrics.jsonl   (``[orchestrator.file_monitor]``, one dict per step)
  <output_dir>/logs/orchestrator.log         (``Step N | ... | Reward r | Trainable a/b | Turns t |
                                              ... | Error e% | Truncation t%`` success lines)
  <output_dir>/logs/trainer.log              (loss / grad-norm lines, optional)

Usage:
  l2_monitor.py <output_dir> [--follow] [--jsonl <path>] [--expect-rollouts N] [--max-error-rate F]

Exit codes (used by the run driver as a fail-closed gate): 0 = fine, 3 = a step exceeded the
error-rate threshold or shipped fewer rollouts than expected, 4 = no step seen at all.
The per-step JSONL written with ``--jsonl`` is the durable coverage record of the run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

STEP_RE = re.compile(
    r"Step (?P<step>\d+) \| +(?P<time>[^|]+?) \| Reward (?P<reward>[-\d.]+) \| "
    r"Trainable (?P<trainable>\d+)/(?P<effective>\d+) \((?P<trainable_rate>[\d.]+)%\) \| "
    r"Turns (?P<turns>[\d.]+) \| Branches (?P<branches>[\d.]+) \| "
    r"Max Off-Policy (?P<off_policy>\d+) \| Error (?P<error>[\d.]+)% \| Truncation (?P<trunc>[\d.]+)%"
)
EVAL_RE = re.compile(r"eval/(?P<env>[^/]+)/(?P<rest>.+)")


def parse_metrics_jsonl(path: Path) -> dict[int, dict]:
    """step -> metrics dict (last write wins)."""
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = d.get("step")
            if step is None:
                continue
            out[int(step)] = d
    return out


def parse_log_steps(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                g = m.groupdict()
                out[int(g["step"])] = {
                    "reward_mean": float(g["reward"]),
                    "trainable": int(g["trainable"]),
                    "effective": int(g["effective"]),
                    "turns_mean": float(g["turns"]),
                    "off_policy_max": int(g["off_policy"]),
                    "error_rate": float(g["error"]) / 100.0,
                    "truncation_rate": float(g["trunc"]) / 100.0,
                    "step_time_txt": g["time"].strip(),
                }
    return out


def pick(d: dict, *patterns: str) -> dict:
    """Metrics whose key matches any regex pattern."""
    res = {}
    for k, v in d.items():
        for p in patterns:
            if re.search(p, k):
                res[k] = v
                break
    return res


def coverage_row(step: int, m: dict | None, lg: dict | None, expect_rollouts: int | None) -> dict:
    row: dict = {"step": step}
    if m:
        row["rollouts"] = m.get("progress/rollouts")
        row["tasks"] = m.get("progress/tasks")
        row["tokens"] = m.get("progress/tokens")
        row["output_tokens"] = m.get("progress/output_tokens")
        row["step_time_s"] = m.get("time/step")
        row["errored_train"] = m.get("dispatcher/errored/train")
        row["cancelled_train"] = m.get("dispatcher/cancelled/train")
        row["prefilter_dropped_rate"] = m.get("pre_filters/all/dropped_rate")
        agg = pick(
            m,
            r"^train/agg/(all|effective)/.*/(reward|has_error|is_truncated|is_filtered|is_trainable)/mean$",
            r"^train/agg/(all|effective)/num_turns/mean$",
            r"^train/agg/(all|effective)/num_output_tokens/mean$",
            r"^train/agg/(all|effective)/.*/filters/.*/mean$",
            r"^train/agg/(all|effective)/.*/(correct_answer|forced_final_penalty|length_efficiency).*mean$",
        )
        row["agg"] = agg
        row["eval"] = {k: v for k, v in m.items() if k.startswith("eval/")}
    if lg:
        row.update({f"log_{k}": v for k, v in lg.items()})
    if expect_rollouts is not None and row.get("rollouts") is not None:
        row["rollouts_expected"] = expect_rollouts
        row["rollouts_short"] = row["rollouts"] < expect_rollouts
    return row


def fmt(row: dict) -> str:
    agg = row.get("agg", {})
    def g(pat: str):
        for k, v in agg.items():
            if re.search(pat, k):
                return v
        return None
    parts = [
        f"[L2-Coverage] step {row['step']}:",
        f"rollouts {row.get('rollouts')}" + (f"/{row['rollouts_expected']}" if "rollouts_expected" in row else ""),
        f"tasks {row.get('tasks')}",
        f"errored {row.get('errored_train')}",
        f"cancelled {row.get('cancelled_train')}",
        f"err_rate {row.get('log_error_rate')}",
        f"trainable {row.get('log_trainable')}/{row.get('log_effective')}",
        f"reward {row.get('log_reward_mean')}",
        f"turns {row.get('log_turns_mean')}",
        f"trunc {row.get('log_truncation_rate')}",
        f"forced_final {g(r'forced_final_penalty.*/mean$')}",
        f"correct {g(r'correct_answer.*/mean$')}",
        f"zero_adv_dropped {g(r'filters/zero_advantage/mean$')}",
        f"out_tok/rollout {g(r'effective/num_output_tokens/mean$')}",
        f"step_time {row.get('step_time_s')}",
    ]
    if row.get("eval"):
        ev = {k: v for k, v in row["eval"].items() if k.endswith("avg@1") or k.endswith("/reward/mean")}
        parts.append(f"eval {ev}")
    if row.get("rollouts_short"):
        parts.append("ROLLOUTS_SHORT")
    return " | ".join(str(p) for p in parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--jsonl", help="append one coverage row per new step here")
    ap.add_argument("--expect-rollouts", type=int, default=None)
    ap.add_argument("--max-error-rate", type=float, default=0.25)
    ap.add_argument("--metrics-name", default="orchestrator_metrics.jsonl")
    args = ap.parse_args()

    out = Path(args.output_dir)
    metrics_path = out / args.metrics_name
    log_path = out / "logs" / "orchestrator.log"
    seen: set[int] = set()
    violations = 0
    jsonl = open(args.jsonl, "a", encoding="utf-8") if args.jsonl else None
    try:
        while True:
            m = parse_metrics_jsonl(metrics_path)
            lg = parse_log_steps(log_path)
            steps = sorted(set(m) | set(lg))
            for s in steps:
                if s in seen:
                    continue
                # wait for both sources unless following is off (post-hoc: emit what exists)
                if args.follow and (s not in m or s not in lg):
                    continue
                seen.add(s)
                row = coverage_row(s, m.get(s), lg.get(s), args.expect_rollouts)
                print(fmt(row), flush=True)
                if jsonl:
                    jsonl.write(json.dumps(row) + "\n"); jsonl.flush()
                er = row.get("log_error_rate")
                if (er is not None and er > args.max_error_rate) or row.get("rollouts_short"):
                    violations += 1
                    print(f"[L2-Coverage] VIOLATION at step {s}: error_rate={er} rollouts_short={row.get('rollouts_short')}", flush=True)
            if not args.follow:
                break
            time.sleep(args.interval)
    finally:
        if jsonl:
            jsonl.close()
    if not seen:
        return 4
    return 3 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
