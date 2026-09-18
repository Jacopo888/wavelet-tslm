#!/usr/bin/env python3
"""L2 coverage monitor: one ``[L2-Coverage]`` line per orchestrator step, built from what
prime-rl c5afcaa2e writes on disk (no GPU, no prime-rl import):

  <output_dir>/run_default/rollouts/step_N/train_rollouts.jsonl   one record per rollout that was
      shipped to the trainer at step N: task (category, length, n_channels, seed), rewards
      ({correct_answer, answer_format}), metrics (dataset metrics + forced_final telemetry), nodes (message graph
      with tool_calls + per-node usage), stop_condition, errors, is_completed, timing
  <output_dir>/logs/orchestrator.log   ``Step N | t | Reward r | Trainable a/b | Turns t | ... |
      Error e% | Truncation t%`` success lines (the sink-level view, errored arrivals included)

Usage:
  l2_monitor.py <output_dir> [--follow] [--jsonl <path>] [--expect-rollouts N] [--max-error-rate F]

Exit codes (the run driver uses them as a fail-closed gate): 0 fine; 3 a step exceeded the error
rate, had incomplete sources, or shipped a different rollout count; 4 no step seen; 5 the
optional collapse guard fired. The JSONL written with --jsonl is the durable coverage record.
"""

from __future__ import annotations

import argparse
import collections
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


def _tool_calls(rec: dict) -> tuple[int, int, int]:
    """(assistant turns, tool calls, wavelet_zoom calls) from the message graph."""
    turns = calls = zoom = 0
    for node in rec.get("nodes") or []:
        msg = node.get("message") or {}
        if msg.get("role") != "assistant":
            continue
        turns += 1
        for tc in msg.get("tool_calls") or []:
            calls += 1
            blob = json.dumps(tc)
            if "wavelet_zoom(" in blob:
                zoom += 1
    return turns, calls, zoom


def _tokens(rec: dict) -> tuple[int, int]:
    """(prompt tokens, completion tokens) summed over sampled nodes' usage when present."""
    p = c = 0
    for node in rec.get("nodes") or []:
        u = node.get("usage") or {}
        if isinstance(u, dict):
            p += int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
            c += int(u.get("completion_tokens") or u.get("output_tokens") or 0)
    return p, c


def _final_text(rec: dict) -> str:
    for node in reversed(rec.get("nodes") or []):
        msg = node.get("message") or {}
        if msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def summarize_step(step: int, path: Path) -> dict:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    n = len(rows)
    errored = sum(1 for r in rows if (r.get("errors") or []) or r.get("is_completed") is False)
    total_rewards = [float(sum((r.get("rewards") or {}).values())) for r in rows]
    correct = [float((r.get("rewards") or {}).get("correct_answer", 0.0)) for r in rows]
    format_valid = sum(1 for r in rows if float((r.get("rewards") or {}).get("answer_format", 0.0)) > 0)
    forced = sum(
        1 for r in rows
        if float((r.get("metrics") or {}).get("forced_final", 0.0)) > 0
        or float((r.get("rewards") or {}).get("forced_final_penalty", 0.0)) < 0  # legacy runs
    )
    empty_final = sum(1 for r in rows if not _final_text(r).strip())
    stops = collections.Counter(str(r.get("stop_condition")) for r in rows)
    by_cat: dict[str, list[float]] = collections.defaultdict(list)
    for r, rew in zip(rows, total_rewards):
        by_cat[str((r.get("task") or {}).get("category"))].append(rew)
    turns = calls = zoom = ptok = ctok = 0
    for r in rows:
        t, c, z = _tool_calls(r)
        turns += t; calls += c; zoom += z
        p, cc = _tokens(r)
        ptok += p; ctok += cc
    gen_s = []
    for r in rows:
        g = (r.get("timing") or {}).get("generation") or {}
        if isinstance(g, dict) and g.get("start") and g.get("end"):
            gen_s.append(float(g["end"]) - float(g["start"]))
    groups: dict[object, list[float]] = collections.defaultdict(list)
    for r, rew in zip(rows, total_rewards):
        groups[(r.get("task") or {}).get("idx")].append(rew)
    zero_var_groups = sum(1 for v in groups.values() if len(v) > 1 and max(v) == min(v))
    return {
        "step": step,
        "rollouts": n,
        "errored": errored,
        "reward_mean": (sum(total_rewards) / n) if n else None,
        "reward_positive": sum(1 for x in total_rewards if x > 0),
        "correct_mean": (sum(correct) / n) if n else None,
        "correct_positive": sum(1 for x in correct if x > 0),
        "format_valid": format_valid,
        "forced_final": forced,
        "empty_final": empty_final,
        "empty_final_rate": (empty_final / n) if n else None,
        "stop_conditions": dict(stops),
        "per_category": {k: {"n": len(v), "reward_mean": sum(v) / len(v)} for k, v in sorted(by_cat.items())},
        "groups": len(groups),
        "zero_variance_groups": zero_var_groups,
        "turns_mean": (turns / n) if n else None,
        "tool_calls_mean": (calls / n) if n else None,
        "wavelet_zoom_calls": zoom,
        "prompt_tokens_mean": (ptok / n) if n else None,
        "completion_tokens_mean": (ctok / n) if n else None,
        "generation_s_mean": (sum(gen_s) / len(gen_s)) if gen_s else None,
    }


def fmt(row: dict) -> str:
    f = lambda x, d=3: ("-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x)))
    cats = " ".join(f"{k[:8]}={v['reward_mean']:.2f}/{v['n']}" for k, v in (row.get("per_category") or {}).items())
    parts = [
        f"[L2-Coverage] step {row['step']}:",
        f"rollouts {row.get('rollouts')}" + (f"/{row['rollouts_expected']}" if "rollouts_expected" in row else ""),
        f"errored {row.get('errored')}",
        f"log_err {f(row.get('log_error_rate'))}",
        f"trainable {row.get('log_trainable')}/{row.get('log_effective')}",
        f"reward {f(row.get('reward_mean'))} (>0: {row.get('reward_positive')})",
        f"correct {f(row.get('correct_mean'))} (>0: {row.get('correct_positive')})",
        f"format {row.get('format_valid')}",
        f"empty_final {row.get('empty_final')} ({f(row.get('empty_final_rate'))})",
        f"forced_final {row.get('forced_final')}",
        f"zero_var_groups {row.get('zero_variance_groups')}/{row.get('groups')}",
        f"turns {f(row.get('turns_mean'), 1)}",
        f"tool_calls {f(row.get('tool_calls_mean'), 1)}",
        f"wavelet_zoom {row.get('wavelet_zoom_calls')}",
        f"tok_in/out {f(row.get('prompt_tokens_mean'), 0)}/{f(row.get('completion_tokens_mean'), 0)}",
        f"gen_s {f(row.get('generation_s_mean'), 0)}",
        f"step_time {row.get('log_step_time_txt')}",
        f"cats[{cats}]",
    ]
    if row.get("rollouts_mismatch"):
        parts.append("ROLLOUTS_MISMATCH")
    if row.get("rollout_data_missing") or row.get("log_step_missing"):
        parts.append("SOURCE_MISSING")
    return " | ".join(parts)


def coverage_row(step: int, summary: dict | None, lg: dict | None, expect_rollouts: int | None) -> dict:
    row: dict = {"step": step, "rollout_data_missing": summary is None, "log_step_missing": lg is None}
    if summary:
        row.update(summary)
    if lg:
        row.update({f"log_{k}": v for k, v in lg.items()})
    if expect_rollouts is not None:
        row["rollouts_expected"] = expect_rollouts
        row["rollouts_mismatch"] = row.get("rollouts") != expect_rollouts
    return row


def collapse_guard(rows: list[dict], consecutive: int, min_trainable: int, max_empty_rate: float) -> tuple[bool, str]:
    """Fail only on a consecutive completed-step streak; one noisy step is tolerated."""
    streak: list[tuple[int, list[str]]] = []
    for row in sorted(rows, key=lambda r: r["step"]):
        reasons = []
        if (row.get("log_trainable") is not None and row["log_trainable"] < min_trainable):
            reasons.append(f"trainable={row['log_trainable']}<{min_trainable}")
        if (row.get("empty_final_rate") is not None and row["empty_final_rate"] > max_empty_rate):
            reasons.append(f"empty_final_rate={row['empty_final_rate']:.3f}>{max_empty_rate:.3f}")
        if reasons:
            if streak and row["step"] != streak[-1][0] + 1:
                streak = []
            streak.append((row["step"], reasons))
            if len(streak) >= consecutive:
                return False, "; ".join(f"step {s}: {', '.join(rs)}" for s, rs in streak[-consecutive:])
        else:
            streak = []
    return True, "no consecutive collapse signature"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--jsonl", help="append one coverage row per new step here")
    ap.add_argument(
        "--log-file",
        action="append",
        default=[],
        help="orchestrator log to parse; repeat for multi-phase runs (default: output_dir/logs/orchestrator.log)",
    )
    ap.add_argument("--expect-rollouts", type=int, default=None)
    ap.add_argument("--max-error-rate", type=float, default=0.25)
    ap.add_argument("--guard-collapse", action="store_true", help="check only consecutive empty/low-trainable steps")
    ap.add_argument("--guard-consecutive", type=int, default=2)
    ap.add_argument("--min-trainable", type=int, default=8)
    ap.add_argument("--max-empty-rate", type=float, default=0.80)
    args = ap.parse_args()
    if args.guard_collapse and args.follow:
        ap.error("--guard-collapse is a one-shot check and cannot be combined with --follow")

    out = Path(args.output_dir)
    rollouts_root = out / "run_default" / "rollouts"
    log_paths = [Path(p) for p in args.log_file] or [out / "logs" / "orchestrator.log"]
    seen: set[int] = set()
    violations = 0
    jsonl = open(args.jsonl, "a", encoding="utf-8") if args.jsonl else None
    try:
        while True:
            step_files = {}
            if rollouts_root.exists():
                for d in rollouts_root.glob("step_*"):
                    p = d / "train_rollouts.jsonl"
                    if p.exists():
                        try:
                            step_files[int(d.name.split("_")[1])] = p
                        except ValueError:
                            continue
            lg = {}
            for log_path in log_paths:
                lg.update(parse_log_steps(log_path))
            if args.guard_collapse:
                rows = [
                    coverage_row(s, summarize_step(s, step_files[s]), lg[s], args.expect_rollouts)
                    for s in sorted(set(step_files) & set(lg))
                ]
                ok, reason = collapse_guard(rows, args.guard_consecutive, args.min_trainable, args.max_empty_rate)
                print(f"[L2-Guard] {'OK' if ok else 'COLLAPSE'}: {reason}", flush=True)
                return 0 if ok else 5
            for s in sorted(set(step_files) | set(lg)):
                if s in seen:
                    continue
                if args.follow and (s not in step_files or s not in lg):
                    continue  # wait until both sources have the step
                if s in step_files and args.follow:
                    # the orchestrator writes the file then logs the step; give the writer a moment
                    time.sleep(2)
                seen.add(s)
                summary = summarize_step(s, step_files[s]) if s in step_files else None
                row = coverage_row(s, summary, lg.get(s), args.expect_rollouts)
                print(fmt(row), flush=True)
                if jsonl:
                    jsonl.write(json.dumps(row) + "\n"); jsonl.flush()
                er = row.get("log_error_rate")
                n = row.get("rollouts")
                err_frac = (row.get("errored") / n) if n else None
                source_missing = row.get("rollout_data_missing") or row.get("log_step_missing")
                if (er is not None and er > args.max_error_rate) or (err_frac is not None and err_frac > args.max_error_rate) or row.get("rollouts_mismatch") or source_missing:
                    violations += 1
                    print(f"[L2-Coverage] VIOLATION at step {s}: log_error_rate={er} errored_frac={err_frac} rollouts_mismatch={row.get('rollouts_mismatch')} source_missing={source_missing}", flush=True)
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
