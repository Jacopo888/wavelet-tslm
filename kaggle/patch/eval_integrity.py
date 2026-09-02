"""Research-integrity helpers for train.py — torch-free and unit-testable.

Born from the 02/09/2026 integrity pass: the m30 run silently dropped all
2,770 16-channel training cells while monitoring reported "zero errors",
and the evaluator swallowed generation errors without a trace. Every
policy here is fail-closed: a sample that cannot be trained on or
evaluated must either abort the run or leave an explicit record.
"""

from __future__ import annotations

import json
import os
from typing import Optional


def coverage_abort(context: str, env=None) -> None:
    """Fail fast on an unrecoverable sample when full coverage is required.

    Called from every skip path in train_epoch (invalid loss, fallback
    failure, generic batch error). With TSLM_REQUIRE_FULL_COVERAGE=1 the
    run dies HERE, at the first lost sample, instead of at epoch end —
    a 5h run must not keep burning GPU after its coverage is already broken.
    """
    env = env if env is not None else os.environ
    if env.get("TSLM_REQUIRE_FULL_COVERAGE") == "1":
        raise SystemExit(
            f"[Coverage] ABORT (fail-fast): training sample lost and "
            f"TSLM_REQUIRE_FULL_COVERAGE=1 — {context}"
        )


def restore_grad_ckpt(lm, env=None) -> None:
    """Restore the GLOBAL gradient-checkpointing setting after a CkptFallback.

    The old code disabled checkpointing unconditionally in the fallback's
    `finally`, so on a run with TSLM_GRAD_CKPT=1 a single OOM would strip
    checkpointing from every later batch (mass-OOM on 16ch cells). The
    global env setting, not the fallback, owns the steady state.
    """
    env = env if env is not None else os.environ
    try:
        if env.get("TSLM_GRAD_CKPT") == "1":
            lm.gradient_checkpointing_enable()
        else:
            lm.gradient_checkpointing_disable()
    except Exception:
        # Best-effort, as before: a restore failure must not mask the
        # original batch error. The next batch surfaces any real breakage.
        pass


class EvalJsonlWriter:
    """One line per evaluated example. `path=None` disables writing but
    still counts rows, so coverage checking works in all code paths."""

    def __init__(self, path: Optional[str]):
        self.path = str(path) if path else None
        self.count = 0
        self.error_count = 0
        self._fh = open(self.path, "w", encoding="utf-8") if self.path else None

    def write(self, row: dict) -> None:
        self.count += 1
        if row.get("parse_status") in ("generation_error", "loss_error"):
            self.error_count += 1
        if self._fh is not None:
            self._fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def check_eval_coverage(n_rows: int, n_expected: Optional[int],
                        split_name: str, env=None) -> None:
    """Positive verification of eval coverage: every example accounted for.

    Prints the count unconditionally; with TSLM_EVAL_STRICT=1 a single
    missing example kills the eval instead of shipping a metric computed
    on a silently smaller denominator.
    """
    env = env if env is not None else os.environ
    if n_expected is None:
        print(f"  [EvalCoverage] {split_name}: {n_rows} rows "
              f"(expected count unknown)", flush=True)
        return
    print(f"  [EvalCoverage] {split_name}: {n_rows}/{n_expected} examples "
          f"accounted for", flush=True)
    if n_rows != n_expected and env.get("TSLM_EVAL_STRICT") == "1":
        raise SystemExit(
            f"[EvalCoverage] ABORT: {split_name} wrote {n_rows} rows but the "
            f"dataset has {n_expected} examples (TSLM_EVAL_STRICT=1)"
        )


def check_resume_checkpoint(path: str, env=None) -> None:
    """Refuse INTRA-EPOCH resume when full coverage is required.

    train.py's resume restarts from `epoch+1`: resuming from a
    checkpoint_epoch_N_batch_M.pt silently skips the rest of epoch N
    (thousands of rows the coverage counters never re-check) and shrinks
    the schedule. Under TSLM_REQUIRE_FULL_COVERAGE=1 only end-of-epoch
    checkpoints (epoch_*_end.pt) or explicit weights-only warm starts are
    acceptable resume points (review finding, 02/09).
    """
    env = env if env is not None else os.environ
    if not path or env.get("TSLM_REQUIRE_FULL_COVERAGE") != "1":
        return
    if env.get("TSLM_RESUME_WEIGHTS_ONLY") == "1":
        return  # warm start, epochs restart from 1: coverage is intact
    name = os.path.basename(str(path))
    if "batch" in name or name == "latest_checkpoint.pt":
        raise SystemExit(
            f"[Coverage] ABORT: resume from intra-epoch checkpoint "
            f"'{name}' would silently skip the remainder of its epoch. "
            f"Resume from an epoch_*_end.pt (or set "
            f"TSLM_RESUME_WEIGHTS_ONLY=1 for a warm start).")


def parse_status_for(category: str, pred_answer: str,
                     parser=None, required_keys=None) -> str:
    """Classify how the official parser saw this prediction.

    `parser`/`required_keys` are scoring.parse_structured_pred and
    scoring.REQUIRED_KEYS, injected by the caller so this module stays
    import-clean. Returns "parsed", "parse_failed", or "unknown" (parser
    unavailable / category outside the precise families).
    """
    if parser is None or required_keys is None:
        return "unknown"
    req = required_keys.get(category)
    if req is None:
        return "unknown"
    return "parsed" if parser(pred_answer or "", req) is not None else "parse_failed"
