#!/usr/bin/env python
"""Integrity-pass test suite — pure Python, no torch/GPU required.

Run:  <anomalyXL venv python> integrity/tests/run_tests.py

Covers each 02/09 fix:
  1. eval_integrity.coverage_abort       (fail-fast on lost samples)
  2. eval_integrity.restore_grad_ckpt    (finally must honor TSLM_GRAD_CKPT)
  3. eval_integrity.EvalJsonlWriter + check_eval_coverage (strict eval)
  4. eval_integrity.parse_status_for
  5. preflight_data.check_variant        (leak detection, synthetic fixtures)
  6. preflight on the REAL quarantined files (the actual pathological case)
  7. diagnostics strict validator / localize decomposition / bootstrap
"""

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # wavelet-tslm/
sys.path.insert(0, os.path.join(ROOT, "kaggle", "patch"))
sys.path.insert(0, os.path.join(ROOT, "integrity"))

import eval_integrity as ei                    # noqa: E402
import diagnostics as dg                       # noqa: E402
import preflight_data as pf                    # noqa: E402

PASS = []
FAIL = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS {name}")
    except AssertionError as e:
        FAIL.append((name, str(e)))
        print(f"  FAIL {name}: {e}")
    except Exception as e:  # noqa: BLE001
        FAIL.append((name, f"{type(e).__name__}: {e}"))
        print(f"  FAIL {name}: {type(e).__name__}: {e}")


# ---- 1. coverage_abort ----------------------------------------------------

def t_coverage_abort_on():
    try:
        ei.coverage_abort("test ctx", env={"TSLM_REQUIRE_FULL_COVERAGE": "1"})
    except SystemExit as e:
        assert "fail-fast" in str(e) and "test ctx" in str(e)
        return
    raise AssertionError("did not raise with flag on")


def t_coverage_abort_off():
    ei.coverage_abort("ctx", env={})                       # must not raise
    ei.coverage_abort("ctx", env={"TSLM_REQUIRE_FULL_COVERAGE": "0"})


# ---- 2. restore_grad_ckpt -------------------------------------------------

class _MockLM:
    def __init__(self):
        self.state = None
    def gradient_checkpointing_enable(self):
        self.state = "on"
    def gradient_checkpointing_disable(self):
        self.state = "off"


def t_restore_honors_global_on():
    lm = _MockLM()
    ei.restore_grad_ckpt(lm, env={"TSLM_GRAD_CKPT": "1"})
    assert lm.state == "on", f"global ckpt stripped: {lm.state}"


def t_restore_honors_global_off():
    lm = _MockLM()
    ei.restore_grad_ckpt(lm, env={})
    assert lm.state == "off"


def t_restore_swallows_errors():
    class Bad:
        def gradient_checkpointing_enable(self):
            raise RuntimeError("boom")
    ei.restore_grad_ckpt(Bad(), env={"TSLM_GRAD_CKPT": "1"})   # must not raise


# ---- 3. JSONL writer + strict coverage ------------------------------------

def t_jsonl_roundtrip_and_counts():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "e.jsonl")
        w = ei.EvalJsonlWriter(p)
        w.write({"id": "a", "parse_status": "parsed", "primary": 0.5})
        w.write({"id": "b", "parse_status": "generation_error"})
        w.close()
        rows = [json.loads(x) for x in open(p, encoding="utf-8")]
        assert len(rows) == 2 and w.count == 2 and w.error_count == 1
        assert rows[1]["parse_status"] == "generation_error"


def t_strict_coverage_aborts_on_missing():
    try:
        ei.check_eval_coverage(799, 800, "test", env={"TSLM_EVAL_STRICT": "1"})
    except SystemExit as e:
        assert "799" in str(e) and "800" in str(e)
        return
    raise AssertionError("799/800 did not abort under strict")


def t_lenient_coverage_only_warns():
    ei.check_eval_coverage(799, 800, "test", env={})          # must not raise
    ei.check_eval_coverage(800, 800, "test", env={"TSLM_EVAL_STRICT": "1"})


# ---- 4. parse_status_for --------------------------------------------------

def t_parse_status():
    parser = lambda text, req: {"ok": 1} if "{" in text else None
    req = {"localize": frozenset({"present"})}
    assert ei.parse_status_for("localize", '{"present": true}', parser, req) == "parsed"
    assert ei.parse_status_for("localize", "nope", parser, req) == "parse_failed"
    assert ei.parse_status_for("other_cat", "{}", parser, req) == "unknown"
    assert ei.parse_status_for("localize", "{}", None, None) == "unknown"


# ---- 5. preflight on synthetic fixtures ------------------------------------

def _write_parquet(path, ids, category="localize"):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.table({"id": ids, "category": [category] * len(ids)}), path)


def t_preflight_detects_leak():
    with tempfile.TemporaryDirectory() as td:
        _write_parquet(os.path.join(td, "anomalyxl-precise", "data.parquet"),
                       ["a", "b", "c", "d"])
        _write_parquet(os.path.join(td, "anomalyxl-train", "anomalyxl-precise",
                                    "train.parquet"), ["a", "b"])   # LEAK
        _write_parquet(os.path.join(td, "anomalyxl-train", "anomalyxl-precise",
                                    "val.parquet"), ["z"])
        old_rows, old_cats = pf.EXPECTED_TEST_ROWS, pf.EXPECTED_TEST_CATEGORIES
        pf.EXPECTED_TEST_ROWS = {"precise": 4}
        pf.EXPECTED_TEST_CATEGORIES = {}
        try:
            problems = []
            pf.check_variant(td, "precise", problems)
        finally:
            pf.EXPECTED_TEST_ROWS, pf.EXPECTED_TEST_CATEGORIES = old_rows, old_cats
        assert any("leaked split" in p for p in problems), problems


def t_preflight_passes_clean():
    with tempfile.TemporaryDirectory() as td:
        _write_parquet(os.path.join(td, "anomalyxl-precise", "data.parquet"),
                       ["a", "b", "c", "d"])
        _write_parquet(os.path.join(td, "anomalyxl-train", "anomalyxl-precise",
                                    "train.parquet"), ["x", "y"])
        _write_parquet(os.path.join(td, "anomalyxl-train", "anomalyxl-precise",
                                    "val.parquet"), ["z"])
        old_rows, old_cats = pf.EXPECTED_TEST_ROWS, pf.EXPECTED_TEST_CATEGORIES
        pf.EXPECTED_TEST_ROWS = {"precise": 4}
        pf.EXPECTED_TEST_CATEGORIES = {}
        try:
            problems = []
            pf.check_variant(td, "precise", problems)
        finally:
            pf.EXPECTED_TEST_ROWS, pf.EXPECTED_TEST_CATEGORIES = old_rows, old_cats
        assert problems == [], problems


# ---- 6. preflight logic against the REAL quarantined leak ------------------

def t_real_quarantined_leak_detected():
    ds = os.path.join(ROOT, "timerlm", "anomalyXL", "datasets")
    test_p = os.path.join(ds, "anomalyxl-precise", "data.parquet")
    leaked = os.path.join(ds, "_QUARANTINE_LEAKED", "anomalyxl-precise",
                          "val.parquet")
    if not (os.path.isfile(test_p) and os.path.isfile(leaked)):
        raise AssertionError("real files not found (layout changed?)")
    inter = pf._ids(leaked) & pf._ids(test_p)
    assert len(inter) == 77, f"expected the real 77-row leak, got {len(inter)}"


def t_real_datasets_root_now_clean():
    ds = os.path.join(ROOT, "timerlm", "anomalyXL", "datasets")
    problems = []
    pf.check_variant(ds, "precise", problems)
    assert problems == [], problems


# ---- 7. diagnostics --------------------------------------------------------

def t_strict_flags_string_false():
    v = dg.strict_violations("localize", {"present": "false"})
    assert any(x.startswith("present_not_bool") for x in v), v
    assert dg.strict_violations("localize",
                                {"present": False}) == []
    assert dg.strict_violations(
        "localize", {"present": True, "start": 1, "end": 2}) == []
    v = dg.strict_violations("localize", {"present": True, "start": "1", "end": 2.5})
    assert "start_not_int" in v and "end_not_int" in v


def t_strict_counts_malformed_events():
    v = dg.strict_violations("localize_all_channels",
                             {"anomalies": [{"channel": "c1", "start": 1, "end": 2},
                                            {"start": 3, "end": 4}]})
    assert "malformed_events:1" in v, v


def t_localize_decomposition_known_iou():
    entries = [
        {"ground_truth": 'localize:{"present": true, "start": 10, "end": 30}',
         "predicted_answer": '{"present": true, "start": 20, "end": 40}',
         "category": "localize", "primary": 1 / 3},
        {"ground_truth": 'localize:{"present": false}',
         "predicted_answer": '{"present": false}',
         "category": "localize", "primary": 1.0},
        {"ground_truth": 'localize:{"present": false}',
         "predicted_answer": '{"present": "false"}',   # string → officially TRUE
         "category": "localize", "primary": 0.0},
        {"ground_truth": 'localize:{"present": true, "start": 0, "end": 10}',
         "predicted_answer": '{"present": false}',
         "category": "localize", "primary": 0.0},
    ]
    r = dg.localize_report(entries)
    assert r["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}, r["confusion"]
    assert abs(r["positives_mean_iou"] - (1 / 3 + 0.0) / 2) < 1e-9
    assert abs(r["matched_mean_iou"] - 1 / 3) < 1e-9
    assert r["negatives_accuracy"] == 0.5


def t_bootstrap_deterministic_and_paired():
    ci1 = dg.boot_ci([0.1, 0.2, 0.3, 0.4])
    ci2 = dg.boot_ci([0.1, 0.2, 0.3, 0.4])
    assert ci1 == ci2, "bootstrap not deterministic"
    pb = dg.paired_boot([0.1] * 50)
    assert pb["significant"] and pb["mean_diff"] > 0


def main():
    print("== integrity test suite ==")
    for name, fn in sorted(
            ((k, v) for k, v in globals().items() if k.startswith("t_")),
            key=lambda kv: kv[0]):
        check(name, fn)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
