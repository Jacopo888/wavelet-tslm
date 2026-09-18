"""CPU-only tests for the L2 kit (no GPU, no prime-rl): config invariants, note variant,
monitor parser. Run from the repo root: `python -m pytest l2/tests -q`."""

from __future__ import annotations

import asyncio
import importlib.util
import ast
import json
import math
import os
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
L2 = ROOT / "l2"
CFG = L2 / "configs"
OVERLAY = L2 / "overlay" / "timerlm"


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            for i, x in enumerate(v):
                out.update(_flatten(x, f"{key}[{i}]."))
        else:
            out[key] = v
    return out


# ---------------------------------------------------------------- arm configs
ALLOWED_DIFF = {
    "output_dir",
    "orchestrator.train.env[0].harness.env.RLM_SKILLS",
}


def test_arms_differ_only_in_skill_and_names():
    code = _flatten(_load(CFG / "l2_pilot_code.toml"))
    wav = _flatten(_load(CFG / "l2_pilot_wavelet.toml"))
    diff = {k for k in set(code) | set(wav) if code.get(k) != wav.get(k)}
    assert diff == ALLOWED_DIFF, f"unexpected differences: {sorted(diff ^ ALLOWED_DIFF)}"
    assert wav["orchestrator.train.env[0].harness.env.RLM_SKILLS"] == "wavelet_zoom"
    assert "orchestrator.train.env[0].harness.env.RLM_SKILLS" not in code


def test_arm_configs_keep_the_declared_pilot_semantics():
    for arm in ("code", "wavelet"):
        c = _load(CFG / f"l2_pilot_{arm}.toml")
        # batch_size = rollouts/step at prime-rl c5afcaa2e (divisible by group_size)
        assert c["orchestrator"]["batch_size"] == 64 and c["orchestrator"]["group_size"] == 8
        assert c["orchestrator"]["batch_size"] % c["orchestrator"]["group_size"] == 0
        assert "weights" not in c["ckpt"], "[ckpt.weights] is not a shared field at this pin"
        assert c["orchestrator"]["advantage"]["length_penalty"]["coef"] == 0.1
        assert c["orchestrator"]["advantage"]["length_penalty"]["gate_by_correctness"] is False
        env = c["orchestrator"]["train"]["env"][0]
        assert env["taskset"]["id"] == "timeseries-qa-v1"
        assert env["taskset"]["repo_or_path"].endswith("/l2/data/val"), "scorer dataset must be the pinned local copy"
        assert env["harness"]["id"] == "timeseries-qa-v1"
        henv = env["harness"]["env"]
        assert int(henv["RLM_MAX_TURNS"]) == 15 and int(henv["RLM_MAX_TOKENS"]) < c["seq_len"]
        assert env["harness"]["summarize_at_tokens"] < c["seq_len"]
        assert c["orchestrator"]["renderer"]["name"] == "qwen3.5"
        assert c["deployment"]["num_infer_gpus"] == 0 and c["deployment"]["num_train_gpus"] == 1
        assert "inference" not in c, "the server is standalone; no [inference] block"
        assert "wandb" not in c, "no [wandb] block (disabled-run teardown crash at the pin)"
        assert c["weight_broadcast"]["type"] == "filesystem"
        assert c["trainer"]["model"]["lora"]["rank"] == 32
        assert c["ckpt"]["keep_interval"] % c["ckpt"]["interval"] == 0, "phase boundary must be a checkpoint"
        assert c["ckpt"]["keep_interval"] < c["max_steps"]
        ev = c["orchestrator"]["eval"]["env"][0]
        assert ev["id"] == "timeseries-qa" and ev["args"]["repo_or_path"].endswith("/l2/data/val")
        assert ev["args"]["max_turns"] == int(henv["RLM_MAX_TURNS"])
        assert c["model"]["name"] == "/root/timerlm/l2/model_snapshot"


def test_smoke_overlay_changes_only_load():
    s = _flatten(_load(CFG / "l2_smoke.toml"))
    allowed_prefixes = ("max_steps", "output_dir", "orchestrator.batch_size", "orchestrator.group_size",
                        "orchestrator.max_inflight_rollouts", "orchestrator.eval.", "ckpt.")
    for k in s:
        assert k.startswith(allowed_prefixes), f"smoke overlay touches {k}"
    assert s["max_steps"] == 2
    assert s["orchestrator.batch_size"] % s["orchestrator.group_size"] == 0


def test_smoke_postcheck_uses_current_rollout_monitor():
    """The smoke gate and monitor must use the same artifact contract."""
    src = (L2 / "node_l2_smoke.sh").read_text(encoding="utf-8")
    assert "orchestrator_metrics.jsonl" not in src
    assert '--expect-rollouts "$expected_rollouts"' in src
    assert 'checkpoints/step_$expected_steps/trainer/.metadata' in src
    assert 'weights/step_$expected_steps/model.safetensors' in src
    assert 'rl_run c_rl_tiny code 8 2 ' in src
    assert 'rl_run e_rescue_wavelet wavelet "${L2_EXPECT_ROLLOUTS:-64}" 5 ' in src


def test_inference_config_matches_arms():
    inf = _load(CFG / "inference_l2.toml")
    arm = _load(CFG / "l2_pilot_code.toml")
    assert inf["model"]["name"] == arm["model"]["name"]
    assert inf["model"]["max_model_len"] == arm["seq_len"]
    assert inf["enable_lora"] is True and inf["max_lora_rank"] == arm["trainer"]["model"]["lora"]["rank"]
    assert inf["model"]["tool_call_parser"] == "qwen3_coder"
    assert "reasoning_parser" not in inf["model"], "plain JSON finals must remain in content"
    assert inf["vllm_extra"]["enable_sleep_mode"] is False
    assert inf["vllm_extra"]["max_num_seqs"] == arm["orchestrator"]["max_inflight_rollouts"]
    port = inf["server"]["port"]
    assert arm["orchestrator"]["student"]["client"]["base_url"] == [f"http://localhost:{port}/v1"]
    assert 0.3 <= inf["gpu_memory_utilization"] <= 0.5


# ---------------------------------------------------------------- note variant
PAPER_NOTE_PREFIX = (
    "This is a critical application: when relevant, precisely locate the anomaly's "
    "start and end using statistical thresholds — be precise. The time series is in "
    'the file `context.json` in your working directory, shaped as '
    '{"series": {channel: [values], ...}}; load it with json. '
)
PAPER_NOTE_SUFFIX = (
    "Always read the actual channel keys from the data (do not guess names). "
    "Analyze it with numpy and scipy in the IPython tool. When you are done, stop "
    "calling the tool and state your final answer as a single JSON object matching "
    'the question\'s schema exactly (e.g. {"start": 12, "end": 87}), with no prose '
    "and no markdown fences. Format failures score zero."
)


def _extract_build_note(path: Path):
    """Load `_build_note` from a source file without importing its heavy siblings."""
    src = path.read_text(encoding="utf-8")
    m = re.search(r"(WAVELET_NOTE_SENTENCE = \([\s\S]*?\)\n)[\s\S]*?(def _build_note\(values_note: str\) -> str:[\s\S]*?return \([\s\S]*?\n    \)\n)", src)
    assert m, f"_build_note not found in {path}"
    ns: dict = {"os": os}
    exec(m.group(1) + "\n" + m.group(2), ns)
    return ns["_build_note"], ns["WAVELET_NOTE_SENTENCE"]


@pytest.mark.parametrize("rel", [
    "environments/timeseries_qa_v1/timeseries_qa_v1/taskset.py",
    "scripts/eval_v1_rlm.py",
])
def test_default_note_is_the_paper_text(rel, monkeypatch):
    build, _ = _extract_build_note(OVERLAY / rel)
    monkeypatch.delenv("TIMERLM_NOTE_VARIANT", raising=False)
    assert build("VN. ") == PAPER_NOTE_PREFIX + "VN. " + PAPER_NOTE_SUFFIX
    monkeypatch.setenv("TIMERLM_NOTE_VARIANT", "wavelet")
    note = build("VN. ")
    assert "print(await wavelet_zoom())" in note
    assert "do not import it" in note and "do not open or print `context.json`" in note
    assert "load it with json" not in note and "Analyze it with numpy" not in note
    assert note.endswith("Format failures score zero.")
    monkeypatch.setenv("TIMERLM_NOTE_VARIANT", "bogus")
    with pytest.raises(ValueError):
        build("VN. ")


def test_note_copies_in_sync():
    a, sa = _extract_build_note(OVERLAY / "environments/timeseries_qa_v1/timeseries_qa_v1/taskset.py")
    b, sb = _extract_build_note(OVERLAY / "scripts/eval_v1_rlm.py")
    assert sa == sb
    for variant in ("default", "wavelet"):
        os.environ["TIMERLM_NOTE_VARIANT"] = variant
        try:
            assert a("X ") == b("X ")
        finally:
            os.environ.pop("TIMERLM_NOTE_VARIANT", None)


def _extract_schema_validator(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"_is_int", "_valid_final_schema"}]
    assert {n.name for n in funcs} == {"_is_int", "_valid_final_schema"}
    mod = ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[]))
    ns = {"json": json, "math": math}
    exec(compile(mod, str(path), "exec"), ns)
    return ns["_valid_final_schema"]


def test_format_bootstrap_is_strict_and_task_aware():
    valid = _extract_schema_validator(OVERLAY / "environments/timeseries_qa_v1/timeseries_qa_v1/taskset.py")
    cases = [
        ('{"present":true,"start":2,"end":5}', 'localize:{"present":true,"start":2,"end":5,"_L":10}'),
        ('{"kind":"Level Shift","start":2,"end":5}', 'classify_with_evidence:{"kind":"Level Shift","start":2,"end":5,"_L":10}'),
        ('{"magnitude_sigma":2.5}', 'measure_magnitude:{"magnitude_sigma":2.5}'),
        ('{"anomalies":[]}', 'localize_all_channels:{"anomalies":[],"_L":10}'),
        ('{"direction":"lead","lag_samples":3}', 'lead_lag_with_magnitude:{"direction":"lead","lag_samples":3,"_L":10}'),
    ]
    assert all(valid(pred, gold) for pred, gold in cases)
    assert not valid('', cases[0][1])
    assert not valid('prose {"present":true,"start":2,"end":5}', cases[0][1])
    assert not valid('{"present":true}', cases[0][1])
    assert not valid('{"magnitude_sigma":NaN}', cases[2][1])
    assert not valid('{"direction":"sideways","lag_samples":3}', cases[4][1])


def test_forced_final_is_telemetry_not_negative_reward():
    src = (OVERLAY / "environments/timeseries_qa_v1/timeseries_qa_v1/taskset.py").read_text(encoding="utf-8")
    assert "async def answer_format" in src and "return 0.05 if _valid_final_schema" in src
    assert "@metric\n    async def forced_final" in src
    assert "async def forced_final_penalty" not in src


def test_overlay_matches_local_checkout_when_present():
    """The tracked overlay is the source of truth; the gitignored checkout must not drift."""
    for rel in ("environments/timeseries_qa_v1/timeseries_qa_v1/taskset.py",
                "environments/timeseries_qa_v1/timeseries_qa_v1/generator.py",
                "scripts/eval_v1_rlm.py", "rlm-harness/src/rlm/prompt.py",
                "rlm-harness/src/rlm/skills/__init__.py",
                "rlm-harness/src/rlm/skills/wavelet_zoom.py", "rlm-harness/pyproject.toml"):
        local = ROOT / "timerlm" / rel
        if local.exists():
            local_text = local.read_text(encoding="utf-8").rstrip()
            overlay_text = (OVERLAY / rel).read_text(encoding="utf-8").rstrip()
            assert local_text == overlay_text, f"overlay drift: {rel}"


def test_wavelet_overview_is_compact_and_finds_a_strong_interval(tmp_path, monkeypatch):
    path = OVERLAY / "rlm-harness/src/rlm/skills/wavelet_zoom.py"
    spec = importlib.util.spec_from_file_location("l2_wavelet_zoom", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = __import__("numpy").random.default_rng(7)
    x = rng.normal(size=4096)
    gold_start, gold_end = 1700, 1850
    x[gold_start:gold_end] += 8.0
    (tmp_path / "context.json").write_text(json.dumps({"series": {"ch_00": x.tolist()}}))
    monkeypatch.chdir(tmp_path)
    text = asyncio.run(mod.run())
    assert len(text) < 4096 and "WAVELET_OVERVIEW" in text
    _, candidates = mod._rank_candidates({"ch_00": x})
    def iou(c):
        overlap = max(0, min(c["end"], gold_end) - max(c["start"], gold_start))
        union = max(c["end"], gold_end) - min(c["start"], gold_start)
        return overlap / union
    assert max(map(iou, candidates[:3])) >= 0.25


def test_bootstrap_curriculum_is_short_and_symmetric():
    src = (OVERLAY / "environments/timeseries_qa_v1/timeseries_qa_v1/generator.py").read_text(encoding="utf-8")
    assert 'ANOMALYXL_CURRICULUM' in src and '{"localize", "measure_magnitude"}' in src
    assert 'magnitude_range=(6.0, 10.0)' in src and 'L <= 32768' in src
    run = (L2 / "node_l2_run.sh").read_text(encoding="utf-8")
    assert 'L2_PHASE1_STEPS:-10' in run and 'ANOMALYXL_CURRICULUM=bootstrap' in run
    for arm in ("code", "wavelet"):
        assert _load(CFG / f"l2_pilot_{arm}.toml")["ckpt"]["keep_interval"] == 10


def test_eval_script_defaults_are_the_paper_protocol():
    src = (OVERLAY / "scripts/eval_v1_rlm.py").read_text(encoding="utf-8")
    assert 'os.environ.get("RLM_MAX_OUTPUT", "8192")' in src
    assert 'os.environ.get("RLM_SUMMARIZE_AT_TOKENS", "45000")' in src
    assert 'os.environ.get("RLM_MAX_TURNS", "15")' in src
    assert "shutil.rmtree(ws, ignore_errors=True)" in src
    assert "if out_log.exists()" in src
    assert "except json.JSONDecodeError:" in src


def test_eval_drops_the_base_shard_index_for_merged_checkpoints():
    src = (L2 / "node_l2_eval.sh").read_text(encoding="utf-8")
    assert '[ -f "$W/model.safetensors" ]' in src
    assert 'rm -f "$MODEL_DIR/model.safetensors.index.json"' in src


def test_pilot_restarts_inference_server_for_each_arm():
    """Sequential arms must not share a loaded LoRA adapter or prefix cache."""
    src = (L2 / "node_l2_run.sh").read_text(encoding="utf-8")
    assert "if ! server_up" not in src
    assert "tmux kill-session -t l2srv" in src
    assert "starting a fresh inference server for arm $ARM" in src
    assert "refusing a non-isolated run" in src


def test_pilot_requires_the_full_batch_smoke_gate():
    src = (L2 / "node_l2_run.sh").read_text(encoding="utf-8")
    assert "for s in a b c d e; do" in src


def test_pilot_passes_critical_overrides_through_nested_tmux():
    src = (L2 / "node_l2_run.sh").read_text(encoding="utf-8")
    runtime = next(line for line in src.splitlines() if line.startswith('RUNTIME_ENV='))
    for assignment in (
        "PRIME_RL_DIST_BACKEND='cpu:gloo,cuda:gloo'",
        "PRIME_RL_TRAINER_ALLOC_CONF=''",
        "FLA_TILELANG=0",
        "PRIME_RL_LORA_GROUPED_MM=0",
    ):
        assert assignment in runtime
    assert "-u ANOMALYXL_P_PRESENT -u ANOMALYXL_ALLCH_MIN_K -u ANOMALYXL_CURRICULUM" in runtime
    assert "$RUNTIME_ENV uv run --no-sync inference" in src
    assert "$phase_env uv run --no-sync rl" in src


# ---------------------------------------------------------------- monitor
def _monitor():
    spec = importlib.util.spec_from_file_location("l2_monitor", L2 / "l2_monitor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_monitor_parses_step_line_and_rollouts(tmp_path):
    m = _monitor()
    out = tmp_path / "run"
    (out / "logs").mkdir(parents=True)
    (out / "logs" / "orchestrator.log").write_text(
        "2026-09-04 [SUCCESS] Step 3 |  12m 4s | Reward 0.1234 | Trainable 60/62 (96.8%) | Turns 7.5 | "
        "Branches 1.0 | Max Off-Policy 2 | Error 3.1% | Truncation 4.8%\n", encoding="utf-8")
    rdir = out / "run_default" / "rollouts" / "step_3"; rdir.mkdir(parents=True)
    def rec(idx, cat, reward, forced, calls, zoom=0, err=None, final="", format_valid=False):
        tcs = [{"name": "call_python_repl", "arguments": {"code": "await wavelet_zoom(mode=\"overview\")" if i < zoom else "print(1)"}} for i in range(calls)]
        nodes = [{"message": {"role": "system", "content": "s"}}, {"message": {"role": "user", "content": "u"}},
                 {"message": {"role": "assistant", "content": final, "tool_calls": tcs}, "usage": {"prompt_tokens": 100, "completion_tokens": 40}}]
        return {"task": {"idx": idx, "category": cat}, "rewards": {"correct_answer": reward, "answer_format": 0.05 if format_valid else 0.0},
                "metrics": {"forced_final": 1.0 if forced else 0.0},
                "nodes": nodes, "stop_condition": "max_turns" if forced else "agent_completed", "errors": [err] if err else [],
                "is_completed": True, "timing": {"generation": {"start": 0.0, "end": 30.0}}}
    rows = [rec(1, "localize", 1.0, False, 3, zoom=1, final='{"present":false,"start":null,"end":null}', format_valid=True),
            rec(1, "localize", 0.0, True, 5), rec(2, "measure_magnitude", 0.0, True, 4),
            rec(2, "measure_magnitude", 0.0, True, 4, err="boom")]
    (rdir / "train_rollouts.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    logs = m.parse_log_steps(out / "logs" / "orchestrator.log")
    assert logs[3]["error_rate"] == pytest.approx(0.031) and logs[3]["trainable"] == 60
    summary = m.summarize_step(3, rdir / "train_rollouts.jsonl")
    assert summary["rollouts"] == 4 and summary["errored"] == 1 and summary["forced_final"] == 3
    assert summary["reward_positive"] == 1 and summary["correct_positive"] == 1 and summary["format_valid"] == 1
    assert summary["empty_final"] == 3 and summary["empty_final_rate"] == pytest.approx(0.75) and summary["wavelet_zoom_calls"] == 1
    assert summary["per_category"]["localize"]["n"] == 2 and summary["groups"] == 2 and summary["zero_variance_groups"] == 1
    assert summary["tool_calls_mean"] == pytest.approx(4.0) and summary["completion_tokens_mean"] == pytest.approx(40.0)
    row = m.coverage_row(3, summary, logs[3], expect_rollouts=4)
    assert not row["rollouts_mismatch"] and not row["rollout_data_missing"] and not row["log_step_missing"]
    line = m.fmt(row)
    assert "[L2-Coverage] step 3" in line and "rollouts 4/4" in line and "wavelet_zoom 1" in line and "zero_var_groups 1/2" in line
    assert "format 1" in line and "empty_final 3 (0.750)" in line
    mismatch = m.coverage_row(3, summary, logs[3], expect_rollouts=3)
    assert mismatch["rollouts_mismatch"] and "ROLLOUTS_MISMATCH" in m.fmt(mismatch)
    missing = m.coverage_row(3, None, logs[3], expect_rollouts=4)
    assert missing["rollout_data_missing"] and missing["rollouts_mismatch"] and "SOURCE_MISSING" in m.fmt(missing)


def test_posthoc_monitor_reads_both_phase_logs():
    src = (L2 / "node_l2_run.sh").read_text(encoding="utf-8")
    assert '--log-file "$LOGS/${ARM}_phase1.log"' in src
    assert '--log-file "$LOGS/${ARM}_phase2.log"' in src
    assert 'rm -f "$RUN/coverage_final.jsonl"' in src


def test_collapse_guard_requires_two_consecutive_bad_steps():
    m = _monitor()
    good = {"step": 0, "log_trainable": 64, "empty_final_rate": 0.2}
    bad1 = {"step": 1, "log_trainable": 7, "empty_final_rate": 0.2}
    bad2 = {"step": 2, "log_trainable": 64, "empty_final_rate": 0.9}
    assert m.collapse_guard([good, bad1], 2, 8, 0.8)[0]
    ok, reason = m.collapse_guard([good, bad1, bad2], 2, 8, 0.8)
    assert not ok and "step 1" in reason and "step 2" in reason
    assert m.collapse_guard([good, bad1, {**good, "step": 2}], 2, 8, 0.8)[0]
