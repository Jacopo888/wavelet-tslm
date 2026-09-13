"""Test della tabella di decisione notturna (l3/tools/decide_branch.py)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l3.tools.decide_branch import decide


def content(mean, low):
    return {"delta": {"mean": mean, "ci_low": low, "ci_high": mean + 0.1, "n": 200}}


def scores(oracle, oracle_low, selector, selector_low):
    return {
        "policy_mean": 0.13,
        "arms": {
            "oracle_matched": {
                "mean": 0.2, "contained_rate": 1.0,
                "delta_vs_policy": {"mean": oracle, "ci_low": oracle_low,
                                    "ci_high": oracle + 0.05, "n": 230}},
            "wavelet_in_policy_channel": {
                "mean": 0.16, "contained_rate": 0.77,
                "delta_vs_policy": {"mean": selector, "ci_low": selector_low,
                                    "ci_high": selector + 0.05, "n": 230}},
            "wavelet_matched": {
                "mean": 0.15, "contained_rate": 0.69,
                "delta_vs_policy": {"mean": selector - 0.01,
                                    "ci_low": selector_low - 0.01,
                                    "ci_high": selector + 0.04, "n": 230}},
        },
    }


THRESHOLDS = {"min_content_delta": 0.03, "min_window_delta": 0.05,
              "min_selector_delta": 0.03}


def test_no_content_effect_stops_the_chain():
    out = decide(content(0.01, -0.02), scores(0.08, 0.04, 0.04, 0.01), **THRESHOLDS)
    assert out["branch"] == "report_only"
    assert out["measured"]["checks"] == {"content_ok": False, "window_ok": True,
                                         "selector_ok": True}


def test_windows_that_do_not_help_point_at_the_reader():
    out = decide(content(0.07, 0.03), scores(0.02, -0.01, 0.01, -0.02), **THRESHOLDS)
    assert out["branch"] == "report_only"
    assert "lettura" in out["reasons"][0]


def test_a_winning_selector_selects_the_supervised_branch():
    out = decide(content(0.07, 0.03), scores(0.08, 0.04, 0.045, 0.012), **THRESHOLDS)
    assert out["branch"] == "selector_sft"
    assert out["measured"]["best_selector"] == "wavelet_in_policy_channel"


def test_a_tied_selector_selects_grpo():
    out = decide(content(0.07, 0.03), scores(0.08, 0.04, 0.01, -0.01), **THRESHOLDS)
    assert out["branch"] == "grpo"
    assert out["measured"]["checks"]["selector_ok"] is False


def test_thresholds_are_recorded_in_the_decision():
    out = decide(content(0.07, 0.03), scores(0.08, 0.04, 0.045, 0.012), **THRESHOLDS)
    assert out["criteria"]["min_window_delta"] == 0.05
    assert "PROTOCOL-L3" in out["criteria"]["declared_in"]


def test_w2_verdict_requires_both_objectives(tmp_path):
    """La tabella del verdetto W2 non deve poter passare con un solo obiettivo."""
    import json
    import subprocess
    import sys

    def write(name, payload):
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def reading(mean, low, high):
        return {"mean": mean, "ci95": [low, high], "n": 208}

    def run(multi, aggregate, primary, contained, gate_passed):
        vs_w1 = write("vs_w1.json", {"per_category": {
            "localize_all_channels": {"reading": reading(*multi)},
            "localize": {"reading": reading(0.0, -0.02, 0.02)}}})
        vs_z1 = write("vs_z1.json", {"reading": reading(*aggregate),
                                     "content": reading(0.1, 0.08, 0.12),
                                     "per_category": {}})
        rows = tmp_path / "rows.jsonl"
        rows.write_text(chr(10).join(
            json.dumps({"id": f"r{i}", "primary": primary}) for i in range(10)),
            encoding="utf-8")
        geometry = write("geo.json", {"overall": {"policy": {"contained_rate": contained}}})
        gate = write("gate.json", {"gate_passed": gate_passed, "checks": {"x": gate_passed}})
        out = tmp_path / "VERDICT.json"
        subprocess.run([sys.executable, str(ROOT / "l3" / "tools" / "w2_verdict.py"),
                        "--vs-w1", str(vs_w1), "--vs-z1bis", str(vs_z1),
                        "--policy-json", str(geometry), "--policy-rows", str(rows),
                        "--geometry", str(geometry), "--gate", str(gate),
                        "--out", str(out)], check=True, capture_output=True)
        return json.loads(out.read_text(encoding="utf-8"))["verdict"]

    assert run((0.05, 0.01, 0.09), (-0.01, -0.03, 0.01), 0.36, 0.62, True) ==         "w2_becomes_candidate"
    # lettura recuperata ma operativo sotto soglia
    assert run((0.05, 0.01, 0.09), (-0.01, -0.03, 0.01), 0.30, 0.62, True) ==         "keep_w1_reading_only"
    # lettura non recuperata (CI del multicanale attraversa zero)
    assert run((0.02, -0.01, 0.05), (-0.01, -0.03, 0.01), 0.36, 0.62, True) ==         "keep_w1_grpo_with_constraint"
    # aggregato ancora compatibile con un calo rilevante
    assert run((0.05, 0.01, 0.09), (-0.03, -0.06, 0.00), 0.36, 0.62, True) ==         "keep_w1_grpo_with_constraint"
