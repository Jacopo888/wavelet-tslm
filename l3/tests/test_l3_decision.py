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
