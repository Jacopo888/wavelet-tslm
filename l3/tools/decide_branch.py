#!/usr/bin/env python3
"""Scelta del ramo notturno secondo la tabella dichiarata in PROTOCOL-L3.md.

Legge due summary gia' prodotti (gate del contenuto sulle finestre autonome e
punteggio dei selettori) e scrive `DECISION.json`. Nessun modello, nessuna GPU.

La tabella e' fissata prima di conoscere i numeri: questo strumento non contiene
soglie proprie, le riceve da riga di comando e le registra nel risultato.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

BRANCHES = ("selector_sft", "grpo", "report_only")


def _delta(node: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not node:
        return None
    return {"mean": float(node["mean"]), "ci_low": float(node["ci95"][0]),
            "ci_high": float(node["ci95"][1]), "n": int(node.get("n", 0))}


def read_content(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    overall = data["overall"]
    return {
        "source": str(path),
        "gate_passed": bool(data.get("gate_passed", False)),
        "delta": _delta(overall.get("true_minus_best_control")),
        "condition_means": overall.get("condition_means"),
        "subhypothesis": data.get("registered_subhypothesis"),
        "width_ladder": {
            name: {"mean": entry["mean"],
                   "delta_vs_self_true": _delta(entry.get("delta_vs_self_true")),
                   "contained_rate": entry.get("contained_rate")}
            for name, entry in (data.get("width_ladder") or {}).items()
        },
    }


def read_scores(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    arms = {}
    for name, arm in data["arms"].items():
        paired = arm.get("paired") or {}
        arms[name] = {
            "mean": arm.get("mean"),
            "contained_rate": arm.get("contained_rate"),
            "delta_vs_policy": _delta(paired.get("delta_vs_policy")),
        }
    return {"source": str(path), "policy_mean": data["baseline"]["mean"], "arms": arms}


def decide(content: Mapping[str, Any], scores: Mapping[str, Any], *,
           min_content_delta: float, min_window_delta: float,
           min_selector_delta: float) -> dict[str, Any]:
    """Tabella dichiarata: contenuto -> utilita' della finestra -> scelta del ramo."""
    reasons: list[str] = []
    content_delta = content.get("delta")
    window = (scores["arms"].get("oracle_matched") or {}).get("delta_vs_policy")
    selector_arms = {
        name: arm["delta_vs_policy"] for name, arm in scores["arms"].items()
        if name != "oracle_matched" and arm.get("delta_vs_policy")
    }
    best_selector, selector = None, None
    if selector_arms:
        best_selector = max(selector_arms, key=lambda name: selector_arms[name]["mean"])
        selector = selector_arms[best_selector]

    def passes(delta: Mapping[str, Any] | None, threshold: float) -> bool:
        return bool(delta and delta["mean"] >= threshold and delta["ci_low"] > 0.0)

    content_ok = passes(content_delta, min_content_delta)
    window_ok = passes(window, min_window_delta)
    selector_ok = passes(selector, min_selector_delta)

    if not content_ok:
        branch = "report_only"
        reasons.append(
            "il contenuto delle finestre scelte dalla policy non porta un vantaggio "
            f"appaiato >= {min_content_delta} con CI oltre lo zero: nessuna run lunga")
    elif not window_ok:
        branch = "report_only"
        reasons.append(
            f"finestre migliori a parita' di budget non valgono >= {min_window_delta} "
            "di primary: il limite e' la lettura, non il puntamento; l'esperimento sul "
            "lettore (encoder/proiettore sbloccati) non e' nel kit di stanotte")
    elif selector_ok:
        branch = "selector_sft"
        reasons.append(
            f"il selettore {best_selector} supera la policy di "
            f"{selector['mean']:+.4f} con CI oltre lo zero: si insegna alla policy a "
            "proporre quelle coordinate")
    else:
        branch = "grpo"
        reasons.append(
            "il puntamento conta ma nessun selettore deterministico batte la policy "
            f"di >= {min_selector_delta}: si affina il comportamento esistente con GRPO")
    return {
        "branch": branch,
        "reasons": reasons,
        "criteria": {
            "min_content_delta": min_content_delta,
            "min_window_delta": min_window_delta,
            "min_selector_delta": min_selector_delta,
            "declared_in": "l3/PROTOCOL-L3.md, sezione Catena notturna condizionale",
        },
        "measured": {
            "content_delta": content_delta,
            "window_delta": window,
            "best_selector": best_selector,
            "selector_delta": selector,
            "checks": {"content_ok": content_ok, "window_ok": window_ok,
                       "selector_ok": selector_ok},
        },
        "content": content,
        "scores": scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--content-summary", required=True, type=Path)
    parser.add_argument("--scores-summary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-content-delta", type=float, default=0.03)
    parser.add_argument("--min-window-delta", type=float, default=0.05)
    parser.add_argument("--min-selector-delta", type=float, default=0.03)
    args = parser.parse_args()
    decision = decide(
        read_content(args.content_summary.expanduser().resolve()),
        read_scores(args.scores_summary.expanduser().resolve()),
        min_content_delta=args.min_content_delta,
        min_window_delta=args.min_window_delta,
        min_selector_delta=args.min_selector_delta,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"L3_BRANCH={decision['branch']}")
    for reason in decision["reasons"]:
        print(f"  motivo: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
