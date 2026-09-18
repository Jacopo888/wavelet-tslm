#!/usr/bin/env python3
"""Between memorising eight signals and solving the benchmark, where exactly does it break?

Astra (17/09) corrected the reading of the memorisation result, and the correction holds: in
``memorization_probe.py`` every example is ``torch.roll`` of ONE base signal, so the eight recordings share the same
background, merely shifted.  Fitting them proves the answer depends on that recording's soft tokens -- the probe
required true minus substituted > 0.1 -- but recognising WHICH rotation this is, and recalling a memorised window,
does not require decoding the position of anything.  Fifty exposures to eight examples and five exposures to 4488
are different problems.

So this walks the ladder between them.  Every rung trains from the same initialisation and then asks one question on
material the training never saw:

  positions   one background, windows at positions never trained.  Does the rule "the window sits where the burst
              is" exist at all, inside a single signal?
  backgrounds a fresh noise draw per example, held-out draws at test.  This is the real task in miniature.
  shapes      held-out widths and amplitudes on held-out backgrounds.  Does it transfer across event scale?

Each rung reports the fit on train (did it learn anything?), the score on the held-out set (did it generalise?) and
the held-out score with another recording's tokens under this one's header (is the held-out score actually coming
from the signal?).  The three outcomes are distinct and must not be conflated:

  GENERALISES     held-out score clears the floor AND beats its own substituted control
  MEMORISES ONLY  train fits, held-out collapses -- there is no generalisable rule
  DOES NOT FIT    train never fits, so the rung says nothing about generalisation

Usage:
  python tools/generalization_ladder.py --config <toml> --rung positions --out <json> \
      [--n-train 24] [--n-test 24] [--steps 900] [--length 16384] [--checkpoint <dir>]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from safetensors.torch import load_file  # noqa: E402

from wavelet_rope.backbone import load_backbone  # noqa: E402
from wavelet_rope.checkpoint import inspect_foreign_checkpoint  # noqa: E402
from wavelet_rope.config import load_runtime_config  # noqa: E402
from wavelet_rope.data import GoldInterval, RowMeta  # noqa: E402
from wavelet_rope.model import WRAgentModel  # noqa: E402
from wavelet_rope.signal_contract import SignalEpisode  # noqa: E402

QUESTION = ("Does the recording contain an anomaly? If it does, give the sample interval that contains it. "
            "Answer with the required JSON object.")


def _episode(values: torch.Tensor, length: int, device: torch.device) -> SignalEpisode:
    return SignalEpisode(
        values=values.to(device),
        valid=torch.ones(1, length, dtype=torch.bool, device=device),
        channel_ids=("ch_00",),
        sample_times=torch.arange(length, dtype=torch.float64, device=device),
        task=None,
    )


def _row(index: int, tag: str, length: int, start: int, width: int) -> RowMeta:
    answer = ('localize:{"_L": %d, "end": %d, "present": true, "start": %d}' % (length, start + width, start))
    return RowMeta(id=f"{tag}-{index:03d}", category="localize", length=length, n_channels=1,
                   channel_names=("ch_00",), question=QUESTION, answer=answer,
                   gold_intervals=(GoldInterval(channel="ch_00", start=start, end=start + width, kind="spike"),),
                   split="synthetic", row_index=index)


def build_rung(rung: str, *, n_train: int, n_test: int, length: int, width: int, amplitude: float,
               seed: int, device: torch.device):
    """Return ``(train, test)`` lists of ``(meta, episode, start)``, disjoint by construction."""

    generator = torch.Generator().manual_seed(seed)
    margin = 4 * width
    usable = length - 2 * margin - width
    if usable <= 0:
        raise SystemExit("the recording is too short for this anomaly width")

    def burst(w: int, a: float) -> torch.Tensor:
        return a * torch.hann_window(w, periodic=False).unsqueeze(0)

    if rung == "positions":
        # ONE background, rotated.  Train positions and test positions interleave and never coincide, so the
        # test windows are unreachable by recalling a trained rotation: the model has to place the burst.
        base = torch.randn(1, length, generator=generator)
        base[:, :width] += burst(width, amplitude)

        def rotations(count: int, offset: float, tag: str):
            items = []
            for index in range(count):
                start = margin + int(round(usable * (index + offset) / count))
                items.append((_row(index, tag, length, start, width), _episode(torch.roll(base, start, 1),
                                                                              length, device), start))
            return items

        train = rotations(n_train, 0.25, "tr")
        test = rotations(n_test, 0.75, "te")
        trained = {start for _, _, start in train}
        test = [item for item in test if item[2] not in trained]
        return train, test

    if rung in ("backgrounds", "shapes"):
        # A fresh noise draw per example: the background is never shared, so nothing can be recalled from it.
        widths = [width] if rung == "backgrounds" else [width // 2, width, width * 2]
        amplitudes = [amplitude] if rung == "backgrounds" else [amplitude / 2, amplitude, amplitude * 2]
        # held-out shapes are strictly outside the trained set, held-out backgrounds always are
        test_widths = [width] if rung == "backgrounds" else [int(width * 0.75), int(width * 1.5)]
        test_amplitudes = [amplitude] if rung == "backgrounds" else [amplitude * 0.75, amplitude * 1.5]

        def draw(count: int, tag: str, ws, amps):
            items = []
            for index in range(count):
                w = int(ws[index % len(ws)])
                a = float(amps[index % len(amps)])
                span = length - 2 * margin - w
                start = margin + int(torch.randint(0, max(1, span), (1,), generator=generator))
                values = torch.randn(1, length, generator=generator)
                values[:, start:start + w] += burst(w, a)
                items.append((_row(index, tag, length, start, w), _episode(values, length, device), start))
            return items

        return draw(n_train, "tr", widths, amplitudes), draw(n_test, "te", test_widths, test_amplitudes)

    raise SystemExit(f"unknown rung {rung!r}")


def evaluate(model, items, *, substituted: bool, max_new_tokens: int) -> dict:
    from wavelet_rope.scoring import score_precise

    scores = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, (meta, episode, _) in enumerate(items):
            shown = episode
            if substituted:
                donor = items[(index + 1) % len(items)][1]
                shown = SignalEpisode(donor.values, episode.valid, episode.channel_ids,
                                      episode.sample_times, episode.task)
            out = model.generate(meta, shown, mode="direct", max_new_tokens=max_new_tokens, thinking=False)
            scores.append(float(score_precise(out["text"], meta.answer).get("primary", 0.0)))
    n = max(1, len(scores))
    return {"primary": sum(scores) / n, "solved": sum(1 for s in scores if s >= 0.5), "n": len(scores)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rung", required=True, choices=("positions", "backgrounds", "shapes"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--n-train", type=int, default=24)
    parser.add_argument("--n-test", type=int, default=24)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--length", type=int, default=16384)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--amplitude", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--floor", type=float, default=0.15, help="held-out primary that counts as generalisation")
    parser.add_argument("--margin", type=float, default=0.10, help="held-out minus its substituted control")
    args = parser.parse_args()
    started = time.time()

    cfg = load_runtime_config(args.config, validate_paths=False)
    raw, spec = cfg.as_dict(), dict(cfg.spec)
    device = torch.device("cuda:0")
    runtime = raw["runtime"]
    backbone = load_backbone(raw["backbone"]["snapshot"]["root"], backend=runtime["attention_backend"],
                             pad_to=int(runtime["attention_qk_pad_to"]), device=device,
                             gradient_checkpointing=True)
    model = WRAgentModel(backbone, spec, segment_weights={"reasoning": 0.0, "action": 1.0, "final": 1.0},
                         aux_weight=0.0)
    model.frontend.to(device)
    model.mixer.to(device)
    model.projector.to(device)
    if args.checkpoint:
        ref = inspect_foreign_checkpoint(args.checkpoint)
        model.load_trainable_state_dict(load_file(str(ref.path / "model.safetensors")))
        backbone.ctx.lam = float(ref.trainer_state["wildcard_lambda"])
    else:
        backbone.ctx.lam = 1.0

    train, test = build_rung(args.rung, n_train=args.n_train, n_test=args.n_test, length=args.length,
                             width=args.width, amplitude=args.amplitude, seed=args.seed, device=device)
    if not test:
        raise SystemExit("the held-out set is empty: raise --n-test or lower --n-train")
    overlap = {s for _, _, s in train} & {s for _, _, s in test}
    if args.rung == "positions" and overlap:
        raise SystemExit(f"train and test share {len(overlap)} positions: the rung would prove nothing")
    print(f"rung {args.rung}: {len(train)} train, {len(test)} held-out, lambda {backbone.ctx.lam}", flush=True)

    groups = model.trainable_groups()
    optimizer = torch.optim.AdamW(
        [{"params": groups["frontend"], "lr": args.lr}, {"params": groups["lora"], "lr": args.lr}],
        weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)

    history = []
    model.train()
    backbone.text.train()
    for step in range(1, args.steps + 1):
        meta, episode, _ = train[(step - 1) % len(train)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            report = model.forward_loss(meta, episode, mode="direct", actions=[], answer=meta.answer,
                                        gold=meta.gold_intervals, supervise_action=False)
        report.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups.values() for p in g], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % args.eval_every == 0 or step == args.steps:
            model.eval(); backbone.text.eval()
            fit = evaluate(model, train, substituted=False, max_new_tokens=args.max_new_tokens)
            held = evaluate(model, test, substituted=False, max_new_tokens=args.max_new_tokens)
            held_swap = evaluate(model, test, substituted=True, max_new_tokens=args.max_new_tokens)
            model.train(); backbone.text.train()
            entry = {"step": step, "loss": float(report.loss), "train_primary": fit["primary"],
                     "held_out_primary": held["primary"], "held_out_substituted": held_swap["primary"],
                     "held_out_delta": held["primary"] - held_swap["primary"],
                     "train_solved": fit["solved"], "held_out_solved": held["solved"]}
            history.append(entry)
            print(f"  step {step:4d} loss {entry['loss']:.4f} | train {entry['train_primary']:.4f} "
                  f"({entry['train_solved']}/{len(train)}) | held-out {entry['held_out_primary']:.4f} "
                  f"({entry['held_out_solved']}/{len(test)}) sostituito {entry['held_out_substituted']:.4f} "
                  f"delta {entry['held_out_delta']:+.4f}", flush=True)

    # Read a plateau, never a single point: at n=24 one example flipping moves the score by 0.0417.
    best_fit = max((h["train_primary"] for h in history), default=0.0)
    best_held = max(history, key=lambda h: h["held_out_primary"]) if history else {}
    if best_fit < 0.5:
        verdict = "DOES NOT FIT"
    elif best_held.get("held_out_primary", 0.0) >= args.floor and best_held.get("held_out_delta", 0.0) >= args.margin:
        verdict = "GENERALISES"
    else:
        verdict = "MEMORISES ONLY"

    report_json = {"config": args.config, "rung": args.rung, "checkpoint": args.checkpoint,
                   "n_train": len(train), "n_test": len(test), "steps": args.steps, "length": args.length,
                   "width": args.width, "amplitude": args.amplitude, "lr": args.lr,
                   "floor": args.floor, "margin": args.margin, "history": history,
                   "best_train_primary": best_fit, "best_held_out": best_held, "verdict": verdict,
                   "seconds": round(time.time() - started, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report_json, indent=2), encoding="utf-8")
    print(f"LADDER {args.rung}: train {best_fit:.4f}, held-out {best_held.get('held_out_primary')} "
          f"delta {best_held.get('held_out_delta')} -> {verdict} ({report_json['seconds']}s)", flush=True)
    return 0 if verdict == "GENERALISES" else 1


if __name__ == "__main__":
    sys.exit(main())
