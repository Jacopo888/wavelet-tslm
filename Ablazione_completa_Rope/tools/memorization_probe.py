#!/usr/bin/env python3
"""Can the whole system memorise a handful of signals that differ ONLY by where the anomaly is?

Astra (16/09): before another long run, the missing control is the most direct one -- 16 to 32 simple signals with
identical headers and anomalies in different positions, on the same GPU and precision as the real run.  The system
must learn those answers, and must get worse when the signals are swapped.

The design makes memorisation itself the proof of reading.  Every example shares the question, the recording length,
the channel and (with the reduced header) the printed statistics; the anomaly always has the same amplitude and
width.  The only thing that distinguishes one example from another is WHERE the anomaly sits, and the only route to
that fact is the soft tokens.  So a model that fits these examples has necessarily read them; one that cannot fit 24
examples it sees hundreds of times has a problem that no amount of data will fix.

Resolution limit, learned the hard way: with ``n`` examples every score is a multiple of 1/n, so a single example
flipping moves the whole number.  At n=24 that is 0.0417, which is the size of the "signals" that appeared at step 500
of one run and step 1000 of another and then vanished.  Read only plateaus over several evaluations, never a single
point, and raise ``--n`` before reading a small delta as real.

Two arms are meant to be compared from the same initialisation, via the spec flag ``tokenizer.meta_proj_fp32``:
the current numerics, where position/L collapses under bf16 autocast, and the fp32 metadata projection.

Usage:
  python tools/memorization_probe.py --config configs/va1r_mi300x.toml --out report.json \
      [--n 24] [--steps 600] [--length 131072] [--meta-fp32] [--checkpoint <dir>]
"""

from __future__ import annotations

import argparse
import json
import math
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
from wavelet_rope.scoring import score_precise  # noqa: E402
from wavelet_rope.signal_contract import SignalEpisode  # noqa: E402

QUESTION = ("Does the recording contain an anomaly? If it does, give the sample interval that contains it. "
            "Answer with the required JSON object.")


def build_examples(n: int, length: int, width: int, amplitude: float, seed: int, device: torch.device):
    """``n`` single-channel recordings identical in everything but the position of one burst."""

    generator = torch.Generator().manual_seed(seed)
    margin = 4 * width
    usable = length - 2 * margin - width
    if usable <= 0:
        raise SystemExit("the recording is too short for this anomaly width")
    # Every example is a CIRCULAR ROTATION of one signal, so all of them hold exactly the same multiset of
    # sample values: median, robust scale, min, max and the valid count are bit-identical, and the printed
    # prompt cannot tell the examples apart.  Position is then reachable only through the soft tokens, which
    # is what makes memorisation here a proof of reading.  Adding the bump to a shared background is NOT
    # enough -- it lands on different noise samples and the statistics drift apart (19 distinct prefixes).
    base = torch.randn(1, length, generator=generator)
    bump = amplitude * torch.hann_window(width, periodic=False).unsqueeze(0)
    base[:, :width] += bump
    examples = []
    for index in range(n):
        # positions spread deterministically over the usable span, never centred, never twice the same
        start = margin + int(round(usable * (index + 0.5) / n))
        values = torch.roll(base, shifts=start, dims=1)
        episode = SignalEpisode(
            values=values.to(device),
            valid=torch.ones(1, length, dtype=torch.bool, device=device),
            channel_ids=("ch_00",),
            sample_times=torch.arange(length, dtype=torch.float64, device=device),
            task=None,
        )
        answer = ('localize:{"_L": %d, "end": %d, "present": true, "start": %d}' % (length, start + width, start))
        meta = RowMeta(id=f"mem-{index:03d}", category="localize", length=length, n_channels=1,
                       channel_names=("ch_00",), question=QUESTION, answer=answer,
                       gold_intervals=(GoldInterval(channel="ch_00", start=start, end=start + width, kind="spike"),),
                       split="synthetic", row_index=index)
        examples.append((meta, episode, start))
    return examples


def oracle_text_for(meta) -> str | None:
    """The true window as text, for the positive control (set by main)."""

    return _ORACLE.get(meta.id) if _ORACLE else None


_ORACLE: dict = {}


def evaluate(model, examples, *, substituted: bool, max_new_tokens: int) -> dict:
    """Primary of every example, optionally with the NEXT example's signal under this example's header."""

    scores, ious = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, (meta, episode, _) in enumerate(examples):
            shown = episode
            if substituted:
                donor = examples[(index + 1) % len(examples)][1]
                shown = SignalEpisode(donor.values, episode.valid, episode.channel_ids,
                                      episode.sample_times, episode.task)
            generated = model.generate(meta, shown, mode="direct", max_new_tokens=max_new_tokens,
                                       thinking=False, oracle_text=oracle_text_for(meta))
            scored = score_precise(generated.get("text") or "", meta.answer)
            scores.append(float(scored["primary"]))
            ious.append(float(scored.get("iou", scored["primary"]) or 0.0))
    return {"primary": sum(scores) / len(scores), "iou": sum(ious) / len(ious), "per_example": scores}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", default=None, help="start from a checkpoint instead of a fresh frontend")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--length", type=int, default=131072)
    parser.add_argument("--width", type=int, default=137)
    parser.add_argument("--amplitude", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--meta-fp32", action="store_true", help="run the metadata projection in fp32")
    parser.add_argument("--wildcard-lambda", type=float, default=None,
                        help="0 = native RoPE, 1 = wildcard; Astra 17/09 wants the two compared from one init")
    parser.add_argument("--oracle-position", action="store_true",
                        help="positive control: the true window is handed to the model as text in the user turn, so "
                             "the task no longer needs the signal. If this does not reach ~1.0 the training loop "
                             "itself is broken and no perceptual conclusion is available")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    started = time.time()

    cfg = load_runtime_config(args.config, validate_paths=False)
    raw, spec = cfg.as_dict(), dict(cfg.spec)
    spec = json.loads(json.dumps(spec))
    spec["tokenizer"]["meta_proj_fp32"] = bool(args.meta_fp32)
    spec["training"]["header_stats"] = "reduced"   # min/max would vary with the burst; median/scale do not carry position
    spec["training"]["reading_segment"] = False
    spec["training"]["token_supervision"] = {"enabled": False, "inputs": ["soft"], "position_weight": 0.0,
                                             "energy_weight": 0.0, "peak_weight": 0.0, "hidden": 0,
                                             "stats_path": None}

    device = torch.device("cuda:0")
    runtime = raw["runtime"]
    torch.manual_seed(args.seed)
    backbone = load_backbone(raw["backbone"]["snapshot"]["root"], backend=runtime["attention_backend"],
                             pad_to=int(runtime["attention_qk_pad_to"]), device=device,
                             gradient_checkpointing=False)
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
    if args.wildcard_lambda is not None:
        backbone.ctx.lam = float(args.wildcard_lambda)
    print(f"wildcard lambda = {backbone.ctx.lam}", flush=True)

    groups = model.trainable_groups()
    optimizer = torch.optim.AdamW(
        [{"params": groups["frontend"], "lr": args.lr}, {"params": groups["lora"], "lr": args.lr}],
        weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)

    examples = build_examples(args.n, args.length, args.width, args.amplitude, args.seed, device)
    headers = set()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for meta, episode, _ in examples:
            encoding = model.encode_episode(episode, [], gold=meta.gold_intervals)
            sequence = model.build_sequence(meta, encoding, mode="direct", actions=[], answer=meta.answer,
                                            supervise_action=False)
            headers.add(sequence.text.split("Question:")[0])
    if len(headers) != 1:
        sep = chr(10) + "---" + chr(10)
        raise SystemExit("the prompt prefixes are NOT identical, so memorisation would not prove "
                         f"reading. {len(headers)} distinct prefixes; first two:"
                         + sep + sep.join(sorted(headers)[:2]))
    print(f"{len(examples)} examples, prompt prefix identical across all of them, "
          f"meta_fp32={args.meta_fp32}", flush=True)

    if args.oracle_position:
        _ORACLE.update({meta.id: f"window {start}-{start + args.width}" for meta, _, start in examples})
        print(f"controllo positivo attivo: la finestra vera e' nel prompt di tutti gli esempi", flush=True)

    history = []
    model.train()
    backbone.text.train()
    for step in range(1, args.steps + 1):
        meta, episode, _ = examples[(step - 1) % len(examples)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            report = model.forward_loss(meta, episode, mode="direct", actions=[], answer=meta.answer,
                                        gold=meta.gold_intervals, supervise_action=False,
                                        oracle_text=oracle_text_for(meta))
        report.loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups.values() for p in g], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % args.eval_every == 0 or step == args.steps:
            model.eval(); backbone.text.eval()
            true_arm = evaluate(model, examples, substituted=False, max_new_tokens=args.max_new_tokens)
            swap_arm = evaluate(model, examples, substituted=True, max_new_tokens=args.max_new_tokens)
            model.train(); backbone.text.train()
            entry = {"step": step, "loss": float(report.loss), "primary_true": true_arm["primary"],
                     "primary_substituted": swap_arm["primary"],
                     "delta": true_arm["primary"] - swap_arm["primary"],
                     "iou_true": true_arm["iou"], "solved": sum(1 for s in true_arm["per_example"] if s >= 0.5)}
            history.append(entry)
            print(f"  step {step:4d} loss {entry['loss']:.4f} | primary veri {entry['primary_true']:.4f} "
                  f"sostituiti {entry['primary_substituted']:.4f} delta {entry['delta']:+.4f} | "
                  f"risolti {entry['solved']}/{len(examples)}", flush=True)

    best = max(history, key=lambda h: h["primary_true"]) if history else {}
    verdict = ("MEMORISED" if best.get("primary_true", 0.0) >= 0.5 and best.get("delta", 0.0) > 0.1
               else "NOT MEMORISED")
    report_json = {"config": args.config, "meta_proj_fp32": bool(args.meta_fp32),
                   "checkpoint": args.checkpoint, "n": args.n, "steps": args.steps, "length": args.length,
                   "width": args.width, "amplitude": args.amplitude, "lr": args.lr,
                   "identical_prompt_prefix": len(headers) == 1, "history": history, "best": best,
                   "verdict": verdict, "seconds": round(time.time() - started, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report_json, indent=2), encoding="utf-8")
    print(f"MEMORIZATION_PROBE meta_fp32={args.meta_fp32}: best primary {best.get('primary_true')} "
          f"delta {best.get('delta')} -> {verdict} ({report_json['seconds']}s)", flush=True)
    return 0 if verdict == "MEMORISED" else 1


if __name__ == "__main__":
    sys.exit(main())
