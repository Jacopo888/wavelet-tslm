#!/usr/bin/env python3
"""Where does the time go? Native vs wrapped Qwen3.5 forward/backward at ~4k tokens, plus a bf16 noise control.

Part A (noise control for the wildcard gate): on pure text, native SDPA vs native eager, and wrapped(MATH) vs
native eager.  If the native SDPA-vs-eager gap is as large as the wrapped-vs-native gap, the wildcard path is
within backend noise (the wildcard reduces to native rotary on text by construction, tests/test_wildcard.py).

Part B (cost): time and peak memory of forward+backward (LoRA grads) for native-unwrapped and wrapped
(flex dynamic=True and False), with a profiler table of the top device ops so the linear-attention (gated delta)
share is visible.  Writes JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wavelet_rope.backbone import load_backbone  # noqa: E402
from wavelet_rope.qwen_temporal_rope import unwrap_full_attention_layers, wrap_full_attention_layers  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--skip-noise", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda:0")
    out: dict = {"tokens": args.tokens, "noise": {}, "cost": {}, "fla_available": False}
    try:
        import fla  # noqa: F401
        from transformers.utils.import_utils import is_flash_linear_attention_available
        out["fla_available"] = bool(is_flash_linear_attention_available())
        out["fla_version"] = getattr(fla, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        out["fla_error"] = str(exc)[:300]

    bb = load_backbone(args.snapshot, backend="FLEX_PAD512", pad_to=512, device=device, gradient_checkpointing=False)
    tok, text, lm_head, ctx = bb.tokenizer, bb.text, bb.lm_head, bb.ctx
    text.eval()
    base = tok.apply_chat_template(
        [{"role": "system", "content": "You analyze long-context time series."},
         {"role": "user", "content": "Describe the difference between a level shift and a transient spike, with examples. "}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(base, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    reps = max(1, args.tokens // ids.shape[1])
    ids = ids.repeat(1, reps)[:, : args.tokens]
    seq = ids.shape[1]
    embeds = text.embed_tokens(ids)
    zeros_m = torch.zeros(1, seq, dtype=torch.bool, device=device)
    zeros_t = torch.zeros(1, seq, device=device)

    def logits_native(impl: str):
        unwrap_full_attention_layers(text)
        old = text.config._attn_implementation
        text.config._attn_implementation = impl
        with torch.no_grad():
            r = lm_head(text(inputs_embeds=embeds, use_cache=False).last_hidden_state).float()
        text.config._attn_implementation = old
        wrap_full_attention_layers(text, ctx)
        return r

    def logits_wrapped(backend: str):
        ctx.backend = backend
        ctx.set_tokens(zeros_m, zeros_t)
        with torch.no_grad():
            r = lm_head(text(inputs_embeds=embeds[:, :512], use_cache=False).last_hidden_state).float()
        ctx.backend = "FLEX_PAD512"
        return r

    def cmp(a, b):
        d = (a - b).abs()
        return {"max_abs": round(float(d.max()), 4), "mean_abs": round(float(d.mean()), 5),
                "argmax_agreement": round(float((a.argmax(-1) == b.argmax(-1)).float().mean()), 4),
                "ref_abs_mean": round(float(b.abs().mean()), 4)}

    if not args.skip_noise:
        embeds_full = embeds
        embeds = embeds[:, :512]
        seq_short = embeds.shape[1]
        zeros_m, zeros_t = zeros_m[:, :seq_short], zeros_t[:, :seq_short]
        eager = logits_native("eager")
        sdpa = logits_native("sdpa")
        w_math = logits_wrapped("MATH")
        w_flex = logits_wrapped("FLEX_PAD512")
        out["noise"] = {"tokens": seq_short, "native_sdpa_vs_native_eager": cmp(sdpa, eager),
                        "wrapped_math_vs_native_eager": cmp(w_math, eager),
                        "wrapped_flex_vs_native_eager": cmp(w_flex, eager),
                        "wrapped_flex_vs_native_sdpa": cmp(w_flex, sdpa)}
        print("noise", json.dumps(out["noise"]), flush=True)
        embeds = embeds_full
        zeros_m = torch.zeros(1, seq, dtype=torch.bool, device=device)
        zeros_t = torch.zeros(1, seq, device=device)

    # Part B: cost
    text.train()
    m = torch.zeros(1, seq, dtype=torch.bool, device=device)
    m[0, 64:64 + 1024] = True
    tau = torch.zeros(1, seq, device=device)
    tau[0, 64:64 + 1024] = torch.linspace(0, 2000, 1024, device=device)

    def run(label: str, *, wrapped: bool, dynamic: bool, profile: bool):
        from torch.profiler import ProfilerActivity, profile as tprofile
        if wrapped:
            wrap_full_attention_layers(text, ctx)
            ctx.flex_dynamic = dynamic
            ctx._flex = None
        else:
            unwrap_full_attention_layers(text)
        for p in text.parameters():
            p.grad = None
        times = []
        for it in range(3):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            ctx.set_tokens(m, tau)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = text(inputs_embeds=embeds, use_cache=False).last_hidden_state
                loss = lm_head(hidden[:, -256:]).float().logsumexp(-1).mean()
            loss.backward()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
            for p in text.parameters():
                p.grad = None
        rec = {"seconds_per_fwd_bwd": [round(t, 2) for t in times], "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        if profile:
            with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                ctx.set_tokens(m, tau)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = text(inputs_embeds=embeds, use_cache=False).last_hidden_state
                    loss = lm_head(hidden[:, -256:]).float().logsumexp(-1).mean()
                loss.backward()
                torch.cuda.synchronize()
            rows = []
            for ev in prof.key_averages():
                dev = getattr(ev, "device_time_total", None)
                if dev is None:
                    dev = getattr(ev, "cuda_time_total", 0.0)
                if dev > 0:
                    rows.append((ev.key[:90], round(dev / 1000.0, 1), ev.count))
            rows.sort(key=lambda r: -r[1])
            rec["top_device_ops_ms"] = rows[:25]
            for p in text.parameters():
                p.grad = None
        out["cost"][label] = rec
        print(label, json.dumps({k: v for k, v in rec.items() if k != "top_device_ops_ms"}), flush=True)
        if profile:
            for r in rec["top_device_ops_ms"][:12]:
                print("   ", r, flush=True)

    run("native_unwrapped", wrapped=False, dynamic=True, profile=True)
    run("wrapped_flex_dynamic", wrapped=True, dynamic=True, profile=True)
    run("wrapped_flex_static", wrapped=True, dynamic=False, profile=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2, default=str))
    print("DONE", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
