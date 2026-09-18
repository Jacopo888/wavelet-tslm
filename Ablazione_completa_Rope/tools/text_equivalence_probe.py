#!/usr/bin/env python3
"""Astra 15/09 §2: pure-text equivalence on the ORIGINAL 144-token prompt (the sentence repeated six times).

Reports mean, p99, max absolute logit difference and argmax agreement for: native SDPA vs native eager (noise
reference on the same input), wrapped FLEX vs native eager, wrapped FLEX vs native SDPA, wrapped MATH vs native
eager. PASS criterion (Astra 15/09 §2): wrapped-vs-native deviation within the native SDPA/eager noise on the same
input. Runs alongside training (small model footprint, no training state touched)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wavelet_rope.backbone import load_backbone  # noqa: E402
from wavelet_rope.qwen_temporal_rope import unwrap_full_attention_layers, wrap_full_attention_layers  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    bb = load_backbone(args.snapshot, backend="FLEX_PAD512", pad_to=512, device=device, gradient_checkpointing=False)
    tok, text, lm_head, ctx = bb.tokenizer, bb.text, bb.lm_head, bb.ctx
    text.eval()
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": "You analyze long-context time series."},
         {"role": "user", "content": "Summarize in one sentence what an anomaly localization task is, then list three "
                                     "kinds of anomalies. " * 6}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    seq = ids.shape[1]
    embeds = text.embed_tokens(ids)
    m = torch.zeros(1, seq, dtype=torch.bool, device=device)
    tau = torch.zeros(1, seq, device=device)

    def native(impl: str):
        unwrap_full_attention_layers(text)
        old = text.config._attn_implementation
        text.config._attn_implementation = impl
        with torch.no_grad():
            r = lm_head(text(inputs_embeds=embeds, use_cache=False).last_hidden_state).float()[0]
        text.config._attn_implementation = old
        wrap_full_attention_layers(text, ctx)
        return r

    def wrapped(backend: str):
        ctx.backend = backend
        ctx.set_tokens(m, tau)
        with torch.no_grad():
            r = lm_head(text(inputs_embeds=embeds, use_cache=False).last_hidden_state).float()[0]
        ctx.backend = "FLEX_PAD512"
        return r

    def cmp(a, b):
        d = (a - b).abs().flatten()
        p99 = float(torch.kthvalue(d, max(1, int(0.99 * d.numel()))).values)  # quantile() refuses > 16M elements
        return {"mean_abs": round(float(d.mean()), 5), "p99_abs": round(p99, 4),
                "max_abs": round(float(d.max()), 4), "argmax_agreement": round(float((a.argmax(-1) == b.argmax(-1)).float().mean()), 4),
                "ref_abs_mean": round(float(b.abs().mean()), 4)}

    eager, sdpa, w_flex, w_math = native("eager"), native("sdpa"), wrapped("FLEX_PAD512"), wrapped("MATH")
    out = {"tokens": seq, "noise_native_sdpa_vs_eager": cmp(sdpa, eager), "wrapped_flex_vs_native_eager": cmp(w_flex, eager),
           "wrapped_flex_vs_native_sdpa": cmp(w_flex, sdpa), "wrapped_math_vs_native_eager": cmp(w_math, eager)}
    noise = out["noise_native_sdpa_vs_eager"]
    w = out["wrapped_flex_vs_native_eager"]
    out["pass"] = bool(w["mean_abs"] <= 1.25 * noise["mean_abs"] and w["p99_abs"] <= 1.25 * noise["p99_abs"]
                       and w["argmax_agreement"] >= noise["argmax_agreement"] - 0.02)
    out["criterion"] = "wrapped-vs-native deviation within 1.25x the native SDPA/eager noise on the same input (mean, p99) and argmax within 2 points"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
