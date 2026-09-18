#!/usr/bin/env python3
"""P0 wildcard gate on the REAL pinned Qwen3.5-4B backbone (DESIGN §7.4 controls 3-4, no dataset, no training).

Checks, in order:
1. loading: vision tower dropped, 8 full-attention layers wrapped, 32 LoRA modules (64 tensors) resolved by name;
2. pure text: wrapped model logits (lambda=1 and lambda=0) vs the native forward on the same prompt — with LoRA at
   its zero initialisation the two must agree up to bf16 kernel noise (argmax agreement reported);
3. mixed sequence (random soft tokens with times inserted in the text): FLEX_PAD512 vs MATH reference;
4. generation with cache vs full recompute on the mixed sequence (greedy, same tokens, close logits);
5. cost of a forward+backward with LoRA gradients on a ~4k-token mixed sequence (time, peak memory).
Writes a JSON report; exit 0 only if every check passes.
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
from wavelet_rope.templates import PLACEHOLDER_TOKEN  # noqa: E402


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--backend", default="FLEX_PAD512")
    ap.add_argument("--pad-to", type=int, default=512)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--long-tokens", type=int, default=4096)
    ap.add_argument("--flex-dynamic", type=int, default=1)
    return ap.parse_args()


def main() -> int:
    args = parse()
    device = torch.device("cuda:0")
    report: dict = {"args": vars(args) | {"report": str(args.report)}, "checks": []}

    def check(name: str, ok: bool, **info):
        report["checks"].append({"check": name, "pass": bool(ok), **info})
        print(f"[{'PASS' if ok else 'FAIL':>4}] {name} {info}", flush=True)

    t0 = time.time()
    bb = load_backbone(args.snapshot, backend=args.backend, pad_to=args.pad_to, device=device,
                       gradient_checkpointing=False, flex_dynamic=bool(args.flex_dynamic))
    check("load", len(bb.wrapped_layers) == 8 and len(bb.lora_parameter_names) == 64,
          wrapped=bb.wrapped_layers, lora_tensors=len(bb.lora_parameter_names), seconds=round(time.time() - t0, 1),
          visual_dropped=bb.model.model.visual is None)
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

    # 2. pure text vs native
    with torch.no_grad():
        for lam in (1.0, 0.0):
            ctx.lam = lam
            ctx.set_tokens(torch.zeros(1, seq, dtype=torch.bool, device=device), torch.zeros(1, seq, device=device))
            wrapped_logits = lm_head(text(inputs_embeds=embeds, use_cache=False).last_hidden_state).float()
            unwrap_full_attention_layers(text)
            native_logits = lm_head(text(inputs_embeds=embeds, use_cache=False).last_hidden_state).float()
            wrap_full_attention_layers(text, ctx)
            diff = (wrapped_logits - native_logits).abs()
            agree = float((wrapped_logits.argmax(-1) == native_logits.argmax(-1)).float().mean())
            scale = float(native_logits.abs().mean())
            check(f"pure_text_native_equivalence_lambda{lam}", agree >= 0.98 and float(diff.mean()) < 0.05 * scale,
                  max_abs=round(float(diff.max()), 4), mean_abs=round(float(diff.mean()), 5),
                  native_abs_mean=round(scale, 4), argmax_agreement=round(agree, 4), tokens=seq)
        ctx.lam = 1.0

    # 3. mixed sequence: FLEX vs MATH
    gen = torch.Generator(device=device).manual_seed(7)
    n_soft = 300
    soft = torch.randn(n_soft, embeds.shape[-1], generator=gen, device=device, dtype=torch.float32)
    soft = (soft / soft.norm(dim=-1, keepdim=True) * embeds.float().norm(dim=-1).mean()).to(embeds.dtype)
    mixed = torch.cat([embeds[:, :40], soft[None], embeds[:, 40:]], dim=1)
    m = torch.zeros(1, mixed.shape[1], dtype=torch.bool, device=device)
    m[0, 40:40 + n_soft] = True
    tau = torch.zeros(1, mixed.shape[1], device=device)
    tau[0, 40:40 + n_soft] = torch.linspace(0, 2000, n_soft, device=device)
    with torch.no_grad():
        ctx.set_tokens(m, tau)
        flex_out = lm_head(text(inputs_embeds=mixed, use_cache=False).last_hidden_state).float()
        ctx.backend = "MATH"
        ctx.set_tokens(m, tau)
        math_out = lm_head(text(inputs_embeds=mixed, use_cache=False).last_hidden_state).float()
        ctx.backend = args.backend
    diff = (flex_out - math_out).abs()
    agree = float((flex_out.argmax(-1) == math_out.argmax(-1)).float().mean())
    check("mixed_sequence_backend_vs_math", agree >= 0.98 and float(diff.mean()) < 0.05 * float(math_out.abs().mean()),
          max_abs=round(float(diff.max()), 4), mean_abs=round(float(diff.mean()), 5), argmax_agreement=round(agree, 4),
          tokens=int(mixed.shape[1]))

    # 4. cached generation vs recompute
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
    steps = 12
    with torch.no_grad():
        cache = Qwen3_5DynamicCache(config=text.config)
        ctx.set_tokens(m, tau)
        out = text(inputs_embeds=mixed, past_key_values=cache, use_cache=True,
                   cache_position=torch.arange(mixed.shape[1], device=device))
        cached_tokens, cached_logits = [], []
        nxt = lm_head(out.last_hidden_state[:, -1]).float()
        pos = mixed.shape[1]
        for _ in range(steps):
            token = int(nxt.argmax(-1))
            cached_tokens.append(token)
            cached_logits.append(nxt[0])
            step = text.embed_tokens(torch.tensor([[token]], device=device))
            ctx.set_tokens(torch.zeros(1, 1, dtype=torch.bool, device=device), torch.zeros(1, 1, device=device))
            out = text(inputs_embeds=step, past_key_values=cache, use_cache=True,
                       cache_position=torch.tensor([pos], device=device))
            nxt = lm_head(out.last_hidden_state[:, -1]).float()
            pos += 1
        full_tokens, full_logits = [], []
        cur, cur_m, cur_tau = mixed, m, tau
        for _ in range(steps):
            ctx.set_tokens(cur_m, cur_tau)
            nxt = lm_head(text(inputs_embeds=cur, use_cache=False).last_hidden_state[:, -1]).float()
            token = int(nxt.argmax(-1))
            full_tokens.append(token)
            full_logits.append(nxt[0])
            cur = torch.cat([cur, text.embed_tokens(torch.tensor([[token]], device=device))], 1)
            cur_m = torch.cat([cur_m, torch.zeros(1, 1, dtype=torch.bool, device=device)], 1)
            cur_tau = torch.cat([cur_tau, torch.zeros(1, 1, device=device)], 1)
    ldiff = max(float((a - b).abs().max()) for a, b in zip(cached_logits, full_logits))
    check("cache_vs_recompute_generation", cached_tokens == full_tokens, cached=cached_tokens, recomputed=full_tokens,
          max_logit_abs_diff=round(ldiff, 4), decoded=tok.decode(cached_tokens))

    # 5. training-cost probe on a long mixed sequence with LoRA gradients
    text.train()
    reps = max(1, args.long_tokens // mixed.shape[1])
    long = torch.cat([mixed] * reps, dim=1)[:, : args.long_tokens].contiguous()
    lm = torch.cat([m] * reps, dim=1)[:, : args.long_tokens].contiguous()
    lt = torch.cat([tau] * reps, dim=1)[:, : args.long_tokens].contiguous()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t1 = time.time()
    ctx.set_tokens(lm, lt)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = text(inputs_embeds=long, use_cache=False).last_hidden_state
        loss = lm_head(hidden[:, -256:]).float().logsumexp(-1).mean()
    loss.backward()
    torch.cuda.synchronize()
    grads = [p.grad for n, p in text.named_parameters() if "lora_" in n and p.grad is not None]
    check("long_sequence_forward_backward", len(grads) == 64 and all(torch.isfinite(g).all() for g in grads),
          tokens=int(long.shape[1]), seconds=round(time.time() - t1, 2),
          peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
          lora_grad_tensors=len(grads), lora_grad_norm=round(float(torch.stack([g.norm() for g in grads]).norm()), 6))

    report["verdict"] = "PASS" if all(c["pass"] for c in report["checks"]) else "FAIL"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))
    print("VERDICT", report["verdict"], "report", args.report)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
