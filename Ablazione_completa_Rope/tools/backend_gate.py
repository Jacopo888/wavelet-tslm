#!/usr/bin/env python3
"""P0 backend gate for the factorized wildcard attention shapes (no Qwen weights needed).

What is verified, as fixed by the Astra decision of 2026-09-14:

* Q/K head dimension 352 (regime) and 384 (curriculum), V head dimension 256,
  16 query heads over 4 KV heads expanded EXPLICITLY to 16 (the cache stays at 4),
  bf16, explicit scale 1/sqrt(256) = 0.0625 (not the SDPA default 1/sqrt(352)).
* Forward AND backward on the requested backend with every other SDPA backend disabled,
  so a silent fallback to MATH is impossible: if the backend cannot serve the shape,
  PyTorch raises and the case FAILS.
* Numerical agreement with an fp32 MATH reference on small tensors, for the output
  and for dQ/dK/dV, under (a) square causal attention and (b) a non-square query block
  attending to a longer key/value prefix (cache offset), i.e. the conversation-order
  causality the design requires.
* Peak memory and wall time at representative lengths (forward+backward only).
* The GPU kernels actually launched, captured with the profiler, so the backend name
  is a measured fact and not a label.

Backends: EFFICIENT_ATTENTION / FLASH_ATTENTION / CUDNN_ATTENTION / MATH go through
torch.nn.functional.scaled_dot_product_attention under torch.nn.attention.sdpa_kernel;
FLEX goes through torch.nn.attention.flex_attention (compiled Triton kernel) with a
block mask expressing the same causal / cache-offset rule.  --pad-qk-to zero-pads the
Q/K head dimension (exact: padded coordinates contribute nothing to the logits) for
kernels that only accept power-of-two dimensions.

The script writes a JSON report and exits 0 only if every REQUIRED case passed.
It never trains, never loads a model, and never touches the dataset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="EFFICIENT_ATTENTION",
                        choices=("EFFICIENT_ATTENTION", "FLASH_ATTENTION", "CUDNN_ATTENTION", "MATH", "FLEX"))
    parser.add_argument("--label", default="", help="free label stored in the report")
    parser.add_argument("--qk-dims", type=int, nargs="+", default=[352, 384])
    parser.add_argument("--v-dim", type=int, default=256)
    parser.add_argument("--pad-qk-to", type=int, default=0, help="zero-pad Q/K head dim to this size (0 = no padding)")
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--scale", type=float, default=1.0 / 16.0, help="explicit attention scale (design: 1/sqrt(256))")
    parser.add_argument("--numeric-length", type=int, default=512)
    parser.add_argument("--cache-prefix", type=int, default=384)
    parser.add_argument("--memory-lengths", type=int, nargs="+", default=[4096, 16384])
    parser.add_argument("--out-atol", type=float, default=2.5e-2)
    parser.add_argument("--out-rtol", type=float, default=2.5e-2)
    parser.add_argument("--grad-atol", type=float, default=5e-2)
    parser.add_argument("--grad-rtol", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--rocm-experimental", action="store_true",
                        help="set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 before importing torch")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--skip-profile", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
if ARGS.rocm_experimental:
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

SDPA_BACKENDS = {
    "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
    "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
    "CUDNN_ATTENTION": getattr(SDPBackend, "CUDNN_ATTENTION", None),
    "MATH": SDPBackend.MATH,
}

_FLEX = None


def flex_fn():
    """Compiled flex_attention, created once; recompiles per shape are allowed."""
    global _FLEX
    if _FLEX is None:
        import torch._dynamo
        from torch.nn.attention.flex_attention import flex_attention
        torch._dynamo.config.cache_size_limit = 64
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "env_TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "gcn_arch": getattr(props, "gcnArchName", None),
            "total_memory_gib": round(props.total_memory / 2**30, 2),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "sdp_flags": {
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "math": torch.backends.cuda.math_sdp_enabled(),
            },
        })
    return info


def expand_kv(x: torch.Tensor) -> torch.Tensor:
    """Explicit 4 -> 16 head expansion for the attention call; parameters and cache keep kv_heads."""
    assert ARGS.heads % ARGS.kv_heads == 0
    return x.repeat_interleave(ARGS.heads // ARGS.kv_heads, dim=1)


def pad_qk(x: torch.Tensor) -> torch.Tensor:
    if ARGS.pad_qk_to and x.shape[-1] < ARGS.pad_qk_to:
        return F.pad(x, (0, ARGS.pad_qk_to - x.shape[-1]))
    return x


def make_inputs(q_len: int, kv_len: int, qk_dim: int, device: torch.device, gen: torch.Generator):
    q = torch.randn(1, ARGS.heads, q_len, qk_dim, generator=gen, device=device)
    k = torch.randn(1, ARGS.kv_heads, kv_len, qk_dim, generator=gen, device=device)
    v = torch.randn(1, ARGS.kv_heads, kv_len, ARGS.v_dim, generator=gen, device=device)
    # unit-RMS per head, as the design normalizes Q/K before the factorized expansion
    q = q / q.norm(dim=-1, keepdim=True) * math.sqrt(qk_dim) * 0.5
    k = k / k.norm(dim=-1, keepdim=True) * math.sqrt(qk_dim) * 0.5
    return q, k, v


def cache_offset_mask(q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    """Boolean mask (True = attend) for q_len new queries appended after kv_len - q_len cached keys."""
    assert kv_len >= q_len
    offset = kv_len - q_len
    qi = torch.arange(q_len, device=device)[:, None] + offset
    kj = torch.arange(kv_len, device=device)[None, :]
    return kj <= qi


def gated_attention(q, k, v, *, q_len: int, kv_len: int, explicit_mask: bool, dtype: torch.dtype):
    """The attention call under test: bf16, explicit KV expansion, explicit scale, forced backend."""
    qq = pad_qk(q.to(dtype))
    kk = pad_qk(expand_kv(k).to(dtype))
    vv = expand_kv(v).to(dtype)
    if ARGS.backend == "FLEX":
        from torch.nn.attention.flex_attention import create_block_mask
        offset = kv_len - q_len

        def mask_mod(b, h, qi, kj):
            return kj <= qi + offset

        block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=qq.device)
        return flex_fn()(qq, kk, vv, block_mask=block_mask, scale=ARGS.scale)
    backend = SDPA_BACKENDS[ARGS.backend]
    mask = cache_offset_mask(q_len, kv_len, qq.device) if explicit_mask else None
    with sdpa_kernel([backend]):
        return F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, is_causal=not explicit_mask, scale=ARGS.scale)


def reference_attention(q, k, v, *, q_len: int, kv_len: int, explicit_mask: bool):
    """fp32 MATH reference on the UNPADDED tensors (padding is exact, so this is the ground truth)."""
    mask = cache_offset_mask(q_len, kv_len, q.device) if explicit_mask else None
    with sdpa_kernel([SDPBackend.MATH]):
        return F.scaled_dot_product_attention(q, expand_kv(k), expand_kv(v), attn_mask=mask,
                                              is_causal=not explicit_mask, scale=ARGS.scale)


def compare(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> dict[str, Any]:
    got32 = got.detach().float()
    ref32 = ref.detach().float()
    if got32.shape != ref32.shape:  # padded gradient: compare the real coordinates only
        got32 = got32[..., : ref32.shape[-1]]
    diff = (got32 - ref32).abs()
    denom = ref32.abs().clamp_min(1e-6)
    ok = bool(torch.allclose(got32, ref32, atol=atol, rtol=rtol))
    return {"tensor": name, "max_abs_err": float(diff.max()), "mean_abs_err": float(diff.mean()),
            "max_rel_err": float((diff / denom).max()), "ref_abs_mean": float(ref32.abs().mean()),
            "finite": bool(torch.isfinite(got32).all()), "atol": atol, "rtol": rtol, "pass": ok}


def base_case(label: str, q_len: int, kv_len: int, qk_dim: int) -> dict[str, Any]:
    return {"case": label, "required": True, "q_len": q_len, "kv_len": kv_len, "qk_dim": qk_dim,
            "qk_dim_padded": max(qk_dim, ARGS.pad_qk_to) if ARGS.pad_qk_to else qk_dim, "v_dim": ARGS.v_dim,
            "heads": ARGS.heads, "kv_heads": ARGS.kv_heads, "dtype": "bfloat16", "backend": ARGS.backend}


def fail(case: dict[str, Any], exc: Exception, status: str = "BACKEND_ERROR") -> dict[str, Any]:
    case["pass"] = False
    case["status"] = status
    case["error"] = f"{type(exc).__name__}: {str(exc)[:1500]}"
    case["traceback"] = traceback.format_exc()[-3000:]
    return case


def numeric_case(label: str, *, q_len: int, kv_len: int, qk_dim: int, device, gen, explicit_mask: bool) -> dict[str, Any]:
    case = base_case(label, q_len, kv_len, qk_dim)
    case["explicit_mask"] = explicit_mask
    try:
        q, k, v = make_inputs(q_len, kv_len, qk_dim, device, gen)
        qr, kr, vr = (t.clone().float().requires_grad_(True) for t in (q, k, v))
        out_ref = reference_attention(qr, kr, vr, q_len=q_len, kv_len=kv_len, explicit_mask=explicit_mask)
        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)

        qb, kb, vb = (t.clone().to(torch.bfloat16).requires_grad_(True) for t in (q, k, v))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = gated_attention(qb, kb, vb, q_len=q_len, kv_len=kv_len, explicit_mask=explicit_mask, dtype=torch.bfloat16)
        out.backward(grad_out.to(torch.bfloat16))
        torch.cuda.synchronize()
        case["wall_seconds_incl_compile"] = round(time.perf_counter() - t0, 4)
        checks = [compare("output", out, out_ref, ARGS.out_atol, ARGS.out_rtol),
                  compare("dQ", qb.grad, qr.grad, ARGS.grad_atol, ARGS.grad_rtol),
                  compare("dK", kb.grad, kr.grad, ARGS.grad_atol, ARGS.grad_rtol),
                  compare("dV", vb.grad, vr.grad, ARGS.grad_atol, ARGS.grad_rtol)]
        case["checks"] = checks
        case["pass"] = all(c["pass"] and c["finite"] for c in checks)
        case["status"] = "PASS" if case["pass"] else "NUMERIC_MISMATCH"
    except Exception as exc:
        fail(case, exc)
    return case


def memory_case(label: str, *, length: int, qk_dim: int, device, gen) -> dict[str, Any]:
    case = base_case(label, length, length, qk_dim)
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        q, k, v = make_inputs(length, length, qk_dim, device, gen)
        qb, kb, vb = (t.to(torch.bfloat16).requires_grad_(True) for t in (q, k, v))
        del q, k, v
        base = torch.cuda.memory_allocated()
        # warm-up (compilation for FLEX), then the timed pass
        for it in range(2):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = gated_attention(qb, kb, vb, q_len=length, kv_len=length, explicit_mask=False, dtype=torch.bfloat16)
            out.backward(torch.randn_like(out))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            if it == 0:
                case["wall_seconds_first_incl_compile"] = round(elapsed, 4)
                qb.grad = kb.grad = vb.grad = None
                torch.cuda.reset_peak_memory_stats()
        case["wall_seconds_fwd_bwd"] = round(elapsed, 4)
        case["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        case["attention_extra_gib"] = round((torch.cuda.max_memory_allocated() - base) / 2**30, 3)
        case["finite"] = bool(torch.isfinite(out).all() and torch.isfinite(qb.grad).all())
        case["pass"] = case["finite"]
        case["status"] = "PASS" if case["pass"] else "NON_FINITE"
    except Exception as exc:
        fail(case, exc)
    return case


def profile_kernels(qk_dim: int, device, gen) -> dict[str, Any]:
    """Record which GPU kernels a forward+backward at L=2048 actually launches."""
    result: dict[str, Any] = {"case": f"profile_qk{qk_dim}", "required": False, "backend": ARGS.backend}
    try:
        from torch.profiler import ProfilerActivity, profile
        q, k, v = make_inputs(2048, 2048, qk_dim, device, gen)
        qb, kb, vb = (t.to(torch.bfloat16).requires_grad_(True) for t in (q, k, v))
        # one untimed pass so compilation does not pollute the profile
        out = gated_attention(qb, kb, vb, q_len=2048, kv_len=2048, explicit_mask=False, dtype=torch.bfloat16)
        out.backward(torch.randn_like(out))
        qb.grad = kb.grad = vb.grad = None
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            out = gated_attention(qb, kb, vb, q_len=2048, kv_len=2048, explicit_mask=False, dtype=torch.bfloat16)
            out.backward(torch.randn_like(out))
            torch.cuda.synchronize()
        rows = []
        for ev in prof.key_averages():
            low = ev.key.lower()
            if any(t in low for t in ("attn", "attention", "aotriton", "fmha", "flash", "sdp", "efficient",
                                       "ck_", "softmax", "triton", "flex", "bmm", "matmul", "gemm")):
                dev_ms = getattr(ev, "device_time_total", None)
                if dev_ms is None:
                    dev_ms = getattr(ev, "cuda_time_total", 0.0)
                rows.append({"name": ev.key[:160], "count": ev.count, "device_time_ms": round(dev_ms / 1000.0, 3)})
        rows.sort(key=lambda r: -r["device_time_ms"])
        result["kernels"] = rows[:30]
        result["math_fallback_seen"] = any("_scaled_dot_product_attention_math" in r["name"] for r in rows)
        result["pass"] = not (result["math_fallback_seen"] and ARGS.backend != "MATH")
        result["status"] = "PASS" if result["pass"] else "MATH_FALLBACK_DETECTED"
    except Exception as exc:
        fail(result, exc, "PROFILE_ERROR")
    return result


def describe(c: dict[str, Any]) -> str:
    if "checks" in c:
        return "; ".join(f"{k['tensor']} max_abs={k['max_abs_err']:.3e}" for k in c["checks"])
    if "peak_allocated_gib" in c:
        return (f"peak={c['peak_allocated_gib']} GiB attn_extra={c['attention_extra_gib']} GiB "
                f"t={c['wall_seconds_fwd_bwd']}s (first {c.get('wall_seconds_first_incl_compile')}s)")
    if "kernels" in c:
        return ", ".join(r["name"][:60] for r in c["kernels"][:4]) or "no attention kernels matched"
    return c.get("error", "")


def main() -> int:
    if ARGS.backend != "FLEX" and SDPA_BACKENDS[ARGS.backend] is None:
        print(f"backend {ARGS.backend} not available in this torch build", file=sys.stderr)
        return 2
    report: dict[str, Any] = {"schema_version": 2, "label": ARGS.label, "started_at_utc": utc_now(),
                              "args": {**vars(ARGS), "report": str(ARGS.report)},
                              "environment": environment(), "cases": []}
    if not torch.cuda.is_available():
        report["verdict"] = "NO_GPU"
        ARGS.report.parent.mkdir(parents=True, exist_ok=True)
        ARGS.report.write_text(json.dumps(report, indent=2, default=str))
        print("no GPU visible; report written", file=sys.stderr)
        return 2

    device = torch.device("cuda:0")
    gen = torch.Generator(device=device).manual_seed(ARGS.seed)
    cases = report["cases"]
    for qk_dim in ARGS.qk_dims:
        new = [numeric_case(f"numeric_causal_square_qk{qk_dim}", q_len=ARGS.numeric_length, kv_len=ARGS.numeric_length,
                            qk_dim=qk_dim, device=device, gen=gen, explicit_mask=False),
               numeric_case(f"numeric_cache_offset_qk{qk_dim}", q_len=ARGS.numeric_length - ARGS.cache_prefix,
                            kv_len=ARGS.numeric_length, qk_dim=qk_dim, device=device, gen=gen, explicit_mask=True)]
        for length in ARGS.memory_lengths:
            new.append(memory_case(f"memory_causal_L{length}_qk{qk_dim}", length=length, qk_dim=qk_dim, device=device, gen=gen))
        if not ARGS.skip_profile:
            new.append(profile_kernels(qk_dim, device, gen))
        for c in new:
            print(f"[{c['status']:>22}] {c['case']:<36} {describe(c)}", flush=True)
        cases.extend(new)

    required = [c for c in cases if c.get("required")]
    verdict_ok = all(c.get("pass") for c in required) and not any(c.get("status") == "MATH_FALLBACK_DETECTED" for c in cases)
    report["verdict"] = "PASS" if verdict_ok else "FAIL"
    report["required_cases"] = len(required)
    report["required_failed"] = [c["case"] for c in required if not c.get("pass")]
    report["finished_at_utc"] = utc_now()
    ARGS.report.parent.mkdir(parents=True, exist_ok=True)
    ARGS.report.write_text(json.dumps(report, indent=2, default=str))
    print(f"VERDICT {report['verdict']}  backend={ARGS.backend} pad={ARGS.pad_qk_to} label={ARGS.label!r}  "
          f"failed={report['required_failed']}  report={ARGS.report}")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
