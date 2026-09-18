"""Algebraic acceptance checks of DESIGN §7.4 (controls 1-3) on small CPU tensors."""

from __future__ import annotations

import math

import pytest
import torch

from wavelet_rope.qwen_temporal_rope import (
    ATTENTION_SCALE,
    QK_DIM_CURRICULUM,
    QK_DIM_REGIME,
    WildcardContext,
    lift,
    native_inv_freq,
    pairwise_rule_scores,
    wildcard_attention,
)

THETA = 10_000_000.0


def _tokens(seed: int, seq: int, heads: int = 4, kv_heads: int = 2):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, heads, seq, 256, generator=gen)
    k = torch.randn(1, kv_heads, seq, 256, generator=gen)
    v = torch.randn(1, kv_heads, seq, 256, generator=gen)
    m = torch.zeros(1, seq, dtype=torch.bool)
    m[0, 5:15] = True  # a block of signal tokens inside text
    m[0, 30:34] = True  # a second observation later in the conversation
    tau = torch.zeros(1, seq)
    tau[0, 5:15] = torch.linspace(3.0, 40.0, 10)
    tau[0, 30:34] = torch.tensor([10.0, 11.0, 12.0, 13.0])
    p = torch.arange(seq, dtype=torch.float32)[None]
    return q, k, v, m, tau, p


@pytest.mark.parametrize("lam", [0.0, 0.15, 0.5, 1.0])
def test_factorization_matches_pairwise_rule(lam: float) -> None:
    q, k, v, m, tau, p = _tokens(1, 40)
    inv_freq = native_inv_freq(THETA)
    q_star = lift(q, p, m, tau, inv_freq, lam, is_key=False)
    k_star = lift(k, p, m, tau, inv_freq, lam, is_key=True)
    assert q_star.shape[-1] == (QK_DIM_REGIME if lam == 1.0 else QK_DIM_CURRICULUM)
    k16 = k_star.repeat_interleave(2, dim=1)
    factorized = q_star @ k16.transpose(-1, -2)
    reference = pairwise_rule_scores(q, k.repeat_interleave(2, dim=1), p, p, m, m, tau, tau, inv_freq, lam)
    assert torch.allclose(factorized, reference, atol=1e-4, rtol=1e-4)


def test_pure_text_equals_native_rotary_for_every_lambda() -> None:
    seq = 24
    gen = torch.Generator().manual_seed(2)
    q = torch.randn(1, 2, seq, 256, generator=gen)
    k = torch.randn(1, 2, seq, 256, generator=gen)
    m = torch.zeros(1, seq, dtype=torch.bool)
    tau = torch.zeros(1, seq)
    p = torch.arange(seq, dtype=torch.float32)[None]
    inv_freq = native_inv_freq(THETA)
    # native rotary attention logits (rotate_half over the first 64 dims, all 32 pairs positional)
    angles = p[..., None] * inv_freq  # [1, S, 32]
    cos = torch.cat((angles.cos(), angles.cos()), -1)[:, None]
    sin = torch.cat((angles.sin(), angles.sin()), -1)[:, None]

    def rope(x):
        rot, rest = x[..., :64], x[..., 64:]
        half = torch.cat((-rot[..., 32:], rot[..., :32]), -1)
        return torch.cat((rot * cos + half * sin, rest), -1)

    native = rope(q) @ rope(k).transpose(-1, -2)
    for lam in (0.0, 0.3, 1.0):
        q_star = lift(q, p, m, tau, inv_freq, lam, is_key=False)
        k_star = lift(k, p, m, tau, inv_freq, lam, is_key=True)
        assert torch.allclose(q_star @ k_star.transpose(-1, -2), native, atol=1e-4, rtol=1e-4)


def test_signal_pairs_are_invariant_to_text_gap_and_common_time_shift() -> None:
    inv_freq = native_inv_freq(THETA)
    gen = torch.Generator().manual_seed(3)
    q = torch.randn(1, 1, 1, 256, generator=gen)
    k = torch.randn(1, 1, 1, 256, generator=gen)
    m = torch.ones(1, 1, dtype=torch.bool)

    def t_term(p_q, p_k, tau_q, tau_k):
        s = pairwise_rule_scores(q, k, torch.tensor([[p_q]]), torch.tensor([[p_k]]), m, m,
                                 torch.tensor([[tau_q]]), torch.tensor([[tau_k]]), inv_freq, 1.0)
        # remove the P and U contributions by differencing against a zero-T copy
        qz, kz = q.clone(), k.clone()
        qz[..., 1:32:2] = 0
        qz[..., 33:64:2] = 0
        s_pu = pairwise_rule_scores(qz, k, torch.tensor([[p_q]]), torch.tensor([[p_k]]), m, m,
                                    torch.tensor([[tau_q]]), torch.tensor([[tau_k]]), inv_freq, 1.0)
        return (s - s_pu).item()

    base = t_term(10.0, 20.0, 5.0, 9.0)
    assert math.isclose(t_term(10.0, 4000.0, 5.0, 9.0), base, rel_tol=1e-4, abs_tol=1e-4)  # CoT gap
    assert math.isclose(t_term(10.0, 20.0, 105.0, 109.0), base, rel_tol=1e-4, abs_tol=1e-4)  # time origin


def test_mixed_pairs_have_no_temporal_phase() -> None:
    inv_freq = native_inv_freq(THETA)
    gen = torch.Generator().manual_seed(4)
    q = torch.randn(1, 1, 1, 256, generator=gen)
    k = torch.randn(1, 1, 1, 256, generator=gen)
    m_q = torch.ones(1, 1, dtype=torch.bool)
    m_k = torch.zeros(1, 1, dtype=torch.bool)
    p_q, p_k = torch.tensor([[3.0]]), torch.tensor([[900.0]])
    ref = None
    for tau_q in (0.0, 17.0, 1234.5):
        s = pairwise_rule_scores(q, k, p_q, p_k, m_q, m_k, torch.tensor([[tau_q]]), torch.zeros(1, 1), inv_freq, 1.0)
        ref = s if ref is None else ref
        assert torch.allclose(s, ref, atol=1e-5)
    q_star = lift(q, p_q, m_q, torch.tensor([[17.0]]), inv_freq, 1.0, is_key=False)
    k_star = lift(k, p_k, m_k, torch.zeros(1, 1), inv_freq, 1.0, is_key=True)
    assert torch.allclose(q_star @ k_star.transpose(-1, -2), ref, atol=1e-4)


def test_math_backend_cache_offset_matches_full_recompute() -> None:
    q, k, v, m, tau, p = _tokens(5, 40)
    inv_freq = native_inv_freq(THETA)
    ctx = WildcardContext(backend="MATH", pad_to=0)
    q_star = lift(q, p, m, tau, inv_freq, 1.0, is_key=False)
    k_star = lift(k, p, m, tau, inv_freq, 1.0, is_key=True)
    full = wildcard_attention(q_star, k_star, v, ctx=ctx, past_len=0, dtype=torch.float32)
    # last 6 queries against the whole key set, as a cached decode step would do
    partial = wildcard_attention(q_star[:, :, 34:], k_star, v, ctx=ctx, past_len=34, dtype=torch.float32)
    assert torch.allclose(full[:, :, 34:], partial, atol=1e-5)
    assert math.isclose(ATTENTION_SCALE, 1 / 16)
