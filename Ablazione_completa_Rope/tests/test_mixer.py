"""CPU checks of the perceptual mixer, two-axis RoPE and the perceptual memory (DESIGN §7.1-7.3)."""

from __future__ import annotations

import torch

from wavelet_rope.perceptual_rope import (
    AXIS_DIM,
    PerceptualMemory,
    PerceptualMixer,
    Projector,
    apply_two_axis_rope,
    axis_inv_freq,
    tau_from_anchor,
)


def test_two_axis_rope_is_relative_per_axis() -> None:
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(1, 2, 1, 64, generator=gen)
    k = torch.randn(1, 2, 1, 64, generator=gen)
    inv = axis_inv_freq()

    def score(p_q, p_k, t_q, t_k):
        rq = apply_two_axis_rope(q, torch.tensor([[p_q]]), torch.tensor([[t_q]]), inv)
        rk = apply_two_axis_rope(k, torch.tensor([[p_k]]), torch.tensor([[t_k]]), inv)
        pos = (rq[..., :AXIS_DIM] * rk[..., :AXIS_DIM]).sum(-1)
        tim = (rq[..., AXIS_DIM:] * rk[..., AXIS_DIM:]).sum(-1)
        return pos, tim

    pos_a, tim_a = score(3.0, 10.0, 1.5, 4.0)
    pos_b, tim_b = score(103.0, 110.0, 1.5, 4.0)  # same ordinal gap, same times
    assert torch.allclose(pos_a, pos_b, atol=1e-4) and torch.allclose(tim_a, tim_b, atol=1e-4)
    pos_c, tim_c = score(3.0, 10.0, 21.5, 24.0)  # same time gap, shifted origin
    assert torch.allclose(tim_a, tim_c, atol=1e-4) and torch.allclose(pos_a, pos_c, atol=1e-4)
    pos_d, tim_d = score(3.0, 10.0, 1.5, 9.0)  # different time gap changes only the time term
    assert torch.allclose(pos_a, pos_d, atol=1e-4) and not torch.allclose(tim_a, tim_d, atol=1e-3)
    pos_e, tim_e = score(3.0, 40.0, 1.5, 4.0)  # different ordinal gap changes only the positional term
    assert torch.allclose(tim_a, tim_e, atol=1e-4) and not torch.allclose(pos_a, pos_e, atol=1e-3)


def test_memory_ordinals_are_immutable_and_tau_is_shared() -> None:
    mem = PerceptualMemory(t_origin=0.0, dt_ref=1.0)
    h0 = torch.randn(10, 512)
    t0 = torch.arange(10, dtype=torch.float64) * 640.0  # anchors every 640 samples
    v0 = mem.append(h0, t0, kind="global", start=0, end=6400, channel_ids=("ch_00",))
    assert torch.equal(v0.p, torch.arange(10, dtype=torch.float32))
    assert torch.allclose(v0.tau, torch.arange(10, dtype=torch.float32) * 10.0)  # 640 samples = 10 units of 64
    h1 = torch.randn(4, 512)
    t1 = torch.tensor([640.0, 660.0, 680.0, 700.0], dtype=torch.float64)
    v1 = mem.append(h1, t1, kind="local", start=640, end=704, channel_ids=("ch_00",))
    assert torch.equal(v1.p, torch.arange(10, 14, dtype=torch.float32))
    assert torch.equal(v0.p, torch.arange(10, dtype=torch.float32))  # unchanged after a zoom
    assert float(v1.tau[0]) == float(v0.tau[1])  # same absolute anchor -> same tau across views
    h, p, tau = mem.stacked()
    assert h.shape == (14, 512) and p.shape == (14,) and tau.shape == (14,)
    assert torch.allclose(tau_from_anchor(torch.tensor([128.0], dtype=torch.float64), 0.0, 1.0), torch.tensor([2.0]))


def test_mixer_and_projector_shapes_and_memory_use() -> None:
    torch.manual_seed(0)
    mixer = PerceptualMixer()
    proj = Projector(2560)
    h0 = torch.randn(1, 12, 512)
    p0 = torch.arange(12, dtype=torch.float32)[None]
    tau0 = torch.rand(1, 12) * 100
    z0 = mixer(h0, p0, tau0, h0, p0, tau0)
    assert z0.shape == (1, 12, 512) and torch.isfinite(z0).all()
    assert proj(z0[0]).shape == (12, 2560)
    # a zoom: 5 new tokens attending to memory = global + themselves
    h1 = torch.randn(1, 5, 512)
    p1 = torch.arange(12, 17, dtype=torch.float32)[None]
    tau1 = torch.rand(1, 5) * 100
    mem_h = torch.cat([h0, h1], 1)
    mem_p = torch.cat([p0, p1], 1)
    mem_tau = torch.cat([tau0, tau1], 1)
    z1 = mixer(h1, p1, tau1, mem_h, mem_p, mem_tau)
    assert z1.shape == (1, 5, 512)
    # the zoom output depends on the global memory content
    z1_other = mixer(h1, p1, tau1, torch.cat([h0 * 0 + 1.0, h1], 1), mem_p, mem_tau)
    assert not torch.allclose(z1, z1_other)
    # masked memory tokens do not contribute
    valid = torch.ones(1, 17, dtype=torch.bool)
    valid[0, :12] = False
    z1_masked = mixer(h1, p1, tau1, mem_h, mem_p, mem_tau, valid)
    z1_self = mixer(h1, p1, tau1, h1, p1, tau1)
    assert torch.allclose(z1_masked, z1_self, atol=1e-5)
    # gradients reach both blocks and the projector
    loss = proj(z1[0]).sum()
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in mixer.parameters())
    assert all(p.grad is not None for p in proj.parameters())
