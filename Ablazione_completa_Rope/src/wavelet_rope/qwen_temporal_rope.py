"""Modality wildcard for the eight Qwen3.5 full-attention layers (DESIGN §7.4, Astra 14/09 §2).

The rule, for query i and key j with native decoder positions ``p`` and physical anchor times ``tau``
(valid only for signal tokens, ``m=1``)::

    S_P(i,j) = qP_i^T R_P(p_j - p_i) kP_j                      (16 native even pairs, always positional)
    S_U(i,j) = qU_i^T kU_j                                     (192 unrotated dims)
    S_T(i,j) = qT_i^T R_T(tau_j - tau_i) kT_j   if m_i = m_j = 1   [signal-signal: physical relative time]
             = qT_i^T R_T(p_j - p_i) kT_j       if m_i = m_j = 0   [text-text: native relative position]
             = qT_i^T kT_j                      if m_i != m_j      [mixed: content without temporal phase]
    logit    = (S_P + S_T + S_U) / sqrt(256), one joint causal softmax over all admitted keys.

It is realised EXACTLY by a deterministic expansion with no new weights (DESIGN §7.4 "Fattorizzazione")::

    a = R_P(p) xP,  b = R_T(tau) xT,  c = R_T(p) xT,  raw = xT
    Q* = [a_q, qU, m*b_q, (1-m)*c_q, m*raw_q, (1-m)*raw_q]                 (32+192+4*32 = 352)
    K* = [a_k, kU, m*b_k, (1-m)*c_k, (1-m)*raw_k, m*raw_k]

During the P1 curriculum (lambda in [0,1)) the T logits are interpolated before the softmax,
``S_T = (1-lambda) * S_T_native + lambda * S_T_wildcard``, which is the 384-dim factorisation
``[a, U, sqrt(lambda) * (four wildcard blocks), sqrt(1-lambda) * c]``.  lambda is a scheduler value, never
a parameter, and the square roots are not differentiated.

Native pair layout: Qwen3.5 rotates the first 64 of the 256 head dims with ``rotate_half``, i.e. pair
``r`` occupies dims ``(r, r+32)`` for ``r = 0..31`` with ``inv_freq[r] = theta^(-2r/64)``.  Even ``r`` are
the P pairs, odd ``r`` the T pairs, so both axes keep the full native frequency range.  On pure text the
rule reduces to the native rotary attention for every lambda.

Backends (measured on 14/09, RISULTATI-GATE-BACKEND-20260914.md):
* ``FLEX_PAD512``: FlexAttention (Triton) with Q*/K* zero-padded to 512 — exact, MI300X lane (user decision).
* ``EFFICIENT_ATTENTION``: PyTorch SDPA memory-efficient kernel with the raw 352/384 dims — CUDA lane.
* ``MATH``: eager reference for tests; never a training backend.

Integration is per instance: ``wrap_full_attention_layers`` overrides ``forward`` on the eight
``Qwen3_5Attention`` objects only (no global monkeypatch), keeping ``q_proj/k_proj/v_proj/o_proj`` so LoRA
targets resolve by name and adapters are applied before the expansion.  The KV cache stores K* (already
lifted) at the native 4 KV heads; GQA expansion 4 -> 16 is explicit at attention time.
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

HEAD_DIM = 256
ROTARY_DIM = 64
N_PAIRS = ROTARY_DIM // 2  # 32 native pairs
UNROTATED_DIM = HEAD_DIM - ROTARY_DIM  # 192
QK_DIM_REGIME = 352
QK_DIM_CURRICULUM = 384
VALUE_DIM = 256
ATTENTION_SCALE = 1.0 / math.sqrt(HEAD_DIM)  # 0.0625, never 1/sqrt(352)
BACKENDS = ("FLEX_PAD512", "EFFICIENT_ATTENTION", "MATH")


def native_inv_freq(theta: float, rotary_dim: int = ROTARY_DIM, device=None) -> Tensor:
    """The native Qwen3.5 inverse frequencies, one per pair (fp32)."""
    return 1.0 / (
        float(theta) ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )


def _rot(x1: Tensor, x2: Tensor, angle: Tensor) -> Tensor:
    """Rotate pairs (x1, x2) by ``angle`` (rotate_half convention); returns cat(y1, y2)."""
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def lift(x: Tensor, p: Tensor, m: Tensor, tau: Tensor, inv_freq: Tensor, lam: float, *, is_key: bool) -> Tensor:
    """Expand normalized Q or K [B, H, S, 256] into Q*/K* [B, H, S, 352 or 384] in fp32.

    ``p``, ``m``, ``tau`` are [B, S] (native position, signal flag as 0/1, physical time); ``inv_freq`` is [32].
    """
    if x.shape[-1] != HEAD_DIM:
        raise ValueError(f"expected head dim {HEAD_DIM}, got {x.shape[-1]}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must lie in [0, 1]")
    x = x.float()
    x1, x2, u = x[..., :N_PAIRS], x[..., N_PAIRS:ROTARY_DIM], x[..., ROTARY_DIM:]
    x_p1, x_p2 = x1[..., 0::2], x2[..., 0::2]  # P pairs: even r
    x_t1, x_t2 = x1[..., 1::2], x2[..., 1::2]  # T pairs: odd r
    f_p = inv_freq[0::2].to(x.device, torch.float32)
    f_t = inv_freq[1::2].to(x.device, torch.float32)
    p4 = p.to(torch.float32)[:, None, :, None]
    tau4 = tau.to(torch.float32)[:, None, :, None]
    m4 = m.to(torch.float32)[:, None, :, None]
    a = _rot(x_p1, x_p2, p4 * f_p)  # [B,H,S,32]
    c = _rot(x_t1, x_t2, p4 * f_t)  # native T
    b = _rot(x_t1, x_t2, tau4 * f_t)  # physical T
    raw = torch.cat((x_t1, x_t2), dim=-1)
    if is_key:
        blocks = [m4 * b, (1.0 - m4) * c, (1.0 - m4) * raw, m4 * raw]
    else:
        blocks = [m4 * b, (1.0 - m4) * c, m4 * raw, (1.0 - m4) * raw]
    if lam < 1.0:
        s_w, s_n = math.sqrt(lam), math.sqrt(1.0 - lam)
        blocks = [s_w * blk for blk in blocks] + [s_n * c]
    return torch.cat([a, u] + blocks, dim=-1)


def pairwise_rule_scores(q: Tensor, k: Tensor, p_q: Tensor, p_k: Tensor, m_q: Tensor, m_k: Tensor,
                         tau_q: Tensor, tau_k: Tensor, inv_freq: Tensor, lam: float = 1.0) -> Tensor:
    """Reference logits [B, H, Sq, Sk] computed case by case (fp32, O(S^2)); for tests only.

    ``k`` must already be expanded to the query heads.  Returns S_P + S_T + S_U (unscaled).
    """
    q = q.float()
    k = k.float()
    f_p = inv_freq[0::2].to(q.device, torch.float32)
    f_t = inv_freq[1::2].to(q.device, torch.float32)

    def parts(x):
        x1, x2, u = x[..., :N_PAIRS], x[..., N_PAIRS:ROTARY_DIM], x[..., ROTARY_DIM:]
        return (x1[..., 0::2], x2[..., 0::2]), (x1[..., 1::2], x2[..., 1::2]), u

    (qp1, qp2), (qt1, qt2), qu = parts(q)
    (kp1, kp2), (kt1, kt2), ku = parts(k)

    def rel_scores(x1, x2, y1, y2, ang_x, ang_y):
        # x rotated by ang_x dotted with y rotated by ang_y == relative rotation by (ang_y - ang_x)
        rx = _rot(x1, x2, ang_x)
        ry = _rot(y1, y2, ang_y)
        return rx @ ry.transpose(-1, -2)

    pq4 = p_q.float()[:, None, :, None]
    pk4 = p_k.float()[:, None, :, None]
    s_p = rel_scores(qp1, qp2, kp1, kp2, pq4 * f_p, pk4 * f_p)
    s_u = qu @ ku.transpose(-1, -2)
    t_native = rel_scores(qt1, qt2, kt1, kt2, pq4 * f_t, pk4 * f_t)
    t_phys = rel_scores(qt1, qt2, kt1, kt2, tau_q.float()[:, None, :, None] * f_t,
                        tau_k.float()[:, None, :, None] * f_t)
    t_raw = torch.cat((qt1, qt2), -1) @ torch.cat((kt1, kt2), -1).transpose(-1, -2)
    mq = m_q.float()[:, None, :, None]
    mk = m_k.float()[:, None, None, :]
    both = mq * mk
    neither = (1 - mq) * (1 - mk)
    mixed = 1 - both - neither
    t_wild = both * t_phys + neither * t_native + mixed * t_raw
    s_t = (1.0 - lam) * t_native + lam * t_wild
    return s_p + s_t + s_u


@dataclass
class WildcardContext:
    """Shared per-forward metadata for the wrapped layers (set by the model before each backbone call)."""

    backend: str = "FLEX_PAD512"
    pad_to: int = 512
    lam: float = 1.0
    flex_dynamic: bool = True
    is_signal: Tensor | None = None  # [B, S_new] bool
    tau: Tensor | None = None  # [B, S_new] float32
    _flex: Callable | None = field(default=None, repr=False)
    calls: int = 0
    cache_lam: float | None = None

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"unknown backend {self.backend!r}; expected one of {BACKENDS}")
        if self.backend == "FLEX_PAD512" and self.pad_to != 512:
            raise ValueError("FLEX_PAD512 requires pad_to == 512")

    def set_tokens(self, is_signal: Tensor, tau: Tensor) -> None:
        if is_signal.ndim != 2 or tau.shape != is_signal.shape:
            raise ValueError("is_signal and tau must both be [B, S]")
        if is_signal.dtype != torch.bool:
            raise TypeError("is_signal must be bool")
        if not torch.isfinite(tau).all():
            raise ValueError("tau must be finite (use 0 on text tokens)")
        self.is_signal = is_signal
        self.tau = tau.to(torch.float32)

    def tokens_for(self, batch: int, seq: int, device) -> tuple[Tensor, Tensor]:
        if self.is_signal is None or self.tau is None:
            raise RuntimeError("WildcardContext.set_tokens must be called before the backbone forward")
        if tuple(self.is_signal.shape) != (batch, seq):
            raise RuntimeError(
                f"token metadata shape {tuple(self.is_signal.shape)} does not match hidden states {(batch, seq)}"
            )
        return self.is_signal.to(device), self.tau.to(device)

    def flex_fn(self) -> Callable:
        if self._flex is None:
            import torch._dynamo
            from torch.nn.attention.flex_attention import flex_attention

            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            # Astra 15/09 §1: recompilations are logged with their reason and cache exhaustion must surface
            # as an error, never as a silent fallback to eager attention.
            if hasattr(torch._dynamo.config, "fail_on_recompile_limit_hit"):
                torch._dynamo.config.fail_on_recompile_limit_hit = True
            try:
                torch._logging.set_logs(recompiles=True)
            except Exception:  # noqa: BLE001 - logging is best effort, the failure flag is not
                pass
            self._flex = torch.compile(flex_attention, dynamic=self.flex_dynamic)
        return self._flex


def _pad_last(x: Tensor, size: int) -> Tensor:
    if x.shape[-1] == size:
        return x
    if x.shape[-1] > size:
        raise ValueError(f"cannot pad {x.shape[-1]} down to {size}")
    return F.pad(x, (0, size - x.shape[-1]))


def wildcard_attention(q_star: Tensor, k_star: Tensor, v: Tensor, *, ctx: WildcardContext, past_len: int,
                       dtype: torch.dtype) -> Tensor:
    """Single joint causal softmax over lifted Q*/K* (explicit GQA expansion), returns [B, H, Sq, 256]."""
    heads = q_star.shape[1]
    kv_heads = k_star.shape[1]
    if heads % kv_heads:
        raise ValueError("query heads must be a multiple of kv heads")
    rep = heads // kv_heads
    k16 = k_star.repeat_interleave(rep, dim=1) if rep > 1 else k_star
    v16 = v.repeat_interleave(rep, dim=1) if rep > 1 else v
    q_len, kv_len = q_star.shape[2], k16.shape[2]
    if kv_len != past_len + q_len:
        raise ValueError("cache length does not match past_len + q_len")

    if ctx.backend == "FLEX_PAD512":
        from torch.nn.attention.flex_attention import create_block_mask

        qq = _pad_last(q_star, ctx.pad_to).to(dtype)
        kk = _pad_last(k16, ctx.pad_to).to(dtype)
        vv = v16.to(dtype)
        offset = past_len

        def mask_mod(b, h, qi, kj):
            return kj <= qi + offset

        block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=qq.device)
        return ctx.flex_fn()(qq, kk, vv, block_mask=block_mask, scale=ATTENTION_SCALE)

    from torch.nn.attention import SDPBackend, sdpa_kernel

    qq, kk, vv = q_star.to(dtype), k16.to(dtype), v16.to(dtype)
    if ctx.backend == "EFFICIENT_ATTENTION":
        backend = SDPBackend.EFFICIENT_ATTENTION
    else:
        backend = SDPBackend.MATH
        qq, kk, vv = qq.float(), kk.float(), vv.float()
    if past_len == 0:
        with sdpa_kernel([backend]):
            out = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True, scale=ATTENTION_SCALE)
    else:
        qi = torch.arange(q_len, device=qq.device)[:, None] + past_len
        kj = torch.arange(kv_len, device=qq.device)[None, :]
        mask = kj <= qi
        with sdpa_kernel([backend]):
            out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask, scale=ATTENTION_SCALE)
    return out.to(dtype)


def wildcard_attention_forward(self, hidden_states: Tensor, position_embeddings=None, attention_mask=None,
                               past_key_values=None, cache_position=None, position_ids=None, **kwargs):
    """Replacement ``forward`` for one ``Qwen3_5Attention`` instance (bound per instance)."""
    ctx: WildcardContext = self._wr_ctx
    batch, seq, _ = hidden_states.shape
    hidden_shape = (batch, seq, -1, self.head_dim)
    query_states, gate = torch.chunk(self.q_proj(hidden_states).view(batch, seq, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(batch, seq, -1)
    query_states = self.q_norm(query_states.reshape(hidden_shape)).transpose(1, 2)  # [B,16,S,256]
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # [B,4,S,256]
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if cache_position is None:
        cache_position = torch.arange(seq, device=hidden_states.device)
    past_len = int(cache_position[0].item())
    if past_len > 0 and past_key_values is None:
        raise RuntimeError("cache_position implies cached keys but no cache was given")
    if past_key_values is not None:
        # cached K* were lifted with one lambda; a generation must keep it constant (validation mid-curriculum
        # uses the current training lambda; final inference uses lambda = 1)
        if past_len == 0:
            ctx.cache_lam = ctx.lam
        elif getattr(ctx, "cache_lam", ctx.lam) != ctx.lam:
            raise RuntimeError("lambda changed while a KV cache built with another lambda is in use")
    if attention_mask is not None and attention_mask.dim() == 2 and not bool(attention_mask.all()):
        raise NotImplementedError("padded batches are not supported: the prototype runs microbatch 1")

    m, tau = ctx.tokens_for(batch, seq, hidden_states.device)
    p = cache_position.to(torch.float32)[None, :].expand(batch, seq)
    inv_freq = self._wr_inv_freq
    q_star = lift(query_states, p, m, tau, inv_freq, ctx.lam, is_key=False)
    k_star = lift(key_states, p, m, tau, inv_freq, ctx.lam, is_key=True)
    if past_key_values is not None:
        k_star, value_states = past_key_values.update(
            k_star.to(hidden_states.dtype), value_states, self.layer_idx, {"cache_position": cache_position}
        )
    attn = wildcard_attention(q_star, k_star, value_states, ctx=ctx, past_len=past_len, dtype=hidden_states.dtype)
    attn = attn.transpose(1, 2).reshape(batch, seq, -1).contiguous()
    attn = attn * torch.sigmoid(gate)
    ctx.calls += 1
    return self.o_proj(attn), None


def wrap_full_attention_layers(text_model: nn.Module, ctx: WildcardContext) -> list[int]:
    """Bind the wildcard forward to every full-attention layer of a Qwen3.5 text model; returns their indices."""
    config = text_model.config
    theta = float(config.rope_parameters["rope_theta"])
    rotary_dim = int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0))
    if rotary_dim != ROTARY_DIM or int(config.head_dim) != HEAD_DIM:
        raise ValueError(f"unexpected Qwen geometry: head_dim={config.head_dim} rotary_dim={rotary_dim}")
    wrapped: list[int] = []
    for index, layer in enumerate(text_model.layers):
        if getattr(layer, "layer_type", None) != "full_attention":
            continue
        attn = layer.self_attn
        if getattr(attn, "_wr_wrapped", False):
            wrapped.append(index)
            continue
        device = attn.q_proj.weight.device
        attn._wr_native_forward = attn.forward
        attn._wr_ctx = ctx
        attn._wr_inv_freq = native_inv_freq(theta, rotary_dim, device=device)
        attn._wr_wrapped = True
        attn.forward = types.MethodType(wildcard_attention_forward, attn)
        wrapped.append(index)
    if len(wrapped) != int(getattr(config, "num_hidden_layers", 0)) // int(getattr(config, "full_attention_interval", 4)):
        # Qwen3.5-4B: 32 layers, one full-attention layer every 4 -> 8 wrapped layers
        raise RuntimeError(f"expected 8 full-attention layers, wrapped {len(wrapped)}: {wrapped}")
    return wrapped


def unwrap_full_attention_layers(text_model: nn.Module) -> None:
    for layer in text_model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and getattr(attn, "_wr_wrapped", False):
            attn.forward = attn._wr_native_forward
            attn._wr_wrapped = False


def lora_target_names(text_model: nn.Module, prefix: str = "") -> list[str]:
    """Fully qualified q/k/v/o projection names of the full-attention layers (DESIGN 10.1: resolved, not assumed)."""
    names: list[str] = []
    for index, layer in enumerate(text_model.layers):
        if getattr(layer, "layer_type", None) != "full_attention":
            continue
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if not hasattr(layer.self_attn, proj):
                raise RuntimeError(f"layer {index} has no {proj}")
            names.append(f"{prefix}layers.{index}.self_attn.{proj}")
    return names


__all__ = [
    "ATTENTION_SCALE",
    "BACKENDS",
    "QK_DIM_CURRICULUM",
    "QK_DIM_REGIME",
    "WildcardContext",
    "lift",
    "lora_target_names",
    "native_inv_freq",
    "pairwise_rule_scores",
    "unwrap_full_attention_layers",
    "wildcard_attention",
    "wildcard_attention_forward",
    "wrap_full_attention_layers",
]
