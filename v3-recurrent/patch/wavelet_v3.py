"""WaveletV3TSModel — WaveletTSModel (v1) with residual recurrent context.

Registered in train.py as ``model.architecture: wavelet_v3``.  Everything
else (placeholder-token merge, MLP projector 512 -> Qwen dim, LoRA'd backbone,
memory-efficient loss, generation, checkpoint format) is inherited from v1
unchanged: ``from_config`` builds the v1 model and swaps ONLY ``ts_encoder``.
Encoder kwargs come from ``model.encoder`` exactly as for v1 (``type`` and
``model_id`` are dropped).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.models.wavelet import WaveletTSModel
from src.models.wavelet_encoder_v3 import RecurrentWaveletTSEncoder

if TYPE_CHECKING:
    from src.utils.config import ExperimentConfig


class WaveletV3TSModel(WaveletTSModel):

    @classmethod
    def from_config(cls, config: "ExperimentConfig", device: str) -> "WaveletV3TSModel":
        model = super().from_config(config, device)  # v1 build, v1 encoder (discarded below)
        enc_cfg = dict(config.model.encoder)
        enc_cfg.pop("type", None)
        enc_cfg.pop("model_id", None)
        v1 = model.ts_encoder
        enc = RecurrentWaveletTSEncoder(**enc_cfg)
        if enc.output_dim != v1.output_dim:
            raise ValueError(
                f"v3 output_dim {enc.output_dim} != v1/projector dim {v1.output_dim}"
            )
        # Preserve the exact va1 starting point.  RecurrentWaveletTSEncoder
        # inherits the v1 token path, and these are precisely the parameters
        # that the just-built v1 model would otherwise have used.
        missing, unexpected = enc.load_state_dict(v1.state_dict(), strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected v1 encoder keys while building v3: {unexpected}")
        base_keys = set(v1.state_dict())
        if base_keys & set(missing):
            raise RuntimeError(f"v3 did not accept all v1 encoder keys: {missing}")
        enc = enc.to(device=device, dtype=next(v1.parameters()).dtype)
        model.ts_encoder = enc
        if getattr(model, "freeze_ts_encoder", False):
            for p in enc.parameters():
                p.requires_grad_(False)
        n_par = sum(p.numel() for p in enc.parameters())
        print(f"  v3 encoder: tied={enc.tied} level_topm={enc.level_topm} "
              f"gate={enc.level_gate} depth_mode={enc.depth_mode} "
              f"context_scale={enc.context_scale.item():.6g} params={n_par/1e6:.2f}M")
        return model
