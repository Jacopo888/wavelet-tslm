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

    def get_trainable_parameters(self):
        """Keep recurrent-only parameters in their own optimizer group.

        The inherited group contains the complete encoder.  Splitting by
        parameter identity lets ``training.learning_rate.context`` accelerate
        only the newly initialized loop while the v1 token path keeps the va1
        learning rate.  If the encoder is frozen, none of these parameters is
        present in the inherited groups and no context group is added.
        """
        groups = {
            name: list(params)
            for name, params in super().get_trainable_parameters().items()
        }
        grouped_ids = {id(parameter) for params in groups.values() for parameter in params}
        encoder = self.ts_encoder
        candidates = [
            *encoder.blocks.parameters(),
            *encoder.step_embed.parameters(),
            *encoder.context_norm.parameters(),
            encoder.context_scale,
        ]
        if encoder.gate_head is not None:
            candidates.extend(encoder.gate_head.parameters())
        context = [parameter for parameter in candidates if id(parameter) in grouped_ids]
        if not context:
            return groups

        context_ids = {id(parameter) for parameter in context}
        split = {
            name: [parameter for parameter in params if id(parameter) not in context_ids]
            for name, params in groups.items()
        }
        split = {name: params for name, params in split.items() if params}
        if "context" in split:
            raise RuntimeError("optimizer group 'context' already exists")
        split["context"] = context
        return split

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
