"""WaveletTSModel — ChatTS-style TSLM with a multiresolution wavelet front-end.

Identical to ChatTSModel in every respect (placeholder-token embedding
replacement, projector, LoRA'd Qwen backbone, loss, generation) except the
perception module: WaveletTSEncoder replaces the frozen Chronos-2 encoder.
Uniform patching becomes adaptive resolution allocation — a fixed budget of
tokens spent on the coarse band plus the largest detail coefficients.

Use ``model.architecture: wavelet`` with ``extra_kwargs.encoding_method: raw``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.models.chatts import ChatTSModel, MLPProjector
from src.models.qwen_backbone import QwenBackbone
from src.models.wavelet_encoder import WaveletTSEncoder

if TYPE_CHECKING:
    from src.utils.config import ExperimentConfig


class WaveletTSModel(ChatTSModel):

    @classmethod
    def from_config(cls, config: "ExperimentConfig", device: str) -> "WaveletTSModel":
        backbone = QwenBackbone(model_id=config.backbone.model_id)

        text_tokenizer = backbone.load_tokenizer()
        if text_tokenizer.pad_token is None:
            text_tokenizer.pad_token = text_tokenizer.eos_token

        text_tokenizer.add_special_tokens(
            {"additional_special_tokens": [cls.TS_PAD_TOKEN]}
        )

        llm_model = backbone.load_model(device_map={"": device})
        llm_model.resize_token_embeddings(len(text_tokenizer))
        llm_dim = llm_model.config.hidden_size

        import os as _os
        if _os.environ.get("TSLM_GRAD_CKPT") == "1":
            llm_model.gradient_checkpointing_enable()
            llm_model.enable_input_require_grads()

        enc_cfg = dict(config.model.encoder)
        enc_cfg.pop("type", None)
        enc_cfg.pop("model_id", None)
        ts_encoder = WaveletTSEncoder(**enc_cfg).to(device)

        extra = config.model.extra_kwargs
        freeze_ts_encoder = extra.get("freeze_ts_encoder", False)
        encoding_method = extra.get("encoding_method", "raw")
        freeze_lm_embeddings = extra.get("freeze_lm_embeddings", False)

        ts_projector = None
        if ts_encoder.output_dim != llm_dim:
            proj_cfg = dict(config.model.projector) if config.model.projector else {}
            proj_cfg.pop("type", None)
            proj_cfg.pop("num_latents", None)
            ts_projector = MLPProjector(
                dim=ts_encoder.output_dim, output_dim=llm_dim, **proj_cfg
            ).to(device)

        llm_dtype = next(llm_model.parameters()).dtype
        ts_encoder.to(llm_dtype)
        if ts_projector is not None:
            ts_projector.to(llm_dtype)

        return cls(
            device=device,
            llm_model=llm_model,
            text_tokenizer=text_tokenizer,
            ts_encoder=ts_encoder,
            backbone=backbone,
            patch_size=1,
            num_features=1,
            freeze_ts_encoder=freeze_ts_encoder,
            ts_projector=ts_projector,
            encoding_method=encoding_method,
            freeze_lm_embeddings=freeze_lm_embeddings,
        )
