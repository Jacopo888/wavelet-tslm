"""WaveletTSModel — ChatTS-style TSLM with a multiresolution wavelet front-end.

Identical to ChatTSModel in every architectural respect (placeholder-token
embedding replacement, projector, LoRA'd Qwen backbone, loss objective,
generation) except the perception module: WaveletTSEncoder replaces the
frozen Chronos-2 encoder.  Its training forward computes that same loss from
supervised hidden-state positions only to avoid sequence-wide logits.
Uniform patching becomes adaptive resolution allocation — a fixed budget of
tokens spent on the coarse band plus the largest detail coefficients.

Use ``model.architecture: wavelet`` with ``extra_kwargs.encoding_method: raw``.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import torch.nn.functional as F

from src.models.chatts import ChatTSModel, MLPProjector
from src.models.qwen_backbone import QwenBackbone
from src.models.wavelet_encoder import WaveletTSEncoder

if TYPE_CHECKING:
    from src.utils.config import ExperimentConfig


class WaveletTSModel(ChatTSModel):

    def forward(
        self,
        time_series: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        memory_efficient_loss: bool = True,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run Qwen without materializing prompt logits during training.

        The inherited ChatTS loss builds logits for the entire multimodal
        sequence and only then masks the prompt with ``ignore_index=-100``.
        For long multichannel examples, cross entropy upcasts that full
        ``(batch, sequence, vocab)`` tensor to fp32 and creates a multi-GiB
        temporary even though only the short answer suffix contributes.

        Qwen's ``logits_to_keep`` indexes the final hidden states *before*
        ``lm_head``.  Causal LM position ``t`` predicts label ``t + 1``, so
        the indices below are deliberately computed from ``labels[..., 1:]``.
        Selecting exactly its non-ignored entries is mathematically identical
        to the inherited shifted CE (batch size is fixed to one by the run
        configs), while reducing logits to answer-token rows only.

        ``memory_efficient_loss=False`` is a diagnostic escape hatch used by
        ``test_loss_equivalence.py`` to execute the unmodified inherited path
        on small real samples.  Inference also keeps the inherited full-logit
        behavior because it has no labels/loss allocation.
        """
        if labels is None or not memory_efficient_loss:
            return super().forward(
                time_series=time_series,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        if labels.ndim != 2 or labels.size(0) != 1:
            raise ValueError(
                "WaveletTSModel's memory-efficient loss requires batch_size=1"
            )

        ts_features = self._encode_and_project(time_series)
        inputs_embeds = self._merge_features(ts_features, input_ids)

        # shift_labels[:, t] is predicted by hidden_states[:, t].  Keeping the
        # non--100 positions here (rather than after lm_head) is the memory win.
        shift_labels = labels[..., 1:].contiguous()
        keep_positions = torch.nonzero(
            shift_labels[0] != -100, as_tuple=False
        ).flatten()

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=keep_positions,
        )
        logits = outputs.logits
        selected_labels = shift_labels.index_select(1, keep_positions)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            selected_labels.reshape(-1),
            ignore_index=-100,
        )

        # With labels, logits intentionally contain supervised positions only;
        # no training/evaluation caller in this patch consumes full logits.
        return {"logits": logits, "loss": loss}

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
