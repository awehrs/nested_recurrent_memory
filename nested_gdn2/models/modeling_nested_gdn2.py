"""NestedGDN-2 language model.

Block structure follows fla's GatedDeltaNetBlock: pre-norm token mixing with a
residual, then pre-norm channel mixing with a residual. Only the token mixer
differs.

Training only -- there is no incremental-decoding path yet, so ``use_cache`` is
rejected rather than silently ignored. ``naive_recurrent_nested_gdn2`` is the
token-step reference if generation is needed before the kernel exists.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from fla.modules import FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as NestedGDN2MLP
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from nested_gdn2.layers.nested_gdn2 import NestedGDN2Attention
from nested_gdn2.models.configuration_nested_gdn2 import NestedGDN2Config

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)

__all__ = [
    "NestedGDN2Block",
    "NestedGDN2ForCausalLM",
    "NestedGDN2Model",
    "NestedGDN2PreTrainedModel",
]


class NestedGDN2Block(GradientCheckpointingLayer):
    def __init__(self, config: NestedGDN2Config, layer_idx: int):
        super().__init__()
        norm = RMSNorm if config.fuse_norm else nn.RMSNorm

        self.attn_norm = norm(config.hidden_size, eps=config.norm_eps)
        self.attn = NestedGDN2Attention(config, layer_idx=layer_idx)
        self.mlp_norm = norm(config.hidden_size, eps=config.norm_eps)
        self.mlp = NestedGDN2MLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.attn_norm(hidden_states))[0]
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class NestedGDN2PreTrainedModel(PreTrainedModel):
    config_class = NestedGDN2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NestedGDN2Block"]

    def _init_weights(self, module: nn.Module):
        # The attention layer sets A_log, dt_bias and the promotion parameters.
        if isinstance(module, NestedGDN2Attention):
            for p in (module.A_log, module.dt_bias):
                p._no_weight_decay = True
            return

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()


class NestedGDN2Model(NestedGDN2PreTrainedModel):
    def __init__(self, config: NestedGDN2Config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            NestedGDN2Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutput:
        if use_cache:
            raise NotImplementedError(
                "NestedGDN2 has no incremental-decoding path yet; use_cache must be False."
            )
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds.")
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(x for x in (hidden_states, all_hidden_states) if x is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states
        )


class NestedGDN2ForCausalLM(NestedGDN2PreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embeddings.weight"}

    def __init__(self, config: NestedGDN2Config):
        super().__init__(config)
        self.model = NestedGDN2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        """Standard causal LM forward.

        With ``fuse_cross_entropy`` the logits are never materialized, which at
        long context is the difference between a [B, T, vocab] tensor fitting
        and not. Evaluations that need per-token losses -- distance-resolved
        loss, for one -- must read ``logits``, so they have to run without it.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=use_cache,
        )
        hidden_states = outputs.last_hidden_state

        fuse = self.config.fuse_cross_entropy and labels is not None and self.training
        logits = None if fuse else self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = torch.cat(
                (labels[..., 1:], labels.new_full((labels.shape[0], 1), -100)), dim=-1
            )
            if fuse:
                if self.criterion is None:
                    self.criterion = FusedLinearCrossEntropyLoss()
                loss = self.criterion(
                    hidden_states.view(-1, self.config.hidden_size),
                    labels.view(-1),
                    self.lm_head.weight,
                    self.lm_head.bias,
                )
            else:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, self.vocab_size).float(), labels.view(-1)
                )

        if not return_dict:
            return tuple(x for x in (loss, logits, outputs.hidden_states) if x is not None)
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states
        )
