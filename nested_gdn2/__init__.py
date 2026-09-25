"""Makes NestedGDN-2 loadable like any other transformers architecture.

The Auto registrations let our own code say ``AutoModelForCausalLM``; the
``register_for_auto_class`` calls make ``save_pretrained`` ship the modeling
code with the weights, so a checkpoint loads under ``trust_remote_code=True``
in scripts that have never heard of this package.
"""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from nested_gdn2.models.configuration_nested_gdn2 import NestedGDN2Config
from nested_gdn2.models.modeling_nested_gdn2 import (
    NestedGDN2ForCausalLM,
    NestedGDN2Model,
)

AutoConfig.register(NestedGDN2Config.model_type, NestedGDN2Config)
AutoModel.register(NestedGDN2Config, NestedGDN2Model)
AutoModelForCausalLM.register(NestedGDN2Config, NestedGDN2ForCausalLM)

NestedGDN2Config.register_for_auto_class()
NestedGDN2Model.register_for_auto_class("AutoModel")
NestedGDN2ForCausalLM.register_for_auto_class("AutoModelForCausalLM")

__all__ = ["NestedGDN2Config", "NestedGDN2ForCausalLM", "NestedGDN2Model"]
