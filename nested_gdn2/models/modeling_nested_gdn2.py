from fla.models.utils import FLAGenerationMixin
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

logger = logging.get_logger(__name__)


class NestedGDN2Block(nn.Module):
    ...


class NestedGDN2PReTrainedModel(PreTrainedModel, FLAGenerationMixin):
    ...


class NestedGDN2Model(NestedGDN2PReTrainedModel):
    ...


class NestedGDN2ForCausalLM(NestedGDN2PReTrainedModel):
    ...
