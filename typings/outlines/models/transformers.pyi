from typing import Any

from outlines.models.base import Model
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PreTrainedTokenizer

class Transformers(Model): ...
class TransformersMultiModal(Model): ...

def from_transformers(
    model: PreTrainedModel | AutoModelForCausalLM | LlavaForConditionalGeneration,
    tokenizer_or_processor: PreTrainedTokenizer | ProcessorMixin | AutoTokenizer | AutoProcessor,
    *,
    device_dtype: Any = None,
) -> Transformers | TransformersMultiModal: ...
