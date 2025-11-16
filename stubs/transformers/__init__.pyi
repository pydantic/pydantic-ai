from typing import Any

from . import modeling_utils, processing_utils, tokenization_utils
from .modeling_utils import PreTrainedModel
from .processing_utils import ProcessorMixin
from .tokenization_utils import PreTrainedTokenizer

class AutoModelForCausalLM(PreTrainedModel):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> PreTrainedModel: ...

class AutoTokenizer(PreTrainedTokenizer):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> PreTrainedTokenizer: ...

class AutoProcessor(ProcessorMixin):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> ProcessorMixin: ...

class LlavaForConditionalGeneration(PreTrainedModel):
    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> PreTrainedModel: ...

def from_pretrained(*args: Any, **kwargs: Any) -> Any: ...
