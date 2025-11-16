from typing import Any

from mlx.nn import Module
from outlines.models.base import Model
from transformers.tokenization_utils import PreTrainedTokenizer

class MLXLM(Model):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

def from_mlxlm(model: Module, tokenizer: PreTrainedTokenizer) -> MLXLM: ...
