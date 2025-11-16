from typing import Any
from outlines.models.base import Model

from mlx.nn import Module
from transformers.tokenization_utils import PreTrainedTokenizer


class MLXLM(Model):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


def from_mlxlm(model: Module, tokenizer: PreTrainedTokenizer) -> MLXLM: ...
