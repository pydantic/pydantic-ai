from __future__ import annotations

from typing import Any

from mlx.nn import Module
from transformers.tokenization_utils import PreTrainedTokenizer

def load(model_path: str) -> tuple[Module, PreTrainedTokenizer]: ...
def generate_step(*args: Any, **kwargs: Any) -> Any: ...
