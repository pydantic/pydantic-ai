from typing import Any

from mlx.nn import Module
from transformers.tokenization_utils import PreTrainedTokenizer

def load(model_path: str | None = None, *args: Any, **kwargs: Any) -> tuple[Module, PreTrainedTokenizer]: ...
def generate_step(*args: Any, **kwargs: Any) -> Any: ...
