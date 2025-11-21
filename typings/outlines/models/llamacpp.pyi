from typing import TYPE_CHECKING

from outlines.models.base import Model

if TYPE_CHECKING:
    from llama_cpp import Llama

class LlamaCpp(Model): ...

def from_llamacpp(model: Llama) -> LlamaCpp: ...
