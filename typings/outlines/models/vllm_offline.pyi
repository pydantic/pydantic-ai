from typing import TYPE_CHECKING

from outlines.models.base import Model

if TYPE_CHECKING:
    from vllm import LLM

class VLLMOffline(Model): ...

def from_vllm_offline(model: LLM) -> VLLMOffline: ...
