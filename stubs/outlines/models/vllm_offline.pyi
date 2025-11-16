from typing import Any

from outlines.models.base import Model

class VLLMOffline(Model): ...

def from_vllm_offline(model: Any) -> VLLMOffline: ...
