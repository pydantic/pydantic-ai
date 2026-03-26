from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.builtin_tools import ImageGenerationTool
from pydantic_ai.tools import AgentBuiltinTool, AgentDepsT, Tool

from .builtin_or_local import BuiltinOrLocalTool


@dataclass(init=False)
class ImageGeneration(BuiltinOrLocalTool[AgentDepsT]):
    """Image generation capability.

    Uses the model's builtin image generation when available. No default local
    fallback — provide a custom `local` tool if needed.
    """

    def __init__(
        self,
        *,
        builtin: ImageGenerationTool | AgentBuiltinTool[AgentDepsT] | bool = True,
        local: Tool[AgentDepsT] | Callable[..., Any] | Literal[False] | None = None,
    ) -> None:
        self.builtin = builtin
        self.local = local
        self.__post_init__()

    def _default_builtin(self) -> ImageGenerationTool:
        return ImageGenerationTool()

    def _builtin_unique_id(self) -> str:
        return ImageGenerationTool.kind
