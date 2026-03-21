from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.builtin_tools import ImageGenerationTool
from pydantic_ai.tools import AgentDepsT, BuiltinToolFunc, Tool

from .builtin_tool import BuiltinToolCapability


@dataclass(init=False)
class ImageGeneration(BuiltinToolCapability[AgentDepsT]):
    """Image generation capability.

    Uses the model's builtin image generation when available. No default local
    fallback — provide a custom ``local`` tool if needed.
    """

    def __init__(
        self,
        builtin: ImageGenerationTool | BuiltinToolFunc[AgentDepsT] | bool = True,
        local: Tool[Any] | Callable[..., Any] | Literal[False] | None = None,
    ) -> None:
        self.builtin = builtin
        self.local = local
        self.__post_init__()

    def _default_builtin(self) -> ImageGenerationTool:
        return ImageGenerationTool()

    def _builtin_unique_id(self) -> str:
        return ImageGenerationTool.kind
