from typing import Any

from .abstract import AbstractCapability, BeforeModelRequestContext
from .builtin_tool import BuiltinToolCapability
from .combined import CombinedCapability
from .history_processor import HistoryProcessorCapability
from .image_generation import ImageGeneration
from .instructions import Instructions
from .mcp import MCP

# Short name is intentional — passing a dict is enough to get type checking,
# and users rarely need both this and settings.ModelSettings in the same scope.
from .model_settings import ModelSettings
from .thinking import Thinking
from .toolset import Toolset
from .web_fetch import WebFetch
from .web_search import WebSearch

DEFAULT_CAPABILITY_TYPES: tuple[type[AbstractCapability[Any]], ...] = (
    ImageGeneration,
    Instructions,
    MCP,
    ModelSettings,
    Thinking,
    WebFetch,
    WebSearch,
)
"""Default capability types that support spec-based construction."""

# Backward-compatible computed registry
CAPABILITY_TYPES: dict[str, type[AbstractCapability[Any]]] = {
    name: cls
    for cls in (
        HistoryProcessorCapability,
        ImageGeneration,
        Instructions,
        MCP,
        ModelSettings,
        Thinking,
        Toolset,
        WebFetch,
        WebSearch,
    )
    if (name := cls.get_serialization_name()) is not None
}

__all__ = [
    'AbstractCapability',
    'BeforeModelRequestContext',
    'BuiltinToolCapability',
    'CAPABILITY_TYPES',
    'DEFAULT_CAPABILITY_TYPES',
    'ImageGeneration',
    'Instructions',
    'HistoryProcessorCapability',
    'MCP',
    'ModelSettings',
    'Thinking',
    'Toolset',
    'WebFetch',
    'WebSearch',
    'CombinedCapability',
]
