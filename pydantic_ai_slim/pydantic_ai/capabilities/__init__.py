from typing import Any

from .abstract import AbstractCapability, ModelRequestContext
from .combined import CombinedCapability
from .history_processor import HistoryProcessor
from .instructions import Instructions

# Short name is intentional — passing a dict is enough to get type checking,
# and users rarely need both this and settings.ModelSettings in the same scope.
from .model_settings import ModelSettings
from .thinking import Thinking
from .toolset import Toolset
from .web_search import WebSearch

DEFAULT_CAPABILITY_TYPES: tuple[type[AbstractCapability[Any]], ...] = (
    Instructions,
    ModelSettings,
    Thinking,
    WebSearch,
)
"""Default capability types that support spec-based construction."""

# Backward-compatible computed registry
CAPABILITY_TYPES: dict[str, type[AbstractCapability[Any]]] = {
    name: cls
    for cls in (
        HistoryProcessor,
        Instructions,
        ModelSettings,
        Thinking,
        Toolset,
        WebSearch,
    )
    if (name := cls.get_serialization_name()) is not None
}

__all__ = [
    'AbstractCapability',
    'ModelRequestContext',
    'CAPABILITY_TYPES',
    'DEFAULT_CAPABILITY_TYPES',
    'Instructions',
    'HistoryProcessor',
    'ModelSettings',
    'Thinking',
    'Toolset',
    'WebSearch',
    'CombinedCapability',
]
