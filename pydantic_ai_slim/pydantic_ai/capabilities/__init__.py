from typing import Any

from .abstract import AbstractCapability, ModelRequestContext
from .builtin_tool import BuiltinTool
from .combined import CombinedCapability
from .history_processor import HistoryProcessor
from .instructions import Instructions

# Short name is intentional — passing a dict is enough to get type checking,
# and users rarely need both this and settings.ModelSettings in the same scope.
from .model_settings import ModelSettings
from .prepare_tools import PrepareTools
from .thinking import Thinking
from .toolset import Toolset
from .web_search import WebSearch

CAPABILITY_TYPES: dict[str, type[AbstractCapability[Any]]] = {
    name: cls
    for cls in (
        BuiltinTool,
        HistoryProcessor,
        Instructions,
        ModelSettings,
        PrepareTools,
        Thinking,
        Toolset,
        WebSearch,
    )
    if (name := cls.get_serialization_name()) is not None
}
"""Registry of all capability types that have a serialization name, mapping name to class."""

__all__ = [
    'AbstractCapability',
    'ModelRequestContext',
    'CAPABILITY_TYPES',
    'BuiltinTool',
    'Instructions',
    'HistoryProcessor',
    'ModelSettings',
    'PrepareTools',
    'Thinking',
    'Toolset',
    'WebSearch',
    'CombinedCapability',
]
