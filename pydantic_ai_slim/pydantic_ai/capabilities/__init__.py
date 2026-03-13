from typing import Any

from .abstract import AbstractCapability
from .combined import CombinedCapability
from .execution_environment import ExecutionEnvironment
from .history_processor import HistoryProcessorCapability
from .instructions import Instructions
from .model_settings import ModelSettingsCapability
from .thinking import Thinking
from .toolset import Toolset
from .web_search import WebSearch

DEFAULT_CAPABILITY_TYPES: tuple[type[AbstractCapability[Any]], ...] = (
    ExecutionEnvironment,
    Instructions,
    ModelSettingsCapability,
    Thinking,
    WebSearch,
)
"""Default capability types that support spec-based construction."""

# Backward-compatible computed registry
CAPABILITY_TYPES: dict[str, type[AbstractCapability[Any]]] = {
    name: cls
    for cls in (
        ExecutionEnvironment,
        HistoryProcessorCapability,
        Instructions,
        ModelSettingsCapability,
        Thinking,
        Toolset,
        WebSearch,
    )
    if (name := cls.get_serialization_name()) is not None
}

__all__ = [
    'AbstractCapability',
    'CAPABILITY_TYPES',
    'DEFAULT_CAPABILITY_TYPES',
    'Instructions',
    'HistoryProcessorCapability',
    'ModelSettingsCapability',
    'Thinking',
    'Toolset',
    'WebSearch',
    'CombinedCapability',
    'ExecutionEnvironment',
]
