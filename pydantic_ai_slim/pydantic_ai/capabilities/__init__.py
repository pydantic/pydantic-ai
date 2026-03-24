from typing import Any

from .abstract import (
    AbstractCapability,
    AgentNode,
    NodeResult,
    RawToolArgs,
    ValidatedToolArgs,
    WrapModelRequestHandler,
    WrapNodeRunHandler,
    WrapRunHandler,
    WrapToolExecuteHandler,
    WrapToolValidateHandler,
)
from .builtin_or_local import BuiltinOrLocalTool, BuiltinTool
from .combined import CombinedCapability
from .history_processor import HistoryProcessor
from .image_generation import ImageGeneration
from .mcp import MCP
from .prepare_tools import PrepareTools
from .toolset import Toolset
from .web_fetch import WebFetch
from .web_search import WebSearch

CAPABILITY_TYPES: dict[str, type[AbstractCapability[Any]]] = {
    name: cls
    for cls in (
        BuiltinTool,
        HistoryProcessor,
        ImageGeneration,
        MCP,
        PrepareTools,
        Toolset,
        WebFetch,
        WebSearch,
    )
    if (name := cls.get_serialization_name()) is not None
}
"""Registry of all capability types that have a serialization name, mapping name to class."""

__all__ = [
    'AbstractCapability',
    'AgentNode',
    'NodeResult',
    'RawToolArgs',
    'ValidatedToolArgs',
    'WrapModelRequestHandler',
    'WrapNodeRunHandler',
    'WrapRunHandler',
    'WrapToolExecuteHandler',
    'WrapToolValidateHandler',
    'BuiltinTool',
    'BuiltinOrLocalTool',
    'CAPABILITY_TYPES',
    'ImageGeneration',
    'HistoryProcessor',
    'MCP',
    'PrepareTools',
    'Toolset',
    'WebFetch',
    'WebSearch',
    'CombinedCapability',
]
