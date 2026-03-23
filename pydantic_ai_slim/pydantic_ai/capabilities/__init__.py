from typing import Any

from typing_extensions import deprecated

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
from .fastmcp import FastMCP
from .history_processor import HistoryProcessor
from .image_generation import ImageGeneration
from .instructions import Instructions
from .mcp import MCP

# Short name is intentional — passing a dict is enough to get type checking,
# and users rarely need both this and settings.ModelSettings in the same scope.
from .model_settings import ModelSettings
from .prepare_tools import PrepareTools
from .toolset import Toolset
from .web_fetch import WebFetch
from .web_search import WebSearch

BuiltinToolCapability = deprecated('BuiltinToolCapability is deprecated, use BuiltinOrLocalTool instead')(
    BuiltinOrLocalTool
)

CAPABILITY_TYPES: dict[str, type[AbstractCapability[Any]]] = {
    name: cls
    for cls in (
        BuiltinTool,
        FastMCP,
        HistoryProcessor,
        ImageGeneration,
        Instructions,
        MCP,
        ModelSettings,
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
    'BuiltinToolCapability',
    'CAPABILITY_TYPES',
    'FastMCP',
    'ImageGeneration',
    'Instructions',
    'HistoryProcessor',
    'MCP',
    'ModelSettings',
    'PrepareTools',
    'Toolset',
    'WebFetch',
    'WebSearch',
    'CombinedCapability',
]
