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
from .fastmcp import FastMCP
from .history_processor import HistoryProcessor
from .hooks import Hooks, HookTimeoutError
from .image_generation import ImageGeneration
from .mcp import MCP
from .prefix_tools import PrefixTools
from .prepare_tools import PrepareTools
from .toolset import Toolset
from .web_fetch import WebFetch
from .web_search import WebSearch
from .wrapper import WrapperCapability

CAPABILITY_TYPES: dict[str, type[AbstractCapability[Any]]] = {
    name: cls
    for cls in (
        BuiltinTool,
        FastMCP,
        HistoryProcessor,
        ImageGeneration,
        MCP,
        PrefixTools,
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
    'FastMCP',
    'ImageGeneration',
    'HistoryProcessor',
    'MCP',
    'PrefixTools',
    'PrepareTools',
    'Toolset',
    'WebFetch',
    'WebSearch',
    'WrapperCapability',
    'CombinedCapability',
    'HookTimeoutError',
    'Hooks',
]
