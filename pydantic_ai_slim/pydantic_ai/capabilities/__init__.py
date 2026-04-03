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
from .builtin_or_local import BuiltinOrLocalTool
from .builtin_tool import BuiltinTool
from .combined import CombinedCapability
from .history_processor import HistoryProcessor
from .hooks import Hooks, HookTimeoutError
from .image_generation import ImageGeneration
from .mcp import MCP
from .prefix_tools import PrefixTools
from .prepare_tools import PrepareTools
from .thinking import Thinking
from .thread_executor import ThreadExecutor
from .toolset import Toolset
from .web_fetch import WebFetch
from .web_search import WebSearch
from .wrapper import WrapperCapability

CAPABILITY_TYPES: dict[str, type[AbstractCapability[Any]]] = {
    name: cls
    for cls in (
        BuiltinTool,
        HistoryProcessor,
        ImageGeneration,
        MCP,
        PrefixTools,
        PrepareTools,
        Thinking,
        Toolset,
        WebFetch,
        WebSearch,
    )
    if (name := cls.get_serialization_name()) is not None
}
"""Registry of all capability types that have a serialization name, mapping name to class."""

# Include provider-specific compaction capabilities if their SDKs are available
try:
    from pydantic_ai.models.openai import OpenAICompaction

    CAPABILITY_TYPES['OpenAICompaction'] = OpenAICompaction  # pragma: no cover
except ImportError:
    pass

try:
    from pydantic_ai.models.anthropic import AnthropicCompaction

    CAPABILITY_TYPES['AnthropicCompaction'] = AnthropicCompaction  # pragma: no cover
except ImportError:
    pass

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
    'PrefixTools',
    'PrepareTools',
    'Thinking',
    'ThreadExecutor',
    'Toolset',
    'WebFetch',
    'WebSearch',
    'WrapperCapability',
    'CombinedCapability',
    'HookTimeoutError',
    'Hooks',
]
