from typing import Any

from ._pending_messages import PendingMessageDrainCapability
from .abstract import (
    AbstractCapability,
    AgentNode,
    CapabilityOrdering,
    CapabilityPosition,
    CapabilityRef,
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
from .deferred_tool_handler import HandleDeferredToolCalls
from .hooks import Hooks, HookTimeoutError
from .image_generation import ImageGeneration
from .include_return_schemas import IncludeToolReturnSchemas
from .mcp import MCP
from .prefix_tools import PrefixTools
from .prepare_tools import PrepareTools
from .process_event_stream import ProcessEventStream
from .process_history import (
    HistoryProcessor,  # pyright: ignore[reportDeprecated]
    ProcessHistory,
)
from .reinject_system_prompt import ReinjectSystemPrompt
from .set_tool_metadata import SetToolMetadata
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
        ImageGeneration,
        IncludeToolReturnSchemas,
        MCP,
        PrefixTools,
        PrepareTools,
        ProcessHistory,
        ReinjectSystemPrompt,
        SetToolMetadata,
        Thinking,
        Toolset,
        WebFetch,
        WebSearch,
    )
    if (name := cls.get_serialization_name()) is not None
}
"""Registry of all capability types that have a serialization name, mapping name to class."""

# Note: OpenAICompaction and AnthropicCompaction have serialization names but can't be
# registered here due to circular imports. Use custom_capability_types in AgentSpec instead.

__all__ = [
    'AbstractCapability',
    'PendingMessageDrainCapability',
    'AgentNode',
    'CapabilityOrdering',
    'CapabilityPosition',
    'CapabilityRef',
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
    'IncludeToolReturnSchemas',
    'MCP',
    'PrefixTools',
    'PrepareTools',
    'ProcessEventStream',
    'ProcessHistory',
    'ReinjectSystemPrompt',
    'SetToolMetadata',
    'Thinking',
    'ThreadExecutor',
    'Toolset',
    'WebFetch',
    'WebSearch',
    'WrapperCapability',
    'CombinedCapability',
    'HandleDeferredToolCalls',
    'HookTimeoutError',
    'Hooks',
]
