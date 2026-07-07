from typing import Any, TypeAlias

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.native_tools._tool_search import (
    ToolSearchFunc as ToolSearchFunc,
    ToolSearchLocalStrategy as ToolSearchLocalStrategy,
    ToolSearchNativeStrategy as ToolSearchNativeStrategy,
    ToolSearchStrategy as ToolSearchStrategy,
)
from pydantic_ai.output import OutputContext

from ._dynamic import CapabilityFunc, DynamicCapability
from ._tool_search import ToolSearch
from .abstract import (
    AbstractCapability,
    AgentNode,
    CapabilityDescription,
    CapabilityOrdering,
    CapabilityPosition,
    CapabilityRef,
    NodeResult,
    RawOutput,
    RawToolArgs,
    ValidatedToolArgs,
    WrapModelRequestHandler,
    WrapNodeRunHandler,
    WrapOutputProcessHandler,
    WrapOutputValidateHandler,
    WrapRunHandler,
    WrapToolExecuteHandler,
    WrapToolValidateHandler,
)
from .capability import Capability
from .combined import CombinedCapability
from .deferred_tool_handler import HandleDeferredToolCalls
from .hooks import Hooks, HookTimeoutError
from .image_generation import ImageGeneration
from .include_return_schemas import IncludeToolReturnSchemas
from .instrumentation import Instrumentation
from .mcp import MCP
from .native_or_local import NativeOrLocalTool
from .native_tool import NativeTool
from .prefix_tools import PrefixTools
from .prepare_tools import PrepareOutputTools, PrepareTools
from .process_event_stream import ProcessEventStream
from .process_history import ProcessHistory
from .reinject_system_prompt import ReinjectSystemPrompt
from .set_tool_metadata import SetToolMetadata
from .thinking import Thinking
from .thread_executor import ThreadExecutor
from .toolset import Toolset
from .web_fetch import WebFetch
from .web_search import WebSearch
from .wrapper import WrapperCapability
from .x_search import XSearch

AgentCapability: TypeAlias = AbstractCapability[AgentDepsT] | CapabilityFunc[AgentDepsT]
"""A capability or a [`CapabilityFunc`][pydantic_ai.capabilities.CapabilityFunc] that takes a run context and returns one.

Use as the item type for `Agent(capabilities=[...])` and `agent.run(capabilities=[...])`.
Functions are wrapped in a [`DynamicCapability`][pydantic_ai.capabilities.DynamicCapability] automatically.
"""


CAPABILITY_TYPES: dict[str, type[AbstractCapability[Any]]] = {
    name: cls
    for cls in (
        NativeTool,
        ImageGeneration,
        IncludeToolReturnSchemas,
        Instrumentation,
        MCP,
        PrefixTools,
        PrepareTools,
        ProcessHistory,
        ReinjectSystemPrompt,
        SetToolMetadata,
        Thinking,
        ToolSearch,
        Toolset,
        WebFetch,
        WebSearch,
        XSearch,
    )
    if (name := cls.get_serialization_name()) is not None
}
"""Registry of all capability types that have a serialization name, mapping name to class."""

# Note: OpenAICompaction and AnthropicCompaction have serialization names but can't be
# registered here due to circular imports. Use custom_capability_types in AgentSpec instead.

__all__ = [
    'AbstractCapability',
    'AgentCapability',
    'AgentNode',
    'CapabilityDescription',
    'CapabilityFunc',
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
    'RawOutput',
    'WrapOutputValidateHandler',
    'WrapOutputProcessHandler',
    'NativeTool',
    'NativeOrLocalTool',
    'Capability',
    'CAPABILITY_TYPES',
    'ImageGeneration',
    'Instrumentation',
    'IncludeToolReturnSchemas',
    'MCP',
    'PrefixTools',
    'PrepareOutputTools',
    'PrepareTools',
    'ProcessEventStream',
    'ProcessHistory',
    'ReinjectSystemPrompt',
    'SetToolMetadata',
    'Thinking',
    'ThreadExecutor',
    'ToolSearch',
    'ToolSearchFunc',
    'ToolSearchLocalStrategy',
    'ToolSearchNativeStrategy',
    'ToolSearchStrategy',
    'Toolset',
    'WebFetch',
    'WebSearch',
    'WrapperCapability',
    'XSearch',
    'CombinedCapability',
    'DynamicCapability',
    'HandleDeferredToolCalls',
    'HookTimeoutError',
    'Hooks',
    'OutputContext',
]
