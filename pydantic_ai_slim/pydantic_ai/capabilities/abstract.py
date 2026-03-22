from __future__ import annotations

from abc import ABC
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentBuiltinTool, AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AgentToolset

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai.agent.abstract import AgentModelSettings
    from pydantic_ai.result import FinalResult
    from pydantic_ai.run import AgentRunResult
    from pydantic_graph import End


@dataclass
class ModelRequestContext:
    """Context passed to and returned from [`AbstractCapability.before_model_request`][pydantic_ai.capabilities.abstract.AbstractCapability.before_model_request].

    Wrapping these parameters in a dataclass instead of a tuple makes the signature
    future-proof: new fields can be added without breaking existing implementations.
    """

    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters


@dataclass
class AbstractCapability(ABC, Generic[AgentDepsT]):
    """Abstract base class for agent capabilities.

    A capability is a reusable, composable unit of agent behavior that can provide
    instructions, model settings, tools, and request/response hooks.

    Lifecycle: capabilities are passed to an [`Agent`][pydantic_ai.Agent] at construction time, where their
    `get_*` methods are called to collect static configuration (instructions, model
    settings, toolsets, builtin tools). Then, on each model request during a run, the
    [`before_model_request`][pydantic_ai.capabilities.AbstractCapability.before_model_request] and
    [`after_model_request`][pydantic_ai.capabilities.AbstractCapability.after_model_request] hooks
    are called to allow dynamic adjustments.

    Built-in capabilities:

    - [`Instructions`][pydantic_ai.capabilities.Instructions] — static or template-based system prompt instructions
    - [`Thinking`][pydantic_ai.capabilities.Thinking] — enables model thinking/reasoning mode
    - [`ModelSettings`][pydantic_ai.capabilities.ModelSettings] — provides extra model settings
    - [`WebSearch`][pydantic_ai.capabilities.WebSearch] — registers the web search builtin tool
    - [`HistoryProcessor`][pydantic_ai.capabilities.HistoryProcessor] — wraps a history processor as a capability
    - [`Toolset`][pydantic_ai.capabilities.Toolset] — registers a toolset with the agent

    Custom capabilities that should work with YAML/JSON specs (via
    [`Agent.from_spec`][pydantic_ai.Agent.from_spec]) can override
    [`get_serialization_name`][pydantic_ai.capabilities.AbstractCapability.get_serialization_name]
    and [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec].
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Return the name used for spec serialization (CamelCase class name by default).

        Return None to opt out of spec-based construction.
        """
        return cls.__name__

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> AbstractCapability[Any]:
        """Create from spec arguments. Default: `cls(*args, **kwargs)`.

        Override when `__init__` takes non-serializable types.
        """
        return cls(*args, **kwargs)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Return the capability instance to use for this agent run.

        Called once per run, before `get_*()` re-extraction and before any hooks fire.
        Override to return a fresh instance for per-run state isolation.
        Default: return `self` (shared across runs).
        """
        return self

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Return static instructions to include in the system prompt, or None."""
        return None

    def get_model_settings(self) -> AgentModelSettings[AgentDepsT] | None:
        """Return model settings to merge into the agent's defaults, or None.

        Return a static `ModelSettings` dict when the settings are known at agent
        construction time and don't change between requests. Return a callable
        that receives [`RunContext`][pydantic_ai.tools.RunContext] when settings
        need to vary per step (e.g. based on `ctx.run_step` or `ctx.deps`).

        When the callable is invoked, `ctx.model_settings` contains the merged
        result of all layers resolved before this capability (model defaults and
        agent-level settings). The returned dict is merged on top of that.
        """
        return None

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Return a toolset to register with the agent, or None."""
        return None

    def get_builtin_tools(self) -> Sequence[AgentBuiltinTool[AgentDepsT]]:
        """Return builtin tools to register with the agent."""
        return []

    # --- Tool preparation hook ---

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Filter or modify tool definitions visible to the model for this step.

        The list contains all tool kinds (function, output, unapproved) distinguished
        by [`tool_def.kind`][pydantic_ai.tools.ToolDefinition.kind]. Return a filtered
        or modified list. Called after the agent-level
        [`prepare_tools`][pydantic_ai.tools.ToolsPrepareFunc] has already run.
        """
        return tool_defs

    # --- Run lifecycle hooks ---

    async def before_run(
        self,
        ctx: RunContext[AgentDepsT],
    ) -> None:
        """Called before the agent run starts. Observe-only; use wrap_run for modification."""

    async def after_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        """Called after the agent run completes. Can modify the result."""
        return result

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: Callable[[], Awaitable[AgentRunResult[Any]]],
    ) -> AgentRunResult[Any]:
        """Wraps the entire agent run. handler() executes the run."""
        return await handler()

    async def wrap_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: _agent_graph.AgentNode[AgentDepsT, Any],
        handler: Callable[
            [_agent_graph.AgentNode[AgentDepsT, Any]],
            Awaitable[_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]],
        ],
    ) -> _agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]:
        """Wraps execution of each agent graph node (run step).

        Called for every node in the agent graph (`UserPromptNode`,
        `ModelRequestNode`, `CallToolsNode`).  `handler(node)` executes
        the node and returns the next node (or `End`).

        Override to inspect or modify nodes before execution, inspect or modify
        the returned next node, call `handler` multiple times (retry), or
        return a different node to redirect graph progression.
        """
        return await handler(node)

    # --- Event stream hook ---

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        """Wraps the event stream for a streamed node. Can observe or transform events."""
        return stream

    # --- Model request lifecycle hooks ---

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Called before each model request. Can modify messages, settings, and parameters."""
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Called after each model response. Can modify the response before further processing."""
        return response

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Wraps the model request. handler() calls the model."""
        return await handler(request_context)

    # --- Tool validate lifecycle hooks ---

    async def before_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: str | dict[str, Any],
    ) -> str | dict[str, Any]:
        """Modify raw args before validation."""
        return args

    async def after_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Modify validated args. Called only on successful validation."""
        return args

    async def wrap_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: str | dict[str, Any],
        handler: Callable[[str | dict[str, Any]], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Wraps tool argument validation. handler() runs the validation."""
        return await handler(args)

    # --- Tool execute lifecycle hooks ---

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """Modify validated args before execution."""
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Modify result after execution."""
        return result

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        """Wraps tool execution. handler() runs the tool."""
        return await handler(args)
