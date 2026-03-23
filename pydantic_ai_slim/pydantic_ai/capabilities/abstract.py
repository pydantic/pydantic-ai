from __future__ import annotations

from abc import ABC
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeAlias

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ToolCallPart
from pydantic_ai.tools import AgentBuiltinTool, AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, AgentToolset

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai.agent.abstract import AgentModelSettings
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.result import FinalResult
    from pydantic_ai.run import AgentRunResult
    from pydantic_graph import End

# --- Handler type aliases for use in hook method signatures ---
# These make it easier to write correct type annotations when subclassing AbstractCapability.

AgentNode: TypeAlias = '_agent_graph.AgentNode[AgentDepsT, Any]'
"""Type alias for an agent graph node (`UserPromptNode`, `ModelRequestNode`, `CallToolsNode`)."""

NodeResult: TypeAlias = '_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]'
"""Type alias for the result of executing an agent graph node: either the next node or `End`."""

WrapRunHandler: TypeAlias = 'Callable[[], Awaitable[AgentRunResult[Any]]]'
"""Handler type for [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run]."""

WrapNodeRunHandler: TypeAlias = 'Callable[[_agent_graph.AgentNode[AgentDepsT, Any]], Awaitable[_agent_graph.AgentNode[AgentDepsT, Any] | End[FinalResult[Any]]]]'
"""Handler type for [`wrap_node_run`][pydantic_ai.capabilities.AbstractCapability.wrap_node_run]."""

WrapModelRequestHandler: TypeAlias = 'Callable[[ModelRequestContext], Awaitable[ModelResponse]]'
"""Handler type for [`wrap_model_request`][pydantic_ai.capabilities.AbstractCapability.wrap_model_request]."""

RawToolArgs: TypeAlias = 'str | dict[str, Any]'
"""Type alias for raw (pre-validation) tool arguments."""

ValidatedToolArgs: TypeAlias = 'dict[str, Any]'
"""Type alias for validated tool arguments."""

WrapToolValidateHandler: TypeAlias = 'Callable[[str | dict[str, Any]], Awaitable[dict[str, Any]]]'
"""Handler type for [`wrap_tool_validate`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_validate]."""

WrapToolExecuteHandler: TypeAlias = 'Callable[[dict[str, Any]], Awaitable[Any]]'
"""Handler type for [`wrap_tool_execute`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_execute]."""


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

    See the [capabilities documentation](capabilities.md) for built-in capabilities.

    [`get_serialization_name`][pydantic_ai.capabilities.AbstractCapability.get_serialization_name]
    and [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] support
    YAML/JSON specs (via [`Agent.from_spec`][pydantic_ai.Agent.from_spec]); they have
    sensible defaults and typically don't need to be overridden.
    """

    @property
    def has_wrap_node_run(self) -> bool:
        """Whether this capability (or any sub-capability) overrides wrap_node_run."""
        return type(self).wrap_node_run is not AbstractCapability.wrap_node_run

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
        """Return instructions to include in the system prompt, or None.

        The returned value can be a static string, a [`TemplateStr`][pydantic_ai.TemplateStr],
        or a callable that receives [`RunContext`][pydantic_ai.tools.RunContext] and returns
        instructions dynamically.
        """
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

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's assembled toolset, or return None to leave it unchanged.

        Called per-run with the combined non-output toolset (after agent-level
        [`prepare_tools`][pydantic_ai.tools.ToolsPrepareFunc] wrapping).
        Output tools are added separately and are not included.

        Unlike the other `get_*` methods which are called once at agent construction,
        this is called each run (after [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run]).
        When multiple capabilities provide wrappers, each receives the already-wrapped
        toolset from earlier capabilities (first capability wraps innermost).

        Use this to apply cross-cutting toolset wrappers like
        [`PreparedToolset`][pydantic_ai.toolsets.PreparedToolset],
        [`FilteredToolset`][pydantic_ai.toolsets.FilteredToolset],
        or custom [`WrapperToolset`][pydantic_ai.toolsets.WrapperToolset] subclasses.
        """
        return None

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
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Wraps the entire agent run. ``handler()`` executes the run.

        Works with both :meth:`agent.run() <pydantic_ai.Agent.run>` and
        :meth:`agent.iter() <pydantic_ai.Agent.iter>`.

        If ``handler()`` raises and this method catches the exception and
        returns a result instead, the error is suppressed and the recovery
        result is used.

        If this method does not call ``handler()`` (short-circuit), the run
        is skipped and the returned result is used directly.

        Note: if the caller cancels the run (e.g. by breaking out of an
        ``iter()`` loop), this method receives an :class:`asyncio.CancelledError`.
        Implementations that hold resources should handle cleanup accordingly.
        """
        return await handler()

    async def wrap_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        handler: WrapNodeRunHandler[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        """Wraps execution of each agent graph node (run step).

        Called for every node in the agent graph (`UserPromptNode`,
        `ModelRequestNode`, `CallToolsNode`).  `handler(node)` executes
        the node and returns the next node (or `End`).

        Override to inspect or modify nodes before execution, inspect or modify
        the returned next node, call `handler` multiple times (retry), or
        return a different node to redirect graph progression.

        Note: this hook fires when using [`agent.run()`][pydantic_ai.Agent.run] and when
        manually driving an [`agent.iter()`][pydantic_ai.Agent.iter] run with
        [`next()`][pydantic_ai.result.AgentRun.next], but it does **not** fire when
        iterating over the run with bare `async for` (which yields stream events, not
        node results).
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
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Wraps the model request. handler() calls the model."""
        return await handler(request_context)

    # --- Tool validate lifecycle hooks ---

    async def before_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: RawToolArgs,
    ) -> RawToolArgs:
        """Modify raw args before validation."""
        return args

    async def after_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        """Modify validated args. Called only on successful validation."""
        return args

    async def wrap_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: RawToolArgs,
        handler: WrapToolValidateHandler,
    ) -> ValidatedToolArgs:
        """Wraps tool argument validation. handler() runs the validation."""
        return await handler(args)

    # --- Tool execute lifecycle hooks ---

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        """Modify validated args before execution."""
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        """Modify result after execution."""
        return result

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Wraps tool execution. handler() runs the tool."""
        return await handler(args)
