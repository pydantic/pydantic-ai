from __future__ import annotations

from abc import ABC
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeAlias

from pydantic import ValidationError

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ToolCallPart
from pydantic_ai.tools import AgentBuiltinTool, AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, AgentToolset

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai.agent.abstract import AgentModelSettings
    from pydantic_ai.capabilities.prefix_tools import PrefixTools
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

    Lifecycle: capabilities are passed to an [`Agent`][pydantic_ai.Agent] at construction time, where
    most `get_*` methods are called to collect static configuration (instructions, model
    settings, toolsets, builtin tools). The exception is
    [`get_wrapper_toolset`][pydantic_ai.capabilities.AbstractCapability.get_wrapper_toolset],
    which is called per-run during toolset assembly. Then, on each model request during a
    run, the [`before_model_request`][pydantic_ai.capabilities.AbstractCapability.before_model_request]
    and [`after_model_request`][pydantic_ai.capabilities.AbstractCapability.after_model_request]
    hooks are called to allow dynamic adjustments.

    See the [capabilities documentation](capabilities.md) for built-in capabilities.

    [`get_serialization_name`][pydantic_ai.capabilities.AbstractCapability.get_serialization_name]
    and [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] support
    YAML/JSON specs (via [`Agent.from_spec`][pydantic_ai.agent.Agent.from_spec]); they have
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

        This method is called once at agent construction time. To get dynamic
        per-request behavior, return a callable that receives
        [`RunContext`][pydantic_ai.tools.RunContext] or a
        `TemplateStr` — not a dynamic string.
        """
        return None

    def get_model_settings(self) -> AgentModelSettings[AgentDepsT] | None:
        """Return model settings to merge into the agent's defaults, or None.

        This method is called once at agent construction time. Return a static
        `ModelSettings` dict when the settings don't change between requests.
        Return a callable that receives [`RunContext`][pydantic_ai.tools.RunContext]
        when settings need to vary per step (e.g. based on `ctx.run_step` or `ctx.deps`).

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
        """Wraps the entire agent run. `handler()` executes the run.

        If `handler()` raises and this method catches the exception and
        returns a result instead, the error is suppressed and the recovery
        result is used.

        If this method does not call `handler()` (short-circuit), the run
        is skipped and the returned result is used directly.

        Note: if the caller cancels the run (e.g. by breaking out of an
        `iter()` loop), this method receives an :class:`asyncio.CancelledError`.
        Implementations that hold resources should handle cleanup accordingly.
        """
        return await handler()

    async def on_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        """Called when the agent run fails with an exception.

        This is the error counterpart to
        [`after_run`][pydantic_ai.capabilities.AbstractCapability.after_run]:
        while `after_run` is called on success, `on_run_error` is called on
        failure (after [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run]
        has had its chance to recover).

        **Raise** the original `error` (or a different exception) to propagate it.
        **Return** an [`AgentRunResult`][pydantic_ai.run.AgentRunResult] to suppress
        the error and recover the run.

        Not called for `GeneratorExit` or `KeyboardInterrupt`.
        """
        raise error

    # --- Node run lifecycle hooks ---

    async def before_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
    ) -> AgentNode[AgentDepsT]:
        """Called before each graph node executes. Can observe or replace the node."""
        return node

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        """Called after each graph node succeeds. Can modify the result (next node or `End`)."""
        return result

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

        Note: this hook fires when using [`agent.run()`][pydantic_ai.agent.AbstractAgent.run],
        [`agent.run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream], and when manually driving
        an [`agent.iter()`][pydantic_ai.agent.Agent.iter] run with
        [`next()`][pydantic_ai.run.AgentRun.next], but it does **not** fire when
        iterating over the run with bare `async for` (which yields stream events, not
        node results).

        When using `agent.run()` with `event_stream_handler`, the handler wraps both
        streaming and graph advancement (i.e. the model call happens inside the wrapper).
        When using `agent.run_stream()`, the handler wraps only graph advancement — streaming
        happens before the wrapper because `run_stream()` must yield the stream to the caller
        while the stream context is still open, which cannot happen from inside a callback.
        """
        return await handler(node)

    async def on_node_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        error: Exception,
    ) -> NodeResult[AgentDepsT]:
        """Called when a graph node fails with an exception.

        This is the error counterpart to
        [`after_node_run`][pydantic_ai.capabilities.AbstractCapability.after_node_run].

        **Raise** the original `error` (or a different exception) to propagate it.
        **Return** a next node or `End` to recover and continue the graph.

        Useful for recovering from
        [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior]
        by redirecting to a different node (e.g. retry with different model settings).
        """
        raise error

    # --- Event stream hook ---

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        """Wraps the event stream for a streamed node. Can observe or transform events."""
        async for event in stream:
            yield event

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

    async def on_model_request_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        """Called when a model request fails with an exception.

        This is the error counterpart to
        [`after_model_request`][pydantic_ai.capabilities.AbstractCapability.after_model_request].

        **Raise** the original `error` (or a different exception) to propagate it.
        **Return** a [`ModelResponse`][pydantic_ai.messages.ModelResponse] to suppress
        the error and use the response as if the model call succeeded.

        Not called for [`SkipModelRequest`][pydantic_ai.exceptions.SkipModelRequest].
        """
        raise error

    # --- Tool validate lifecycle hooks ---

    async def before_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
    ) -> RawToolArgs:
        """Modify raw args before validation."""
        return args

    async def after_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        """Modify validated args. Called only on successful validation."""
        return args

    async def wrap_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
        handler: WrapToolValidateHandler,
    ) -> ValidatedToolArgs:
        """Wraps tool argument validation. handler() runs the validation."""
        return await handler(args)

    async def on_tool_validate_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
        error: ValidationError | ModelRetry,
    ) -> ValidatedToolArgs:
        """Called when tool argument validation fails.

        This is the error counterpart to
        [`after_tool_validate`][pydantic_ai.capabilities.AbstractCapability.after_tool_validate].
        Fires for `ValidationError` (schema mismatch) and
        [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] (custom validator rejection).

        **Raise** the original `error` (or a different exception) to propagate it.
        **Return** validated args to suppress the error and continue as if validation passed.

        Not called for [`SkipToolValidation`][pydantic_ai.exceptions.SkipToolValidation].
        """
        raise error

    # --- Tool execute lifecycle hooks ---

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        """Modify validated args before execution."""
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
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
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        """Wraps tool execution. handler() runs the tool."""
        return await handler(args)

    async def on_tool_execute_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        error: Exception,
    ) -> Any:
        """Called when tool execution fails with an exception.

        This is the error counterpart to
        [`after_tool_execute`][pydantic_ai.capabilities.AbstractCapability.after_tool_execute].

        **Raise** the original `error` (or a different exception) to propagate it.
        **Return** any value to suppress the error and use it as the tool result.

        Not called for control flow exceptions
        ([`SkipToolExecution`][pydantic_ai.exceptions.SkipToolExecution],
        [`CallDeferred`][pydantic_ai.exceptions.CallDeferred],
        [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired])
        or retry signals ([`ToolRetryError`][pydantic_ai.exceptions.ToolRetryError]
        from [`ModelRetry`][pydantic_ai.exceptions.ModelRetry]).
        Use [`wrap_tool_execute`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_execute]
        to intercept retries.
        """
        raise error

    # --- Convenience methods ---

    def prefix_tools(self, prefix: str) -> PrefixTools[AgentDepsT]:
        """Returns a new capability that wraps this one and prefixes its tool names.

        Only this capability's tools are prefixed; other agent tools are unaffected.
        """
        from .prefix_tools import PrefixTools

        return PrefixTools(wrapped=self, prefix=prefix)
