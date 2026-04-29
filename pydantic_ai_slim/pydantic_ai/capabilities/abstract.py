from __future__ import annotations

from abc import ABC
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias

from pydantic import ValidationError

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ToolCallPart
from pydantic_ai.tools import (
    AgentBuiltinTool,
    AgentDepsT,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDefinition,
)
from pydantic_ai.toolsets import AbstractToolset, AgentToolset

if TYPE_CHECKING:
    from pydantic_ai import _agent_graph
    from pydantic_ai.agent.abstract import AgentModelSettings
    from pydantic_ai.capabilities.prefix_tools import PrefixTools
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.output import OutputContext
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

RawToolArgs: TypeAlias = str | dict[str, Any]
"""Type alias for raw (pre-validation) tool arguments."""

ValidatedToolArgs: TypeAlias = dict[str, Any]
"""Type alias for validated tool arguments."""

WrapToolValidateHandler: TypeAlias = Callable[[RawToolArgs], Awaitable[ValidatedToolArgs]]
"""Handler type for [`wrap_tool_validate`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_validate]."""

WrapToolExecuteHandler: TypeAlias = Callable[[ValidatedToolArgs], Awaitable[Any]]
"""Handler type for [`wrap_tool_execute`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_execute]."""

RawOutput: TypeAlias = str | dict[str, Any]
"""Type alias for raw output data (text or tool args)."""

WrapOutputValidateHandler: TypeAlias = Callable[[RawOutput], Awaitable[Any]]
"""Handler type for wrap_output_validate."""

WrapOutputProcessHandler: TypeAlias = Callable[[Any], Awaitable[Any]]
"""Handler type for wrap_output_process."""


CapabilityPosition = Literal['outermost', 'innermost']
"""Position tier for a capability in the middleware chain.

- `'outermost'`: in the outermost tier, before all non-outermost capabilities.
  Multiple capabilities can declare `'outermost'`; original list order breaks ties
  within the tier, and `wraps`/`wrapped_by` edges refine order further.
- `'innermost'`: in the innermost tier, after all non-innermost capabilities.
  Same tie-breaking rules apply.
"""

CapabilityRef: TypeAlias = 'type[AbstractCapability[Any]] | AbstractCapability[Any]'
"""Reference to a capability — either a type (matches all instances of that type) or a specific instance (matches by identity)."""


@dataclass
class CapabilityOrdering:
    """Ordering constraints for a capability within a combined capability chain.

    Capabilities follow middleware semantics: the first capability in the list is the
    **outermost** layer, wrapping all others. Declare ordering constraints via
    [`get_ordering`][pydantic_ai.capabilities.AbstractCapability.get_ordering]
    to control a capability's position in the chain regardless of how the user lists them.

    When a [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability] is
    constructed, it topologically sorts its children to satisfy these constraints,
    preserving user-provided order as a tiebreaker.
    """

    position: CapabilityPosition | None = None
    """Fixed position in the chain, or `None` for user-provided order."""

    wraps: Sequence[CapabilityRef] = ()
    """This capability wraps around (is outside of) these capabilities in the middleware chain.

    Each entry can be a capability **type** (matches all instances of that type via `issubclass`)
    or a specific capability **instance** (matches by identity via `is`).

    Note: instance refs use identity (`is`) matching, so if a capability's
    [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run] returns a
    new instance, refs to the original will no longer match. Use type refs
    when the target capability uses per-run state isolation.
    """

    wrapped_by: Sequence[CapabilityRef] = ()
    """This capability is wrapped by (is inside of) these capabilities in the middleware chain.

    Each entry can be a capability **type** (matches all instances of that type via `issubclass`)
    or a specific capability **instance** (matches by identity via `is`).

    Note: instance refs use identity (`is`) matching, so if a capability's
    [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run] returns a
    new instance, refs to the original will no longer match. Use type refs
    when the target capability uses per-run state isolation.
    """

    requires: Sequence[type[AbstractCapability[Any]]] = ()
    """These types must be present in the chain (no ordering implied)."""


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
    YAML/JSON specs (via [`Agent.from_spec`][pydantic_ai.Agent.from_spec]); they have
    sensible defaults and typically don't need to be overridden.
    """

    def apply(self, visitor: Callable[[AbstractCapability[AgentDepsT]], None]) -> None:
        """Run a visitor function on all leaf capabilities in this tree.

        For a single capability, calls the visitor on itself.
        Overridden by [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability]
        to recursively visit all child capabilities, and by
        [`WrapperCapability`][pydantic_ai.capabilities.WrapperCapability]
        to delegate to the wrapped capability.
        """
        visitor(self)

    @property
    def has_wrap_node_run(self) -> bool:
        """Whether this capability (or any sub-capability) overrides wrap_node_run."""
        return type(self).wrap_node_run is not AbstractCapability.wrap_node_run

    @property
    def has_wrap_run_event_stream(self) -> bool:
        """Whether this capability (or any sub-capability) overrides wrap_run_event_stream."""
        return type(self).wrap_run_event_stream is not AbstractCapability.wrap_run_event_stream

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

    def get_ordering(self) -> CapabilityOrdering | None:
        """Return ordering constraints for this capability, or `None` for default behavior.

        Override to declare a fixed position (`'outermost'` / `'innermost'`),
        relative ordering (`wraps` / `wrapped_by` other capability types or instances),
        or dependency requirements (`requires`).

        [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability] uses
        these to topologically sort its children at construction time.
        """
        return None

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
        [`TemplateStr`][pydantic_ai.TemplateStr] — not a dynamic string.
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

        Called per-run with the combined non-output toolset (after the
        [`prepare_tools`][pydantic_ai.capabilities.AbstractCapability.prepare_tools] hook
        has already wrapped it). Output tools are added separately and are not included.

        Unlike the other `get_*` methods which are called once at agent construction,
        this is called each run (after [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run]).
        When multiple capabilities provide wrappers, they follow middleware semantics:
        the first capability in the list wraps outermost (matching `wrap_*` hooks).

        Use this to apply cross-cutting toolset wrappers like
        [`PreparedToolset`][pydantic_ai.toolsets.PreparedToolset],
        [`FilteredToolset`][pydantic_ai.toolsets.FilteredToolset],
        or custom [`WrapperToolset`][pydantic_ai.toolsets.WrapperToolset] subclasses.
        """
        return None

    # --- Tool preparation hooks ---

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Filter or modify function tool definitions for this step.

        Receives **function** tools only. For [output tools][pydantic_ai.output.ToolOutput],
        override
        [`prepare_output_tools`][pydantic_ai.capabilities.AbstractCapability.prepare_output_tools]
        — it runs separately, with `ctx.retry`/`ctx.max_retries` reflecting the **output**
        retry budget instead of the function-tool budget.

        Return a filtered or modified list. The result flows into both the model's request
        parameters and `ToolManager.tools`, so filtering also blocks tool execution.
        """
        return tool_defs

    async def prepare_output_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Filter or modify output tool definitions for this step.

        Receives only [output tools][pydantic_ai.output.ToolOutput]. `ctx.retry` and
        `ctx.max_retries` reflect the **output** retry budget (agent-level
        `max_result_retries`), matching the output hook lifecycle.

        Return a filtered or modified list. The result flows into both the model's request
        parameters and `ToolManager.tools`, so filtering also blocks tool execution.
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
        `iter()` loop), this method receives an `asyncio.CancelledError`.
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

        Note: this hook fires when using [`agent.run()`][pydantic_ai.Agent.run],
        [`agent.run_stream()`][pydantic_ai.Agent.run_stream], and when manually driving
        an [`agent.iter()`][pydantic_ai.Agent.iter] run with
        [`next()`][pydantic_ai.result.AgentRun.next], but it does **not** fire when
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
        """Wraps the event stream for a streamed node. Can observe or transform events.

        Note: when this method is overridden (or [`Hooks.on.event`][pydantic_ai.capabilities.hooks.Hooks.on]
        / [`Hooks.on.run_event_stream`][pydantic_ai.capabilities.hooks.Hooks.on] are registered),
        [`agent.run()`][pydantic_ai.Agent.run] automatically enables streaming mode so this hook
        fires even without an explicit `event_stream_handler`.
        """
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
        """Called after each model response. Can modify the response before further processing.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to reject the response and
        ask the model to try again. The original response is still appended to message history
        so the model can see what it said. Retries count against `max_result_retries`.
        """
        return response

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Wraps the model request. handler() calls the model.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to skip `on_model_request_error`
        and directly retry the model request with a retry prompt. If the handler was called,
        the model response is preserved in history for context (same as `after_model_request`).
        """
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
        **Raise** [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to retry the model request
        with a retry prompt instead of recovering or propagating.

        Not called for [`SkipModelRequest`][pydantic_ai.exceptions.SkipModelRequest]
        or [`ModelRetry`][pydantic_ai.exceptions.ModelRetry].
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
        """Modify raw args before validation.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to skip validation and
        ask the model to redo the tool call.
        """
        return args

    async def after_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        """Modify validated args. Called only on successful validation.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to reject the validated args
        and ask the model to redo the tool call.
        """
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
        Fires for [`ValidationError`][pydantic.ValidationError] (schema mismatch) and
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
        """Modify validated args before execution.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to skip execution and
        ask the model to redo the tool call.
        """
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
        """Modify result after execution.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to reject the tool result
        and ask the model to redo the tool call.
        """
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
        **Raise** [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to ask the model to
        redo the tool call instead of recovering or propagating.

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

    # --- Output validate lifecycle hooks ---

    async def before_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
    ) -> RawOutput:
        """Modify raw model output before validation/parsing.

        The primary hook for pre-parse repair and normalization of model output.
        Fires only for structured output that requires parsing: prompted, native,
        tool, and union output. Does **not** fire for plain text or image output.

        For structured text output, `output` is the raw text string from the model.
        For tool output, `output` is the raw tool arguments (string or dict).

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to skip validation and
        ask the model to try again with a custom message.

        During streaming, this hook fires on every partial validation attempt as well as
        the final result. Check `ctx.partial_output` to distinguish and avoid expensive
        work on partial results.
        """
        return output

    async def after_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        """Modify validated output after successful parsing. Called only on success.

        `output` is the **semantic value** the model was asked to produce — e.g., a
        `MyModel` instance for `output_type=MyModel`, or `42` for `output_type=int`, or
        the input to a single-arg output function. For multi-arg output functions, this
        is the `dict` of arguments (the genuine multi-value input).

        Note: this differs from *tool* hooks (`after_tool_validate`), which always see
        `dict[str, Any]` — tool args follow the schema contract. Output hooks see the
        semantic output value, regardless of how it's internally represented during
        validation.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to reject the validated
        output and ask the model to try again.
        """
        return output

    async def wrap_output_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
        handler: WrapOutputValidateHandler,
    ) -> Any:
        """Wraps output validation. handler(output) performs the validation.

        [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] from within the handler goes to
        [`on_output_validate_error`][pydantic_ai.capabilities.AbstractCapability.on_output_validate_error].
        `ModelRetry` raised directly (not from the handler) bypasses the error hook.
        """
        return await handler(output)

    async def on_output_validate_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: RawOutput,
        error: ValidationError | ModelRetry,
    ) -> Any:
        """Called when output validation fails.

        This is the error counterpart to
        [`after_output_validate`][pydantic_ai.capabilities.AbstractCapability.after_output_validate].

        **Raise** the original `error` (or a different exception) to propagate it.
        **Return** validated output to suppress the error and continue.
        """
        raise error

    # --- Output process lifecycle hooks ---

    async def before_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        """Modify validated output before processing (extraction, output function call).

        `output` is the **semantic value** — e.g., a `MyModel` instance or `42`, matching
        `after_output_validate`. For multi-arg output functions, it's the `dict` of args.
        See [`after_output_validate`][pydantic_ai.capabilities.AbstractCapability.after_output_validate]
        for a full explanation of the semantic-value contract.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to skip processing and
        ask the model to try again.
        """
        return output

    async def after_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
    ) -> Any:
        """Modify result after output processing.

        Raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to reject the result
        and ask the model to try again.
        """
        return output

    async def wrap_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
        handler: WrapOutputProcessHandler,
    ) -> Any:
        """Wraps output processing. handler(output) runs extraction + output function call.

        [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] bypasses
        [`on_output_process_error`][pydantic_ai.capabilities.AbstractCapability.on_output_process_error]
        (treated as control flow, not an error).

        During streaming, this fires only when partial validation succeeds, and on the
        final result. Check `ctx.partial_output` to skip expensive work on partial results.
        """
        return await handler(output)

    async def on_output_process_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: Any,
        error: Exception,
    ) -> Any:
        """Called when output processing fails with an exception.

        This is the error counterpart to
        [`after_output_process`][pydantic_ai.capabilities.AbstractCapability.after_output_process].

        **Raise** the original `error` (or a different exception) to propagate it.
        **Return** any value to suppress the error and use it as the output.

        Not called for retry signals ([`ToolRetryError`][pydantic_ai.exceptions.ToolRetryError]
        from [`ModelRetry`][pydantic_ai.exceptions.ModelRetry]).
        """
        raise error

    # --- Deferred tool call hooks ---

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        """Handle deferred tool calls (approval-required or externally-executed) inline during an agent run.

        Called by [`ToolManager`][pydantic_ai.tool_manager.ToolManager] when:

        - a tool raises [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] or
          [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] during execution, or
        - the model calls a tool registered with `requires_approval=True` (see
          [Human-in-the-Loop Tool Approval](../deferred-tools.md#human-in-the-loop-tool-approval))
          or a tool backed by [external execution](../deferred-tools.md#external-tool-execution).

        Uses accumulation dispatch: each capability in the chain receives remaining
        unresolved requests and can resolve some or all of them. Results are merged
        and unresolved calls are passed to the next capability.

        **Return** a [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] to resolve
        some or all calls.
        **Return** `None` to leave all calls unresolved.
        """
        return None

    # --- Convenience methods ---

    def prefix_tools(self, prefix: str) -> PrefixTools[AgentDepsT]:
        """Returns a new capability that wraps this one and prefixes its tool names.

        Only this capability's tools are prefixed; other agent tools are unaffected.
        """
        from .prefix_tools import PrefixTools

        return PrefixTools(wrapped=self, prefix=prefix)
