from __future__ import annotations as _annotations

import asyncio
import dataclasses
from collections.abc import Awaitable, Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Generic, Literal, cast

from opentelemetry.trace import Tracer
from typing_extensions import TypeVar

from pydantic_graph import GraphRunContext

from .. import _enqueue, _output, exceptions, messages as _messages, models, usage as _usage
from .._cancel import RunCancellation
from .._deferred_capabilities import parse_loaded_capabilities, registered_loaded_capability_ids
from .._instrumentation import DEFAULT_INSTRUMENTATION_VERSION
from .._run_context import EventStreamBuffer
from .._uuid import uuid7
from ..capabilities.abstract import AbstractCapability, ModelSelector
from ..output import OutputDataT
from ..settings import ModelSettings
from ..tool_manager import ToolManager
from ..tools import AgentNativeTool, RunContext, ToolDefinition
from ..toolsets._tool_search import parse_discovered_tools

if TYPE_CHECKING:
    from ..agent import Agent
    from ..models.instrumented import InstrumentationSettings


DepsT = TypeVar('DepsT')

__all__ = (
    'AgentGraphSleepFunc',
    'EndStrategy',
    'GraphAgentDeps',
    'GraphAgentState',
    'NEW_CONVERSATION',
    '_agent_graph_sleep',
    '_build_output_run_context',
    '_refresh_discovered_tool_names',
    '_refresh_loaded_capability_ids',
    '_revealed_tool_names',
    '_select_model',
    '_with_outgoing_reveal_state',
    'build_run_context',
    'build_validation_context',
    'resolve_conversation_id',
    'resolve_run_id',
    'run_cancelled_snapshot',
    'set_agent_graph_sleep',
)


EndStrategy = Literal['early', 'graceful', 'exhaustive']
"""How to handle function tool calls a model requests alongside a result that ends the run.

The final result usually comes from an output tool call, but with
[`NativeOutput`][pydantic_ai.output.NativeOutput], [`PromptedOutput`][pydantic_ai.output.PromptedOutput],
or image output it comes from the text or image the model returns in the same response.

- `'early'`: Output tools run in the order the model emitted them and the run ends at the first one
  that succeeds; function tools are not executed. If every output tool fails, function tools run so
  the model can correct on the next round. Likewise, if the response contains a valid structured
  output (`NativeOutput`/`PromptedOutput` text, or an image) alongside function tool calls, that output
  ends the run and the function tools are skipped. Plain, unstructured text output (`str` or
  `TextOutput`) does *not* skip tools this way — the model isn't told its text is final, so its
  preamble shouldn't silently cancel a tool call; the function tools run and the run continues.
- `'graceful'` (default): Tools run in the order the model emitted them — function tools that precede
  an output tool complete before it. Output tools run in order and the first success wins; subsequent
  output tools are skipped (their side effects don't run). If a function tool raises
  [`ModelRetry`][pydantic_ai.exceptions.ModelRetry], the output result is suppressed and the retry is
  surfaced to the model instead.
- `'exhaustive'`: Every tool runs (in parallel by default); the first valid output by emission order
  becomes the final result. As with `'graceful'`, a function tool's
  [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] suppresses the output result. Use `sequential=True`
  on a tool (including via [`ToolOutput`][pydantic_ai.output.ToolOutput]) to make it a barrier that
  doesn't overlap with others.

Under `'graceful'` and `'exhaustive'`, a structured output (`NativeOutput`/`PromptedOutput` text, or an
image) returned alongside function tool calls does *not* end the run early: the function tools run and
the run continues, so their results can inform the model's eventual output. Only `'early'` skips them.

The default changed from `'early'` to `'graceful'` in v2. Set `end_strategy='early'` to keep the v1
behavior where the run ends the instant an output tool succeeds.
"""


AgentGraphSleepFunc = Callable[[float], Awaitable[None]]
"""Type for async sleep functions used by the agent graph."""

_AGENT_GRAPH_SLEEP: ContextVar[AgentGraphSleepFunc | None] = ContextVar(
    'pydantic_ai.agent_graph_sleep',
    default=None,
)


@contextmanager
def set_agent_graph_sleep(sleep_func: AgentGraphSleepFunc) -> Generator[None]:
    """Set a custom async sleep function for agent graph delays.

    By default, the agent graph uses `asyncio.sleep` when it needs to wait during
    a run. Durable execution frameworks (Temporal, Prefect, DBOS, Restate, etc.)
    should use this context manager to register their own durable sleep so that
    delays survive workflow replays and don't waste activity time.

    Example:
    ```python
    from pydantic_ai import Agent


    async def durable_sleep(seconds: float) -> None:
        ...  # e.g. `await workflow.sleep(seconds)` under Temporal

    with Agent.using_sleep(durable_sleep):
        ...
    ```
    """
    token = _AGENT_GRAPH_SLEEP.set(sleep_func)
    try:
        yield
    finally:
        _AGENT_GRAPH_SLEEP.reset(token)


async def _agent_graph_sleep(delay: float) -> None:
    """Sleep using the registered agent graph sleep function, or asyncio.sleep."""
    sleep_func = _AGENT_GRAPH_SLEEP.get()
    if sleep_func is not None:
        await sleep_func(delay)
    else:
        await asyncio.sleep(delay)


NEW_CONVERSATION: Literal['new'] = 'new'
"""Sentinel value for `conversation_id` that forces a fresh conversation, ignoring any
`conversation_id` present in `message_history`. See `resolve_conversation_id`."""


def resolve_conversation_id(
    explicit: str | None,
    message_history: Sequence[_messages.ModelMessage] | None,
) -> str:
    """Resolve the `conversation_id` to use for an agent run.

    Priority:

    1. `explicit == 'new'` → fresh UUID7 (forks a conversation off the supplied history).
    2. Explicit string → used as-is.
    3. Most recent non-`None` `conversation_id` on `message_history` (scanned from the end).
    4. Fresh UUID7.

    A fresh UUID7 is intentionally distinct from the run's `run_id`, so callers can
    treat the two identifiers as independent.
    """
    if explicit == NEW_CONVERSATION:
        return str(uuid7())
    if explicit is not None:
        return explicit
    if message_history:
        for message in reversed(message_history):
            if (cid := message.conversation_id) is not None:
                return cid
    return str(uuid7())


def resolve_run_id(
    explicit: str | None,
    message_history: Sequence[_messages.ModelMessage] | None,
) -> str:
    """Resolve the `run_id` to use for an agent run.

    Unlike `conversation_id`, `run_id` is never inherited from `message_history`.
    Each agent run — including a deferred-tool resume — gets its own id so
    `new_messages()` can key off stamped `run_id` values.

    Priority:

    1. Explicit string → used as-is (raises `UserError` if empty, or if that id already
       appears on `message_history`).
    2. Fresh UUID7.
    """
    if explicit is not None:
        if explicit == '':
            raise exceptions.UserError(
                '`run_id` must be a non-empty string when provided. '
                'Empty `run_id` breaks `new_messages()` boundary detection.'
            )
        from .history import _first_run_id_index

        if message_history and _first_run_id_index(message_history, explicit) < len(message_history):
            raise exceptions.UserError(
                f'`run_id={explicit!r}` already appears in `message_history`. '
                'Each agent run needs a distinct `run_id`; reuse breaks `new_messages()`. '
                'Use `conversation_id` to correlate across turns or deferred-tool resume. '
                'When retrying a failed run with the same `run_id`, rebuild `message_history` '
                "without the failed attempt's messages."
            )
        return explicit
    return str(uuid7())


@dataclasses.dataclass(kw_only=True)
class GraphAgentState:
    """State kept across the execution of the agent graph."""

    message_history: list[_messages.ModelMessage] = dataclasses.field(default_factory=list[_messages.ModelMessage])
    usage: _usage.RunUsage = dataclasses.field(default_factory=_usage.RunUsage)
    output_retries_used: int = 0
    run_step: int = 0
    run_id: str = dataclasses.field(default_factory=lambda: str(uuid7()))
    """The unique identifier of this agent run.

    Resolved from the `run_id` argument to `Agent.run` (etc.), or a freshly generated
    UUID7. Unlike `conversation_id`, this is never inherited from `message_history`.
    """
    conversation_id: str = dataclasses.field(default_factory=lambda: str(uuid7()))
    """The unique identifier of the conversation this run belongs to.

    Resolved from the `conversation_id` argument to `Agent.run` (etc.), the most recent
    `conversation_id` on `message_history`, or a freshly generated UUID7. See the
    `Agent.iter` docstring for the resolution priority.
    """
    metadata: dict[str, Any] | None = None
    last_max_tokens: int | None = None
    """Last-resolved `max_tokens` from model settings, used only in error messages."""
    last_model_request_parameters: models.ModelRequestParameters | None = None
    """Last-resolved model request parameters, used for OTel span attributes."""
    pending_messages: list[_enqueue.PendingMessage] = dataclasses.field(default_factory=list[_enqueue.PendingMessage])
    """Internal: queue used by [`PendingMessageDrainCapability`][pydantic_ai.capabilities._pending_messages.PendingMessageDrainCapability]
    for messages enqueued via [`enqueue`][pydantic_ai.tools.RunContext.enqueue] or [`AgentRun.enqueue`][pydantic_ai.run.AgentRun.enqueue]."""
    event_stream_buffer: list[_messages.AgentStreamEvent] = dataclasses.field(default_factory=EventStreamBuffer)
    """Internal: run event buffer, shared by reference into every `RunContext` this run (see `build_run_context`)
    as the private `_event_stream_buffer` field. Framework code appends events to it (e.g.
    [`EnqueuedMessagesEvent`][pydantic_ai.messages.EnqueuedMessagesEvent]s from
    [`PendingMessageDrainCapability`][pydantic_ai.capabilities._pending_messages.PendingMessageDrainCapability]);
    the graph drains it into the agent event stream around node events."""
    mcp_tool_defs_cache: dict[str, dict[str, ToolDefinition]] = dataclasses.field(
        default_factory=dict[str, dict[str, ToolDefinition]]
    )
    """Per-run cache of durable-execution MCP toolset tool definitions, keyed by toolset `id`.

    Shared by reference into every `RunContext` this run (see `build_run_context`), where it is
    exposed as the private `_mcp_tool_defs_cache` field. Recreated per run and reconstructed
    identically on durable replay/recovery, which is what keeps the Temporal/DBOS MCP wrappers'
    `get_tools` scheduling replay-deterministic."""

    def check_incomplete_tool_call(self) -> None:
        """Raise `IncompleteToolCall` if the last model response was truncated mid-tool-call."""
        if (
            self.message_history
            and isinstance(model_response := self.message_history[-1], _messages.ModelResponse)
            and model_response.finish_reason == 'length'
            and model_response.parts
            and isinstance(tool_call := model_response.parts[-1], _messages.ToolCallPart)
        ):
            try:
                tool_call.args_as_dict(raise_if_invalid=True)
            except Exception:
                raise exceptions.IncompleteToolCall(
                    f'Model token limit ({self.last_max_tokens or "provider default"}) exceeded while generating a tool call, resulting in incomplete arguments. Increase the `max_tokens` model setting, or simplify the prompt to result in a shorter response that will fit within the limit.'
                )

    def consume_output_retry(
        self,
        max_output_retries: int,
        error: BaseException | None = None,
    ) -> None:
        """Record one unit of output-retry budget consumption.

        Raises `UnexpectedModelBehavior` when `output_retries_used` would exceed
        `max_output_retries`. Called for `ModelRetry`s from output validators (text path)
        and for `ToolRetryError`s from output-tool dispatch / empty-or-non-actionable
        responses; per-tool retry limits are still enforced separately by
        `ToolManager._check_max_retries`.
        """
        self.output_retries_used += 1
        if self.output_retries_used > max_output_retries:
            self.check_incomplete_tool_call()
            message = f'Exceeded maximum output retries ({max_output_retries})'
            raise exceptions.UnexpectedModelBehavior(message) from error


@dataclasses.dataclass(kw_only=True)
class GraphAgentDeps(Generic[DepsT, OutputDataT]):
    """Dependencies/config passed to the agent graph."""

    user_deps: DepsT

    prompt: str | Sequence[_messages.UserContent] | None
    new_message_index: int
    resumed_request: _messages.ModelRequest | None
    resumed_request_index: int | None

    model: models.Model
    model_selector: ModelSelector[DepsT] | None
    model_selected_for_step: int | None
    evaluate_model_selector: Callable[
        [ModelSelector[DepsT], models.ModelSelectionContext[DepsT]], Awaitable[tuple[models.Model, str | None]]
    ]
    enter_model: Callable[[models.Model], Awaitable[None]]
    get_model_settings: Callable[[RunContext[DepsT]], ModelSettings | None]
    usage_limits: _usage.UsageLimits
    max_output_retries: int
    end_strategy: EndStrategy
    get_instructions: Callable[[RunContext[DepsT]], Awaitable[list[_messages.InstructionPart] | None]]

    output_schema: _output.OutputSchema[OutputDataT]
    output_validators: list[_output.OutputValidator[DepsT, OutputDataT]]
    validation_context: Any | Callable[[RunContext[DepsT]], Any]

    root_capability: AbstractCapability[DepsT]

    capabilities: dict[str, AbstractCapability[DepsT]]

    # Invariant: these two sets are shared by reference into every `RunContext` this run (their
    # identity survives `replace(ctx, ...)`, which shallow-copies) and are only ever mutated in
    # place — never reassigned. The per-step refresh relies on that shared identity for both, and
    # `discovered_tool_names` additionally on the in-step reveals written by tool execution.
    # `loaded_capability_ids` is refreshed from history only: a capability loaded during a step
    # lands from the next one, so nothing writes it mid-step. Reassigning either (here, or by
    # passing it to a `replace(ctx, ...=...)`) would silently break in-step tool reveals.
    loaded_capability_ids: set[str]
    discovered_tool_names: set[str]

    native_tools: list[AgentNativeTool[DepsT]] = dataclasses.field(repr=False)
    tool_manager: ToolManager[DepsT]

    tracer: Tracer
    instrumentation_settings: InstrumentationSettings | None

    agent: Agent[DepsT, Any] | None = None

    cancellation: RunCancellation = dataclasses.field(default_factory=RunCancellation, repr=False)
    """The run's first-party cancellation controller. Runtime-only: holds a live task reference."""

    pending_immediate_dispatches: dict[int, list[asyncio.Event]] = dataclasses.field(
        default_factory=dict[int, list[asyncio.Event]], repr=False
    )
    """Settlement signals for buffered events dispatched immediately, keyed by `id(event)`.

    Runtime-only, deliberately not on `GraphAgentState`: raw object ids are meaningless in a revived
    process (a stale persisted id could even collide with a new event's address), so a revived run
    starts empty and buffered events degrade to dispatching at stream position."""

    event_stream_replacements: dict[int, _messages.AgentStreamEvent] = dataclasses.field(
        default_factory=dict[int, _messages.AgentStreamEvent], repr=False
    )
    """Legacy `hooks.on.event` replacements to apply at the consumer-facing stream position.

    Runtime-only and id-keyed like `pending_immediate_dispatches`, and excluded from persistence for
    the same reason."""

    model_id: str | None = None
    """The model-id string `model` was resolved from, if the run's model came from a string.

    Stamped onto `ModelRequestContext.model_id` so durable-execution capabilities can
    round-trip the original selection token across the activity/step/task boundary.
    """


async def _select_model(ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, Any]]) -> None:
    selector = ctx.deps.model_selector
    if selector is None or ctx.deps.model_selected_for_step == ctx.state.run_step:
        return

    agent = ctx.deps.agent
    assert agent is not None
    selection_ctx = models.ModelSelectionContext(
        agent=agent,
        deps=ctx.deps.user_deps,
        model=ctx.deps.model,
        run_step=ctx.state.run_step,
        # The current request has already been appended, but selection describes the model
        # that will handle it. Expose the history available before this request step, matching
        # bootstrap selection, and do not let selectors mutate graph state through the context.
        messages=list(ctx.state.message_history[:-1]),
        usage=ctx.state.usage,
    )
    model, model_id = await ctx.deps.evaluate_model_selector(selector, selection_ctx)
    await ctx.deps.enter_model(model)
    ctx.deps.model = model
    ctx.deps.model_id = model_id
    ctx.deps.model_selected_for_step = ctx.state.run_step


def build_run_context(ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, Any]]) -> RunContext[DepsT]:
    """Build a `RunContext` object from the current agent graph run context."""
    run_context = RunContext[DepsT](
        deps=ctx.deps.user_deps,
        agent=ctx.deps.agent,
        model=ctx.deps.model,
        _model_id=ctx.deps.model_id,
        usage=ctx.state.usage,
        usage_limits=ctx.deps.usage_limits,
        prompt=ctx.deps.prompt,
        messages=ctx.state.message_history,
        validation_context=None,
        tracer=ctx.deps.tracer,
        trace_include_content=ctx.deps.instrumentation_settings is not None
        and ctx.deps.instrumentation_settings.include_content,
        instrumentation_version=ctx.deps.instrumentation_settings.version
        if ctx.deps.instrumentation_settings
        else DEFAULT_INSTRUMENTATION_VERSION,
        run_step=ctx.state.run_step,
        run_id=ctx.state.run_id,
        conversation_id=ctx.state.conversation_id,
        metadata=ctx.state.metadata,
        tool_manager=ctx.deps.tool_manager,
        root_capability=ctx.deps.root_capability,
        capabilities=ctx.deps.capabilities,
        loaded_capability_ids=ctx.deps.loaded_capability_ids,
        discovered_tool_names=ctx.deps.discovered_tool_names,
        pending_messages=ctx.state.pending_messages,
        _cancellation=ctx.deps.cancellation,
        _event_stream_buffer=ctx.state.event_stream_buffer,
        _pending_immediate_dispatches=ctx.deps.pending_immediate_dispatches,
        _event_stream_replacements=ctx.deps.event_stream_replacements,
        _mcp_tool_defs_cache=ctx.state.mcp_tool_defs_cache,
    )
    validation_context = build_validation_context(ctx.deps.validation_context, run_context)
    # Only `validation_context` may be passed to `replace`: it shallow-copies, preserving the shared
    # identity of the mutable members passed by reference above — `loaded_capability_ids`,
    # `discovered_tool_names`, `pending_messages`, `_cancellation`, `_event_stream_buffer`,
    # `_mcp_tool_defs_cache` (see the invariant on `GraphAgentDeps.loaded_capability_ids`). Never
    # add any of them as a `replace` kwarg — forking the object would silently break in-step
    # capability loads / tool reveals / message enqueues / cancellation / event delivery /
    # tool-defs caching.
    run_context = replace(run_context, validation_context=validation_context)
    return run_context


def run_cancelled_snapshot(
    message: str, state: GraphAgentState, deps: GraphAgentDeps[Any, Any]
) -> exceptions.RunCancelled:
    """Build a `RunCancelled` carrying a detached snapshot of the run's current state."""
    return exceptions.RunCancelled(
        message,
        messages=state.message_history,
        new_message_index=deps.new_message_index,
        usage=state.usage,
        metadata=state.metadata,
        run_id=state.run_id,
        conversation_id=state.conversation_id,
    )


def _refresh_loaded_capability_ids(ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, Any]]) -> None:
    """Refresh the history-derived loaded capability ids from the current graph state."""
    # The `load_capability` tool (and therefore any `LoadCapability*` history parts) only exists
    # when a deferred capability is configured — the same condition that injects the loader. Without
    # one, the set can never change during the run, so the seeded value stays in sync without rescanning.
    # (`discovered_tool_names` has no equally-cheap guard: tool search is auto-injected and its trigger
    # is "deferred tools exist", which isn't known without resolving toolsets, so its refresh stays
    # unconditional.)
    if not any(capability.defer_loading is True for capability in ctx.deps.capabilities.values()):
        return

    loaded_capability_ids = registered_loaded_capability_ids(ctx.state.message_history, ctx.deps.capabilities.keys())

    # Mutate in place (not reassign): this set is shared by reference with the run's `RunContext`
    # copies made via `replace(ctx, ...)`, so clear + update keeps them all in sync.
    ctx.deps.loaded_capability_ids.clear()
    ctx.deps.loaded_capability_ids.update(loaded_capability_ids)


def _refresh_discovered_tool_names(ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, Any]]) -> None:
    """Refresh the history-derived discovered tool names from the current graph state."""
    discovered_tool_names = parse_discovered_tools(ctx.state.message_history)

    # Mutate in place (not reassign), for the same shared-by-reference reason as the set above.
    ctx.deps.discovered_tool_names.clear()
    ctx.deps.discovered_tool_names.update(discovered_tool_names)


def _revealed_tool_names(
    discovered: Iterable[str],
    function_tools: Iterable[ToolDefinition],
    *,
    deferred_capability_ids: set[str],
    loaded_capability_ids: set[str],
) -> set[str]:
    """Drop reveals for tools this run doesn't define, and those whose owning capability isn't active yet.

    History outlives configuration, so it can name a tool the current run has no definition for. Such
    a name can't be revealed — there is no schema to show — and every consumer already guards on
    membership in the definitions, so dropping it here changes nothing observable; what it buys is
    that `revealed_tool_names` is a subset of `function_tools`' names by construction, and a future
    consumer can't be caught out by an entry that resolves to nothing.

    The ordering a run holds to is load, then reveal, then call: a capability's instructions and
    hooks come as a bundle, and its tools should not reach the model ahead of the runbook for using
    them. A reveal says a schema *may* be shown; it cannot stand in for the load.

    Not a trust boundary, and not trying to be one. Any history the model could plausibly have
    produced is honoured — fabricating a coherent `load_capability` exchange is equivalent to the
    model having called it, and history integrity is the deployment's job. What is rejected is a
    history no legitimate run could have produced: a capability tool revealed with no load behind it
    describes a world that never existed, and honouring it would put the run in a state its own
    rules forbid — including advertising a tool `ToolManager` will refuse to run.

    Only *deferred* capabilities gate their tools this way. An always-on capability's search-gated
    tool is revealed by discovery alone, which is why this needs `deferred_capability_ids` read from
    the capability instances rather than a guess from the tool definitions.
    """
    owner_by_name = {tool_def.name: tool_def.capability_id for tool_def in function_tools}
    # The complement of `RunContext.active_capability_ids` over the run's capabilities: active
    # is "not deferred, or loaded", so inactive is "deferred and not loaded". Spelled from the
    # two history-derived sets because this also runs against a bare message list, with no
    # `RunContext` to ask — but it must keep answering exactly what `is_tool_available` answers.
    inactive_capability_ids = deferred_capability_ids - loaded_capability_ids
    return {name for name in discovered if name in owner_by_name and owner_by_name[name] not in inactive_capability_ids}


def _with_outgoing_reveal_state(
    parameters: models.ModelRequestParameters, messages: list[_messages.ModelMessage]
) -> models.ModelRequestParameters:
    """Make per-request reveal state match the history that will be sent to the model.

    Gated on the same availability rule as the run-level state: a reveal naming a tool whose
    deferred capability this history does not show as loaded is dropped, so the model is never
    offered a tool it has not been properly given — and never one `ToolManager` would refuse to
    run. An always-on capability's search-gated tools are unaffected: they carry no load marker by
    design, and `deferred_capability_ids` is read from the capability instances, so they are not
    in it.
    """
    return replace(
        parameters,
        revealed_tool_names=_revealed_tool_names(
            parse_discovered_tools(messages),
            parameters.function_tools,
            deferred_capability_ids=parameters.deferred_capability_ids,
            loaded_capability_ids=parse_loaded_capabilities(messages),
        ),
    )


def build_validation_context(
    validation_ctx: Any | Callable[[RunContext[DepsT]], Any],
    run_context: RunContext[DepsT],
) -> Any:
    """Build a Pydantic validation context, potentially from the current agent run context."""
    if callable(validation_ctx):
        fn = cast(Callable[[RunContext[DepsT]], Any], validation_ctx)
        return fn(run_context)
    else:
        return validation_ctx


def _build_output_run_context(
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, Any]],
) -> RunContext[DepsT]:
    """Build a RunContext with global output retry info for output validation.

    Starts from `tool_manager.ctx` (when available) so per-tool retry counts
    (`ctx.retries[name]`) populated by `for_run_step` propagate to output hooks
    like `prepare_output_tools` and output validators. Then overrides `retry`
    and `max_retries` with the **output** budget (`max_output_retries`),
    distinct from the tool budget on `tool_manager.ctx`.
    """
    base = ctx.deps.tool_manager.ctx if ctx.deps.tool_manager.ctx is not None else build_run_context(ctx)
    return replace(
        base,
        retry=ctx.state.output_retries_used,
        max_retries=ctx.deps.max_output_retries,
    )
