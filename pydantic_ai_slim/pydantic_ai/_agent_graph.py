from __future__ import annotations as _annotations

import asyncio
import dataclasses
import inspect
import time
from asyncio import Task
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Generator, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import field, replace
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeGuard, cast

from opentelemetry.trace import Tracer
from typing_extensions import TypeVar, assert_never

from pydantic_ai._history_processor import HistoryProcessor
from pydantic_ai._instrumentation import (
    DEFAULT_INSTRUMENTATION_VERSION,
    capture_current_context,
    get_instructions as _get_history_instructions,
    time_to_first_chunk_ctx,
)
from pydantic_ai._tool_execution import process_tool_calls
from pydantic_ai._utils import cancel_and_drain, dataclasses_no_defaults_repr, fill_run_metadata, now_utc
from pydantic_ai._uuid import uuid7
from pydantic_ai.capabilities.abstract import AbstractCapability, ModelSelector
from pydantic_ai.models import (
    CompletedStreamedResponse,
    ModelRequestContext,
)
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.native_tools._tool_search import ToolSearchTool
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets._tool_search import parse_discovered_tools
from pydantic_graph import BaseNode, End, Graph, GraphBuilder, GraphRunContext
from pydantic_graph.basenode import NodeRunEndT

from . import _enqueue, _output, _system_prompt, exceptions, messages as _messages, models, result, usage as _usage
from ._deferred_capabilities import parse_loaded_capabilities
from ._instructions import normalize_toolset_instructions
from ._run_context import set_current_run_context
from .exceptions import ToolRetryError

# `_ContinuationStreamedResponse` is an intentionally-exported member of the private
# `_continuation` module (see its `__all__`); the leading underscore is a module-privacy
# marker, not a within-package one, so the private-usage check doesn't apply here.
from .models._continuation import (
    MAX_BACKGROUND_POLLS,
    MAX_GENERATION_CONTINUATIONS,
    MergeMode,
    _ContinuationStreamedResponse,
    cancel_suspended_job,
    merge_mode,
    merge_responses,
)
from .output import OutputDataT, OutputSpec
from .settings import ModelSettings
from .tools import (
    AgentNativeTool,
    DeferredToolResult,
    DeferredToolResults,
    RunContext,
    ToolDefinition,
)

if TYPE_CHECKING:
    from .agent import Agent
    from .models.instrumented import InstrumentationSettings

__all__ = (
    'GraphAgentState',
    'GraphAgentDeps',
    'UserPromptNode',
    'ModelRequestNode',
    'CallToolsNode',
    'build_run_context',
    'capture_run_messages',
    'HistoryProcessor',
    'resolve_conversation_id',
    'process_tool_calls',
)


T = TypeVar('T')
S = TypeVar('S')
NoneType = type(None)
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


DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')


async def _with_event_stream_buffer(
    stream: AsyncIterator[_messages.AgentStreamEvent],
    event_stream_buffer: list[_messages.AgentStreamEvent],
) -> AsyncIterator[_messages.AgentStreamEvent]:
    """Drain buffered run events around each event from a node stream, preserving order."""
    while event_stream_buffer:
        yield event_stream_buffer.pop(0)
    async for event in stream:
        while event_stream_buffer:
            yield event_stream_buffer.pop(0)
        yield event
    while event_stream_buffer:
        yield event_stream_buffer.pop(0)


async def _cancel_task(task: Task[Any]) -> None:
    # `cancel()` is a documented no-op on an already-finished task, so there's no need to guard it.
    task.cancel()
    try:
        await task
    except BaseException:
        # Called while another stream error is already propagating; await only
        # to finish cleanup and retrieve the task exception, not replace it.
        pass


async def _resolve_interrupted_stream_state(
    model: models.Model,
    stream_error: BaseException,
    partial: _messages.ModelResponse,
) -> _messages.ModelResponseState:
    """State to record for a streamed turn the consumer stopped, cancelling a leaked job when appropriate.

    The composite treats every `aclose()` (which the handler teardown triggers) as a *detach*, so
    `partial.state` is `'suspended'` whenever the last segment is a still-pending job — regardless of
    *why* the consumer stopped. Only the graph knows why, from `stream_error`'s type:

    - `GeneratorExit` is a walk-away detach (the consumer broke out of `run_stream`/`stream_text`). Mirror
      the non-streaming detach: keep `'suspended'` so the run is resumable, and leave the job alive.
    - any other exception is a genuine downstream failure. Mirror the non-streaming cancel-on-error policy:
      force `'interrupted'` (non-resumable) and best-effort cancel the still-live job so it doesn't leak.
    """
    if isinstance(stream_error, GeneratorExit) and partial.state == 'suspended':
        return 'suspended'
    if partial.state == 'suspended':
        await cancel_suspended_job(model, partial)
    return 'interrupted'


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


@dataclasses.dataclass(kw_only=True)
class GraphAgentState:
    """State kept across the execution of the agent graph."""

    message_history: list[_messages.ModelMessage] = dataclasses.field(default_factory=list[_messages.ModelMessage])
    usage: _usage.RunUsage = dataclasses.field(default_factory=_usage.RunUsage)
    output_retries_used: int = 0
    run_step: int = 0
    run_id: str = dataclasses.field(default_factory=lambda: str(uuid7()))
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
    event_stream_buffer: list[_messages.AgentStreamEvent] = dataclasses.field(
        default_factory=list[_messages.AgentStreamEvent]
    )
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
    # place — never reassigned. The per-step refresh and the `load_capability` tool body rely on
    # that shared identity. Reassigning either (here, or by passing it to a `replace(ctx, ...=...)`)
    # would silently break in-step capability loads / tool reveals.
    loaded_capability_ids: set[str]
    discovered_tool_names: set[str]

    native_tools: list[AgentNativeTool[DepsT]] = dataclasses.field(repr=False)
    tool_manager: ToolManager[DepsT]

    tracer: Tracer
    instrumentation_settings: InstrumentationSettings | None

    agent: Agent[DepsT, Any] | None = None

    model_id: str | None = None
    """The model-id string `model` was resolved from, if the run's model came from a string.

    Stamped onto `ModelRequestContext.model_id` so durable-execution capabilities can
    round-trip the original selection token across the activity/step/task boundary.
    """


class AgentNode(BaseNode[GraphAgentState, GraphAgentDeps[DepsT, Any], result.FinalResult[NodeRunEndT]]):
    """The base class for all agent nodes.

    Using subclass of `BaseNode` for all nodes reduces the amount of boilerplate of generics everywhere
    """


def is_agent_node(
    node: BaseNode[GraphAgentState, GraphAgentDeps[T, Any], result.FinalResult[S]] | End[result.FinalResult[S]],
) -> TypeGuard[AgentNode[T, S]]:
    """Check if the provided node is an instance of `AgentNode`.

    Usage:

        if is_agent_node(node):
            # `node` is an AgentNode
            ...

    This method preserves the generic parameters on the narrowed type, unlike `isinstance(node, AgentNode)`.
    """
    return isinstance(node, AgentNode)


@dataclasses.dataclass
class UserPromptNode(AgentNode[DepsT, NodeRunEndT]):
    """The node that handles the user prompt and instructions."""

    user_prompt: str | Sequence[_messages.UserContent] | None

    _: dataclasses.KW_ONLY

    deferred_tool_results: DeferredToolResults | None = None

    instructions: str | None = None
    instructions_functions: list[_system_prompt.SystemPromptRunner[DepsT]] = dataclasses.field(
        default_factory=list[_system_prompt.SystemPromptRunner[DepsT]]
    )

    system_prompts: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    system_prompt_functions: list[_system_prompt.SystemPromptRunner[DepsT]] = dataclasses.field(
        default_factory=list[_system_prompt.SystemPromptRunner[DepsT]]
    )
    system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[DepsT]] = dataclasses.field(
        default_factory=dict[str, _system_prompt.SystemPromptRunner[DepsT]]
    )

    async def run(  # noqa: C901
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> ModelRequestNode[DepsT, NodeRunEndT] | CallToolsNode[DepsT, NodeRunEndT]:
        try:
            ctx_messages = get_captured_run_messages()
        except LookupError:
            messages: list[_messages.ModelMessage] = []
        else:
            if ctx_messages.used:
                messages = []
            else:
                messages = ctx_messages.messages
                ctx_messages.used = True

        # Replace the `capture_run_messages` list with the message history
        messages[:] = _clean_message_history(ctx.state.message_history)
        # Use the `capture_run_messages` list as the message history so that new messages are added to it
        ctx.state.message_history = messages
        ctx.deps.new_message_index = len(messages)

        if self.deferred_tool_results is not None:
            return await self._handle_deferred_tool_results(self.deferred_tool_results, messages, ctx)

        if (
            messages
            and isinstance(last_message := messages[-1], _messages.ModelRequest)
            and last_message.state == 'interrupted'
        ):
            # A trailing request interrupted during tool execution means the last response's
            # still-unanswered calls will never be executed, so they are closed out with
            # synthesized returns. A 'complete' trailing request (e.g. from a run that ended in
            # `DeferredToolRequests`) is left alone: its response's open calls may still receive
            # `deferred_tool_results`.
            messages[:] = _repair_dangling_tool_calls(messages, repair_last_response=True)

        next_message: _messages.ModelRequest | None = None
        is_resuming_without_prompt = False

        run_context: RunContext[DepsT] | None = None

        if messages and (last_message := messages[-1]):
            if isinstance(last_message, _messages.ModelRequest) and self.user_prompt is None:
                # Drop last message from history and reuse its parts
                messages.pop()
                next_message = _messages.ModelRequest(
                    parts=last_message.parts,
                    run_id=last_message.run_id,
                    conversation_id=last_message.conversation_id,
                    metadata=last_message.metadata,
                )
                is_resuming_without_prompt = True

                # Extract `UserPromptPart` content from the popped message and add to `ctx.deps.prompt`
                user_prompt_parts = [part for part in last_message.parts if isinstance(part, _messages.UserPromptPart)]
                if user_prompt_parts:
                    if len(user_prompt_parts) == 1:
                        ctx.deps.prompt = user_prompt_parts[0].content
                    else:
                        combined_content: list[_messages.UserContent] = []
                        for part in user_prompt_parts:
                            if isinstance(part.content, str):
                                combined_content.append(part.content)
                            else:
                                combined_content.extend(part.content)
                        ctx.deps.prompt = combined_content
            elif isinstance(last_message, _messages.ModelResponse):
                if last_message.state == 'suspended' and self.user_prompt is None:
                    # The history ends in a turn a provider paused mid-flight (Anthropic
                    # `pause_turn`, OpenAI background mode) and persisted. Resume it in
                    # `ModelRequestNode`, which re-issues the suspended turn and stitches the
                    # continuation into a single response with hooks firing once around the chain.
                    # `request` is an empty placeholder to satisfy `ModelRequestNode`'s dataclass:
                    # it is intentionally NOT appended to history (the suspended response is the real
                    # tail that gets echoed back), and nothing is sent for it. `_resume_suspended`
                    # drives `_prepare_resume_request` instead of the normal `_prepare_request` path.
                    return ModelRequestNode[DepsT, NodeRunEndT](
                        request=_messages.ModelRequest(parts=[]), _resume_suspended=last_message
                    )
                if self.user_prompt is None:
                    # Align with the upcoming request step so we don't resolve dynamic toolsets twice.
                    run_context = replace(
                        build_run_context(ctx),
                        run_step=ctx.state.run_step + 1,
                        retry=ctx.state.output_retries_used,
                        max_retries=ctx.deps.tool_manager.default_max_retries,
                    )
                    ctx.deps.tool_manager = await ctx.deps.tool_manager.for_run_step(run_context)
                    if last_message.tool_calls:
                        # Pending tool calls must be processed before any new ModelRequest, regardless
                        # of instructions.  Instructions will be applied by ModelRequestNode.run() on
                        # the subsequent request after tool results are collected.
                        return CallToolsNode[DepsT, NodeRunEndT](last_message)
                    instruction_parts = await _get_instructions(ctx, run_context)
                    if not instruction_parts:
                        # No pending tool calls and no instructions — nothing new to send to the model.
                        return CallToolsNode[DepsT, NodeRunEndT](last_message)
                elif last_message.state == 'suspended':
                    # A new prompt on top of a suspended turn would abandon it, leaking the provider's
                    # server-side job (e.g. an OpenAI background run). Resume it first (run with this
                    # history and no new prompt) before starting a new turn.
                    raise exceptions.UserError(
                        'Cannot provide a new user prompt when the message history ends in a suspended response. '
                        'Resume it by running the agent with this message history and no new prompt.'
                    )
                elif last_message.tool_calls:
                    if last_message.state == 'interrupted':
                        # The response was cut off (e.g. a cancelled stream), so its tool calls
                        # will never be executed; close them out with synthesized returns instead
                        # of refusing the new prompt.
                        messages[:] = _repair_dangling_tool_calls(messages, repair_last_response=True)
                    else:
                        raise exceptions.UserError(
                            'Cannot provide a new user prompt when the message history contains unprocessed tool calls.'
                        )

        if not run_context:
            run_context = build_run_context(ctx)

        if messages:
            await self._reevaluate_dynamic_prompts(messages, run_context)

        if next_message:
            await self._reevaluate_dynamic_prompts([next_message], run_context)
        else:
            parts: list[_messages.ModelRequestPart] = []
            if not messages:
                parts.extend(await self._sys_parts(run_context))

            if self.user_prompt is not None:
                parts.append(_messages.UserPromptPart(self.user_prompt))

            next_message = _messages.ModelRequest(parts=parts)

        return ModelRequestNode[DepsT, NodeRunEndT](
            request=next_message, is_resuming_without_prompt=is_resuming_without_prompt
        )

    async def _handle_deferred_tool_results(
        self,
        deferred_tool_results: DeferredToolResults,
        messages: list[_messages.ModelMessage],
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
    ) -> CallToolsNode[DepsT, NodeRunEndT]:
        if not messages:
            raise exceptions.UserError('Tool call results were provided, but the message history is empty.')

        last_model_request: _messages.ModelRequest | None = None
        last_model_response: _messages.ModelResponse | None = None
        for message in reversed(messages):
            if isinstance(message, _messages.ModelRequest):
                last_model_request = message
            elif isinstance(message, _messages.ModelResponse):  # pragma: no branch
                last_model_response = message
                break

        if not last_model_response:
            raise exceptions.UserError(
                'Tool call results were provided, but the message history does not contain a `ModelResponse`.'
            )
        if not last_model_response.tool_calls:
            raise exceptions.UserError(
                'Tool call results were provided, but the message history does not contain any unprocessed tool calls.'
            )

        tool_call_results: dict[str, DeferredToolResult | Literal['skip']] = {}
        tool_call_results.update(deferred_tool_results.to_tool_call_results())

        if last_model_request:
            for part in last_model_request.parts:
                if isinstance(part, _messages.ToolReturnPart | _messages.RetryPromptPart):
                    if part.tool_call_id in tool_call_results:
                        raise exceptions.UserError(
                            f'Tool call {part.tool_call_id!r} was already executed and its result cannot be overridden.'
                        )
                    tool_call_results[part.tool_call_id] = 'skip'

        # Skip ModelRequestNode and go directly to CallToolsNode
        return CallToolsNode[DepsT, NodeRunEndT](
            last_model_response,
            tool_call_results=tool_call_results,
            tool_call_metadata=deferred_tool_results.metadata or None,
            user_prompt=self.user_prompt,
        )

    async def _reevaluate_dynamic_prompts(
        self, messages: list[_messages.ModelMessage], run_context: RunContext[DepsT]
    ) -> None:
        """Reevaluate any `SystemPromptPart` with dynamic_ref in the provided messages by running the associated runner function."""
        # Only proceed if there's at least one dynamic runner.
        if self.system_prompt_dynamic_functions:
            for msg in messages:
                if isinstance(msg, _messages.ModelRequest):
                    reevaluated_message_parts: list[_messages.ModelRequestPart] = []
                    for part in msg.parts:
                        if isinstance(part, _messages.SystemPromptPart) and part.dynamic_ref:
                            # Look up the runner by its ref
                            if runner := self.system_prompt_dynamic_functions.get(  # pragma: lax no cover
                                part.dynamic_ref
                            ):
                                # To enable dynamic system prompt refs in future runs, use a placeholder string
                                updated_part_content = await runner.run(run_context)
                                part = _messages.SystemPromptPart(
                                    updated_part_content or '', dynamic_ref=part.dynamic_ref
                                )

                        reevaluated_message_parts.append(part)

                    # Replace message parts with reevaluated ones to prevent mutating parts list
                    if reevaluated_message_parts != msg.parts:
                        msg.parts = reevaluated_message_parts

    async def _sys_parts(self, run_context: RunContext[DepsT]) -> list[_messages.SystemPromptPart]:
        """Build the initial system-prompt messages for the conversation."""
        return await _system_prompt.resolve_system_prompts(
            self.system_prompts, self.system_prompt_functions, run_context
        )

    __repr__ = dataclasses_no_defaults_repr


async def _get_instructions(
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
    run_context: RunContext[DepsT],
) -> list[_messages.InstructionPart] | None:
    """Combine base instructions (from agent/capabilities) with toolset instructions.

    Toolset instructions are fetched from the current tool manager's toolset,
    which reflects any changes from for_run_step.
    """
    parts: list[_messages.InstructionPart] = []

    base = await ctx.deps.get_instructions(run_context)
    if base:
        parts.extend(base)

    toolset_result = await ctx.deps.tool_manager.toolset.get_instructions(run_context)
    parts.extend(normalize_toolset_instructions(toolset_result))

    return parts or None


async def _prepare_request_parameters(
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
    instruction_parts: list[_messages.InstructionPart] | None,
) -> models.ModelRequestParameters:
    """Build tools and create an agent model."""
    output_schema = ctx.deps.output_schema

    prompted_output_template = (
        output_schema.template if isinstance(output_schema, _output.StructuredTextOutputSchema) else None
    )

    # `tool_manager.tool_defs` already reflects the `prepare_tools`/`prepare_output_tools`
    # capability hooks — they're dispatched at `get_tools()` time via `PreparedToolset`
    # wrappers in `Agent._get_toolset`, so the filtered/modified defs are baked into
    # `ToolManager.tools` (and execution lookups) as well as the model's request parameters.
    function_tools: list[ToolDefinition] = []
    output_tools: list[ToolDefinition] = []
    for tool_def in ctx.deps.tool_manager.tool_defs:
        if tool_def.kind == 'output':
            output_tools.append(tool_def)
        else:
            function_tools.append(tool_def)

    run_context = build_run_context(ctx)

    raw_native_tools: list[AgentNativeTool[DepsT]] = list(ctx.deps.native_tools)

    # resolve dynamic native tools
    native_tools: list[AbstractNativeTool] = []
    if raw_native_tools:
        for tool in raw_native_tools:
            if isinstance(tool, AbstractNativeTool):
                native_tools.append(tool)
            else:
                t = tool(run_context)
                if inspect.isawaitable(t):
                    t = await t
                if t is not None:
                    native_tools.append(t)

    # Drop the auto-injected `ToolSearchTool` native tool when the search corpus is empty —
    # the toolset has nothing to manage, so emitting the native tool would waste a tool slot
    # and surface an inert native tool in `ModelRequestParameters` snapshots. Filtering
    # here (at MRP-construction time) keeps the request shape honest before
    # `prepare_request` runs. Non-optional `ToolSearchTool` instances (user-passed) are
    # preserved so the request still fails loudly on unsupported models.
    has_tool_search_corpus = any(t.with_native == ToolSearchTool.kind for t in function_tools)
    if not has_tool_search_corpus:
        # Confine the corpus-empty drop to `ToolSearchTool`: other optional native tools
        # (e.g. a hypothetical `WebSearchTool(optional=True)`) don't have a corpus and
        # shouldn't be dropped here — they only get dropped on the unsupported-on-this-model
        # path in `Model.prepare_request`.
        native_tools = [t for t in native_tools if not (isinstance(t, ToolSearchTool) and t.optional)]

    return models.ModelRequestParameters(
        function_tools=function_tools,
        native_tools=native_tools,
        output_mode=output_schema.mode,
        output_tools=output_tools,
        output_object=output_schema.object_def,
        prompted_output_template=prompted_output_template,
        allow_text_output=output_schema.allows_text,
        allow_image_output=output_schema.allows_image,
        instruction_parts=instruction_parts,
    )


# --- Innermost model-call helpers ---
#
# Both the agent graph (ModelRequestNode) and durable execution capabilities
# (TemporalDurability/DBOSDurability/PrefectDurability inside their
# activity/step/task) need to invoke the model as the innermost operation of a
# request. Routing both call sites through these helpers keeps the innermost
# logic in one place — any future addition (telemetry, caching, error
# translation) lands in both paths automatically. In particular, the
# suspended → complete continuation loop (Anthropic `pause_turn`, OpenAI
# background mode) lives here, so a durability capability's activity/step/task
# runs the whole chain inside one durable unit.


def _split_resume_seed(
    messages: list[_messages.ModelMessage],
) -> tuple[list[_messages.ModelMessage], _messages.ModelResponse | None]:
    """Split a trailing suspended `ModelResponse` off `messages` as the continuation seed.

    A history ending in a `ModelResponse` with `state == 'suspended'` is the wire-truthful
    encoding of a paused turn to resume: the continuation loop echoes that response back to
    the provider itself, so it must not be part of the base history handed to the model.
    A normal request ends in a `ModelRequest`, so the seed is `None` and the messages pass
    through untouched.
    """
    if messages and isinstance(last := messages[-1], _messages.ModelResponse) and last.state == 'suspended':
        return messages[:-1], last
    return messages, None


def _check_continuation_usage(run_context: RunContext[Any], continuation_usage: _usage.RequestUsage) -> None:
    """Enforce token limits mid-turn against a provisional total during continuations.

    Continuation segments accumulate usage but aren't committed to the run usage until the
    final merged response is appended exactly once by `ModelRequestNode._append_response`
    (so a continuation isn't double-counted, nor counted as a separate request step). To
    still fail fast when a segment blows the token budget, check the limit against a
    throwaway copy of the run usage plus the accumulated continuation usage. Works both in
    the agent graph (where `run_context.usage` is the live run usage) and inside a durable
    boundary (where it's the serialized snapshot the activity/step/task received — the final
    workflow-side check still applies when the merged response is committed).
    """
    if run_context.usage_limits:
        provisional = deepcopy(run_context.usage)
        provisional.incr(continuation_usage)
        run_context.usage_limits.check_tokens(provisional)


async def model_request(
    model: models.Model,
    *,
    request_context: ModelRequestContext,
    run_context: RunContext[Any],
    on_progress: Callable[[_messages.ModelResponse], None],
) -> _messages.ModelResponse:
    """Run the innermost non-streaming model request, resolving any continuation chain.

    Loops over any suspended → complete continuation segments (Anthropic `pause_turn`,
    OpenAI background mode), echoing each suspended response back and merging the segments
    into one response. Only the final merged response is returned, so `wrap_model_request`
    spans the whole chain and `after_model_request` sees just the final response.
    Continuations are not separate request steps, so usage is committed exactly once when
    the merged response is appended to history. When `request_context.messages` ends in a
    suspended `ModelResponse` (a resumed run), that response seeds the loop.

    Under the bundled durable-execution capabilities (Temporal/DBOS/Prefect) this loop runs
    in workflow code: the capability swaps `request_context.model` for a wrapper that
    dispatches each segment's `model.request(...)` through its own activity/step/task, so a
    failed segment retries alone and each suspended response is checkpointed between
    segments.

    Args:
        model: The model to call.
        request_context: The merged request context (messages, settings, parameters).
        run_context: The current run context, made available via `get_current_run_context`.
        on_progress: Callback invoked with the merged response after each segment, so the
            caller can preserve partial progress when a later segment's error is converted
            to a retry.

    Returns:
        The (merged) model response.
    """
    base_messages, seed = _split_resume_seed(request_context.messages)

    # Two independent ceilings distinguished by the generic `merge_mode` signal, mirroring the
    # streamed composite in `_continuation`: every *fresh-generation* re-suspension (accumulate
    # `pause_turn`, a model change, or a `FallbackModel` replace directive) keeps the small
    # `MAX_GENERATION_CONTINUATIONS` cap against an unbounded model, while only a *same-id* re-suspension
    # re-polling one background job (OpenAI background mode, same `provider_response_id`) gets the
    # far more generous `MAX_BACKGROUND_POLLS` backstop so a legitimately long job isn't killed.
    accumulate_count = 0
    replace_count = 0
    # Mode of the merge that produced the current suspended `response`. A chain is homogeneous in
    # practice, so the previous merge's mode reliably classifies the next re-issue; the first
    # re-issue (`last_mode is None`) counts as strict, harmless since both ceilings allow ≥1.
    last_mode: MergeMode | None = None
    response = seed
    with set_current_run_context(run_context):
        while True:
            if response is None:
                messages = base_messages
            elif response.state == 'suspended':
                job_id = response.provider_response_id
                if last_mode == 'replace-same-id':
                    replace_count += 1
                    over_limit = replace_count > MAX_BACKGROUND_POLLS
                    limit_message = (
                        f'Model response for job {job_id!r} remained suspended after polling the maximum '
                        f'of {MAX_BACKGROUND_POLLS} times'
                    )
                else:
                    accumulate_count += 1
                    over_limit = accumulate_count > MAX_GENERATION_CONTINUATIONS
                    limit_message = (
                        f'Model response {job_id!r} was suspended more than the maximum of '
                        f'{MAX_GENERATION_CONTINUATIONS} times'
                    )
                if over_limit:
                    # Giving up on a still-suspended job: cancel it before raising so it doesn't leak.
                    await cancel_suspended_job(model, response)
                    raise exceptions.UnexpectedModelBehavior(limit_message)
                if delay := model.continuation_delay(response):
                    try:
                        await _agent_graph_sleep(delay)
                    except BaseException:
                        # A `CancelledError` (or any error) raised while parked in the inter-poll
                        # sleep sits outside the request's cancel guard below, so cancel the job
                        # here too before propagating.
                        await cancel_suspended_job(model, response)
                        raise
                messages = [*base_messages, response]
            else:
                return response

            try:
                new_response = await model.request(
                    messages, request_context.model_settings, request_context.model_request_parameters
                )
            except BaseException:
                # The broad catch is deliberate: `BaseException` also covers `CancelledError`,
                # `KeyboardInterrupt`, and `SystemExit`, and we must cancel the server-side
                # suspended/background job before letting any of them propagate so it doesn't leak.
                if response is not None:
                    await cancel_suspended_job(model, response)
                raise

            new_response = _narrow_tool_call_parts(new_response, request_context.model_request_parameters)
            if response is None:
                response = new_response
            else:
                # Classify this transition (replace vs accumulate) so the next re-issue is
                # counted against the right ceiling.
                last_mode = merge_mode(response, new_response)
                response = merge_responses(response, new_response)
                # Enforce token limits early against a provisional total so a runaway
                # continuation can't blow the budget; the total is committed once later.
                try:
                    _check_continuation_usage(run_context, response.usage)
                except BaseException:
                    # The limit tripped on a still-suspended merge: cancel the live
                    # server-side job before propagating so it doesn't leak (mirrors the
                    # request-failure guard above and the streamed composite's check).
                    if response.state == 'suspended':
                        await cancel_suspended_job(model, response)
                    raise
            on_progress(response)


@asynccontextmanager
async def model_request_stream(
    model: models.Model,
    *,
    request_context: ModelRequestContext,
    run_context: RunContext[Any],
) -> AsyncGenerator[models.StreamedResponse]:
    """Open the innermost streaming model request, stitching any continuation chain.

    Under the bundled durable-execution capabilities (Temporal/DBOS/Prefect) this runs in
    workflow code: the capability swaps `request_context.model` for a wrapper whose
    `request_stream` drains one segment inside its own activity/step/task and replays its
    buffered events, so the composite below stitches per-segment replays.

    The yielded stream is a composite that stitches the (possibly suspended → complete)
    segments into one continuous stream: it opens a `model.request_stream(...)` per segment
    as it's iterated, so the whole chain is presented as a single stream and the
    model-request hooks wrap it once. When `request_context.messages` ends in a suspended
    `ModelResponse` (a resumed run), that response seeds the loop. On exit, the helper tears
    down any in-flight segment's connection (`aclose()`), which deliberately does *not*
    cancel a still-pending server-side job — cancellation stays on the
    `AgentStream.cancel()` → `close_stream()` path.

    Args:
        model: The model to call.
        request_context: The merged request context.
        run_context: The current run context.

    Yields:
        A `StreamedResponse` to iterate inside the durable boundary.
    """
    base_messages, seed = _split_resume_seed(request_context.messages)
    with set_current_run_context(run_context):
        sr = _ContinuationStreamedResponse(
            model_request_parameters=request_context.model_request_parameters,
            model=model,
            model_settings=request_context.model_settings,
            base_messages=base_messages,
            run_context=run_context,
            max_generation_continuations=MAX_GENERATION_CONTINUATIONS,
            max_background_polls=MAX_BACKGROUND_POLLS,
            sleep_func=_agent_graph_sleep,
            check_usage=lambda continuation_usage: _check_continuation_usage(run_context, continuation_usage),
            initial_suspended_response=seed,
            # The composite opens each segment lazily in the consumer task, which doesn't share
            # this task's OTel context (where `wrap_model_request` opened the `chat` span). Capture
            # it here so re-attaching it around each segment keeps `get_current_span()`-driven span
            # updates (e.g. `FallbackModel` recording the resolved inner model) on the right span.
            segment_context=capture_current_context(),
        )
        try:
            yield sr
        finally:
            # Deterministically tear down an in-flight segment's connection once the
            # consumer has stopped (mirrors the pre-stitching `async with request_stream`
            # teardown; a no-op after a fully-drained stream). Server-side cancellation
            # stays on the `AgentStream.cancel()` → `close_stream()` path.
            await sr.aclose()


@dataclasses.dataclass
class ModelRequestNode(AgentNode[DepsT, NodeRunEndT]):
    """The node that makes a request to the model using the last message in state.message_history."""

    request: _messages.ModelRequest
    is_resuming_without_prompt: bool = False

    _: dataclasses.KW_ONLY

    _resume_suspended: _messages.ModelResponse | None = None
    """A suspended `ModelResponse` from a prior run to resume, when the run's `message_history`
    ends in a provider-paused turn (Anthropic `pause_turn`, OpenAI background mode). Set by
    `UserPromptNode`; dispatches `_prepare_request` to the resume path, which keeps the
    suspended tail on the request messages so the continuation loop in the innermost
    `model_request`/`model_request_stream` helpers can echo it back to complete the turn."""

    _result: CallToolsNode[DepsT, NodeRunEndT] | ModelRequestNode[DepsT, NodeRunEndT] | None = field(
        repr=False, init=False, default=None
    )
    _did_stream: bool = field(repr=False, init=False, default=False)
    last_request_context: ModelRequestContext | None = field(repr=False, init=False, default=None)

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> CallToolsNode[DepsT, NodeRunEndT] | ModelRequestNode[DepsT, NodeRunEndT]:
        if self._result is not None:
            return self._result

        if self._did_stream:
            # `self._result` gets set when exiting the `stream` contextmanager, so hitting this
            # means that the stream was started but not finished before `run()` was called
            raise exceptions.AgentRunError('You must finish streaming before calling run()')  # pragma: no cover

        return await self._make_request(ctx)

    @asynccontextmanager
    async def stream(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, T]],
    ) -> AsyncGenerator[result.AgentStream[DepsT, T]]:
        assert not self._did_stream, 'stream() should only be called once per node'

        try:
            model, model_settings, model_request_parameters, message_history, run_context = await self._prepare_request(
                ctx, streaming=True
            )
        except exceptions.SkipModelRequest as e:
            # SkipModelRequest in stream path: yield an empty stream and finish handling
            # new_message_index wasn't updated in _prepare_request, fix it here
            ctx.deps.new_message_index = _first_new_message_index(
                ctx.state.message_history,
                ctx.state.run_id,
                resumed_request=ctx.deps.resumed_request,
                resumed_request_index=ctx.deps.resumed_request_index,
            )
            self._did_stream = True
            ctx.state.usage.requests += 1
            # instruction_parts=None is fine here: the model isn't called, we just need MRP for the wrapper
            skip_mrp = await _prepare_request_parameters(ctx, instruction_parts=None)
            skip_sr = CompletedStreamedResponse(e.response, model_request_parameters=skip_mrp)
            agent_stream = self._build_agent_stream(ctx, skip_sr, skip_mrp)
            yield agent_stream
            await self._finish_handling(ctx, e.response)
            assert self._result is not None
            return

        # Cooperative hand-off between this coroutine and the wrap_model_request task:
        # 1. The task runs capability middleware, then calls _streaming_handler which opens the stream.
        # 2. _streaming_handler sets stream_ready once the stream is open, then waits on stream_done.
        # 3. This coroutine waits for stream_ready (or early task completion), yields the stream
        #    to the caller, and sets stream_done when the caller is finished consuming it.
        # 4. The handler resumes, the stream context manager closes, and the task completes.
        stream_ready = asyncio.Event()
        stream_done = asyncio.Event()
        agent_stream_holder: list[result.AgentStream[DepsT, T]] = []

        _handler_response: _messages.ModelResponse | None = None

        async def _streaming_handler(
            req_ctx: ModelRequestContext,
        ) -> _messages.ModelResponse:
            nonlocal _handler_response
            # Stamp the request-issue instant so the instrumentation capability can record
            # `gen_ai.client.operation.time_to_first_chunk` (TTFT). `StreamedResponse` records
            # the first-chunk instant; the delta is the client-side time to first token.
            request_start = time.perf_counter()
            # `model_request_stream` stitches the (possibly suspended → complete) segments
            # into one continuous stream, so the whole chain is presented as a single
            # `AgentStream` and the model-request hooks wrap it once.
            # `ctx.state.usage.requests` is bumped once here: continuations aren't
            # separate request steps.
            async with model_request_stream(req_ctx.model, request_context=req_ctx, run_context=run_context) as sr:
                self._did_stream = True
                ctx.state.usage.requests += 1
                agent_stream = self._build_agent_stream(ctx, sr, req_ctx.model_request_parameters)
                agent_stream_holder.append(agent_stream)
                stream_ready.set()
                try:
                    await stream_done.wait()
                finally:
                    # Report TTFT in a `finally` so it also lands when the consumer raises
                    # mid-iteration and `_cancel_task(wrap_task)` injects CancelledError at
                    # the `wait()` above, mirroring `InstrumentedModel.request_stream`. On
                    # that cancelled path `finish` is never reached today (no metrics of any
                    # kind are recorded), so this is symmetry rather than an observable fix.
                    time_to_first_chunk_ctx.set(sr.time_to_first_chunk(request_start))
            response = sr.get()
            _handler_response = response
            return response

        wrap_request_context = ModelRequestContext(
            model=model,
            messages=message_history,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
        )
        wrap_request_context.model_id = ctx.deps.model_id
        # Signal to hooks that the agent loop expects a real event stream.
        wrap_request_context.streaming = True
        wrap_task = asyncio.create_task(
            ctx.deps.root_capability.wrap_model_request(
                run_context,
                request_context=wrap_request_context,
                handler=_streaming_handler,
            )
        )

        # Wait for handler to start or wrap to complete (short-circuit).
        # If outer cancellation arrives during this wait, drain both tasks before re-raising
        # so the user's `wrap_model_request` cleanup runs instead of orphaning.
        ready_waiter = asyncio.create_task(stream_ready.wait())
        try:
            await asyncio.wait({ready_waiter, wrap_task}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            # `BaseException` to also catch `CancelledError`. Handoff hasn't completed,
            # so both tasks are still ours; drain them so cleanup runs before we re-raise.
            #
            # Unblock `_streaming_handler` before draining: if wrap_task's model
            # absorbed the CancelledError (e.g. Temporal's cooperative cancellation),
            # the handler is parked on `stream_done.wait()`. Setting stream_done lets
            # it exit so cancel_and_drain's gather can complete. Harmless no-op when
            # the task was actually cancelled — it's already unwinding. See https://github.com/pydantic/pydantic-ai/issues/6422.
            stream_done.set()
            await cancel_and_drain(ready_waiter, wrap_task)
            raise
        else:
            # Handoff succeeded: `wrap_task` is owned by the rest of the streaming
            # lifecycle below. Only the throwaway readiness waiter is ours to clean up.
            await cancel_and_drain(ready_waiter)

        if wrap_task.done() and not stream_ready.is_set():
            # wrap_model_request completed without calling handler — short-circuited or raised SkipModelRequest
            try:
                result_or_exc: _messages.ModelResponse | Exception
                try:
                    result_or_exc = wrap_task.result()
                except Exception as e:
                    result_or_exc = e
                model_response = await self._resolve_wrap_result(ctx, run_context, wrap_request_context, result_or_exc)
            except exceptions.ModelRetry as e:
                self._did_stream = True
                # Don't increment usage.requests — handler was never called (short-circuit)
                run_context = build_run_context(ctx)
                await self._build_retry_node(ctx, e)
                # Must still yield from @asynccontextmanager — yield an empty stream
                dummy_sr = CompletedStreamedResponse(
                    _messages.ModelResponse(parts=[]), model_request_parameters=model_request_parameters
                )
                yield self._build_agent_stream(ctx, dummy_sr, model_request_parameters)
                return
            self._did_stream = True
            ctx.state.usage.requests += 1
            replay_sr = CompletedStreamedResponse(
                model_response,
                model_request_parameters=model_request_parameters,
                replay_events=True,
            )
            agent_stream = self._build_agent_stream(ctx, replay_sr, model_request_parameters)
            yield agent_stream
            self.last_request_context = wrap_request_context
            await self._finish_handling(ctx, model_response)
            assert self._result is not None
            return

        # Normal path: handler was called, stream is ready
        stream_error: BaseException | None = None
        try:
            yield agent_stream_holder[0]
        except BaseException as exc:
            stream_error = exc
            raise
        finally:
            stream_done.set()

            if stream_error is not None:
                await _cancel_task(wrap_task)
                # Capture the partial response so `capture_run_messages` and `all_messages()`
                # include what was streamed before the interruption.
                # We append directly rather than via `_append_response` to skip the usage-limit
                # check; raising `UsageLimitExceeded` here would mask `stream_error`.
                if agent_stream_holder:  # pragma: no branch
                    partial = agent_stream_holder[0].response
                    recorded_state = await _resolve_interrupted_stream_state(model, stream_error, partial)
                    partial_response = replace(
                        partial,
                        state=recorded_state,
                        run_id=ctx.state.run_id,
                        conversation_id=ctx.state.conversation_id,
                    )
                    ctx.state.usage.incr(partial_response.usage)
                    ctx.state.message_history.append(partial_response)
            else:
                try:
                    try:
                        model_response = await wrap_task
                    except exceptions.ModelRetry:
                        raise  # Propagate to outer handler
                    except Exception as e:
                        model_response = await ctx.deps.root_capability.on_model_request_error(
                            run_context, request_context=wrap_request_context, error=e
                        )
                except exceptions.ModelRetry as e:
                    # Don't increment usage.requests — _streaming_handler already did
                    # In the normal streaming path the handler was always called (that's
                    # how the stream was created), so _handler_response is always set.
                    assert _handler_response is not None
                    self._append_response(ctx, _handler_response)
                    await self._build_retry_node(ctx, e)
                else:
                    self.last_request_context = wrap_request_context
                    await self._finish_handling(ctx, model_response)
                    assert self._result is not None

    @staticmethod
    def _build_agent_stream(
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, T]],
        stream_response: models.StreamedResponse,
        model_request_parameters: models.ModelRequestParameters,
    ) -> result.AgentStream[DepsT, T]:
        """Build an AgentStream from the given stream response and context."""
        return result.AgentStream[DepsT, T](
            _raw_stream_response=stream_response,
            _output_schema=ctx.deps.output_schema,
            _model_request_parameters=model_request_parameters,
            _output_validators=ctx.deps.output_validators,
            _run_ctx=build_run_context(ctx),
            _usage_limits=ctx.deps.usage_limits,
            _tool_manager=ctx.deps.tool_manager,
            _root_capability=ctx.deps.root_capability,
            _metadata_getter=lambda: ctx.state.metadata,
            _event_stream_buffer_getter=lambda: ctx.state.event_stream_buffer,
        )

    async def _make_request(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> CallToolsNode[DepsT, NodeRunEndT] | ModelRequestNode[DepsT, NodeRunEndT]:
        if self._result is not None:
            return self._result  # pragma: no cover

        try:
            model, model_settings, model_request_parameters, message_history, run_context = await self._prepare_request(
                ctx, streaming=False
            )
        except exceptions.SkipModelRequest as e:
            # new_message_index wasn't updated in _prepare_request, fix it here
            ctx.deps.new_message_index = _first_new_message_index(
                ctx.state.message_history,
                ctx.state.run_id,
                resumed_request=ctx.deps.resumed_request,
                resumed_request_index=ctx.deps.resumed_request_index,
            )
            ctx.state.usage.requests += 1
            return await self._finish_handling(ctx, e.response)

        _handler_response: _messages.ModelResponse | None = None

        async def model_handler(req_ctx: ModelRequestContext) -> _messages.ModelResponse:
            nonlocal _handler_response

            # `model_request` resolves any suspended → complete continuation chain (Anthropic
            # `pause_turn`, OpenAI background mode) and returns the final merged response, so
            # `wrap_model_request` spans the whole chain and `after_model_request` sees just
            # the final response. Continuations are not separate request steps, so usage is
            # committed exactly once by `_finish_handling` → `_append_response`.
            def on_progress(response: _messages.ModelResponse) -> None:
                nonlocal _handler_response
                _handler_response = response

            response = await model_request(
                req_ctx.model, request_context=req_ctx, run_context=run_context, on_progress=on_progress
            )
            _handler_response = response
            return response

        request_context = ModelRequestContext(
            model=model,
            messages=message_history,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
        )
        request_context.model_id = ctx.deps.model_id
        try:
            try:
                model_response = await ctx.deps.root_capability.wrap_model_request(
                    run_context,
                    request_context=request_context,
                    handler=model_handler,
                )
            except exceptions.SkipModelRequest as e:
                model_response = e.response
            except exceptions.ModelRetry:
                raise  # Propagate to outer handler
            except Exception as e:
                model_response = await ctx.deps.root_capability.on_model_request_error(
                    run_context, request_context=request_context, error=e
                )
        except exceptions.ModelRetry as e:
            # ModelRetry from wrap_model_request or on_model_request_error — retry the model request.
            # If the handler was called, preserve the response in history for context.
            if _handler_response is not None:
                ctx.state.usage.requests += 1
                self._append_response(ctx, _handler_response)
            return await self._build_retry_node(ctx, e)
        self.last_request_context = request_context
        ctx.state.usage.requests += 1

        return await self._finish_handling(ctx, model_response)

    async def _prepare_request(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        *,
        streaming: bool,
    ) -> tuple[
        models.Model,
        ModelSettings | None,
        models.ModelRequestParameters,
        list[_messages.ModelMessage],
        RunContext[DepsT],
    ]:
        if self._resume_suspended is not None:
            return await self._prepare_resume_request(ctx, streaming=streaming)

        self.request.timestamp = now_utc()
        if not self.is_resuming_without_prompt:
            self.request.run_id = self.request.run_id or ctx.state.run_id
            self.request.conversation_id = self.request.conversation_id or ctx.state.conversation_id
        ctx.state.message_history.append(self.request)

        ctx.state.run_step += 1

        await _select_model(ctx)

        _refresh_loaded_capability_ids(ctx)

        _refresh_discovered_tool_names(ctx)

        run_context = build_run_context(ctx)
        run_context = replace(
            run_context,
            retry=ctx.state.output_retries_used,
            max_retries=ctx.deps.tool_manager.default_max_retries,
        )

        # This will raise errors for any tool name conflicts.
        # Note: for_run_step may already have been called by UserPromptNode for the
        # resume-without-prompt path; ToolManager.for_run_step is a no-op for the same step.
        ctx.deps.tool_manager = await ctx.deps.tool_manager.for_run_step(run_context)

        # Fetch instructions now that dynamic toolsets have been resolved by for_run_step.
        instruction_parts = await _get_instructions(ctx, run_context)
        if instruction_parts:
            instruction_parts = _messages.InstructionPart.sorted(instruction_parts) or None
        self.request.instructions = _messages.InstructionPart.join(instruction_parts) if instruction_parts else None

        # Validate after instructions are resolved; self.request was appended above so [:-1] is prior history
        if not ctx.state.message_history[:-1] and not self.request.parts and not self.request.instructions:
            raise exceptions.UserError('No message history, user prompt, or instructions provided')

        model_request_parameters = await _prepare_request_parameters(ctx, instruction_parts)
        model_settings = ctx.deps.get_model_settings(run_context) or ModelSettings()
        run_context.model_settings = model_settings

        request_context = ModelRequestContext(
            model=ctx.deps.model,
            messages=ctx.state.message_history[:],
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
        )
        request_context.model_id = ctx.deps.model_id
        request_context.streaming = streaming
        messages_before_processing = len(request_context.messages)
        self.last_request_context = request_context
        request_context = await ctx.deps.root_capability.before_model_request(
            run_context,
            request_context,
        )
        self.last_request_context = request_context
        model = request_context.model
        messages = request_context.messages
        model_settings = request_context.model_settings
        model_request_parameters = request_context.model_request_parameters

        if len(messages) == 0:
            raise exceptions.UserError('Processed history cannot be empty.')

        if not isinstance(messages[-1], _messages.ModelRequest):
            raise exceptions.UserError('Processed history must end with a `ModelRequest`.')

        # Fill in framework metadata the history processors may have left unset on a new `ModelRequest`.
        fill_run_metadata(messages[-1], run_id=ctx.state.run_id, conversation_id=ctx.state.conversation_id)

        if self.is_resuming_without_prompt:
            # No separate user-prompt request this run: the trailing request that arrived via
            # `message_history` *is* the request being sent, so it's prior context, not new. Track it
            # two ways so `_first_new_message_index` can exclude it however capabilities/processors
            # mutate the list: by object (identity/value, survives reordering and removal) and by
            # position (survives an in-place rebuild that changes its fields). It's the last message
            # here, before the model output is appended, so its index is `len(messages) - 1`.
            ctx.deps.resumed_request = self.request
            ctx.deps.resumed_request_index = len(messages) - 1
        elif ctx.deps.resumed_request_index is not None:
            # Later steps (e.g. a tool-call loop) may prepend/truncate/rebuild messages ahead of the
            # resumed request, shifting it. Translate the pinned index by the net count change; drop
            # it (falling back to object/value matching, then run_id) if processing removed the
            # resumed request itself. The object reference is left untouched — it still points at the
            # step-1 request, so identity/value matching keeps working across steps.
            shifted = ctx.deps.resumed_request_index - (messages_before_processing - len(messages))
            ctx.deps.resumed_request_index = shifted if shifted >= 0 else None
        # `ctx.state.message_history` is the same list used by `capture_run_messages`, so we should replace its contents, not the reference
        ctx.state.message_history[:] = messages
        # Update the new message index to ensure `result.new_messages()` returns the correct messages
        ctx.deps.new_message_index = _first_new_message_index(
            messages,
            ctx.state.run_id,
            resumed_request=ctx.deps.resumed_request,
            resumed_request_index=ctx.deps.resumed_request_index,
        )

        # Merge possible consecutive trailing `ModelRequest`s into one, with tool call parts before user parts,
        # but don't store it in the message history on state. This is just for the benefit of model classes that want clear user/assistant boundaries.
        # See `tests/test_tools.py::test_parallel_tool_return_with_deferred` for an example where this is necessary.
        #
        # Run a first pass so `prepare_messages` sees a normalized history.
        # The history is definitively being sent to the model at this point, so even the last
        # response's dangling tool calls (e.g. left by a history processor) can be repaired.
        messages = _clean_message_history(messages, repair_last_response=True)

        # Hand off to the model class for any history shapes the active provider can't
        # ship on the wire — currently typed `NativeToolSearch*Part` instances translated
        # to local-shape `ToolSearch*Part` when the profile doesn't support `ToolSearchTool`.
        #
        # Lives on `Model.prepare_messages` rather than inline here for two reasons:
        # 1. The translation depends on `self.profile`, which is per-model state.
        # 2. `FallbackModel` defers the decision until it's picked an underlying model — so
        #    each candidate runs `prepare_messages` itself with its own profile when chosen.
        prepared = model.prepare_messages(messages)

        # If `prepare_messages` produced a new list (e.g. tool-search synthesis split a
        # `ModelResponse(call+return)` into `ModelResponse(call) + ModelRequest(return)`
        # adjacent to an existing `ModelRequest`), re-run cleanup so consecutive same-role
        # messages are merged. The default `prepare_messages` returns the input list
        # unchanged, so the identity check skips the redundant second pass.
        if prepared is not messages:
            messages = _clean_message_history(prepared, repair_last_response=True)
        else:
            messages = prepared

        ctx.state.last_max_tokens = model_settings.get('max_tokens') if model_settings else None
        ctx.state.last_model_request_parameters = model_request_parameters
        usage = ctx.state.usage
        if ctx.deps.usage_limits.count_tokens_before_request:
            # Copy to avoid modifying the original usage object with the counted usage
            usage = deepcopy(usage)

            counted_usage = await model.count_tokens(messages, model_settings, model_request_parameters)
            usage.incr(counted_usage)

        ctx.deps.usage_limits.check_before_request(usage)

        return model, model_settings or None, model_request_parameters, messages, run_context

    async def _prepare_resume_request(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        *,
        streaming: bool,
    ) -> tuple[
        models.Model,
        ModelSettings | None,
        models.ModelRequestParameters,
        list[_messages.ModelMessage],
        RunContext[DepsT],
    ]:
        """Prepare a request that resumes a turn the provider paused mid-flight.

        Unlike `_prepare_request`, the `message_history` already ends in the suspended
        `ModelResponse` (there is no new `ModelRequest` to append). The request messages keep
        that suspended tail — the innermost `model_request`/`model_request_stream` helpers
        split it off as the continuation seed, so it also crosses a durable-execution
        boundary as part of the messages. Instructions are rehydrated from the recorded
        `ModelRequest` rather than re-evaluated, since a continuation completes the same
        logical turn and providers (e.g. Anthropic) require the exact prior history back.
        """
        assert self._resume_suspended is not None

        ctx.state.run_step += 1

        _refresh_loaded_capability_ids(ctx)
        _refresh_discovered_tool_names(ctx)

        run_context = build_run_context(ctx)
        run_context = replace(
            run_context,
            retry=ctx.state.output_retries_used,
            max_retries=ctx.deps.tool_manager.default_max_retries,
        )
        ctx.deps.tool_manager = await ctx.deps.tool_manager.for_run_step(run_context)

        instructions = _get_history_instructions(ctx.state.message_history)
        instruction_parts = [_messages.InstructionPart(content=instructions)] if instructions else None

        model_request_parameters = await _prepare_request_parameters(ctx, instruction_parts)
        model_settings = ctx.deps.get_model_settings(run_context) or ModelSettings()
        run_context.model_settings = model_settings

        # Show the hooks the exact history that will be echoed back (ending in the suspended
        # response); the innermost helpers split that response off as the continuation seed.
        request_context = ModelRequestContext(
            model=ctx.deps.model,
            messages=ctx.state.message_history[:],
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
        )
        request_context.model_id = ctx.deps.model_id
        request_context.streaming = streaming
        self.last_request_context = request_context
        request_context = await ctx.deps.root_capability.before_model_request(run_context, request_context)
        self.last_request_context = request_context
        model = request_context.model
        messages = request_context.messages
        model_settings = request_context.model_settings
        model_request_parameters = request_context.model_request_parameters

        if not (
            messages
            and isinstance(suspended := messages[-1], _messages.ModelResponse)
            and suspended.state == 'suspended'
        ):
            raise exceptions.UserError('Processed history must end with a suspended `ModelResponse` to resume.')

        # History bookkeeping operates on the base history ending in the `ModelRequest` that
        # triggered the turn; the request messages keep the suspended tail (the continuation
        # loop in the innermost helpers re-appends it to the wire history itself).
        base_messages = messages[:-1]

        # `resumed_request` = the request that triggered the paused turn, so `new_messages()`
        # yields just the completed (merged) response. Track it by object (identity/value) and by
        # position so `_first_new_message_index` can exclude it however processors mutate the list.
        for index in range(len(base_messages) - 1, -1, -1):
            if isinstance(message := base_messages[index], _messages.ModelRequest):
                ctx.deps.resumed_request = message
                ctx.deps.resumed_request_index = index
                break

        # `ctx.state.message_history` is the same list used by `capture_run_messages`, so
        # replace its contents (dropping the suspended response) rather than the reference;
        # `_finish_handling` then appends the final merged response after the base history.
        ctx.state.message_history[:] = base_messages
        ctx.deps.new_message_index = _first_new_message_index(
            base_messages,
            ctx.state.run_id,
            resumed_request=ctx.deps.resumed_request,
            resumed_request_index=ctx.deps.resumed_request_index,
        )

        ctx.state.last_max_tokens = model_settings.get('max_tokens') if model_settings else None
        ctx.state.last_model_request_parameters = model_request_parameters
        ctx.deps.usage_limits.check_before_request(ctx.state.usage)

        return model, model_settings or None, model_request_parameters, messages, run_context

    async def _finish_handling(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        response: _messages.ModelResponse,
    ) -> CallToolsNode[DepsT, NodeRunEndT] | ModelRequestNode[DepsT, NodeRunEndT]:
        fill_run_metadata(response, run_id=ctx.state.run_id, conversation_id=ctx.state.conversation_id)

        run_context = build_run_context(ctx)
        assert self.last_request_context is not None, 'last_request_context must be set before _finish_handling'
        request_context = self.last_request_context
        run_context.model_settings = request_context.model_settings
        try:
            response = await ctx.deps.root_capability.after_model_request(
                run_context, request_context=request_context, response=response
            )
        except exceptions.ModelRetry as e:
            # Hook rejected the response — append it to history (model DID respond) and retry
            self._append_response(ctx, response)
            return await self._build_retry_node(ctx, e)

        # Append the model response to state.message_history
        self._append_response(ctx, response)

        # Set the `_result` attribute since we can't use `return` in an async iterator
        self._result = CallToolsNode(response)

        return self._result

    async def _resolve_wrap_result(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        run_context: RunContext[DepsT],
        request_context: ModelRequestContext,
        result_or_exc: _messages.ModelResponse | Exception,
    ) -> _messages.ModelResponse:
        """Resolve a wrap_model_request result, handling SkipModelRequest and errors.

        Returns ModelResponse on success.
        Raises ModelRetry if the result or on_model_request_error raises it.
        """
        if isinstance(result_or_exc, Exception):
            exc = result_or_exc
            if isinstance(exc, exceptions.SkipModelRequest):
                return exc.response
            if isinstance(exc, exceptions.ModelRetry):
                raise exc
            return await ctx.deps.root_capability.on_model_request_error(
                run_context, request_context=request_context, error=exc
            )
        return result_or_exc

    @staticmethod
    def _append_response(
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[Any, Any]],
        response: _messages.ModelResponse,
    ) -> None:
        """Append a model response to history, updating usage tracking."""
        fill_run_metadata(response, run_id=ctx.state.run_id, conversation_id=ctx.state.conversation_id)
        ctx.state.usage.incr(response.usage)
        if ctx.deps.usage_limits:  # pragma: no branch
            ctx.deps.usage_limits.check_tokens(ctx.state.usage)
        ctx.state.message_history.append(response)

    async def _build_retry_node(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        error: exceptions.ModelRetry,
    ) -> ModelRequestNode[DepsT, NodeRunEndT]:
        """Build a retry ModelRequestNode from a ModelRetry exception.

        Increments the retry counter and creates a new request with a RetryPromptPart.
        """
        ctx.state.consume_output_retry(ctx.deps.max_output_retries, error=error)
        m = _messages.RetryPromptPart(content=error.message)
        retry_node = ModelRequestNode[DepsT, NodeRunEndT](_messages.ModelRequest(parts=[m]))
        self._result = retry_node
        return retry_node

    __repr__ = dataclasses_no_defaults_repr


@dataclasses.dataclass
class CallToolsNode(AgentNode[DepsT, NodeRunEndT]):
    """The node that processes a model response, and decides whether to end the run or make a new request."""

    model_response: _messages.ModelResponse
    tool_call_results: dict[str, DeferredToolResult | Literal['skip']] | None = None
    tool_call_metadata: dict[str, dict[str, Any]] | None = None
    """Metadata for deferred tool calls, keyed by `tool_call_id`."""
    user_prompt: str | Sequence[_messages.UserContent] | None = None
    """Optional user prompt to include alongside tool call results.

    This prompt is only sent to the model when the `model_response` contains tool calls.
    If the `model_response` has final output instead, this user prompt is ignored.
    The user prompt will be appended after all tool return parts in the next model request.
    """

    _events_iterator: AsyncIterator[_messages.HandleResponseEvent] | None = field(default=None, init=False, repr=False)
    _next_node: ModelRequestNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]] | None = field(
        default=None, init=False, repr=False
    )
    _stream_error: BaseException | None = field(default=None, init=False, repr=False)

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> ModelRequestNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]]:
        async with self.stream(ctx):
            pass
        if self._next_node is not None:
            return self._next_node
        # If the stream raised an error that was caught by an external consumer
        # (e.g. UIEventStream.transform_stream), _next_node will not have been set.
        # Re-raise the original error instead of a confusing assertion.
        if self._stream_error is not None:
            raise self._stream_error.with_traceback(self._stream_error.__traceback__)
        raise exceptions.AgentRunError('the stream should set `self._next_node` before it ends')  # pragma: no cover

    @asynccontextmanager
    async def stream(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> AsyncGenerator[AsyncIterator[_messages.AgentStreamEvent]]:
        """Process the model response and yield events for the start and end of each function tool call."""
        stream = _with_event_stream_buffer(self._run_stream(ctx), ctx.state.event_stream_buffer)
        yield stream

        # Run the stream to completion if it was not finished:
        async for _event in stream:
            pass

    async def _run_stream(  # noqa: C901
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> AsyncIterator[_messages.HandleResponseEvent]:
        if self._events_iterator is None:
            # Ensure that the stream is only run once

            output_schema = ctx.deps.output_schema

            async def _run_stream() -> AsyncIterator[_messages.HandleResponseEvent]:  # noqa: C901
                if self.model_response.state == 'suspended':
                    # A suspended turn is not a completed response to handle: its partial parts could
                    # match an output schema and end the run on mid-turn output while the provider's
                    # server-side job keeps running. This is reachable when a consumer detaches a
                    # streamed background run under `agent.iter` and then keeps driving the graph, handing
                    # this node the suspended response. Symmetric with `UserPromptNode`'s suspended guard.
                    raise exceptions.UserError(
                        'Cannot handle a suspended model response as a completed turn. '
                        'Resume it by running the agent with this message history and no new prompt.'
                    )

                is_empty = not self.model_response.parts
                is_thinking_only = not is_empty and all(
                    isinstance(p, _messages.ThinkingPart) for p in self.model_response.parts
                )

                if is_empty or is_thinking_only:
                    # No actionable output was returned by the model.

                    # Don't retry if the token limit was exceeded, possibly during thinking.
                    if self.model_response.finish_reason == 'length':
                        raise exceptions.UnexpectedModelBehavior(
                            f'Model token limit ({ctx.state.last_max_tokens or "provider default"}) exceeded before any response was generated. Increase the `max_tokens` model setting, or simplify the prompt to result in a shorter response that will fit within the limit.'
                        )

                    # Check for content filter on empty response
                    if is_empty and self.model_response.finish_reason == 'content_filter':
                        details = self.model_response.provider_details or {}
                        body = _messages.ModelMessagesTypeAdapter.dump_json([self.model_response]).decode()

                        if reason := details.get('finish_reason'):
                            message = f"Content filter triggered. Finish reason: '{reason}'"
                        elif reason := details.get('block_reason'):
                            message = f"Content filter triggered. Block reason: '{reason}'"
                        elif refusal := details.get('refusal'):
                            message = f'Content filter triggered. Refusal: {refusal!r}'
                        else:  # pragma: no cover
                            message = 'Content filter triggered.'

                        raise exceptions.ContentFilterError(message, body=body)

                    # If the output type allows `None`, an empty or thinking-only response is a valid result:
                    # both signal that the model has no text output to give. Some models emit only thinking
                    # after completing the task via a tool call, and forcing a retry just makes them produce
                    # unnecessary follow-up text.
                    if output_schema.allows_none:
                        run_context = _build_output_run_context(ctx)
                        try:
                            result_data = await _output.run_none_process_hooks(
                                capability=ctx.deps.root_capability,
                                run_context=run_context,
                                schema=output_schema,
                                output_validators=ctx.deps.output_validators,
                            )
                            self._next_node = self._handle_final_result(
                                ctx, result.FinalResult(cast(NodeRunEndT, result_data)), []
                            )
                        except ToolRetryError as e:
                            ctx.state.consume_output_retry(ctx.deps.max_output_retries, error=e)
                            self._next_node = ModelRequestNode[DepsT, NodeRunEndT](
                                _messages.ModelRequest(parts=[e.tool_retry])
                            )
                        return

                    # For empty or thinking-only responses, fall through to the normal retry prompt
                    # below. That prompt is built from the output schema and available tools, so it
                    # tells the model which kinds of output are actually valid (text, tool call,
                    # and/or image) rather than assuming text is always an option.

                text = ''
                compaction_text = ''
                tool_calls: list[_messages.ToolCallPart] = []
                files: list[_messages.BinaryContent] = []

                for part in self.model_response.parts:
                    if isinstance(part, _messages.TextPart):
                        text += part.content
                    elif isinstance(part, _messages.ToolCallPart):
                        tool_calls.append(part)
                    elif isinstance(part, _messages.FilePart):
                        files.append(part.content)
                    elif isinstance(part, _messages.NativeToolCallPart):
                        # Text parts before a native tool call are essentially thoughts,
                        # not part of the final result output, so we reset the accumulated text.
                        # The part itself was already surfaced through `PartStartEvent` / `PartDeltaEvent`.
                        text = ''
                    elif isinstance(part, _messages.NativeToolReturnPart):
                        # Already surfaced through `PartStartEvent` / `PartDeltaEvent`.
                        pass
                    elif isinstance(part, _messages.ThinkingPart):
                        pass
                    elif isinstance(part, _messages.CompactionPart):
                        if part.content:
                            compaction_text += part.content
                    else:
                        assert_never(part)

                # Use compaction content as text fallback when the response has no other
                # actionable text (e.g. Anthropic pause_after_compaction=True)
                if not text and compaction_text:
                    text = compaction_text

                try:
                    # We generally prioritize at least executing tool calls if they are present.
                    # This accounts for cases like Anthropic returns that might contain a text response
                    # and a tool call response, where the text response just indicates the tool call will happen.
                    # The exception is `end_strategy='early'`: if the response also carries a valid non-tool
                    # output (schema-validated text, or an image) alongside plain function tool calls, that
                    # output is already the final result, so `_handle_tool_calls` skips those tools and ends the
                    # run — matching the way `'early'` skips function tools once an output tool call succeeds.
                    # (Output tool calls and deferred tool calls are left to normal processing, so a co-emitted
                    # one still wins/surfaces rather than being preempted by the text.)
                    alternatives: list[str] = []
                    if tool_calls:
                        response_output = (text, files) if ctx.deps.end_strategy == 'early' else None
                        async for event in self._handle_tool_calls(ctx, tool_calls, response_output=response_output):
                            yield event
                        return
                    elif output_schema.toolset:
                        alternatives.append('include your response in a tool call')
                    elif ctx.deps.tool_manager.tools is None or ctx.deps.tool_manager.tools:
                        # tools is None when the tool manager is unprepared (e.g. UserPromptNode
                        # skips to CallToolsNode, bypassing for_run_step); in that case we
                        # default to suggesting tools to be safe
                        alternatives.append('call a tool')

                    if output_schema.allows_image:
                        if image := next((file for file in files if isinstance(file, _messages.BinaryImage)), None):
                            self._next_node = await self._handle_image_response(ctx, image)
                            return
                        alternatives.append('return an image')

                    if text_processor := output_schema.text_processor:
                        if text:
                            self._next_node = await self._handle_text_response(ctx, text, text_processor)
                            return
                        alternatives.insert(0, 'return text')

                    # handle responses with only parts that don't constitute output.
                    # This can happen with models that support thinking mode when they don't provide
                    # actionable output alongside their thinking content. so we tell the model to try again.
                    m = _messages.RetryPromptPart(
                        content=f'Please {" or ".join(alternatives)}.',
                    )
                    raise ToolRetryError(m)
                except ToolRetryError as e:
                    ctx.state.consume_output_retry(ctx.deps.max_output_retries, error=e)
                    self._next_node = ModelRequestNode[DepsT, NodeRunEndT](_messages.ModelRequest(parts=[e.tool_retry]))

            self._events_iterator = _run_stream()

        try:
            async for event in self._events_iterator:
                yield event
        except BaseException as e:
            self._stream_error = e
            raise

    async def _handle_tool_calls(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        tool_calls: list[_messages.ToolCallPart],
        *,
        response_output: tuple[str, list[_messages.BinaryContent]] | None = None,
    ) -> AsyncIterator[_messages.HandleResponseEvent]:
        run_context = build_run_context(ctx)
        run_context = replace(
            run_context,
            retry=ctx.state.output_retries_used,
            max_retries=ctx.deps.tool_manager.default_max_retries,
        )

        # This will raise errors for any tool name conflicts
        ctx.deps.tool_manager = await ctx.deps.tool_manager.for_run_step(run_context)

        # Under `end_strategy='early'`, `response_output` holds the response's `(text, files)`. If it carries a
        # valid non-tool output (schema-validated text, or an image) and every co-emitted tool call is a plain
        # function tool, that output is the final result and the tools are recorded as skipped.
        #
        # We check the tool kinds here (rather than letting `process_tool_calls` sort it out) for two reasons:
        # output and deferred (external/unapproved) tool calls must go through normal processing, and
        # `_process_response_output` runs the output validators, so we only want to invoke it once we know the
        # response output can actually win. `for_run_step` above populated the tool defs used here.
        #
        # The precedence is deliberate: calling an output tool is an explicit "finish the run" signal, and a
        # deferred call may need an external result or human approval — whereas the model's text may just be
        # supporting prose (it doesn't know we might treat that text as final), so text must not silently
        # cancel either. A co-emitted output tool call therefore still produces the final result, and a
        # co-emitted deferred call is still surfaced, rather than being preempted by the text.
        final_result: result.FinalResult[NodeRunEndT] | None = None
        if response_output is not None and all(
            (tool_def := ctx.deps.tool_manager.get_tool_def(call.tool_name)) is None or tool_def.kind == 'function'
            for call in tool_calls
        ):
            text, files = response_output
            final_result = await self._process_response_output(ctx, text=text, files=files)

        output_parts: list[_messages.ModelRequestPart] = []
        output_final_result: deque[result.FinalResult[NodeRunEndT]] = deque(maxlen=1)

        try:
            # When `final_result` is set (schema-validated text or image output already won under
            # `end_strategy='early'`), `process_tool_calls` records the tool calls as skipped rather than
            # executing them.
            async for event in process_tool_calls(
                tool_manager=ctx.deps.tool_manager,
                tool_calls=tool_calls,
                tool_call_results=self.tool_call_results,
                tool_call_metadata=self.tool_call_metadata,
                final_result=final_result,
                ctx=ctx,
                output_parts=output_parts,
                output_final_result=output_final_result,
            ):
                yield event
        except BaseException:
            # Capture the partial tool returns collected so far. State is 'interrupted'
            # so `capture_run_messages` consumers can detect partial state. The user prompt
            # is intentionally omitted: this request was never sent to the model.
            if output_parts:
                ctx.state.message_history.append(
                    _messages.ModelRequest(
                        parts=list(output_parts),
                        run_id=ctx.state.run_id,
                        conversation_id=ctx.state.conversation_id,
                        timestamp=now_utc(),
                        state='interrupted',
                    )
                )
            raise

        if output_final_result:
            final_result = output_final_result[0]
            self._next_node = self._handle_final_result(ctx, final_result, output_parts)
        else:
            # Add user prompt if provided, after all tool return parts
            if self.user_prompt is not None:
                output_parts.append(_messages.UserPromptPart(self.user_prompt))

            self._next_node = ModelRequestNode[DepsT, NodeRunEndT](_messages.ModelRequest(parts=output_parts))

    async def _process_response_output(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        *,
        text: str,
        files: list[_messages.BinaryContent],
    ) -> result.FinalResult[NodeRunEndT] | None:
        """Build the response's non-tool output result (an image, or schema-validated text), or `None`.

        Used under `end_strategy='early'` to decide whether a response that also contains function tool calls
        already carries a final result. Images take precedence over text, matching the order the no-tool-calls
        path handles them in.

        Only text that's validated against a schema can preempt tool calls — i.e. the object output processor
        used by [`NativeOutput`][pydantic_ai.output.NativeOutput],
        [`PromptedOutput`][pydantic_ai.output.PromptedOutput], and a bare structured type (auto mode). There
        the model was told to produce the final output as its text, so text that validates is a deliberate
        final result. Plain, unstructured text output (`str`, [`TextOutput`][pydantic_ai.output.TextOutput], or
        a `str` fallback in a larger schema) accepts *any* text, so the model's preamble — which it emits with
        no signal that we'd treat it as final — must not silently win and skip the tools.

        Returns `None` when the response carries no usable output — e.g. schema-validated text or an image that
        fails validation — so the caller runs the tool calls instead. Unlike a failed output *tool* call, this
        doesn't consume an output retry or surface a retry prompt: running the tools is the correction.
        """
        output_schema = ctx.deps.output_schema
        try:
            if output_schema.allows_image:
                if image := next((file for file in files if isinstance(file, _messages.BinaryImage)), None):
                    return await self._process_image_response(ctx, image)
            if (
                (text_processor := output_schema.text_processor)
                and isinstance(text_processor, _output.BaseObjectOutputProcessor)
                and text
            ):
                return await self._process_text_response(ctx, text, text_processor)
        except ToolRetryError:
            return None
        return None

    async def _handle_text_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        text: str,
        text_processor: _output.BaseOutputProcessor[NodeRunEndT],
    ) -> ModelRequestNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]]:
        return self._handle_final_result(ctx, await self._process_text_response(ctx, text, text_processor), [])

    async def _process_text_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        text: str,
        text_processor: _output.BaseOutputProcessor[NodeRunEndT],
    ) -> result.FinalResult[NodeRunEndT]:
        run_context = _build_output_run_context(ctx)
        schema = ctx.deps.output_schema

        result_data = await _output.run_output_with_hooks(
            text_processor,
            text=text,
            run_context=run_context,
            capability=ctx.deps.root_capability,
            schema=schema,
            output_validators=ctx.deps.output_validators,
        )

        return result.FinalResult(result_data)

    async def _handle_image_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        image: _messages.BinaryImage,
    ) -> ModelRequestNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]]:
        return self._handle_final_result(ctx, await self._process_image_response(ctx, image), [])

    async def _process_image_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        image: _messages.BinaryImage,
    ) -> result.FinalResult[NodeRunEndT]:
        run_context = _build_output_run_context(ctx)
        schema = ctx.deps.output_schema
        result_data = await _output.run_image_process_hooks(
            image,
            capability=ctx.deps.root_capability,
            run_context=run_context,
            schema=schema,
            output_validators=ctx.deps.output_validators,
        )

        return result.FinalResult(result_data)

    def _handle_final_result(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        final_result: result.FinalResult[NodeRunEndT],
        tool_responses: list[_messages.ModelRequestPart],
    ) -> End[result.FinalResult[NodeRunEndT]]:
        messages = ctx.state.message_history

        # To allow this message history to be used in a future run without dangling tool calls,
        # append a new ModelRequest using the tool returns and retries
        if tool_responses:
            messages.append(
                _messages.ModelRequest(
                    parts=tool_responses,
                    run_id=ctx.state.run_id,
                    conversation_id=ctx.state.conversation_id,
                    timestamp=now_utc(),
                )
            )

        return End(final_result)

    __repr__ = dataclasses_no_defaults_repr


@dataclasses.dataclass
class SetFinalResult(AgentNode[DepsT, NodeRunEndT]):
    """A node that immediately ends the graph run after a streaming response produced a final result."""

    final_result: result.FinalResult[NodeRunEndT]

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> End[result.FinalResult[NodeRunEndT]]:
        return End(self.final_result)


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
        _event_stream_buffer=ctx.state.event_stream_buffer,
        _mcp_tool_defs_cache=ctx.state.mcp_tool_defs_cache,
    )
    validation_context = build_validation_context(ctx.deps.validation_context, run_context)
    # Only `validation_context` may be passed to `replace`: it shallow-copies, preserving the shared
    # identity of the mutable members passed by reference above — `loaded_capability_ids`,
    # `discovered_tool_names`, `pending_messages`, `_event_stream_buffer`, `_mcp_tool_defs_cache` (see the
    # invariant on `GraphAgentDeps.loaded_capability_ids`). Never add any of them as a `replace` kwarg — forking the
    # object would silently break in-step capability loads / tool reveals / message enqueues / event delivery /
    # tool-defs caching.
    run_context = replace(run_context, validation_context=validation_context)
    return run_context


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

    loaded_capability_ids = parse_loaded_capabilities(ctx.state.message_history)

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


@dataclasses.dataclass
class _RunMessages:
    messages: list[_messages.ModelMessage]
    used: bool = False


_messages_ctx_var: ContextVar[_RunMessages] = ContextVar('var')


@contextmanager
def capture_run_messages() -> Generator[list[_messages.ModelMessage]]:
    """Context manager to access the messages used in a [`run`][pydantic_ai.agent.AbstractAgent.run], [`run_sync`][pydantic_ai.agent.AbstractAgent.run_sync], or [`run_stream`][pydantic_ai.agent.AbstractAgent.run_stream] call.

    Useful when a run may raise an exception, see [model errors](../agent.md#model-errors) for more information.

    Examples:
    ```python
    from pydantic_ai import Agent, capture_run_messages

    agent = Agent('test')

    with capture_run_messages() as messages:
        try:
            result = agent.run_sync('foobar')
        except Exception:
            print(messages)
            raise
    ```

    !!! note
        If you call `run`, `run_sync`, or `run_stream` more than once within a single `capture_run_messages` context,
        `messages` will represent the messages exchanged during the first call only.

        Contexts can be nested: each `capture_run_messages` context captures the runs for which it is the
        innermost active context. A run started inside a nested context is captured by that nested context,
        not by any enclosing one, so wrapping a nested agent run (e.g. inside a tool) in its own
        `capture_run_messages` lets you inspect that inner run's messages independently.

    If a run is interrupted by an exception or cancellation while streaming a response or executing
    tool calls, the partial [`ModelResponse`][pydantic_ai.messages.ModelResponse] or
    [`ModelRequest`][pydantic_ai.messages.ModelRequest] is still captured here with
    `state='interrupted'`, so consumers can detect and inspect partial state.
    """
    messages: list[_messages.ModelMessage] = []
    # Always push a fresh context so nested `capture_run_messages` contexts each capture their own runs,
    # rather than sharing (and overwriting) the enclosing context's messages.
    token = _messages_ctx_var.set(_RunMessages(messages))
    try:
        yield messages
    finally:
        _messages_ctx_var.reset(token)


def get_captured_run_messages() -> _RunMessages:
    return _messages_ctx_var.get()


def build_agent_graph(
    name: str | None,
    deps_type: type[DepsT],
    output_type: OutputSpec[OutputT],
) -> Graph[
    GraphAgentState,
    GraphAgentDeps[DepsT, OutputT],
    UserPromptNode[DepsT, OutputT],
    result.FinalResult[OutputT],
]:
    """Build the execution [Graph][pydantic_graph.Graph] for a given agent."""
    g = GraphBuilder(
        name=name or 'Agent',
        state_type=GraphAgentState,
        deps_type=GraphAgentDeps[DepsT, OutputT],
        input_type=UserPromptNode[DepsT, OutputT],
        output_type=result.FinalResult[OutputT],
        auto_instrument=False,
    )

    g.add(
        g.edge_from(g.start_node).to(UserPromptNode[DepsT, OutputT]),
        g.node(UserPromptNode[DepsT, OutputT]),
        g.node(ModelRequestNode[DepsT, OutputT]),
        g.node(CallToolsNode[DepsT, OutputT]),
        g.node(
            SetFinalResult[DepsT, OutputT],
        ),
    )
    return g.build(validate_graph_structure=False)


def _narrow_tool_call_parts(
    response: _messages.ModelResponse, model_request_parameters: models.ModelRequestParameters
) -> _messages.ModelResponse:
    """Promote each base `ToolCallPart` in the response to its typed subclass via `ToolDefinition.tool_kind`.

    Lives here rather than in each model adapter so adapter authors emit base
    `ToolCallPart`s freely and the framework owns the typed-identity translation. Streaming
    parts are typed up-front by `ModelResponsePartsManager` via the same lookup; this
    function handles the non-streaming `Model.request()` return path. Either path produces
    the same typed end state — `isinstance(part, ToolSearchCallPart)` is true from the
    moment the call is emitted by the model.
    """
    tool_kind_by_name: dict[str, _messages.ToolPartKind] = {
        td.name: td.tool_kind for td in model_request_parameters.function_tools if td.tool_kind
    }
    if not tool_kind_by_name:
        return response

    changed = False
    new_parts: list[_messages.ModelResponsePart] = []
    for part in response.parts:
        if (
            isinstance(part, _messages.ToolCallPart)
            and part.tool_kind is None
            and (tool_kind := tool_kind_by_name.get(part.tool_name)) is not None
        ):
            promoted = _messages.ToolCallPart.narrow_type(part, tool_kind=tool_kind)
            new_parts.append(promoted)
            changed = True
        else:
            new_parts.append(part)
    return replace(response, parts=new_parts) if changed else response


def _first_run_id_index(messages: list[_messages.ModelMessage], run_id: str) -> int:
    """Return the index of the first message for the current run, or len(messages) if none are found."""
    for index, message in enumerate(messages):
        if message.run_id == run_id:
            return index
    return len(messages)


def _first_new_message_index(
    messages: list[_messages.ModelMessage],
    run_id: str,
    *,
    resumed_request: _messages.ModelRequest | None,
    resumed_request_index: int | None,
) -> int:
    """Return the first index that should be included in `new_messages()`.

    When resuming from `message_history` without a new user prompt, the trailing
    `ModelRequest` is prior context even though the framework stamps it with the current
    `run_id` for adapter bookkeeping, so it must be excluded. A capability or history processor
    can mutate the message list before this runs, so the resumed request is located by trying
    progressively looser fallbacks, each robust to a different kind of mutation:

    1. Object identity (`is`) — survives reordering, insertion, and removal of *other* messages.
    2. Value match (`_is_same_request`) — survives loss of identity (e.g. a deep-copying
       processor) as long as the request's fields are unchanged.
    3. Position (`resumed_request_index`, pinned while preparing the request) — survives an
       in-place rebuild that changes the request's fields (e.g. system-prompt reinjection),
       which defeats both matches above.

    Falling back to the first message carrying the current `run_id` is the last resort. Note the
    layers cover different *single* mutations: a rebuild that also shifts the request's position
    by adding/removing messages after it on the same step defeats all three, and detection falls
    back to `run_id` (which includes the resumed request); this is rarer than any layer's own
    blind spot and no built-in capability triggers it.
    """
    if resumed_request is not None:
        for index, message in enumerate(messages):
            if message is resumed_request:
                return index + 1

        for index in range(len(messages) - 1, -1, -1):
            if _is_same_request(messages[index], resumed_request):
                return index + 1

    if resumed_request_index is not None and 0 <= resumed_request_index < len(messages):
        return resumed_request_index + 1

    return _first_run_id_index(messages, run_id)


def _is_same_request(message: _messages.ModelMessage, request: _messages.ModelRequest) -> bool:
    if not isinstance(message, _messages.ModelRequest):
        return False
    if message is request:  # pragma: no cover
        return True
    # Intentionally excludes `run_id`: the resumed request may not have `run_id` set yet when
    # this comparison is performed.
    return (
        message.parts == request.parts
        and message.timestamp == request.timestamp
        and message.instructions == request.instructions
        and message.metadata == request.metadata
    )


SYNTHESIZED_TOOL_RETURN_METADATA_KEY = 'pydantic_ai_synthesized_tool_return'
"""Metadata key set to `True` on `ToolReturnPart`s synthesized for tool calls that never received a result."""

_SYNTHESIZED_TOOL_RETURN_CONTENT = 'The tool call was interrupted before a result was produced.'


def _dangling_tool_calls_by_response(messages: list[_messages.ModelMessage]) -> dict[int, list[_messages.ToolCallPart]]:
    """Find tool calls that will never receive a result, keyed by the index of their response.

    Matching is an ordered walk: a tool result (`_is_tool_result_part` — a `ToolReturnPart` or
    *tool-bound* `RetryPromptPart`; plain validation feedback doesn't answer a call even if its
    `tool_call_id` collides) only answers a call that is open (produced by an earlier response and
    not already answered) at that point. An out-of-place result — one preceding its call, a
    duplicate, or one reusing the ID of an already-answered call — doesn't mask a genuinely
    dangling call.
    """
    open_calls: dict[str, tuple[int, _messages.ToolCallPart]] = {}
    dangling_by_response: dict[int, list[_messages.ToolCallPart]] = {}
    for index, message in enumerate(messages):
        if isinstance(message, _messages.ModelResponse):
            for part in message.parts:
                if isinstance(part, _messages.ToolCallPart):
                    if shadowed := open_calls.get(part.tool_call_id):
                        # A new call reusing the ID of an open call means the open call can no
                        # longer be answered: any later result answers the new call instead.
                        dangling_by_response.setdefault(shadowed[0], []).append(shadowed[1])
                    open_calls[part.tool_call_id] = (index, part)
        elif isinstance(message, _messages.ModelRequest):  # pragma: no branch
            for part in message.parts:
                if _is_tool_result_part(part):
                    open_calls.pop(part.tool_call_id, None)
    for response_index, call in open_calls.values():
        dangling_by_response.setdefault(response_index, []).append(call)
    return dangling_by_response


def _insert_synthesized_returns(
    request: _messages.ModelRequest, synthesized: list[_messages.ToolReturnPart]
) -> _messages.ModelRequest:
    """Insert synthesized returns after the request's existing tool results (if any).

    They go ahead of user-facing parts — including a plain (non-tool-bound) `RetryPromptPart`,
    which renders as user text — matching where providers expect tool results.
    """
    insert_at = next(
        (
            part_index + 1
            for part_index in range(len(request.parts) - 1, -1, -1)
            if _is_tool_result_part(request.parts[part_index])
        ),
        0,
    )
    return replace(request, parts=[*request.parts[:insert_at], *synthesized, *request.parts[insert_at:]])


def _is_tool_result_part(
    part: _messages.ModelRequestPart | _messages.ModelResponsePart,
) -> TypeGuard[_messages.ToolReturnPart | _messages.RetryPromptPart]:
    """Whether a part is a regular (locally-executed) tool result answering a `ToolCallPart`.

    A `RetryPromptPart` with no `tool_name` is validation feedback rendered as a plain user message,
    not a tool result, so it doesn't need (or have) a matching tool call. `NativeToolReturnPart` (a
    sibling `BaseToolReturnPart` subclass) is intentionally excluded: native/builtin results are
    co-located with their call in one `ModelResponse` and shaped by each model's own serializer, so
    the pipeline leaves them alone.
    """
    return isinstance(part, _messages.ToolReturnPart) or (
        isinstance(part, _messages.RetryPromptPart) and part.tool_name is not None
    )


def _drop_orphaned_tool_results(messages: list[_messages.ModelMessage]) -> list[_messages.ModelMessage]:
    """REMOVE regular tool results whose call is missing.

    A `ToolReturnPart` or tool-bound `RetryPromptPart` (in a `ModelRequest`) whose `tool_call_id`
    never appeared as a regular `ToolCallPart` in any preceding `ModelResponse` is "orphaned".
    Providers reject a tool result without a matching tool call (Anthropic rejects a `tool_result`
    with no `tool_use`; OpenAI Responses raises `No tool call found for function call output`).
    Orphans arise from context eviction dropping the response that made the call, from a result
    placed before its call in a hand-built history, or from adapter round-trips.

    This operates only on regular, locally-executed tool call/result pairing across message
    boundaries. Native/builtin parts (`NativeToolCallPart`/`NativeToolReturnPart`) are left untouched:
    they're produced and resulted by the provider inline and shaped by each model's own serializer,
    and a native result can even arrive in a *later* response (e.g. Anthropic tool search), so the
    core pipeline must not treat them as droppable.

    Removal, not reordering: a result that precedes its call could in principle be moved after it,
    but that crosses message boundaries and reorders content, so the fundamentally-invalid result is
    dropped instead (its now-unanswered call is later synthesized a result by
    `_repair_dangling_tool_calls`). If dropping empties an interior `ModelRequest` the request is
    dropped; if it empties the last message an empty `ModelRequest` is kept so the history still ends
    on a request. Returns the input unchanged when there are no orphans.
    """
    seen_call_ids: set[str] = set()
    repaired: list[_messages.ModelMessage] = []
    changed = False
    for index, message in enumerate(messages):
        for part in message.parts:
            if isinstance(part, _messages.ToolCallPart):
                seen_call_ids.add(part.tool_call_id)
        kept_parts = [
            part
            for part in message.parts
            if not (_is_tool_result_part(part) and part.tool_call_id not in seen_call_ids)
        ]
        if len(kept_parts) == len(message.parts):
            repaired.append(message)
            continue
        changed = True
        if kept_parts or (isinstance(message, _messages.ModelRequest) and index == len(messages) - 1):
            repaired.append(replace(message, parts=kept_parts))
        # else: interior emptied `ModelRequest` — drop the message
    return repaired if changed else messages


def _repair_dangling_tool_calls(
    messages: list[_messages.ModelMessage], *, repair_last_response: bool = False
) -> list[_messages.ModelMessage]:
    """Repair tool calls that are missing a matching result ("dangling" tool calls).

    A run that was cancelled or crashed mid-tool-execution — or a hand-built history — can contain
    `ToolCallPart`s with no matching `ToolReturnPart`/`RetryPromptPart` in a later `ModelRequest`.
    Providers reject histories with dangling tool calls, so before a request is sent, every dangling
    tool call gets a synthesized `ToolReturnPart` — marked with `SYNTHESIZED_TOOL_RETURN_METADATA_KEY`
    in its `metadata` — inserted after the existing tool returns of the immediately following
    `ModelRequest`, or as a new `ModelRequest` when no request follows.

    This includes a call whose args string was cut off mid-stream (unparsable JSON): the call is
    kept verbatim and closed out like any other dangling call, never removed. Malformed args are
    already sendable — serializers degrade them gracefully (see `ToolCallPart.args_as_dict`), as
    the tool-call retry flow relies on — and removing the call would disturb the response's shape,
    e.g. leaving a thinking-only response whose signature was computed over a turn that included
    the call.

    The last `ModelResponse` is only repaired when `repair_last_response` is set: its tool calls
    are the live frontier that run resumption and `deferred_tool_results` may still answer, and a
    trailing unparsable-args call is left for local args validation to turn into a retry prompt.

    Matching is an ordered walk: a result only answers a call that is open (produced by an earlier
    response and not already answered) at that point. An out-of-place result — one preceding its
    call, a duplicate, or one reusing the ID of an already-answered call — doesn't mask a genuinely
    dangling call; such orphaned results themselves are not repaired.

    The repair is deterministic and idempotent: synthesized parts derive their timestamp from the
    response they repair and contain no wall-clock or random data, so repairing the same history
    twice (or on every run) yields the same output and never churns provider prompt-cache prefixes.
    If there is nothing to repair, the input list is returned unchanged. Repair is silent — like
    the other pipeline passes — with the `SYNTHESIZED_TOOL_RETURN_METADATA_KEY` marker as the
    mechanism for inspecting what was synthesized.
    """
    dangling_by_response = _dangling_tool_calls_by_response(messages)
    if not repair_last_response:
        last_response_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], _messages.ModelResponse)
            ),
            None,
        )
        if last_response_index is not None:
            dangling_by_response.pop(last_response_index, None)
    if not dangling_by_response:
        return messages

    repaired: list[_messages.ModelMessage] = []
    synthesized: list[_messages.ToolReturnPart] = []
    for index, message in enumerate(messages):
        if isinstance(message, _messages.ModelResponse):
            if synthesized:
                # The dangling calls of the previous response are followed by another response,
                # so the synthesized returns need a new request in between.
                repaired.append(_messages.ModelRequest(parts=synthesized))
                synthesized = []

            if dangling := dangling_by_response.get(index):
                for call in dangling:
                    synthesized.append(
                        _messages.ToolReturnPart(
                            tool_name=call.tool_name,
                            content=_SYNTHESIZED_TOOL_RETURN_CONTENT,
                            tool_call_id=call.tool_call_id,
                            metadata={SYNTHESIZED_TOOL_RETURN_METADATA_KEY: True},
                            timestamp=message.timestamp,
                            outcome='interrupted',
                        )
                    )
            repaired.append(message)
        elif isinstance(message, _messages.ModelRequest):  # pragma: no branch
            if synthesized:
                message = _insert_synthesized_returns(message, synthesized)
                synthesized = []
            repaired.append(message)

    if synthesized:
        repaired.append(_messages.ModelRequest(parts=synthesized))

    return repaired


def _merge_consecutive_messages(messages: list[_messages.ModelMessage]) -> list[_messages.ModelMessage]:
    """Normalize the history's shape by merging consecutive same-role messages into one.

    Neither adds nor removes content — it only combines adjacent `ModelRequest`s (or adjacent
    synthetic `ModelResponse`s) that providers expect as a single turn, and within a merged request
    hoists tool results ahead of user-facing parts (where providers require them). Runs last, after
    the repair passes have settled call/result pairing, so it operates on a valid history and never
    separates a result from the call it answers.
    """
    clean_messages: list[_messages.ModelMessage] = []
    for message in messages:
        last_message = clean_messages[-1] if len(clean_messages) > 0 else None

        if isinstance(message, _messages.ModelRequest):
            if (
                last_message
                and isinstance(last_message, _messages.ModelRequest)
                # Requests can only be merged if they have the same instructions
                and (
                    not last_message.instructions
                    or not message.instructions
                    or last_message.instructions == message.instructions
                )
                # We intentionally don't block merging when `conversation_id` or `metadata` differ,
                # nor try to preserve them across the merge. These fields are only bookkeeping for
                # callers; they're never part of what gets sent to the model. Refusing to merge on a
                # mismatch would leave two consecutive requests where the model expects one, breaking
                # providers (and provider-side conversation state) that require a single request per
                # turn -- a real regression -- just to preserve fields the model request node never reads.
            ):
                parts = [*last_message.parts, *message.parts]
                parts.sort(
                    # Tool return parts always need to be at the start
                    key=lambda x: 0 if isinstance(x, _messages.ToolReturnPart | _messages.RetryPromptPart) else 1
                )
                merged_message = _messages.ModelRequest(
                    parts=parts,
                    instructions=last_message.instructions or message.instructions,
                    timestamp=message.timestamp or last_message.timestamp,
                )
                clean_messages[-1] = merged_message
            else:
                clean_messages.append(message)
        elif isinstance(message, _messages.ModelResponse):  # pragma: no branch
            if (
                last_message
                and isinstance(last_message, _messages.ModelResponse)
                # Responses can only be merged if they didn't really come from an API
                and last_message.provider_response_id is None
                and last_message.provider_name is None
                and last_message.model_name is None
                and message.provider_response_id is None
                and message.provider_name is None
                and message.model_name is None
            ):
                merged_message = replace(last_message, parts=[*last_message.parts, *message.parts])
                clean_messages[-1] = merged_message
            else:
                clean_messages.append(message)
    return clean_messages


def _clean_message_history(
    messages: list[_messages.ModelMessage], *, repair_last_response: bool = False
) -> list[_messages.ModelMessage]:
    """Make the message history provider-valid and normalize its shape, out of the box.

    An ordered pipeline of pure `list[ModelMessage] -> list[ModelMessage]` passes over regular,
    locally-executed tool call/result pairing across message boundaries. Following the principle
    "massage the history however we can to make the model API accept it, and drop only what's
    fundamentally unsendable", each pass ADDs (synthesizes) or REMOVEs content, never silently
    dropping anything a provider could accept. Native/builtin parts are left entirely untouched:
    they're produced and resulted by the provider inline and shaped by each model's own serializer
    (which handles their own dangling/empty-id cases), and a native result can even arrive in a
    later response, so the core pipeline must not touch them. Ordering matters:

    1. `_drop_orphaned_tool_results` (REMOVE) — first, so an orphaned result can't survive into the
       merge (which would hoist it to the front of a request) and so dropping it can expose a call
       that then needs a synthesized result in pass 2.
    2. `_repair_dangling_tool_calls` (ADD synthesized results) — the matching-graph repair; runs
       before the merge changes message boundaries. Frontier-gated by `repair_last_response` so
       the last response's still-answerable calls are left alone.
    3. `_merge_consecutive_messages` (normalize) — last, once call/result pairing is valid, so it
       never separates a result from its call.
    """
    messages = _drop_orphaned_tool_results(messages)
    messages = _repair_dangling_tool_calls(messages, repair_last_response=repair_last_response)
    messages = _merge_consecutive_messages(messages)
    return messages
