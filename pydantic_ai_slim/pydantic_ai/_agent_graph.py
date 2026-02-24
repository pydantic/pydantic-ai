from __future__ import annotations as _annotations

import asyncio
import dataclasses
import inspect
import uuid
from asyncio import Task
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import field, replace
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeGuard, cast

from opentelemetry.trace import Tracer
from typing_extensions import TypeVar, assert_never

from pydantic_ai._function_schema import _takes_ctx as is_takes_ctx  # type: ignore
from pydantic_ai._instrumentation import DEFAULT_INSTRUMENTATION_VERSION
from pydantic_ai._tool_manager import ToolManager, ValidatedToolCall
from pydantic_ai._utils import dataclasses_no_defaults_repr, get_union_args, is_async_callable, now_utc, run_in_executor
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_graph import BaseNode, GraphRunContext
from pydantic_graph.beta import Graph, GraphBuilder
from pydantic_graph.nodes import End, NodeRunEndT

from . import _output, _system_prompt, exceptions, messages as _messages, models, result, usage as _usage
from ._run_context import set_current_run_context
from .exceptions import ToolRetryError
from .output import OutputDataT, OutputSpec
from .settings import ModelSettings
from .tools import (
    BuiltinToolFunc,
    DeferredToolCallResult,
    DeferredToolResult,
    DeferredToolResults,
    RunContext,
    ToolApproved,
    ToolDefinition,
    ToolDenied,
    ToolKind,
)

if TYPE_CHECKING:
    from .models.instrumented import InstrumentationSettings

__all__ = (
    'GraphAgentState',
    'GraphAgentDeps',
    'UserPromptNode',
    'ModelRequestNode',
    'CallToolsNode',
    'build_agent_graph',
    'build_run_context',
    'capture_run_messages',
    'HistoryProcessor',
    'process_tool_calls',
    'SetFinalResult',
)


T = TypeVar('T')
S = TypeVar('S')
NoneType = type(None)
EndStrategy = Literal['early', 'exhaustive']
DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')

_HistoryProcessorSync = Callable[[list[_messages.ModelMessage]], list[_messages.ModelMessage]]
_HistoryProcessorAsync = Callable[[list[_messages.ModelMessage]], Awaitable[list[_messages.ModelMessage]]]
_HistoryProcessorSyncWithCtx = Callable[[RunContext[DepsT], list[_messages.ModelMessage]], list[_messages.ModelMessage]]
_HistoryProcessorAsyncWithCtx = Callable[
    [RunContext[DepsT], list[_messages.ModelMessage]], Awaitable[list[_messages.ModelMessage]]
]
HistoryProcessor = (
    _HistoryProcessorSync
    | _HistoryProcessorAsync
    | _HistoryProcessorSyncWithCtx[DepsT]
    | _HistoryProcessorAsyncWithCtx[DepsT]
)
"""A function that processes a list of model messages and returns a list of model messages.

Can optionally accept a `RunContext` as a parameter.
"""


@dataclasses.dataclass(kw_only=True)
class GraphAgentState:
    """State kept across the execution of the agent graph."""

    message_history: list[_messages.ModelMessage] = dataclasses.field(default_factory=list[_messages.ModelMessage])
    usage: _usage.RunUsage = dataclasses.field(default_factory=_usage.RunUsage)
    retries: int = 0
    run_step: int = 0
    run_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] | None = None

    def increment_retries(
        self,
        max_result_retries: int,
        error: BaseException | None = None,
        model_settings: ModelSettings | None = None,
    ) -> None:
        self.retries += 1
        if self.retries > max_result_retries:
            if (
                self.message_history
                and isinstance(model_response := self.message_history[-1], _messages.ModelResponse)
                and model_response.finish_reason == 'length'
                and model_response.parts
                and isinstance(tool_call := model_response.parts[-1], _messages.ToolCallPart)
            ):
                try:
                    tool_call.args_as_dict()
                except Exception:
                    max_tokens = model_settings.get('max_tokens') if model_settings else None
                    raise exceptions.IncompleteToolCall(
                        f'Model token limit ({max_tokens or "provider default"}) exceeded while generating a tool call, resulting in incomplete arguments. Increase the `max_tokens` model setting, or simplify the prompt to result in a shorter response that will fit within the limit.'
                    )
            message = f'Exceeded maximum retries ({max_result_retries}) for output validation'
            if error:
                if isinstance(error, exceptions.UnexpectedModelBehavior) and error.__cause__ is not None:
                    error = error.__cause__
                raise exceptions.UnexpectedModelBehavior(message) from error
            else:
                raise exceptions.UnexpectedModelBehavior(message)


@dataclasses.dataclass(kw_only=True)
class GraphAgentDeps(Generic[DepsT, OutputDataT]):
    """Dependencies/config passed to the agent graph."""

    user_deps: DepsT

    prompt: str | Sequence[_messages.UserContent] | None
    new_message_index: int
    resumed_request: _messages.ModelRequest | None

    model: models.Model
    model_settings: ModelSettings | None
    usage_limits: _usage.UsageLimits
    max_result_retries: int
    end_strategy: EndStrategy
    get_instructions: Callable[[RunContext[DepsT]], Awaitable[str | None]]

    output_schema: _output.OutputSchema[OutputDataT]
    output_validators: list[_output.OutputValidator[DepsT, OutputDataT]]
    validation_context: Any | Callable[[RunContext[DepsT]], Any]

    history_processors: Sequence[HistoryProcessor[DepsT]]

    builtin_tools: list[AbstractBuiltinTool | BuiltinToolFunc[DepsT]] = dataclasses.field(repr=False)
    tool_manager: ToolManager[DepsT]

    tracer: Tracer
    instrumentation_settings: InstrumentationSettings | None


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

        next_message: _messages.ModelRequest | None = None
        is_resuming_without_prompt = False

        run_context: RunContext[DepsT] | None = None
        instructions: str | None = None

        if messages and (last_message := messages[-1]):
            if isinstance(last_message, _messages.ModelRequest) and self.user_prompt is None:
                # Drop last message from history and reuse its parts
                messages.pop()
                next_message = _messages.ModelRequest(
                    parts=last_message.parts,
                    run_id=last_message.run_id,
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
                if self.user_prompt is None:
                    run_context = build_run_context(ctx)
                    instructions = await ctx.deps.get_instructions(run_context)
                    if not instructions:
                        # If there's no new prompt or instructions, skip ModelRequestNode and go directly to CallToolsNode
                        return CallToolsNode[DepsT, NodeRunEndT](last_message)
                elif last_message.tool_calls:
                    raise exceptions.UserError(
                        'Cannot provide a new user prompt when the message history contains unprocessed tool calls.'
                    )

        if not run_context:
            run_context = build_run_context(ctx)
            instructions = await ctx.deps.get_instructions(run_context)

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

        next_message.instructions = instructions

        if not messages and not next_message.parts and not next_message.instructions:
            raise exceptions.UserError('No message history, user prompt, or instructions provided')

        return ModelRequestNode[DepsT, NodeRunEndT](
            request=next_message, is_resuming_without_prompt=is_resuming_without_prompt
        )

    async def _handle_deferred_tool_results(  # noqa: C901
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

        tool_call_results: dict[str, DeferredToolResult | Literal['skip']] | None = None
        tool_call_results = {}
        for tool_call_id, approval in deferred_tool_results.approvals.items():
            if approval is True:
                approval = ToolApproved()
            elif approval is False:
                approval = ToolDenied()
            tool_call_results[tool_call_id] = approval

        if calls := deferred_tool_results.calls:
            call_result_types = get_union_args(DeferredToolCallResult)
            for tool_call_id, result in calls.items():
                if not isinstance(result, call_result_types):
                    result = _messages.ToolReturn(result)
                tool_call_results[tool_call_id] = result

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

    async def _sys_parts(self, run_context: RunContext[DepsT]) -> list[_messages.ModelRequestPart]:
        """Build the initial messages for the conversation."""
        messages: list[_messages.ModelRequestPart] = [_messages.SystemPromptPart(p) for p in self.system_prompts]
        for sys_prompt_runner in self.system_prompt_functions:
            prompt = await sys_prompt_runner.run(run_context)
            if sys_prompt_runner.dynamic:
                # To enable dynamic system prompt refs in future runs, use a placeholder string
                messages.append(
                    _messages.SystemPromptPart(prompt or '', dynamic_ref=sys_prompt_runner.function.__qualname__)
                )
            elif prompt:
                # omit empty system prompts
                messages.append(_messages.SystemPromptPart(prompt))
        return messages

    __repr__ = dataclasses_no_defaults_repr


async def _prepare_request_parameters(
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
) -> models.ModelRequestParameters:
    """Build tools and create an agent model."""
    output_schema = ctx.deps.output_schema

    prompted_output_template = (
        output_schema.template if isinstance(output_schema, _output.StructuredTextOutputSchema) else None
    )

    function_tools: list[ToolDefinition] = []
    output_tools: list[ToolDefinition] = []
    for tool_def in ctx.deps.tool_manager.tool_defs:
        if tool_def.kind == 'output':
            output_tools.append(tool_def)
        else:
            function_tools.append(tool_def)

    # resolve dynamic builtin tools
    builtin_tools: list[AbstractBuiltinTool] = []
    if ctx.deps.builtin_tools:
        run_context = build_run_context(ctx)
        for tool in ctx.deps.builtin_tools:
            if isinstance(tool, AbstractBuiltinTool):
                builtin_tools.append(tool)
            else:
                t = tool(run_context)
                if inspect.isawaitable(t):
                    t = await t
                if t is not None:
                    builtin_tools.append(t)

    return models.ModelRequestParameters(
        function_tools=function_tools,
        builtin_tools=builtin_tools,
        output_mode=output_schema.mode,
        output_tools=output_tools,
        output_object=output_schema.object_def,
        prompted_output_template=prompted_output_template,
        allow_text_output=output_schema.allows_text,
        allow_image_output=output_schema.allows_image,
    )


@dataclasses.dataclass
class ModelRequestNode(AgentNode[DepsT, NodeRunEndT]):
    """The node that makes a request to the model using the last message in state.message_history."""

    request: _messages.ModelRequest
    is_resuming_without_prompt: bool = False

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> CallToolsNode[DepsT, NodeRunEndT]:
        """Run the model request and return the next node."""
        return await self._make_request(ctx)

    async def _make_request(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> CallToolsNode[DepsT, NodeRunEndT]:
        """Make the request to the model."""
        model_settings, model_request_parameters, message_history, run_context = await self._prepare_request(ctx)
        with set_current_run_context(run_context):
            model_response = await ctx.deps.model.request(message_history, model_settings, model_request_parameters)
        ctx.state.usage.requests += 1

        return CallToolsNode[DepsT, NodeRunEndT](model_response)

    async def _prepare_request(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> tuple[ModelSettings | None, models.ModelRequestParameters, list[_messages.ModelMessage], RunContext[DepsT]]:
        """Prepare the request parameters."""
        self.request.timestamp = now_utc()
        # Always set run_id if it's not already set, regardless of is_resuming_without_prompt
        self.request.run_id = self.request.run_id or ctx.state.run_id
        ctx.state.message_history.append(self.request)

        ctx.state.run_step += 1

        # Build the run context
        run_context = build_run_context(ctx)

        # Prepare the request parameters
        model_request_parameters = await _prepare_request_parameters(ctx)

        # Get the model settings
        model_settings = ctx.deps.model_settings

        # Process the message history
        message_history = ctx.state.message_history
        for processor in ctx.deps.history_processors:
            if is_takes_ctx(processor):
                message_history = await processor(run_context, message_history)
            else:
                message_history = await processor(message_history)

        return model_settings, model_request_parameters, message_history, run_context


@dataclasses.dataclass
class CallToolsNode(AgentNode[DepsT, NodeRunEndT]):
    """The node that handles tool calls."""

    model_response: _messages.ModelResponse
    tool_call_results: dict[str, DeferredToolResult | Literal['skip']] | None = None
    tool_call_metadata: dict[str, Any] | None = None
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

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> ModelRequestNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]]:
        async with self.stream(ctx):
            pass
        assert self._next_node is not None, 'the stream should set `self._next_node` before it ends'
        return self._next_node

    @asynccontextmanager
    async def stream(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> AsyncIterator[AsyncIterator[_messages.HandleResponseEvent]]:
        """Process the model response and yield events for the start and end of each function tool call."""
        stream = self._run_stream(ctx)
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
                if not self.model_response.parts:
                    # Don't retry if the model returned an empty response because the token limit was exceeded, possibly during thinking.
                    if self.model_response.finish_reason == 'length':
                        model_settings = ctx.deps.model_settings
                        max_tokens = model_settings.get('max_tokens') if model_settings else None
                        raise exceptions.UnexpectedModelBehavior(
                            f'Model token limit ({max_tokens or "provider default"}) exceeded before any response was generated. Increase the `max_tokens` model setting, or simplify the prompt to result in a shorter response that will fit within the limit.'
                        )

                    # Check for content filter on empty response
                    if self.model_response.finish_reason == 'content_filter':
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

                    # we got an empty response.
                    # this sometimes happens with anthropic (and perhaps other models)
                    # when the model has already returned text along side tool calls
                    if text_processor := output_schema.text_processor:  # pragma: no branch
                        # in this scenario, if text responses are allowed, we return text from the most recent model
                        # response, if any
                        for message in reversed(ctx.state.message_history):
                            if isinstance(message, _messages.ModelResponse):
                                text = ''
                                for part in message.parts:
                                    if isinstance(part, _messages.TextPart):
                                        text += part.content
                                    elif isinstance(part, _messages.BuiltinToolCallPart):
                                        # Text parts before a built-in tool call are essentially thoughts,
                                        # not part of the final result output, so we reset the accumulated text
                                        text = ''  # pragma: no cover
                                if text:
                                    try:
                                        self._next_node = await self._handle_text_response(ctx, text, text_processor)
                                        return
                                    except ToolRetryError:  # pragma: no cover
                                        # If the text from the previous response was invalid, ignore it.
                                        pass

                    # Go back to the model request node with an empty request, which means we'll essentially
                    # resubmit the most recent request that resulted in an empty response,
                    # as the empty response and request will not create any items in the API payload,
                    # in the hope the model will return a non-empty response this time.
                    ctx.state.increment_retries(ctx.deps.max_result_retries, model_settings=ctx.deps.model_settings)
                    run_context = build_run_context(ctx)
                    instructions = await ctx.deps.get_instructions(run_context)
                    self._next_node = ModelRequestNode[DepsT, NodeRunEndT](
                        _messages.ModelRequest(parts=[], instructions=instructions)
                    )
                    return

                text = ''
                tool_calls: list[_messages.ToolCallPart] = []
                files: list[_messages.BinaryContent] = []

                for part in self.model_response.parts:
                    if isinstance(part, _messages.TextPart):
                        text += part.content
                    elif isinstance(part, _messages.ToolCallPart):
                        tool_calls.append(part)
                    elif isinstance(part, _messages.FilePart):
                        files.append(part.content)
                    elif isinstance(part, _messages.BuiltinToolCallPart):
                        # Text parts before a built-in tool call are essentially thoughts,
                        # not part of the final result output, so we reset the accumulated text
                        text = ''
                        yield _messages.BuiltinToolCallEvent(part)  # pyright: ignore[reportDeprecated]
                    elif isinstance(part, _messages.BuiltinToolReturnPart):
                        yield _messages.BuiltinToolResultEvent(part)  # pyright: ignore[reportDeprecated]
                    elif isinstance(part, _messages.ThinkingPart):
                        pass
                    else:
                        assert_never(part)

                try:
                    # At the moment, we prioritize at least executing tool calls if they are present.
                    # In the future, we'd consider making this configurable at the agent or run level.
                    # This accounts for cases like anthropic returns that might contain a text response
                    # and a tool call response, where the text response just indicates the tool call will happen.
                    alternatives: list[str] = []
                    if tool_calls:
                        async for event in self._handle_tool_calls(ctx, tool_calls):
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
                    ctx.state.increment_retries(
                        ctx.deps.max_result_retries, error=e, model_settings=ctx.deps.model_settings
                    )
                    run_context = build_run_context(ctx)
                    instructions = await ctx.deps.get_instructions(run_context)
                    self._next_node = ModelRequestNode[DepsT, NodeRunEndT](
                        _messages.ModelRequest(parts=[e.tool_retry], instructions=instructions)
                    )

            self._events_iterator = _run_stream()

        async for event in self._events_iterator:
            yield event

    async def _handle_tool_calls(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        tool_calls: list[_messages.ToolCallPart],
    ) -> AsyncIterator[_messages.HandleResponseEvent]:
        run_context = build_run_context(ctx)

        # This will raise errors for any tool name conflicts
        ctx.deps.tool_manager = await ctx.deps.tool_manager.for_run_step(run_context)

        output_parts: list[_messages.ModelRequestPart] = []
        output_final_result: deque[result.FinalResult[NodeRunEndT]] = deque(maxlen=1)

        async for event in process_tool_calls(
            tool_manager=ctx.deps.tool_manager,
            tool_calls=tool_calls,
            tool_call_results=self.tool_call_results,
            tool_call_metadata=self.tool_call_metadata,
            final_result=None,
            ctx=ctx,
            output_parts=output_parts,
            output_final_result=output_final_result,
        ):
            yield event

        if output_final_result:
            final_result = output_final_result[0]
            self._next_node = self._handle_final_result(ctx, final_result, output_parts)
        else:
            # Add user prompt if provided, after all tool return parts
            if self.user_prompt is not None:
                output_parts.append(_messages.UserPromptPart(self.user_prompt))

            run_context = build_run_context(ctx)
            instructions = await ctx.deps.get_instructions(run_context)
            self._next_node = ModelRequestNode[DepsT, NodeRunEndT](
                _messages.ModelRequest(parts=output_parts, instructions=instructions)
            )

    async def _handle_text_response(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]], text: str, text_processor: Any
    ) -> End[result.FinalResult[NodeRunEndT]]:
        """Handle a text response from the model."""
        run_context = build_run_context(ctx)
        validation_context = ctx.deps.validation_context
        if callable(validation_context):
            validation_context = validation_context(run_context)

        for validator in ctx.deps.output_validators:
            text = await validator(text, run_context, validation_context)

        result_obj = text_processor(text)
        return self._handle_final_result(ctx, result.Result(result_obj), [])

    async def _handle_image_response(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]], image: _messages.BinaryImage
    ) -> End[result.FinalResult[NodeRunEndT]]:
        """Handle an image response from the model."""
        return self._handle_final_result(ctx, result.Result(image), [])

    def _handle_final_result(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]], final_result: result.FinalResult[NodeRunEndT], output_parts: list[_messages.ModelRequestPart]
    ) -> End[result.FinalResult[NodeRunEndT]]:
        """Handle a final result from the model."""
        return End(final_result)


class SetFinalResult(BaseNode[GraphAgentState, GraphAgentDeps[DepsT, OutputDataT], result.FinalResult[OutputDataT]]):
    """A node that sets the final result of the agent run."""

    result: result.FinalResult[OutputDataT]

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, OutputDataT]]
    ) -> End[result.FinalResult[OutputDataT]]:
        """Run the node and return the final result."""
        return End(self.result)


async def process_tool_calls(
    tool_manager: ToolManager[DepsT],
    tool_calls: list[_messages.ToolCallPart],
    tool_call_results: dict[str, DeferredToolResult | Literal['skip']] | None,
    tool_call_metadata: dict[str, Any] | None,
    final_result: result.FinalResult[NodeRunEndT] | None,
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
    output_parts: list[_messages.ModelRequestPart],
    output_final_result: deque[result.FinalResult[NodeRunEndT]],
) -> AsyncIterator[_messages.HandleResponseEvent]:
    """Process tool calls and yield events for each tool call."""
    # This is a placeholder implementation
    # In the actual code, this would process the tool calls and yield events
    pass


def build_run_context(ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, Any]]) -> RunContext[DepsT]:
    """Build a run context from the graph run context."""
    return RunContext(
        user_deps=ctx.deps.user_deps,
        tool_manager=ctx.deps.tool_manager,
        model=ctx.deps.model,
        model_settings=ctx.deps.model_settings,
        run_id=ctx.state.run_id,
        run_step=ctx.state.run_step,
        metadata=ctx.state.metadata,
    )


_captured_run_messages: ContextVar[_CapturedRunMessages | None] = ContextVar('_captured_run_messages', default=None)


@dataclasses.dataclass
class _CapturedRunMessages:
    """A class to capture run messages."""

    messages: list[_messages.ModelMessage]
    used: bool = False


def capture_run_messages(messages: list[_messages.ModelMessage]) -> contextmanager[None]:
    """Capture run messages in a context manager."""

    @contextmanager
    def _capture_run_messages():
        token = _captured_run_messages.set(_CapturedRunMessages(messages))
        try:
            yield
        finally:
            _captured_run_messages.reset(token)

    return _capture_run_messages()


def get_captured_run_messages() -> _CapturedRunMessages:
    """Get the captured run messages."""
    if (ctx := _captured_run_messages.get()) is None:
        raise LookupError('No captured run messages found. Did you forget to use `capture_run_messages`?')
    return ctx


def _clean_message_history(message_history: list[_messages.ModelMessage]) -> list[_messages.ModelMessage]:
    """Clean the message history by removing any messages that are not needed."""
    # This is a placeholder implementation
    # In the actual code, this would clean the message history
    return message_history


def build_agent_graph(
    user_prompt: str | Sequence[_messages.UserContent] | None,
    *,  # Force keyword arguments
    deferred_tool_results: DeferredToolResults | None = None,
    instructions: str | None = None,
    instructions_functions: list[_system_prompt.SystemPromptRunner[DepsT]] | None = None,
    system_prompts: Sequence[str] | None = None,
    system_prompt_functions: list[_system_prompt.SystemPromptRunner[DepsT]] | None = None,
    system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[DepsT]] | None = None,
) -> Graph[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT], result.FinalResult[NodeRunEndT]]:
    """Build an agent graph."""
    # This is a placeholder implementation
    # In the actual code, this would build the agent graph
    pass