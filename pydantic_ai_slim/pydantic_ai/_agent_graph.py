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
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai._utils import dataclasses_no_defaults_repr, get_union_args, is_async_callable, run_in_executor
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_graph import BaseNode, GraphRunContext
from pydantic_graph.beta import Graph, GraphBuilder
from pydantic_graph.nodes import End, NodeRunEndT

from . import _output, _system_prompt, exceptions, messages as _messages, models, result, usage as _usage
from .exceptions import ToolRetryError
from .output import OutputDataT, OutputSpec
from .settings import ModelSettings
from .tools import (
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
    'build_run_context',
    'capture_run_messages',
    'HistoryProcessor',
)


T = TypeVar('T')
S = TypeVar('S')
NoneType = type(None)
EndStrategy = Literal['early', 'exhaustive']
"""The strategy for handling multiple tool calls when a final result is found.

- `'early'`: Stop processing other tool calls once a final result is found
- `'exhaustive'`: Process all tool calls even after finding a final result
"""
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

    message_history: list[_messages.ModelMessage] = dataclasses.field(default_factory=list)
    usage: _usage.RunUsage = dataclasses.field(default_factory=_usage.RunUsage)
    retries: int = 0
    run_step: int = 0
    run_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))

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
                    max_tokens = (model_settings or {}).get('max_tokens') if model_settings else None
                    raise exceptions.IncompleteToolCall(
                        f'Model token limit ({max_tokens if max_tokens is not None else "provider default"}) exceeded while emitting a tool call, resulting in incomplete arguments. Increase max tokens or simplify tool call arguments to fit within limit.'
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

    model: models.Model
    model_settings: ModelSettings | None
    usage_limits: _usage.UsageLimits
    max_result_retries: int
    end_strategy: EndStrategy
    get_instructions: Callable[[RunContext[DepsT]], Awaitable[str | None]]

    output_schema: _output.OutputSchema[OutputDataT]
    output_validators: list[_output.OutputValidator[DepsT, OutputDataT]]

    history_processors: Sequence[HistoryProcessor[DepsT]]

    builtin_tools: list[AbstractBuiltinTool] = dataclasses.field(repr=False)
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
    instructions_functions: list[_system_prompt.SystemPromptRunner[DepsT]] = dataclasses.field(default_factory=list)

    system_prompts: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    system_prompt_functions: list[_system_prompt.SystemPromptRunner[DepsT]] = dataclasses.field(default_factory=list)
    system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[DepsT]] = dataclasses.field(
        default_factory=dict
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

        if messages and (last_message := messages[-1]):
            if isinstance(last_message, _messages.ModelRequest) and self.user_prompt is None:
                # Drop last message from history and reuse its parts
                messages.pop()
                next_message = _messages.ModelRequest(parts=last_message.parts)

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
                    # Skip ModelRequestNode and go directly to CallToolsNode
                    return CallToolsNode[DepsT, NodeRunEndT](last_message)
                elif last_message.tool_calls:
                    raise exceptions.UserError(
                        'Cannot provide a new user prompt when the message history contains unprocessed tool calls.'
                    )

        # Build the run context after `ctx.deps.prompt` has been updated
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

        next_message.instructions = await ctx.deps.get_instructions(run_context)

        if not messages and not next_message.parts and not next_message.instructions:
            raise exceptions.UserError('No message history, user prompt, or instructions provided')

        return ModelRequestNode[DepsT, NodeRunEndT](request=next_message)

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
        if self.user_prompt is not None:
            raise exceptions.UserError(
                'Cannot provide a new user prompt when the message history contains unprocessed tool calls.'
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
        return CallToolsNode[DepsT, NodeRunEndT](last_model_response, tool_call_results=tool_call_results)

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
                                updated_part_content = await runner.run(run_context)
                                part = _messages.SystemPromptPart(updated_part_content, dynamic_ref=part.dynamic_ref)

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
                messages.append(_messages.SystemPromptPart(prompt, dynamic_ref=sys_prompt_runner.function.__qualname__))
            else:
                messages.append(_messages.SystemPromptPart(prompt))
        return messages

    __repr__ = dataclasses_no_defaults_repr


async def _prepare_request_parameters(
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
) -> models.ModelRequestParameters:
    """Build tools and create an agent model."""
    output_schema = ctx.deps.output_schema

    prompted_output_template = (
        output_schema.template if isinstance(output_schema, _output.PromptedOutputSchema) else None
    )

    function_tools: list[ToolDefinition] = []
    output_tools: list[ToolDefinition] = []
    for tool_def in ctx.deps.tool_manager.tool_defs:
        if tool_def.kind == 'output':
            output_tools.append(tool_def)
        else:
            function_tools.append(tool_def)

    return models.ModelRequestParameters(
        function_tools=function_tools,
        builtin_tools=ctx.deps.builtin_tools,
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

    _result: CallToolsNode[DepsT, NodeRunEndT] | None = field(repr=False, init=False, default=None)
    _did_stream: bool = field(repr=False, init=False, default=False)

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> CallToolsNode[DepsT, NodeRunEndT]:
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
    ) -> AsyncIterator[result.AgentStream[DepsT, T]]:
        assert not self._did_stream, 'stream() should only be called once per node'

        model_settings, model_request_parameters, message_history, run_context = await self._prepare_request(ctx)
        async with ctx.deps.model.request_stream(
            message_history, model_settings, model_request_parameters, run_context
        ) as streamed_response:
            self._did_stream = True
            ctx.state.usage.requests += 1
            agent_stream = result.AgentStream[DepsT, T](
                _raw_stream_response=streamed_response,
                _output_schema=ctx.deps.output_schema,
                _model_request_parameters=model_request_parameters,
                _output_validators=ctx.deps.output_validators,
                _run_ctx=build_run_context(ctx),
                _usage_limits=ctx.deps.usage_limits,
                _tool_manager=ctx.deps.tool_manager,
            )
            yield agent_stream
            # In case the user didn't manually consume the full stream, ensure it is fully consumed here,
            # otherwise usage won't be properly counted:
            async for _ in agent_stream:
                pass

        model_response = streamed_response.get()

        self._finish_handling(ctx, model_response)
        assert self._result is not None  # this should be set by the previous line

    async def _make_request(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> CallToolsNode[DepsT, NodeRunEndT]:
        if self._result is not None:
            return self._result  # pragma: no cover

        model_settings, model_request_parameters, message_history, _ = await self._prepare_request(ctx)
        model_response = await ctx.deps.model.request(message_history, model_settings, model_request_parameters)
        ctx.state.usage.requests += 1

        return self._finish_handling(ctx, model_response)

    async def _prepare_request(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> tuple[ModelSettings | None, models.ModelRequestParameters, list[_messages.ModelMessage], RunContext[DepsT]]:
        self.request.run_id = self.request.run_id or ctx.state.run_id
        ctx.state.message_history.append(self.request)

        ctx.state.run_step += 1

        run_context = build_run_context(ctx)

        # This will raise errors for any tool name conflicts
        ctx.deps.tool_manager = await ctx.deps.tool_manager.for_run_step(run_context)

        original_history = ctx.state.message_history[:]
        message_history = await _process_message_history(original_history, ctx.deps.history_processors, run_context)
        # `ctx.state.message_history` is the same list used by `capture_run_messages`, so we should replace its contents, not the reference
        ctx.state.message_history[:] = message_history
        # Update the new message index to ensure `result.new_messages()` returns the correct messages
        ctx.deps.new_message_index -= len(original_history) - len(message_history)

        # Merge possible consecutive trailing `ModelRequest`s into one, with tool call parts before user parts,
        # but don't store it in the message history on state. This is just for the benefit of model classes that want clear user/assistant boundaries.
        # See `tests/test_tools.py::test_parallel_tool_return_with_deferred` for an example where this is necessary
        message_history = _clean_message_history(message_history)

        model_request_parameters = await _prepare_request_parameters(ctx)

        model_settings = ctx.deps.model_settings
        usage = ctx.state.usage
        if ctx.deps.usage_limits.count_tokens_before_request:
            # Copy to avoid modifying the original usage object with the counted usage
            usage = deepcopy(usage)

            counted_usage = await ctx.deps.model.count_tokens(message_history, model_settings, model_request_parameters)
            usage.incr(counted_usage)

        ctx.deps.usage_limits.check_before_request(usage)

        return model_settings, model_request_parameters, message_history, run_context

    def _finish_handling(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        response: _messages.ModelResponse,
    ) -> CallToolsNode[DepsT, NodeRunEndT]:
        response.run_id = response.run_id or ctx.state.run_id
        # Update usage
        ctx.state.usage.incr(response.usage)
        if ctx.deps.usage_limits:  # pragma: no branch
            ctx.deps.usage_limits.check_tokens(ctx.state.usage)

        # Append the model response to state.message_history
        ctx.state.message_history.append(response)

        # Set the `_result` attribute since we can't use `return` in an async iterator
        self._result = CallToolsNode(response)

        return self._result

    __repr__ = dataclasses_no_defaults_repr


@dataclasses.dataclass
class CallToolsNode(AgentNode[DepsT, NodeRunEndT]):
    """The node that processes a model response, and decides whether to end the run or make a new request."""

    model_response: _messages.ModelResponse
    tool_call_results: dict[str, DeferredToolResult | Literal['skip']] | None = None

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
                                    except ToolRetryError:
                                        # If the text from the preview response was invalid, ignore it.
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
                    else:
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
            instructions = await ctx.deps.get_instructions(run_context)
            self._next_node = ModelRequestNode[DepsT, NodeRunEndT](
                _messages.ModelRequest(parts=output_parts, instructions=instructions)
            )

    async def _handle_text_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        text: str,
        text_processor: _output.BaseOutputProcessor[NodeRunEndT],
    ) -> ModelRequestNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]]:
        run_context = build_run_context(ctx)

        result_data = await text_processor.process(text, run_context)

        for validator in ctx.deps.output_validators:
            result_data = await validator.validate(result_data, run_context)
        return self._handle_final_result(ctx, result.FinalResult(result_data), [])

    async def _handle_image_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        image: _messages.BinaryImage,
    ) -> ModelRequestNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]]:
        result_data = cast(NodeRunEndT, image)
        return self._handle_final_result(ctx, result.FinalResult(result_data), [])

    def _handle_final_result(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        final_result: result.FinalResult[NodeRunEndT],
        tool_responses: list[_messages.ModelRequestPart],
    ) -> End[result.FinalResult[NodeRunEndT]]:
        messages = ctx.state.message_history

        # For backwards compatibility, append a new ModelRequest using the tool returns and retries
        if tool_responses:
            messages.append(_messages.ModelRequest(parts=tool_responses, run_id=ctx.state.run_id))

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


def build_run_context(ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, Any]]) -> RunContext[DepsT]:
    """Build a `RunContext` object from the current agent graph run context."""
    return RunContext[DepsT](
        deps=ctx.deps.user_deps,
        model=ctx.deps.model,
        usage=ctx.state.usage,
        prompt=ctx.deps.prompt,
        messages=ctx.state.message_history,
        tracer=ctx.deps.tracer,
        trace_include_content=ctx.deps.instrumentation_settings is not None
        and ctx.deps.instrumentation_settings.include_content,
        instrumentation_version=ctx.deps.instrumentation_settings.version
        if ctx.deps.instrumentation_settings
        else DEFAULT_INSTRUMENTATION_VERSION,
        run_step=ctx.state.run_step,
        run_id=ctx.state.run_id,
    )


async def process_tool_calls(  # noqa: C901
    tool_manager: ToolManager[DepsT],
    tool_calls: list[_messages.ToolCallPart],
    tool_call_results: dict[str, DeferredToolResult | Literal['skip']] | None,
    final_result: result.FinalResult[NodeRunEndT] | None,
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
    output_parts: list[_messages.ModelRequestPart],
    output_final_result: deque[result.FinalResult[NodeRunEndT]] = deque(maxlen=1),
) -> AsyncIterator[_messages.HandleResponseEvent]:
    """Process function (i.e., non-result) tool calls in parallel.

    Also add stub return parts for any other tools that need it.

    Because async iterators can't have return values, we use `output_parts` and `output_final_result` as output arguments.
    """
    tool_calls_by_kind: dict[ToolKind | Literal['unknown'], list[_messages.ToolCallPart]] = defaultdict(list)
    for call in tool_calls:
        tool_def = tool_manager.get_tool_def(call.tool_name)
        if tool_def:
            kind = tool_def.kind
        else:
            kind = 'unknown'
        tool_calls_by_kind[kind].append(call)

    # First, we handle output tool calls
    for call in tool_calls_by_kind['output']:
        if final_result:
            if final_result.tool_call_id == call.tool_call_id:
                part = _messages.ToolReturnPart(
                    tool_name=call.tool_name,
                    content='Final result processed.',
                    tool_call_id=call.tool_call_id,
                )
            else:
                yield _messages.FunctionToolCallEvent(call)
                part = _messages.ToolReturnPart(
                    tool_name=call.tool_name,
                    content='Output tool not used - a final result was already processed.',
                    tool_call_id=call.tool_call_id,
                )
                yield _messages.FunctionToolResultEvent(part)

            output_parts.append(part)
        else:
            try:
                result_data = await tool_manager.handle_call(call)
            except exceptions.UnexpectedModelBehavior as e:
                ctx.state.increment_retries(
                    ctx.deps.max_result_retries, error=e, model_settings=ctx.deps.model_settings
                )
                raise e  # pragma: lax no cover
            except ToolRetryError as e:
                ctx.state.increment_retries(
                    ctx.deps.max_result_retries, error=e, model_settings=ctx.deps.model_settings
                )
                yield _messages.FunctionToolCallEvent(call)
                output_parts.append(e.tool_retry)
                yield _messages.FunctionToolResultEvent(e.tool_retry)
            else:
                part = _messages.ToolReturnPart(
                    tool_name=call.tool_name,
                    content='Final result processed.',
                    tool_call_id=call.tool_call_id,
                )
                output_parts.append(part)
                final_result = result.FinalResult(result_data, call.tool_name, call.tool_call_id)

    # Then, we handle function tool calls
    calls_to_run: list[_messages.ToolCallPart] = []
    if final_result and ctx.deps.end_strategy == 'early':
        for call in tool_calls_by_kind['function']:
            output_parts.append(
                _messages.ToolReturnPart(
                    tool_name=call.tool_name,
                    content='Tool not executed - a final result was already processed.',
                    tool_call_id=call.tool_call_id,
                )
            )
    else:
        calls_to_run.extend(tool_calls_by_kind['function'])

    # Then, we handle unknown tool calls
    if tool_calls_by_kind['unknown']:
        ctx.state.increment_retries(ctx.deps.max_result_retries, model_settings=ctx.deps.model_settings)
        calls_to_run.extend(tool_calls_by_kind['unknown'])

    calls_to_run_results: dict[str, DeferredToolResult] = {}
    if tool_call_results is not None:
        # Deferred tool calls are "run" as well, by reading their value from the tool call results
        calls_to_run.extend(tool_calls_by_kind['external'])
        calls_to_run.extend(tool_calls_by_kind['unapproved'])

        result_tool_call_ids = set(tool_call_results.keys())
        tool_call_ids_to_run = {call.tool_call_id for call in calls_to_run}
        if tool_call_ids_to_run != result_tool_call_ids:
            raise exceptions.UserError(
                'Tool call results need to be provided for all deferred tool calls. '
                f'Expected: {tool_call_ids_to_run}, got: {result_tool_call_ids}'
            )

        # Filter out calls that were already executed before and should now be skipped
        calls_to_run_results = {call_id: result for call_id, result in tool_call_results.items() if result != 'skip'}
        calls_to_run = [call for call in calls_to_run if call.tool_call_id in calls_to_run_results]

    deferred_calls: dict[Literal['external', 'unapproved'], list[_messages.ToolCallPart]] = defaultdict(list)
    deferred_metadata: dict[str, dict[str, Any]] = {}

    if calls_to_run:
        async for event in _call_tools(
            tool_manager=tool_manager,
            tool_calls=calls_to_run,
            tool_call_results=calls_to_run_results,
            tracer=ctx.deps.tracer,
            usage=ctx.state.usage,
            usage_limits=ctx.deps.usage_limits,
            output_parts=output_parts,
            output_deferred_calls=deferred_calls,
            output_deferred_metadata=deferred_metadata,
        ):
            yield event

    # Finally, we handle deferred tool calls (unless they were already included in the run because results were provided)
    if tool_call_results is None:
        calls = [*tool_calls_by_kind['external'], *tool_calls_by_kind['unapproved']]
        if final_result:
            # If the run was already determined to end on deferred tool calls,
            # we shouldn't insert return parts as the deferred tools will still get a real result.
            if not isinstance(final_result.output, _output.DeferredToolRequests):
                for call in calls:
                    output_parts.append(
                        _messages.ToolReturnPart(
                            tool_name=call.tool_name,
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=call.tool_call_id,
                        )
                    )
        elif calls:
            deferred_calls['external'].extend(tool_calls_by_kind['external'])
            deferred_calls['unapproved'].extend(tool_calls_by_kind['unapproved'])

            for call in calls:
                yield _messages.FunctionToolCallEvent(call)

    if not final_result and deferred_calls:
        if not ctx.deps.output_schema.allows_deferred_tools:
            raise exceptions.UserError(
                'A deferred tool call was present, but `DeferredToolRequests` is not among output types. To resolve this, add `DeferredToolRequests` to the list of output types for this agent.'
            )
        deferred_tool_requests = _output.DeferredToolRequests(
            calls=deferred_calls['external'],
            approvals=deferred_calls['unapproved'],
            metadata=deferred_metadata,
        )

        final_result = result.FinalResult(cast(NodeRunEndT, deferred_tool_requests), None, None)

    if final_result:
        output_final_result.append(final_result)


async def _call_tools(
    tool_manager: ToolManager[DepsT],
    tool_calls: list[_messages.ToolCallPart],
    tool_call_results: dict[str, DeferredToolResult],
    tracer: Tracer,
    usage: _usage.RunUsage,
    usage_limits: _usage.UsageLimits,
    output_parts: list[_messages.ModelRequestPart],
    output_deferred_calls: dict[Literal['external', 'unapproved'], list[_messages.ToolCallPart]],
    output_deferred_metadata: dict[str, dict[str, Any]],
) -> AsyncIterator[_messages.HandleResponseEvent]:
    tool_parts_by_index: dict[int, _messages.ModelRequestPart] = {}
    user_parts_by_index: dict[int, _messages.UserPromptPart] = {}
    deferred_calls_by_index: dict[int, Literal['external', 'unapproved']] = {}
    deferred_metadata_by_index: dict[int, dict[str, Any] | None] = {}

    if usage_limits.tool_calls_limit is not None:
        projected_usage = deepcopy(usage)
        projected_usage.tool_calls += len(tool_calls)
        usage_limits.check_before_tool_call(projected_usage)

    for call in tool_calls:
        yield _messages.FunctionToolCallEvent(call)

    with tracer.start_as_current_span(
        'running tools',
        attributes={
            'tools': [call.tool_name for call in tool_calls],
            'logfire.msg': f'running {len(tool_calls)} tool{"" if len(tool_calls) == 1 else "s"}',
        },
    ):

        async def handle_call_or_result(
            coro_or_task: Awaitable[
                tuple[
                    _messages.ToolReturnPart | _messages.RetryPromptPart, str | Sequence[_messages.UserContent] | None
                ]
            ]
            | Task[
                tuple[
                    _messages.ToolReturnPart | _messages.RetryPromptPart, str | Sequence[_messages.UserContent] | None
                ]
            ],
            index: int,
        ) -> _messages.HandleResponseEvent | None:
            try:
                tool_part, tool_user_content = (
                    (await coro_or_task) if inspect.isawaitable(coro_or_task) else coro_or_task.result()
                )
            except exceptions.CallDeferred as e:
                deferred_calls_by_index[index] = 'external'
                deferred_metadata_by_index[index] = e.metadata
            except exceptions.ApprovalRequired as e:
                deferred_calls_by_index[index] = 'unapproved'
                deferred_metadata_by_index[index] = e.metadata
            else:
                tool_parts_by_index[index] = tool_part
                if tool_user_content:
                    user_parts_by_index[index] = _messages.UserPromptPart(content=tool_user_content)

                return _messages.FunctionToolResultEvent(tool_part, content=tool_user_content)

        if tool_manager.should_call_sequentially(tool_calls):
            for index, call in enumerate(tool_calls):
                if event := await handle_call_or_result(
                    _call_tool(tool_manager, call, tool_call_results.get(call.tool_call_id)),
                    index,
                ):
                    yield event

        else:
            tasks = [
                asyncio.create_task(
                    _call_tool(tool_manager, call, tool_call_results.get(call.tool_call_id)),
                    name=call.tool_name,
                )
                for call in tool_calls
            ]

            pending = tasks
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    index = tasks.index(task)
                    if event := await handle_call_or_result(coro_or_task=task, index=index):
                        yield event

    # We append the results at the end, rather than as they are received, to retain a consistent ordering
    # This is mostly just to simplify testing
    output_parts.extend([tool_parts_by_index[k] for k in sorted(tool_parts_by_index)])
    output_parts.extend([user_parts_by_index[k] for k in sorted(user_parts_by_index)])

    _populate_deferred_calls(
        tool_calls, deferred_calls_by_index, deferred_metadata_by_index, output_deferred_calls, output_deferred_metadata
    )


def _populate_deferred_calls(
    tool_calls: list[_messages.ToolCallPart],
    deferred_calls_by_index: dict[int, Literal['external', 'unapproved']],
    deferred_metadata_by_index: dict[int, dict[str, Any] | None],
    output_deferred_calls: dict[Literal['external', 'unapproved'], list[_messages.ToolCallPart]],
    output_deferred_metadata: dict[str, dict[str, Any]],
) -> None:
    """Populate deferred calls and metadata from indexed mappings."""
    for k in sorted(deferred_calls_by_index):
        call = tool_calls[k]
        output_deferred_calls[deferred_calls_by_index[k]].append(call)
        metadata = deferred_metadata_by_index[k]
        if metadata is not None:
            output_deferred_metadata[call.tool_call_id] = metadata


async def _call_tool(
    tool_manager: ToolManager[DepsT],
    tool_call: _messages.ToolCallPart,
    tool_call_result: DeferredToolResult | None,
) -> tuple[_messages.ToolReturnPart | _messages.RetryPromptPart, str | Sequence[_messages.UserContent] | None]:
    try:
        if tool_call_result is None:
            tool_result = await tool_manager.handle_call(tool_call)
        elif isinstance(tool_call_result, ToolApproved):
            if tool_call_result.override_args is not None:
                tool_call = dataclasses.replace(tool_call, args=tool_call_result.override_args)
            tool_result = await tool_manager.handle_call(tool_call, approved=True)
        elif isinstance(tool_call_result, ToolDenied):
            return _messages.ToolReturnPart(
                tool_name=tool_call.tool_name,
                content=tool_call_result.message,
                tool_call_id=tool_call.tool_call_id,
            ), None
        elif isinstance(tool_call_result, exceptions.ModelRetry):
            m = _messages.RetryPromptPart(
                content=tool_call_result.message,
                tool_name=tool_call.tool_name,
                tool_call_id=tool_call.tool_call_id,
            )
            raise ToolRetryError(m)
        elif isinstance(tool_call_result, _messages.RetryPromptPart):
            tool_call_result.tool_name = tool_call.tool_name
            tool_call_result.tool_call_id = tool_call.tool_call_id
            raise ToolRetryError(tool_call_result)
        else:
            tool_result = tool_call_result
    except ToolRetryError as e:
        return e.tool_retry, None

    if isinstance(tool_result, _messages.ToolReturn):
        tool_return = tool_result
    else:
        result_is_list = isinstance(tool_result, list)
        contents = cast(list[Any], tool_result) if result_is_list else [tool_result]

        return_values: list[Any] = []
        user_contents: list[str | _messages.UserContent] = []
        for content in contents:
            if isinstance(content, _messages.ToolReturn):
                raise exceptions.UserError(
                    f'The return value of tool {tool_call.tool_name!r} contains invalid nested `ToolReturn` objects. '
                    f'`ToolReturn` should be used directly.'
                )
            elif isinstance(content, _messages.MultiModalContent):
                identifier = content.identifier

                return_values.append(f'See file {identifier}')
                user_contents.extend([f'This is file {identifier}:', content])
            else:
                return_values.append(content)

        tool_return = _messages.ToolReturn(
            return_value=return_values[0] if len(return_values) == 1 and not result_is_list else return_values,
            content=user_contents,
        )

    if (
        isinstance(tool_return.return_value, _messages.MultiModalContent)
        or isinstance(tool_return.return_value, list)
        and any(
            isinstance(content, _messages.MultiModalContent)
            for content in tool_return.return_value  # type: ignore
        )
    ):
        raise exceptions.UserError(
            f'The `return_value` of tool {tool_call.tool_name!r} contains invalid nested `MultiModalContent` objects. '
            f'Please use `content` instead.'
        )

    return_part = _messages.ToolReturnPart(
        tool_name=tool_call.tool_name,
        tool_call_id=tool_call.tool_call_id,
        content=tool_return.return_value,  # type: ignore
        metadata=tool_return.metadata,
    )

    return return_part, tool_return.content or None


@dataclasses.dataclass
class _RunMessages:
    messages: list[_messages.ModelMessage]
    used: bool = False


_messages_ctx_var: ContextVar[_RunMessages] = ContextVar('var')


@contextmanager
def capture_run_messages() -> Iterator[list[_messages.ModelMessage]]:
    """Context manager to access the messages used in a [`run`][pydantic_ai.agent.AbstractAgent.run], [`run_sync`][pydantic_ai.agent.AbstractAgent.run_sync], or [`run_stream`][pydantic_ai.agent.AbstractAgent.run_stream] call.

    Useful when a run may raise an exception, see [model errors](../agents.md#model-errors) for more information.

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
    """
    token = None
    messages: list[_messages.ModelMessage] = []

    # Try to reuse existing message context if available
    try:
        messages = _messages_ctx_var.get().messages
    except LookupError:
        # No existing context, create a new one
        token = _messages_ctx_var.set(_RunMessages(messages))

    try:
        yield messages
    finally:
        # Clean up context if we created it
        if token is not None:
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


async def _process_message_history(
    messages: list[_messages.ModelMessage],
    processors: Sequence[HistoryProcessor[DepsT]],
    run_context: RunContext[DepsT],
) -> list[_messages.ModelMessage]:
    """Process message history through a sequence of processors."""
    for processor in processors:
        takes_ctx = is_takes_ctx(processor)

        if is_async_callable(processor):
            if takes_ctx:
                messages = await processor(run_context, messages)
            else:
                async_processor = cast(_HistoryProcessorAsync, processor)
                messages = await async_processor(messages)
        else:
            if takes_ctx:
                sync_processor_with_ctx = cast(_HistoryProcessorSyncWithCtx[DepsT], processor)
                messages = await run_in_executor(sync_processor_with_ctx, run_context, messages)
            else:
                sync_processor = cast(_HistoryProcessorSync, processor)
                messages = await run_in_executor(sync_processor, messages)

    if len(messages) == 0:
        raise exceptions.UserError('Processed history cannot be empty.')

    if not isinstance(messages[-1], _messages.ModelRequest):
        raise exceptions.UserError('Processed history must end with a `ModelRequest`.')

    return messages


def _clean_message_history(messages: list[_messages.ModelMessage]) -> list[_messages.ModelMessage]:
    """Clean the message history by merging consecutive messages of the same type."""
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
            ):
                parts = [*last_message.parts, *message.parts]
                parts.sort(
                    # Tool return parts always need to be at the start
                    key=lambda x: 0 if isinstance(x, _messages.ToolReturnPart | _messages.RetryPromptPart) else 1
                )
                merged_message = _messages.ModelRequest(
                    parts=parts,
                    instructions=last_message.instructions or message.instructions,
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
