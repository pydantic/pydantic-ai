from __future__ import annotations as _annotations

import asyncio
import dataclasses
import hashlib
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeGuard, cast

from opentelemetry.trace import Tracer
from typing_extensions import TypeVar, assert_never

from pydantic_ai._function_schema import _takes_ctx as is_takes_ctx  # type: ignore
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai._utils import is_async_callable, run_in_executor
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_graph import BaseNode, Graph, GraphRunContext
from pydantic_graph.nodes import End, NodeRunEndT

from . import _output, _system_prompt, exceptions, messages as _messages, models, result, usage as _usage
from .exceptions import ToolRetryError
from .output import OutputDataT, OutputSpec
from .settings import ModelSettings
from .tools import (
    DeferredToolResult,
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

    message_history: list[_messages.ModelMessage]
    usage: _usage.RunUsage
    retries: int
    run_step: int

    def increment_retries(self, max_result_retries: int, error: BaseException | None = None) -> None:
        self.retries += 1
        if self.retries > max_result_retries:
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
    tool_call_results: dict[str, DeferredToolResult] | None

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

    instructions: str | None
    instructions_functions: list[_system_prompt.SystemPromptRunner[DepsT]]

    system_prompts: tuple[str, ...]
    system_prompt_functions: list[_system_prompt.SystemPromptRunner[DepsT]]
    system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[DepsT]]

    async def run(
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

        # Add message history to the `capture_run_messages` list, which will be empty at this point
        messages.extend(ctx.state.message_history)
        # Use the `capture_run_messages` list as the message history so that new messages are added to it
        ctx.state.message_history = messages

        run_context = build_run_context(ctx)

        parts: list[_messages.ModelRequestPart] = []
        if messages:
            # Reevaluate any dynamic system prompt parts
            await self._reevaluate_dynamic_prompts(messages, run_context)
        else:
            parts.extend(await self._sys_parts(run_context))

        if (tool_call_results := ctx.deps.tool_call_results) is not None:
            if messages and (last_message := messages[-1]) and isinstance(last_message, _messages.ModelRequest):
                # If tool call results were provided, that means the previous run ended on deferred tool calls.
                # That run would typically have ended on a `ModelResponse`, but if it had a mix of deferred tool calls and ones that could already be executed,
                # a `ModelRequest` would already have been added to the history with the preliminary results, even if it wouldn't have been sent to the model yet.
                # So now that we have all of the deferred results, we roll back to the last `ModelResponse` and store the contents of the `ModelRequest` on `deferred_tool_results` to be handled by `CallToolsNode`.
                ctx.deps.tool_call_results = self._update_tool_call_results_from_model_request(
                    tool_call_results, last_message
                )
                messages.pop()

            if not messages:
                raise exceptions.UserError('Tool call results were provided, but the message history is empty.')

        if messages and (last_message := messages[-1]):
            if isinstance(last_message, _messages.ModelRequest) and self.user_prompt is None:
                # Drop last message from history and reuse its parts
                messages.pop()
                parts.extend(last_message.parts)
            elif isinstance(last_message, _messages.ModelResponse):
                call_tools_node = await self._handle_message_history_model_response(ctx, last_message)
                if call_tools_node is not None:
                    return call_tools_node

        if self.user_prompt is not None:
            parts.append(_messages.UserPromptPart(self.user_prompt))

        instructions = await ctx.deps.get_instructions(run_context)
        next_message = _messages.ModelRequest(parts, instructions=instructions)

        return ModelRequestNode[DepsT, NodeRunEndT](request=next_message)

    async def _handle_message_history_model_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        message: _messages.ModelResponse,
    ) -> CallToolsNode[DepsT, NodeRunEndT] | None:
        unprocessed_tool_calls = any(isinstance(part, _messages.ToolCallPart) for part in message.parts)
        if unprocessed_tool_calls:
            if self.user_prompt is not None:
                raise exceptions.UserError(
                    'Cannot provide a new user prompt when the message history contains unprocessed tool calls.'
                )
        else:
            if ctx.deps.tool_call_results is not None:
                raise exceptions.UserError(
                    'Tool call results were provided, but the message history does not contain any unprocessed tool calls.'
                )

        if unprocessed_tool_calls or self.user_prompt is None:
            # `CallToolsNode` requires the tool manager to be prepared for the run step
            # This will raise errors for any tool name conflicts
            run_context = build_run_context(ctx)
            ctx.deps.tool_manager = await ctx.deps.tool_manager.for_run_step(run_context)

            # Skip ModelRequestNode and go directly to CallToolsNode
            return CallToolsNode[DepsT, NodeRunEndT](model_response=message)

    def _update_tool_call_results_from_model_request(
        self, tool_call_results: dict[str, DeferredToolResult], message: _messages.ModelRequest
    ) -> dict[str, DeferredToolResult]:
        last_tool_return: _messages.ToolReturn | None = None
        user_content: list[str | _messages.UserContent] = []
        for part in message.parts:
            if isinstance(part, _messages.ToolReturnPart):
                if part.tool_call_id in tool_call_results:
                    raise exceptions.UserError(
                        f'Tool call {part.tool_call_id!r} was already executed and its result cannot be overridden.'
                    )

                last_tool_return = _messages.ToolReturn(return_value=part.content, metadata=part.metadata)
                tool_call_results[part.tool_call_id] = last_tool_return
            elif isinstance(part, _messages.RetryPromptPart):
                if part.tool_call_id in tool_call_results:
                    raise exceptions.UserError(
                        f'Tool call {part.tool_call_id!r} was already executed and its result cannot be overridden.'
                    )

                tool_call_results[part.tool_call_id] = part
            elif isinstance(part, _messages.UserPromptPart):
                # Tools can return user parts via `ToolReturn.content` or by returning multi-modal content.
                # These go together with a specific `ToolReturnPart`, but we don't have a way to know which,
                # so (below) we just add them to the last one, matching the tool-results-before-user-parts order of the request.
                if isinstance(part.content, str):
                    user_content.append(part.content)
                else:
                    user_content.extend(part.content)
            else:
                raise exceptions.UserError(f'Unexpected message part type: {type(part)}')  # pragma: no cover

        if user_content:
            if last_tool_return is None:
                raise exceptions.UserError(
                    'Tool call results were provided, but the last message in the history was a `ModelRequest` with user parts not tied to preliminary tool results.'
                )
            assert last_tool_return is not None
            last_tool_return.content = user_content

        return tool_call_results

    async def _reevaluate_dynamic_prompts(
        self, messages: list[_messages.ModelMessage], run_context: RunContext[DepsT]
    ) -> None:
        """Reevaluate any `SystemPromptPart` with dynamic_ref in the provided messages by running the associated runner function."""
        # Only proceed if there's at least one dynamic runner.
        if self.system_prompt_dynamic_functions:
            for msg in messages:
                if isinstance(msg, _messages.ModelRequest):
                    for i, part in enumerate(msg.parts):
                        if isinstance(part, _messages.SystemPromptPart) and part.dynamic_ref:
                            # Look up the runner by its ref
                            if runner := self.system_prompt_dynamic_functions.get(  # pragma: lax no cover
                                part.dynamic_ref
                            ):
                                updated_part_content = await runner.run(run_context)
                                msg.parts[i] = _messages.SystemPromptPart(
                                    updated_part_content, dynamic_ref=part.dynamic_ref
                                )

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


async def _prepare_request_parameters(
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
) -> models.ModelRequestParameters:
    """Build tools and create an agent model."""
    output_schema = ctx.deps.output_schema
    output_object = None
    if isinstance(output_schema, _output.NativeOutputSchema):
        output_object = output_schema.object_def

    # ToolOrTextOutputSchema, NativeOutputSchema, and PromptedOutputSchema all inherit from TextOutputSchema
    allow_text_output = isinstance(output_schema, _output.TextOutputSchema)

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
        output_object=output_object,
        allow_text_output=allow_text_output,
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
        ctx.state.message_history.append(self.request)

        ctx.state.run_step += 1

        run_context = build_run_context(ctx)

        # This will raise errors for any tool name conflicts
        ctx.deps.tool_manager = await ctx.deps.tool_manager.for_run_step(run_context)

        message_history = await _process_message_history(ctx.state, ctx.deps.history_processors, run_context)

        model_request_parameters = await _prepare_request_parameters(ctx)
        model_request_parameters = ctx.deps.model.customize_request_parameters(model_request_parameters)

        model_settings = ctx.deps.model_settings
        usage = ctx.state.usage
        if ctx.deps.usage_limits.count_tokens_before_request:
            # Copy to avoid modifying the original usage object with the counted usage
            usage = dataclasses.replace(usage)

            counted_usage = await ctx.deps.model.count_tokens(message_history, model_settings, model_request_parameters)
            usage.incr(counted_usage)

        ctx.deps.usage_limits.check_before_request(usage)

        return model_settings, model_request_parameters, message_history, run_context

    def _finish_handling(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        response: _messages.ModelResponse,
    ) -> CallToolsNode[DepsT, NodeRunEndT]:
        # Update usage
        ctx.state.usage.incr(response.usage)
        if ctx.deps.usage_limits:  # pragma: no branch
            ctx.deps.usage_limits.check_tokens(ctx.state.usage)

        # Append the model response to state.message_history
        ctx.state.message_history.append(response)

        # Set the `_result` attribute since we can't use `return` in an async iterator
        self._result = CallToolsNode(response)

        return self._result


@dataclasses.dataclass
class CallToolsNode(AgentNode[DepsT, NodeRunEndT]):
    """The node that processes a model response, and decides whether to end the run or make a new request."""

    model_response: _messages.ModelResponse

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

            async def _run_stream() -> AsyncIterator[_messages.HandleResponseEvent]:  # noqa: C901
                texts: list[str] = []
                tool_calls: list[_messages.ToolCallPart] = []
                thinking_parts: list[_messages.ThinkingPart] = []

                for part in self.model_response.parts:
                    if isinstance(part, _messages.TextPart):
                        # ignore empty content for text parts, see #437
                        if part.content:
                            texts.append(part.content)
                    elif isinstance(part, _messages.ToolCallPart):
                        tool_calls.append(part)
                    elif isinstance(part, _messages.BuiltinToolCallPart):
                        yield _messages.BuiltinToolCallEvent(part)
                    elif isinstance(part, _messages.BuiltinToolReturnPart):
                        yield _messages.BuiltinToolResultEvent(part)
                    elif isinstance(part, _messages.ThinkingPart):
                        thinking_parts.append(part)
                    else:
                        assert_never(part)

                # At the moment, we prioritize at least executing tool calls if they are present.
                # In the future, we'd consider making this configurable at the agent or run level.
                # This accounts for cases like anthropic returns that might contain a text response
                # and a tool call response, where the text response just indicates the tool call will happen.
                if tool_calls:
                    async for event in self._handle_tool_calls(ctx, tool_calls):
                        yield event
                elif texts:
                    # No events are emitted during the handling of text responses, so we don't need to yield anything
                    self._next_node = await self._handle_text_response(ctx, texts)
                elif thinking_parts:
                    # handle thinking-only responses (responses that contain only ThinkingPart instances)
                    # this can happen with models that support thinking mode when they don't provide
                    # actionable output alongside their thinking content.
                    self._next_node = ModelRequestNode[DepsT, NodeRunEndT](
                        _messages.ModelRequest(
                            parts=[_messages.RetryPromptPart('Responses without text or tool calls are not permitted.')]
                        )
                    )
                else:
                    # we got an empty response with no tool calls, text, or thinking
                    # this sometimes happens with anthropic (and perhaps other models)
                    # when the model has already returned text along side tool calls
                    # in this scenario, if text responses are allowed, we return text from the most recent model
                    # response, if any
                    if isinstance(ctx.deps.output_schema, _output.TextOutputSchema):
                        for message in reversed(ctx.state.message_history):
                            if isinstance(message, _messages.ModelResponse):
                                last_texts = [p.content for p in message.parts if isinstance(p, _messages.TextPart)]
                                if last_texts:
                                    self._next_node = await self._handle_text_response(ctx, last_texts)
                                    return

                    raise exceptions.UnexpectedModelBehavior('Received empty model response')

            self._events_iterator = _run_stream()

        async for event in self._events_iterator:
            yield event

    async def _handle_tool_calls(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        tool_calls: list[_messages.ToolCallPart],
    ) -> AsyncIterator[_messages.HandleResponseEvent]:
        run_context = build_run_context(ctx)

        output_parts: list[_messages.ModelRequestPart] = []
        output_final_result: deque[result.FinalResult[NodeRunEndT]] = deque(maxlen=1)

        async for event in process_function_tools(
            ctx.deps.tool_manager, tool_calls, None, ctx, output_parts, output_final_result
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

    def _handle_final_result(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        final_result: result.FinalResult[NodeRunEndT],
        tool_responses: list[_messages.ModelRequestPart],
    ) -> End[result.FinalResult[NodeRunEndT]]:
        messages = ctx.state.message_history

        # For backwards compatibility, append a new ModelRequest using the tool returns and retries
        if tool_responses:
            messages.append(_messages.ModelRequest(parts=tool_responses))

        return End(final_result)

    async def _handle_text_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        texts: list[str],
    ) -> ModelRequestNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]]:
        output_schema = ctx.deps.output_schema

        text = '\n\n'.join(texts)
        try:
            run_context = build_run_context(ctx)
            if isinstance(output_schema, _output.TextOutputSchema):
                result_data = await output_schema.process(text, run_context)
            else:
                m = _messages.RetryPromptPart(
                    content='Plain text responses are not permitted, please include your response in a tool call',
                )
                raise ToolRetryError(m)

            for validator in ctx.deps.output_validators:
                result_data = await validator.validate(result_data, run_context)
        except ToolRetryError as e:
            ctx.state.increment_retries(ctx.deps.max_result_retries, e)
            return ModelRequestNode[DepsT, NodeRunEndT](_messages.ModelRequest(parts=[e.tool_retry]))
        else:
            return self._handle_final_result(ctx, result.FinalResult(result_data), [])


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
        run_step=ctx.state.run_step,
        tool_call_approved=ctx.state.run_step == 0 and ctx.deps.tool_call_results is not None,
    )


def multi_modal_content_identifier(identifier: str | bytes) -> str:
    """Generate stable identifier for multi-modal content to help LLM in finding a specific file in tool call responses."""
    if isinstance(identifier, str):
        identifier = identifier.encode('utf-8')
    return hashlib.sha1(identifier).hexdigest()[:6]


async def process_function_tools(  # noqa: C901
    tool_manager: ToolManager[DepsT],
    tool_calls: list[_messages.ToolCallPart],
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
                ctx.state.increment_retries(ctx.deps.max_result_retries, e)
                raise e  # pragma: lax no cover
            except ToolRetryError as e:
                ctx.state.increment_retries(ctx.deps.max_result_retries, e)
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
        output_parts.extend(
            [
                _messages.ToolReturnPart(
                    tool_name=call.tool_name,
                    content='Tool not executed - a final result was already processed.',
                    tool_call_id=call.tool_call_id,
                )
                for call in tool_calls_by_kind['function']
            ]
        )
    else:
        calls_to_run.extend(tool_calls_by_kind['function'])

    # Then, we handle unknown tool calls
    if tool_calls_by_kind['unknown']:
        ctx.state.increment_retries(ctx.deps.max_result_retries)
        calls_to_run.extend(tool_calls_by_kind['unknown'])

    deferred_tool_results: dict[str, DeferredToolResult] = {}
    if build_run_context(ctx).tool_call_approved and ctx.deps.tool_call_results is not None:
        deferred_tool_results = ctx.deps.tool_call_results

        # Deferred tool calls are "run" as well, by reading their value from the tool call results
        calls_to_run.extend(tool_calls_by_kind['external'])
        calls_to_run.extend(tool_calls_by_kind['unapproved'])

        result_tool_call_ids = set(deferred_tool_results.keys())
        tool_call_ids_to_run = {call.tool_call_id for call in calls_to_run}
        if tool_call_ids_to_run != result_tool_call_ids:
            raise exceptions.UserError(
                'Tool call results need to be provided for all deferred tool calls. '
                f'Expected: {tool_call_ids_to_run}, got: {result_tool_call_ids}'
            )

    deferred_calls: dict[Literal['external', 'unapproved'], list[_messages.ToolCallPart]] = defaultdict(list)

    if calls_to_run:
        async for event in _call_tools(
            tool_manager,
            calls_to_run,
            deferred_tool_results,
            ctx.deps.tracer,
            output_parts,
            deferred_calls,
        ):
            yield event

    # Finally, we handle deferred tool calls (unless they were already included in the run because results were provided)
    if not deferred_tool_results:
        if final_result:
            for call in [*tool_calls_by_kind['external'], *tool_calls_by_kind['unapproved']]:
                output_parts.append(
                    _messages.ToolReturnPart(
                        tool_name=call.tool_name,
                        content='Tool not executed - a final result was already processed.',
                        tool_call_id=call.tool_call_id,
                    )
                )
        else:
            for call in tool_calls_by_kind['external']:
                deferred_calls['external'].append(call)
                yield _messages.FunctionToolCallEvent(call)

            for call in tool_calls_by_kind['unapproved']:
                deferred_calls['unapproved'].append(call)
                yield _messages.FunctionToolCallEvent(call)

    if not final_result and deferred_calls:
        if not ctx.deps.output_schema.allows_deferred_tools:
            raise exceptions.UserError(
                'A deferred tool call was present, but `DeferredToolRequests` is not among output types. To resolve this, add `DeferredToolRequests` to the list of output types for this agent.'
            )
        deferred_tool_requests = _output.DeferredToolRequests(
            calls=deferred_calls['external'],
            approvals=deferred_calls['unapproved'],
        )

        final_result = result.FinalResult(cast(NodeRunEndT, deferred_tool_requests), None, None)

    if final_result:
        output_final_result.append(final_result)


async def _call_tools(
    tool_manager: ToolManager[DepsT],
    tool_calls: list[_messages.ToolCallPart],
    deferred_tool_results: dict[str, DeferredToolResult],
    tracer: Tracer,
    output_parts: list[_messages.ModelRequestPart],
    output_deferred_calls: dict[Literal['external', 'unapproved'], list[_messages.ToolCallPart]],
) -> AsyncIterator[_messages.HandleResponseEvent]:
    tool_parts_by_index: dict[int, _messages.ModelRequestPart] = {}
    user_parts_by_index: dict[int, _messages.UserPromptPart] = {}
    deferred_calls_by_index: dict[int, Literal['external', 'unapproved']] = {}

    for call in tool_calls:
        yield _messages.FunctionToolCallEvent(call)

    # Run all tool tasks in parallel
    with tracer.start_as_current_span(
        'running tools',
        attributes={
            'tools': [call.tool_name for call in tool_calls],
            'logfire.msg': f'running {len(tool_calls)} tool{"" if len(tool_calls) == 1 else "s"}',
        },
    ):
        tasks = [
            asyncio.create_task(
                _call_tool(tool_manager, call, deferred_tool_results.get(call.tool_call_id)),
                name=call.tool_name,
            )
            for call in tool_calls
        ]

        pending = tasks
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                index = tasks.index(task)
                try:
                    tool_part, tool_user_part = task.result()
                except exceptions.CallDeferred:
                    deferred_calls_by_index[index] = 'external'
                except exceptions.ApprovalRequired:
                    deferred_calls_by_index[index] = 'unapproved'
                else:
                    yield _messages.FunctionToolResultEvent(tool_part)

                    tool_parts_by_index[index] = tool_part
                    if tool_user_part:
                        user_parts_by_index[index] = tool_user_part

    # We append the results at the end, rather than as they are received, to retain a consistent ordering
    # This is mostly just to simplify testing
    for k in sorted(tool_parts_by_index):
        output_parts.append(tool_parts_by_index[k])

    for k in sorted(user_parts_by_index):
        output_parts.append(user_parts_by_index[k])

    for k in sorted(deferred_calls_by_index):
        output_deferred_calls[deferred_calls_by_index[k]].append(tool_calls[k])


async def _call_tool(
    tool_manager: ToolManager[DepsT],
    tool_call: _messages.ToolCallPart,
    tool_call_result: DeferredToolResult | None,
) -> tuple[_messages.ToolReturnPart | _messages.RetryPromptPart, _messages.UserPromptPart | None]:
    try:
        if tool_call_result is None:
            tool_result = await tool_manager.handle_call(tool_call)
        elif isinstance(tool_call_result, ToolApproved):
            if tool_call_result.override_args is not None:
                tool_call = dataclasses.replace(tool_call, args=tool_call_result.override_args)
            tool_result = await tool_manager.handle_call(tool_call)
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
                if isinstance(content, _messages.BinaryContent):
                    identifier = content.identifier or multi_modal_content_identifier(content.data)
                else:
                    identifier = multi_modal_content_identifier(content.url)

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

    user_part: _messages.UserPromptPart | None = None
    if tool_return.content:
        user_part = _messages.UserPromptPart(
            content=tool_return.content,
            part_kind='user-prompt',
        )

    return return_part, user_part


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
) -> Graph[GraphAgentState, GraphAgentDeps[DepsT, result.FinalResult[OutputT]], result.FinalResult[OutputT]]:
    """Build the execution [Graph][pydantic_graph.Graph] for a given agent."""
    nodes = (
        UserPromptNode[DepsT],
        ModelRequestNode[DepsT],
        CallToolsNode[DepsT],
    )
    graph = Graph[GraphAgentState, GraphAgentDeps[DepsT, Any], result.FinalResult[OutputT]](
        nodes=nodes,
        name=name or 'Agent',
        state_type=GraphAgentState,
        run_end_type=result.FinalResult[OutputT],
        auto_instrument=False,
    )
    return graph


async def _process_message_history(
    state: GraphAgentState,
    processors: Sequence[HistoryProcessor[DepsT]],
    run_context: RunContext[DepsT],
) -> list[_messages.ModelMessage]:
    """Process message history through a sequence of processors."""
    messages = state.message_history
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

    # Replaces the message history in the state with the processed messages
    state.message_history = messages
    return messages
