from __future__ import annotations as _annotations

import asyncio
import dataclasses
import hashlib
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Union, cast

from opentelemetry.trace import Tracer
from typing_extensions import TypeGuard, TypeVar, assert_never

from pydantic_ai._function_schema import _takes_ctx as is_takes_ctx  # type: ignore
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai._utils import is_async_callable, run_in_executor
from pydantic_graph import BaseNode, Graph, GraphRunContext
from pydantic_graph.nodes import End, NodeRunEndT

from . import _output, _system_prompt, exceptions, messages as _messages, models, result, usage as _usage
from .exceptions import ToolRetryError
from .output import OutputDataT, OutputSpec
from .settings import ModelSettings, merge_model_settings
from .tools import RunContext, ToolDefinition, ToolKind

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
HistoryProcessor = Union[
    _HistoryProcessorSync,
    _HistoryProcessorAsync,
    _HistoryProcessorSyncWithCtx[DepsT],
    _HistoryProcessorAsyncWithCtx[DepsT],
]
"""A function that processes a list of model messages and returns a list of model messages.

Can optionally accept a `RunContext` as a parameter.
"""


@dataclasses.dataclass
class GraphAgentState:
    """State kept across the execution of the agent graph."""

    message_history: list[_messages.ModelMessage]
    usage: _usage.Usage
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


@dataclasses.dataclass
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

    tool_manager: ToolManager[DepsT]

    tracer: Tracer
    instrumentation_settings: InstrumentationSettings | None = None


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

    instructions: str | None
    instructions_functions: list[_system_prompt.SystemPromptRunner[DepsT]]

    system_prompts: tuple[str, ...]
    system_prompt_functions: list[_system_prompt.SystemPromptRunner[DepsT]]
    system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[DepsT]]

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> ModelRequestNode[DepsT, NodeRunEndT]:
        return ModelRequestNode[DepsT, NodeRunEndT](request=await self._get_first_message(ctx))

    async def _get_first_message(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> _messages.ModelRequest:
        run_context = build_run_context(ctx)
        history, next_message = await self._prepare_messages(
            self.user_prompt, ctx.state.message_history, ctx.deps.get_instructions, run_context
        )
        ctx.state.message_history = history
        run_context.messages = history

        return next_message

    async def _prepare_messages(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None,
        message_history: list[_messages.ModelMessage] | None,
        get_instructions: Callable[[RunContext[DepsT]], Awaitable[str | None]],
        run_context: RunContext[DepsT],
    ) -> tuple[list[_messages.ModelMessage], _messages.ModelRequest]:
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

        parts: list[_messages.ModelRequestPart] = []
        instructions = await get_instructions(run_context)
        if message_history:
            # Shallow copy messages
            messages.extend(message_history)
            # Reevaluate any dynamic system prompt parts
            await self._reevaluate_dynamic_prompts(messages, run_context)
        else:
            parts.extend(await self._sys_parts(run_context))

        if user_prompt is not None:
            parts.append(_messages.UserPromptPart(user_prompt))
        elif (
            len(parts) == 0
            and message_history
            and (last_message := message_history[-1])
            and isinstance(last_message, _messages.ModelRequest)
        ):
            # Drop last message that came from history and reuse its parts
            messages.pop()
            parts.extend(last_message.parts)

        return messages, _messages.ModelRequest(parts, instructions=instructions)

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
    run_context = build_run_context(ctx)
    ctx.deps.tool_manager = await ctx.deps.tool_manager.for_run_step(run_context)

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
        output_mode=output_schema.mode,
        output_tools=output_tools,
        output_object=output_object,
        allow_text_output=allow_text_output,
    )


@dataclasses.dataclass
class ModelRequestNode(AgentNode[DepsT, NodeRunEndT]):
    """The node that makes a request to the model using the last message in state.message_history."""

    request: _messages.ModelRequest

    _result: CallToolsNode[DepsT, NodeRunEndT] | None = field(default=None, repr=False)
    _did_stream: bool = field(default=False, repr=False)

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
        async with self._stream(ctx) as streamed_response:
            agent_stream = result.AgentStream[DepsT, T](
                streamed_response,
                ctx.deps.output_schema,
                ctx.deps.output_validators,
                build_run_context(ctx),
                ctx.deps.usage_limits,
                ctx.deps.tool_manager,
            )
            yield agent_stream
            # In case the user didn't manually consume the full stream, ensure it is fully consumed here,
            # otherwise usage won't be properly counted:
            async for _ in agent_stream:
                pass

    @asynccontextmanager
    async def _stream(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, T]],
    ) -> AsyncIterator[models.StreamedResponse]:
        assert not self._did_stream, 'stream() should only be called once per node'

        model_settings, model_request_parameters = await self._prepare_request(ctx)
        model_request_parameters = ctx.deps.model.customize_request_parameters(model_request_parameters)
        message_history = await _process_message_history(
            ctx.state.message_history, ctx.deps.history_processors, build_run_context(ctx)
        )
        async with ctx.deps.model.request_stream(
            message_history, model_settings, model_request_parameters
        ) as streamed_response:
            self._did_stream = True
            ctx.state.usage.requests += 1
            yield streamed_response
            # In case the user didn't manually consume the full stream, ensure it is fully consumed here,
            # otherwise usage won't be properly counted:
            async for _ in streamed_response:
                pass
        model_response = streamed_response.get()

        self._finish_handling(ctx, model_response)
        assert self._result is not None  # this should be set by the previous line

    async def _make_request(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> CallToolsNode[DepsT, NodeRunEndT]:
        if self._result is not None:
            return self._result  # pragma: no cover

        model_settings, model_request_parameters = await self._prepare_request(ctx)
        model_request_parameters = ctx.deps.model.customize_request_parameters(model_request_parameters)
        message_history = await _process_message_history(
            ctx.state.message_history, ctx.deps.history_processors, build_run_context(ctx)
        )
        model_response = await ctx.deps.model.request(message_history, model_settings, model_request_parameters)
        ctx.state.usage.incr(_usage.Usage())

        return self._finish_handling(ctx, model_response)

    async def _prepare_request(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> tuple[ModelSettings | None, models.ModelRequestParameters]:
        ctx.state.message_history.append(self.request)

        # Check usage
        if ctx.deps.usage_limits:  # pragma: no branch
            ctx.deps.usage_limits.check_before_request(ctx.state.usage)

        # Increment run_step
        ctx.state.run_step += 1

        model_settings = merge_model_settings(ctx.deps.model_settings, None)
        model_request_parameters = await _prepare_request_parameters(ctx)
        return model_settings, model_request_parameters

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

    _events_iterator: AsyncIterator[_messages.HandleResponseEvent] | None = field(default=None, repr=False)
    _next_node: ModelRequestNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]] | None = field(
        default=None, repr=False
    )

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> Union[ModelRequestNode[DepsT, NodeRunEndT], End[result.FinalResult[NodeRunEndT]]]:  # noqa UP007
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

            async def _run_stream() -> AsyncIterator[_messages.HandleResponseEvent]:
                texts: list[str] = []
                tool_calls: list[_messages.ToolCallPart] = []
                for part in self.model_response.parts:
                    if isinstance(part, _messages.TextPart):
                        # ignore empty content for text parts, see #437
                        if part.content:
                            texts.append(part.content)
                    elif isinstance(part, _messages.ToolCallPart):
                        tool_calls.append(part)
                    elif isinstance(part, _messages.ThinkingPart):
                        # We don't need to do anything with thinking parts in this tool-calling node.
                        # We need to handle text parts in case there are no tool calls and/or the desired output comes
                        # from the text, but thinking parts should not directly influence the execution of tools or
                        # determination of the next node of graph execution here.
                        pass
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
                else:
                    # we've got an empty response, this sometimes happens with anthropic (and perhaps other models)
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
        elif deferred_tool_calls := ctx.deps.tool_manager.get_deferred_tool_calls(tool_calls):
            if not ctx.deps.output_schema.allows_deferred_tool_calls:
                raise exceptions.UserError(
                    'A deferred tool call was present, but `DeferredToolCalls` is not among output types. To resolve this, add `DeferredToolCalls` to the list of output types for this agent.'
                )
            final_result = result.FinalResult(cast(NodeRunEndT, deferred_tool_calls), None, None)
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
            return self._handle_final_result(ctx, result.FinalResult(result_data, None, None), [])


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
        kind = tool_def.kind if tool_def else 'unknown'
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
                raise e  # pragma: no cover
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

    for call in calls_to_run:
        yield _messages.FunctionToolCallEvent(call)

    user_parts_by_index: dict[int, list[_messages.UserPromptPart]] = defaultdict(list)

    if calls_to_run:
        # Run all tool tasks in parallel
        tool_parts_by_index: dict[int, _messages.ModelRequestPart] = {}
        with ctx.deps.tracer.start_as_current_span(
            'running tools',
            attributes={
                'tools': [call.tool_name for call in calls_to_run],
                'logfire.msg': f'running {len(calls_to_run)} tool{"" if len(calls_to_run) == 1 else "s"}',
            },
        ):
            tasks = [
                asyncio.create_task(_call_function_tool(tool_manager, call), name=call.tool_name)
                for call in calls_to_run
            ]

            pending = tasks
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    index = tasks.index(task)
                    tool_part, tool_user_parts = task.result()
                    yield _messages.FunctionToolResultEvent(tool_part)

                    tool_parts_by_index[index] = tool_part
                    user_parts_by_index[index] = tool_user_parts

        # We append the results at the end, rather than as they are received, to retain a consistent ordering
        # This is mostly just to simplify testing
        for k in sorted(tool_parts_by_index):
            output_parts.append(tool_parts_by_index[k])

    # Finally, we handle deferred tool calls
    for call in tool_calls_by_kind['deferred']:
        if final_result:
            output_parts.append(
                _messages.ToolReturnPart(
                    tool_name=call.tool_name,
                    content='Tool not executed - a final result was already processed.',
                    tool_call_id=call.tool_call_id,
                )
            )
        else:
            yield _messages.FunctionToolCallEvent(call)

    for k in sorted(user_parts_by_index):
        output_parts.extend(user_parts_by_index[k])

    if final_result:
        output_final_result.append(final_result)


async def _call_function_tool(
    tool_manager: ToolManager[DepsT],
    tool_call: _messages.ToolCallPart,
) -> tuple[_messages.ToolReturnPart | _messages.RetryPromptPart, list[_messages.UserPromptPart]]:
    try:
        tool_result = await tool_manager.handle_call(tool_call)
    except ToolRetryError as e:
        return (e.tool_retry, [])

    tool_part = _messages.ToolReturnPart(
        tool_name=tool_call.tool_name,
        content=tool_result,
        tool_call_id=tool_call.tool_call_id,
    )
    user_parts: list[_messages.UserPromptPart] = []

    if isinstance(tool_result, _messages.ToolReturn):
        if (
            isinstance(tool_result.return_value, _messages.MultiModalContentTypes)
            or isinstance(tool_result.return_value, list)
            and any(
                isinstance(content, _messages.MultiModalContentTypes)
                for content in tool_result.return_value  # type: ignore
            )
        ):
            raise exceptions.UserError(
                f'The `return_value` of tool {tool_call.tool_name!r} contains invalid nested `MultiModalContentTypes` objects. '
                f'Please use `content` instead.'
            )

        tool_part.content = tool_result.return_value  # type: ignore
        tool_part.metadata = tool_result.metadata
        if tool_result.content:
            user_parts.append(
                _messages.UserPromptPart(
                    content=tool_result.content,
                    part_kind='user-prompt',
                )
            )
    else:

        def process_content(content: Any) -> Any:
            if isinstance(content, _messages.ToolReturn):
                raise exceptions.UserError(
                    f'The return value of tool {tool_call.tool_name!r} contains invalid nested `ToolReturn` objects. '
                    f'`ToolReturn` should be used directly.'
                )
            elif isinstance(content, _messages.MultiModalContentTypes):
                if isinstance(content, _messages.BinaryContent):
                    identifier = content.identifier or multi_modal_content_identifier(content.data)
                else:
                    identifier = multi_modal_content_identifier(content.url)

                user_parts.append(
                    _messages.UserPromptPart(
                        content=[f'This is file {identifier}:', content],
                        part_kind='user-prompt',
                    )
                )
                return f'See file {identifier}'

            return content

        if isinstance(tool_result, list):
            contents = cast(list[Any], tool_result)
            tool_part.content = [process_content(content) for content in contents]
        else:
            tool_part.content = process_content(tool_result)

    return (tool_part, user_parts)


@dataclasses.dataclass
class _RunMessages:
    messages: list[_messages.ModelMessage]
    used: bool = False


_messages_ctx_var: ContextVar[_RunMessages] = ContextVar('var')


@contextmanager
def capture_run_messages() -> Iterator[list[_messages.ModelMessage]]:
    """Context manager to access the messages used in a [`run`][pydantic_ai.Agent.run], [`run_sync`][pydantic_ai.Agent.run_sync], or [`run_stream`][pydantic_ai.Agent.run_stream] call.

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
    return messages
