from __future__ import annotations as _annotations

import asyncio
import dataclasses
import hashlib
import json
from collections import defaultdict
from collections.abc import AsyncIterator, Awaitable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Union, cast

from opentelemetry.trace import Tracer
from pydantic import ValidationError
from typing_extensions import TypeGuard, TypeVar, assert_never

from pydantic_ai._function_schema import _takes_ctx as is_takes_ctx  # type: ignore
from pydantic_ai._utils import is_async_callable, run_in_executor
from pydantic_ai.toolset import AbstractToolset, CombinedToolset, RunToolset
from pydantic_graph import BaseNode, Graph, GraphRunContext
from pydantic_graph.nodes import End, NodeRunEndT

from . import _output, _system_prompt, exceptions, messages as _messages, models, result, usage as _usage
from .output import OutputDataT, OutputSpec, PendingToolCalls
from .settings import ModelSettings, merge_model_settings
from .tools import RunContext, ToolDefinition, ToolKind

if TYPE_CHECKING:
    pass

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

    def increment_retries(self, max_result_retries: int, error: Exception | None = None) -> None:
        self.retries += 1
        if self.retries > max_result_retries:
            message = f'Exceeded maximum retries ({max_result_retries}) for result validation'
            if error:
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
    output_toolset: RunToolset[DepsT]
    output_validators: list[_output.OutputValidator[DepsT, OutputDataT]]

    history_processors: Sequence[HistoryProcessor[DepsT]]

    toolset: RunToolset[DepsT]

    tracer: Tracer


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
    ctx.deps.toolset = toolset = await ctx.deps.toolset.prepare_for_run(run_context)
    ctx.deps.output_toolset = output_toolset = await ctx.deps.output_toolset.prepare_for_run(run_context)

    # This will raise errors for any name conflicts
    CombinedToolset[DepsT]([output_toolset, toolset])

    output_schema = ctx.deps.output_schema
    output_object = None
    if isinstance(output_schema, _output.ModelStructuredOutputSchema):
        output_object = output_schema.object_def

    # ToolOrTextOutputSchema, ModelStructuredOutputSchema, and PromptedStructuredOutputSchema all inherit from TextOutputSchema
    allow_text_output = isinstance(output_schema, _output.TextOutputSchema)

    return models.ModelRequestParameters(
        function_tools=toolset.tool_defs,
        output_mode=output_schema.mode,
        output_tools=output_toolset.tool_defs,
        output_object=output_object,
        allow_text_output=allow_text_output,
    )


@dataclasses.dataclass
class ModelRequestNode(AgentNode[DepsT, NodeRunEndT]):
    """Make a request to the model using the last message in state.message_history."""

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
    """Process a model response, and decide whether to end the run or make a new request."""

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

    async def _handle_tool_calls(  # noqa: C901
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        tool_calls: list[_messages.ToolCallPart],
    ) -> AsyncIterator[_messages.HandleResponseEvent]:
        run_context = build_run_context(ctx)

        final_result: result.FinalResult[NodeRunEndT] | None = None
        parts: list[_messages.ModelRequestPart] = []

        toolset = await CombinedToolset([ctx.deps.toolset, ctx.deps.output_toolset]).prepare_for_run(run_context)

        unknown_calls: list[_messages.ToolCallPart] = []
        tool_calls_by_kind: dict[ToolKind, list[_messages.ToolCallPart]] = defaultdict(list)
        # TODO: Make Toolset.tool_defs a dict
        tool_defs_by_name: dict[str, ToolDefinition] = {tool_def.name: tool_def for tool_def in toolset.tool_defs}
        for call in tool_calls:
            try:
                tool_def = tool_defs_by_name[call.tool_name]
                tool_calls_by_kind[tool_def.kind].append(call)
            except KeyError:
                unknown_calls.append(call)

        # first, look for the output tool call
        for call in tool_calls_by_kind['output']:
            if final_result:
                part = _messages.ToolReturnPart(
                    tool_name=call.tool_name,
                    content='Output tool not used - a final result was already processed.',
                    tool_call_id=call.tool_call_id,
                )
                parts.append(part)
            else:
                try:
                    result_data = await _call_tool(toolset, call, run_context)
                except _output.ToolRetryError as e:
                    parts.append(e.tool_retry)
                else:
                    part = _messages.ToolReturnPart(
                        tool_name=call.tool_name,
                        content='Final result processed.',
                        tool_call_id=call.tool_call_id,
                    )
                    parts.append(part)
                    final_result = result.FinalResult(result_data, call.tool_name, call.tool_call_id)

        # Then build the other request parts based on end strategy
        if final_result and ctx.deps.end_strategy == 'early':
            for call in tool_calls_by_kind['function']:
                parts.append(
                    _messages.ToolReturnPart(
                        tool_name=call.tool_name,
                        content='Tool not executed - a final result was already processed.',
                        tool_call_id=call.tool_call_id,
                    )
                )
        else:
            async for event in process_function_tools(
                toolset,
                tool_calls_by_kind['function'],
                ctx,
                parts,
            ):
                yield event

        if unknown_calls:
            ctx.state.increment_retries(ctx.deps.max_result_retries)
            async for event in process_function_tools(
                toolset,
                unknown_calls,
                ctx,
                parts,
            ):
                yield event

        pending_calls: list[_messages.ToolCallPart] = []
        for call in tool_calls_by_kind['pending']:
            if final_result:
                parts.append(
                    _messages.ToolReturnPart(
                        tool_name=call.tool_name,
                        content='Tool not executed - a final result was already processed.',
                        tool_call_id=call.tool_call_id,
                    )
                )
            else:
                yield _messages.FunctionToolCallEvent(call)
                pending_calls.append(call)

        if pending_calls:
            if not ctx.deps.output_schema.pending_tool_calls:
                raise exceptions.UserError(
                    'There are pending tool calls but PendingToolCalls is not among output types.'
                )

            pending_tool_names = [call.tool_name for call in pending_calls]
            pending_tool_defs = {
                tool_def.name: tool_def for tool_def in toolset.tool_defs if tool_def.name in pending_tool_names
            }
            output_data = cast(NodeRunEndT, PendingToolCalls(pending_calls, pending_tool_defs))
            final_result = result.FinalResult(output_data)

        if final_result:
            self._next_node = self._handle_final_result(ctx, final_result, parts)
        else:
            instructions = await ctx.deps.get_instructions(run_context)
            self._next_node = ModelRequestNode[DepsT, NodeRunEndT](
                _messages.ModelRequest(parts=parts, instructions=instructions)
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
            if isinstance(output_schema, _output.TextOutputSchema):
                run_context = build_run_context(ctx)
                result_data = await output_schema.process(text, run_context)
            else:
                m = _messages.RetryPromptPart(
                    content='Plain text responses are not permitted, please include your response in a tool call',
                )
                raise _output.ToolRetryError(m)

            result_data = await _validate_output(result_data, ctx, None)
        except _output.ToolRetryError as e:
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
        run_step=ctx.state.run_step,
    )


def multi_modal_content_identifier(identifier: str | bytes) -> str:
    """Generate stable identifier for multi-modal content to help LLM in finding a specific file in tool call responses."""
    if isinstance(identifier, str):
        identifier = identifier.encode('utf-8')
    return hashlib.sha1(identifier).hexdigest()[:6]


async def process_function_tools(
    toolset: AbstractToolset[DepsT],
    tool_calls: list[_messages.ToolCallPart],
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
    output_parts: list[_messages.ModelRequestPart],
) -> AsyncIterator[_messages.HandleResponseEvent]:
    """Process function (i.e., non-result) tool calls in parallel.

    Also add stub return parts for any other tools that need it.

    Because async iterators can't have return values, we use `output_parts` as an output argument.
    """
    run_context = build_run_context(ctx)

    calls_to_run: list[_messages.ToolCallPart] = []
    call_index_to_event_id: dict[int, str] = {}
    for call in tool_calls:
        event = _messages.FunctionToolCallEvent(call)
        yield event
        call_index_to_event_id[len(calls_to_run)] = event.call_id
        calls_to_run.append(call)

    user_parts: list[_messages.UserPromptPart] = []

    # Run all tool tasks in parallel
    results_by_index: dict[int, _messages.ModelRequestPart] = {}
    with ctx.deps.tracer.start_as_current_span(
        'running tools',
        attributes={
            'tools': [call.tool_name for call in calls_to_run],
            'logfire.msg': f'running {len(calls_to_run)} tool{"" if len(calls_to_run) == 1 else "s"}',
        },
    ):
        tasks = [
            asyncio.create_task(_call_function_tool(toolset, call, run_context, ctx.deps.tracer), name=call.tool_name)
            for call in calls_to_run
        ]

        pending = tasks
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                index = tasks.index(task)
                result = task.result()
                yield _messages.FunctionToolResultEvent(result, tool_call_id=call_index_to_event_id[index])

                if isinstance(result, _messages.RetryPromptPart):
                    results_by_index[index] = result
                elif isinstance(result, _messages.ToolReturnPart):

                    def process_content(content: Any) -> Any:
                        if isinstance(content, _messages.MultiModalContentTypes):
                            if isinstance(content, _messages.BinaryContent):
                                identifier = multi_modal_content_identifier(content.data)
                            else:
                                identifier = multi_modal_content_identifier(content.url)

                            user_parts.append(
                                _messages.UserPromptPart(
                                    content=[f'This is file {identifier}:', content],
                                    timestamp=result.timestamp,
                                    part_kind='user-prompt',
                                )
                            )
                            return f'See file {identifier}'
                        else:
                            return content

                    if isinstance(result.content, list):
                        contents = cast(list[Any], result.content)  # type: ignore
                        result.content = [process_content(content) for content in contents]
                    else:
                        result.content = process_content(result.content)

                    results_by_index[index] = result
                else:
                    assert_never(result)

    # We append the results at the end, rather than as they are received, to retain a consistent ordering
    # This is mostly just to simplify testing
    for k in sorted(results_by_index):
        output_parts.append(results_by_index[k])

    output_parts.extend(user_parts)


async def _call_function_tool(
    toolset: AbstractToolset[DepsT],
    tool_call: _messages.ToolCallPart,
    run_context: RunContext[DepsT],
    tracer: Tracer,
) -> _messages.ToolReturnPart | _messages.RetryPromptPart:
    """Run the tool function asynchronously.

    See <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span>.
    """
    span_attributes = {
        'gen_ai.tool.name': tool_call.tool_name,
        # NOTE: this means `gen_ai.tool.call.id` will be included even if it was generated by pydantic-ai
        'gen_ai.tool.call.id': tool_call.tool_call_id,
        'tool_arguments': tool_call.args_as_json_str(),
        'logfire.msg': f'running tool: {tool_call.tool_name}',
        # add the JSON schema so these attributes are formatted nicely in Logfire
        'logfire.json_schema': json.dumps(
            {
                'type': 'object',
                'properties': {
                    'tool_arguments': {'type': 'object'},
                    'gen_ai.tool.name': {},
                    'gen_ai.tool.call.id': {},
                },
            }
        ),
    }

    with tracer.start_as_current_span('running tool', attributes=span_attributes):
        try:
            response_content = await _call_tool(toolset, tool_call, run_context)
        except _output.ToolRetryError as e:
            return e.tool_retry
        else:
            return _messages.ToolReturnPart(
                tool_name=tool_call.tool_name,
                content=response_content,
                tool_call_id=tool_call.tool_call_id,
            )


async def _call_tool(
    toolset: AbstractToolset[DepsT], tool_call: _messages.ToolCallPart, run_context: RunContext[DepsT]
) -> Any:
    run_context = dataclasses.replace(run_context, tool_call_id=tool_call.tool_call_id)

    try:
        args_dict = toolset.validate_tool_args(run_context, tool_call.tool_name, tool_call.args)
        response_content = await toolset.call_tool(run_context, tool_call.tool_name, args_dict)
    except (ValidationError, exceptions.ModelRetry) as e:
        if isinstance(e, ValidationError):
            m = _messages.RetryPromptPart(
                tool_name=tool_call.tool_name,
                content=e.errors(include_url=False, include_context=False),
                tool_call_id=tool_call.tool_call_id,
            )
        else:
            m = _messages.RetryPromptPart(
                tool_name=tool_call.tool_name,
                content=e.message,
                tool_call_id=tool_call.tool_call_id,
            )
        raise _output.ToolRetryError(m)

    return response_content


async def _validate_output(
    result_data: T,
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, T]],
    tool_call: _messages.ToolCallPart | None,
) -> T:
    for validator in ctx.deps.output_validators:
        run_context = build_run_context(ctx)
        result_data = await validator.validate(result_data, tool_call, run_context)
    return result_data


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
    try:
        yield _messages_ctx_var.get().messages
    except LookupError:
        messages: list[_messages.ModelMessage] = []
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
