from __future__ import annotations as _annotations

import asyncio
import dataclasses
from abc import ABC
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import field
from typing import Any, Generic, Literal, Union, cast

import logfire_api
from typing_extensions import TypeVar, assert_never

from pydantic_graph import BaseNode, Graph, GraphRunContext
from pydantic_graph.nodes import End, NodeRunEndT

from . import (
    _result,
    _system_prompt,
    exceptions,
    messages as _messages,
    models,
    result,
    usage as _usage,
)
from .result import ResultDataT
from .settings import ModelSettings, merge_model_settings
from .tools import (
    RunContext,
    Tool,
    ToolDefinition,
)

_logfire = logfire_api.Logfire(otel_scope='pydantic-ai')

# while waiting for https://github.com/pydantic/logfire/issues/745
try:
    import logfire._internal.stack_info
except ImportError:
    pass
else:
    from pathlib import Path

    logfire._internal.stack_info.NON_USER_CODE_PREFIXES += (str(Path(__file__).parent.absolute()),)

T = TypeVar('T')
NoneType = type(None)
EndStrategy = Literal['early', 'exhaustive']
"""The strategy for handling multiple tool calls when a final result is found.

- `'early'`: Stop processing other tool calls once a final result is found
- `'exhaustive'`: Process all tool calls even after finding a final result
"""
DepsT = TypeVar('DepsT')
ResultT = TypeVar('ResultT')


@dataclasses.dataclass
class MarkFinalResult(Generic[ResultDataT]):
    """Marker class to indicate that the result is the final result.

    This allows us to use `isinstance`, which wouldn't be possible if we were returning `ResultDataT` directly.

    It also avoids problems in the case where the result type is itself `None`, but is set.
    """

    data: ResultDataT
    """The final result data."""
    tool_name: str | None
    """Name of the final result tool, None if the result is a string."""


@dataclasses.dataclass
class GraphAgentState:
    """State kept across the execution of the agent graph."""

    message_history: list[_messages.ModelMessage]
    usage: _usage.Usage
    retries: int
    run_step: int

    def increment_retries(self, max_result_retries: int) -> None:
        self.retries += 1
        if self.retries > max_result_retries:
            raise exceptions.UnexpectedModelBehavior(
                f'Exceeded maximum retries ({max_result_retries}) for result validation'
            )


@dataclasses.dataclass
class GraphAgentDeps(Generic[DepsT, ResultDataT]):
    """Dependencies/config passed to the agent graph."""

    user_deps: DepsT

    prompt: str
    new_message_index: int

    model: models.Model
    model_settings: ModelSettings | None
    usage_limits: _usage.UsageLimits
    max_result_retries: int
    end_strategy: EndStrategy

    result_schema: _result.ResultSchema[ResultDataT] | None
    result_tools: list[ToolDefinition]
    result_validators: list[_result.ResultValidator[DepsT, ResultDataT]]

    function_tools: dict[str, Tool[DepsT]] = dataclasses.field(repr=False)

    run_span: logfire_api.LogfireSpan


@dataclasses.dataclass
class BaseUserPromptNode(BaseNode[GraphAgentState, GraphAgentDeps[DepsT, Any], NodeRunEndT], ABC):
    user_prompt: str

    system_prompts: tuple[str, ...]
    system_prompt_functions: list[_system_prompt.SystemPromptRunner[DepsT]]
    system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[DepsT]]

    async def _get_first_message(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, Any]]
    ) -> _messages.ModelRequest:
        run_context = _build_run_context(ctx)
        history, next_message = await self._prepare_messages(self.user_prompt, ctx.state.message_history, run_context)
        ctx.state.message_history = history
        run_context.messages = history

        # TODO: We need to make it so that function_tools are not shared between runs
        #   See comment on the current_retry field of `Tool` for more details.
        for tool in ctx.deps.function_tools.values():
            tool.current_retry = 0
        return next_message

    async def _prepare_messages(
        self, user_prompt: str, message_history: list[_messages.ModelMessage] | None, run_context: RunContext[DepsT]
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

        if message_history:
            # Shallow copy messages
            messages.extend(message_history)
            # Reevaluate any dynamic system prompt parts
            await self._reevaluate_dynamic_prompts(messages, run_context)
            return messages, _messages.ModelRequest([_messages.UserPromptPart(user_prompt)])
        else:
            parts = await self._sys_parts(run_context)
            parts.append(_messages.UserPromptPart(user_prompt))
            return messages, _messages.ModelRequest(parts)

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
                            if runner := self.system_prompt_dynamic_functions.get(part.dynamic_ref):
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


@dataclasses.dataclass
class UserPromptNode(BaseUserPromptNode[DepsT, NodeRunEndT]):
    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, Any]]
    ) -> ModelRequestNode[DepsT, NodeRunEndT]:
        return ModelRequestNode[DepsT, NodeRunEndT](request=await self._get_first_message(ctx))


@dataclasses.dataclass
class StreamUserPromptNode(BaseUserPromptNode[DepsT, NodeRunEndT]):
    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, Any]]
    ) -> StreamModelRequestNode[DepsT, NodeRunEndT]:
        return StreamModelRequestNode[DepsT, NodeRunEndT](request=await self._get_first_message(ctx))


async def _prepare_request_parameters(
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
) -> models.ModelRequestParameters:
    """Build tools and create an agent model."""
    function_tool_defs: list[ToolDefinition] = []

    run_context = _build_run_context(ctx)

    async def add_tool(tool: Tool[DepsT]) -> None:
        ctx = run_context.replace_with(retry=tool.current_retry, tool_name=tool.name)
        if tool_def := await tool.prepare_tool_def(ctx):
            function_tool_defs.append(tool_def)

    await asyncio.gather(*map(add_tool, ctx.deps.function_tools.values()))

    result_schema = ctx.deps.result_schema
    return models.ModelRequestParameters(
        function_tools=function_tool_defs,
        allow_text_result=_allow_text_result(result_schema),
        result_tools=result_schema.tool_defs() if result_schema is not None else [],
    )


@dataclasses.dataclass
class ModelRequestNode(BaseNode[GraphAgentState, GraphAgentDeps[DepsT, Any], NodeRunEndT]):
    """Make a request to the model using the last message in state.message_history."""

    request: _messages.ModelRequest

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> HandleResponseNode[DepsT, NodeRunEndT]:
        ctx.state.message_history.append(self.request)

        # Check usage
        if ctx.deps.usage_limits:
            ctx.deps.usage_limits.check_before_request(ctx.state.usage)

        # Increment run_step
        ctx.state.run_step += 1

        with _logfire.span('preparing model request params {run_step=}', run_step=ctx.state.run_step):
            model_request_parameters = await _prepare_request_parameters(ctx)

        # Actually make the model request
        model_settings = merge_model_settings(ctx.deps.model_settings, None)
        with _logfire.span('model request') as span:
            model_response, request_usage = await ctx.deps.model.request(
                ctx.state.message_history, model_settings, model_request_parameters
            )
            span.set_attribute('response', model_response)
            span.set_attribute('usage', request_usage)

        # Update usage
        ctx.state.usage.incr(request_usage, requests=1)
        if ctx.deps.usage_limits:
            ctx.deps.usage_limits.check_tokens(ctx.state.usage)

        # Append the model response to state.message_history
        ctx.state.message_history.append(model_response)
        return HandleResponseNode(model_response)


@dataclasses.dataclass
class HandleResponseNode(BaseNode[GraphAgentState, GraphAgentDeps[DepsT, Any], NodeRunEndT]):
    """Process e response from a model, decide whether to end the run or make a new request."""

    model_response: _messages.ModelResponse

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> Union[ModelRequestNode[DepsT, NodeRunEndT], FinalResultNode[DepsT, NodeRunEndT]]:  # noqa UP007
        with _logfire.span('handle model response', run_step=ctx.state.run_step) as handle_span:
            texts: list[str] = []
            tool_calls: list[_messages.ToolCallPart] = []
            for part in self.model_response.parts:
                if isinstance(part, _messages.TextPart):
                    # ignore empty content for text parts, see #437
                    if part.content:
                        texts.append(part.content)
                elif isinstance(part, _messages.ToolCallPart):
                    tool_calls.append(part)
                else:
                    assert_never(part)

            # At the moment, we prioritize at least executing tool calls if they are present.
            # In the future, we'd consider making this configurable at the agent or run level.
            # This accounts for cases like anthropic returns that might contain a text response
            # and a tool call response, where the text response just indicates the tool call will happen.
            if tool_calls:
                return await self._handle_tool_calls_response(ctx, tool_calls, handle_span)
            elif texts:
                return await self._handle_text_response(ctx, texts, handle_span)
            else:
                raise exceptions.UnexpectedModelBehavior('Received empty model response')

    async def _handle_tool_calls_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        tool_calls: list[_messages.ToolCallPart],
        handle_span: logfire_api.LogfireSpan,
    ):
        result_schema = ctx.deps.result_schema

        # first look for the result tool call
        final_result: MarkFinalResult[NodeRunEndT] | None = None
        parts: list[_messages.ModelRequestPart] = []
        if result_schema is not None:
            if match := result_schema.find_tool(tool_calls):
                call, result_tool = match
                try:
                    result_data = result_tool.validate(call)
                    result_data = await _validate_result(result_data, ctx, call)
                except _result.ToolRetryError as e:
                    # TODO: Should only increment retry stuff once per node execution, not for each tool call
                    #   Also, should increment the tool-specific retry count rather than the run retry count
                    ctx.state.increment_retries(ctx.deps.max_result_retries)
                    parts.append(e.tool_retry)
                else:
                    final_result = MarkFinalResult(result_data, call.tool_name)

        # Then build the other request parts based on end strategy
        tool_responses = await _process_function_tools(tool_calls, final_result and final_result.tool_name, ctx)

        if final_result:
            handle_span.set_attribute('result', final_result.data)
            handle_span.message = 'handle model response -> final result'
            return FinalResultNode[DepsT, NodeRunEndT](final_result, tool_responses)
        else:
            if tool_responses:
                handle_span.set_attribute('tool_responses', tool_responses)
                tool_responses_str = ' '.join(r.part_kind for r in tool_responses)
                handle_span.message = f'handle model response -> {tool_responses_str}'
                parts.extend(tool_responses)
            return ModelRequestNode[DepsT, NodeRunEndT](_messages.ModelRequest(parts=parts))

    async def _handle_text_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        texts: list[str],
        handle_span: logfire_api.LogfireSpan,
    ):
        result_schema = ctx.deps.result_schema

        text = '\n\n'.join(texts)
        if _allow_text_result(result_schema):
            result_data_input = cast(NodeRunEndT, text)
            try:
                result_data = await _validate_result(result_data_input, ctx, None)
            except _result.ToolRetryError as e:
                ctx.state.increment_retries(ctx.deps.max_result_retries)
                return ModelRequestNode[DepsT, NodeRunEndT](_messages.ModelRequest(parts=[e.tool_retry]))
            else:
                handle_span.set_attribute('result', result_data)
                handle_span.message = 'handle model response -> final result'
                return FinalResultNode[DepsT, NodeRunEndT](MarkFinalResult(result_data, None))
        else:
            ctx.state.increment_retries(ctx.deps.max_result_retries)
            return ModelRequestNode[DepsT, NodeRunEndT](
                _messages.ModelRequest(
                    parts=[
                        _messages.RetryPromptPart(
                            content='Plain text responses are not permitted, please call one of the functions instead.',
                        )
                    ]
                )
            )


@dataclasses.dataclass
class StreamModelRequestNode(BaseNode[GraphAgentState, GraphAgentDeps[DepsT, Any], NodeRunEndT]):
    """Make a request to the model using the last message in state.message_history (or a specified request)."""

    request: _messages.ModelRequest
    _result: StreamModelRequestNode[DepsT, NodeRunEndT] | End[result.StreamedRunResult[DepsT, NodeRunEndT]] | None = (
        field(default=None, repr=False)
    )

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> Union[StreamModelRequestNode[DepsT, NodeRunEndT], End[result.StreamedRunResult[DepsT, NodeRunEndT]]]:  # noqa UP007
        if self._result is not None:
            return self._result

        async with self.run_to_result(ctx) as final_node:
            return final_node

    @asynccontextmanager
    async def run_to_result(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> AsyncIterator[StreamModelRequestNode[DepsT, NodeRunEndT] | End[result.StreamedRunResult[DepsT, NodeRunEndT]]]:
        result_schema = ctx.deps.result_schema

        ctx.state.message_history.append(self.request)

        # Check usage
        if ctx.deps.usage_limits:
            ctx.deps.usage_limits.check_before_request(ctx.state.usage)

        # Increment run_step
        ctx.state.run_step += 1

        with _logfire.span('preparing model and tools {run_step=}', run_step=ctx.state.run_step):
            model_request_parameters = await _prepare_request_parameters(ctx)

        # Actually make the model request
        model_settings = merge_model_settings(ctx.deps.model_settings, None)
        with _logfire.span('model request {run_step=}', run_step=ctx.state.run_step) as model_req_span:
            async with ctx.deps.model.request_stream(
                ctx.state.message_history, model_settings, model_request_parameters
            ) as streamed_response:
                ctx.state.usage.requests += 1
                model_req_span.set_attribute('response_type', streamed_response.__class__.__name__)
                # We want to end the "model request" span here, but we can't exit the context manager
                # in the traditional way
                model_req_span.__exit__(None, None, None)

                with _logfire.span('handle model response') as handle_span:
                    received_text = False

                    async for maybe_part_event in streamed_response:
                        if isinstance(maybe_part_event, _messages.PartStartEvent):
                            new_part = maybe_part_event.part
                            if isinstance(new_part, _messages.TextPart):
                                received_text = True
                                if _allow_text_result(result_schema):
                                    handle_span.message = 'handle model response -> final result'
                                    streamed_run_result = _build_streamed_run_result(streamed_response, None, ctx)
                                    self._result = End(streamed_run_result)
                                    yield self._result
                                    return
                            elif isinstance(new_part, _messages.ToolCallPart):
                                if result_schema is not None and (match := result_schema.find_tool([new_part])):
                                    call, _ = match
                                    handle_span.message = 'handle model response -> final result'
                                    streamed_run_result = _build_streamed_run_result(
                                        streamed_response, call.tool_name, ctx
                                    )
                                    self._result = End(streamed_run_result)
                                    yield self._result
                                    return
                            else:
                                assert_never(new_part)

                    tasks: list[asyncio.Task[_messages.ModelRequestPart]] = []
                    parts: list[_messages.ModelRequestPart] = []
                    model_response = streamed_response.get()
                    if not model_response.parts:
                        raise exceptions.UnexpectedModelBehavior('Received empty model response')
                    ctx.state.message_history.append(model_response)

                    run_context = _build_run_context(ctx)
                    for p in model_response.parts:
                        if isinstance(p, _messages.ToolCallPart):
                            if tool := ctx.deps.function_tools.get(p.tool_name):
                                tasks.append(asyncio.create_task(tool.run(p, run_context), name=p.tool_name))
                            else:
                                parts.append(_unknown_tool(p.tool_name, ctx))

                    if received_text and not tasks and not parts:
                        # Can only get here if self._allow_text_result returns `False` for the provided result_schema
                        ctx.state.increment_retries(ctx.deps.max_result_retries)
                        self._result = StreamModelRequestNode[DepsT, NodeRunEndT](
                            _messages.ModelRequest(
                                parts=[
                                    _messages.RetryPromptPart(
                                        content='Plain text responses are not permitted, please call one of the functions instead.',
                                    )
                                ]
                            )
                        )
                        yield self._result
                        return

                    with _logfire.span('running {tools=}', tools=[t.get_name() for t in tasks]):
                        task_results: Sequence[_messages.ModelRequestPart] = await asyncio.gather(*tasks)
                        parts.extend(task_results)

                    next_request = _messages.ModelRequest(parts=parts)
                    if any(isinstance(part, _messages.RetryPromptPart) for part in parts):
                        try:
                            ctx.state.increment_retries(ctx.deps.max_result_retries)
                        except:
                            # TODO: This is janky, so I think we should probably change it, but how?
                            ctx.state.message_history.append(next_request)
                            raise

                    handle_span.set_attribute('tool_responses', parts)
                    tool_responses_str = ' '.join(r.part_kind for r in parts)
                    handle_span.message = f'handle model response -> {tool_responses_str}'
                    # the model_response should have been fully streamed by now, we can add its usage
                    streamed_response_usage = streamed_response.usage()
                    run_context.usage.incr(streamed_response_usage)
                    ctx.deps.usage_limits.check_tokens(run_context.usage)
                    self._result = StreamModelRequestNode[DepsT, NodeRunEndT](next_request)
                    yield self._result
                    return


@dataclasses.dataclass
class FinalResultNode(BaseNode[GraphAgentState, GraphAgentDeps[DepsT, Any], MarkFinalResult[NodeRunEndT]]):
    """Produce the final result of the run."""

    data: MarkFinalResult[NodeRunEndT]
    """The final result data."""
    extra_parts: list[_messages.ModelRequestPart] = dataclasses.field(default_factory=list)

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> End[MarkFinalResult[NodeRunEndT]]:
        run_span = ctx.deps.run_span
        usage = ctx.state.usage
        messages = ctx.state.message_history

        # TODO: For backwards compatibility, append a new ModelRequest using the tool returns and retries
        if self.extra_parts:
            messages.append(_messages.ModelRequest(parts=self.extra_parts))

        # TODO: Set this attribute somewhere
        # handle_span = self.handle_model_response_span
        # handle_span.set_attribute('final_data', self.data)
        run_span.set_attribute('usage', usage)
        run_span.set_attribute('all_messages', messages)

        # End the run with self.data
        return End(self.data)


def _build_run_context(ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, Any]]) -> RunContext[DepsT]:
    return RunContext[DepsT](
        deps=ctx.deps.user_deps,
        model=ctx.deps.model,
        usage=ctx.state.usage,
        prompt=ctx.deps.prompt,
        messages=ctx.state.message_history,
        run_step=ctx.state.run_step,
    )


def _build_streamed_run_result(
    result_stream: models.StreamedResponse,
    result_tool_name: str | None,
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
) -> result.StreamedRunResult[DepsT, NodeRunEndT]:
    new_message_index = ctx.deps.new_message_index
    result_schema = ctx.deps.result_schema
    run_span = ctx.deps.run_span
    usage_limits = ctx.deps.usage_limits
    messages = ctx.state.message_history
    run_context = _build_run_context(ctx)

    async def on_complete():
        """Called when the stream has completed.

        The model response will have been added to messages by now
        by `StreamedRunResult._marked_completed`.
        """
        last_message = messages[-1]
        assert isinstance(last_message, _messages.ModelResponse)
        tool_calls = [part for part in last_message.parts if isinstance(part, _messages.ToolCallPart)]
        parts = await _process_function_tools(
            tool_calls,
            result_tool_name,
            ctx,
        )
        # TODO: Should we do something here related to the retry count?
        #   Maybe we should move the incrementing of the retry count to where we actually make a request?
        # if any(isinstance(part, _messages.RetryPromptPart) for part in parts):
        #     ctx.state.increment_retries(ctx.deps.max_result_retries)
        if parts:
            messages.append(_messages.ModelRequest(parts))
        run_span.set_attribute('all_messages', messages)

    return result.StreamedRunResult[DepsT, NodeRunEndT](
        messages,
        new_message_index,
        usage_limits,
        result_stream,
        result_schema,
        run_context,
        ctx.deps.result_validators,
        result_tool_name,
        on_complete,
    )


async def _process_function_tools(
    tool_calls: list[_messages.ToolCallPart],
    result_tool_name: str | None,
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
) -> list[_messages.ModelRequestPart]:
    """Process function (non-result) tool calls in parallel.

    Also add stub return parts for any other tools that need it.
    """
    parts: list[_messages.ModelRequestPart] = []
    tasks: list[asyncio.Task[_messages.ToolReturnPart | _messages.RetryPromptPart]] = []

    stub_function_tools = bool(result_tool_name) and ctx.deps.end_strategy == 'early'
    result_schema = ctx.deps.result_schema

    # we rely on the fact that if we found a result, it's the first result tool in the last
    found_used_result_tool = False
    run_context = _build_run_context(ctx)

    for call in tool_calls:
        if call.tool_name == result_tool_name and not found_used_result_tool:
            found_used_result_tool = True
            parts.append(
                _messages.ToolReturnPart(
                    tool_name=call.tool_name,
                    content='Final result processed.',
                    tool_call_id=call.tool_call_id,
                )
            )
        elif tool := ctx.deps.function_tools.get(call.tool_name):
            if stub_function_tools:
                parts.append(
                    _messages.ToolReturnPart(
                        tool_name=call.tool_name,
                        content='Tool not executed - a final result was already processed.',
                        tool_call_id=call.tool_call_id,
                    )
                )
            else:
                tasks.append(asyncio.create_task(tool.run(call, run_context), name=call.tool_name))
        elif result_schema is not None and call.tool_name in result_schema.tools:
            # if tool_name is in _result_schema, it means we found a result tool but an error occurred in
            # validation, we don't add another part here
            if result_tool_name is not None:
                parts.append(
                    _messages.ToolReturnPart(
                        tool_name=call.tool_name,
                        content='Result tool not used - a final result was already processed.',
                        tool_call_id=call.tool_call_id,
                    )
                )
        else:
            parts.append(_unknown_tool(call.tool_name, ctx))

    # Run all tool tasks in parallel
    if tasks:
        with _logfire.span('running {tools=}', tools=[t.get_name() for t in tasks]):
            task_results: Sequence[_messages.ToolReturnPart | _messages.RetryPromptPart] = await asyncio.gather(*tasks)
            for result in task_results:
                if isinstance(result, _messages.ToolReturnPart):
                    parts.append(result)
                elif isinstance(result, _messages.RetryPromptPart):
                    parts.append(result)
                else:
                    assert_never(result)
    return parts


def _unknown_tool(
    tool_name: str,
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
) -> _messages.RetryPromptPart:
    ctx.state.increment_retries(ctx.deps.max_result_retries)
    tool_names = list(ctx.deps.function_tools.keys())
    if result_schema := ctx.deps.result_schema:
        tool_names.extend(result_schema.tool_names())

    if tool_names:
        msg = f'Available tools: {", ".join(tool_names)}'
    else:
        msg = 'No tools available.'

    return _messages.RetryPromptPart(content=f'Unknown tool name: {tool_name!r}. {msg}')


async def _validate_result(
    result_data: T,
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, T]],
    tool_call: _messages.ToolCallPart | None,
) -> T:
    for validator in ctx.deps.result_validators:
        run_context = _build_run_context(ctx)
        result_data = await validator.validate(result_data, tool_call, run_context)
    return result_data


def _allow_text_result(result_schema: _result.ResultSchema[Any] | None) -> bool:
    return result_schema is None or result_schema.allow_text_result


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
    name: str | None, deps_type: type[DepsT], result_type: type[ResultT]
) -> Graph[GraphAgentState, GraphAgentDeps[DepsT, Any], MarkFinalResult[ResultT]]:
    # We'll define the known node classes:
    nodes = (
        UserPromptNode[DepsT],
        ModelRequestNode[DepsT],
        HandleResponseNode[DepsT],
        FinalResultNode[DepsT, ResultT],
    )
    graph = Graph[GraphAgentState, GraphAgentDeps[DepsT, Any], MarkFinalResult[ResultT]](
        nodes=nodes,
        name=name or 'Agent',
        state_type=GraphAgentState,
        run_end_type=MarkFinalResult[result_type],
        auto_instrument=False,
    )
    return graph


def build_agent_stream_graph(
    name: str | None, deps_type: type[DepsT], result_type: type[ResultT] | None
) -> Graph[GraphAgentState, GraphAgentDeps[DepsT, Any], result.StreamedRunResult[DepsT, Any]]:
    nodes = [
        StreamUserPromptNode[DepsT, result.StreamedRunResult[DepsT, ResultT]],
        StreamModelRequestNode[DepsT, result.StreamedRunResult[DepsT, ResultT]],
    ]
    graph = Graph[GraphAgentState, GraphAgentDeps[DepsT, Any], result.StreamedRunResult[DepsT, Any]](
        nodes=nodes,
        name=name or 'Agent',
        state_type=GraphAgentState,
        run_end_type=result.StreamedRunResult[DepsT, result_type],
    )
    return graph
