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
    'build_run_context',
    'capture_run_messages',
    'HistoryProcessor',
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
        output_tools=output_tools,
        builtin_tools=builtin_tools,
        prompted_output_template=prompted_output_template,
    )


class ModelRequestNode(AgentNode[DepsT, NodeRunEndT]):
    """The node that makes a request to the model using the last message in state.message_history."""

    request: _messages.ModelRequest
    is_resuming_without_prompt: bool = False

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
        with set_current_run_context(run_context):
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
                    _metadata_getter=lambda: ctx.state.metadata,
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

        model_settings, model_request_parameters, message_history, run_context = await self._prepare_request(ctx)
        with set_current_run_context(run_context):
            model_response = await ctx.deps.model.request(message_history, model_settings, model_request_parameters, run_context)
        ctx.state.usage.requests += 1

        return self._finish_handling(ctx, model_response)

    async def _prepare_request(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> tuple[ModelSettings | None, models.ModelRequestParameters, list[_messages.ModelMessage], RunContext[DepsT]]:
        self.request.timestamp = now_utc()
        # Always set run_id if it's not already set, regardless of is_resuming_without_prompt
        self.request.run_id = self.request.run_id or ctx.state.run_id
        ctx.state.message_history.append(self.request)

        ctx.state.run_step += 1

        run_context = build_run_context(ctx)

        # This will raise errors for any tool name conflicts
        ctx.deps.tool_manager = await ctx.deps.tool_manager.for_run_step(run_context)

        message_history = await _process_message_history(
            ctx.state.message_history[:], ctx.deps.history_processors, run_context
        )
        if message_history and message_history[-1].run_id is None:
            is_resumed_tail = self.is_resuming_without_prompt and _is_same_request(message_history[-1], self.request)
            if not is_resumed_tail:
                message_history[-1].run_id = ctx.state.run_id

        if self.is_resuming_without_prompt:
            ctx.deps.resumed_request = self.request
        # `ctx.state.message_history` is the same list used by `capture_run_messages`, so we should replace its contents, not the reference
        ctx.state.message_history[:] = message_history
        # Update the new message index to ensure `result.new_messages()` returns the correct messages
        ctx.deps.new_message_index = _first_new_message_index(
            message_history, ctx.state.run_id, resumed_request=ctx.deps.resumed_request
        )

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
    tool_call_metadata: dict[str, dict[str, Any]] | None = None
    """Metadata for deferred tool calls, keyed by `tool_call_id`."""
    user_prompt: str | Sequence[_messages.UserContent] | None = None
    """Optional user prompt to include alongside tool call results.

    This is used when resuming a run with tool call results and a new user prompt."""

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> UserPromptNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]]:
        """Process the model response and decide what to do next."""
        # Check if we have tool calls to process
        if self.model_response.tool_calls:
            return await self._handle_tool_calls(ctx)
        else:
            return await self._handle_final_result(ctx)

    async def _handle_tool_calls(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> UserPromptNode[DepsT, NodeRunEndT]:
        """Handle tool calls in the model response."""
        # Get tool call results if provided
        tool_call_results = self.tool_call_results or {}
        tool_call_metadata = self.tool_call_metadata or {}

        # Process each tool call
        tool_return_parts: list[_messages.ToolReturnPart | _messages.BuiltinToolReturnPart] = []
        for tool_call in self.model_response.tool_calls:
            tool_call_id = tool_call.tool_call_id
            tool_name = tool_call.tool_name

            # Check if we have a result for this tool call
            if tool_call_id in tool_call_results:
                result = tool_call_results[tool_call_id]
                if result == 'skip':
                    continue
                elif isinstance(result, _messages.ToolReturn):
                    # Tool result was provided directly
                    tool_return_parts.append(
                        _messages.ToolReturnPart(
                            tool_name=tool_name,
                            content=result.content,
                            tool_call_id=tool_call_id,
                        )
                    )
                elif isinstance(result, (_messages.ToolApproved, _messages.ToolDenied)):
                    # Tool approval/denial was provided
                    tool_return_parts.append(
                        _messages.RetryPromptPart(
                            tool_name=tool_name,
                            tool_call_id=tool_call_id,
                            approval=result,
                            args=tool_call.args,
                        )
                    )
                else:
                    # This should not happen
                    raise exceptions.AgentRunError(f'Unexpected tool call result type: {type(result)}')  # pragma: no cover
            else:
                # No result provided, we need to execute the tool
                tool_return = await self._execute_tool(ctx, tool_call, tool_call_metadata.get(tool_call_id))
                tool_return_parts.append(tool_return)

        # Create a new user prompt node with the tool return parts
        return UserPromptNode[DepsT, NodeRunEndT](
            user_prompt=None,
            deferred_tool_results=None,
        )

    async def _execute_tool(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        tool_call: _messages.BaseToolCallPart,
        tool_call_metadata: dict[str, Any] | None = None,
    ) -> _messages.ToolReturnPart | _messages.BuiltinToolReturnPart:
        """Execute a tool call and return the result."""
        run_context = build_run_context(ctx)
        tool_manager = ctx.deps.tool_manager

        # Validate tool call
        validated_tool_call = tool_manager.validate_tool_call(tool_call)

        # Execute tool
        tool_result = await tool_manager.execute_tool(validated_tool_call, run_context, tool_call_metadata)

        # Create tool return part
        if isinstance(tool_result, _messages.ToolReturn):
            return _messages.ToolReturnPart(
                tool_name=validated_tool_call.tool_name,
                content=tool_result.content,
                tool_call_id=validated_tool_call.tool_call_id,
            )
        elif isinstance(tool_result, _messages.BuiltinToolReturn):
            return _messages.BuiltinToolReturnPart(
                tool_name=validated_tool_call.tool_name,
                content=tool_result.content,
                tool_call_id=validated_tool_call.tool_call_id,
                provider_name=tool_result.provider_name,
            )
        else:
            # This should not happen
            raise exceptions.AgentRunError(f'Unexpected tool result type: {type(tool_result)}')  # pragma: no cover

    async def _handle_final_result(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> End[result.FinalResult[NodeRunEndT]]:
        """Handle a final result from the model."""
        run_context = build_run_context(ctx)

        # Validate output
        try:
            validated_output = await ctx.deps.output_schema.validate(
                self.model_response, run_context, ctx.deps.output_validators
            )
        except exceptions.UnexpectedModelBehavior as e:
            # Increment retries and try again
            ctx.state.increment_retries(ctx.deps.max_result_retries, e, ctx.deps.model_settings)
            # Create a retry prompt
            retry_prompt = _messages.RetryPromptPart(
                tool_name='',  # No tool name for general retries
                tool_call_id=str(uuid.uuid4()),  # Generate a unique ID
                approval=_messages.ToolApproved(),  # Always approve retries
                args={},  # No args for general retries
            )
            # Create a new user prompt node with the retry prompt
            return UserPromptNode[DepsT, NodeRunEndT](
                user_prompt=None,
                deferred_tool_results=None,
            )

        # Check if we have a user prompt to include
        if self.user_prompt is not None:
            # Create a new user prompt node with the user prompt
            return UserPromptNode[DepsT, NodeRunEndT](
                user_prompt=self.user_prompt,
                deferred_tool_results=None,
            )
        else:
            # Create a final result
            final_result = result.FinalResult(
                output=validated_output,
                model_response=self.model_response,
            )
            # Return an end node
            return End(final_result)

    __repr__ = dataclasses_no_defaults_repr


# Helper functions

def build_run_context(ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, Any]]) -> RunContext[DepsT]:
    """Build a run context from the graph run context."""
    return RunContext(
        deps=ctx.deps.user_deps,
        run_id=ctx.state.run_id,
        metadata=ctx.state.metadata,
    )


def _clean_message_history(messages: list[_messages.ModelMessage]) -> list[_messages.ModelMessage]:
    """Clean up message history by merging consecutive ModelRequest or ModelResponse messages."""
    if not messages:
        return messages

    cleaned: list[_messages.ModelMessage] = []
    for msg in messages:
        if isinstance(msg, _messages.ModelRequest):
            if cleaned and isinstance(cleaned[-1], _messages.ModelRequest):
                # Merge consecutive ModelRequest messages
                cleaned[-1].parts = [*cleaned[-1].parts, *msg.parts]
                if msg.instructions and not cleaned[-1].instructions:
                    cleaned[-1].instructions = msg.instructions
            else:
                cleaned.append(msg)
        elif isinstance(msg, _messages.ModelResponse):
            if cleaned and isinstance(cleaned[-1], _messages.ModelResponse):
                # Merge consecutive ModelResponse messages
                cleaned[-1].parts = [*cleaned[-1].parts, *msg.parts]
                if msg.usage:
                    cleaned[-1].usage = cleaned[-1].usage + msg.usage if cleaned[-1].usage else msg.usage
            else:
                cleaned.append(msg)
        else:
            cleaned.append(msg)

    return cleaned


async def _process_message_history(
    messages: list[_messages.ModelMessage],
    history_processors: Sequence[HistoryProcessor[DepsT]],
    run_context: RunContext[DepsT],
) -> list[_messages.ModelMessage]:
    """Process message history using the provided processors."""
    for processor in history_processors:
        if is_takes_ctx(processor):
            # Processor takes a run context
            if is_async_callable(processor):
                messages = await processor(run_context, messages)
            else:
                messages = processor(run_context, messages)
        else:
            # Processor doesn't take a run context
            if is_async_callable(processor):
                messages = await processor(messages)
            else:
                messages = processor(messages)
    return messages


def _is_same_request(msg1: _messages.ModelMessage, msg2: _messages.ModelMessage) -> bool:
    """Check if two messages are the same request."""
    if not isinstance(msg1, _messages.ModelRequest) or not isinstance(msg2, _messages.ModelRequest):
        return False
    # Check if parts are the same
    if len(msg1.parts) != len(msg2.parts):
        return False
    for p1, p2 in zip(msg1.parts, msg2.parts):
        if type(p1) != type(p2):
            return False
        if isinstance(p1, _messages.UserPromptPart) and isinstance(p2, _messages.UserPromptPart):
            if p1.content != p2.content:
                return False
        elif isinstance(p1, _messages.SystemPromptPart) and isinstance(p2, _messages.SystemPromptPart):
            if p1.content != p2.content:
                return False
        elif isinstance(p1, _messages.ToolReturnPart) and isinstance(p2, _messages.ToolReturnPart):
            if p1.tool_name != p2.tool_name or p1.content != p2.content or p1.tool_call_id != p2.tool_call_id:
                return False
        elif isinstance(p1, _messages.RetryPromptPart) and isinstance(p2, _messages.RetryPromptPart):
            if p1.tool_name != p2.tool_name or p1.tool_call_id != p2.tool_call_id or p1.args != p2.args:
                return False
        else:
            # For other part types, we'll just assume they're the same if they're the same type
            pass
    return True

def _first_new_message_index(
    message_history: list[_messages.ModelMessage],
    run_id: str,
    resumed_request: _messages.ModelRequest | None = None,
) -> int:
    """Find the index of the first message in the history that belongs to the current run."""
    # If we're resuming a request, the first new message is the resumed request
    if resumed_request:
        for i, msg in enumerate(message_history):
            if msg is resumed_request:
                return i
    # Otherwise, find the first message with the current run_id
    for i, msg in enumerate(message_history):
        if msg.run_id == run_id:
            return i
    # If no message has the current run_id, return the length of the history
    return len(message_history)


# Context var for captured run messages
_captured_run_messages: ContextVar[_CapturedRunMessages | None] = ContextVar('_captured_run_messages', default=None)


@dataclasses.dataclass
class _CapturedRunMessages:
    """Captured run messages."""

    messages: list[_messages.ModelMessage]
    used: bool = False


def capture_run_messages() -> list[_messages.ModelMessage]:
    """Capture run messages in a context var.

    This is used to capture messages from a run so they can be reused in a subsequent run.

    Usage:

        messages = capture_run_messages()
        async with agent.iter('What is the capital of France?') as agent_run:
            async for node in agent_run:
                pass
        # messages now contains all messages from the run
    """
    captured = _captured_run_messages.get()
    if captured is None:
        captured = _CapturedRunMessages(messages=[])
        _captured_run_messages.set(captured)
    return captured.messages

def get_captured_run_messages() -> _CapturedRunMessages:
    """Get captured run messages.

    This is used internally by the agent graph to get captured messages.
    """
    captured = _captured_run_messages.get()
    if captured is None:
        raise LookupError('No captured run messages found')
    return captured