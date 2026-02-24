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
    """The node that handles tool calls and returns the final result."""

    model_response: _messages.ModelResponse

    _: dataclasses.KW_ONLY

    tool_call_results: dict[str, DeferredToolResult | Literal['skip']] | None = None
    tool_call_metadata: dict[str, dict[str, Any]] | None = None
    user_prompt: str | Sequence[_messages.UserContent] | None = None

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> End[result.FinalResult[NodeRunEndT]]:
        """Run the tool calls and return the final result."""
        tool_manager = ctx.deps.tool_manager
        tool_calls = self.model_response.tool_calls

        output_final_result: list[result.FinalResult[NodeRunEndT]] = []
        output_parts: list[_messages.ModelRequestPart] = []

        if tool_calls:
            # Process tool calls
            async for event in process_tool_calls(
                ctx,
                tool_manager,
                tool_calls,
                self.tool_call_results,
                self.tool_call_metadata,
                output_final_result,
                output_parts,
            ):
                pass

        if not output_final_result:
            # No tool calls, or no final result from tool calls
            result_data = cast(NodeRunEndT, self.model_response.content)
            output_final_result.append(result.FinalResult(result_data))

        # Validate the final result
        run_context = build_run_context(ctx)
        for validator in ctx.deps.output_validators:
            result_data = await validator.validate(output_final_result[0].data, run_context)
            output_final_result[0] = result.FinalResult(result_data)

        return End(output_final_result[0])


async def process_tool_calls(  # noqa: C901
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
    tool_manager: ToolManager[DepsT],
    tool_calls: list[_messages.ToolCallPart],
    tool_call_results: dict[str, DeferredToolResult | Literal['skip']] | None,
    tool_call_metadata: dict[str, dict[str, Any]] | None,
    output_final_result: list[result.FinalResult[NodeRunEndT]],
    output_parts: list[_messages.ModelRequestPart],
) -> AsyncIterator[_messages.HandleResponseEvent]:
    """Process tool calls and return the final result."""
    # Group tool calls by kind
    tool_calls_by_kind: dict[Literal['output', 'function', 'unknown', 'external', 'unapproved'], list[_messages.ToolCallPart]] = {
        'output': [],
        'function': [],
        'unknown': [],
        'external': [],
        'unapproved': [],
    }

    for call in tool_calls:
        if await tool_manager.is_output_tool(call.tool_name):
            tool_calls_by_kind['output'].append(call)
        elif await tool_manager.is_function_tool(call.tool_name):
            tool_calls_by_kind['function'].append(call)
        elif await tool_manager.is_external_tool(call.tool_name):
            tool_calls_by_kind['external'].append(call)
        elif await tool_manager.is_unapproved_tool(call.tool_name):
            tool_calls_by_kind['unapproved'].append(call)
        else:
            tool_calls_by_kind['unknown'].append(call)

    # First, we handle output tool calls
    final_result: result.FinalResult[NodeRunEndT] | None = None
    for call in tool_calls_by_kind['output']:
        try:
            validated = await tool_manager.validate_tool_call(call)
        except exceptions.UnexpectedModelBehavior as e:
            if final_result:
                # If output tool fails when we already have a final result, skip it without retrying
                for event in _emit_skipped_output_tool(
                    call, 'Output tool not used - output failed validation.', output_parts, args_valid=False
                ):
                    yield event
                continue
            ctx.state.increment_retries(
                ctx.deps.max_result_retries, error=e, model_settings=ctx.deps.model_settings
            )
            raise  # pragma: lax no cover

        if not validated.args_valid:
            assert validated.validation_error is not None
            if final_result:
                for event in _emit_skipped_output_tool(
                    call, 'Output tool not used - output failed validation.', output_parts, args_valid=False
                ):
                    yield event
                continue

            ctx.state.increment_retries(
                ctx.deps.max_result_retries,
                error=validated.validation_error,
                model_settings=ctx.deps.model_settings,
            )
            yield _messages.FunctionToolCallEvent(call, args_valid=False)
            output_parts.append(validated.validation_error.tool_retry)
            yield _messages.FunctionToolResultEvent(validated.validation_error.tool_retry)
            continue

        # Validation passed - execute the tool
        try:
            result_data = await tool_manager.execute_tool_call(validated)
        except exceptions.UnexpectedModelBehavior as e:
            if final_result:
                for event in _emit_skipped_output_tool(
                    call, 'Output tool not used - output function execution failed.', output_parts, args_valid=True
                ):
                    yield event
                continue
            ctx.state.increment_retries(
                ctx.deps.max_result_retries, error=e, model_settings=ctx.deps.model_settings
            )
            raise  # pragma: lax no cover
        except ToolRetryError as e:
            # If we already have a valid final result, don't increment retries for invalid output tools
            # This allows the run to succeed if at least one output tool returned a valid result
            if not final_result:
                ctx.state.increment_retries(
                    ctx.deps.max_result_retries, error=e, model_settings=ctx.deps.model_settings
                )
            yield _messages.FunctionToolCallEvent(call, args_valid=True)
            output_parts.append(e.tool_retry)
            yield _messages.FunctionToolResultEvent(e.tool_retry)
            continue

        part = _messages.ToolReturnPart(
            tool_name=call.tool_name,
            content='Final result processed.',
            tool_call_id=call.tool_call_id,
        )
        output_parts.append(part)

        # In both `early` and `exhaustive` modes, use the first output tool's result as the final result
        if not final_result:
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
        calls_to_run_results = {call_id: value for call_id, value in tool_call_results.items() if value != 'skip'}
        calls_to_run = [call for call in calls_to_run if call.tool_call_id in calls_to_run_results]

    deferred_calls: dict[Literal['external', 'unapproved'], list[_messages.ToolCallPart]] = defaultdict(list)
    deferred_metadata: dict[str, dict[str, Any]] = {}

    if calls_to_run:
        # Check usage limits before running tools
        if ctx.deps.usage_limits.tool_calls_limit is not None:
            projected_usage = deepcopy(ctx.state.usage)
            projected_usage.tool_calls += len(calls_to_run)
            ctx.deps.usage_limits.check_before_tool_call(projected_usage)

        # Validate upfront and cache results. For ToolApproved deferred results, validate
        # with the approval context so the event reflects the actual validation outcome.
        # Other deferred result types (ToolDenied, ModelRetry, etc.) skip validation since
        # the tool won't actually execute.
        validated_calls: dict[str, ValidatedToolCall[DepsT]] = {}
        for call in calls_to_run:
            deferred_result = calls_to_run_results.get(call.tool_call_id)
            if deferred_result is not None and not isinstance(deferred_result, ToolApproved):
                yield _messages.FunctionToolCallEvent(call)
                continue
            try:
                if isinstance(deferred_result, ToolApproved):
                    validate_call = call
                    if deferred_result.override_args is not None:
                        validate_call = dataclasses.replace(call, args=deferred_result.override_args)
                    metadata = tool_call_metadata.get(call.tool_call_id) if tool_call_metadata else None
                    validated = await tool_manager.validate_tool_call(validate_call, approved=True, metadata=metadata)
                else:
                    validated = await tool_manager.validate_tool_call(call)
            except exceptions.UnexpectedModelBehavior:
                yield _messages.FunctionToolCallEvent(call, args_valid=False)
                raise
            validated_calls[call.tool_call_id] = validated
            yield _messages.FunctionToolCallEvent(call, args_valid=validated.args_valid)

        async for event in _call_tools(
            tool_manager=tool_manager,
            tool_calls=calls_to_run,
            tool_call_results=calls_to_run_results,
            validated_calls=validated_calls,
            tracer=ctx.deps.tracer,
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
            for call in calls:
                try:
                    validated = await tool_manager.validate_tool_call(call)
                except exceptions.UnexpectedModelBehavior:
                    yield _messages.FunctionToolCallEvent(call, args_valid=False)
                    raise

                yield _messages.FunctionToolCallEvent(call, args_valid=validated.args_valid)

                if validated.args_valid:
                    if call in tool_calls_by_kind['external']:
                        deferred_calls['external'].append(call)
                    else:
                        deferred_calls['unapproved'].append(call)
                else:
                    # Call execute_tool_call to raise the validation error inside a trace span;
                    # retries are already tracked by validate_tool_call() via failed_tools.
                    try:
                        await tool_manager.execute_tool_call(validated)
                    except ToolRetryError as e:
                        output_parts.append(e.tool_retry)
                        yield _messages.FunctionToolResultEvent(e.tool_retry)

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


async def _call_tools(  # noqa: C901
    tool_manager: ToolManager[DepsT],
    tool_calls: list[_messages.ToolCallPart],
    tool_call_results: dict[str, DeferredToolResult],
    validated_calls: dict[str, ValidatedToolCall[DepsT]],
    tracer: Tracer,
    output_parts: list[_messages.ModelRequestPart],
    output_deferred_calls: dict[Literal['external', 'unapproved'], list[_messages.ToolCallPart]],
    output_deferred_metadata: dict[str, dict[str, Any]],
) -> AsyncIterator[_messages.HandleResponseEvent]:
    tool_parts_by_index: dict[int, _messages.ModelRequestPart] = {}
    user_parts_by_index: dict[int, _messages.UserPromptPart] = {}
    deferred_calls_by_index: dict[int, Literal['external', 'unapproved']] = {}
    deferred_metadata_by_index: dict[int, dict[str, Any] | None] = {}

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

        parallel_execution_mode = tool_manager.get_parallel_execution_mode(tool_calls)
        if parallel_execution_mode == 'sequential':
            for index, call in enumerate(tool_calls):
                if event := await handle_call_or_result(
                    _call_tool(
                        tool_manager,
                        validated_calls.get(call.tool_call_id, call),
                        tool_call_results.get(call.tool_call_id),
                    ),
                    index,
                ):
                    yield event

        else:
            tasks = [
                asyncio.create_task(
                    _call_tool(
                        tool_manager,
                        validated_calls.get(call.tool_call_id, call),
                        tool_call_results.get(call.tool_call_id),
                    ),
                    name=call.tool_name,
                )
                for call in tool_calls
            ]
            try:
                if parallel_execution_mode == 'parallel_ordered_events':
                    # Wait for all tasks to complete before yielding any events
                    await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
                    for index, task in enumerate(tasks):
                        if event := await handle_call_or_result(coro_or_task=task, index=index):
                            yield event
                else:
                    pending: set[
                        asyncio.Task[
                            tuple[_messages.ToolReturnPart | _messages.RetryPromptPart, _messages.UserPromptPart | None]
                        ]
                    ] = set(tasks)  # pyright: ignore[reportAssignmentType]
                    while pending:
                        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            index = tasks.index(task)  # pyright: ignore[reportArgumentType]
                            if event := await handle_call_or_result(coro_or_task=task, index=index):  # pyright: ignore[reportArgumentType]
                                yield event

            except asyncio.CancelledError as e:
                for task in tasks:
                    task.cancel(msg=e.args[0] if len(e.args) != 0 else None)

                raise

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
    for index, kind in deferred_calls_by_index.items():
        call = tool_calls[index]
        output_deferred_calls[kind].append(call)
        if metadata := deferred_metadata_by_index.get(index):
            output_deferred_metadata[call.tool_call_id] = metadata


async def _call_tool(
    tool_manager: ToolManager[DepsT],
    call: _messages.ToolCallPart | ValidatedToolCall[DepsT],
    deferred_result: DeferredToolResult | None,
) -> tuple[_messages.ToolReturnPart | _messages.RetryPromptPart, str | Sequence[_messages.UserContent] | None]:
    """Call a tool and return the result."""
    if isinstance(deferred_result, ToolApproved):
        # For approved tools, we need to execute them with the approval context
        if deferred_result.override_args is not None:
            call = dataclasses.replace(call, args=deferred_result.override_args)
        result_data = await tool_manager.execute_tool_call(call)
        part = _messages.ToolReturnPart(
            tool_name=call.tool_name,
            content=result_data,
            tool_call_id=call.tool_call_id,
        )
        return part, None
    elif isinstance(deferred_result, ToolDenied):
        # For denied tools, we return a message saying the tool was denied
        part = _messages.ToolReturnPart(
            tool_name=call.tool_name,
            content='Tool call denied.',
            tool_call_id=call.tool_call_id,
        )
        return part, None
    elif isinstance(deferred_result, _messages.ModelRetry):
        # For model retries, we return a retry prompt
        part = _messages.RetryPromptPart(content=deferred_result.content)
        return part, None
    else:
        # For other deferred results, we execute the tool normally
        result_data = await tool_manager.execute_tool_call(call)
        part = _messages.ToolReturnPart(
            tool_name=call.tool_name,
            content=result_data,
            tool_call_id=call.tool_call_id,
        )
        return part, None


def _emit_skipped_output_tool(
    call: _messages.ToolCallPart,
    message: str,
    output_parts: list[_messages.ModelRequestPart],
    args_valid: bool | None,
) -> list[_messages.HandleResponseEvent]:
    """Emit events for a skipped output tool call."""
    events: list[_messages.HandleResponseEvent] = []
    events.append(_messages.FunctionToolCallEvent(call, args_valid=args_valid))
    part = _messages.ToolReturnPart(
        tool_name=call.tool_name,
        content=message,
        tool_call_id=call.tool_call_id,
    )
    output_parts.append(part)
    events.append(_messages.FunctionToolResultEvent(part))
    return events


def _clean_message_history(messages: list[_messages.ModelMessage]) -> list[_messages.ModelMessage]:
    """Clean the message history by removing any messages that are not needed."""
    # Remove any messages that are not ModelRequest or ModelResponse
    return [msg for msg in messages if isinstance(msg, (_messages.ModelRequest, _messages.ModelResponse))]


@dataclasses.dataclass
class _CapturedRunMessages:
    """Captured run messages."""

    messages: list[_messages.ModelMessage]
    used: bool = False


_captured_run_messages: ContextVar[_CapturedRunMessages | None] = ContextVar('_captured_run_messages', default=None)


def capture_run_messages(messages: list[_messages.ModelMessage]) -> None:
    """Capture run messages for use in the agent graph."""
    _captured_run_messages.set(_CapturedRunMessages(messages))


def get_captured_run_messages() -> _CapturedRunMessages:
    """Get captured run messages."""
    if messages := _captured_run_messages.get():
        return messages
    raise LookupError('No captured run messages found.')


def build_run_context(ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, Any]]) -> RunContext[DepsT]:
    """Build a run context from the graph run context."""
    return RunContext(
        deps=ctx.deps.user_deps,
        run_id=ctx.state.run_id,
        run_step=ctx.state.run_step,
        message_history=ctx.state.message_history,
        usage=ctx.state.usage,
        metadata=ctx.state.metadata,
        tracer=ctx.deps.tracer,
        instrumentation_settings=ctx.deps.instrumentation_settings,
    )
