from __future__ import annotations as _annotations

import asyncio
import dataclasses
import inspect
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

from pydantic_ai._history_processor import HistoryProcessor
from pydantic_ai._instrumentation import DEFAULT_INSTRUMENTATION_VERSION
from pydantic_ai._utils import dataclasses_no_defaults_repr, get_union_args, now_utc
from pydantic_ai._uuid import uuid7
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tool_manager import ToolManager, ValidatedToolCall
from pydantic_graph import BaseNode, GraphRunContext
from pydantic_graph.beta import Graph, GraphBuilder
from pydantic_graph.nodes import End, NodeRunEndT

from . import _output, _system_prompt, exceptions, messages as _messages, models, result, usage as _usage
from ._run_context import set_current_run_context
from .exceptions import ToolRetryError
from .output import OutputDataT, OutputSpec
from .settings import ModelSettings
from .tools import (
    AgentBuiltinTool,
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
    from datetime import datetime

    from .agent.abstract import AbstractAgent
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
EndStrategy = Literal['early', 'graceful', 'exhaustive']
DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')


@dataclasses.dataclass(kw_only=True)
class GraphAgentState:
    """State kept across the execution of the agent graph."""

    message_history: list[_messages.ModelMessage] = dataclasses.field(default_factory=list[_messages.ModelMessage])
    usage: _usage.RunUsage = dataclasses.field(default_factory=_usage.RunUsage)
    retries: int = 0
    run_step: int = 0
    run_id: str = dataclasses.field(default_factory=lambda: str(uuid7()))
    metadata: dict[str, Any] | None = None
    last_max_tokens: int | None = None
    """Last-resolved `max_tokens` from model settings, used only in error messages."""
    last_model_request_parameters: models.ModelRequestParameters | None = None
    """Last-resolved model request parameters, used for OTel span attributes."""

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

    def increment_retries(
        self,
        max_result_retries: int,
        error: BaseException | None = None,
    ) -> None:
        self.retries += 1
        if self.retries > max_result_retries:
            self.check_incomplete_tool_call()
            message = f'Exceeded maximum retries ({max_result_retries}) for output validation'
            raise exceptions.UnexpectedModelBehavior(message) from error


@dataclasses.dataclass(kw_only=True)
class GraphAgentDeps(Generic[DepsT, OutputDataT]):
    """Dependencies/config passed to the agent graph."""

    user_deps: DepsT

    prompt: str | Sequence[_messages.UserContent] | None
    new_message_index: int
    resumed_request: _messages.ModelRequest | None

    model: models.Model
    get_model_settings: Callable[[RunContext[DepsT]], ModelSettings | None]
    usage_limits: _usage.UsageLimits
    max_result_retries: int
    end_strategy: EndStrategy
    get_instructions: Callable[[RunContext[DepsT]], Awaitable[list[_messages.InstructionPart] | None]]

    output_schema: _output.OutputSchema[OutputDataT]
    output_validators: list[_output.OutputValidator[DepsT, OutputDataT]]
    validation_context: Any | Callable[[RunContext[DepsT]], Any]

    root_capability: AbstractCapability[DepsT]

    builtin_tools: list[AgentBuiltinTool[DepsT]] = dataclasses.field(repr=False)
    tool_manager: ToolManager[DepsT]

    tracer: Tracer
    instrumentation_settings: InstrumentationSettings | None

    agent: AbstractAgent[DepsT, Any] | None = None


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
                    # Align with the upcoming request step so we don't resolve dynamic toolsets twice.
                    run_context = replace(
                        build_run_context(ctx),
                        run_step=ctx.state.run_step + 1,
                        retry=ctx.state.retries,
                        max_retries=ctx.deps.max_result_retries,
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
                elif last_message.tool_calls:
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
    if toolset_result:
        # The top-level toolset is always a CombinedToolset which returns a list,
        # but the return type also allows a single str or InstructionPart for custom subclasses.
        items = [toolset_result] if isinstance(toolset_result, (str, _messages.InstructionPart)) else toolset_result
        for item in items:
            if isinstance(item, _messages.InstructionPart):
                if item.content.strip():
                    parts.append(item)
            else:
                # Plain str from toolsets: treat as dynamic (external/changeable source)
                if item.strip():
                    parts.append(_messages.InstructionPart(content=item, dynamic=True))

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

    all_tool_defs = list(ctx.deps.tool_manager.tool_defs)

    # Let capabilities filter/modify tool definitions
    run_context = build_run_context(ctx)
    all_tool_defs = await ctx.deps.root_capability.prepare_tools(run_context, all_tool_defs)

    function_tools: list[ToolDefinition] = []
    output_tools: list[ToolDefinition] = []
    for tool_def in all_tool_defs:
        if tool_def.kind == 'output':
            output_tools.append(tool_def)
        else:
            function_tools.append(tool_def)

    # resolve dynamic builtin tools
    builtin_tools: list[AbstractBuiltinTool] = []
    if ctx.deps.builtin_tools:
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
        instruction_parts=instruction_parts,
    )


@dataclasses.dataclass
class _SkipStreamedResponse(models.StreamedResponse):
    """Minimal StreamedResponse for SkipModelRequest — yields no events.

    These properties implement the StreamedResponse ABC but are never accessed:
    the streaming skip path always resolves via the _run_result shortcut in
    StreamedRunResult, so the AgentStream wrapping this response is discarded.
    """

    _response: _messages.ModelResponse = field(repr=False)

    @property
    def model_name(self) -> str:  # pragma: no cover
        return self._response.model_name or ''

    @property
    def provider_name(self) -> str | None:  # pragma: no cover
        return None

    @property
    def provider_url(self) -> str | None:  # pragma: no cover
        return None

    @property
    def timestamp(self) -> datetime:  # pragma: no cover
        return self._response.timestamp

    async def _get_event_iterator(self) -> AsyncIterator[_messages.ModelResponseStreamEvent]:
        return
        yield  # pragma: no cover

    def get(self) -> _messages.ModelResponse:  # pragma: no cover
        return self._response


@dataclasses.dataclass
class ModelRequestNode(AgentNode[DepsT, NodeRunEndT]):
    """The node that makes a request to the model using the last message in state.message_history."""

    request: _messages.ModelRequest
    is_resuming_without_prompt: bool = False

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
    ) -> AsyncIterator[result.AgentStream[DepsT, T]]:
        assert not self._did_stream, 'stream() should only be called once per node'

        try:
            model, model_settings, model_request_parameters, message_history, run_context = await self._prepare_request(
                ctx
            )
        except exceptions.SkipModelRequest as e:
            # SkipModelRequest in stream path: yield an empty stream and finish handling
            # new_message_index wasn't updated in _prepare_request, fix it here
            ctx.deps.new_message_index = _first_new_message_index(
                ctx.state.message_history, ctx.state.run_id, resumed_request=ctx.deps.resumed_request
            )
            self._did_stream = True
            ctx.state.usage.requests += 1
            # instruction_parts=None is fine here: the model isn't called, we just need MRP for the wrapper
            skip_mrp = await _prepare_request_parameters(ctx, instruction_parts=None)
            skip_sr = _SkipStreamedResponse(model_request_parameters=skip_mrp, _response=e.response)
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
            with set_current_run_context(run_context):
                async with req_ctx.model.request_stream(
                    req_ctx.messages, req_ctx.model_settings, req_ctx.model_request_parameters, run_context
                ) as sr:
                    self._did_stream = True
                    ctx.state.usage.requests += 1
                    agent_stream = self._build_agent_stream(ctx, sr, req_ctx.model_request_parameters)
                    agent_stream_holder.append(agent_stream)
                    stream_ready.set()
                    await stream_done.wait()
            response = sr.get()
            _handler_response = response
            return response

        wrap_request_context = ModelRequestContext(
            model=model,
            messages=message_history,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
        )
        wrap_task = asyncio.create_task(
            ctx.deps.root_capability.wrap_model_request(
                run_context,
                request_context=wrap_request_context,
                handler=_streaming_handler,
            )
        )

        # Wait for handler to start or wrap to complete (short-circuit)
        ready_waiter = asyncio.create_task(stream_ready.wait())
        await asyncio.wait({ready_waiter, wrap_task}, return_when=asyncio.FIRST_COMPLETED)
        ready_waiter.cancel()

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
                dummy_sr = _SkipStreamedResponse(
                    model_request_parameters=model_request_parameters,
                    _response=_messages.ModelResponse(parts=[]),
                )
                yield self._build_agent_stream(ctx, dummy_sr, model_request_parameters)
                return
            self._did_stream = True
            ctx.state.usage.requests += 1
            skip_sr = _SkipStreamedResponse(model_request_parameters=model_request_parameters, _response=model_response)
            agent_stream = self._build_agent_stream(ctx, skip_sr, model_request_parameters)
            yield agent_stream
            self.last_request_context = wrap_request_context
            await self._finish_handling(ctx, model_response)
            assert self._result is not None
            return

        # Normal path: handler was called, stream is ready
        stream_error: BaseException | None = None
        try:
            yield agent_stream_holder[0]
            # Ensure stream is fully consumed for proper usage counting
            async for _ in agent_stream_holder[0]:
                pass
        except BaseException as exc:
            stream_error = exc
            raise
        finally:
            stream_done.set()

            if stream_error is not None:
                wrap_task.cancel()
                try:
                    await wrap_task
                except (asyncio.CancelledError, BaseException):
                    pass
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
            _metadata_getter=lambda: ctx.state.metadata,
        )

    async def _make_request(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> CallToolsNode[DepsT, NodeRunEndT] | ModelRequestNode[DepsT, NodeRunEndT]:
        if self._result is not None:
            return self._result  # pragma: no cover

        try:
            model, model_settings, model_request_parameters, message_history, run_context = await self._prepare_request(
                ctx
            )
        except exceptions.SkipModelRequest as e:
            # new_message_index wasn't updated in _prepare_request, fix it here
            ctx.deps.new_message_index = _first_new_message_index(
                ctx.state.message_history, ctx.state.run_id, resumed_request=ctx.deps.resumed_request
            )
            ctx.state.usage.requests += 1
            return await self._finish_handling(ctx, e.response)

        _handler_response: _messages.ModelResponse | None = None

        async def model_handler(req_ctx: ModelRequestContext) -> _messages.ModelResponse:
            nonlocal _handler_response
            with set_current_run_context(run_context):
                response = await req_ctx.model.request(
                    req_ctx.messages, req_ctx.model_settings, req_ctx.model_request_parameters
                )
                _handler_response = response
                return response

        request_context = ModelRequestContext(
            model=model,
            messages=message_history,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
        )
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
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> tuple[
        models.Model,
        ModelSettings | None,
        models.ModelRequestParameters,
        list[_messages.ModelMessage],
        RunContext[DepsT],
    ]:
        self.request.timestamp = now_utc()
        if not self.is_resuming_without_prompt:
            self.request.run_id = self.request.run_id or ctx.state.run_id
        ctx.state.message_history.append(self.request)

        ctx.state.run_step += 1

        run_context = build_run_context(ctx)
        run_context = replace(
            run_context,
            retry=ctx.state.retries,
            max_retries=ctx.deps.max_result_retries,
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

        # Ensure the last request has a timestamp (history processors may create new ModelRequest objects without one)
        if messages[-1].timestamp is None:
            messages[-1].timestamp = now_utc()

        if messages and messages[-1].run_id is None:
            messages[-1].run_id = ctx.state.run_id

        if self.is_resuming_without_prompt:
            ctx.deps.resumed_request = self.request
        # `ctx.state.message_history` is the same list used by `capture_run_messages`, so we should replace its contents, not the reference
        ctx.state.message_history[:] = messages
        # Update the new message index to ensure `result.new_messages()` returns the correct messages
        ctx.deps.new_message_index = _first_new_message_index(
            messages, ctx.state.run_id, resumed_request=ctx.deps.resumed_request
        )

        # Merge possible consecutive trailing `ModelRequest`s into one, with tool call parts before user parts,
        # but don't store it in the message history on state. This is just for the benefit of model classes that want clear user/assistant boundaries.
        # See `tests/test_tools.py::test_parallel_tool_return_with_deferred` for an example where this is necessary
        messages = _clean_message_history(messages)

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

    async def _finish_handling(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        response: _messages.ModelResponse,
    ) -> CallToolsNode[DepsT, NodeRunEndT] | ModelRequestNode[DepsT, NodeRunEndT]:
        response.run_id = response.run_id or ctx.state.run_id

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
        response.run_id = response.run_id or ctx.state.run_id
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
        ctx.state.increment_retries(ctx.deps.max_result_retries, error=error)
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

                    # If the output type allows None, an empty response is a valid result.
                    if is_empty and output_schema.allows_none:
                        run_context = _build_output_run_context(ctx)
                        try:
                            result_data = await _run_output_validators(ctx, cast(NodeRunEndT, None), run_context)
                            self._next_node = self._handle_final_result(ctx, result.FinalResult(result_data), [])
                        except ToolRetryError as e:
                            ctx.state.increment_retries(ctx.deps.max_result_retries, error=e)
                            self._next_node = ModelRequestNode[DepsT, NodeRunEndT](
                                _messages.ModelRequest(parts=[e.tool_retry])
                            )
                        return

                    # Try to recover text from a previous model response.
                    # This handles the case where the model returned text alongside tool calls
                    # (so the text was discarded in favor of executing the tools) and subsequently
                    # returned an empty or thinking-only response.
                    if text_processor := output_schema.text_processor:
                        text = self._recover_text_from_message_history(ctx.state.message_history)
                        if text is not None:
                            try:
                                self._next_node = await self._handle_text_response(ctx, text, text_processor)
                                return
                            except ToolRetryError:  # pragma: no cover
                                # If the recovered text was invalid, fall through.
                                pass

                    if is_empty:
                        # Go back to the model request node with an empty request, which means we'll
                        # essentially resubmit the most recent request that resulted in an empty response,
                        # as the empty response and request will not create any items in the API payload,
                        # in the hope the model will return a non-empty response this time.
                        ctx.state.increment_retries(ctx.deps.max_result_retries)
                        self._next_node = ModelRequestNode[DepsT, NodeRunEndT](_messages.ModelRequest(parts=[]))
                        return

                    # For thinking-only responses without recoverable text, fall through to the
                    # normal retry prompt below.

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
                    elif isinstance(part, _messages.BuiltinToolCallPart):
                        # Text parts before a built-in tool call are essentially thoughts,
                        # not part of the final result output, so we reset the accumulated text
                        text = ''
                        yield _messages.BuiltinToolCallEvent(part)  # pyright: ignore[reportDeprecated]
                    elif isinstance(part, _messages.BuiltinToolReturnPart):
                        yield _messages.BuiltinToolResultEvent(part)  # pyright: ignore[reportDeprecated]
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
                    ctx.state.increment_retries(ctx.deps.max_result_retries, error=e)
                    self._next_node = ModelRequestNode[DepsT, NodeRunEndT](_messages.ModelRequest(parts=[e.tool_retry]))

            self._events_iterator = _run_stream()

        async for event in self._events_iterator:
            yield event

    async def _handle_tool_calls(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        tool_calls: list[_messages.ToolCallPart],
    ) -> AsyncIterator[_messages.HandleResponseEvent]:
        run_context = build_run_context(ctx)
        run_context = replace(
            run_context,
            retry=ctx.state.retries,
            max_retries=ctx.deps.max_result_retries,
        )

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

            self._next_node = ModelRequestNode[DepsT, NodeRunEndT](_messages.ModelRequest(parts=output_parts))

    @staticmethod
    def _recover_text_from_message_history(message_history: list[_messages.ModelMessage]) -> str | None:
        """Search backward through message history for recoverable text from a previous model response.

        This handles cases where the model returned text alongside tool calls (so the text was
        discarded in favor of executing the tools) and subsequently returned an empty or
        thinking-only response. Returns the recovered text, or None if no text was found.
        """
        for message in reversed(message_history):
            if isinstance(message, _messages.ModelResponse):
                text = ''
                for part in message.parts:
                    if isinstance(part, _messages.TextPart):
                        text += part.content
                    elif isinstance(part, _messages.BuiltinToolCallPart):
                        # Text parts before a built-in tool call are essentially thoughts,
                        # not part of the final result output, so we reset the accumulated text.
                        text = ''  # pragma: no cover
                if text:
                    return text
        return None

    async def _handle_text_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        text: str,
        text_processor: _output.BaseOutputProcessor[NodeRunEndT],
    ) -> ModelRequestNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]]:
        run_context = _build_output_run_context(ctx)
        result_data = await text_processor.process(text, run_context=run_context)
        result_data = await _run_output_validators(ctx, result_data, run_context)
        return self._handle_final_result(ctx, result.FinalResult(result_data), [])

    async def _handle_image_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        image: _messages.BinaryImage,
    ) -> ModelRequestNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]]:
        run_context = _build_output_run_context(ctx)
        result_data = await _run_output_validators(ctx, cast(NodeRunEndT, image), run_context)
        return self._handle_final_result(ctx, result.FinalResult(result_data), [])

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
            messages.append(_messages.ModelRequest(parts=tool_responses, run_id=ctx.state.run_id, timestamp=now_utc()))

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
    run_context = RunContext[DepsT](
        deps=ctx.deps.user_deps,
        agent=ctx.deps.agent,
        model=ctx.deps.model,
        usage=ctx.state.usage,
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
        metadata=ctx.state.metadata,
        tool_manager=ctx.deps.tool_manager,
    )
    validation_context = build_validation_context(ctx.deps.validation_context, run_context)
    run_context = replace(run_context, validation_context=validation_context)
    return run_context


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
    """Build a RunContext with global output retry info for output validation."""
    run_context = build_run_context(ctx)
    return replace(
        run_context,
        retry=ctx.state.retries,
        max_retries=ctx.deps.max_result_retries,
    )


async def _run_output_validators(
    ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
    result_data: NodeRunEndT,
    run_context: RunContext[DepsT],
) -> NodeRunEndT:
    """Run all output validators on the result data."""
    for validator in ctx.deps.output_validators:
        result_data = await validator.validate(result_data, run_context)
    return result_data


def _emit_skipped_output_tool(
    call: _messages.ToolCallPart,
    message: str,
    output_parts: list[_messages.ModelRequestPart],
    *,
    args_valid: bool | None = None,
) -> Iterator[_messages.HandleResponseEvent]:
    """Yield events for an output tool call that was skipped, and append the result to output_parts."""
    yield _messages.FunctionToolCallEvent(call, args_valid=args_valid)
    part = _messages.ToolReturnPart(
        tool_name=call.tool_name,
        content=message,
        tool_call_id=call.tool_call_id,
    )
    output_parts.append(part)
    yield _messages.FunctionToolResultEvent(part)


async def process_tool_calls(  # noqa: C901
    tool_manager: ToolManager[DepsT],
    tool_calls: list[_messages.ToolCallPart],
    tool_call_results: dict[str, DeferredToolResult | Literal['skip']] | None,
    tool_call_metadata: dict[str, dict[str, Any]] | None,
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
        # `final_result` can be passed into `process_tool_calls` from `Agent.run_stream`
        # when streaming and there's already a final result
        if final_result and final_result.tool_call_id == call.tool_call_id:
            part = _messages.ToolReturnPart(
                tool_name=call.tool_name,
                content='Final result processed.',
                tool_call_id=call.tool_call_id,
            )
            output_parts.append(part)
        # A final result is already set and this strategy skips remaining output tools
        elif ctx.deps.end_strategy in ('early', 'graceful') and final_result:
            for event in _emit_skipped_output_tool(
                call, 'Output tool not used - a final result was already processed.', output_parts, args_valid=None
            ):
                yield event
        # No final result yet, or exhaustive strategy processes all output tools
        else:
            # Validate and execute the output tool call — unlike deferred tools,
            # output tools track retries and can be skipped if a final_result exists.
            try:
                validated = await tool_manager.validate_tool_call(call)
            except exceptions.UnexpectedModelBehavior as e:
                # If we already have a valid final result, don't fail the entire run
                # This allows exhaustive strategy to complete successfully when at least one output tool is valid
                if final_result:
                    # If output tool fails when we already have a final result, skip it without retrying
                    for event in _emit_skipped_output_tool(
                        call, 'Output tool not used - output failed validation.', output_parts, args_valid=False
                    ):
                        yield event
                    continue
                ctx.state.check_incomplete_tool_call()  # pragma: lax no cover
                tool = tool_manager.tools.get(call.tool_name) if tool_manager.tools else None  # pragma: lax no cover
                max_retries = tool.max_retries if tool else ctx.deps.max_result_retries  # pragma: lax no cover
                raise exceptions.UnexpectedModelBehavior(  # pragma: lax no cover
                    f'Exceeded maximum retries ({max_retries}) for output validation'
                ) from (e.__cause__ or e)

            if not validated.args_valid:
                assert validated.validation_error is not None
                if final_result:
                    for event in _emit_skipped_output_tool(
                        call, 'Output tool not used - output failed validation.', output_parts, args_valid=False
                    ):
                        yield event
                    continue

                yield _messages.FunctionToolCallEvent(call, args_valid=False)
                output_parts.append(validated.validation_error.tool_retry)
                yield _messages.FunctionToolResultEvent(validated.validation_error.tool_retry)
                ctx.state.retries += 1
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
                ctx.state.check_incomplete_tool_call()  # pragma: lax no cover
                max_retries = (
                    validated.tool.max_retries if validated.tool else ctx.deps.max_result_retries
                )  # pragma: lax no cover
                raise exceptions.UnexpectedModelBehavior(  # pragma: lax no cover
                    f'Exceeded maximum retries ({max_retries}) for output validation'
                ) from (e.__cause__ or e)
            except ToolRetryError as e:
                yield _messages.FunctionToolCallEvent(call, args_valid=True)
                output_parts.append(e.tool_retry)
                yield _messages.FunctionToolResultEvent(e.tool_retry)
                ctx.state.retries += 1
                continue

            part = _messages.ToolReturnPart(
                tool_name=call.tool_name,
                content='Final result processed.',
                tool_call_id=call.tool_call_id,
            )
            output_parts.append(part)

            # Use the first valid output tool's result as the final result
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

    # Unknown tools use per-tool retry handling via ModelRetry in ToolManager
    if tool_calls_by_kind['unknown']:
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
                ctx.state.check_incomplete_tool_call()
                yield _messages.FunctionToolCallEvent(call, args_valid=False)
                raise
            validated_calls[call.tool_call_id] = validated
            yield _messages.FunctionToolCallEvent(call, args_valid=validated.args_valid)

        async for event in _call_tools(
            tool_manager=tool_manager,
            tool_calls=calls_to_run,
            tool_call_results=calls_to_run_results,
            validated_calls=validated_calls,
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
    output_parts: list[_messages.ModelRequestPart],
    output_deferred_calls: dict[Literal['external', 'unapproved'], list[_messages.ToolCallPart]],
    output_deferred_metadata: dict[str, dict[str, Any]],
) -> AsyncIterator[_messages.HandleResponseEvent]:
    tool_parts_by_index: dict[int, _messages.ModelRequestPart] = {}
    user_parts_by_index: dict[int, _messages.UserPromptPart] = {}
    deferred_calls_by_index: dict[int, Literal['external', 'unapproved']] = {}
    deferred_metadata_by_index: dict[int, dict[str, Any] | None] = {}

    async def handle_call_or_result(
        coro_or_task: Awaitable[
            tuple[_messages.ToolReturnPart | _messages.RetryPromptPart, str | Sequence[_messages.UserContent] | None]
        ]
        | Task[
            tuple[_messages.ToolReturnPart | _messages.RetryPromptPart, str | Sequence[_messages.UserContent] | None]
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
        except BaseException:
            # Cancel any still-running sibling tasks so they don't become
            # orphaned asyncio tasks when a non-CancelledError exception
            # (e.g. RuntimeError, ConnectionError) propagates out of
            # handle_call_or_result().
            for task in tasks:
                task.cancel()
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
    """Populate deferred calls and metadata from indexed mappings."""
    for k in sorted(deferred_calls_by_index):
        call = tool_calls[k]
        output_deferred_calls[deferred_calls_by_index[k]].append(call)
        metadata = deferred_metadata_by_index[k]
        if metadata is not None:
            output_deferred_metadata[call.tool_call_id] = metadata


async def _call_tool(
    tool_manager: ToolManager[DepsT],
    tool_call: ValidatedToolCall[DepsT] | _messages.ToolCallPart,
    tool_call_result: DeferredToolResult | None,
) -> tuple[_messages.ToolReturnPart | _messages.RetryPromptPart, str | Sequence[_messages.UserContent] | None]:
    if isinstance(tool_call, ValidatedToolCall):
        validated = tool_call
        call = tool_call.call
    else:
        validated = None
        call = tool_call

    tool_result: Any
    try:
        if tool_call_result is None or isinstance(tool_call_result, ToolApproved):
            if validated is not None:
                tool_result = await tool_manager.execute_tool_call(validated)
            else:
                raise RuntimeError('Expected validated tool call')  # pragma: no cover
        elif isinstance(tool_call_result, ToolDenied):
            return _messages.ToolReturnPart(
                tool_name=call.tool_name,
                content=tool_call_result.message,
                tool_call_id=call.tool_call_id,
                outcome='denied',
            ), None
        elif isinstance(tool_call_result, exceptions.ModelRetry):
            m = _messages.RetryPromptPart(
                content=tool_call_result.message,
                tool_name=call.tool_name,
                tool_call_id=call.tool_call_id,
            )
            raise ToolRetryError(m)
        elif isinstance(tool_call_result, _messages.RetryPromptPart):
            tool_call_result.tool_name = call.tool_name
            tool_call_result.tool_call_id = call.tool_call_id
            raise ToolRetryError(tool_call_result)
        else:
            tool_result = tool_call_result
    except ToolRetryError as e:
        return e.tool_retry, None

    if isinstance(tool_result, _messages.ToolReturn):
        tool_return = cast(_messages.ToolReturn[Any], tool_result)
    elif isinstance(tool_result, list) and any(
        isinstance(i, _messages.ToolReturn) for i in cast(list[Any], tool_result)
    ):
        raise exceptions.UserError(
            f'The return value of tool {call.tool_name!r} contains invalid nested `ToolReturn` objects. '
            f'`ToolReturn` should be used directly.'
        )
    else:
        tool_return = _messages.ToolReturn[Any](return_value=cast(Any, tool_result))

    return_part = _messages.ToolReturnPart(
        tool_name=call.tool_name,
        tool_call_id=call.tool_call_id,
        content=tool_return.return_value,
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
) -> int:
    """Return the first index that should be included in `new_messages()`."""
    if resumed_request is not None:
        for index, message in enumerate(messages):
            if message is resumed_request:
                # Requests passed in via `message_history` are prior context,
                # even if they are stamped with the current `run_id` for adapter
                # bookkeeping.
                return index + 1

        for index in range(len(messages) - 1, -1, -1):
            if _is_same_request(messages[index], resumed_request):
                return index + 1
    return _first_run_id_index(messages, run_id)


def _is_same_request(message: _messages.ModelMessage, request: _messages.ModelRequest) -> bool:
    if not isinstance(message, _messages.ModelRequest):
        return False
    if message is request:
        return True  # pragma: no cover
    # Intentionally excludes run_id: the resumed request may not have
    # run_id set yet when this comparison is performed.
    return (
        message.parts == request.parts
        and message.timestamp == request.timestamp
        and message.instructions == request.instructions
        and message.metadata == request.metadata
    )


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
