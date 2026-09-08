from __future__ import annotations as _annotations

import asyncio
import dataclasses
import inspect
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import field, replace
from typing import TYPE_CHECKING, Any

from typing_extensions import TypeVar

from pydantic_graph import GraphRunContext
from pydantic_graph.basenode import NodeRunEndT

from .. import _output, exceptions, messages as _messages, models, result
from .._genai_prices import best_effort_price, fill_response_cost
from .._instrumentation import (
    get_instructions as _get_history_instructions,
    get_instructions_source as _get_history_instructions_source,
    time_to_first_chunk_ctx,
)
from .._utils import cancel_and_drain, dataclasses_no_defaults_repr, fill_run_metadata, now_utc
from ..models import CompletedStreamedResponse, ModelRequestContext
from ..native_tools import AbstractNativeTool
from ..native_tools._tool_search import ToolSearchTool
from ..settings import ModelSettings
from ..tools import AgentNativeTool, RunContext, ToolDefinition
from ..toolsets._instruction_collection import collect_toolset_instructions
from .graph import AgentNode
from .history import _clean_message_history, _first_new_message_index
from .model_call import _cancel_task, _resolve_interrupted_stream_state, model_request, model_request_stream
from .state import (
    GraphAgentDeps,
    GraphAgentState,
    _refresh_discovered_tool_names,
    _refresh_loaded_capability_ids,
    _revealed_tool_names,
    _select_model,
    _with_outgoing_reveal_state,
    build_run_context,
)

if TYPE_CHECKING:
    from .model_response import CallToolsNode as CallToolsNode


DepsT = TypeVar('DepsT')
T = TypeVar('T')

__all__ = 'ModelRequestNode', '_get_instructions'


def _ensure_model_supports_streaming(model: models.Model) -> None:
    if type(model).request_stream is models.Model.request_stream:
        raise exceptions.UserError(
            f'{type(model).__name__} does not support streamed requests. This step needs to stream '
            'either because the run itself is streamed (`agent.run_stream()`, `agent.run_stream_events()`), '
            'or because a capability registers a `wrap_run_event_stream` hook and so needs events to observe. '
            'Implement `request_stream()` on the model, or use a non-streamed run without such a capability.'
        )


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

    parts.extend(await collect_toolset_instructions(ctx.deps.tool_manager.toolset, run_context))

    return parts or None


def _apply_instruction_parts(
    request: _messages.ModelRequest, instruction_parts: list[_messages.InstructionPart] | None
) -> None:
    """Render the instruction parts being sent onto the request that records them.

    `ModelRequestParameters.instruction_parts` is what the model reads, so a `before_model_request`
    hook that rewrites the parts would otherwise leave message history and OTel reporting
    instructions the model never received.

    `None` means "unset" rather than "no instructions" — it's what makes `Model._get_instruction_parts`
    fall back to the request's own `instructions` — so it leaves the request alone.
    """
    if instruction_parts is not None:
        request.instructions = _messages.InstructionPart.join(instruction_parts)


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
    # and surface an inert native tool in `ModelRequestParameters` snapshots. `prepare_request`
    # applies the same drop during resolution, but instrumentation and durable-execution
    # payloads observe the parameters BEFORE resolution, so filtering here too is what keeps
    # the observed request shape honest. Non-optional `ToolSearchTool` instances (user-passed)
    # are preserved so the request still fails loudly on unsupported models.
    has_tool_search_corpus = any(t.with_native == ToolSearchTool.kind for t in function_tools)
    if not has_tool_search_corpus:
        # Confine the corpus-empty drop to `ToolSearchTool`: other optional native tools
        # (e.g. a hypothetical `WebSearchTool(optional=True)`) don't have a corpus and
        # shouldn't be dropped here — they only get dropped on the unsupported-on-this-model
        # path in `Model.prepare_request`.
        native_tools = [t for t in native_tools if not (isinstance(t, ToolSearchTool) and t.optional)]

    deferred_capability_ids = {
        capability_id
        for capability_id, capability in run_context.capabilities.items()
        if capability.defer_loading is True
    }

    return models.ModelRequestParameters(
        function_tools=function_tools,
        native_tools=native_tools,
        deferred_capability_ids=deferred_capability_ids,
        revealed_tool_names=_revealed_tool_names(
            run_context.discovered_tool_names,
            function_tools,
            deferred_capability_ids=deferred_capability_ids,
            loaded_capability_ids=run_context.loaded_capability_ids,
        ),
        output_mode=output_schema.mode,
        output_tools=output_tools,
        output_object=output_schema.object_def,
        prompted_output_template=prompted_output_template,
        allow_text_output=output_schema.allows_text,
        allow_image_output=output_schema.allows_image,
        instruction_parts=instruction_parts,
    )


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
    async def stream(  # noqa: C901
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
            try:
                yield agent_stream
            finally:
                await agent_stream.aclose_events()
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
            _ensure_model_supports_streaming(req_ctx.model)
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
        root_capability = ctx.deps.root_capability
        if root_capability._has_wrap_model_request:  # pyright: ignore[reportPrivateUsage]
            wrap_awaitable = root_capability.wrap_model_request(
                run_context,
                request_context=wrap_request_context,
                handler=_streaming_handler,
            )
        else:
            wrap_awaitable = _streaming_handler(wrap_request_context)
        wrap_task = asyncio.create_task(wrap_awaitable)

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
                agent_stream = self._build_agent_stream(ctx, dummy_sr, model_request_parameters)
                try:
                    yield agent_stream
                finally:
                    await agent_stream.aclose_events()
                return
            self._did_stream = True
            ctx.state.usage.requests += 1
            replay_sr = CompletedStreamedResponse(
                model_response,
                model_request_parameters=model_request_parameters,
                replay_events=True,
            )
            agent_stream = self._build_agent_stream(ctx, replay_sr, model_request_parameters)
            try:
                yield agent_stream
            finally:
                # The event iterator is memoized on the stream, so a consumer that broke out early
                # leaves the capability chain suspended. Close it now that the node is done with it.
                await agent_stream.aclose_events()
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
            try:
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
                        fill_response_cost(partial_response)
                        ctx.state.usage.incr(partial_response.usage)
                        ctx.state.message_history.append(partial_response)
                else:
                    try:
                        try:
                            model_response = await wrap_task
                        except exceptions.ModelRetry:
                            raise  # Propagate to outer handler
                        except Exception as e:
                            if not root_capability._has_on_model_request_error:  # pyright: ignore[reportPrivateUsage]
                                raise
                            model_response = await root_capability.on_model_request_error(
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
            finally:
                # The event iterator is memoized on the stream, so a consumer that broke out early
                # leaves the capability chain suspended. Close it now that the node is done with it.
                await agent_stream_holder[0].aclose_events()

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
        root_capability = ctx.deps.root_capability
        try:
            try:
                if root_capability._has_wrap_model_request:  # pyright: ignore[reportPrivateUsage]
                    model_response = await root_capability.wrap_model_request(
                        run_context,
                        request_context=request_context,
                        handler=model_handler,
                    )
                else:
                    model_response = await model_handler(request_context)
            except exceptions.SkipModelRequest as e:
                model_response = e.response
            except exceptions.ModelRetry:
                raise  # Propagate to outer handler
            except Exception as e:
                if not root_capability._has_on_model_request_error:  # pyright: ignore[reportPrivateUsage]
                    raise
                model_response = await root_capability.on_model_request_error(
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
            fill_run_metadata(self.request, run_id=ctx.state.run_id, conversation_id=ctx.state.conversation_id)
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

        # The hook may have rewritten the instruction parts the model will actually be sent, so bring
        # the request that records them back in step. It's the request this step created and set
        # instructions on above, which is not necessarily the last message anymore: a hook can append
        # further messages (e.g. `ToolSearch`'s auto-load synthesizes a call/return pair).
        _apply_instruction_parts(self.request, model_request_parameters.instruction_parts)

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

        # Reveal state is a property of the history actually sent to the model. A history
        # processor may remove or replace availability deltas, so the per-request parameters
        # must not retain the state derived from the unprocessed durable history. Derive AFTER
        # cleanup: cleanup strips evidence the processors orphaned (e.g. a search return whose
        # call was dropped), and counting stripped evidence would ship a "revealed" tool with no
        # reveal on the wire. (`prepare_messages` below only ever adds or reshapes evidence —
        # synthesis/translation — so this list is final for derivation purposes.)
        model_request_parameters = _with_outgoing_reveal_state(model_request_parameters, messages)

        # Hand off to the model class for any history shapes the active provider can't
        # ship on the wire — currently typed `NativeToolSearch*Part` instances translated
        # to local-shape `ToolSearch*Part` when they came from another provider or the
        # profile doesn't support `ToolSearchTool`.
        #
        # Lives on `Model.prepare_messages` rather than inline here for two reasons:
        # 1. The translation depends on `self.profile`, which is per-model state.
        # 2. `FallbackModel` defers the decision until it's picked an underlying model — so
        #    each candidate runs `prepare_messages` itself with its own profile when chosen.
        prepared = model.prepare_messages(messages, model_request_parameters)

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
            # Price this request's input tokens so the accumulated cost reflects them. Output tokens don't
            # exist yet, so this is a lower bound: it only catches a request whose input alone exceeds the limit.
            counted_price = best_effort_price(
                counted_usage,
                model_name=model.model_name,
                provider_api_url=model.base_url,
                provider_name=model.system,
            )
            counted_usage.cost = counted_price.total_price if counted_price is not None else None
            usage.incr(counted_usage)

            ctx.deps.usage_limits.check_per_request_input_tokens(counted_usage.input_tokens)

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

        model_request_parameters = _with_outgoing_reveal_state(model_request_parameters, messages)

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

        # A hook's rewrite has to land on the message that records the instructions this
        # continuation echoes back, which is the one they were read from — not necessarily the
        # resumed request. A trailing tool-return-only request carries none of its own (which is
        # why `_get_history_instructions` looks past it), so stamping that one would put
        # instructions on a message that was sent without any. Falling back to the resumed request
        # covers a hook adding instructions where the history had none to source.
        instructions_target = _get_history_instructions_source(base_messages) or ctx.deps.resumed_request
        if instructions_target is not None:
            _apply_instruction_parts(instructions_target, model_request_parameters.instruction_parts)

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
        from .model_response import CallToolsNode

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
            root_capability = ctx.deps.root_capability
            if not root_capability._has_on_model_request_error:  # pyright: ignore[reportPrivateUsage]
                raise exc
            return await root_capability.on_model_request_error(run_context, request_context=request_context, error=exc)
        return result_or_exc

    @staticmethod
    def _append_response(
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[Any, Any]],
        response: _messages.ModelResponse,
    ) -> None:
        """Append a model response to history, updating usage tracking."""
        fill_run_metadata(response, run_id=ctx.state.run_id, conversation_id=ctx.state.conversation_id)
        fill_response_cost(response)
        ctx.state.usage.incr(response.usage)
        if ctx.deps.usage_limits:  # pragma: no branch
            ctx.deps.usage_limits.check_tokens(ctx.state.usage)
            # More model responses may provide priceable usage, so only warn after the run successfully finishes.
            ctx.deps.usage_limits.check_cost(ctx.state.usage, warn_if_cost_unavailable=False)
            # For a continuation chain (Anthropic `pause_turn`, OpenAI background mode) the merged
            # response sums usage across segments (see `_check_continuation_usage`), so this caps the
            # chain's combined input rather than any single segment's — conservative, not lenient.
            ctx.deps.usage_limits.check_per_request_input_tokens(response.usage.input_tokens)
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
