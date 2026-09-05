from __future__ import annotations as _annotations

import dataclasses
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

from typing_extensions import TypeVar, assert_never

from pydantic_graph import End, GraphRunContext
from pydantic_graph.basenode import NodeRunEndT

from .. import _output, exceptions, messages as _messages, result
from .._deferred_capabilities import _parse_loaded_capabilities  # pyright: ignore[reportPrivateUsage]
from .._run_context import AnchoredEvidence, dispatch_event_stream
from .._tool_execution import process_tool_calls
from .._utils import dataclasses_no_defaults_repr, now_utc
from ..exceptions import ToolRetryError
from ..tools import DeferredToolResult
from ..toolsets._tool_search import _discovered_tool_names_in_order  # pyright: ignore[reportPrivateUsage]
from .graph import AgentNode
from .state import (
    GraphAgentDeps,
    GraphAgentState,
    _build_output_run_context,
    _refresh_discovered_tool_names,
    build_run_context,
)

if TYPE_CHECKING:
    from .model_request import ModelRequestNode as ModelRequestNode


DepsT = TypeVar('DepsT')

__all__ = 'CallToolsNode', '_with_event_stream_buffer'


async def _with_event_stream_buffer(
    stream: AsyncIterator[_messages.AgentStreamEvent],
    event_stream_buffer: list[_messages.AgentStreamEvent],
) -> AsyncIterator[_messages.AgentStreamEvent]:
    """Drain buffered run events at the start and end of a node stream.

    Events buffered while the node stream is live are yielded by the stream itself, as soon as they
    are emitted (see `_iter_completed_or_buffered`); draining them here as well could yield them
    ahead of an earlier event the stream is about to deliver, inverting emission order.
    """
    while event_stream_buffer:
        yield event_stream_buffer.pop(0)
    async for event in stream:
        yield event
    while event_stream_buffer:
        yield event_stream_buffer.pop(0)


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

    _wrapped_events_iterator: AsyncIterator[_messages.AgentStreamEvent] | None = field(
        default=None, init=False, repr=False
    )
    _next_node: ModelRequestNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]] | None = field(
        default=None, init=False, repr=False
    )
    _stream_error: BaseException | None = field(default=None, init=False, repr=False)

    async def run(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> ModelRequestNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]]:
        async with self.stream(ctx):
            pass
        if self._next_node is not None:
            return self._next_node
        # If the stream raised an error that was caught by an external consumer
        # (e.g. UIEventStream.transform_stream), _next_node will not have been set.
        # Re-raise the original error instead of a confusing assertion.
        if self._stream_error is not None:
            raise self._stream_error.with_traceback(self._stream_error.__traceback__)
        raise exceptions.AgentRunError('the stream should set `self._next_node` before it ends')  # pragma: no cover

    @asynccontextmanager
    async def stream(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> AsyncGenerator[AsyncIterator[_messages.AgentStreamEvent]]:
        """Process the model response and yield events for the start and end of each function tool call."""
        stream = self._wrapped_stream(ctx)
        try:
            yield stream

            # Run the stream to completion if it was not finished:
            async for _event in stream:
                pass
        finally:
            # The capability-wrapped stream is memoized on the node, so a consumer that bails out
            # leaves the chain suspended along with anything a capability parked on it.
            # The root capability's wrapper is always a generator, so the guard never falls through
            # today; it's here because `wrap_run_event_stream` may return any `AsyncIterable`.
            aclose: Callable[[], Awaitable[None]] | None = getattr(stream, 'aclose', None)
            if aclose is not None:  # pragma: no branch
                await aclose()

    def _wrapped_stream(
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> AsyncIterator[_messages.AgentStreamEvent]:
        """This node's events, wrapped in the capability chain exactly once.

        `run()` enters `stream()` itself, so a caller that already streamed this node under
        `agent.iter()` makes that the second entry. The wrapper has to be built once and reused:
        rebuilding it would run every capability's `wrap_run_event_stream` again over an exhausted
        stream, duplicating whatever setup or teardown it does outside its own iteration.
        """
        if self._wrapped_events_iterator is None:
            run_context = build_run_context(ctx)
            inner = dispatch_event_stream(
                run_context, _with_event_stream_buffer(self._run_stream(ctx), ctx.state.event_stream_buffer)
            )
            self._wrapped_events_iterator = aiter(
                ctx.deps.root_capability.wrap_run_event_stream(run_context, stream=inner)
            )
        return self._wrapped_events_iterator

    async def _run_stream(  # noqa: C901
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> AsyncIterator[_messages.AgentStreamEvent]:
        from .model_request import ModelRequestNode

        # `_wrapped_stream` builds this generator once per node, so there is no caching to do here.
        output_schema = ctx.deps.output_schema

        async def _run_stream() -> AsyncIterator[_messages.AgentStreamEvent]:  # noqa: C901
            if self.model_response.state == 'suspended':
                # A suspended turn is not a completed response to handle: its partial parts could
                # match an output schema and end the run on mid-turn output while the provider's
                # server-side job keeps running. This is reachable when a consumer detaches a
                # streamed background run under `agent.iter` and then keeps driving the graph, handing
                # this node the suspended response. Symmetric with `UserPromptNode`'s suspended guard.
                raise exceptions.UserError(
                    'Cannot handle a suspended model response as a completed turn. '
                    'Resume it by running the agent with this message history and no new prompt.'
                )

            is_empty = not self.model_response.parts
            # A `TextPart` with empty content carries no text output; adapters preserve such parts
            # (e.g. when a gateway returns a text item with `text: null`) so their IDs round-trip.
            is_blank_text_only = not is_empty and all(
                isinstance(p, _messages.TextPart) and not p.content for p in self.model_response.parts
            )
            is_thinking_only = (
                not is_empty
                and not is_blank_text_only
                and all(
                    isinstance(p, _messages.ThinkingPart) or (isinstance(p, _messages.TextPart) and not p.content)
                    for p in self.model_response.parts
                )
            )

            if is_empty or is_blank_text_only or is_thinking_only:
                # No actionable output was returned by the model.

                # Don't retry if the token limit was exceeded, possibly during thinking.
                if self.model_response.finish_reason == 'length':
                    raise exceptions.UnexpectedModelBehavior(
                        f'Model token limit ({ctx.state.last_max_tokens or "provider default"}) exceeded before any response was generated. Increase the `max_tokens` model setting, or simplify the prompt to result in a shorter response that will fit within the limit.'
                    )

                # Check for content filter on a response with no content
                if (is_empty or is_blank_text_only) and self.model_response.finish_reason == 'content_filter':
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

                # If the output type allows `None`, a response with no text output is a valid result:
                # it signals that the model has nothing to say. Some models emit only thinking after
                # completing the task via a tool call, and forcing a retry just makes them produce
                # unnecessary follow-up text.
                if output_schema.allows_none:
                    run_context = _build_output_run_context(ctx)
                    try:
                        result_data = await _output.run_none_process_hooks(
                            capability=ctx.deps.root_capability,
                            run_context=run_context,
                            schema=output_schema,
                            output_validators=ctx.deps.output_validators,
                        )
                        self._next_node = self._handle_final_result(
                            ctx, result.FinalResult(cast(NodeRunEndT, result_data)), []
                        )
                    except ToolRetryError as e:
                        ctx.state.consume_output_retry(ctx.deps.max_output_retries, error=e)
                        self._next_node = ModelRequestNode[DepsT, NodeRunEndT](
                            _messages.ModelRequest(parts=[e.tool_retry])
                        )
                    return

                # For responses with no text output, fall through to the normal retry prompt
                # below. That prompt is built from the output schema and available tools, so it
                # tells the model which kinds of output are actually valid (text, tool call,
                # and/or image) rather than assuming text is always an option.

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
                elif isinstance(part, _messages.NativeToolCallPart):
                    # Text parts before a native tool call are essentially thoughts,
                    # not part of the final result output, so we reset the accumulated text.
                    # The part itself was already surfaced through `PartStartEvent` / `PartDeltaEvent`.
                    text = ''
                elif isinstance(part, _messages.NativeToolReturnPart):
                    # Already surfaced through `PartStartEvent` / `PartDeltaEvent`.
                    pass
                elif isinstance(part, _messages.ThinkingPart):
                    pass
                elif isinstance(part, _messages.CompactionPart):
                    if part.content:
                        compaction_text += part.content
                elif isinstance(part, _messages.SpeechPart):
                    # No standard model produces realtime audio parts, but a custom model (e.g. a
                    # `FunctionModel` bridging one) can. Its transcript is the response's text —
                    # `ModelResponse.text` already reads it that way — so treat it like a `TextPart`
                    # rather than judging the response empty and forcing a retry.
                    text += part.content
                else:
                    assert_never(part)

            # Use compaction content as text fallback when the response has no other
            # actionable text (e.g. Anthropic pause_after_compaction=True)
            if not text and compaction_text:
                text = compaction_text

            try:
                # We generally prioritize at least executing tool calls if they are present.
                # This accounts for cases like Anthropic returns that might contain a text response
                # and a tool call response, where the text response just indicates the tool call will happen.
                # The exception is `end_strategy='early'`: if the response also carries a valid non-tool
                # output (schema-validated text, or an image) alongside plain function tool calls, that
                # output is already the final result, so `_handle_tool_calls` skips those tools and ends the
                # run — matching the way `'early'` skips function tools once an output tool call succeeds.
                # (Output tool calls and deferred tool calls are left to normal processing, so a co-emitted
                # one still wins/surfaces rather than being preempted by the text.)
                alternatives: list[str] = []
                if tool_calls:
                    response_output = (text, files) if ctx.deps.end_strategy == 'early' else None
                    async for event in self._handle_tool_calls(ctx, tool_calls, response_output=response_output):
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
                ctx.state.consume_output_retry(ctx.deps.max_output_retries, error=e)
                self._next_node = ModelRequestNode[DepsT, NodeRunEndT](_messages.ModelRequest(parts=[e.tool_retry]))

        try:
            async for event in _run_stream():
                yield event
        except GeneratorExit:
            # Being closed is teardown, not a stream failure. `run()` re-raises `_stream_error` when
            # the stream ended without setting a next node, and a bare `GeneratorExit` surfacing from
            # a coroutine there would tell the caller nothing about what actually went wrong.
            raise
        except BaseException as e:
            self._stream_error = e
            raise

    async def _handle_tool_calls(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        tool_calls: list[_messages.ToolCallPart],
        *,
        response_output: tuple[str, list[_messages.BinaryContent]] | None = None,
    ) -> AsyncIterator[_messages.AgentStreamEvent]:
        from .model_request import ModelRequestNode

        # Re-derive reveals now that the response is in history: a provider-side tool search
        # reveals a tool *inside* the response that goes on to call it, and the model saw that
        # schema before emitting the call. The step-start refresh ran before the response existed.
        # A `load_capability` call in the same response is deliberately *not* covered — it has not
        # executed yet, so its capability stays unavailable and its tools stay uncallable until the
        # next request carries the capability's instructions.
        _refresh_discovered_tool_names(ctx)

        run_context = build_run_context(ctx)
        evidence_window = _messages._post_compaction_window_for_response(  # pyright: ignore[reportPrivateUsage]
            ctx.state.message_history, self.model_response
        )
        # Held in a local because it lands in two places below.
        anchored_evidence = AnchoredEvidence(
            discovered_tool_names=frozenset(_discovered_tool_names_in_order(evidence_window))
            - ctx.deps.discovered_tool_names,
            loaded_capability_ids=frozenset(_parse_loaded_capabilities(evidence_window))
            - ctx.deps.loaded_capability_ids,
        )
        run_context = replace(
            run_context,
            retry=ctx.state.output_retries_used,
            max_retries=ctx.deps.tool_manager.default_max_retries,
            _anchored_evidence=anchored_evidence,
        )

        # This will raise errors for any tool name conflicts
        ctx.deps.tool_manager = await ctx.deps.tool_manager.for_run_step(run_context)
        # The manager was already prepared for this same run step before the model request, so
        # `for_run_step` deliberately returns it unchanged, keeping the retries it accumulated —
        # which is why the evidence lands field by field rather than by swapping in `run_context`.
        # Only the retrospective evidence is carried: replacing the prospective shared sets would
        # affect the next request's reveal pruning and search ranking.
        assert ctx.deps.tool_manager.ctx is not None
        ctx.deps.tool_manager.ctx._anchored_evidence = anchored_evidence  # pyright: ignore[reportPrivateUsage]

        # Under `end_strategy='early'`, `response_output` holds the response's `(text, files)`. If it carries a
        # valid non-tool output (schema-validated text, or an image) and every co-emitted tool call is a plain
        # function tool, that output is the final result and the tools are recorded as skipped.
        #
        # We check the tool kinds here (rather than letting `process_tool_calls` sort it out) for two reasons:
        # output and deferred (external/unapproved) tool calls must go through normal processing, and
        # `_process_response_output` runs the output validators, so we only want to invoke it once we know the
        # response output can actually win. `for_run_step` above populated the tool defs used here.
        #
        # The precedence is deliberate: calling an output tool is an explicit "finish the run" signal, and a
        # deferred call may need an external result or human approval — whereas the model's text may just be
        # supporting prose (it doesn't know we might treat that text as final), so text must not silently
        # cancel either. A co-emitted output tool call therefore still produces the final result, and a
        # co-emitted deferred call is still surfaced, rather than being preempted by the text.
        final_result: result.FinalResult[NodeRunEndT] | None = None
        if response_output is not None and all(
            (tool_def := ctx.deps.tool_manager.get_tool_def(call.tool_name)) is None or tool_def.kind == 'function'
            for call in tool_calls
        ):
            text, files = response_output
            final_result = await self._process_response_output(ctx, text=text, files=files)

        output_parts: list[_messages.ModelRequestPart] = []
        output_final_result: deque[result.FinalResult[NodeRunEndT]] = deque(maxlen=1)

        try:
            # When `final_result` is set (schema-validated text or image output already won under
            # `end_strategy='early'`), `process_tool_calls` records the tool calls as skipped rather than
            # executing them.
            async for event in process_tool_calls(
                tool_manager=ctx.deps.tool_manager,
                tool_calls=tool_calls,
                tool_call_results=self.tool_call_results,
                tool_call_metadata=self.tool_call_metadata,
                final_result=final_result,
                ctx=ctx,
                output_parts=output_parts,
                output_final_result=output_final_result,
            ):
                yield event
        except BaseException:
            # Capture the partial tool returns collected so far. State is 'interrupted'
            # so `capture_run_messages` consumers can detect partial state. The user prompt
            # is intentionally omitted: this request was never sent to the model.
            #
            # It's appended even when no tool finished and it's therefore empty: this node only runs
            # for a response that made tool calls, so the marker is what tells the resume path that
            # those calls will never be answered and need synthesized `'interrupted'` returns.
            # Without it, a run cancelled during its first (or only) tool call would leave a history
            # that can't take a new prompt.
            ctx.state.message_history.append(
                _messages.ModelRequest(
                    parts=list(output_parts),
                    run_id=ctx.state.run_id,
                    conversation_id=ctx.state.conversation_id,
                    timestamp=now_utc(),
                    state='interrupted',
                )
            )
            raise

        if output_final_result:
            final_result = output_final_result[0]
            self._next_node = self._handle_final_result(ctx, final_result, output_parts)
        else:
            # Add user prompt if provided, after all tool return parts
            if self.user_prompt is not None:
                output_parts.append(_messages.UserPromptPart(self.user_prompt))

            self._next_node = ModelRequestNode[DepsT, NodeRunEndT](_messages.ModelRequest(parts=output_parts))

    async def _process_response_output(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        *,
        text: str,
        files: list[_messages.BinaryContent],
    ) -> result.FinalResult[NodeRunEndT] | None:
        """Build the response's non-tool output result (an image, or schema-validated text), or `None`.

        Used under `end_strategy='early'` to decide whether a response that also contains function tool calls
        already carries a final result. Images take precedence over text, matching the order the no-tool-calls
        path handles them in.

        Only text that's validated against a schema can preempt tool calls — i.e. the object output processor
        used by [`NativeOutput`][pydantic_ai.output.NativeOutput],
        [`PromptedOutput`][pydantic_ai.output.PromptedOutput], and a bare structured type (auto mode). There
        the model was told to produce the final output as its text, so text that validates is a deliberate
        final result. Plain, unstructured text output (`str`, [`TextOutput`][pydantic_ai.output.TextOutput], or
        a `str` fallback in a larger schema) accepts *any* text, so the model's preamble — which it emits with
        no signal that we'd treat it as final — must not silently win and skip the tools.

        Returns `None` when the response carries no usable output — e.g. schema-validated text or an image that
        fails validation — so the caller runs the tool calls instead. Unlike a failed output *tool* call, this
        doesn't consume an output retry or surface a retry prompt: running the tools is the correction.
        """
        output_schema = ctx.deps.output_schema
        try:
            if output_schema.allows_image:
                if image := next((file for file in files if isinstance(file, _messages.BinaryImage)), None):
                    return await self._process_image_response(ctx, image)
            if (
                (text_processor := output_schema.text_processor)
                and isinstance(text_processor, _output.BaseObjectOutputProcessor)
                and text
            ):
                return await self._process_text_response(ctx, text, text_processor)
        except ToolRetryError:
            return None
        return None

    async def _handle_text_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        text: str,
        text_processor: _output.BaseOutputProcessor[NodeRunEndT],
    ) -> ModelRequestNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]]:
        return self._handle_final_result(ctx, await self._process_text_response(ctx, text, text_processor), [])

    async def _process_text_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        text: str,
        text_processor: _output.BaseOutputProcessor[NodeRunEndT],
    ) -> result.FinalResult[NodeRunEndT]:
        run_context = _build_output_run_context(ctx)
        schema = ctx.deps.output_schema

        result_data = await _output.run_output_with_hooks(
            text_processor,
            text=text,
            run_context=run_context,
            capability=ctx.deps.root_capability,
            schema=schema,
            output_validators=ctx.deps.output_validators,
        )

        return result.FinalResult(result_data)

    async def _handle_image_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        image: _messages.BinaryImage,
    ) -> ModelRequestNode[DepsT, NodeRunEndT] | End[result.FinalResult[NodeRunEndT]]:
        return self._handle_final_result(ctx, await self._process_image_response(ctx, image), [])

    async def _process_image_response(
        self,
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
        image: _messages.BinaryImage,
    ) -> result.FinalResult[NodeRunEndT]:
        run_context = _build_output_run_context(ctx)
        schema = ctx.deps.output_schema
        result_data = await _output.run_image_process_hooks(
            image,
            capability=ctx.deps.root_capability,
            run_context=run_context,
            schema=schema,
            output_validators=ctx.deps.output_validators,
        )

        return result.FinalResult(result_data)

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
            messages.append(
                _messages.ModelRequest(
                    parts=tool_responses,
                    run_id=ctx.state.run_id,
                    conversation_id=ctx.state.conversation_id,
                    timestamp=now_utc(),
                )
            )

        return End(final_result)

    __repr__ = dataclasses_no_defaults_repr
