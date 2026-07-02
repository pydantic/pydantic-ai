"""A realtime session that wraps a [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection] with automatic tool execution."""

from __future__ import annotations as _annotations

import asyncio
import dataclasses
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

import pydantic_core
from opentelemetry.trace import Span, SpanKind

from ..exceptions import UsageLimitExceeded
from ..usage import RunUsage, UsageLimits
from ._base import (
    AudioInput,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    ImageInput,
    InputTranscript,
    RealtimeConnection,
    RealtimeEvent,
    RealtimeInput,
    RealtimeSessionEvent,
    SessionError,
    TextInput,
    ToolCall,
    ToolCallCompleted,
    ToolCallStarted,
    ToolResult,
    Transcript,
    TruncateOutput,
    Usage,
)

if TYPE_CHECKING:
    from ..models.instrumented import InstrumentationSettings

ToolRunner = Callable[[str, dict[str, Any], str], Awaitable[str]]
"""Async callable executing a tool given its name, parsed arguments, and call id; returns the string result."""


def _as_event(item: object) -> RealtimeSessionEvent:
    """Unwrap a queue item: re-raise a background tool's exception, otherwise return the event."""
    if isinstance(item, Exception):
        raise item
    return cast('RealtimeSessionEvent', item)


def _transcript_message(event: RealtimeSessionEvent) -> dict[str, Any] | None:
    """Map a final transcript event to an OpenTelemetry GenAI message, or `None` for anything else."""
    if isinstance(event, InputTranscript) and event.is_final and event.text:
        return {'role': 'user', 'parts': [{'type': 'text', 'content': event.text}]}
    if isinstance(event, Transcript) and event.is_final and event.text:
        return {'role': 'assistant', 'parts': [{'type': 'text', 'content': event.text}]}
    return None


def _parse_tool_args(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a tool call's raw JSON arguments.

    Returns `(args, None)` on success, or `(None, error_message)` when the payload is not a JSON
    object, so the caller can report the error back to the model rather than guessing.
    """
    if not raw:
        return {}, None
    try:
        parsed = pydantic_core.from_json(raw)
    except ValueError as e:
        return None, f'Error: could not parse tool arguments as JSON: {e}'
    if not isinstance(parsed, dict):
        return None, f'Error: expected tool arguments to be a JSON object, got {type(parsed).__name__}'
    return cast('dict[str, Any]', parsed), None


class RealtimeSession:
    """Wraps a [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection] and auto-executes tool calls.

    When iterating, a `ToolCall` from the connection is intercepted: a `ToolCallStarted` is emitted,
    the tool runs via `tool_runner`, its result is sent back to the model as a `ToolResult`, and a
    `ToolCallCompleted` is emitted. Every other event passes through unchanged.

    Tools execute in one of two modes:

    - **Synchronous** (default): the session waits for the tool to finish before reading further
      events from the model, mirroring a blocking call.
    - **Background**: tools whose name is in `background_tools` run concurrently. The session keeps
      streaming the model's events (so it can keep speaking) and sends the result back once it is
      ready, mirroring firing off a subagent and continuing work while it runs.

    A `ToolResult` is always sent for every `ToolCall`, even when argument parsing or the tool itself
    fails, so the model never stalls waiting on a result.
    """

    def __init__(
        self,
        connection: RealtimeConnection,
        tool_runner: ToolRunner,
        *,
        background_tools: Iterable[str] = (),
        instrumentation: InstrumentationSettings | None = None,
        model_name: str | None = None,
        agent_name: str | None = None,
        usage: RunUsage | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> None:
        self._connection = connection
        self._tool_runner = tool_runner
        self._background_tools = frozenset(background_tools)
        self._instrumentation = instrumentation
        self._model_name = model_name
        self._agent_name = agent_name
        self._usage_limits = usage_limits
        self.usage = usage if usage is not None else RunUsage()
        """Cumulative token usage and tool-call counts for the session, updated as events stream in.

        Pass `usage` to [`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session] to accumulate
        into a shared [`RunUsage`][pydantic_ai.usage.RunUsage]; otherwise a fresh one is used.
        """

    async def send(self, content: RealtimeInput) -> None:
        """Feed content into the underlying connection."""
        await self._connection.send(content)

    async def send_audio(self, data: bytes) -> None:
        """Stream a chunk of audio to the model."""
        await self._connection.send(AudioInput(data=data))

    async def send_text(self, text: str) -> None:
        """Send a complete text turn to the model."""
        await self._connection.send(TextInput(text=text))

    async def send_image(self, data: bytes, *, mime_type: str = 'image/jpeg') -> None:
        """Send an image frame as conversation context (e.g. a video frame)."""
        await self._connection.send(ImageInput(data=data, mime_type=mime_type))

    async def commit_audio(self) -> None:
        """Commit buffered input audio as a user turn (manual turn-taking / push-to-talk)."""
        await self._connection.send(CommitAudio())

    async def clear_audio(self) -> None:
        """Discard buffered, uncommitted input audio."""
        await self._connection.send(ClearAudio())

    async def create_response(self) -> None:
        """Ask the model to respond now (manual turn-taking, after `commit_audio`)."""
        await self._connection.send(CreateResponse())

    async def truncate_output(self, audio_end_ms: int) -> None:
        """Truncate the model's current audio output at `audio_end_ms` (see `TruncateOutput`)."""
        await self._connection.send(TruncateOutput(audio_end_ms=audio_end_ms))

    async def interrupt(self, *, audio_end_ms: int | None = None) -> None:
        """Barge-in: cancel the model's in-progress response, optionally truncating its audio first.

        This is server-side only — it stops generation and (when `audio_end_ms` is given) syncs the
        provider's transcript to what was actually heard. Flushing locally buffered playback is the
        caller's responsibility.

        Args:
            audio_end_ms: Milliseconds of the current output audio that were actually played. When
                given, the output item is truncated to this point before the response is cancelled.
        """
        # Truncate before cancelling: cancellation triggers `response.done`, which clears the tracked
        # output item, so a truncate sent afterwards could no-op.
        if audio_end_ms is not None:
            await self._connection.send(TruncateOutput(audio_end_ms=audio_end_ms))
        await self._connection.send(CancelResponse())

    async def _run_tool(self, call: ToolCall) -> str:
        settings = self._instrumentation
        if settings is None:
            return await self._execute_tool(call)
        attributes: dict[str, Any] = {
            'gen_ai.operation.name': 'execute_tool',
            'gen_ai.tool.name': call.tool_name,
            'gen_ai.tool.call.id': call.tool_call_id,
        }
        if settings.include_content:
            attributes['gen_ai.tool.call.arguments'] = call.args
        with settings.tracer.start_as_current_span(
            f'execute_tool {call.tool_name}', attributes=attributes, kind=SpanKind.INTERNAL
        ) as span:
            result = await self._execute_tool(call)
            if settings.include_content:
                span.set_attribute('gen_ai.tool.call.result', result)
            return result

    async def _execute_tool(self, call: ToolCall) -> str:
        args, error = _parse_tool_args(call.args)
        if error is not None:
            result = error
        else:
            assert args is not None
            try:
                result = await self._tool_runner(call.tool_name, args, call.tool_call_id)
            except Exception as e:
                result = f'Error: {e}'
        await self._connection.send(ToolResult(tool_call_id=call.tool_call_id, output=result))
        return result

    async def __aiter__(self) -> AsyncIterator[RealtimeSessionEvent]:
        settings = self._instrumentation
        if settings is None:
            async for event in self._stream():
                yield event
            return
        # Open a session-level span; tool spans created in the pump task inherit it through the
        # OpenTelemetry context that `asyncio.create_task` copies.
        attributes: dict[str, Any] = {'gen_ai.operation.name': 'realtime'}
        if self._model_name:
            attributes['gen_ai.request.model'] = self._model_name
        if self._agent_name:
            attributes['gen_ai.agent.name'] = self._agent_name
        span_name = f'realtime {self._model_name}' if self._model_name else 'realtime'
        with settings.tracer.start_as_current_span(span_name, attributes=attributes, kind=SpanKind.CLIENT) as span:
            # The conversation transcript, captured turn-by-turn so it lands on the span as messages.
            # User input and model output are kept separate to match the GenAI semantic conventions.
            input_messages: list[dict[str, Any]] = []
            output_messages: list[dict[str, Any]] = []
            try:
                async for event in self._stream():
                    if settings.include_content and (message := _transcript_message(event)) is not None:
                        bucket = output_messages if message['role'] == 'assistant' else input_messages
                        bucket.append(message)
                    yield event
            finally:
                self._finalize_span(settings, span, attributes, input_messages, output_messages)

    def _finalize_span(
        self,
        settings: InstrumentationSettings,
        span: Span,
        base_attributes: dict[str, Any],
        input_messages: list[dict[str, Any]],
        output_messages: list[dict[str, Any]],
    ) -> None:
        """Attach cumulative usage and the conversation transcript to the session span."""
        span.set_attributes(self.usage.opentelemetry_attributes())
        if input_messages:
            span.set_attribute('gen_ai.input.messages', json.dumps(input_messages))
        if output_messages:
            span.set_attribute('gen_ai.output.messages', json.dumps(output_messages))
        for token_type in ('input', 'output'):
            tokens: int = getattr(self.usage, f'{token_type}_tokens')
            if tokens:
                settings.tokens_histogram.record(tokens, {**base_attributes, 'gen_ai.token.type': token_type})

    def _tool_call_limit_error(self) -> SessionError | None:
        """A non-recoverable `SessionError` if running one more tool would breach the limits, else `None`."""
        if self._usage_limits is None:
            return None
        projected = dataclasses.replace(self.usage, tool_calls=self.usage.tool_calls + 1)
        try:
            self._usage_limits.check_before_tool_call(projected)
        except UsageLimitExceeded as e:
            return SessionError(message=str(e), type='usage_limit_exceeded', recoverable=False)
        return None

    def _token_limit_error(self) -> SessionError | None:
        """A non-recoverable `SessionError` if accumulated token usage breaches the limits, else `None`."""
        if self._usage_limits is None:
            return None
        try:
            self._usage_limits.check_tokens(self.usage)
        except UsageLimitExceeded as e:
            return SessionError(message=str(e), type='usage_limit_exceeded', recoverable=False)
        return None

    async def _run_background_tool(self, call: ToolCall, queue: asyncio.Queue[RealtimeSessionEvent | object]) -> None:
        """Run a background tool and feed its completion (or failure) back through the queue."""
        try:
            result = await self._run_tool(call)
        except Exception as e:
            # Surface the failure through the queue so the consumer re-raises it, instead of letting it
            # vanish into the final `gather(..., return_exceptions=True)` and hang the session on a
            # completion that never arrives.
            await queue.put(e)
            return
        await queue.put(ToolCallCompleted(tool_name=call.tool_name, tool_call_id=call.tool_call_id, result=result))

    async def _handle_pump_event(
        self,
        event: RealtimeEvent,
        queue: asyncio.Queue[RealtimeSessionEvent | object],
        background: set[asyncio.Task[None]],
    ) -> bool:
        """Process one upstream event onto the queue; return `True` to stop the pump (a limit tripped)."""
        if isinstance(event, ToolCall):
            if (limit_error := self._tool_call_limit_error()) is not None:
                await queue.put(limit_error)
                return True
            self.usage.tool_calls += 1
            await queue.put(ToolCallStarted(tool_name=event.tool_name, tool_call_id=event.tool_call_id))
            if event.tool_name in self._background_tools:
                task = asyncio.create_task(self._run_background_tool(event, queue))
                background.add(task)
                task.add_done_callback(background.discard)
            else:
                result = await self._run_tool(event)
                await queue.put(
                    ToolCallCompleted(tool_name=event.tool_name, tool_call_id=event.tool_call_id, result=result)
                )
            return False
        if isinstance(event, Usage):
            self.usage.incr(event.usage)
            self.usage.requests += 1
            if (limit_error := self._token_limit_error()) is not None:
                await queue.put(limit_error)
                return True
        await queue.put(event)
        return False

    async def _stream(self) -> AsyncIterator[RealtimeSessionEvent]:
        # Both the upstream connection and finished background tools feed a single queue, so a
        # background completion wakes the consumer immediately instead of waiting for the next
        # provider event (which may never come while the model is idle).
        queue: asyncio.Queue[RealtimeSessionEvent | object] = asyncio.Queue()
        closed = object()  # sentinel: the upstream connection has been fully drained
        background: set[asyncio.Task[None]] = set()
        pump_error: Exception | None = None

        async def pump() -> None:
            nonlocal pump_error
            try:
                async for event in self._connection:
                    if await self._handle_pump_event(event, queue, background):
                        return  # a usage limit tripped: stop reading the upstream
            except Exception as e:
                pump_error = e
            finally:
                await queue.put(closed)

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                item = await queue.get()
                if item is closed:
                    break
                yield _as_event(item)  # re-raises if a background tool failed
            # Upstream is done: wait for any in-flight background tools, then flush their completions.
            if background:
                await asyncio.gather(*background, return_exceptions=True)
            while not queue.empty():
                yield _as_event(queue.get_nowait())
            if pump_error is not None:
                raise pump_error
        finally:
            pump_task.cancel()
            for task in list(background):
                task.cancel()
            await asyncio.gather(pump_task, *background, return_exceptions=True)
