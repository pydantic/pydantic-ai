"""A realtime session that wraps a [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection] with automatic tool execution."""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import Any, cast

import pydantic_core

from ._base import (
    AudioInput,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    ImageInput,
    RealtimeConnection,
    RealtimeInput,
    RealtimeSessionEvent,
    TextInput,
    ToolCall,
    ToolCallCompleted,
    ToolCallStarted,
    ToolResult,
    TruncateOutput,
)

ToolRunner = Callable[[str, dict[str, Any], str], Awaitable[str]]
"""Async callable executing a tool given its name, parsed arguments, and call id; returns the string result."""


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
    ) -> None:
        self._connection = connection
        self._tool_runner = tool_runner
        self._background_tools = frozenset(background_tools)

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
        # Both the upstream connection and finished background tools feed a single queue, so a
        # background completion wakes the consumer immediately instead of waiting for the next
        # provider event (which may never come while the model is idle).
        queue: asyncio.Queue[RealtimeSessionEvent | object] = asyncio.Queue()
        closed = object()  # sentinel: the upstream connection has been fully drained
        background: set[asyncio.Task[None]] = set()
        pump_error: Exception | None = None

        async def run_background(call: ToolCall) -> None:
            result = await self._run_tool(call)
            await queue.put(ToolCallCompleted(tool_name=call.tool_name, tool_call_id=call.tool_call_id, result=result))

        async def pump() -> None:
            nonlocal pump_error
            try:
                async for event in self._connection:
                    if isinstance(event, ToolCall):
                        await queue.put(ToolCallStarted(tool_name=event.tool_name, tool_call_id=event.tool_call_id))
                        if event.tool_name in self._background_tools:
                            task = asyncio.create_task(run_background(event))
                            background.add(task)
                            task.add_done_callback(background.discard)
                        else:
                            result = await self._run_tool(event)
                            await queue.put(
                                ToolCallCompleted(
                                    tool_name=event.tool_name, tool_call_id=event.tool_call_id, result=result
                                )
                            )
                    else:
                        await queue.put(event)
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
                yield cast('RealtimeSessionEvent', item)
            # Upstream is done: wait for any in-flight background tools, then flush their completions.
            if background:
                await asyncio.gather(*background, return_exceptions=True)
            while not queue.empty():
                yield cast('RealtimeSessionEvent', queue.get_nowait())
            if pump_error is not None:
                raise pump_error
        finally:
            pump_task.cancel()
            for task in list(background):
                task.cancel()
            await asyncio.gather(pump_task, *background, return_exceptions=True)
