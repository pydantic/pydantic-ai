"""A realtime session that wraps a [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection] with automatic tool execution."""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import Any, cast

import pydantic_core

from ._base import (
    AudioInput,
    RealtimeConnection,
    RealtimeInput,
    RealtimeSessionEvent,
    TextInput,
    ToolCall,
    ToolCallCompleted,
    ToolCallStarted,
    ToolResult,
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
        background: set[asyncio.Task[None]] = set()
        completed: list[ToolCallCompleted] = []

        async def run_background(call: ToolCall) -> None:
            result = await self._run_tool(call)
            completed.append(ToolCallCompleted(tool_name=call.tool_name, tool_call_id=call.tool_call_id, result=result))

        try:
            async for event in self._connection:
                while completed:
                    yield completed.pop(0)
                if isinstance(event, ToolCall):
                    yield ToolCallStarted(tool_name=event.tool_name, tool_call_id=event.tool_call_id)
                    if event.tool_name in self._background_tools:
                        task = asyncio.create_task(run_background(event))
                        background.add(task)
                        task.add_done_callback(background.discard)
                    else:
                        result = await self._run_tool(event)
                        yield ToolCallCompleted(
                            tool_name=event.tool_name, tool_call_id=event.tool_call_id, result=result
                        )
                else:
                    yield event
            if background:
                await asyncio.gather(*background, return_exceptions=True)
            while completed:
                yield completed.pop(0)
        finally:
            for task in list(background):
                task.cancel()
            if background:
                await asyncio.gather(*background, return_exceptions=True)
