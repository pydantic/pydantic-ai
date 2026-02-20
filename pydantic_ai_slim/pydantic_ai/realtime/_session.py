"""Realtime session that wraps a `RealtimeConnection` with automatic tool execution."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pydantic_core

from ._base import (
    AudioInput,
    RealtimeConnection,
    RealtimeInput,
    RealtimeSessionEvent,
    ToolCall,
    ToolCallCompleted,
    ToolCallStarted,
    ToolResult,
)

ToolRunner = Callable[[str, dict[str, Any]], Awaitable[str]]
"""Async callable that executes a tool by name with parsed args, returning the string result."""


class RealtimeSession:
    """Wraps a `RealtimeConnection` and auto-executes tool calls.

    When iterating, `ToolCall` events from the connection are intercepted:
    a `ToolCallStarted` is emitted, the tool is executed via the provided
    ``tool_runner``, the result is sent back to the model, and a
    `ToolCallCompleted` is emitted. All other events pass through directly.
    """

    def __init__(
        self,
        connection: RealtimeConnection,
        tool_runner: ToolRunner,
    ) -> None:
        self._connection = connection
        self._tool_runner = tool_runner

    async def send(self, content: RealtimeInput) -> None:
        """Feed content into the underlying connection."""
        await self._connection.send(content)

    async def send_audio(self, data: bytes) -> None:
        """Convenience method to send audio data."""
        await self._connection.send(AudioInput(data=data))

    async def __aiter__(self) -> AsyncIterator[RealtimeSessionEvent]:
        async for event in self._connection:
            if isinstance(event, ToolCall):
                yield ToolCallStarted(tool_name=event.tool_name, tool_call_id=event.tool_call_id)

                try:
                    args: dict[str, Any] = pydantic_core.from_json(event.args) if event.args else {}
                except ValueError:
                    args = {}

                result = await self._tool_runner(event.tool_name, args)
                await self._connection.send(ToolResult(tool_call_id=event.tool_call_id, output=result))

                yield ToolCallCompleted(
                    tool_name=event.tool_name,
                    tool_call_id=event.tool_call_id,
                    result=result,
                )
            else:
                yield event
