"""Voice session that wraps a `RealtimeConnection` with automatic tool execution."""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from ..models.realtime import (
    RealtimeConnection,
    ToolCall,
    ToolCallCompleted,
    ToolCallStarted,
    VoiceSessionEvent,
)

ToolRunner = Callable[[str, dict[str, Any]], Awaitable[str]]
"""Async callable that executes a tool by name with parsed args, returning the string result."""


class VoiceSession:
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

    async def send_audio(self, data: bytes) -> None:
        """Forward audio data to the underlying connection."""
        await self._connection.send_audio(data)

    async def __aiter__(self) -> AsyncIterator[VoiceSessionEvent]:
        async for event in self._connection:
            if isinstance(event, ToolCall):
                yield ToolCallStarted(tool_name=event.tool_name, tool_call_id=event.tool_call_id)

                try:
                    args = json.loads(event.args) if event.args else {}
                except json.JSONDecodeError:
                    args = {}

                result = await self._tool_runner(event.tool_name, args)
                await self._connection.send_tool_result(event.tool_call_id, result)

                yield ToolCallCompleted(
                    tool_name=event.tool_name,
                    tool_call_id=event.tool_call_id,
                    result=result,
                )
            else:
                yield event
