"""An in-memory `MCPToolset` shared by the durable-engine MCP tests.

It is a genuine `MCPToolset` subclass, so a durability capability's `isinstance` wrapping and
`tool_for_tool_def` rebuild apply, but its wire methods return in-memory results, which keeps the
tests off Docker and the network.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip('pydantic_ai.mcp')

from pydantic_ai import ToolsetTool
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import InstructionPart
from pydantic_ai.tools import RunContext, ToolDefinition

_ADD_SCHEMA = {
    'type': 'object',
    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
    'required': ['a', 'b'],
}


class FakeMCPToolset(MCPToolset[object]):
    """In-memory `MCPToolset` whose I/O methods return canned results.

    Bypasses `MCPToolset.__init__` (which would build a real transport) and sets only the
    attributes the durable wrapper and the run touch.
    """

    def __init__(
        self,
        *,
        id: str | None,
        instructions: str | None = None,
        include_instructions: bool = True,
        tool_metadata: dict[str, object] | None = None,
        tool_name: str = 'add',
    ) -> None:
        self._id = id
        self._tool_name = tool_name
        self.max_retries = None
        self.cache_tools = True
        self.include_instructions = include_instructions
        self.include_return_schema = None
        self._instructions_text = instructions
        self._tool_metadata = tool_metadata
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.enter_count = 0
        self._session_depth = 0
        self.implicit_sessions = 0

    async def __aenter__(self) -> FakeMCPToolset:
        self.enter_count += 1
        self._session_depth += 1
        return self

    async def __aexit__(self, *args: object) -> None:
        self._session_depth -= 1

    async def _require_session(self) -> None:
        """Model a real server: I/O needs an active session, and a call without one opens its
        own implicit session for the duration of the call, as `MCPToolset` does."""
        if self._session_depth == 0:
            self.implicit_sessions += 1
            await self.__aenter__()
            await self.__aexit__(None, None, None)

    async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
        await self._require_session()
        tool_def = ToolDefinition(
            name=self._tool_name,
            description='Add two integers.',
            parameters_json_schema=_ADD_SCHEMA,
            metadata=self._tool_metadata,
        )
        return {self._tool_name: self.tool_for_tool_def(tool_def, ctx=ctx)}

    async def get_instructions(self, ctx: RunContext[object]) -> InstructionPart | None:
        await self._require_session()
        if not self.include_instructions or self._instructions_text is None:
            return None
        return InstructionPart(content=self._instructions_text)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
    ) -> int:
        await self._require_session()
        self.tool_calls.append((name, dict(tool_args)))
        return int(tool_args['a']) + int(tool_args['b'])
