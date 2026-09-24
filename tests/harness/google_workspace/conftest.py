"""A Google-style MCP server exercised in process."""

from __future__ import annotations

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import AgentDepsT


@pytest.fixture
def connections(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str | None]]:
    server = FastMCP('google', instructions='Google instructions.')

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def read_item() -> str:
        return 'read'

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def write_item() -> str:
        return 'written'

    @server.tool()
    def unmarked_item() -> str:
        return 'unmarked'

    connections: list[tuple[str, str | None]] = []

    class TestConnection(MCPToolset[AgentDepsT]):
        def __init__(self, url: str, *, id: str, auth: str, include_instructions: bool) -> None:
            connections.append((url, auth))
            super().__init__(server, id=id, include_instructions=include_instructions)

    monkeypatch.setattr('pydantic_ai_harness.google_workspace._capability.MCPToolset', TestConnection)
    return connections
