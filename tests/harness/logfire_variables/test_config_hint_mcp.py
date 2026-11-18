"""The baseline masks the hostname an MCP server's toolset id is derived from, as core telemetry does."""

from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip('fastmcp')

import httpx2
from fastmcp import FastMCP
from logfire.testing import CaptureLogfire
from logfire.variables import VariablesConfig
from starlette.applications import Starlette

from pydantic_ai import Agent
from pydantic_ai._mcp_compat import is_mcp_sdk_v2
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.logfire import AgentControl

from ._helpers import variables_provider

MCP_SDK_V2 = is_mcp_sdk_v2()


@pytest.fixture
def server_app() -> Starlette:
    """An MCP server with instructions and one tool, as an HTTP app built outside the event loop."""
    server = FastMCP('test', instructions='server guidance')

    @server.tool
    def ping() -> str:
        """Ping the server."""
        return 'pong'  # pragma: no cover

    return server.http_app(stateless_http=True, json_response=True)


async def test_a_url_derived_mcp_toolset_id_is_masked_in_the_baseline(
    capfire: CaptureLogfire, server_app: Starlette
) -> None:
    client = (
        httpx2.AsyncClient(transport=httpx2.ASGITransport(app=server_app))
        if MCP_SDK_V2
        else httpx.AsyncClient(transport=httpx.ASGITransport(app=server_app))
    )
    toolset = MCPToolset('http://testserver/mcp', http_client=client, include_instructions=True)
    assert toolset.id == 'testserver'

    with variables_provider(capfire, VariablesConfig(variables={})):
        agent = Agent(TestModel(call_tools=[]), name='mcp_agent', toolsets=[toolset], capabilities=[AgentControl()])
        try:
            async with server_app.router.lifespan_context(server_app):
                await agent.run('hello')
        finally:
            await client.aclose()

    (serialized,) = [
        span['attributes']['agent_control.baseline']
        for span in capfire.exporter.exported_spans_as_dict()
        if span['name'] == 'agent_control_config_hint'
    ]
    assert 'testserver' not in serialized
    baseline = json.loads(serialized)
    assert [entry['id'] for entry in baseline['instructions']] == ['toolset:[REDACTED]']
    assert [tool['toolset'] for tool in baseline['tool_definitions']] == ['[REDACTED]']
