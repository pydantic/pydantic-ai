"""Test Notion's connection settings and tool selection through an agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic_ai import Agent
from pydantic_ai.capabilities import DynamicCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.notion import Notion

# MCP's test server leaves its lifespan annotation unresolved with pydantic-settings 2.15.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Field 'lifespan' has an incomplete definition:UserWarning:pydantic_settings.sources.utils"
)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def server() -> FastMCP:
    server = FastMCP('provider', instructions='Provider instructions.')

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def read_resource() -> str:
        return 'read'

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def write_resource() -> str:
        return 'written'

    @server.tool()
    def unmarked_resource() -> str:
        return 'unmarked'

    return server


def transport(capability: Notion[None]) -> StreamableHttpTransport:
    toolset = capability.get_toolset()
    assert isinstance(toolset, MCPToolset)
    result = toolset.client.transport
    assert isinstance(result, StreamableHttpTransport)
    return result


async def connections_for(capability: Notion[str | None], deps: str | None) -> list[MCPToolset[str | None]]:
    """The MCP connections a run with `deps` would open."""
    ctx = RunContext[str | None](deps=deps, model=TestModel(), usage=RunUsage())
    toolset = await capability.get_toolset().for_run(ctx)
    connections: list[MCPToolset[str | None]] = []

    def collect(leaf: AbstractToolset[str | None]) -> None:
        if isinstance(leaf, MCPToolset):
            connections.append(leaf)

    toolset.apply(collect)
    return connections


@dataclass
class Tenant:
    token: str | None


def no_credential(ctx: RunContext[object]) -> None:
    return None


def bearer(connection: MCPToolset[str | None]) -> str:
    transport = connection.client.transport
    assert isinstance(transport, StreamableHttpTransport) and transport.auth is not None
    request = next(transport.auth.auth_flow(httpx.Request('POST', 'https://example.com/mcp')))
    return request.headers['Authorization']


class TestNotion:
    @pytest.mark.parametrize(
        ('read_only', 'expected'),
        [
            (False, '{"read_resource":"read","write_resource":"written","unmarked_resource":"unmarked"}'),
            (True, '{"read_resource":"read"}'),
        ],
    )
    async def test_agent_executes_selected_tools(self, server: FastMCP, read_only: bool, expected: str) -> None:
        agent = Agent(TestModel(), capabilities=[Notion(client=server, read_only=read_only)])
        result = await agent.run('Use the tools')
        assert result.output == expected

    @pytest.mark.parametrize('include', [True, False])
    async def test_server_instructions(self, server: FastMCP, include: bool) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[Notion(client=server, include_instructions=include)])
        result = await agent.run('Hello')
        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert ('Provider instructions.' in (request.instructions or '')) is include

    def test_custom_client_owns_authentication(self) -> None:
        client = StreamableHttpTransport('https://example.com/mcp', auth=httpx.BasicAuth('user', 'secret'))
        assert transport(Notion(client=client)).auth is client.auth

    @pytest.mark.parametrize('settings', [{'auth': 'key'}, {'auth': no_credential}])
    def test_client_cannot_be_combined_with_connection_settings(self, settings: dict[str, Any]) -> None:
        with pytest.raises(UserError, match='`client` owns the connection'):
            Notion(client='https://example.com/mcp', **settings)

    def test_defer_loading_needs_no_id(self, server: FastMCP) -> None:
        Agent(TestModel(), capabilities=[Notion(client=server, defer_loading=True)])

    def test_two_that_differ_raise_when_the_agent_is_built(self) -> None:
        with pytest.raises(
            UserError,
            match="Capability id 'notion' is used by multiple Notion capabilities that disagree on 'auth', 'read_only'",
        ):
            Agent(TestModel(), capabilities=[Notion(auth='a'), Notion(auth='b', read_only=True)])

    def test_credential_is_not_in_repr(self) -> None:
        assert 'secret-token' not in repr(Notion(auth='secret-token'))

    def test_environment_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('NOTION_ACCESS_TOKEN', 'environment-token')
        auth = transport(Notion()).auth
        assert isinstance(auth, httpx.Auth)
        request = next(auth.auth_flow(httpx.Request('POST', 'https://example.com/mcp')))
        assert request.headers['Authorization'] == 'Bearer environment-token'

    def test_hosted_endpoint(self) -> None:
        assert transport(Notion(auth='token')).url == 'https://mcp.notion.com/mcp'

    def test_missing_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('NOTION_ACCESS_TOKEN', raising=False)
        with pytest.raises(UserError, match='Set `NOTION_ACCESS_TOKEN`'):
            Notion().get_toolset()


class TestPerRunAuth:
    async def test_each_run_connects_with_its_own_credential(self) -> None:
        capability = Notion[str | None](auth=lambda ctx: ctx.deps)
        [alice] = await connections_for(capability, 'alice-token')
        [bob] = await connections_for(capability, 'bob-token')
        assert (bearer(alice), bearer(bob)) == ('Bearer alice-token', 'Bearer bob-token')

    @pytest.mark.parametrize('missing', [None, ''])
    async def test_provider_returning_none_does_not_fall_back(
        self, missing: str | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('NOTION_ACCESS_TOKEN', 'deployment-token')
        capability = Notion[str | None](auth=lambda ctx: ctx.deps)
        assert await connections_for(capability, missing) == []
        agent = Agent(TestModel(), capabilities=[Notion[object](auth=no_credential)])
        result = await agent.run('Use the tools')
        assert result.output == 'success (no tool calls)'

    async def test_provider_returning_oauth_raises(self) -> None:
        capability = Notion[str | None](auth=lambda ctx: ctx.deps)
        with pytest.raises(UserError, match="must return an API key or token, not 'oauth'"):
            await connections_for(capability, 'oauth')

    @pytest.mark.filterwarnings('ignore:Using in-memory token storage')
    def test_fixed_oauth_uses_browser_login(self) -> None:
        assert isinstance(transport(Notion(auth='oauth')).auth, OAuth)

    async def test_read_only_applies_per_run(self) -> None:
        capability = Notion[str | None](auth=lambda ctx: ctx.deps, read_only=True)
        assert len(await connections_for(capability, 'alice-token')) == 1

    async def test_dynamic_capability_builds_per_run(self, server: FastMCP) -> None:
        def notion(ctx: RunContext[Tenant]) -> Notion[Tenant] | None:
            return None if ctx.deps.token is None else Notion(client=server, read_only=True)

        agent = Agent(TestModel(), deps_type=Tenant, capabilities=[DynamicCapability(notion, id='notion')])
        alice = await agent.run('Use the tools', deps=Tenant('alice-token'))
        nobody = await agent.run('Use the tools', deps=Tenant(None))
        assert (alice.output, nobody.output) == ('{"read_resource":"read"}', 'success (no tool calls)')
