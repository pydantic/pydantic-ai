"""Behavioral tests for Grain through `Agent(capabilities=[...])`."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import httpx
import pytest
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from mcp.server.fastmcp.server import FastMCP, Settings
from mcp.types import ToolAnnotations

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.grain import Grain
from tests.harness._hosted_mcp import mcp_connection, only_leaf

# The MCP SDK leaves a settings annotation unresolved in some supported dependency
# combinations. Rebuild it before warnings are escalated by the test suite.
Settings.model_rebuild()

DepsT = TypeVar('DepsT')


def _http_transport(toolset: AbstractToolset[DepsT]) -> StreamableHttpTransport:
    toolset = mcp_connection(toolset)
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


def bearer(toolset: AbstractToolset[DepsT]) -> str:
    auth = _http_transport(toolset).auth
    assert auth is not None
    request = next(auth.auth_flow(httpx.Request('POST', 'https://example.com/mcp')))
    return request.headers['Authorization']


async def connections_for(capability: Grain[str | None], deps: str | None) -> list[MCPToolset[str | None]]:
    """The MCP connections a run with `deps` would open."""
    ctx = RunContext[str | None](deps=deps, model=TestModel(), usage=RunUsage())
    toolset = await capability.get_toolset().for_run(ctx)
    connections: list[MCPToolset[str | None]] = []

    def collect(leaf: AbstractToolset[str | None]) -> None:
        if isinstance(leaf, MCPToolset):
            connections.append(leaf)

    toolset.apply(collect)
    return connections


def per_user_token(ctx: RunContext[str | None]) -> str | None:
    """Read the run's token from its deps, as an app serving many users would."""
    return ctx.deps


def fake_grain() -> FastMCP:
    """A stand-in for Grain's server with one read-only tool and one that changes data."""
    server = FastMCP('grain-fake')

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def search_meetings() -> str:
        return 'found'  # pragma: no cover

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def tag_meeting() -> str:
        return 'tagged'  # pragma: no cover

    return server


async def tool_names_for(capability: Grain[str | None], deps: str | None) -> list[str]:
    """The tools a run with `deps` would get, with each hosted Grain connection swapped for `fake_grain()`."""
    ctx = RunContext[str | None](deps=deps, model=TestModel(), usage=RunUsage())
    toolset = await capability.get_toolset().for_run(ctx)

    def to_fake(leaf: AbstractToolset[str | None]) -> AbstractToolset[str | None]:
        assert isinstance(leaf, MCPToolset)
        return MCPToolset(fake_grain())

    toolset = toolset.visit_and_replace(to_fake)
    async with toolset:
        return sorted(await toolset.get_tools(ctx))


class TestGrain:
    @pytest.mark.parametrize(
        ('read_only', 'expected'),
        [
            (False, '{"grain_read":"read","grain_write":"written","grain_unmarked":"unmarked"}'),
            (True, '{"grain_read":"read"}'),
        ],
    )
    async def test_read_only_keeps_only_read_only_tools(self, read_only: bool, expected: str) -> None:
        server = FastMCP('grain-fake')

        @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
        def read() -> str:
            return 'read'

        @server.tool(annotations=ToolAnnotations(readOnlyHint=False))
        def write() -> str:
            return 'written'

        @server.tool()
        def unmarked() -> str:
            return 'unmarked'

        agent = Agent(TestModel(), capabilities=[Grain(client=server, read_only=read_only)])
        assert (await agent.run('Use the tools')).output == expected

    @pytest.mark.parametrize('auth', ['grain-token', per_user_token], ids=['token', 'function'])
    async def test_read_only_applies_to_the_hosted_connection(
        self, auth: str | Callable[[RunContext[str | None]], str | None]
    ) -> None:
        capability = Grain[str | None](auth=auth, read_only=True)
        assert await tool_names_for(capability, 'alice-token') == ['grain_search_meetings']

    @pytest.mark.parametrize('include', [True, False])
    async def test_server_instructions(self, include: bool) -> None:
        server = FastMCP('grain-fake', instructions='Grain instructions.')
        agent = Agent(TestModel(call_tools=[]), capabilities=[Grain(client=server, include_instructions=include)])
        request = (await agent.run('Hello')).all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert ('Grain instructions.' in (request.instructions or '')) is include

    def test_connects_to_grain_with_the_token(self) -> None:
        toolset = Grain(auth='grain-token').get_toolset()
        assert _http_transport(toolset).url == 'https://api.grain.com/_/mcp'
        assert bearer(toolset) == 'Bearer grain-token'

    @pytest.mark.parametrize(
        ('capability', 'include'),
        [(Grain[None](auth='grain-token'), True), (Grain[None](auth='grain-token', include_instructions=False), False)],
        ids=['default', 'disabled'],
    )
    def test_hosted_connection_forwards_include_instructions(self, capability: Grain[None], include: bool) -> None:
        # `MCPToolset` defaults to False, so this proves the capability passes its own setting on.
        toolset = mcp_connection(capability.get_toolset())
        assert toolset.include_instructions is include

    def test_environment_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('GRAIN_ACCESS_TOKEN', 'environment-token')
        assert bearer(Grain().get_toolset()) == 'Bearer environment-token'

    def test_missing_auth_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('GRAIN_ACCESS_TOKEN', raising=False)
        with pytest.raises(UserError, match='Set `GRAIN_ACCESS_TOKEN`'):
            Grain().get_toolset()

    @pytest.mark.filterwarnings('ignore:Using in-memory token storage')
    def test_oauth_uses_browser_login(self) -> None:
        assert isinstance(_http_transport(Grain(auth='oauth').get_toolset()).auth, OAuth)

    def test_credential_is_not_in_repr(self) -> None:
        assert 'secret-token' not in repr(Grain(auth='secret-token'))

    def test_client_accepts_empty_auth(self) -> None:
        # The `auth` table documents `''` as unset, so it does not conflict with `client`.
        toolset = Grain[None](client='https://example.com/mcp', auth='').get_toolset()
        assert _http_transport(toolset).url == 'https://example.com/mcp'

    @pytest.mark.parametrize('auth', ['grain-token', per_user_token], ids=['token', 'function'])
    def test_client_cannot_be_combined_with_auth(
        self, auth: str | Callable[[RunContext[str | None]], str | None]
    ) -> None:
        with pytest.raises(UserError, match='`client` owns the connection'):
            Grain[str | None](client='https://example.com/mcp', auth=auth)

    @pytest.mark.parametrize(
        'capability',
        [
            Grain[str | None](id='tenant-grain', auth='grain-token'),
            Grain[str | None](id='tenant-grain', auth=per_user_token),
            Grain[str | None](id='tenant-grain', client='https://example.com/mcp'),
        ],
        ids=['token', 'function', 'client'],
    )
    def test_custom_id_is_forwarded(self, capability: Grain[str | None]) -> None:
        assert only_leaf(capability.get_toolset()).id == 'tenant-grain'

    @pytest.mark.parametrize('id', [None, ''])
    def test_unset_id_means_the_default(self, id: str | None) -> None:
        capability = Grain[None](id=id, auth='grain-token')
        assert (capability.id, only_leaf(capability.get_toolset()).id) == ('grain', 'grain')

    def test_unset_ids_that_differ_raise_when_the_agent_is_built(self) -> None:
        with pytest.raises(UserError, match="Capability id 'grain' is used by multiple Grain capabilities"):
            Agent(TestModel(), capabilities=[Grain(id=None, auth='a'), Grain(id=None, auth='b')])

    def test_defer_loading_needs_no_id(self) -> None:
        Agent(TestModel(), capabilities=[Grain(auth='grain-token', defer_loading=True)])

    def test_two_that_differ_raise_when_the_agent_is_built(self) -> None:
        with pytest.raises(
            UserError,
            match="Capability id 'grain' is used by multiple Grain capabilities that disagree on 'auth', 'read_only'",
        ):
            Agent(TestModel(), capabilities=[Grain(auth='a'), Grain(auth='b', read_only=True)])


class TestPerRunAuth:
    async def test_each_run_connects_with_its_own_credential(self) -> None:
        capability = Grain[str | None](auth=per_user_token)
        [alice] = await connections_for(capability, 'alice-token')
        [bob] = await connections_for(capability, 'bob-token')
        assert (bearer(alice), bearer(bob)) == ('Bearer alice-token', 'Bearer bob-token')

    @pytest.mark.parametrize('missing', [None, ''])
    async def test_no_credential_means_no_tools(self, missing: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
        # The environment token is set to show a function never falls back to it.
        monkeypatch.setenv('GRAIN_ACCESS_TOKEN', 'deployment-token')
        capability = Grain[str | None](auth=per_user_token)
        assert await connections_for(capability, missing) == []

    async def test_function_returning_oauth_raises(self) -> None:
        capability = Grain[str | None](auth=per_user_token)
        with pytest.raises(UserError, match="must return an API key or token, not 'oauth'"):
            await connections_for(capability, 'oauth')
