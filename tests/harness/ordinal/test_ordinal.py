"""Behavioral tests for Ordinal through `Agent(capabilities=[...])`."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import httpx
import pytest
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from mcp.server.fastmcp.server import FastMCP, Settings
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.ordinal import Ordinal

# The MCP SDK leaves a settings annotation unresolved in some supported dependency
# combinations. Rebuild it before warnings are escalated by the test suite.
Settings.model_rebuild()

DepsT = TypeVar('DepsT')


def _http_transport(toolset: AbstractToolset[DepsT]) -> StreamableHttpTransport:
    assert isinstance(toolset, MCPToolset)
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


def bearer(toolset: AbstractToolset[DepsT]) -> str:
    auth = _http_transport(toolset).auth
    assert auth is not None
    request = next(auth.auth_flow(httpx.Request('POST', 'https://example.com/mcp')))
    return request.headers['Authorization']


async def connections_for(capability: Ordinal[str | None], deps: str | None) -> list[MCPToolset[str | None]]:
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


class TestOrdinal:
    @pytest.mark.anyio
    async def test_agent_runs_with_ordinal_tools(self) -> None:
        server = FastMCP('ordinal-fake')

        @server.tool()
        def ordinal_get_workspace_context() -> dict[str, str]:
            """List Ordinal workspaces."""
            return {'slug': 'acme'}

        agent = Agent(
            TestModel(call_tools=['ordinal_get_workspace_context']),
            capabilities=[Ordinal(client=server)],
        )
        result = await agent.run('List my workspaces')
        assert 'acme' in result.output

    @pytest.mark.anyio
    @pytest.mark.parametrize('include', [True, False])
    async def test_server_instructions(self, include: bool) -> None:
        server = FastMCP('ordinal-fake', instructions='Ordinal instructions.')
        agent = Agent(TestModel(call_tools=[]), capabilities=[Ordinal(client=server, include_instructions=include)])
        request = (await agent.run('Hello')).all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert ('Ordinal instructions.' in (request.instructions or '')) is include

    def test_connects_to_ordinal_with_the_token(self) -> None:
        toolset = Ordinal(auth='ordinal-token').get_toolset()
        assert _http_transport(toolset).url == 'https://app.tryordinal.com/mcp'
        assert bearer(toolset) == 'Bearer ordinal-token'

    @pytest.mark.parametrize(
        ('capability', 'include'),
        [
            (Ordinal[str | None](auth='ordinal-token'), True),
            (Ordinal[str | None](auth='ordinal-token', include_instructions=False), False),
        ],
        ids=['default', 'disabled'],
    )
    def test_hosted_connection_forwards_include_instructions(
        self, capability: Ordinal[str | None], include: bool
    ) -> None:
        toolset = capability.get_toolset()
        assert isinstance(toolset, MCPToolset)
        assert toolset.include_instructions is include

    def test_environment_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ORDINAL_ACCESS_TOKEN', 'environment-token')
        assert bearer(Ordinal().get_toolset()) == 'Bearer environment-token'

    def test_missing_auth_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('ORDINAL_ACCESS_TOKEN', raising=False)
        with pytest.raises(UserError, match='Set `ORDINAL_ACCESS_TOKEN`'):
            Ordinal().get_toolset()

    @pytest.mark.filterwarnings('ignore:Using in-memory token storage')
    def test_oauth_uses_browser_login(self) -> None:
        assert isinstance(_http_transport(Ordinal(auth='oauth').get_toolset()).auth, OAuth)

    def test_credential_is_not_in_repr(self) -> None:
        assert 'secret-token' not in repr(Ordinal(auth='secret-token'))

    @pytest.mark.parametrize('auth', ['ordinal-token', per_user_token])
    def test_client_cannot_be_combined_with_auth(
        self, auth: str | Callable[[RunContext[str | None]], str | None]
    ) -> None:
        with pytest.raises(UserError, match='`client` owns the connection'):
            Ordinal(client='https://example.com/mcp', auth=auth)

    @pytest.mark.parametrize(
        'capability',
        [
            Ordinal[str | None](id='tenant-ordinal', auth='ordinal-token'),
            Ordinal[str | None](id='tenant-ordinal', auth=per_user_token),
            Ordinal[str | None](id='tenant-ordinal', client='https://example.com/mcp'),
        ],
        ids=['token', 'function', 'client'],
    )
    def test_custom_id_is_forwarded(self, capability: Ordinal[str | None]) -> None:
        assert capability.get_toolset().id == 'tenant-ordinal'

    def test_defer_loading_needs_no_id(self) -> None:
        Agent(TestModel(), capabilities=[Ordinal(auth='ordinal-token', defer_loading=True)])

    def test_two_that_differ_raise_when_the_agent_is_built(self) -> None:
        with pytest.raises(
            UserError, match="Capability id 'ordinal' is used by multiple Ordinal capabilities that disagree on 'auth'"
        ):
            Agent(TestModel(), capabilities=[Ordinal(auth='a'), Ordinal(auth='b')])


class TestPerRunAuth:
    @pytest.mark.anyio
    async def test_each_run_connects_with_its_own_credential(self) -> None:
        capability = Ordinal[str | None](auth=per_user_token)
        [alice] = await connections_for(capability, 'alice-token')
        [bob] = await connections_for(capability, 'bob-token')
        assert (bearer(alice), bearer(bob)) == ('Bearer alice-token', 'Bearer bob-token')

    @pytest.mark.anyio
    @pytest.mark.parametrize('missing', [None, ''])
    async def test_no_credential_means_no_tools(self, missing: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
        # The environment token is set to show a function never falls back to it.
        monkeypatch.setenv('ORDINAL_ACCESS_TOKEN', 'deployment-token')
        capability = Ordinal[str | None](auth=per_user_token)
        assert await connections_for(capability, missing) == []

    @pytest.mark.anyio
    async def test_function_returning_oauth_raises(self) -> None:
        capability = Ordinal[str | None](auth=per_user_token)
        with pytest.raises(UserError, match="must return an API key or token, not 'oauth'"):
            await connections_for(capability, 'oauth')
