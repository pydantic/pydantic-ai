"""Test GitHub's connection settings and tool selection through an agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import httpx
import pytest
from fastmcp.client.transports import StreamableHttpTransport
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.github import GITHUB_MCP_URL, GitHub

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


DepsT = TypeVar('DepsT')


def transport(toolset: AbstractToolset[DepsT]) -> StreamableHttpTransport:
    assert isinstance(toolset, MCPToolset)
    result = toolset.client.transport
    assert isinstance(result, StreamableHttpTransport)
    return result


async def connections_for(capability: GitHub[str | None], deps: str | None) -> list[MCPToolset[str | None]]:
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


def bearer(connection: AbstractToolset[DepsT]) -> str:
    auth = transport(connection).auth
    assert auth is not None
    request = next(auth.auth_flow(httpx.Request('POST', 'https://example.com/mcp')))
    return request.headers['Authorization']


class TestGitHub:
    @pytest.mark.parametrize(
        ('read_only', 'expected'),
        [
            (False, '{"read_resource":"read","write_resource":"written","unmarked_resource":"unmarked"}'),
            (True, '{"read_resource":"read"}'),
        ],
    )
    async def test_agent_executes_selected_tools(self, server: FastMCP, read_only: bool, expected: str) -> None:
        agent = Agent(TestModel(), capabilities=[GitHub(client=server, read_only=read_only)])
        result = await agent.run('Use the tools')
        assert result.output == expected

    @pytest.mark.parametrize('include', [True, False])
    async def test_server_instructions(self, server: FastMCP, include: bool) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[GitHub(client=server, include_instructions=include)])
        result = await agent.run('Hello')
        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert ('Provider instructions.' in (request.instructions or '')) is include

    @pytest.mark.parametrize(
        ('auth', 'url', 'toolsets'),
        [
            ('token', GITHUB_MCP_URL, None),
            (per_user_token, GITHUB_MCP_URL, None),
            (None, 'https://example.com/mcp', None),
            (None, GITHUB_MCP_URL, ['repos']),
        ],
    )
    def test_client_cannot_be_combined_with_connection_settings(
        self, auth: str | Callable[[RunContext[str | None]], str | None] | None, url: str, toolsets: list[str] | None
    ) -> None:
        with pytest.raises(UserError, match='`client` owns the connection'):
            GitHub(client='https://example.com/mcp', auth=auth, url=url, toolsets=toolsets)

    @pytest.mark.parametrize(
        'capability',
        [
            GitHub[str | None](id='work-github', auth='token'),
            GitHub[str | None](id='work-github', auth=per_user_token),
            GitHub[str | None](id='work-github', client='https://example.com/mcp'),
        ],
        ids=['token', 'function', 'client'],
    )
    def test_custom_id_is_forwarded(self, capability: GitHub[str | None]) -> None:
        assert capability.get_toolset().id == 'work-github'

    def test_defer_loading_needs_no_id(self, server: FastMCP) -> None:
        Agent(TestModel(), capabilities=[GitHub(client=server, defer_loading=True)])

    def test_two_that_differ_raise_when_the_agent_is_built(self) -> None:
        with pytest.raises(
            UserError,
            match="Capability id 'github' is used by multiple GitHub capabilities that disagree on 'auth', 'read_only'",
        ):
            Agent(TestModel(), capabilities=[GitHub(auth='a'), GitHub(auth='b', read_only=True)])

    @pytest.mark.parametrize(
        ('capability', 'include'),
        [
            (GitHub[str | None](auth='token'), True),
            (GitHub[str | None](auth='token', include_instructions=False), False),
        ],
        ids=['default', 'disabled'],
    )
    def test_hosted_connection_forwards_include_instructions(
        self, capability: GitHub[str | None], include: bool
    ) -> None:
        # `MCPToolset` defaults to False, so this proves the capability passes its own setting on.
        toolset = capability.get_toolset()
        assert isinstance(toolset, MCPToolset)
        assert toolset.include_instructions is include

    def test_credential_is_not_in_repr(self) -> None:
        assert 'secret-token' not in repr(GitHub(auth='secret-token'))

    def test_connects_to_github_with_the_token(self) -> None:
        toolset = GitHub(auth='github-token').get_toolset()
        assert transport(toolset).url == 'https://api.githubcopilot.com/mcp/'
        assert bearer(toolset) == 'Bearer github-token'

    def test_environment_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('GITHUB_TOKEN', 'environment-token')
        assert bearer(GitHub().get_toolset()) == 'Bearer environment-token'

    def test_missing_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('GITHUB_TOKEN', raising=False)
        with pytest.raises(UserError, match='Set `GITHUB_TOKEN`'):
            GitHub().get_toolset()

    def test_empty_toolsets_raise(self) -> None:
        with pytest.raises(UserError, match='at least one tool group'):
            GitHub(auth='token', toolsets=[])

    @pytest.mark.parametrize('group', ['repos,actions', '', ' '])
    def test_each_toolset_names_one_group(self, group: str) -> None:
        with pytest.raises(UserError, match='must name one tool group'):
            GitHub(auth='token', toolsets=['issues', group])

    @pytest.mark.parametrize(
        ('capability', 'headers'),
        [
            (GitHub[str | None](auth='token'), {}),
            (
                GitHub[str | None](auth='token', read_only=True, toolsets=['actions', 'notifications']),
                {'X-MCP-Readonly': 'true', 'X-MCP-Toolsets': 'actions,notifications'},
            ),
        ],
        ids=['default', 'configured'],
    )
    def test_server_settings_are_sent_as_headers(self, capability: GitHub[str | None], headers: dict[str, str]) -> None:
        assert transport(capability.get_toolset()).headers == headers

    def test_enterprise_endpoint(self) -> None:
        url = 'https://copilot-api.acme.ghe.com/mcp'
        assert transport(GitHub(auth='token', url=url).get_toolset()).url == url


class TestPerRunAuth:
    async def test_each_run_connects_with_its_own_credential(self) -> None:
        capability = GitHub[str | None](auth=per_user_token)
        [alice] = await connections_for(capability, 'alice-token')
        [bob] = await connections_for(capability, 'bob-token')
        assert (bearer(alice), bearer(bob)) == ('Bearer alice-token', 'Bearer bob-token')

    @pytest.mark.parametrize('missing', [None, ''])
    async def test_no_credential_means_no_tools(self, missing: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
        # The environment token is set to show a function never falls back to it.
        monkeypatch.setenv('GITHUB_TOKEN', 'deployment-token')
        capability = GitHub[str | None](auth=per_user_token)
        assert await connections_for(capability, missing) == []

    async def test_provider_returning_oauth_raises(self) -> None:
        capability = GitHub[str | None](auth=per_user_token)
        with pytest.raises(UserError, match="must return an API key or token, not 'oauth'"):
            await connections_for(capability, 'oauth')

    async def test_read_only_applies_per_run(self) -> None:
        capability = GitHub[str | None](auth=per_user_token, read_only=True, toolsets=['repos'])
        [connection] = await connections_for(capability, 'alice-token')
        assert transport(connection).headers == {'X-MCP-Readonly': 'true', 'X-MCP-Toolsets': 'repos'}
