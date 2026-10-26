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
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.github import GITHUB_MCP_URL, GitHub
from tests.harness._hosted_mcp import mcp_connection, only_leaf

# MCP's test server leaves its lifespan annotation unresolved with pydantic-settings 2.15.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Field 'lifespan' has an incomplete definition:UserWarning:pydantic_settings.sources.utils"
    ),
]


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
    toolset = mcp_connection(toolset)
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
            (
                False,
                '{"github_read_resource":"read","github_write_resource":"written","github_unmarked_resource":"unmarked"}',
            ),
            (True, '{"github_read_resource":"read"}'),
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
        ('auth', 'url', 'toolsets', 'lockdown'),
        [
            ('token', GITHUB_MCP_URL, None, False),
            (per_user_token, GITHUB_MCP_URL, None, False),
            (None, 'https://example.com/mcp', None, False),
            (None, GITHUB_MCP_URL, ['repos'], False),
            (None, GITHUB_MCP_URL, None, True),
        ],
    )
    def test_client_cannot_be_combined_with_connection_settings(
        self,
        auth: str | Callable[[RunContext[str | None]], str | None] | None,
        url: str,
        toolsets: list[str] | None,
        lockdown: bool,
    ) -> None:
        with pytest.raises(UserError, match='`client` owns the connection'):
            GitHub(client='https://example.com/mcp', auth=auth, url=url, toolsets=toolsets, lockdown=lockdown)

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
        assert only_leaf(capability.get_toolset()).id == 'work-github'

    @pytest.mark.parametrize(
        'capability',
        [
            GitHub[str | None](id='work-github', auth='token', read_only=True, tool_prefix=None),
            GitHub[str | None](id='work-github', auth=per_user_token, read_only=True, tool_prefix=None),
        ],
        ids=['token', 'function'],
    )
    def test_hosted_read_only_keeps_the_id(self, capability: GitHub[str | None]) -> None:
        # Durable execution finds a run's held connection by the toolset's `id`. The default `tool_prefix` wraps the
        # filter, so only the connection under it keeps the id then (see `test_custom_id_is_forwarded`).
        assert capability.get_toolset().id == 'work-github'

    def test_defer_loading_needs_no_id(self, server: FastMCP) -> None:
        Agent(TestModel(), capabilities=[GitHub(client=server, defer_loading=True)])

    def test_two_that_differ_raise_when_the_agent_is_built(self) -> None:
        with pytest.raises(
            UserError,
            match="Capability id 'github' is used by multiple GitHub capabilities that disagree on 'auth', 'read_only'",
        ):
            Agent(TestModel(), capabilities=[GitHub(auth='a'), GitHub(auth='b', read_only=True)])

    def test_lockdown_is_part_of_the_connection(self) -> None:
        # Two that differ only in `lockdown` must raise, so a same-id duplicate cannot silently drop the filter.
        first = GitHub[None](auth='secret-token', lockdown=True)
        assert GitHub[None].combine([first, GitHub[None](auth='secret-token', lockdown=True)]) is first
        with pytest.raises(UserError, match=r"disagree on 'lockdown'") as exc_info:
            Agent(TestModel(), capabilities=[GitHub(auth='secret-token'), GitHub(auth='secret-token', lockdown=True)])
        assert 'secret-token' not in str(exc_info.value)

    @pytest.mark.parametrize('unset', [None, ''])
    def test_unset_id_connects_under_the_default_id(self, unset: str | None) -> None:
        assert only_leaf(GitHub[str | None](id=unset, auth='token').get_toolset()).id == 'github'

    def test_unset_ids_are_not_combined(self) -> None:
        # Without an id there is nothing to combine on, so differing settings keep working as separate connections.
        Agent(
            TestModel(),
            capabilities=[
                GitHub(id=None, auth='token', toolsets=['repos'], include_instructions=False),
                GitHub(id=None, auth='token', toolsets=['issues'], include_instructions=False),
            ],
        )

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
        toolset = mcp_connection(capability.get_toolset())
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

    def test_missing_token_raises_when_the_agent_is_built(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('GITHUB_TOKEN', raising=False)
        with pytest.raises(UserError, match='Set `GITHUB_TOKEN`'):
            Agent(TestModel(), capabilities=[GitHub()])

    @pytest.mark.parametrize('auth', ['oauth', None], ids=['auth', 'environment'])
    def test_oauth_token_raises(self, auth: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('GITHUB_TOKEN', 'oauth')
        capability = GitHub[str | None](auth=auth)
        with pytest.raises(UserError, match="GitHub needs a token, not 'oauth'"):
            capability.get_toolset()

    @pytest.mark.parametrize(
        'url',
        ['https:///mcp', 'api.githubcopilot.com/mcp', 'ftp://api.githubcopilot.com/mcp/'],
    )
    def test_url_must_be_https(self, url: str) -> None:
        with pytest.raises(UserError, match=r"`url` must be an `https://` URL; got '"):
            GitHub(auth='token', url=url)

    @pytest.mark.parametrize('url', ['http://api.githubcopilot.com/mcp/', 'http://127.0.0.1:8082/mcp'])
    def test_plain_http_url_is_deprecated(self, url: str) -> None:
        with pytest.warns(HarnessDeprecationWarning, match='sends the GitHub token unencrypted') as record:
            capability = GitHub(auth='token', url=url)
        assert record[0].filename == __file__
        assert transport(capability.get_toolset()).url == url

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
            (GitHub[str | None](auth='token', lockdown=True), {'X-MCP-Lockdown': 'true'}),
            (
                GitHub[str | None](auth='token', read_only=True, toolsets=['actions']),
                {'X-MCP-Readonly': 'true', 'X-MCP-Toolsets': 'actions'},
            ),
            (
                GitHub[str | None](auth='token', read_only=True, lockdown=True, toolsets=['actions', 'notifications']),
                {'X-MCP-Readonly': 'true', 'X-MCP-Lockdown': 'true', 'X-MCP-Toolsets': 'actions,notifications'},
            ),
        ],
        ids=['default', 'lockdown', 'read-only', 'configured'],
    )
    def test_server_settings_are_sent_as_headers(self, capability: GitHub[str | None], headers: dict[str, str]) -> None:
        assert transport(capability.get_toolset()).headers == headers

    def test_enterprise_endpoint(self) -> None:
        url = 'https://copilot-api.acme.ghe.com/mcp'
        assert transport(GitHub(auth='token', url=url).get_toolset()).url == url

    async def test_hosted_read_only_keeps_only_marked_tools(self, server: FastMCP) -> None:
        # Stand in a server that ignores `X-MCP-Readonly` for the hosted connection, to show the client-side filter.
        def ignore_header(leaf: AbstractToolset[None]) -> AbstractToolset[None]:
            return MCPToolset[None](server) if isinstance(leaf, MCPToolset) else leaf

        toolset = GitHub[None](auth='token', read_only=True).get_toolset().visit_and_replace(ignore_header)
        result = await Agent(TestModel(), deps_type=type(None), toolsets=[toolset]).run('Use the tools')
        assert result.output == '{"github_read_resource":"read"}'


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
        with pytest.raises(UserError, match="must return a GitHub token, not 'oauth'"):
            await connections_for(capability, 'oauth')

    async def test_server_settings_apply_per_run(self) -> None:
        capability = GitHub[str | None](auth=per_user_token, read_only=True, lockdown=True, toolsets=['repos'])
        [connection] = await connections_for(capability, 'alice-token')
        assert transport(connection).headers == {
            'X-MCP-Readonly': 'true',
            'X-MCP-Lockdown': 'true',
            'X-MCP-Toolsets': 'repos',
        }
