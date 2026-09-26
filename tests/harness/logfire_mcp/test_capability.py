"""Test LogfireMCP's connection settings and tool selection through an agent."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone

import httpx
import pytest
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.logfire_mcp import LOGFIRE_EU_MCP_URL, LOGFIRE_US_MCP_URL, LogfireMCP

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


def transport(capability: LogfireMCP[None]) -> StreamableHttpTransport:
    toolset = capability.get_toolset()
    assert isinstance(toolset, MCPToolset)
    result = toolset.client.transport
    assert isinstance(result, StreamableHttpTransport)
    return result


async def connections_for(capability: LogfireMCP[str | None], deps: str | None) -> list[MCPToolset[str | None]]:
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


def bearer(capability: LogfireMCP[None]) -> str:
    auth = transport(capability).auth
    assert isinstance(auth, httpx.Auth)
    request = next(auth.auth_flow(httpx.Request('POST', 'https://example.com/mcp')))
    return request.headers['Authorization']


class TestLogfireMCP:
    @pytest.mark.parametrize(
        ('read_only', 'expected'),
        [
            (False, '{"read_resource":"read","write_resource":"written","unmarked_resource":"unmarked"}'),
            (True, '{"read_resource":"read"}'),
        ],
    )
    async def test_agent_executes_selected_tools(self, server: FastMCP, read_only: bool, expected: str) -> None:
        agent = Agent(TestModel(), capabilities=[LogfireMCP(client=server, read_only=read_only)])
        result = await agent.run('Use the tools')
        assert result.output == expected

    @pytest.mark.parametrize('include', [True, False])
    async def test_server_instructions(self, server: FastMCP, include: bool) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[LogfireMCP(client=server, include_instructions=include)])
        result = await agent.run('Hello')
        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert ('Provider instructions.' in (request.instructions or '')) is include

    @pytest.mark.parametrize(
        ('auth', 'url'),
        [('key', LOGFIRE_US_MCP_URL), (per_user_token, LOGFIRE_US_MCP_URL), (None, LOGFIRE_EU_MCP_URL)],
    )
    def test_client_cannot_be_combined_with_connection_settings(
        self, auth: str | Callable[[RunContext[str | None]], str | None] | None, url: str
    ) -> None:
        with pytest.raises(UserError, match='`client` owns the connection'):
            LogfireMCP(client='https://example.com/mcp', auth=auth, url=url)

    def test_defer_loading_needs_no_id(self, server: FastMCP) -> None:
        Agent(TestModel(), capabilities=[LogfireMCP(client=server, defer_loading=True)])

    def test_two_that_differ_raise_when_the_agent_is_built(self) -> None:
        with pytest.raises(
            UserError,
            match="Capability id 'logfire-mcp' is used by multiple LogfireMCP capabilities that disagree on 'auth', 'url'",
        ):
            Agent(TestModel(), capabilities=[LogfireMCP(auth='a'), LogfireMCP(auth='b', url=LOGFIRE_EU_MCP_URL)])

    @pytest.mark.parametrize(
        'capability',
        [
            LogfireMCP[str | None](id='tenant-logfire', auth='token'),
            LogfireMCP[str | None](id='tenant-logfire', auth=per_user_token),
            LogfireMCP[str | None](id='tenant-logfire', client='https://example.com/mcp'),
        ],
        ids=['token', 'function', 'client'],
    )
    def test_custom_id_is_forwarded(self, capability: LogfireMCP[str | None]) -> None:
        assert capability.get_toolset().id == 'tenant-logfire'

    @pytest.mark.parametrize(
        ('capability', 'include'),
        [
            (LogfireMCP[str | None](auth='token'), True),
            (LogfireMCP[str | None](auth='token', include_instructions=False), False),
        ],
        ids=['default', 'disabled'],
    )
    def test_hosted_connection_forwards_include_instructions(
        self, capability: LogfireMCP[str | None], include: bool
    ) -> None:
        # `MCPToolset` defaults to False, so this proves the capability passes its own setting on.
        toolset = capability.get_toolset()
        assert isinstance(toolset, MCPToolset)
        assert toolset.include_instructions is include

    def test_credential_is_not_in_repr(self) -> None:
        assert 'secret-token' not in repr(LogfireMCP(auth='secret-token'))

    def test_environment_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('LOGFIRE_API_KEY', 'environment-token')
        assert bearer(LogfireMCP()) == 'Bearer environment-token'

    @pytest.mark.parametrize(
        ('capability', 'expected'),
        [
            (LogfireMCP[None](auth='key'), 'https://logfire-us.pydantic.dev/mcp'),
            (LogfireMCP[None](auth='key', url=LOGFIRE_EU_MCP_URL), 'https://logfire-eu.pydantic.dev/mcp'),
            (LogfireMCP[None](auth='key', url='https://logfire.example/mcp'), 'https://logfire.example/mcp'),
        ],
        ids=['us', 'eu', 'self-hosted'],
    )
    def test_endpoint(self, capability: LogfireMCP[None], expected: str) -> None:
        assert transport(capability).url == expected

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('LOGFIRE_API_KEY', raising=False)
        with pytest.raises(UserError, match='Set `LOGFIRE_API_KEY`'):
            LogfireMCP().get_toolset()

    @pytest.mark.parametrize('year', [2020, 2100])
    async def test_current_time_ignores_message_history(self, server: FastMCP, year: int) -> None:
        stamp = datetime(year, 1, 1, tzinfo=timezone.utc)
        history = [
            ModelRequest(parts=[UserPromptPart('Recent errors', timestamp=stamp)], timestamp=stamp),
            ModelResponse(parts=[TextPart('None.')], timestamp=stamp),
        ]
        before = datetime.now(timezone.utc).replace(microsecond=0)
        result = await Agent(TestModel(call_tools=[]), capabilities=[LogfireMCP(client=server)]).run(
            message_history=history
        )
        request = next(message for message in reversed(result.all_messages()) if isinstance(message, ModelRequest))
        instructions = request.instructions or ''
        timestamp = instructions.split('Current UTC time is `')[1].split('`')[0]
        assert before <= datetime.fromisoformat(timestamp) <= datetime.now(timezone.utc)

    @pytest.mark.parametrize('messages', [[], [ModelResponse(parts=[TextPart('Hello')])]])
    def test_no_current_time_without_a_request(self, messages: list[ModelMessage]) -> None:
        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), messages=messages)
        assert LogfireMCP[None]()._current_utc(ctx) is None  # pyright: ignore[reportPrivateUsage]


class TestPerRunAuth:
    async def test_concurrent_runs_use_their_own_credentials(self, whoami_url: str) -> None:
        agent = Agent(TestModel(), deps_type=str, capabilities=[LogfireMCP[str](url=whoami_url, auth=per_user_token)])
        alice, bob = await asyncio.gather(
            agent.run('Who am I?', deps='alice-token'), agent.run('Who am I?', deps='bob-token')
        )
        assert (alice.output, bob.output) == ('{"whoami":"Bearer alice-token"}', '{"whoami":"Bearer bob-token"}')

    @pytest.mark.parametrize(('token', 'connections'), [('alice-token', 1), (None, 0), ('', 0)])
    async def test_connects_only_with_a_credential(
        self, token: str | None, connections: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The environment key is set to show a function never falls back to it.
        monkeypatch.setenv('LOGFIRE_API_KEY', 'deployment-token')
        capability = LogfireMCP[str | None](auth=per_user_token)
        assert len(await connections_for(capability, token)) == connections

    async def test_provider_returning_oauth_raises(self) -> None:
        capability = LogfireMCP[str | None](auth=per_user_token)
        with pytest.raises(UserError, match="must return an API key or token, not 'oauth'"):
            await connections_for(capability, 'oauth')

    @pytest.mark.filterwarnings('ignore:Using in-memory token storage')
    def test_fixed_oauth_uses_browser_login(self) -> None:
        assert isinstance(transport(LogfireMCP(auth='oauth')).auth, OAuth)
