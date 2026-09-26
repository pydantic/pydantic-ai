"""Google Workspace connection composition and tool selection."""

from __future__ import annotations

import httpx
import pytest
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.google_workspace import GoogleWorkspace

# MCP's test server leaves its lifespan annotation unresolved with pydantic-settings 2.15.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Field 'lifespan' has an incomplete definition:UserWarning:pydantic_settings.sources.utils"
)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def connections_for(capability: GoogleWorkspace[str | None], deps: str | None) -> list[MCPToolset[str | None]]:
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


def user_token(ctx: RunContext[str]) -> str:
    return ctx.deps


def bearer(connection: MCPToolset[str | None]) -> str:
    transport = connection.client.transport
    assert isinstance(transport, StreamableHttpTransport) and transport.auth is not None
    request = next(transport.auth.auth_flow(httpx.Request('POST', 'https://example.com/mcp')))
    return request.headers['Authorization']


def first_instructions(messages: list[ModelMessage]) -> str:
    request = messages[0]
    assert isinstance(request, ModelRequest)
    return request.instructions or ''


class TestGoogleWorkspace:
    @pytest.mark.parametrize(
        ('services', 'message'), [([], 'at least one'), (['mail'], 'Unknown Google')], ids=['empty', 'unknown']
    )
    def test_invalid_services_raise(self, services: list[str], message: str) -> None:
        with pytest.raises(UserError, match=message):
            GoogleWorkspace(services=services)  # pyright: ignore[reportArgumentType]

    def test_missing_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('GOOGLE_ACCESS_TOKEN', raising=False)
        with pytest.raises(UserError, match='Set `GOOGLE_ACCESS_TOKEN`'):
            GoogleWorkspace('gmail').get_toolset()

    def test_defer_loading_needs_no_id(self, connections: list[tuple[str, str | None]]) -> None:
        Agent(TestModel(), capabilities=[GoogleWorkspace('gmail', auth='token', defer_loading=True)])

    def test_id_is_derived_from_the_services(self) -> None:
        assert GoogleWorkspace(['gmail', 'calendar', 'gmail']).id == 'google-workspace-calendar-gmail'
        assert GoogleWorkspace('gmail', id='mail').id == 'mail'

    def test_two_for_the_same_services_in_any_order_raise_when_the_agent_is_built(self) -> None:
        with pytest.raises(UserError, match="Capability id 'google-workspace-calendar-gmail' is used by multiple"):
            Agent(
                TestModel(),
                capabilities=[
                    GoogleWorkspace(['gmail', 'calendar'], auth='a'),
                    GoogleWorkspace(['calendar', 'gmail'], auth='b'),
                ],
            )

    def test_credential_is_not_in_repr(self) -> None:
        assert 'secret-token' not in repr(GoogleWorkspace('gmail', auth='secret-token'))

    @pytest.mark.parametrize(
        ('read_only', 'expected'),
        [
            (False, '{"gmail_read_item":"read","gmail_write_item":"written","gmail_unmarked_item":"unmarked"}'),
            (True, '{"gmail_read_item":"read"}'),
        ],
    )
    async def test_agent_executes_selected_tools(
        self, connections: list[tuple[str, str | None]], read_only: bool, expected: str
    ) -> None:
        agent = Agent(TestModel(), capabilities=[GoogleWorkspace('gmail', auth='token', read_only=read_only)])
        assert (await agent.run('Use the tools')).output == expected

    def test_environment_token(
        self, connections: list[tuple[str, str | None]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('GOOGLE_ACCESS_TOKEN', 'environment-token')
        GoogleWorkspace('people').get_toolset()
        assert connections == [('https://people.googleapis.com/mcp/v1', 'environment-token')]

    @pytest.mark.parametrize('include', [True, False])
    async def test_server_instructions(self, connections: list[tuple[str, str | None]], include: bool) -> None:
        capability = GoogleWorkspace('gmail', auth='token', include_instructions=include)
        result = await Agent(TestModel(call_tools=[]), capabilities=[capability]).run('Hello')
        assert ('Google instructions.' in first_instructions(result.all_messages())) is include


class TestPerRunAuth:
    async def test_each_run_connects_with_its_own_credential(self) -> None:
        capability = GoogleWorkspace[str | None](['gmail', 'calendar'], auth=per_user_token)
        alice = await connections_for(capability, 'alice-token')
        bob = await connections_for(capability, 'bob-token')
        assert [bearer(connection) for connection in alice] == ['Bearer alice-token', 'Bearer alice-token']
        assert [bearer(connection) for connection in bob] == ['Bearer bob-token', 'Bearer bob-token']

    @pytest.mark.parametrize('missing', [None, ''])
    async def test_provider_returning_none_does_not_fall_back(
        self, missing: str | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('GOOGLE_ACCESS_TOKEN', 'deployment-token')
        capability = GoogleWorkspace[str | None]('gmail', auth=per_user_token)
        assert await connections_for(capability, missing) == []

    async def test_provider_returning_oauth_raises(self) -> None:
        capability = GoogleWorkspace[str | None]('gmail', auth=per_user_token)
        with pytest.raises(UserError, match="must return an API key or token, not 'oauth'"):
            await connections_for(capability, 'oauth')

    def test_provider_does_not_need_environment_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('GOOGLE_ACCESS_TOKEN', raising=False)
        GoogleWorkspace[str | None]('gmail', auth=per_user_token).get_toolset()

    async def test_read_only_applies_per_run(self, connections: list[tuple[str, str | None]]) -> None:
        capability = GoogleWorkspace[str]('gmail', auth=user_token, read_only=True)
        result = await Agent(TestModel(), deps_type=str, capabilities=[capability]).run('Read', deps='alice-token')
        assert result.output == '{"gmail_read_item":"read"}'
        assert connections == [('https://gmailmcp.googleapis.com/mcp/v1', 'alice-token')]
