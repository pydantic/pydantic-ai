"""Behavioral tests for PostHog through `Agent(capabilities=[...])`."""

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

from pydantic_ai_harness.posthog import PostHog

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


async def connections_for(capability: PostHog[str | None], deps: str | None) -> list[MCPToolset[str | None]]:
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


class TestPostHog:
    @pytest.mark.anyio
    async def test_agent_runs_with_posthog_tools(self) -> None:
        server = FastMCP('posthog-fake')

        @server.tool()
        def insight_query() -> dict[str, str]:
            """Query PostHog insights."""
            return {'name': 'Weekly signups'}

        agent = Agent(TestModel(call_tools=['insight_query']), capabilities=[PostHog(client=server)])
        result = await agent.run('Show weekly signups')
        assert 'Weekly signups' in result.output

    @pytest.mark.anyio
    @pytest.mark.parametrize('include', [True, False])
    async def test_server_instructions(self, include: bool) -> None:
        server = FastMCP('posthog-fake', instructions='PostHog instructions.')
        agent = Agent(TestModel(call_tools=[]), capabilities=[PostHog(client=server, include_instructions=include)])
        request = (await agent.run('Hello')).all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert ('PostHog instructions.' in (request.instructions or '')) is include

    def test_connects_to_posthog_with_the_key(self) -> None:
        toolset = PostHog(auth='posthog-key').get_toolset()
        assert _http_transport(toolset).url == 'https://mcp.posthog.com/mcp'
        assert bearer(toolset) == 'Bearer posthog-key'

    @pytest.mark.parametrize(
        ('capability', 'include'),
        [
            (PostHog[None](auth='posthog-key'), True),
            (PostHog[None](auth='posthog-key', include_instructions=False), False),
        ],
        ids=['default', 'disabled'],
    )
    def test_hosted_connection_forwards_include_instructions(self, capability: PostHog[None], include: bool) -> None:
        # `MCPToolset` defaults to False, so this proves the capability passes its own setting on.
        toolset = capability.get_toolset()
        assert isinstance(toolset, MCPToolset)
        assert toolset.include_instructions is include

    def test_environment_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('POSTHOG_PERSONAL_API_KEY', 'environment-key')
        assert bearer(PostHog().get_toolset()) == 'Bearer environment-key'

    def test_missing_auth_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('POSTHOG_PERSONAL_API_KEY', raising=False)
        with pytest.raises(UserError, match='Set `POSTHOG_PERSONAL_API_KEY`'):
            PostHog().get_toolset()

    @pytest.mark.filterwarnings('ignore:Using in-memory token storage')
    def test_oauth_uses_browser_login(self) -> None:
        assert isinstance(_http_transport(PostHog(auth='oauth').get_toolset()).auth, OAuth)

    def test_credential_is_not_in_repr(self) -> None:
        assert 'secret-key' not in repr(PostHog(auth='secret-key'))

    @pytest.mark.parametrize(
        ('auth', 'features'),
        [('posthog-key', None), (per_user_token, None), (None, ['flags'])],
        ids=['token', 'function', 'features'],
    )
    def test_client_cannot_be_combined_with_connection_settings(
        self, auth: str | Callable[[RunContext[str | None]], str | None] | None, features: list[str] | None
    ) -> None:
        with pytest.raises(UserError, match='`client` owns the connection'):
            PostHog[str | None](client='https://example.com/mcp', auth=auth, features=features)

    @pytest.mark.parametrize(
        'capability',
        [
            PostHog[str | None](id='tenant-posthog', auth='posthog-key'),
            PostHog[str | None](id='tenant-posthog', auth=per_user_token),
            PostHog[str | None](id='tenant-posthog', client='https://example.com/mcp'),
        ],
        ids=['token', 'function', 'client'],
    )
    def test_custom_id_is_forwarded(self, capability: PostHog[str | None]) -> None:
        assert capability.get_toolset().id == 'tenant-posthog'

    def test_defer_loading_needs_no_id(self) -> None:
        Agent(TestModel(), capabilities=[PostHog(auth='posthog-key', defer_loading=True)])

    def test_two_that_differ_raise_when_the_agent_is_built(self) -> None:
        with pytest.raises(
            UserError, match="Capability id 'posthog' is used by multiple PostHog capabilities that disagree on 'auth'"
        ):
            Agent(TestModel(), capabilities=[PostHog(auth='a'), PostHog(auth='b')])


class TestPerRunAuth:
    @pytest.mark.anyio
    async def test_each_run_connects_with_its_own_credential(self) -> None:
        capability = PostHog[str | None](auth=per_user_token)
        [alice] = await connections_for(capability, 'alice-key')
        [bob] = await connections_for(capability, 'bob-key')
        assert (bearer(alice), bearer(bob)) == ('Bearer alice-key', 'Bearer bob-key')

    @pytest.mark.anyio
    @pytest.mark.parametrize('missing', [None, ''])
    async def test_no_credential_means_no_tools(self, missing: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
        # The environment token is set to show a function never falls back to it.
        monkeypatch.setenv('POSTHOG_PERSONAL_API_KEY', 'deployment-key')
        capability = PostHog[str | None](auth=per_user_token)
        assert await connections_for(capability, missing) == []

    @pytest.mark.anyio
    async def test_function_returning_oauth_raises(self) -> None:
        capability = PostHog[str | None](auth=per_user_token)
        with pytest.raises(UserError, match="must return an API key or token, not 'oauth'"):
            await connections_for(capability, 'oauth')


class TestServerSettings:
    @pytest.mark.parametrize(('read_only', 'headers'), [(False, {}), (True, {'x-posthog-read-only': 'true'})])
    def test_read_only_is_asked_of_the_server(self, read_only: bool, headers: dict[str, str]) -> None:
        assert _http_transport(PostHog(auth='posthog-key', read_only=read_only).get_toolset()).headers == headers

    @pytest.mark.anyio
    async def test_read_only_with_a_client_keeps_only_read_only_tools(self) -> None:
        server = FastMCP('posthog-fake')

        @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
        def insight_query() -> str:
            return 'queried'

        @server.tool(annotations=ToolAnnotations(readOnlyHint=False))
        def feature_flag_create() -> str:
            return 'created'  # pragma: no cover

        agent = Agent(TestModel(), capabilities=[PostHog(client=server, read_only=True)])
        assert (await agent.run('Use the tools')).output == '{"insight_query":"queried"}'

    @pytest.mark.parametrize(
        ('features', 'query'), [('flags', 'flags'), (['flags', 'error_tracking'], 'flags%2Cerror_tracking')]
    )
    def test_features_select_feature_groups(self, features: str | list[str], query: str) -> None:
        toolset = PostHog(auth='posthog-key', features=features).get_toolset()
        assert _http_transport(toolset).url == f'https://mcp.posthog.com/mcp?features={query}'

    def test_features_are_fixed_when_the_capability_is_built(self) -> None:
        groups = ['flags']
        capability = PostHog(auth='posthog-key', features=groups)
        groups.clear()
        assert _http_transport(capability.get_toolset()).url == 'https://mcp.posthog.com/mcp?features=flags'

    def test_empty_features_raise(self) -> None:
        with pytest.raises(UserError, match='at least one feature group'):
            PostHog(auth='posthog-key', features=[])

    @pytest.mark.parametrize('group', ['flags,insights', '', ' ', 'flags '])
    def test_each_feature_names_one_group(self, group: str) -> None:
        with pytest.raises(UserError, match='must name one feature group'):
            PostHog(auth='posthog-key', features=['dashboards', group])
