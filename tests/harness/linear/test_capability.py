"""Test Linear's connection settings and tool selection through an agent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeGuard, TypeVar

import httpx
import pytest
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    DeferredToolRequests,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.linear import Linear
from tests.harness._hosted_mcp import mcp_connection, only_leaf

from .._docs_examples import python_blocks, run_block

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


async def connections_for(capability: Linear[str | None], deps: str | None) -> list[MCPToolset[str | None]]:
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


def bearer(toolset: AbstractToolset[DepsT]) -> str:
    auth = transport(toolset).auth
    assert auth is not None
    request = next(auth.auth_flow(httpx.Request('POST', 'https://example.com/mcp')))
    return request.headers['Authorization']


class TestLinear:
    @pytest.mark.parametrize(
        ('read_only', 'expected'),
        [
            (
                False,
                '{"linear_read_resource":"read","linear_write_resource":"written","linear_unmarked_resource":"unmarked"}',
            ),
            (True, '{"linear_read_resource":"read"}'),
        ],
    )
    async def test_agent_executes_selected_tools(self, server: FastMCP, read_only: bool, expected: str) -> None:
        agent = Agent(TestModel(), capabilities=[Linear(client=server, read_only=read_only)])
        result = await agent.run('Use the tools')
        assert result.output == expected

    @pytest.mark.parametrize('include', [True, False])
    async def test_server_instructions(self, server: FastMCP, include: bool) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[Linear(client=server, include_instructions=include)])
        result = await agent.run('Hello')
        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert ('Provider instructions.' in (request.instructions or '')) is include

    @pytest.mark.parametrize('auth', ['key', per_user_token])
    def test_client_cannot_be_combined_with_connection_settings(
        self, auth: str | Callable[[RunContext[str | None]], str | None]
    ) -> None:
        with pytest.raises(UserError, match='`client` owns the connection'):
            Linear(client='https://example.com/mcp', auth=auth)

    @pytest.mark.parametrize(
        'capability',
        [
            Linear[str | None](id='tenant-linear', auth='linear-token'),
            Linear[str | None](id='tenant-linear', auth=per_user_token),
            Linear[str | None](id='tenant-linear', client='https://example.com/mcp'),
        ],
        ids=['token', 'function', 'client'],
    )
    def test_custom_id_is_forwarded(self, capability: Linear[str | None]) -> None:
        assert only_leaf(capability.get_toolset()).id == 'tenant-linear'

    def test_read_only_client_keeps_the_id_on_its_mcp_toolset(self) -> None:
        # The read-only filter wraps the connection, so the id lives on the `MCPToolset` inside it.
        toolset = Linear[None](id='tenant-linear', client='https://example.com/mcp', read_only=True).get_toolset()
        ids: list[str | None] = []
        toolset.apply(lambda leaf: ids.append(leaf.id))
        assert ids == ['tenant-linear']

    @pytest.mark.parametrize('id', [None, ''])
    def test_unset_id_connects_under_the_default_id(self, id: str | None) -> None:
        assert only_leaf(Linear[None](id=id, auth='linear-token').get_toolset()).id == 'linear'

    def test_unset_ids_are_not_combined(self) -> None:
        # An unset id has nothing to combine on, so differing settings don't raise when the agent is built,
        # as with `GitHub(id=None)`.
        Agent(
            TestModel(),
            capabilities=[
                Linear(id=None, auth='a', include_instructions=False),
                Linear(id=None, auth='b', include_instructions=False),
            ],
        )

    def test_client_accepts_empty_auth(self) -> None:
        # The `auth` table documents `''` as unset, so it does not conflict with `client`.
        toolset = Linear[None](client='https://example.com/mcp', auth='').get_toolset()
        assert transport(toolset).url == 'https://example.com/mcp'

    def test_defer_loading_needs_no_id(self, server: FastMCP) -> None:
        Agent(TestModel(), capabilities=[Linear(client=server, defer_loading=True)])

    def test_two_that_differ_raise_when_the_agent_is_built(self) -> None:
        with pytest.raises(
            UserError,
            match="Capability id 'linear' is used by multiple Linear capabilities that disagree on 'auth', 'read_only'",
        ):
            Agent(TestModel(), capabilities=[Linear(auth='a'), Linear(auth='b', read_only=True)])

    @pytest.mark.parametrize(
        ('capability', 'include'),
        [
            (Linear[str | None](auth='token'), True),
            (Linear[str | None](auth='token', include_instructions=False), False),
        ],
        ids=['default', 'disabled'],
    )
    def test_hosted_connection_forwards_include_instructions(
        self, capability: Linear[str | None], include: bool
    ) -> None:
        # `MCPToolset` defaults to False, so this proves the capability passes its own setting on.
        toolset = mcp_connection(capability.get_toolset())
        assert toolset.include_instructions is include

    def test_credential_is_not_in_repr(self) -> None:
        assert 'secret-token' not in repr(Linear(auth='secret-token'))

    def test_environment_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('LINEAR_ACCESS_TOKEN', 'environment-token')
        assert bearer(Linear().get_toolset()) == 'Bearer environment-token'

    @pytest.mark.parametrize('auth', ['token', per_user_token], ids=['fixed', 'per-run'])
    @pytest.mark.parametrize('read_only', [True, False])
    async def test_native_read_only_endpoint(
        self, auth: str | Callable[[RunContext[str | None]], str | None], read_only: bool
    ) -> None:
        suffix = '/readonly' if read_only else ''
        [toolset] = await connections_for(Linear[str | None](auth=auth, read_only=read_only), 'token')
        assert transport(toolset).url == 'https://mcp.linear.app/mcp' + suffix

    def test_missing_token_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('LINEAR_ACCESS_TOKEN', raising=False)
        with pytest.raises(UserError, match='Set `LINEAR_ACCESS_TOKEN`'):
            Linear().get_toolset()

    @pytest.mark.filterwarnings('ignore:Using in-memory token storage')
    def test_fixed_oauth_uses_browser_login(self) -> None:
        assert isinstance(transport(Linear(auth='oauth').get_toolset()).auth, OAuth)


async def per_user_token_async(ctx: RunContext[str | None]) -> str | None:
    """The same as `per_user_token`, for an app that loads the token asynchronously."""
    return ctx.deps


class TestPerRunAuth:
    @pytest.mark.parametrize('auth', [per_user_token, per_user_token_async], ids=['sync', 'async'])
    async def test_each_run_connects_with_its_own_credential(
        self, auth: Callable[[RunContext[str | None]], str | None | Awaitable[str | None]]
    ) -> None:
        capability = Linear[str | None](auth=auth)
        [alice] = await connections_for(capability, 'alice-token')
        [bob] = await connections_for(capability, 'bob-token')
        assert (bearer(alice), bearer(bob)) == ('Bearer alice-token', 'Bearer bob-token')

    @pytest.mark.parametrize('missing', [None, ''])
    async def test_no_credential_means_no_tools(self, missing: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
        # The environment token is set to show a function never falls back to it.
        monkeypatch.setenv('LINEAR_ACCESS_TOKEN', 'deployment-token')
        capability = Linear[str | None](auth=per_user_token)
        assert await connections_for(capability, missing) == []

    async def test_function_returning_oauth_raises(self) -> None:
        capability = Linear[str | None](auth=per_user_token)
        with pytest.raises(UserError, match="must return an API key or token, not 'oauth'"):
            await connections_for(capability, 'oauth')


def fake_linear() -> FastMCP:
    """A stand-in for Linear's server, so the documented examples run offline."""
    server = FastMCP('linear-fake')

    @server.tool()
    def list_issues() -> str:
        return 'issues'

    return server


def calls_list_issues(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Calls `linear_list_issues` once when the run offers it, then answers."""
    if len(messages) == 1 and any(tool.name == 'linear_list_issues' for tool in info.function_tools):
        return ModelResponse(parts=[ToolCallPart('linear_list_issues', {})])
    return ModelResponse(parts=[TextPart('done')])


def is_agent(value: object) -> TypeGuard[Agent[object, object]]:
    """Whether a docs block bound `agent` to an `Agent`, checked at run time like any `isinstance`."""
    return isinstance(value, Agent)


def tool_returns(result: AgentRunResult[object]) -> list[object]:
    return [part.content for part in result.new_messages()[-2].parts if isinstance(part, ToolReturnPart)]


class TestDocs:
    @pytest.mark.parametrize(
        'page', ['docs/harness/linear.md', 'src/pydantic_ai_harness/pydantic_ai_harness/linear/README.md']
    )
    def test_documented_examples_run(self, page: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('LINEAR_ACCESS_TOKEN', 'docs-token')
        connections: list[tuple[str, str | None]] = []

        # Takes the arguments `Linear` passes `MCPToolset`, and connects to `fake_linear()` instead.
        def offline(
            client: str, *, id: str | None, include_instructions: bool, auth: str | None = None
        ) -> MCPToolset[object]:
            connections.append((client, auth))
            return MCPToolset(fake_linear(), id=id, include_instructions=include_instructions)

        monkeypatch.setattr('pydantic_ai_harness.linear._capability.MCPToolset', offline)
        model = FunctionModel(calls_list_issues)
        quickstart, per_user, approval = python_blocks(page)

        namespace, runs = run_block(quickstart)
        assert len(runs) == 1
        agent = namespace['agent']
        assert is_agent(agent)
        assert tool_returns(agent.run_sync('List my open issues', model=model)) == ['issues']
        assert connections == [('https://mcp.linear.app/mcp/readonly', 'docs-token')]

        namespace, _ = run_block(per_user)
        agent, deps_type = namespace['agent'], namespace['Deps']
        assert is_agent(agent) and isinstance(deps_type, type)
        assert tool_returns(agent.run_sync('List my open issues', deps=deps_type('user-token'), model=model)) == [
            'issues'
        ]
        assert connections[-1] == ('https://mcp.linear.app/mcp', 'user-token')
        assert agent.run_sync('List my open issues', deps=deps_type(None), model=model).output == 'done'
        assert len(connections) == 2

        namespace, _ = run_block(approval)
        agent = namespace['agent']
        assert is_agent(agent)
        output = agent.run_sync('List my open issues', model=model).output
        assert isinstance(output, DeferredToolRequests)
        assert [call.tool_name for call in output.approvals] == ['linear_list_issues']
        assert connections[-1] == ('https://mcp.linear.app/mcp', 'docs-token')
