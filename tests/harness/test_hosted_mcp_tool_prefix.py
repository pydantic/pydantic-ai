"""Test how hosted MCP capabilities name the server's tools: `tool_prefix`, its defaults, and read-only selection."""

from __future__ import annotations

from typing import Protocol

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolsetClient
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.day_ai import DayAI
from pydantic_ai_harness.github import GitHub
from pydantic_ai_harness.grain import Grain
from pydantic_ai_harness.linear import Linear
from pydantic_ai_harness.logfire_mcp import LogfireMCP
from pydantic_ai_harness.notion import Notion
from pydantic_ai_harness.ordinal import Ordinal
from pydantic_ai_harness.posthog import PostHog
from pydantic_ai_harness.pylon import Pylon
from pydantic_ai_harness.slack import Slack

# MCP's test server leaves its lifespan annotation unresolved with pydantic-settings 2.15.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:Field 'lifespan' has an incomplete definition:UserWarning:pydantic_settings.sources.utils"
    ),
]


class HostedMCP(Protocol):
    __name__: str

    def __call__(
        self, *, client: MCPToolsetClient, id: str | None = ..., tool_prefix: str | None = ...
    ) -> AbstractCapability[None]: ...


class ReadOnlyCapable(Protocol):
    __name__: str

    def __call__(
        self,
        *,
        client: MCPToolsetClient,
        id: str | None = ...,
        tool_prefix: str | None = ...,
        read_only: bool = ...,
    ) -> AbstractCapability[None]: ...


READ_ONLY_CAPABLE: list[ReadOnlyCapable] = [
    LogfireMCP[None],
    GitHub[None],
    Linear[None],
    Notion[None],
    Slack[None],
    PostHog[None],
    Grain[None],
    Pylon[None],
]
ALL: list[HostedMCP] = [*READ_ONLY_CAPABLE, DayAI[None], Ordinal[None]]


@pytest.fixture
def server() -> FastMCP:
    server = FastMCP('provider')

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def read() -> str:
        return 'read'

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False))
    def write() -> str:
        return 'written'

    return server


async def offered_tools(agent: Agent[None, str]) -> list[str]:
    """The tool names the agent sends the model."""
    model = TestModel(call_tools=[])
    await agent.run('List the tools', model=model)
    assert model.last_model_request_parameters is not None
    return [tool.name for tool in model.last_model_request_parameters.function_tools]


@pytest.mark.parametrize(
    ('capability', 'expected'),
    [
        (LogfireMCP[None], ['logfire_read', 'logfire_write']),
        (GitHub[None], ['github_read', 'github_write']),
        (Linear[None], ['linear_read', 'linear_write']),
        (PostHog[None], ['posthog_read', 'posthog_write']),
        (Grain[None], ['grain_read', 'grain_write']),
        (DayAI[None], ['day_ai_read', 'day_ai_write']),
        (Pylon[None], ['pylon_read', 'pylon_write']),
        # These servers already start their tool names with the service.
        (Notion[None], ['read', 'write']),
        (Slack[None], ['read', 'write']),
        (Ordinal[None], ['read', 'write']),
    ],
    ids=lambda value: getattr(value, '__name__', None),
)
async def test_default_prefix(server: FastMCP, capability: HostedMCP, expected: list[str]) -> None:
    assert await offered_tools(Agent(deps_type=type(None), capabilities=[capability(client=server)])) == expected


@pytest.mark.parametrize('capability', ALL, ids=lambda capability: capability.__name__)
async def test_none_keeps_the_server_names(server: FastMCP, capability: HostedMCP) -> None:
    assert await offered_tools(
        Agent(deps_type=type(None), capabilities=[capability(client=server, tool_prefix=None)])
    ) == ['read', 'write']


@pytest.mark.parametrize('capability', ALL, ids=lambda capability: capability.__name__)
async def test_two_connections_with_distinct_prefixes(server: FastMCP, capability: HostedMCP) -> None:
    agent = Agent(
        TestModel(),
        deps_type=type(None),
        capabilities=[
            capability(client=server, id='work', tool_prefix='work'),
            capability(client=server, id='personal', tool_prefix='personal'),
        ],
    )
    result = await agent.run('Use the tools')
    assert result.output == (
        '{"work_read":"read","work_write":"written","personal_read":"read","personal_write":"written"}'
    )


@pytest.mark.parametrize('capability', READ_ONLY_CAPABLE, ids=lambda capability: capability.__name__)
async def test_read_only_selects_by_the_server_annotations_under_a_prefix(
    server: FastMCP, capability: ReadOnlyCapable
) -> None:
    agent = Agent(
        TestModel(), deps_type=type(None), capabilities=[capability(client=server, tool_prefix='work', read_only=True)]
    )
    assert (await agent.run('Use the tools')).output == '{"work_read":"read"}'


@pytest.mark.parametrize('capability', ALL, ids=lambda capability: capability.__name__)
def test_two_that_differ_under_one_id_point_at_tool_prefix(server: FastMCP, capability: HostedMCP) -> None:
    with pytest.raises(UserError, match=r"disagree on 'tool_prefix'\. Give each its own `id` and `tool_prefix`"):
        Agent(
            TestModel(),
            deps_type=type(None),
            capabilities=[capability(client=server, tool_prefix='work'), capability(client=server, tool_prefix='oss')],
        )
