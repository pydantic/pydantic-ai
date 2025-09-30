from __future__ import annotations

import os
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    RunContext,
)
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.usage import RequestUsage

from .conftest import IsDatetime, IsStr

try:
    from prefect import flow, task
    from prefect.testing.utilities import prefect_test_harness

    from pydantic_ai.durable_exec.prefect import PrefectAgent, PrefectMCPServer, PrefectModel
except ImportError:  # pragma: lax no cover
    pytest.skip('Prefect is not installed', allow_module_level=True)

try:
    import logfire
    from logfire.testing import CaptureLogfire
except ImportError:  # pragma: lax no cover
    pytest.skip('logfire not installed', allow_module_level=True)

try:
    from pydantic_ai.mcp import MCPServerStdio
except ImportError:  # pragma: lax no cover
    pytest.skip('mcp not installed', allow_module_level=True)


try:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:  # pragma: lax no cover
    pytest.skip('openai not installed', allow_module_level=True)

from inline_snapshot import snapshot

from pydantic_ai import ExternalToolset, FunctionToolset
from pydantic_ai.tools import DeferredToolResults, ToolDefinition

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
]

# We need to use a custom cached HTTP client here as the default one created for OpenAIProvider will be closed automatically
# at the end of each test, but we need this one to live longer.
http_client = cached_async_http_client(provider='prefect')


@pytest.fixture(autouse=True, scope='module')
async def close_cached_httpx_client(anyio_backend: str) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await http_client.aclose()


@pytest.fixture(autouse=True, scope='module')
def setup_logfire_instrumentation() -> Iterator[None]:
    import warnings

    # Set up logfire for the tests.
    logfire.configure(metrics=False)

    # Filter out the propagated trace context warning from logfire
    # This warning is expected when using Prefect with logfire
    warnings.filterwarnings('ignore', message='Found propagated trace context.*', category=RuntimeWarning)

    yield


@pytest.fixture(autouse=True, scope='session')
def setup_prefect_test_harness() -> Iterator[None]:
    """Set up Prefect test harness for all tests."""
    with prefect_test_harness():
        yield


@contextmanager
def flow_raises(exc_type: type[Exception], exc_message: str) -> Iterator[None]:
    """Helper for asserting that a Prefect flow fails with the expected error."""
    with pytest.raises(Exception) as exc_info:
        yield
    assert isinstance(exc_info.value, Exception)
    assert str(exc_info.value) == exc_message


model = OpenAIChatModel(
    'gpt-4o',
    provider=OpenAIProvider(
        api_key=os.getenv('OPENAI_API_KEY', 'mock-api-key'),
        http_client=http_client,
    ),
)

# Simple agent for basic testing
simple_agent = Agent(model, name='simple_agent')
simple_prefect_agent = PrefectAgent(simple_agent)


async def test_simple_agent_run_in_flow(allow_model_requests: None, openai_api_key: str) -> None:
    """Test that a simple agent can run in a Prefect flow."""

    @flow(name='test_simple_agent_run_in_flow')
    async def run_simple_agent() -> str:
        result = await simple_prefect_agent.run('What is the capital of Mexico?')
        return result.output

    output = await run_simple_agent()
    assert output == snapshot('The capital of Mexico is Mexico City.')


class Deps(BaseModel):
    country: str


# Wrap event_stream_handler as a Prefect task because it's non-deterministic (uses logfire)
@task(name='event_stream_handler')
async def event_stream_handler(
    ctx: RunContext[Deps],
    stream: AsyncIterable[AgentStreamEvent],
):
    logfire.info(f'{ctx.run_step=}')
    async for event in stream:
        logfire.info('event', event=event)


# This doesn't need to be a task
async def get_country(ctx: RunContext[Deps]) -> str:
    return ctx.deps.country


class WeatherArgs(BaseModel):
    city: str


@task(name='get_weather')
def get_weather(args: WeatherArgs) -> str:
    if args.city == 'Mexico City':
        return 'sunny'
    else:
        return 'unknown'  # pragma: no cover


from dataclasses import dataclass


@dataclass
class Answer:
    label: str
    answer: str


@dataclass
class Response:
    answers: list[Answer]


complex_agent = Agent(
    model,
    deps_type=Deps,
    output_type=Response,
    toolsets=[
        FunctionToolset[Deps](tools=[get_country], id='country'),
        MCPServerStdio('python', ['-m', 'tests.mcp_server'], timeout=20, id='mcp'),
        ExternalToolset(tool_defs=[ToolDefinition(name='external')], id='external'),
    ],
    tools=[get_weather],
    event_stream_handler=event_stream_handler,
    instrument=True,  # Enable instrumentation for testing
    name='complex_agent',
)
complex_prefect_agent = PrefectAgent(complex_agent)


async def test_complex_agent_run_in_flow(allow_model_requests: None, capfire: CaptureLogfire) -> None:
    """Test a complex agent with tools, MCP servers, and event stream handler."""

    @flow(name='test_complex_agent_run_in_flow')
    async def run_complex_agent() -> Response:
        # PrefectAgent already wraps the `run` function as a Prefect flow, so we can just call it directly.
        result = await complex_prefect_agent.run(
            'Tell me: the capital of the country; the weather there; the product name', deps=Deps(country='Mexico')
        )
        return result.output

    output = await run_complex_agent()
    assert output == snapshot(
        Response(
            answers=[
                Answer(label='Capital of the country', answer='Mexico City'),
                Answer(label='Weather in the capital', answer='Sunny'),
                Answer(label='Product Name', answer='Pydantic AI'),
            ]
        )
    )


async def test_agent_requires_name(allow_model_requests: None) -> None:
    """Test that PrefectAgent requires a name."""
    agent_without_name = Agent(model)

    with pytest.raises(Exception) as exc_info:
        PrefectAgent(agent_without_name)

    assert 'unique' in str(exc_info.value).lower() and 'name' in str(exc_info.value).lower()


async def test_agent_requires_model_at_creation(allow_model_requests: None) -> None:
    """Test that PrefectAgent requires model to be set at creation time."""
    agent_without_model = Agent(None, name='test_agent')

    with pytest.raises(Exception) as exc_info:
        PrefectAgent(agent_without_model)

    assert 'model' in str(exc_info.value).lower()


async def test_run_sync_in_flow(allow_model_requests: None, openai_api_key: str) -> None:
    """Test that run_sync works in a Prefect flow."""

    @flow(name='test_run_sync_in_flow')
    def run_simple_agent_sync() -> str:
        result = simple_prefect_agent.run_sync('What is the capital of France?')
        return result.output

    output = run_simple_agent_sync()
    assert output == snapshot('The capital of France is Paris.')
