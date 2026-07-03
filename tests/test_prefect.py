# pyright: reportDeprecated=false
# `PrefectAgent` (the wrapper-agent path) is deprecated in favor of the
# `PrefectDurability` capability, but this file still exercises both paths in
# parallel for parity. Silenced at file level rather than annotating every
# individual usage.
from __future__ import annotations

import os
import uuid
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    AgentRunResult,
    AgentRunResultEvent,
    AgentStreamEvent,
    ExternalToolset,
    FunctionToolset,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    RunContext,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.usage import RequestUsage, RunUsage

try:
    from prefect import flow, task
    from prefect.testing.utilities import prefect_test_harness

    from pydantic_ai.durable_exec.prefect import (
        DEFAULT_PYDANTIC_AI_CACHE_POLICY,
        PrefectAgent,
        PrefectDurability,
        PrefectFunctionToolset,
        PrefectMCPToolset,
        PrefectModel,
        TaskConfig,
    )
    from pydantic_ai.durable_exec.prefect._cache_policies import (
        PrefectAgentInputs,
        _replace_run_context,  # pyright: ignore[reportPrivateUsage]
    )
except ImportError:  # pragma: lax no cover
    pytest.skip('Prefect is not installed', allow_module_level=True)

try:
    import logfire
    from logfire.testing import CaptureLogfire
except ImportError:  # pragma: lax no cover
    pytest.skip('logfire not installed', allow_module_level=True)

try:
    from fastmcp.client.transports import StdioTransport

    from pydantic_ai.mcp import MCPToolset
except ImportError:  # pragma: lax no cover
    pytest.skip('mcp not installed', allow_module_level=True)

try:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:  # pragma: lax no cover
    pytest.skip('openai not installed', allow_module_level=True)

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsStr

# `PrefectAgent` is deprecated in favor of `capabilities=[PrefectDurability(...)]`, and
# the legacy `MCPServer*` / `FastMCPToolset` classes are deprecated in favor of `MCPToolset`.
# These tests exercise the wrapper-agent path on purpose; suppress the warnings here
# rather than globally in `pyproject.toml`. The `pytestmark` entries below cover warnings
# emitted *inside* test functions; the `filterwarnings` calls below cover warnings emitted
# at module import time (e.g. module-level construction of `PrefectAgent`).
warnings.filterwarnings('ignore', message='`PrefectAgent` is deprecated', category=DeprecationWarning)
warnings.filterwarnings(
    'ignore',
    message=r'`(MCPServerStdio|MCPServerSSE|MCPServerStreamableHTTP|FastMCPToolset)` is deprecated',
    category=DeprecationWarning,
)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.xdist_group(name='prefect'),
    # TODO(Marcelo): We are temporarily disabling it. We should enable them again.
    # Root cause + latest context: https://github.com/pydantic/pydantic-ai/issues/3929 — VCR's
    # httpcore stub wraps the in-process Prefect temp-server's localhost connections despite
    # `ignore_localhost`, exhausting the pool until `httpcore.PoolTimeout`. Version-independent; a
    # fix attempt (#3930) was reverted (#3932). The hang is confined to tests that run an agent
    # flow (`await prefect_agent.run(...)` inside an `@flow`): the `test_cache_policy_*` and
    # `test_cache_key_run_context_projection_is_exhaustive` unit tests run no flow and pass in
    # seconds, so this blanket skip could be narrowed to only the flow-running tests (which would
    # also mean revisiting the `durable_exec/prefect/*` + `tests/test_prefect.py` coverage omit).
    pytest.mark.skip('This test suite is hanging with the latest versions of all packages.'),
    pytest.mark.filterwarnings('ignore:`PrefectAgent` is deprecated:DeprecationWarning'),
    pytest.mark.filterwarnings(
        'ignore:`(MCPServerStdio|MCPServerSSE|MCPServerStreamableHTTP|FastMCPToolset)` is deprecated:DeprecationWarning'
    ),
]

# We need to use a custom cached HTTP client here as the default one created for OpenAIProvider will be closed automatically
# at the end of each test, but we need this one to live longer.
http_client = create_async_http_client()


@pytest.fixture(autouse=True, scope='module')
async def close_cached_httpx_client(anyio_backend: str) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await http_client.aclose()


@pytest.fixture(autouse=True)
def setup_logfire_instrumentation() -> Iterator[None]:
    # Set up logfire for the tests. Prefect sets the `traceparent` header, so we explicitly enable
    # distributed tracing the tests to avoid the warning.
    logfire.configure(metrics=False, distributed_tracing=False)

    yield


@pytest.fixture(autouse=True, scope='session')
def setup_prefect_test_harness() -> Iterator[None]:
    """Set up Prefect test harness for all tests."""
    with prefect_test_harness(server_startup_timeout=60):
        yield


@contextmanager
def flow_raises(exc_type: type[Exception], exc_message: str) -> Generator[None]:
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


async def test_simple_agent_run_in_flow(allow_model_requests: None) -> None:
    """Test that a simple agent can run in a Prefect flow."""

    @flow(name='test_simple_agent_run_in_flow')
    async def run_simple_agent() -> str:
        result = await simple_prefect_agent.run('What is the capital of Mexico?')
        return result.output

    output = await run_simple_agent()
    assert output == snapshot('The capital of Mexico is Mexico City.')


class Deps(BaseModel):
    country: str


async def event_stream_handler(
    ctx: RunContext[Deps],
    stream: AsyncIterable[AgentStreamEvent],
):
    logfire.info(f'{ctx.run_step=}')
    async for event in stream:
        logfire.info('event', event=event)


async def runtime_event_stream_handler(
    ctx: RunContext[object],
    stream: AsyncIterable[AgentStreamEvent],
):
    logfire.info(f'{ctx.run_step=}')
    async for event in stream:
        logfire.info('runtime_event', event=event)


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


@dataclass
class Answer:
    label: str
    answer: str


@dataclass
class Response:
    answers: list[Answer]


@dataclass
class BasicSpan:
    content: str
    children: list[BasicSpan] = field(default_factory=list['BasicSpan'])
    parent_id: int | None = field(repr=False, compare=False, default=None)


complex_agent = Agent(
    model,
    deps_type=Deps,
    output_type=Response,
    toolsets=[
        FunctionToolset[Deps](tools=[get_country], id='country'),
        MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='mcp', init_timeout=20),
        ExternalToolset(tool_defs=[ToolDefinition(name='external')], id='external'),
    ],
    tools=[get_weather],
    capabilities=[Instrumentation(settings=InstrumentationSettings())],
    name='complex_agent',
)
complex_prefect_agent = PrefectAgent(complex_agent, event_stream_handler=event_stream_handler)


async def runtime_handler_stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[str]:
    del messages, agent_info
    yield 'Hello'
    yield ' world'


runtime_handler_stream_agent = Agent(
    FunctionModel(stream_function=runtime_handler_stream_function),
    name='runtime_handler_stream_agent',
)
runtime_handler_stream_prefect_agent = PrefectAgent(runtime_handler_stream_agent)


async def test_complex_agent_run_in_flow(allow_model_requests: None, capfire: CaptureLogfire) -> None:
    """Test a complex agent with tools, MCP servers, and event stream handler."""

    @flow(name='test_complex_agent_run_in_flow')
    async def run_complex_agent() -> Response:
        # Use sequential tool calls to avoid flaky test due to non-deterministic ordering
        with Agent.parallel_tool_call_execution_mode('sequential'):
            result = await complex_prefect_agent.run(
                'Tell me: the capital of the country; the weather there; the product name', deps=Deps(country='Mexico')
            )
        return result.output

    # Prefect sets the `traceparent` header, so we explicitly disable distributed tracing for the tests to avoid the warning,
    # but we can't set that configuration for the capfire fixture, so we ignore the warning here.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        output = await run_complex_agent()
    assert output == snapshot(
        Response(
            answers=[
                Answer(label='Capital of the country', answer='Mexico City'),
                Answer(label='Weather in the capital', answer='Sunny'),
                Answer(label='Product name', answer='Pydantic AI'),
            ]
        )
    )

    # Verify logfire instrumentation with full span tree
    exporter = capfire.exporter
    spans = exporter.exported_spans_as_dict()
    basic_spans_by_id = {
        span['context']['span_id']: BasicSpan(
            parent_id=span['parent']['span_id'] if span['parent'] else None,
            content=attributes.get('event') or attributes['logfire.msg'],
        )
        for span in spans
        if (attributes := span.get('attributes'))
    }
    root_span = None
    for basic_span in basic_spans_by_id.values():
        if basic_span.parent_id is None:
            root_span = basic_span
        else:
            parent_id = basic_span.parent_id
            parent_span = basic_spans_by_id[parent_id]
            parent_span.children.append(basic_span)

    assert root_span == snapshot(
        BasicSpan(
            content=IsStr(regex=r'\w+-\w+'),  # Random Prefect flow run name
            children=[
                BasicSpan(
                    content='Found propagated trace context. See https://logfire.pydantic.dev/docs/how-to-guides/distributed-tracing/#unintentional-distributed-tracing.'
                ),
                BasicSpan(
                    content=IsStr(regex=r'\w+-\w+'),  # Random Prefect flow run name
                    children=[
                        BasicSpan(
                            content='complex_agent run',
                            children=[
                                BasicSpan(
                                    content='chat gpt-4o',
                                    children=[
                                        BasicSpan(
                                            content=IsStr(regex=r'Model Request \(Streaming\): gpt-4o-\w+'),
                                            children=[
                                                BasicSpan(content='ctx.run_step=1'),
                                                BasicSpan(
                                                    content='{"index":0,"part":{"tool_name":"get_country","args":"","tool_call_id":"call_rI3WKPYvVwlOgCGRjsPP2hEx","id":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":null,"event_kind":"part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{}","tool_call_id":"call_rI3WKPYvVwlOgCGRjsPP2hEx","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"part":{"tool_name":"get_country","args":"{}","tool_call_id":"call_rI3WKPYvVwlOgCGRjsPP2hEx","id":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":null,"event_kind":"part_end"}'
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                                BasicSpan(
                                    content=IsStr(regex=r'Handle Stream Event-\w+'),
                                    children=[
                                        BasicSpan(content='ctx.run_step=1'),
                                        BasicSpan(
                                            content='{"part":{"tool_name":"get_country","args":"{}","tool_call_id":"call_rI3WKPYvVwlOgCGRjsPP2hEx","id":null,"provider_details":null,"part_kind":"tool-call"},"event_kind":"function_tool_call"}'
                                        ),
                                    ],
                                ),
                                BasicSpan(
                                    content='running tool: get_country',
                                    children=[BasicSpan(content=IsStr(regex=r'Call Tool: get_country-\w+'))],
                                ),
                                BasicSpan(
                                    content=IsStr(regex=r'Handle Stream Event-\w+'),
                                    children=[
                                        BasicSpan(content='ctx.run_step=1'),
                                        BasicSpan(
                                            content=IsStr(
                                                regex=r'\{"result":\{"tool_name":"get_country","content":"Mexico","tool_call_id":"call_rI3WKPYvVwlOgCGRjsPP2hEx","metadata":null,"timestamp":"[^"]+","part_kind":"tool-return"\},"content":null,"event_kind":"function_tool_result"\}'
                                            )
                                        ),
                                    ],
                                ),
                                BasicSpan(
                                    content='chat gpt-4o',
                                    children=[
                                        BasicSpan(
                                            content=IsStr(regex=r'Model Request \(Streaming\): gpt-4o-\w+'),
                                            children=[
                                                BasicSpan(content='ctx.run_step=2'),
                                                BasicSpan(
                                                    content='{"index":0,"part":{"tool_name":"get_weather","args":"","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","id":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":null,"event_kind":"part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"ci","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"ty\\": ","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"Mexic","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"o Ci","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"ty\\"}","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"part":{"tool_name":"get_weather","args":"{\\"city\\": \\"Mexico City\\"}","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","id":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":"tool-call","event_kind":"part_end"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":1,"part":{"tool_name":"get_product_name","args":"","tool_call_id":"call_SkGkkGDvHQEEk0CGbnAh2AQw","id":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":"tool-call","event_kind":"part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":1,"delta":{"tool_name_delta":null,"args_delta":"{}","tool_call_id":"call_SkGkkGDvHQEEk0CGbnAh2AQw","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":1,"part":{"tool_name":"get_product_name","args":"{}","tool_call_id":"call_SkGkkGDvHQEEk0CGbnAh2AQw","id":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":null,"event_kind":"part_end"}'
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                                BasicSpan(
                                    content=IsStr(regex=r'Handle Stream Event-\w+'),
                                    children=[
                                        BasicSpan(content='ctx.run_step=2'),
                                        BasicSpan(
                                            content='{"part":{"tool_name":"get_weather","args":"{\\"city\\": \\"Mexico City\\"}","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","id":null,"provider_details":null,"part_kind":"tool-call"},"event_kind":"function_tool_call"}'
                                        ),
                                    ],
                                ),
                                BasicSpan(
                                    content=IsStr(regex=r'Handle Stream Event-\w+'),
                                    children=[
                                        BasicSpan(content='ctx.run_step=2'),
                                        BasicSpan(
                                            content='{"part":{"tool_name":"get_product_name","args":"{}","tool_call_id":"call_SkGkkGDvHQEEk0CGbnAh2AQw","id":null,"provider_details":null,"part_kind":"tool-call"},"event_kind":"function_tool_call"}'
                                        ),
                                    ],
                                ),
                                BasicSpan(
                                    content='running tool: get_weather',
                                    children=[
                                        BasicSpan(
                                            content=IsStr(regex=r'Call Tool: get_weather-\w+'),
                                            children=[BasicSpan(content=IsStr(regex=r'get_weather-\w+'))],
                                        )
                                    ],
                                ),
                                BasicSpan(
                                    content=IsStr(regex=r'Handle Stream Event-\w+'),
                                    children=[
                                        BasicSpan(content='ctx.run_step=2'),
                                        BasicSpan(
                                            content=IsStr(
                                                regex=r'\{"result":\{"tool_name":"get_weather","content":"sunny","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","metadata":null,"timestamp":"[^"]+","part_kind":"tool-return"\},"content":null,"event_kind":"function_tool_result"\}'
                                            )
                                        ),
                                    ],
                                ),
                                BasicSpan(
                                    content='running tool: get_product_name',
                                    children=[BasicSpan(content=IsStr(regex=r'Call MCP Tool: get_product_name-\w+'))],
                                ),
                                BasicSpan(
                                    content=IsStr(regex=r'Handle Stream Event-\w+'),
                                    children=[
                                        BasicSpan(content='ctx.run_step=2'),
                                        BasicSpan(
                                            content=IsStr(
                                                regex=r'\{"result":\{"tool_name":"get_product_name","content":"Pydantic AI","tool_call_id":"call_SkGkkGDvHQEEk0CGbnAh2AQw","metadata":null,"timestamp":"[^"]+","part_kind":"tool-return"\},"content":null,"event_kind":"function_tool_result"\}'
                                            )
                                        ),
                                    ],
                                ),
                                BasicSpan(
                                    content='chat gpt-4o',
                                    children=[
                                        BasicSpan(
                                            content=IsStr(regex=r'Model Request \(Streaming\): gpt-4o-\w+'),
                                            children=[
                                                BasicSpan(content='ctx.run_step=3'),
                                                BasicSpan(
                                                    content='{"index":0,"part":{"tool_name":"final_result","args":"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","id":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":null,"event_kind":"part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"tool_name":"final_result","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","event_kind":"final_result"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answers","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":[","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Capital","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" of","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" the","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" country","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Mexico","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" City","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"},{\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Weather","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" in","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" the","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" capital","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Sunny","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"},{\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Product","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" name","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"P","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"yd","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"antic","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" AI","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"}","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"]}","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"part":{"tool_name":"final_result","args":"{\\"answers\\":[{\\"label\\":\\"Capital of the country\\",\\"answer\\":\\"Mexico City\\"},{\\"label\\":\\"Weather in the capital\\",\\"answer\\":\\"Sunny\\"},{\\"label\\":\\"Product name\\",\\"answer\\":\\"Pydantic AI\\"}]}","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","id":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":null,"event_kind":"part_end"}'
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
            ],
        )
    )


async def test_multiple_agents(allow_model_requests: None) -> None:
    """Test that multiple agents can run in a Prefect flow."""

    @flow(name='test_multiple_agents')
    async def run_multiple_agents() -> tuple[str, Response]:
        result1 = await simple_prefect_agent.run('What is the capital of Mexico?')
        result2 = await complex_prefect_agent.run(
            'Tell me: the capital of the country; the weather there; the product name', deps=Deps(country='Mexico')
        )
        return result1.output, result2.output

    output1, output2 = await run_multiple_agents()
    assert output1 == snapshot('The capital of Mexico is Mexico City.')
    assert output2 == snapshot(
        Response(
            answers=[
                Answer(label='Capital of the Country', answer='The capital of Mexico is Mexico City.'),
                Answer(label='Weather in the Capital', answer='The weather in Mexico City is currently sunny.'),
                Answer(label='Product Name', answer='The product name is Pydantic AI.'),
            ]
        )
    )


async def test_prefect_agent_run_in_flow_with_runtime_event_stream_handler(
    allow_model_requests: None, capfire: CaptureLogfire
) -> None:
    @flow(name='test_prefect_agent_run_in_flow_with_runtime_event_stream_handler')
    async def run_agent() -> AgentRunResult[str]:
        return await runtime_handler_stream_prefect_agent.run(
            'Say hello', event_stream_handler=runtime_event_stream_handler
        )

    result = await run_agent()
    assert result.output == snapshot('Hello world')

    exported_messages = [
        attributes['logfire.msg']
        for span in capfire.exporter.exported_spans_as_dict()
        if (attributes := span.get('attributes')) and attributes.get('logfire.msg') == 'runtime_event'
    ]
    assert exported_messages != []


async def test_agent_requires_name() -> None:
    """Test that PrefectAgent requires a name."""
    agent_without_name = Agent(model)

    with pytest.raises(UserError) as exc_info:
        PrefectAgent(agent_without_name)

    assert 'unique' in str(exc_info.value).lower() and 'name' in str(exc_info.value).lower()


async def test_agent_requires_model_at_creation() -> None:
    """Test that PrefectAgent requires model to be set at creation time."""
    agent_without_model = Agent(name='test_agent')

    with pytest.raises(UserError) as exc_info:
        PrefectAgent(agent_without_model)

    assert 'model' in str(exc_info.value).lower()


async def test_toolset_without_id():
    """Test that agents can be created with toolsets without IDs."""
    # This is allowed in Prefect
    PrefectAgent(Agent(model=model, name='test_agent', toolsets=[FunctionToolset()]))


async def test_prefect_agent():
    """Test that PrefectAgent properly wraps model and toolsets."""
    assert isinstance(complex_prefect_agent.model, PrefectModel)
    assert complex_prefect_agent.model.wrapped == complex_agent.model

    # Prefect wraps MCP servers and function toolsets
    toolsets = complex_prefect_agent.toolsets
    # Note: toolsets include the output toolset which is not wrapped
    assert len(toolsets) >= 4

    # Find the wrapped toolsets (skip the internal output toolset)
    prefect_function_toolsets = [ts for ts in toolsets if isinstance(ts, PrefectFunctionToolset)]
    prefect_mcp_toolsets = [ts for ts in toolsets if isinstance(ts, PrefectMCPToolset)]
    external_toolsets = [ts for ts in toolsets if isinstance(ts, ExternalToolset)]

    # Verify we have the expected wrapped toolsets
    assert len(prefect_function_toolsets) >= 2  # agent tools + country toolset
    assert len(prefect_mcp_toolsets) == 1  # mcp toolset
    assert len(external_toolsets) == 1  # external toolset

    # Verify MCP toolset is wrapped (complex_agent.toolsets[1] is the `MCPToolset` for mcp).
    mcp_toolset = prefect_mcp_toolsets[0]
    assert mcp_toolset.id == 'mcp'
    assert isinstance(mcp_toolset.wrapped, MCPToolset)

    # Verify external toolset is NOT wrapped (passed through)
    external_toolset = external_toolsets[0]
    assert external_toolset.id == 'external'


def test_prefect_wrapper_visit_and_replace():
    """Prefect wrapper toolsets should not be replaced by visit_and_replace."""
    toolsets = complex_prefect_agent.toolsets
    prefect_function_toolsets = [ts for ts in toolsets if isinstance(ts, PrefectFunctionToolset)]
    assert len(prefect_function_toolsets) >= 1

    prefect_toolset = prefect_function_toolsets[0]

    # visit_and_replace should return self for Prefect wrappers
    result = prefect_toolset.visit_and_replace(lambda t: FunctionToolset(id='replaced'))
    assert result is prefect_toolset


async def test_prefect_agent_run(allow_model_requests: None) -> None:
    """Test that agent.run() works (auto-wrapped as flow)."""
    result = await simple_prefect_agent.run('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is Mexico City.')


def test_prefect_agent_run_sync(allow_model_requests: None):
    """Test that agent.run_sync() works."""
    result = simple_prefect_agent.run_sync('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is Mexico City.')


async def test_prefect_agent_run_stream(allow_model_requests: None):
    """Test that agent.run_stream() works outside of flows."""
    async with simple_prefect_agent.run_stream('What is the capital of Mexico?') as result:
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(
            [
                'The',
                'The capital',
                'The capital of',
                'The capital of Mexico',
                'The capital of Mexico is',
                'The capital of Mexico is Mexico',
                'The capital of Mexico is Mexico City',
                'The capital of Mexico is Mexico City.',
            ]
        )


async def test_prefect_agent_run_stream_events(allow_model_requests: None):
    """Test that agent.run_stream_events() works."""
    async with simple_prefect_agent.run_stream_events('What is the capital of Mexico?') as event_stream:
        events = [event async for event in event_stream]
    assert events == snapshot(
        [AgentRunResultEvent(result=AgentRunResult(output='The capital of Mexico is Mexico City.'))]
    )


async def test_prefect_agent_iter(allow_model_requests: None):
    """Test that agent.iter() works."""
    outputs: list[str] = []
    async with simple_prefect_agent.iter('What is the capital of Mexico?') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for chunk in stream.stream_text(debounce_by=None):
                        outputs.append(chunk)
    assert outputs == snapshot(
        [
            'The',
            'The capital',
            'The capital of',
            'The capital of Mexico',
            'The capital of Mexico is',
            'The capital of Mexico is Mexico',
            'The capital of Mexico is Mexico City',
            'The capital of Mexico is Mexico City.',
        ]
    )


def test_run_sync_in_flow(allow_model_requests: None) -> None:
    """Test that run_sync works inside a Prefect flow."""

    @flow(name='test_run_sync_in_flow')
    def run_simple_agent_sync() -> str:
        result = simple_prefect_agent.run_sync('What is the capital of Mexico?')
        return result.output

    output = run_simple_agent_sync()
    assert output == snapshot('The capital of Mexico is Mexico City.')


async def test_run_stream_in_flow(allow_model_requests: None) -> None:
    """Test that run_stream errors when used inside a Prefect flow."""

    @flow(name='test_run_stream_in_flow')
    async def run_stream_workflow():
        async with simple_prefect_agent.run_stream('What is the capital of Mexico?') as result:
            return await result.get_output()  # pragma: no cover

    with flow_raises(
        UserError,
        snapshot(
            '`agent.run_stream()` cannot be used inside a Prefect flow. '
            'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
        ),
    ):
        await run_stream_workflow()


async def test_run_stream_events_in_flow(allow_model_requests: None) -> None:
    """Test that run_stream_events errors when used inside a Prefect flow."""

    @flow(name='test_run_stream_events_in_flow')
    async def run_stream_events_workflow():
        async with simple_prefect_agent.run_stream_events('What is the capital of Mexico?') as event_stream:
            return [event async for event in event_stream]

    with flow_raises(
        UserError,
        snapshot(
            '`agent.run_stream_events()` cannot be used inside a Prefect flow. '
            'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
        ),
    ):
        await run_stream_events_workflow()


async def test_iter_in_flow(allow_model_requests: None) -> None:
    """Test that iter works inside a Prefect flow."""

    @flow(name='test_iter_in_flow')
    async def run_iter_workflow():
        outputs: list[str] = []
        async with simple_prefect_agent.iter('What is the capital of Mexico?') as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for chunk in stream.stream_text(debounce_by=None):
                            outputs.append(chunk)
        return outputs

    outputs = await run_iter_workflow()
    # If called in a workflow, the output is a single concatenated string.
    assert outputs == snapshot(
        [
            'The capital of Mexico is Mexico City.',
        ]
    )


async def test_prefect_agent_run_with_model(allow_model_requests: None) -> None:
    """Test that passing model at runtime errors appropriately."""
    with flow_raises(
        UserError,
        snapshot(
            'Non-Prefect model cannot be set at agent run time inside a Prefect flow, it must be set at agent creation time.'
        ),
    ):
        await simple_prefect_agent.run('What is the capital of Mexico?', model=model)


async def test_prefect_agent_override_model() -> None:
    """Test that overriding model in a flow context errors."""

    @flow(name='test_override_model')
    async def override_model_flow():
        with simple_prefect_agent.override(model=model):
            pass

    with flow_raises(
        UserError,
        snapshot(
            'Non-Prefect model cannot be contextually overridden inside a Prefect flow, it must be set at agent creation time.'
        ),
    ):
        await override_model_flow()


async def test_prefect_agent_override_toolsets(allow_model_requests: None) -> None:
    """Test that overriding toolsets works."""

    @flow(name='test_override_toolsets')
    async def override_toolsets_flow():
        with simple_prefect_agent.override(toolsets=[FunctionToolset()]):
            result = await simple_prefect_agent.run('What is the capital of Mexico?')
            return result.output

    output = await override_toolsets_flow()
    assert output == snapshot('The capital of Mexico is Mexico City.')


async def test_prefect_agent_run_with_runtime_external_toolset() -> None:
    def request_external_tool(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('external', {'query': 'runtime'}, tool_call_id='call-1')])

    agent = Agent(
        FunctionModel(request_external_tool),
        name='runtime_external_toolset_prefect_agent',
        output_type=[str, DeferredToolRequests],
    )
    prefect_agent = PrefectAgent(agent)

    result = await prefect_agent.run(
        'Call the runtime external tool.',
        toolsets=[
            ExternalToolset(
                tool_defs=[
                    ToolDefinition(
                        name='external',
                        parameters_json_schema={
                            'type': 'object',
                            'properties': {'query': {'type': 'string'}},
                            'required': ['query'],
                        },
                    )
                ],
                id='external',
            )
        ],
    )

    assert result.output == DeferredToolRequests(
        calls=[ToolCallPart('external', {'query': 'runtime'}, tool_call_id='call-1')]
    )


@pytest.mark.parametrize('kind', ['function', 'mcp', 'dynamic'])
async def test_prefect_agent_run_rejects_executing_runtime_toolsets(kind: str) -> None:
    # Prefect wraps both function tools and MCP servers in tasks registered up front, and dynamic toolsets
    # can't be introspected ahead of time, so none of them can be added per-run.
    toolset_factories = {
        'function': lambda: FunctionToolset(),
        'mcp': lambda: MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='runtime_mcp'),
        'dynamic': lambda: DynamicToolset(lambda _: FunctionToolset(), id='runtime_dynamic'),
    }
    labels = {'function': 'FunctionToolset', 'mcp': 'MCPToolset', 'dynamic': 'DynamicToolset'}

    prefect_agent = PrefectAgent(Agent(TestModel(), name=f'reject_{kind}_prefect_agent'))
    with pytest.raises(UserError, match=f'{labels[kind]} cannot be passed to '):
        await prefect_agent.run('Hello', toolsets=[toolset_factories[kind]()])


async def test_prefect_agent_override_tools(allow_model_requests: None) -> None:
    """Test that overriding tools works."""

    @flow(name='test_override_tools')
    async def override_tools_flow():
        with simple_prefect_agent.override(tools=[get_weather]):
            result = await simple_prefect_agent.run('What is the capital of Mexico?')
            return result.output

    output = await override_tools_flow()
    assert output == snapshot('The capital of Mexico is Mexico City.')


async def test_prefect_agent_override_deps(allow_model_requests: None) -> None:
    """Test that overriding deps works."""

    @flow(name='test_override_deps')
    async def override_deps_flow():
        with simple_prefect_agent.override(deps=None):
            result = await simple_prefect_agent.run('What is the capital of Mexico?')
            return result.output

    output = await override_deps_flow()
    assert output == snapshot('The capital of Mexico is Mexico City.')


# Test human-in-the-loop with HITL tool
hitl_agent = Agent(
    model,
    name='hitl_agent',
    output_type=[str, DeferredToolRequests],
    instructions='Just call tools without asking for confirmation.',
)


@task(name='create_file')
@hitl_agent.tool
def create_file(ctx: RunContext, path: str) -> None:
    raise CallDeferred


@task(name='delete_file')
@hitl_agent.tool
def delete_file(ctx: RunContext, path: str) -> bool:
    if not ctx.tool_call_approved:
        raise ApprovalRequired
    return True


hitl_prefect_agent = PrefectAgent(hitl_agent)


async def test_prefect_agent_with_hitl_tool(allow_model_requests: None) -> None:
    """Test human-in-the-loop with deferred tool calls and approvals."""

    @flow(name='test_hitl_tool')
    async def hitl_main_loop(prompt: str) -> AgentRunResult[str | DeferredToolRequests]:
        messages: list[ModelMessage] = [ModelRequest.user_text_prompt(prompt)]
        deferred_tool_results: DeferredToolResults | None = None

        result = await hitl_prefect_agent.run(message_history=messages, deferred_tool_results=deferred_tool_results)
        messages = result.all_messages()

        if isinstance(result.output, DeferredToolRequests):  # pragma: no branch
            # Handle deferred requests
            results = DeferredToolResults()
            for tool_call in result.output.approvals:
                results.approvals[tool_call.tool_call_id] = True
            for tool_call in result.output.calls:
                results.calls[tool_call.tool_call_id] = 'Success'

            # Second run with results
            result = await hitl_prefect_agent.run(message_history=messages, deferred_tool_results=results)

        return result

    result = await hitl_main_loop('Delete the file `.env` and create `test.txt`')
    assert isinstance(result.output, str)
    assert 'deleted' in result.output.lower() or 'created' in result.output.lower()


def test_prefect_agent_with_hitl_tool_sync(allow_model_requests: None) -> None:
    """Test human-in-the-loop with sync version."""

    @flow(name='test_hitl_tool_sync')
    def hitl_main_loop_sync(prompt: str) -> AgentRunResult[str | DeferredToolRequests]:
        messages: list[ModelMessage] = [ModelRequest.user_text_prompt(prompt)]
        deferred_tool_results: DeferredToolResults | None = None

        result = hitl_prefect_agent.run_sync(message_history=messages, deferred_tool_results=deferred_tool_results)
        messages = result.all_messages()

        if isinstance(result.output, DeferredToolRequests):  # pragma: no branch
            results = DeferredToolResults()
            for tool_call in result.output.approvals:
                results.approvals[tool_call.tool_call_id] = True
            for tool_call in result.output.calls:
                results.calls[tool_call.tool_call_id] = 'Success'

            result = hitl_prefect_agent.run_sync(message_history=messages, deferred_tool_results=results)

        return result

    result = hitl_main_loop_sync('Delete the file `.env` and create `test.txt`')
    assert isinstance(result.output, str)


# Test model retry
model_retry_agent = Agent(model, name='model_retry_agent')


@task(name='get_weather_in_city')
@model_retry_agent.tool_plain
def get_weather_in_city(city: str) -> str:
    if city != 'Mexico City':
        raise ModelRetry('Did you mean Mexico City?')
    return 'sunny'


model_retry_prefect_agent = PrefectAgent(model_retry_agent)


async def test_prefect_agent_with_model_retry(allow_model_requests: None) -> None:
    """Test that ModelRetry works correctly."""
    result = await model_retry_prefect_agent.run('What is the weather in CDMX?')
    assert 'sunny' in result.output.lower() or 'mexico city' in result.output.lower()


# Test dynamic toolsets
@dataclass
class ToggleableDeps:
    active: Literal['weather', 'datetime']

    def toggle(self):
        if self.active == 'weather':
            self.active = 'datetime'
        else:
            self.active = 'weather'


@task(name='temperature_celsius')
def temperature_celsius(city: str) -> float:
    return 21.0


@task(name='temperature_fahrenheit')
def temperature_fahrenheit(city: str) -> float:
    return 69.8


@task(name='conditions')
def conditions(city: str) -> str:
    # Simplified version without RunContext
    return "It's raining"


weather_toolset = FunctionToolset(tools=[temperature_celsius, temperature_fahrenheit, conditions])

datetime_toolset = FunctionToolset()


@task(name='now')
def now_func() -> datetime:
    return datetime.now()


datetime_toolset.add_function(now_func, name='now')

test_model = TestModel()
dynamic_agent = Agent(name='dynamic_agent', model=test_model, deps_type=ToggleableDeps)


@dynamic_agent.toolset
def toggleable_toolset(ctx: RunContext[ToggleableDeps]) -> FunctionToolset:
    if ctx.deps.active == 'weather':
        return weather_toolset
    else:
        return datetime_toolset


@dynamic_agent.tool
def toggle(ctx: RunContext[ToggleableDeps]):
    ctx.deps.toggle()


dynamic_prefect_agent = PrefectAgent(dynamic_agent)


def test_dynamic_toolset():
    """Test that dynamic toolsets work correctly."""
    weather_deps = ToggleableDeps('weather')

    result = dynamic_prefect_agent.run_sync('Toggle the toolset', deps=weather_deps)
    assert isinstance(result.output, str)

    result = dynamic_prefect_agent.run_sync('Toggle the toolset', deps=weather_deps)
    assert isinstance(result.output, str)


# Test cache policies
async def test_cache_policy_default():
    """Test that the default cache policy is set correctly."""
    assert DEFAULT_PYDANTIC_AI_CACHE_POLICY is not None
    # It's a CompoundCachePolicy instance with policies attribute
    assert hasattr(DEFAULT_PYDANTIC_AI_CACHE_POLICY, 'policies')


async def test_cache_policy_custom():
    """
    Test that custom cache policy PrefectAgentInputs works.
    Timestamps must be excluded from computed cache keys to avoid
    duplicate calls when runs are restarted.
    """
    cache_policy = PrefectAgentInputs()

    # Create two sets of messages with same content but different timestamps
    time1 = datetime.now()
    time2 = time1 + timedelta(minutes=5)

    # First set of messages
    messages1 = [
        ModelRequest(
            parts=[UserPromptPart(content='What is the capital of France?', timestamp=time1)], timestamp=IsDatetime()
        ),
        ModelResponse(
            parts=[TextPart(content='The capital of France is Paris.')],
            usage=RequestUsage(input_tokens=10, output_tokens=10),
            model_name='test-model',
            timestamp=time1,
        ),
    ]

    # Second set of messages - same content, different timestamps
    messages2 = [
        ModelRequest(
            parts=[UserPromptPart(content='What is the capital of France?', timestamp=time2)], timestamp=IsDatetime()
        ),
        ModelResponse(
            parts=[TextPart(content='The capital of France is Paris.')],
            usage=RequestUsage(input_tokens=10, output_tokens=10),
            model_name='test-model',
            timestamp=time2,
        ),
    ]

    mock_task_ctx = MagicMock()

    # Compute hashes using the cache policy
    hash1 = cache_policy.compute_key(
        task_ctx=mock_task_ctx,
        inputs={'messages': messages1},
        flow_parameters={},
    )

    hash2 = cache_policy.compute_key(
        task_ctx=mock_task_ctx,
        inputs={'messages': messages2},
        flow_parameters={},
    )

    # The hashes should be the same since timestamps are excluded
    assert hash1 == hash2

    # Also test that different content produces different hashes
    messages3 = [
        ModelRequest(
            parts=[UserPromptPart(content='What is the capital of Spain?', timestamp=time1)], timestamp=IsDatetime()
        ),
        ModelResponse(
            parts=[TextPart(content='The capital of Spain is Madrid.')],
            usage=RequestUsage(input_tokens=10, output_tokens=10),
            model_name='test-model',
            timestamp=time1,
        ),
    ]

    hash3 = cache_policy.compute_key(
        task_ctx=mock_task_ctx,
        inputs={'messages': messages3},
        flow_parameters={},
    )

    # This hash should be different from the others
    assert hash3 != hash1


async def test_cache_policy_with_tuples():
    """Test that cache policy handles tuples with timestamps correctly."""
    cache_policy = PrefectAgentInputs()
    mock_task_ctx = MagicMock()

    time1 = datetime.now()
    time2 = time1 + timedelta(minutes=5)

    time3 = time2 + timedelta(minutes=5)
    time4 = time3 + timedelta(minutes=5)

    # Create a tuple with timestamps
    data_with_tuple_1 = (
        UserPromptPart(content='Question', timestamp=time1),
        TextPart(content='Answer'),
        UserPromptPart(content='Follow-up', timestamp=time2),
    )

    data_with_tuple_2 = (
        UserPromptPart(content='Question', timestamp=time3),
        TextPart(content='Answer'),
        UserPromptPart(content='Follow-up', timestamp=time4),
    )

    assert cache_policy.compute_key(
        task_ctx=mock_task_ctx,
        inputs={'messages': data_with_tuple_1},
        flow_parameters={},
    ) == cache_policy.compute_key(
        task_ctx=mock_task_ctx,
        inputs={'messages': data_with_tuple_2},
        flow_parameters={},
    )


async def test_cache_policy_empty_inputs():
    """Test that cache policy returns None for empty inputs."""
    cache_policy = PrefectAgentInputs()

    mock_task_ctx = MagicMock()

    # Test with empty inputs
    result = cache_policy.compute_key(
        task_ctx=mock_task_ctx,
        inputs={},
        flow_parameters={},
    )

    assert result is None


def test_cache_key_run_context_projection_is_exhaustive():
    """Every `RunContext` field must be consciously categorized for Prefect cache-key hashing.

    A task's cache key is derived from a hashable projection of `RunContext` (see
    `_replace_run_context`). A field that affects a step's behavior but is omitted from the
    projection causes cache collisions: two runs differing only in that field share a key and
    one replays the other's result. This test fails when a `RunContext` field is added until
    it's either included in the projection or listed in `cache_irrelevant` with a reason — the
    same drift that left `loaded_capability_ids`/`discovered_tool_names` out of the key.
    """
    # Fields that legitimately don't belong in the cache key, each with its reason.
    cache_irrelevant = {
        'deps',  # user object; hashed separately as a task input, not via RunContext
        'agent',  # the agent instance, identified by task source not run state
        'model',  # live Model instance, not hashable run state
        'usage',  # accumulates per run; not an input that should fork the cache
        'tracer',  # tracing plumbing, not run state
        'tool_manager',  # live ToolManager, not hashable run state
        'capabilities',  # live capability objects, not hashable run state
        'pending_messages',  # live run queue, not hashable run state
        'messages',  # hashed as the separate `messages` task input
        'prompt',  # hashed as the separate prompt task input
        'validation_context',  # arbitrary user object, not run state
        'trace_include_content',  # tracing config, not run state
        'instrumentation_version',  # tracing config, not run state
        'partial_output',  # output-validator flag, not a tool-execution input
        'run_id',  # per-run id; deliberately excluded so keys are stable across runs
        'conversation_id',  # per-conversation id; same rationale as run_id
        'metadata',  # free-form run metadata, not a tool-execution input
        'model_settings',  # hashed via the model request inputs, not RunContext
        'capability_loaded',  # transient per-hook flag; `None` during tool execution
    }
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    projected = set(_replace_run_context({'ctx': ctx})['ctx'])
    all_fields = set(RunContext.__dataclass_fields__)

    overlap = projected & cache_irrelevant
    assert not overlap, f'Fields both projected and marked irrelevant: {overlap}'

    uncategorized = all_fields - (projected | cache_irrelevant)
    assert not uncategorized, (
        f'Uncategorized `RunContext` fields: {uncategorized}. Add each to the `_replace_run_context` '
        'projection (if it should fork the cache key) or to `cache_irrelevant` (with a reason).'
    )


async def test_repeated_run_hits_cache():
    """Same prompt across two separate flow runs must only call the model once.

    `PrefectAgent.run()` wraps each call in its own Prefect flow, so a cross-flow
    cache hit requires the Model Request task's cache key to be stable across flow
    runs. This is a field-agnostic regression guard: any per-run field that leaks
    into the hashed inputs (today `run_id`/`timestamp`, or anything added to
    `ModelMessage` in the future) will make the two keys differ, miss the cache,
    and fail this test with `call_count == 2`. The UUID in the prompt keeps the
    test isolated from any other run in the session-scoped Prefect test harness.
    """
    call_count = 0

    def counting_model(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        return ModelResponse(parts=[TextPart('4')])

    prefect_agent = PrefectAgent(
        Agent(FunctionModel(counting_model), name='cache_test_agent'),
        model_task_config=TaskConfig(cache_policy=PrefectAgentInputs()),
    )

    prompt = f'What is 2+2? {uuid.uuid4()}'
    await prefect_agent.run(prompt)
    await prefect_agent.run(prompt)
    assert call_count == 1


# Test custom model settings
class CustomModelSettings(ModelSettings, total=False):
    custom_setting: str


def return_settings(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(str(agent_info.model_settings))])


model_settings = CustomModelSettings(max_tokens=123, custom_setting='custom_value')
function_model = FunctionModel(return_settings, settings=model_settings)

settings_agent = Agent(function_model, name='settings_agent')
settings_prefect_agent = PrefectAgent(settings_agent)


async def test_custom_model_settings(allow_model_requests: None):
    """Test that custom model settings are passed through correctly."""
    result = await settings_prefect_agent.run('Give me those settings')
    assert result.output == snapshot("{'max_tokens': 123, 'custom_setting': 'custom_value'}")


@dataclass
class SimpleDeps:
    value: str


async def test_tool_call_outside_flow():
    """Test that tools work when called outside a Prefect flow."""

    # Create an agent with a simple tool
    test_agent = Agent(TestModel(), deps_type=SimpleDeps, name='test_outside_flow')

    @test_agent.tool
    def simple_tool(ctx: RunContext[SimpleDeps]) -> str:
        return f'Tool called with: {ctx.deps.value}'

    test_prefect_agent = PrefectAgent(test_agent)

    # Call run() outside a flow - tools should still work
    result = await test_prefect_agent.run('Call the tool', deps=SimpleDeps(value='test'))
    # Check that the tool was actually called by looking at the messages
    messages = result.all_messages()
    assert any('simple_tool' in str(msg) for msg in messages)


async def test_disabled_tool():
    """Test that tools can be disabled via tool_task_config_by_name."""

    # Create an agent with a tool
    test_agent = Agent(TestModel(), name='test_disabled_tool')

    @test_agent.tool_plain
    def my_tool() -> str:
        return 'Tool executed'

    # Create PrefectAgent with the tool disabled
    test_prefect_agent = PrefectAgent(
        test_agent,
        tool_task_config_by_name={
            'my_tool': None,
        },
    )

    # Test outside a flow
    result = await test_prefect_agent.run('Call my_tool')
    messages = result.all_messages()
    assert any('my_tool' in str(msg) for msg in messages)

    # Test inside a flow to ensure disabled tools work there too
    @flow
    async def test_flow():
        result = await test_prefect_agent.run('Call my_tool')
        return result

    flow_result = await test_flow()
    flow_messages = flow_result.all_messages()
    assert any('my_tool' in str(msg) for msg in flow_messages)


# ==========================================
# PrefectDurability capability tests
# ==========================================


def _durability_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Simple model function for durability tests."""
    for msg in reversed(messages):
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                return ModelResponse(parts=[TextPart(content=f'Echo: {part.content}')])
    return ModelResponse(parts=[TextPart(content='no prompt')])  # pragma: no cover


_durability_fn_model = FunctionModel(_durability_model_fn)


async def test_prefect_durability_simple_agent(prefect_harness: None) -> None:
    """PrefectDurability routes model requests through Prefect tasks."""
    agent = Agent(_durability_fn_model, name='durability_simple', capabilities=[PrefectDurability()])

    @flow
    async def run_durable_agent() -> str:
        result = await agent.run('Hello Prefect')
        return result.output

    output = await run_durable_agent()
    assert output == 'Echo: Hello Prefect'


async def test_prefect_durability_auto_wraps_run_as_flow(prefect_harness: None) -> None:
    """`agent.run` outside an active flow auto-wraps in one.

    The capability hooks `agent.run` so callers don't need a manual `@flow`. Inside an
    existing flow the wrapper short-circuits to avoid creating a redundant nested flow.
    """
    agent = Agent(_durability_fn_model, name='durability_auto', capabilities=[PrefectDurability()])

    result = await agent.run('Auto-wrapped')
    assert result.output == 'Echo: Auto-wrapped'


def test_prefect_durability_idempotent_for_agent() -> None:
    """Binding a second `PrefectDurability` to the same agent doesn't re-wrap `agent.run`.

    Without the idempotency guard, re-binding would stack flow decorators on top
    of each other.
    """
    agent = Agent(_durability_fn_model, name='prefect_idempotent_test', capabilities=[PrefectDurability()])
    first_run = agent.run

    # Re-binding another PrefectDurability should leave agent.run untouched.
    PrefectDurability().for_agent(agent)
    assert agent.run is first_run


@pytest.mark.parametrize('kind', ['function', 'mcp', 'dynamic'])
async def test_prefect_durability_rejects_executing_runtime_toolsets(kind: str) -> None:
    """Capability-path equivalent of `test_prefect_agent_run_rejects_executing_runtime_toolsets`."""
    toolset_factories = {
        'function': lambda: FunctionToolset(),
        'mcp': lambda: MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='runtime_mcp'),
        'dynamic': lambda: DynamicToolset(lambda _: FunctionToolset(), id='runtime_dynamic'),
    }
    labels = {'function': 'FunctionToolset', 'mcp': 'MCPToolset', 'dynamic': 'DynamicToolset'}

    agent = Agent(TestModel(), name=f'durability_reject_{kind}', capabilities=[PrefectDurability()])
    with pytest.raises(UserError, match=f'{labels[kind]} cannot be passed to '):
        await agent.run('Hello', toolsets=[toolset_factories[kind]()])
