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
from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    AgentRunResult,
    AgentRunResultEvent,
    AgentStreamEvent,
    ExternalToolset,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    FunctionToolset,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RunContext,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import (
    MCP,
    Capability,
    DynamicCapability,
    Instrumentation,
    ProcessEventStream,
    ResolveModelId,
    Toolset,
)
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UsageLimitExceeded, UserError
from pydantic_ai.models import ModelRequestParameters, ModelResolutionContext, create_async_http_client
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

try:
    from prefect import flow, task
    from prefect.context import TaskRunContext
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
from .conftest import IsDatetime, IsSameStr, IsStr
from .continuation_utils import ScriptedContinuationModel, StreamSegment, scripted_response

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
                                BasicSpan(content='tools/list'),
                                BasicSpan(
                                    content='chat gpt-4o',
                                    children=[
                                        BasicSpan(
                                            content=IsStr(regex=r'Model Request \(Streaming\): gpt-4o-\w+'),
                                            children=[
                                                BasicSpan(content='ctx.run_step=1'),
                                                BasicSpan(
                                                    content='{"index":0,"part":{"tool_name":"get_country","args":"","tool_call_id":"call_rI3WKPYvVwlOgCGRjsPP2hEx","tool_kind":null,"id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":null,"event_kind":"part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{}","tool_call_id":"call_rI3WKPYvVwlOgCGRjsPP2hEx","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"part":{"tool_name":"get_country","args":"{}","tool_call_id":"call_rI3WKPYvVwlOgCGRjsPP2hEx","tool_kind":null,"id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":null,"event_kind":"part_end"}'
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
                                            content='{"part":{"tool_name":"get_country","args":"{}","tool_call_id":"call_rI3WKPYvVwlOgCGRjsPP2hEx","tool_kind":null,"id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"args_valid":true,"event_kind":"function_tool_call"}'
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
                                                regex=r'\{"part":\{"tool_name":"get_country","content":"Mexico","tool_call_id":"call_rI3WKPYvVwlOgCGRjsPP2hEx","tool_kind":null,"metadata":null,"timestamp":"[^"]+","outcome":"success","part_kind":"tool-return"\},"content":null,"event_kind":"function_tool_result"\}'
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
                                                    content='{"index":0,"part":{"tool_name":"get_weather","args":"","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","tool_kind":null,"id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":null,"event_kind":"part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"ci","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"ty\\": ","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"Mexic","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"o Ci","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"ty\\"}","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"part":{"tool_name":"get_weather","args":"{\\"city\\": \\"Mexico City\\"}","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","tool_kind":null,"id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":"tool-call","event_kind":"part_end"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":1,"part":{"tool_name":"get_product_name","args":"","tool_call_id":"call_SkGkkGDvHQEEk0CGbnAh2AQw","tool_kind":null,"id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":"tool-call","event_kind":"part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":1,"delta":{"tool_name_delta":null,"args_delta":"{}","tool_call_id":"call_SkGkkGDvHQEEk0CGbnAh2AQw","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":1,"part":{"tool_name":"get_product_name","args":"{}","tool_call_id":"call_SkGkkGDvHQEEk0CGbnAh2AQw","tool_kind":null,"id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":null,"event_kind":"part_end"}'
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
                                            content='{"part":{"tool_name":"get_weather","args":"{\\"city\\": \\"Mexico City\\"}","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","tool_kind":null,"id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"args_valid":true,"event_kind":"function_tool_call"}'
                                        ),
                                    ],
                                ),
                                BasicSpan(
                                    content=IsStr(regex=r'Handle Stream Event-\w+'),
                                    children=[
                                        BasicSpan(content='ctx.run_step=2'),
                                        BasicSpan(
                                            content='{"part":{"tool_name":"get_product_name","args":"{}","tool_call_id":"call_SkGkkGDvHQEEk0CGbnAh2AQw","tool_kind":null,"id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"args_valid":true,"event_kind":"function_tool_call"}'
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
                                                regex=r'\{"part":\{"tool_name":"get_weather","content":"sunny","tool_call_id":"call_NS4iQj14cDFwc0BnrKqDHavt","tool_kind":null,"metadata":null,"timestamp":"[^"]+","outcome":"success","part_kind":"tool-return"\},"content":null,"event_kind":"function_tool_result"\}'
                                            )
                                        ),
                                    ],
                                ),
                                BasicSpan(
                                    content='running tool: get_product_name',
                                    children=[
                                        BasicSpan(
                                            content=IsStr(regex=r'Call MCP Tool: get_product_name-\w+'),
                                            children=[BasicSpan(content='tools/call get_product_name')],
                                        )
                                    ],
                                ),
                                BasicSpan(
                                    content=IsStr(regex=r'Handle Stream Event-\w+'),
                                    children=[
                                        BasicSpan(content='ctx.run_step=2'),
                                        BasicSpan(
                                            content=IsStr(
                                                regex=r'\{"part":\{"tool_name":"get_product_name","content":"Pydantic AI","tool_call_id":"call_SkGkkGDvHQEEk0CGbnAh2AQw","tool_kind":null,"metadata":null,"timestamp":"[^"]+","outcome":"success","part_kind":"tool-return"\},"content":null,"event_kind":"function_tool_result"\}'
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
                                                    content='{"index":0,"part":{"tool_name":"final_result","args":"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","tool_kind":null,"id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":null,"event_kind":"part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"tool_name":"final_result","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","event_kind":"final_result"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answers","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":[","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Capital","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" of","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" the","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" country","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Mexico","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" City","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"},{\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Weather","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" in","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" the","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" capital","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Sunny","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"},{\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Product","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" name","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"P","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"yd","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"antic","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" AI","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"}","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"]}","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"part":{"tool_name":"final_result","args":"{\\"answers\\":[{\\"label\\":\\"Capital of the country\\",\\"answer\\":\\"Mexico City\\"},{\\"label\\":\\"Weather in the capital\\",\\"answer\\":\\"Sunny\\"},{\\"label\\":\\"Product name\\",\\"answer\\":\\"Pydantic AI\\"}]}","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","tool_kind":null,"id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":null,"event_kind":"part_end"}'
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                                BasicSpan(
                                    content=IsStr(regex=r'Handle Stream Event-\w+'),
                                    children=[
                                        BasicSpan(content='ctx.run_step=3'),
                                        BasicSpan(
                                            content='{"part":{"tool_name":"final_result","args":"{\\"answers\\":[{\\"label\\":\\"Capital of the country\\",\\"answer\\":\\"Mexico City\\"},{\\"label\\":\\"Weather in the capital\\",\\"answer\\":\\"Sunny\\"},{\\"label\\":\\"Product name\\",\\"answer\\":\\"Pydantic AI\\"}]}","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","tool_kind":null,"id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"args_valid":true,"event_kind":"output_tool_call"}'
                                        ),
                                    ],
                                ),
                                BasicSpan(
                                    content=IsStr(regex=r'Handle Stream Event-\w+'),
                                    children=[
                                        BasicSpan(content='ctx.run_step=3'),
                                        BasicSpan(
                                            content=IsStr(
                                                regex=r'\{"part":\{"tool_name":"final_result","content":"Final result processed\.","tool_call_id":"call_QcKhHXwXzqOXJUUHJb1TB2V5","tool_kind":null,"metadata":null,"timestamp":"[^"]+","outcome":"success","part_kind":"tool-return"\},"event_kind":"output_tool_result"\}'
                                            )
                                        ),
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


async def test_event_stream_handler_property_outside_flow() -> None:
    # Outside a Prefect flow, the `event_stream_handler` property resolves to the effective handler
    # directly, rather than the in-flow per-event dispatcher.
    agent = Agent(TestModel(), name='event_stream_handler_property_agent')
    prefect_agent = PrefectAgent(agent, event_stream_handler=runtime_event_stream_handler)
    assert prefect_agent.event_stream_handler is runtime_event_stream_handler


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


async def test_capability_contributed_toolset_id_from_capability():
    """A capability's `id` flows to its contributed leaf toolset, so a capability combined with a
    local MCP server is swapped for its Prefect task wrapper under a stable id. An `MCP` with no
    explicit `id` derives one from its URL.

    This isn't a VCR test: it inspects the constructed toolset tree during local agent construction,
    before any model or MCP request, so there's no network round-trip to record.

    Regression for https://github.com/pydantic/pydantic-ai/issues/6334.
    """

    def add(x: int) -> int:
        return x + 1  # pragma: no cover

    agent = Agent(
        model,
        name='capability_agent',
        capabilities=[
            Capability(id='billing', tools=[add]),
            MCP(url='https://mcp.example.com/api'),
        ],
    )
    prefect_agent = PrefectAgent(agent)

    leaves: list[AbstractToolset[object]] = []
    for toolset in prefect_agent.toolsets:
        toolset.apply(leaves.append)
    # The contributed MCP leaf carries the URL-derived id, so its `PrefectMCPToolset` wrapper is built
    # under a stable id; the `billing` function toolset carries the capability id.
    assert any(isinstance(ts, MCPToolset) and ts.id == 'mcp.example.com-api' for ts in leaves)
    assert any(isinstance(ts, FunctionToolset) and ts.id == 'billing' for ts in leaves)


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
        [
            PartStartEvent(index=0, part=TextPart(content='The')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' capital')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' Mexico')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' Mexico')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' City')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='.')),
            PartEndEvent(index=0, part=TextPart(content='The capital of Mexico is Mexico City.')),
            AgentRunResultEvent(result=AgentRunResult(output='The capital of Mexico is Mexico City.')),
        ]
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
            return [event async for event in event_stream]  # pragma: no cover

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


async def test_prefect_cancel_suspended_response_runs_in_task(allow_model_requests: None) -> None:
    """`PrefectModel.cancel_suspended_response` must run inside a Prefect task, not inline in the flow.

    The provider teardown that cancels a server-side suspended/background job is a raw HTTP call;
    wrapping it as a task makes it durable and retried. We assert a `TaskRunContext` is active when
    the wrapped model's cancel runs, proving it executed inside a task rather than inline.
    """
    ran_in_task: list[bool] = []

    class RecordingModel(TestModel):
        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            ran_in_task.append(TaskRunContext.get() is not None)

    prefect_model = PrefectModel(
        RecordingModel(),
        task_config=TaskConfig(),
        get_event_stream_handler=lambda: None,
    )
    response = ModelResponse(parts=[TextPart('paused')], state='suspended')

    @flow(name='test_cancel_suspended_response')
    async def cancel_in_flow() -> None:
        await prefect_model.cancel_suspended_response(response)

    await cancel_in_flow()
    assert ran_in_task == [True]


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


async def test_cache_policy_per_run_ids_excluded_but_dict_keys_kept():
    """Per-run message fields must not fork the cache key, but identically named plain dict keys must.

    `ModelRequest`/`ModelResponse` grow a fresh `run_id`/`conversation_id` per run, so two runs with
    identical content would never share a cache entry if those fields were hashed. Plain dict keys
    with the same names are a different story: they are user or provider data (tool args,
    `provider_details['conversation_id']` used for OpenAI server-side continuation) where the value
    is meaningful and must fork the key. Unit test because key stability across separately
    constructed inputs can't be observed through a recorded run.
    """
    cache_policy = PrefectAgentInputs()
    mock_task_ctx = MagicMock()

    def messages_with_ids(run_id: str, conversation_id: str) -> list[ModelMessage]:
        return [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2+2?')],
                timestamp=IsDatetime(),
                run_id=run_id,
                conversation_id=conversation_id,
            ),
            ModelResponse(
                parts=[TextPart(content='4')],
                usage=RequestUsage(input_tokens=10, output_tokens=10),
                model_name='test-model',
                run_id=run_id,
                conversation_id=conversation_id,
            ),
        ]

    hash1 = cache_policy.compute_key(
        task_ctx=mock_task_ctx, inputs={'messages': messages_with_ids('run-1', 'conv-1')}, flow_parameters={}
    )
    hash2 = cache_policy.compute_key(
        task_ctx=mock_task_ctx, inputs={'messages': messages_with_ids('run-2', 'conv-2')}, flow_parameters={}
    )
    assert hash1 == hash2

    def messages_with_provider_details(conversation_id: str) -> list[ModelMessage]:
        return [
            ModelResponse(
                parts=[TextPart(content='4')],
                usage=RequestUsage(input_tokens=10, output_tokens=10),
                model_name='test-model',
                provider_details={'conversation_id': conversation_id},
            ),
        ]

    provider_hash1 = cache_policy.compute_key(
        task_ctx=mock_task_ctx, inputs={'messages': messages_with_provider_details('conv-a')}, flow_parameters={}
    )
    provider_hash2 = cache_policy.compute_key(
        task_ctx=mock_task_ctx, inputs={'messages': messages_with_provider_details('conv-b')}, flow_parameters={}
    )
    assert provider_hash1 != provider_hash2

    tool_args_hash1 = cache_policy.compute_key(
        task_ctx=mock_task_ctx, inputs={'tool_args': {'conversation_id': 'conv-a'}}, flow_parameters={}
    )
    tool_args_hash2 = cache_policy.compute_key(
        task_ctx=mock_task_ctx, inputs={'tool_args': {'conversation_id': 'conv-b'}}, flow_parameters={}
    )
    assert tool_args_hash1 != tool_args_hash2


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
        'root_capability',  # live capability tree (static config); run-varying loaded state is projected via loaded_capability_ids/discovered_tool_names
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
        '_mcp_tool_defs_cache',  # live per-run memo of MCP tool defs, reconstructed from messages
        '_event_stream_buffer',  # live per-run event buffer drained in workflow code, not a tool-execution input
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
    result1 = await prefect_agent.run(prompt)
    result2 = await prefect_agent.run(prompt)
    assert call_count == 1

    # A replayed response must keep the run/conversation that produced it. If the cached payload
    # were re-stamped with the replaying run's IDs, provider server-side state guards (e.g. OpenAI
    # `openai_conversation_id='auto'`) would treat another conversation's response as their own and
    # continue its provider-side conversation.
    response1, response2 = result1.all_messages()[-1], result2.all_messages()[-1]
    assert [response1.run_id, response1.conversation_id, response2.run_id, response2.conversation_id] == [
        (producing_run_id := IsSameStr()),
        (producing_conversation_id := IsSameStr()),
        producing_run_id,
        producing_conversation_id,
    ]
    # The replay belongs to a different run: run 2's own request carries its own fresh `run_id`.
    request2 = result2.all_messages()[0]
    assert request2.run_id == IsStr()
    assert request2.run_id != response2.run_id


async def test_durability_repeated_run_hits_cache_preserves_provenance():
    """The capability path stamps cached responses with their producing run and conversation."""
    call_count = 0

    def counting_model(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        return ModelResponse(parts=[TextPart('4')])

    agent = Agent(
        FunctionModel(counting_model),
        name='durability_cache_test_agent',
        capabilities=[PrefectDurability(model_task_config=TaskConfig(cache_policy=PrefectAgentInputs()))],
    )

    @flow
    async def run_agent(prompt: str) -> AgentRunResult[str]:
        return await agent.run(prompt)

    prompt = f'What is 2+2? {uuid.uuid4()}'
    result1 = await run_agent(prompt)
    result2 = await run_agent(prompt)
    assert call_count == 1
    response1, response2 = result1.all_messages()[-1], result2.all_messages()[-1]
    assert [response1.run_id, response1.conversation_id, response2.run_id, response2.conversation_id] == [
        (producing_run_id := IsSameStr()),
        (producing_conversation_id := IsSameStr()),
        producing_run_id,
        producing_conversation_id,
    ]
    assert result2.all_messages()[0].run_id != response2.run_id


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
    for msg in reversed(messages):  # pragma: no branch - first message carries the prompt
        for part in msg.parts:  # pragma: no branch - first part is the UserPromptPart
            if isinstance(part, UserPromptPart):  # pragma: no branch - same reason
                return ModelResponse(parts=[TextPart(content=f'Echo: {part.content}')])
    return ModelResponse(parts=[TextPart(content='no prompt')])  # pragma: no cover


_durability_fn_model = FunctionModel(_durability_model_fn)


async def test_prefect_durability_simple_agent() -> None:
    """PrefectDurability routes model requests through Prefect tasks."""
    agent = Agent(_durability_fn_model, name='durability_simple', capabilities=[PrefectDurability()])

    @flow
    async def run_durable_agent() -> str:
        result = await agent.run('Hello Prefect')
        return result.output

    output = await run_durable_agent()
    assert output == 'Echo: Hello Prefect'


def test_resolve_tool_task_config_reads_metadata() -> None:
    """Per-tool Prefect config from `tool_def.metadata['prefect']` takes priority over the by-name dict."""
    from pydantic_ai.durable_exec.prefect._toolset import resolve_tool_task_config
    from pydantic_ai.tools import ToolDefinition
    from pydantic_ai.toolsets import ToolsetTool

    metadata_config = TaskConfig(timeout_seconds=120.0)

    fn_toolset = FunctionToolset[None](id='resolve_meta_toolset')

    def fn_tool() -> str:
        return 'ok'  # pragma: no cover - registered with toolset; test only resolves metadata

    fn_toolset.add_function(fn_tool, metadata={'prefect': metadata_config})
    tool_def = ToolDefinition(name='fn_tool', metadata={'prefect': metadata_config})
    tool = ToolsetTool[None](
        toolset=fn_toolset,
        tool_def=tool_def,
        max_retries=0,
        args_validator=None,  # pyright: ignore[reportArgumentType]
    )

    # Metadata wins over the per-tool (deprecated `PrefectAgent`) dict.
    resolved = resolve_tool_task_config(tool, 'fn_tool', {'fn_tool': TaskConfig(timeout_seconds=1.0)})
    assert resolved is metadata_config

    # `False` in metadata disables task wrapping.
    tool.tool_def.metadata = {'prefect': False}
    assert resolve_tool_task_config(tool, 'fn_tool', {}) is False

    # No metadata: an explicit `None` in the fallback dict disables wrapping, a missing key uses the base config.
    tool.tool_def.metadata = None
    assert resolve_tool_task_config(tool, 'fn_tool', {'fn_tool': None}) is False
    assert resolve_tool_task_config(tool, 'fn_tool', {}) == {}

    # Metadata present but without a `'prefect'` key: falls through to the by-name fallback.
    tool.tool_def.metadata = {'other': 'x'}
    assert resolve_tool_task_config(tool, 'fn_tool', {'fn_tool': None}) is False
    assert resolve_tool_task_config(tool, 'fn_tool', {}) == {}

    # Invalid metadata (e.g. a string from a misuse like `metadata={'prefect': '5s'}`)
    # raises `UserError` instead of silently passing the wrong shape to Prefect.
    tool.tool_def.metadata = {'prefect': '5s'}
    with pytest.raises(UserError, match=r"Tool 'fn_tool' has invalid 'prefect' metadata"):
        resolve_tool_task_config(tool, 'fn_tool', {})


@pytest.mark.parametrize('kind', ['function', 'mcp'])
def test_prefect_durability_rejects_idless_toolsets(kind: str) -> None:
    """Wrapped leaf toolsets without an `id` fail loudly at construction.

    The Prefect task wrapper is swapped in by toolset ID at run time, so without one the
    toolset's calls would silently run untracked inside the Prefect flow and re-execute
    on retries. Temporal raises the equivalent error for id-less leaves.
    """

    def greet() -> str:
        return 'hi'  # pragma: no cover

    toolset_factories = {
        'function': lambda: FunctionToolset([greet]),
        'mcp': lambda: MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server'])),
    }
    with pytest.raises(UserError, match='need to have a unique `id` in order to be used with Prefect'):
        Agent(
            _durability_fn_model,
            name=f'prefect_idless_{kind}',
            toolsets=[toolset_factories[kind]()],
            capabilities=[PrefectDurability()],
        )


def test_prefect_durability_wraps_capability_contributed_toolsets() -> None:
    """Toolsets contributed by other capabilities are wrapped as Prefect tasks too.

    Durability capabilities are in the `innermost` ordering tier, so `Agent.__init__` binds
    them only after every other capability's contributed toolsets have been extracted into
    `agent.toolsets`. Without that two-phase binding, this toolset would be invisible to
    `for_agent` and its tools would run untracked inside the Prefect flow.
    """

    def greet() -> str:
        return 'hi'  # pragma: no cover

    agent = Agent(
        _durability_fn_model,
        name='prefect_cap_toolset',
        capabilities=[Toolset(FunctionToolset([greet], id='cap_tools')), PrefectDurability()],
    )
    bound = PrefectDurability.from_agent(agent)
    assert bound is not None
    assert 'cap_tools' in bound._toolsets_by_id  # pyright: ignore[reportPrivateUsage]


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

    @flow
    async def run_agent() -> None:
        await agent.run('Hello', toolsets=[toolset_factories[kind]()])

    with pytest.raises(UserError, match=f'{labels[kind]} cannot be passed to '):
        await run_agent()


async def test_prefect_durability_allows_fully_opted_out_runtime_function_toolset() -> None:
    def model(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('runtime_tool', {}, tool_call_id='call-1')])

    async def runtime_tool() -> str:
        return 'tool-result'

    toolset = FunctionToolset(id='runtime')
    toolset.add_function(runtime_tool, metadata={'prefect': False})
    agent = Agent(FunctionModel(model), name='runtime_opt_out', capabilities=[PrefectDurability()])

    @flow
    async def run_agent() -> str:
        return (await agent.run('Hello', toolsets=[toolset])).output

    assert await run_agent() == 'done'


async def test_prefect_durability_rejects_partially_opted_out_runtime_function_toolset() -> None:
    async def opted_out() -> str:  # pragma: no cover — rejected before any tool runs
        return 'ok'

    async def wrapped() -> str:  # pragma: no cover — rejected before any tool runs
        return 'no'

    toolset = FunctionToolset(id='runtime')
    toolset.add_function(opted_out, metadata={'prefect': False})
    toolset.add_function(wrapped)
    agent = Agent(TestModel(), name='runtime_partial_opt_out', capabilities=[PrefectDurability()])

    @flow
    async def run_agent() -> None:
        await agent.run('Hello', toolsets=[toolset])

    with pytest.raises(UserError, match='FunctionToolset cannot be passed'):
        await run_agent()


async def test_prefect_durability_rejects_runtime_toolset_in_iter() -> None:
    """`agent.iter(toolsets=...)` inside a user flow is guarded like `run(toolsets=...)`.

    The rejection lives in run setup (`get_wrapper_toolset`), which every entry point routes
    through so `iter` inside a flow cannot execute the toolset's tools un-tasked.
    """
    agent = Agent(TestModel(), name='durability_reject_iter', capabilities=[PrefectDurability()])

    @flow
    async def run_agent() -> None:
        async with agent.iter('Hello', toolsets=[FunctionToolset(id='iter_fn')]):
            pass  # pragma: no cover — run setup raises before any node runs

    with pytest.raises(UserError, match='FunctionToolset cannot be passed to '):
        await run_agent()


async def test_prefect_durability_rejects_per_run_capability_toolset() -> None:
    """A toolset contributed by a per-run capability is rejected like `run(toolsets=...)`.

    Construction-time capability toolsets are wrapped by `for_agent` (see the
    capability-contributed test above); a per-run capability's toolset arrives after that
    wrapping has happened, so its tools would run un-tasked inside the flow.
    """
    agent = Agent(TestModel(), name='durability_reject_per_run_cap', capabilities=[PrefectDurability()])

    @flow
    async def run_agent() -> None:
        await agent.run('Hello', capabilities=[Toolset(FunctionToolset(id='per_run_fn'))])

    with pytest.raises(UserError, match='FunctionToolset cannot be passed to '):
        await run_agent()


def test_prefect_durability_rejects_duplicate_toolset_id() -> None:
    """Two distinct toolsets under one `id` are rejected at binding time.

    The registry maps `id` → task wrapper, so a duplicate would silently replace the first
    entry and route both toolsets' calls through the last one's tasks.
    """
    with pytest.raises(UserError, match="Two toolsets have the same `id` 'dup'"):
        Agent(
            _durability_fn_model,
            name='durability_dup_toolset',
            toolsets=[FunctionToolset(id='dup'), FunctionToolset(id='dup')],
            capabilities=[PrefectDurability()],
        )


def test_prefect_durability_same_toolset_instance_reused() -> None:
    """The same toolset instance appearing twice maps to one wrapper, not an `id` conflict."""
    toolset = FunctionToolset(id='shared_fn')
    agent = Agent(
        _durability_fn_model,
        name='durability_shared_toolset',
        toolsets=[toolset, toolset],
        capabilities=[PrefectDurability()],
    )
    bound = PrefectDurability.from_agent(agent)
    assert bound is not None
    assert sorted(bound._toolsets_by_id) == ['<agent>', 'shared_fn']  # pyright: ignore[reportPrivateUsage]


async def test_prefect_durability_outside_flow() -> None:
    """PrefectDurability is transparent outside a Prefect flow."""
    agent = Agent(_durability_fn_model, name='durability_outside', capabilities=[PrefectDurability()])

    result = await agent.run('Hello outside')
    assert result.output == 'Echo: Hello outside'


async def test_prefect_durability_dynamic_capability_tool_runs_as_task() -> None:
    """A dynamic capability's tool calls run as Prefect tasks."""
    calls: list[str] = []
    task_run_names: list[str] = []

    def dynamic_tool() -> str:
        calls.append('called')
        task_run_context = TaskRunContext.get()
        assert task_run_context is not None
        task_run_names.append(task_run_context.task_run.name)
        return 'dynamic result'

    def factory(ctx: RunContext[Any]) -> Capability[Any]:
        return Capability(tools=[dynamic_tool])

    agent = Agent(
        TestModel(),
        name='prefect_dynamic_capability',
        capabilities=[DynamicCapability(capability_func=factory, id='dyn'), PrefectDurability()],
    )

    @flow
    async def run_agent() -> str:
        return (await agent.run('Call the tool')).output

    assert await run_agent() == '{"dynamic_tool":"dynamic result"}'
    assert calls == ['called']
    assert len(task_run_names) == 1
    assert task_run_names[0].startswith('Call Tool: dynamic_tool')


def test_prefect_durability_dynamic_capability_requires_id() -> None:
    def factory(ctx: RunContext[Any]) -> Capability[Any]:
        return Capability()

    with pytest.raises(UserError, match=r"DynamicCapability\(\.\.\., id='user-tools'\)"):
        Agent(
            TestModel(),
            name='prefect_dynamic_capability_no_id',
            capabilities=[DynamicCapability(capability_func=factory), PrefectDurability()],
        )


async def test_prefect_durability_dynamic_capability_tool_opts_out_of_task() -> None:
    task_contexts: list[TaskRunContext[Any] | None] = []

    def dynamic_tool() -> str:
        task_contexts.append(TaskRunContext.get())
        return 'dynamic result'

    def factory(ctx: RunContext[Any]) -> Capability[Any]:
        toolset = FunctionToolset()
        toolset.add_function(dynamic_tool, metadata={'prefect': False})
        return Capability(toolsets=[toolset])

    agent = Agent(
        TestModel(),
        name='prefect_dynamic_capability_inline_tool',
        capabilities=[DynamicCapability(capability_func=factory, id='dyn_inline'), PrefectDurability()],
    )

    @flow
    async def run_agent() -> str:
        return (await agent.run('Call the tool')).output

    assert await run_agent() == '{"dynamic_tool":"dynamic result"}'
    assert task_contexts == [None]


def test_prefect_durability_requires_agent_name() -> None:
    """PrefectDurability raises UserError when the agent has no name."""
    with pytest.raises(UserError, match='unique `name`'):
        Agent(_durability_fn_model, capabilities=[PrefectDurability()])


def test_prefect_durability_explicit_name_overrides_agent_name_and_supports_unnamed_agent() -> None:
    named_agent = Agent(_durability_fn_model, name='agent-name', capabilities=[PrefectDurability(name='custom')])
    bound = PrefectDurability.from_agent(named_agent)
    assert bound is not None
    assert bound.name == 'custom'

    unnamed_agent = Agent(_durability_fn_model, capabilities=[PrefectDurability(name='unnamed-custom')])
    unnamed_bound = PrefectDurability.from_agent(unnamed_agent)
    assert unnamed_bound is not None
    assert unnamed_bound.name == 'unnamed-custom'


def test_prefect_durability_requires_model() -> None:
    """PrefectDurability raises UserError when the agent has no model at all."""
    with pytest.raises(UserError, match='needs to have a `model`'):
        Agent(name='needs_model', capabilities=[PrefectDurability()])


def _prefect_alt_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='alt-response')])


_prefect_alt_model = FunctionModel(_prefect_alt_model_fn, model_name='alt')


async def test_prefect_durability_runtime_registered_model() -> None:
    """A model registered in `models=` can be selected at run time, by key or instance.

    The `model_id` crosses the task boundary and the task rebuilds the model from the
    registry, so the response is produced by the selected model inside the Prefect task.
    """
    agent = Agent(
        _durability_fn_model,
        name='durability_runtime_registered',
        capabilities=[PrefectDurability(models={'alt': _prefect_alt_model})],
    )

    # Separate flow runs so each request gets its own task-cache scope (Prefect caches
    # tasks by input hash, and both requests would otherwise share the `'alt'` model task).
    @flow
    async def run_by_key() -> str:
        return (await agent.run('hello', model='alt')).output

    @flow
    async def run_by_instance() -> str:
        return (await agent.run('hello', model=_prefect_alt_model)).output

    assert await run_by_key() == 'alt-response'
    assert await run_by_instance() == 'alt-response'


async def test_prefect_durability_override_registered_model() -> None:
    """A model set via `override(model=...)` round-trips the task boundary like a per-run `model=`."""
    agent = Agent(
        _durability_fn_model,
        name='durability_override_registered',
        capabilities=[PrefectDurability(models={'alt': _prefect_alt_model})],
    )

    @flow
    async def run_agent() -> str:
        with agent.override(model='alt'):
            result = await agent.run('hello')
        return result.output

    assert await run_agent() == 'alt-response'


async def test_prefect_durability_unrebuildable_runtime_model_errors() -> None:
    """An unregistered instance whose `model_id` can't be fed back through `infer_model` errors helpfully.

    `TestModel()` round-trips as `'test:test'`, which `infer_model` can't rebuild; instead of a
    bare 'Unknown provider' the task points at the `models=` / `ResolveModelId` escape hatches.
    """
    agent = Agent(_durability_fn_model, name='durability_unrebuildable', capabilities=[PrefectDurability()])

    @flow
    async def run_agent() -> None:
        await agent.run('hello', model=TestModel())

    with pytest.raises(UserError, match='could not be rebuilt'):
        await run_agent()


def _prefect_tenant_resolver(ctx: ModelResolutionContext[str], model_id: str) -> FunctionModel | None:
    """Resolve the 'tenant-model' alias to a model built from the run's deps.

    Matches the alias exactly: the run's original model-id string (not the resolved
    model's `'function:tenant-model'`) is what crosses the durable boundary, so the
    worker-side re-resolution sees the same string the caller wrote.
    """
    if model_id != 'tenant-model':
        return None
    tenant = ctx.deps

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=f'tenant:{tenant}')])

    return FunctionModel(fn, model_name='tenant-model')


async def test_prefect_durability_resolve_model_id_capability_is_deps_aware() -> None:
    """A deps-aware `ResolveModelId` resolver rebuilds the model with the run's deps inside the task."""
    agent = Agent(
        _durability_fn_model,
        name='durability_tenant',
        deps_type=str,
        capabilities=[ResolveModelId(_prefect_tenant_resolver), PrefectDurability()],
    )

    # Separate flow runs so each request gets its own task-cache scope: within one flow the
    # `'tenant-model'` model task caches by input and would replay the first tenant's result.
    @flow
    async def run_agent(model_id: str, tenant: str) -> str:
        return (await agent.run('hi', model=model_id, deps=tenant)).output

    assert await run_agent('tenant-model', 'acme') == 'tenant:acme'
    assert await run_agent('tenant-model', 'globex') == 'tenant:globex'
    # A string the resolver doesn't recognize defers to the default `infer_model` flow.
    assert await run_agent('test', 'acme') == 'success (no tool calls)'


async def test_prefect_durability_alias_default_model() -> None:
    """An agent whose *default* model is an alias only a `ResolveModelId` capability can resolve.

    `infer_model` can't build `'tenant-model'`, so binding registers no concrete default;
    every request carries the raw alias string across the task boundary and the task
    re-resolves it with the run's deps.
    """
    agent = Agent(
        'tenant-model',
        name='durability_alias_default',
        deps_type=str,
        capabilities=[ResolveModelId(_prefect_tenant_resolver), PrefectDurability()],
    )

    @flow
    async def run_agent() -> str:
        result = await agent.run('hi', deps='acme')
        return result.output

    assert await run_agent() == 'tenant:acme'


async def test_prefect_durability_allows_instrumented_default_model() -> None:
    """An outer `Instrumentation` capability wraps the model, but the default model is still accepted.

    `_find_model_id` unwraps the `InstrumentedModel` wrapper before comparing instances by
    identity, so an instrumented run still takes the default's `model_id=None` fast path.
    """
    agent = Agent(
        _durability_fn_model,
        name='durability_instrumented_default',
        capabilities=[Instrumentation(settings=InstrumentationSettings()), PrefectDurability()],
    )

    @flow
    async def run_agent() -> str:
        result = await agent.run('hello')
        return result.output

    assert await run_agent() == 'Echo: hello'


def test_prefect_durability_get_ordering() -> None:
    """PrefectDurability declares innermost ordering."""
    from pydantic_ai.capabilities.abstract import CapabilityOrdering

    assert PrefectDurability().get_ordering() == CapabilityOrdering(position='innermost')


def test_prefect_durability_get_serialization_name() -> None:
    """PrefectDurability is not spec-serializable."""
    assert PrefectDurability.get_serialization_name() is None


async def test_prefect_durability_passes_through_non_wrappable_leaf() -> None:
    """Leaf toolsets that aren't function/MCP toolsets are left as-is, not Prefect-wrapped.

    `ExternalToolset` doesn't perform I/O of its own, so it isn't wrapped in a task and
    isn't registered for run-time swapping. Running the agent exercises the run-time swap's
    pass-through for such an unregistered leaf.
    """
    agent = Agent(
        _durability_fn_model,
        name='durability_external',
        toolsets=[ExternalToolset([ToolDefinition(name='ext_tool')], id='ext')],
        capabilities=[PrefectDurability()],
    )
    bound = PrefectDurability.from_agent(agent)
    assert bound is not None
    assert 'ext' not in bound._toolsets_by_id  # pyright: ignore[reportPrivateUsage]

    result = await agent.run('Hello external')
    assert result.output == 'Echo: Hello external'


async def _durability_stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    for msg in reversed(messages):  # pragma: no branch - first message carries the prompt
        for part in msg.parts:  # pragma: no branch - first part is the UserPromptPart
            if isinstance(part, UserPromptPart):  # pragma: no branch - same reason
                yield f'Echo: {part.content}'
                return
    yield 'no prompt'  # pragma: no cover


async def test_prefect_durability_streaming_in_flow() -> None:
    """`ProcessEventStream` receives captured model events in flow code."""
    events_in_task: list[tuple[AgentStreamEvent, bool]] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events_in_task.append((event, TaskRunContext.get() is not None))

    stream_model = FunctionModel(_durability_model_fn, stream_function=_durability_stream_fn)
    agent = Agent(
        stream_model,
        name='durability_streaming',
        capabilities=[ProcessEventStream(handler), PrefectDurability()],
    )

    @flow
    async def run_durable_streaming_agent() -> str:
        result = await agent.run('Hello streaming')
        return result.output

    output = await run_durable_streaming_agent()
    assert output == 'Echo: Hello streaming'
    model_events_in_task = [
        in_task for event, in_task in events_in_task if isinstance(event, (PartStartEvent, PartDeltaEvent))
    ]
    assert model_events_in_task
    assert not any(model_events_in_task)


async def _chunks_stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'Stream'
    yield 'ed '
    yield 'response'


async def test_prefect_durability_process_event_stream_fires_flow_side() -> None:
    """`ProcessEventStream` sees the real captured events replayed in the flow."""
    events_received: list[AgentStreamEvent] = []

    async def collect(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            assert TaskRunContext.get() is None
            events_received.append(event)

    stream_model = FunctionModel(_durability_model_fn, stream_function=_chunks_stream_fn)
    agent = Agent(
        stream_model,
        name='durability_process_stream',
        capabilities=[ProcessEventStream(collect), PrefectDurability()],
    )

    @flow
    async def run_durable_agent() -> str:
        result = await agent.run('Hello')
        return result.output

    output = await run_durable_agent()
    assert output == 'Streamed response'

    delta_events = [
        e.delta.content_delta
        for e in events_received
        if isinstance(e, PartDeltaEvent) and isinstance(e.delta, TextPartDelta)
    ]
    assert delta_events == ['ed ', 'response']


async def test_prefect_durability_buffers_caller_streams_and_keeps_handlers_distinct() -> None:
    live_events: list[AgentStreamEvent] = []
    buffered_events: list[AgentStreamEvent] = []

    async def live_handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            assert TaskRunContext.get() is not None
            live_events.append(event)

    async def buffered_handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            assert TaskRunContext.get() is None
            buffered_events.append(event)

    agent = Agent(
        TestModel(custom_output_text='hello world'),
        name='durability_buffered_streams',
        capabilities=[ProcessEventStream(buffered_handler), PrefectDurability(event_stream_handler=live_handler)],
    )

    @flow
    async def run_durable_streams() -> tuple[list[str], str, list[str], int, int]:
        async with agent.run_stream('Hello') as stream:
            chunks = [chunk async for chunk in stream.stream_text(debounce_by=None)]
            output = await stream.get_output()
        live_handler_calls = sum(isinstance(event, PartStartEvent) for event in live_events)
        buffered_handler_calls = sum(isinstance(event, PartStartEvent) for event in buffered_events)

        async with agent.run_stream_events('Hello') as event_stream:
            events = [event async for event in event_stream]
        deltas = [
            event.delta.content_delta
            for event in events
            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta)
        ]
        return chunks, output, deltas, live_handler_calls, buffered_handler_calls

    assert await run_durable_streams() == (
        ['hello ', 'hello world'],
        'hello world',
        ['hello ', 'world'],
        1,
        1,
    )


async def test_prefect_durability_event_stream_handler() -> None:
    events_in_boundary: list[tuple[AgentStreamEvent, bool]] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events_in_boundary.append((event, TaskRunContext.get() is not None))

    async def handled_tool() -> str:
        return 'handled'

    durability = PrefectDurability(event_stream_handler=handler)
    agent = Agent(TestModel(), name='durability_handler', tools=[handled_tool], capabilities=[durability])

    @flow
    async def run_durable_agent() -> str:
        return (await agent.run('Hello')).output

    await run_durable_agent()
    events = [event for event, _ in events_in_boundary]
    assert events
    assert all(in_boundary for _, in_boundary in events_in_boundary)
    assert sum(isinstance(event, FunctionToolCallEvent) for event in events) == 1
    assert sum(isinstance(event, FunctionToolResultEvent) for event in events) == 1
    assert any(isinstance(event, PartStartEvent) for event in events)
    assert any(isinstance(event, FinalResultEvent) for event in events)


async def test_prefect_durability_event_stream_handler_outside_flow() -> None:
    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    durability = PrefectDurability(event_stream_handler=handler)
    agent = Agent(TestModel(custom_output_text='done'), name='outside_handler', capabilities=[durability])
    with agent.override():
        await agent.run('Hello')
    assert any(isinstance(event, PartStartEvent) for event in events)


def test_prefect_durability_without_handler_does_not_wrap_event_stream() -> None:
    assert PrefectDurability().has_wrap_run_event_stream is False


async def test_prefect_durability_runtime_handler_receives_buffered_events() -> None:
    """A per-run `event_stream_handler` passed to `agent.run()` inside a flow receives events.

    The buffered replay preserves real granular deltas — the per-run handler sees the same
    multi-chunk stream the construction-time handler would see.
    """
    events_received: list[AgentStreamEvent] = []

    async def runtime_collect(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events_received.append(event)

    stream_model = FunctionModel(_durability_model_fn, stream_function=_chunks_stream_fn)
    agent = Agent(stream_model, name='durability_runtime_handler', capabilities=[PrefectDurability()])

    @flow
    async def run_durable_agent() -> str:
        result = await agent.run('Hello', event_stream_handler=runtime_collect)
        return result.output

    output = await run_durable_agent()
    assert output == 'Streamed response'

    delta_events = [
        e.delta.content_delta
        for e in events_received
        if isinstance(e, PartDeltaEvent) and isinstance(e.delta, TextPartDelta)
    ]
    assert delta_events == ['ed ', 'response']


# --- Continuation chains (suspended → complete) run one task per segment ---
#
# When a model suspends a turn (Anthropic `pause_turn`, OpenAI background mode), the
# continuation loop in the innermost `model_request`/`model_request_stream` helpers runs
# flow-side under `PrefectDurability`, dispatching each segment through its own model
# request task. These tests use a scripted model (no cassettes: `FunctionModel` can't emit
# suspended streaming segments, and VCR matchers wouldn't pin the chain shape).


async def test_prefect_durability_continuation_chain_in_flow() -> None:
    """A suspended → complete chain resolves across per-segment Prefect tasks, as one merged response.

    Usage is counted once — a continuation isn't a separate request step.
    """
    model = ScriptedContinuationModel(
        responses=[
            scripted_response(
                texts=['The answer '],
                state='suspended',
                provider_response_id='cont1',
                input_tokens=5,
                output_tokens=2,
            ),
            scripted_response(texts=['is 42.'], provider_response_id='cont2', input_tokens=3, output_tokens=4),
        ]
    )
    agent = Agent(model, name='durability_continuation', capabilities=[PrefectDurability()])

    results: list[AgentRunResult[str]] = []

    @flow
    async def run_durable_agent() -> str:
        result = await agent.run('go')
        results.append(result)
        return result.output

    output = await run_durable_agent()

    assert output == 'The answer is 42.'
    result = results[0]
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'complete'
    assert [part.content for part in response.parts if isinstance(part, TextPart)] == ['The answer ', 'is 42.']
    assert result.usage.requests == 1
    assert result.usage.input_tokens == 8
    assert result.usage.output_tokens == 6
    # Each segment ran in its own durable boundary.
    assert model.request_calls == 2


async def test_prefect_durability_continuation_usage_limit_cancels_suspended() -> None:
    """A usage limit tripped between segments cancels the live suspended job in its own task.

    The continuation loop runs flow-side and checks the limit as each segment merges; the
    provider teardown of the abandoned server-side job is I/O, so it must cross the boundary
    through the dedicated cancellation task. We assert a `TaskRunContext` is active inside
    the model's `request` and `cancel_suspended_response`, proving each segment and the
    teardown ran in their own Prefect tasks rather than inline in the flow, and that the
    error surfaces to flow code with its real type.
    """
    calls_in_task: list[tuple[str, bool]] = []

    class RecordingContinuationModel(ScriptedContinuationModel):
        async def request(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            calls_in_task.append(('request', TaskRunContext.get() is not None))
            return await super().request(messages, model_settings, model_request_parameters)

        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            calls_in_task.append(('cancel', TaskRunContext.get() is not None))
            await super().cancel_suspended_response(response)

    model = RecordingContinuationModel(
        responses=[
            scripted_response(
                texts=['still going '],
                state='suspended',
                provider_response_id='cont1',
                input_tokens=10,
                output_tokens=5,
            ),
            scripted_response(
                texts=['keeps going '],
                state='suspended',
                provider_response_id='cont2',
                input_tokens=100,
                output_tokens=50,
            ),
        ]
    )
    agent = Agent(model, name='durability_continuation_usage_limit', capabilities=[PrefectDurability()])

    @flow
    async def run_agent() -> None:
        await agent.run('go', usage_limits=UsageLimits(total_tokens_limit=20))

    with pytest.raises(UsageLimitExceeded, match='total_tokens_limit'):
        await run_agent()

    # The over-budget merge was still suspended, so the live job was cancelled before raising.
    assert [cancelled.provider_response_id for cancelled in model.cancelled] == ['cont2']
    assert calls_in_task == [('request', True), ('request', True), ('cancel', True)]


async def test_prefect_durability_streaming_continuation_chain_in_flow() -> None:
    """A streamed suspended → complete chain is stitched across per-segment tasks.

        `ProcessEventStream` receives each captured segment in flow code, and the
    final response merges both segments' text with usage summed once.
    """
    model = ScriptedContinuationModel(
        segments=[
            StreamSegment(
                texts=['The answer '],
                state='suspended',
                provider_response_id='cont1',
                input_tokens=5,
                output_tokens=2,
            ),
            StreamSegment(
                texts=['is 42.'], state='complete', provider_response_id='cont2', input_tokens=3, output_tokens=4
            ),
        ]
    )

    events_received: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events_received.append(event)

    agent = Agent(
        model,
        name='durability_continuation_stream',
        capabilities=[ProcessEventStream(handler), PrefectDurability()],
    )

    results: list[AgentRunResult[str]] = []

    @flow
    async def run_durable_agent() -> str:
        result = await agent.run('go')
        results.append(result)
        return result.output

    output = await run_durable_agent()

    assert output == 'The answer is 42.'
    result = results[0]
    assert result.usage.requests == 1
    assert result.usage.input_tokens == 8
    assert result.usage.output_tokens == 6
    indices = [
        (type(event).__name__, event.index)
        for event in events_received
        if isinstance(event, (PartStartEvent, PartDeltaEvent))
    ]
    assert indices == snapshot(
        [('PartStartEvent', 0), ('PartDeltaEvent', 0), ('PartStartEvent', 1), ('PartDeltaEvent', 1)]
    )
    assert model.request_stream_calls == 2


async def test_prefect_durability_continuation_resume_from_history() -> None:
    """A `message_history` ending in a suspended response resumes inside the Prefect task.

    The suspended tail crosses the task boundary as the last request message and seeds the
    continuation loop there, so the run completes the paused turn instead of starting a
    fresh generation.
    """
    model = ScriptedContinuationModel(
        responses=[scripted_response(texts=['is 42.'], provider_response_id='cont2', input_tokens=3, output_tokens=4)]
    )
    agent = Agent(model, name='durability_continuation_resume', capabilities=[PrefectDurability()])

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='go')]),
        scripted_response(
            texts=['The answer '], state='suspended', provider_response_id='cont1', input_tokens=5, output_tokens=2
        ),
    ]

    results: list[AgentRunResult[str]] = []

    @flow
    async def run_durable_agent() -> str:
        result = await agent.run(message_history=history)
        results.append(result)
        return result.output

    output = await run_durable_agent()

    assert output == 'The answer is 42.'
    result = results[0]
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'complete'
    assert [part.content for part in response.parts if isinstance(part, TextPart)] == ['The answer ', 'is 42.']
    assert result.usage.requests == 1
    assert result.usage.input_tokens == 8
    assert result.usage.output_tokens == 6
    # The continuation request ran inside the boundary — the seed wasn't re-generated.
    assert model.request_calls == 1
