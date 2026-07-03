# pyright: reportDeprecated=false
# `DBOSAgent` (the wrapper-agent path) is deprecated in favor of the
# `DBOSDurability` capability, but this file still exercises both paths in
# parallel for parity. Silenced at file level rather than annotating every
# individual usage.
from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, cast

import pytest
from httpx import AsyncClient
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    RetryPromptPart,
    RunContext,
    RunUsage,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    ToolsetTool,
    UserPromptPart,
)
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RequestUsage

from .conftest import IsDatetime, IsNow, IsStr

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSAgent, DBOSDurability, DBOSModel
    from pydantic_ai.durable_exec.dbos._mcp import DBOSMCPToolsetBase
    from pydantic_ai.durable_exec.dbos._mcp_toolset import DBOSMCPToolset

except ImportError:  # pragma: lax no cover
    pytest.skip('DBOS is not installed', allow_module_level=True)

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

from pydantic_ai import ExternalToolset, FunctionToolset
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._inline_snapshot import snapshot

# `DBOSAgent` is deprecated in favor of `capabilities=[DBOSDurability(...)]`.
# These tests exercise the wrapper-agent path on purpose; suppress the warning here
# rather than globally in `pyproject.toml`. The `pytestmark` entry below covers warnings
# emitted *inside* test functions; the `filterwarnings` call below covers warnings emitted
# at module import time (e.g. `simple_dbos_agent = DBOSAgent(...)`).
warnings.filterwarnings('ignore', message='`DBOSAgent` is deprecated', category=DeprecationWarning)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.xdist_group(name='dbos'),
    pytest.mark.filterwarnings('ignore:`DBOSAgent` is deprecated:DeprecationWarning'),
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


@pytest.fixture(autouse=True, scope='module')
def setup_logfire_instrumentation() -> Iterator[None]:
    # Set up logfire for the tests.
    logfire.configure(metrics=False)
    yield


@contextmanager
def workflow_raises(exc_type: type[Exception], exc_message: str) -> Generator[None]:
    """Helper for asserting that a DBOS workflow fails with the expected error."""
    with pytest.raises(Exception) as exc_info:
        yield
    assert isinstance(exc_info.value, Exception)
    assert str(exc_info.value) == exc_message


@pytest.fixture(scope='module')
def dbos(tmp_path_factory: pytest.TempPathFactory) -> Generator[DBOS, Any, None]:
    dbos_sqlite_file = tmp_path_factory.mktemp('dbos') / 'dbostest.sqlite'
    dbos_config: DBOSConfig = {
        'name': 'pydantic_dbos_tests',
        'system_database_url': f'sqlite:///{dbos_sqlite_file}',
        'run_admin_server': False,
        'enable_otlp': True,
    }
    dbos = DBOS(config=dbos_config)
    DBOS.launch()
    try:
        yield dbos
    finally:
        DBOS.destroy()


model = OpenAIChatModel(
    'gpt-4o',
    provider=OpenAIProvider(
        api_key=os.getenv('OPENAI_API_KEY', 'mock-api-key'),
        http_client=http_client,
    ),
)

# Not necessarily need to define it outside of the function. DBOS just requires workflows to be statically defined so recovery would be able to find those workflows. It's nice to reuse it in multiple tests.
simple_agent = Agent(model, name='simple_agent')
simple_dbos_agent = DBOSAgent(simple_agent)


async def test_simple_agent_run_in_workflow(allow_model_requests: None, dbos: DBOS, openai_api_key: str) -> None:
    """Test that a simple agent can run in a DBOS workflow."""

    @DBOS.workflow()
    async def run_simple_agent() -> str:
        result = await simple_dbos_agent.run('What is the capital of Mexico?')
        return result.output

    output = await run_simple_agent()
    assert output == snapshot('The capital of Mexico is Mexico City.')


class Deps(BaseModel):
    country: str


# Wrap event_stream_handler as a DBOS step because it's non-deterministic (uses logfire)
@DBOS.step()
async def event_stream_handler(
    ctx: RunContext[Deps],
    stream: AsyncIterable[AgentStreamEvent],
):
    logfire.info(f'{ctx.run_step=}')
    async for event in stream:
        logfire.info('event', event=event)


@DBOS.step()
async def runtime_event_stream_handler(
    ctx: RunContext[object],
    stream: AsyncIterable[AgentStreamEvent],
):
    logfire.info(f'{ctx.run_step=}')
    async for event in stream:
        logfire.info('runtime_event', event=event)


# This doesn't need to be a step
async def get_country(ctx: RunContext[Deps]) -> str:
    return ctx.deps.country


class WeatherArgs(BaseModel):
    city: str


@DBOS.step()
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
    capabilities=[Instrumentation(settings=InstrumentationSettings())],  # Enable instrumentation for testing
    name='complex_agent',
)
complex_dbos_agent = DBOSAgent(complex_agent, event_stream_handler=event_stream_handler)
seq_complex_dbos_agent = DBOSAgent(
    complex_agent,
    event_stream_handler=event_stream_handler,
    parallel_execution_mode='sequential',
    name='seq_complex_agent',
)


async def runtime_handler_stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[str]:
    del messages, agent_info
    yield 'Hello'
    yield ' world'


runtime_handler_stream_agent = Agent(
    FunctionModel(stream_function=runtime_handler_stream_function),
    name='runtime_handler_stream_agent',
)
runtime_handler_stream_dbos_agent = DBOSAgent(runtime_handler_stream_agent)


async def test_complex_agent_run_in_workflow(allow_model_requests: None, dbos: DBOS, capfire: CaptureLogfire) -> None:
    # Set a workflow ID for testing list steps
    wfid = str(uuid.uuid4())
    with SetWorkflowID(wfid):
        # DBOSAgent already wraps the `run` function as a DBOS workflow, so we can just call it directly.
        result = await complex_dbos_agent.run(
            'Tell me: the capital of the country; the weather there; the product name', deps=Deps(country='Mexico')
        )
    assert result.output == snapshot(
        Response(
            answers=[
                Answer(label='Capital of the country', answer='Mexico City'),
                Answer(label='Weather in the capital', answer='Sunny'),
                Answer(label='Product Name', answer='Pydantic AI'),
            ]
        )
    )

    # Make sure the steps are persisted correctly in the DBOS database.
    steps = await dbos.list_workflow_steps_async(wfid)
    assert [step['function_name'] for step in steps] == snapshot(
        [
            'complex_agent__mcp_server__mcp.get_tools',
            'complex_agent__model.request_stream',
            'event_stream_handler',
            'event_stream_handler',
            'complex_agent__mcp_server__mcp.call_tool',
            'event_stream_handler',
            'event_stream_handler',
            'complex_agent__model.request_stream',
            'event_stream_handler',
            'get_weather',
            'event_stream_handler',
            'complex_agent__model.request_stream',
            'event_stream_handler',
            'event_stream_handler',
        ]
    )

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

    assert len(basic_spans_by_id) > 0, 'No spans were exported'
    root_span = None
    for basic_span in basic_spans_by_id.values():
        if basic_span.parent_id is None:
            root_span = basic_span
        else:
            parent_id = basic_span.parent_id
            parent_span = basic_spans_by_id[parent_id]
            parent_span.children.append(basic_span)

    def _normalize_json_spans(span: BasicSpan) -> None:
        """Normalize non-deterministic tool_call_ids in JSON event spans."""
        import json

        for child in span.children:
            if child.content.startswith('{'):
                try:
                    data = json.loads(child.content)
                    _strip_volatile_fields(data)
                    child.content = json.dumps(data)
                except json.JSONDecodeError:
                    pass
            _normalize_json_spans(child)

    def _strip_volatile_fields(obj: dict[str, Any]) -> None:
        for k, v in obj.items():
            if k in ('tool_call_id', 'timestamp'):
                obj[k] = None
            elif isinstance(v, dict):
                _strip_volatile_fields(cast(dict[str, Any], v))

    assert root_span is not None
    _normalize_json_spans(root_span)

    # Assert the root span and its structure matches expected hierarchy
    assert root_span == snapshot(
        BasicSpan(
            content='complex_agent.run',
            children=[
                BasicSpan(
                    content='complex_agent run',
                    children=[
                        BasicSpan(
                            content='complex_agent__mcp_server__mcp.get_tools',
                            children=[BasicSpan(content='tools/list')],
                        ),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='complex_agent__model.request_stream',
                                    children=[
                                        BasicSpan(content='ctx.run_step=1'),
                                        BasicSpan(
                                            content='{"index": 0, "part": {"tool_name": "get_country", "args": "", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "previous_part_kind": null, "event_kind": "part_start"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "{}", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "part": {"tool_name": "get_country", "args": "{}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "next_part_kind": "tool-call", "event_kind": "part_end"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 1, "part": {"tool_name": "get_product_name", "args": "", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "previous_part_kind": "tool-call", "event_kind": "part_start"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 1, "delta": {"tool_name_delta": null, "args_delta": "{}", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 1, "part": {"tool_name": "get_product_name", "args": "{}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "next_part_kind": null, "event_kind": "part_end"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='event_stream_handler',
                            children=[
                                BasicSpan(content='ctx.run_step=1'),
                                BasicSpan(
                                    content='{"part": {"tool_name": "get_country", "args": "{}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "args_valid": true, "event_kind": "function_tool_call"}'
                                ),
                            ],
                        ),
                        BasicSpan(
                            content='event_stream_handler',
                            children=[
                                BasicSpan(content='ctx.run_step=1'),
                                BasicSpan(
                                    content='{"part": {"tool_name": "get_product_name", "args": "{}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "args_valid": true, "event_kind": "function_tool_call"}'
                                ),
                            ],
                        ),
                        BasicSpan(content='running tool: get_country'),
                        BasicSpan(
                            content='running tool: get_product_name',
                            children=[
                                BasicSpan(
                                    content='complex_agent__mcp_server__mcp.call_tool',
                                    children=[BasicSpan(content='tools/call get_product_name')],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='event_stream_handler',
                            children=[
                                BasicSpan(content='ctx.run_step=1'),
                                BasicSpan(
                                    content='{"part": {"tool_name": "get_country", "content": "Mexico", "tool_call_id": null, "tool_kind": null, "metadata": null, "timestamp": null, "outcome": "success", "part_kind": "tool-return"}, "content": null, "event_kind": "function_tool_result"}'
                                ),
                            ],
                        ),
                        BasicSpan(
                            content='event_stream_handler',
                            children=[
                                BasicSpan(content='ctx.run_step=1'),
                                BasicSpan(
                                    content='{"part": {"tool_name": "get_product_name", "content": "Pydantic AI", "tool_call_id": null, "tool_kind": null, "metadata": null, "timestamp": null, "outcome": "success", "part_kind": "tool-return"}, "content": null, "event_kind": "function_tool_result"}'
                                ),
                            ],
                        ),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='complex_agent__model.request_stream',
                                    children=[
                                        BasicSpan(content='ctx.run_step=2'),
                                        BasicSpan(
                                            content='{"index": 0, "part": {"tool_name": "get_weather", "args": "", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "previous_part_kind": null, "event_kind": "part_start"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "{\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "city", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "Mexico", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " City", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\"}", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "part": {"tool_name": "get_weather", "args": "{\\"city\\":\\"Mexico City\\"}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "next_part_kind": null, "event_kind": "part_end"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='event_stream_handler',
                            children=[
                                BasicSpan(content='ctx.run_step=2'),
                                BasicSpan(
                                    content='{"part": {"tool_name": "get_weather", "args": "{\\"city\\":\\"Mexico City\\"}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "args_valid": true, "event_kind": "function_tool_call"}'
                                ),
                            ],
                        ),
                        BasicSpan(content='running tool: get_weather', children=[BasicSpan(content='get_weather')]),
                        BasicSpan(
                            content='event_stream_handler',
                            children=[
                                BasicSpan(content='ctx.run_step=2'),
                                BasicSpan(
                                    content='{"part": {"tool_name": "get_weather", "content": "sunny", "tool_call_id": null, "tool_kind": null, "metadata": null, "timestamp": null, "outcome": "success", "part_kind": "tool-return"}, "content": null, "event_kind": "function_tool_result"}'
                                ),
                            ],
                        ),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='complex_agent__model.request_stream',
                                    children=[
                                        BasicSpan(content='ctx.run_step=3'),
                                        BasicSpan(
                                            content='{"index": 0, "part": {"tool_name": "final_result", "args": "", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "previous_part_kind": null, "event_kind": "part_start"}'
                                        ),
                                        BasicSpan(
                                            content='{"tool_name": "final_result", "tool_call_id": null, "event_kind": "final_result"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "{\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "answers", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":[", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "{\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "label", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "Capital", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " of", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " the", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " country", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\",\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "answer", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "Mexico", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " City", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\"},{\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "label", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "Weather", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " in", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " the", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " capital", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\",\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "answer", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "Sunny", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\"},{\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "label", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "Product", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " Name", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\",\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "answer", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "P", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "yd", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "antic", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " AI", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\"}", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "]}", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index": 0, "part": {"tool_name": "final_result", "args": "{\\"answers\\":[{\\"label\\":\\"Capital of the country\\",\\"answer\\":\\"Mexico City\\"},{\\"label\\":\\"Weather in the capital\\",\\"answer\\":\\"Sunny\\"},{\\"label\\":\\"Product Name\\",\\"answer\\":\\"Pydantic AI\\"}]}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "next_part_kind": null, "event_kind": "part_end"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='event_stream_handler',
                            children=[
                                BasicSpan(content='ctx.run_step=3'),
                                BasicSpan(
                                    content='{"part": {"tool_name": "final_result", "args": "{\\"answers\\":[{\\"label\\":\\"Capital of the country\\",\\"answer\\":\\"Mexico City\\"},{\\"label\\":\\"Weather in the capital\\",\\"answer\\":\\"Sunny\\"},{\\"label\\":\\"Product Name\\",\\"answer\\":\\"Pydantic AI\\"}]}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "args_valid": true, "event_kind": "output_tool_call"}'
                                ),
                            ],
                        ),
                        BasicSpan(
                            content='event_stream_handler',
                            children=[
                                BasicSpan(content='ctx.run_step=3'),
                                BasicSpan(
                                    content='{"part": {"tool_name": "final_result", "content": "Final result processed.", "tool_call_id": null, "tool_kind": null, "metadata": null, "timestamp": null, "outcome": "success", "part_kind": "tool-return"}, "event_kind": "output_tool_result"}'
                                ),
                            ],
                        ),
                    ],
                )
            ],
        )
    )


async def test_dbos_agent_run_in_workflow_with_runtime_event_stream_handler(
    allow_model_requests: None, dbos: DBOS, capfire: CaptureLogfire
) -> None:
    wfid = str(uuid.uuid4())
    with SetWorkflowID(wfid):
        result = await runtime_handler_stream_dbos_agent.run(
            'Say hello', event_stream_handler=runtime_event_stream_handler
        )

    assert result.output == snapshot('Hello world')

    steps = await dbos.list_workflow_steps_async(wfid)
    step_names = [step['function_name'] for step in steps]
    assert step_names[0] == 'runtime_handler_stream_agent__model.request_stream'
    # The per-run handler fires live, nested inside the model-request step (delivering the streamed
    # events asserted below). It is no longer invoked a second time at the graph level against the
    # already-consumed, empty stream, so it doesn't appear as a separate top-level workflow step.
    assert 'runtime_event_stream_handler' not in step_names

    exported_event_messages = [
        event
        for span in capfire.exporter.exported_spans_as_dict()
        if (attributes := span.get('attributes'))
        and attributes.get('logfire.msg') == 'runtime_event'
        and isinstance((event := attributes.get('event')), str)
    ]
    assert exported_event_messages != []


async def test_dbos_agent_event_stream_handler_property_outside_workflow(dbos: DBOS) -> None:
    # Outside a DBOS workflow, the `event_stream_handler` property resolves to the effective handler
    # directly, rather than the in-workflow per-event dispatcher.
    agent = Agent(TestModel(), name='event_stream_handler_property_agent')
    dbos_agent = DBOSAgent(agent, event_stream_handler=runtime_event_stream_handler)
    assert dbos_agent.event_stream_handler is runtime_event_stream_handler


async def test_mcp_tools_not_cached_when_disabled(allow_model_requests: None, dbos: DBOS) -> None:
    """Verify that wrapper-level caching is skipped when cache_tools=False.

    With caching disabled, every model request should be preceded by a get_tools step,
    rather than only the first one populating the cache.
    """
    mcp_toolset = next(ts for ts in complex_dbos_agent.toolsets if isinstance(ts, DBOSMCPToolset))
    wrapped = cast(MCPToolset[Deps], mcp_toolset.wrapped)

    original_cache_tools = wrapped.cache_tools
    wrapped.cache_tools = False

    try:
        wfid = str(uuid.uuid4())
        with SetWorkflowID(wfid):
            result = await complex_dbos_agent.run(
                'Tell me: the capital of the country; the weather there; the product name', deps=Deps(country='Mexico')
            )
        assert result.output == snapshot(
            Response(
                answers=[
                    Answer(label='Capital of the country', answer='Mexico City'),
                    Answer(label='Weather in the capital', answer='Sunny'),
                    Answer(label='Product Name', answer='Pydantic AI'),
                ]
            )
        )

        steps = await dbos.list_workflow_steps_async(wfid)
        step_names = [step['function_name'] for step in steps]
        # Without caching, get_tools should be called 3 times (once per model request)
        assert step_names.count('complex_agent__mcp_server__mcp.get_tools') == 3
    finally:
        wrapped.cache_tools = original_cache_tools


# Test sequential tool call works
async def test_complex_agent_run_sequential_tool(allow_model_requests: None, dbos: DBOS) -> None:
    # Set a workflow ID for testing list steps
    wfid = str(uuid.uuid4())
    with SetWorkflowID(wfid):
        # DBOSAgent already wraps the `run` function as a DBOS workflow, so we can just call it directly.
        result = await seq_complex_dbos_agent.run(
            'Tell me: the capital of the country; the weather there; the product name', deps=Deps(country='Mexico')
        )
    assert result.output == snapshot(
        Response(
            answers=[
                Answer(label='Capital of the country', answer='Mexico City'),
                Answer(label='Weather in the capital', answer='Sunny'),
                Answer(label='Product Name', answer='Pydantic AI'),
            ]
        )
    )

    # Make sure the steps are persisted correctly in the DBOS database.
    steps = await dbos.list_workflow_steps_async(wfid)
    assert [step['function_name'] for step in steps] == snapshot(
        [
            'seq_complex_agent__mcp_server__mcp.get_tools',
            'seq_complex_agent__model.request_stream',
            'event_stream_handler',
            'event_stream_handler',
            'event_stream_handler',
            'seq_complex_agent__mcp_server__mcp.call_tool',
            'event_stream_handler',
            'seq_complex_agent__model.request_stream',
            'event_stream_handler',
            'get_weather',
            'event_stream_handler',
            'seq_complex_agent__model.request_stream',
            'event_stream_handler',
            'event_stream_handler',
        ]
    )


async def test_multiple_agents(allow_model_requests: None, dbos: DBOS):
    """Test that multiple agents can run in a DBOS workflow."""
    # This is just a smoke test to ensure that multiple agents can run in a DBOS workflow.
    # We don't need to check the output as it's already tested in the individual agent tests.
    result = await simple_dbos_agent.run('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is Mexico City.')

    result = await complex_dbos_agent.run(
        'Tell me: the capital of the country; the weather there; the product name', deps=Deps(country='Mexico')
    )
    assert result.output == snapshot(
        Response(
            answers=[
                Answer(label='Capital of the Country', answer='Mexico City'),
                Answer(label='Weather in Mexico City', answer='Sunny'),
                Answer(label='Product Name', answer='Pydantic AI'),
            ]
        )
    )


async def test_agent_name_collision(allow_model_requests: None, dbos: DBOS):
    with pytest.raises(
        Exception, match="Duplicate instance registration for class 'DBOSAgent' instance 'simple_agent'"
    ):
        DBOSAgent(simple_agent)


async def test_agent_without_name():
    with pytest.raises(
        UserError,
        match=re.escape(
            "An agent needs to have a unique `name` in order to be used with DBOS. The name will be used to identify the agent's workflows and steps."
        ),
    ):
        DBOSAgent(Agent())


async def test_agent_without_model():
    with pytest.raises(
        UserError,
        match=re.escape(
            'An agent needs to have a `model` in order to be used with DBOS, it cannot be set at agent run time.'
        ),
    ):
        DBOSAgent(Agent(name='test_agent'))


async def test_toolset_without_id():
    # Note: this is allowed in DBOS because we don't wrap the tools automatically in a workflow. It's up to the user to define the tools as DBOS steps if they want to use them as steps in a workflow.
    DBOSAgent(Agent(model=model, name='test_agent', toolsets=[FunctionToolset()]))


async def test_dbos_agent():
    assert isinstance(complex_dbos_agent.model, DBOSModel)
    assert complex_dbos_agent.model.wrapped == complex_agent.model

    # DBOS only wraps the MCP server toolsets. Other toolsets are not wrapped.
    toolsets = complex_dbos_agent.toolsets
    assert len(toolsets) == 5

    # Empty function toolset for the agent's own tools
    assert isinstance(toolsets[0], FunctionToolset)
    assert toolsets[0].id == '<agent>'
    assert toolsets[0].tools == {}

    # Function toolset for the wrapped agent's own tools
    assert isinstance(toolsets[1], FunctionToolset)
    assert toolsets[1].id == '<agent>'
    assert toolsets[1].tools.keys() == {'get_weather'}

    # Wrapped 'country' toolset
    assert isinstance(toolsets[2], FunctionToolset)
    assert toolsets[2].id == 'country'
    assert toolsets[2].tools.keys() == {'get_country'}

    # Wrapped 'mcp' MCP server
    assert isinstance(toolsets[3], DBOSMCPToolset)
    assert toolsets[3].id == 'mcp'
    assert toolsets[3].wrapped == complex_agent.toolsets[2]

    # Unwrapped 'external' toolset
    assert isinstance(toolsets[4], ExternalToolset)
    assert toolsets[4].id == 'external'
    assert toolsets[4] == complex_agent.toolsets[3]


async def test_dbos_agent_run(allow_model_requests: None, dbos: DBOS):
    # Note: this runs as a DBOS workflow because we automatically wrap the run function.
    result = await simple_dbos_agent.run('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is Mexico City.')


def test_dbos_agent_run_sync(allow_model_requests: None, dbos: DBOS):
    # Note: this runs as a DBOS workflow because we automatically wrap the run_sync function.
    # This is equivalent to test_dbos_agent_run_sync_in_workflow
    result = simple_dbos_agent.run_sync('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is Mexico City.')


async def test_dbos_agent_run_stream(allow_model_requests: None):
    # Run stream is not a DBOS workflow, so we can use it directly.
    async with simple_dbos_agent.run_stream('What is the capital of Mexico?') as result:
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


async def test_dbos_agent_run_stream_events(allow_model_requests: None):
    # This doesn't work because `run_stream_events` calls `run` internally, which is automatically wrapped in a DBOS workflow.
    with pytest.raises(
        UserError,
        match=re.escape(
            '`agent.run_stream_events()` cannot be used with DBOS. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
        ),
    ):
        async with simple_dbos_agent.run_stream_events('What is the capital of Mexico?'):
            pass


async def test_dbos_agent_iter(allow_model_requests: None):
    output: list[str] = []
    async with simple_dbos_agent.iter('What is the capital of Mexico?') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for chunk in stream.stream_text(debounce_by=None):
                        output.append(chunk)
    assert output == snapshot(
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


def test_dbos_agent_run_sync_in_workflow(allow_model_requests: None, dbos: DBOS):
    # DBOS allows calling `run_sync` inside a workflow as a child workflow.
    @DBOS.workflow()
    def run_sync_workflow():
        result = simple_dbos_agent.run_sync('What is the capital of Mexico?')
        return result.output

    output = run_sync_workflow()
    assert output == snapshot('The capital of Mexico is Mexico City.')


async def test_dbos_agent_run_stream_in_workflow(allow_model_requests: None, dbos: DBOS):
    @DBOS.workflow()
    async def run_stream_workflow():
        async with simple_dbos_agent.run_stream('What is the capital of Mexico?') as result:
            pass
        return await result.get_output()  # pragma: no cover

    with workflow_raises(
        UserError,
        snapshot(
            '`agent.run_stream()` cannot be used inside a DBOS workflow. '
            'Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
        ),
    ):
        await run_stream_workflow()


async def test_dbos_agent_run_stream_events_in_workflow(allow_model_requests: None, dbos: DBOS):
    @DBOS.workflow()
    async def run_stream_events_workflow():
        async with simple_dbos_agent.run_stream_events('What is the capital of Mexico?') as event_stream:
            return [event async for event in event_stream]  # pragma: no cover

    with workflow_raises(
        UserError,
        snapshot(
            '`agent.run_stream_events()` cannot be used with DBOS. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
        ),
    ):
        await run_stream_events_workflow()


async def test_dbos_agent_iter_in_workflow(allow_model_requests: None, dbos: DBOS):
    # DBOS allows calling `iter` inside a workflow as a step.
    @DBOS.workflow()
    async def run_iter_workflow():
        output: list[str] = []
        async with simple_dbos_agent.iter('What is the capital of Mexico?') as run:
            async for node in run:
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for chunk in stream.stream_text(debounce_by=None):
                            output.append(chunk)
        return output

    output = await run_iter_workflow()
    # If called in a workflow, the output is a single concatenated string.
    assert output == snapshot(
        [
            'The capital of Mexico is Mexico City.',
        ]
    )


async def test_dbos_agent_run_in_workflow_with_event_stream_handler(allow_model_requests: None, dbos: DBOS) -> None:
    # DBOS workflow input must be serializable, so we cannot use an inner function as an argument.
    # It's fine to pass in an event_stream_handler that is defined as a top-level function.
    async def simple_event_stream_handler(
        ctx: RunContext,
        stream: AsyncIterable[AgentStreamEvent],
    ):
        pass

    with pytest.raises(Exception) as exc_info:
        await simple_dbos_agent.run('What is the capital of Mexico?', event_stream_handler=simple_event_stream_handler)

    # 3.14+: _pickle.PicklingError("Can't pickle local object <function ...>")
    # <=3.13: AttributeError("Can't get local object '...<locals>...'")
    assert 'local object' in str(exc_info.value) and 'simple_event_stream_handler' in str(exc_info.value)


async def test_dbos_agent_run_in_workflow_with_model(allow_model_requests: None, dbos: DBOS):
    # A non-DBOS model is not wrapped as steps so it's not deterministic and cannot be used in a DBOS workflow.
    with workflow_raises(
        UserError,
        snapshot(
            'Non-DBOS model cannot be set at agent run time inside a DBOS workflow, it must be set at agent creation time.'
        ),
    ):
        await simple_dbos_agent.run('What is the capital of Mexico?', model=model)


async def test_dbos_agent_run_in_workflow_with_toolsets(allow_model_requests: None, dbos: DBOS):
    # Since DBOS does not automatically wrap the tools in a workflow, and allows dynamic steps, we can pass in toolsets directly.
    result = await simple_dbos_agent.run('What is the capital of Mexico?', toolsets=[FunctionToolset()])
    assert result.output == snapshot('The capital of Mexico is Mexico City.')


async def test_dbos_agent_run_in_workflow_with_runtime_external_toolset(dbos: DBOS):
    def request_external_tool(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('external', {'query': 'runtime'}, tool_call_id='call-1')])

    agent = Agent(
        FunctionModel(request_external_tool),
        name='runtime_external_toolset_agent',
        output_type=[str, DeferredToolRequests],
    )
    dbos_agent = DBOSAgent(agent)

    result = await dbos_agent.run(
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


def runtime_tool() -> str:
    return 'tool-result'


async def test_dbos_agent_run_in_workflow_with_runtime_function_toolset(dbos: DBOS):
    # Unlike Temporal and Prefect, DBOS runs function tools inline rather than wrapping them, so a
    # `FunctionToolset` added per-run is allowed and executes like a constructor-time one. (The toolset,
    # like all DBOS workflow arguments, must be serializable, so the tool is a module-level function.)
    def call_then_answer(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('runtime_tool', {}, tool_call_id='call-1')])

    agent = Agent(FunctionModel(call_then_answer), name='runtime_function_toolset_agent')
    dbos_agent = DBOSAgent(agent)

    result = await dbos_agent.run(
        'Call the runtime tool.', toolsets=[FunctionToolset(tools=[runtime_tool], id='runtime_fn')]
    )
    assert result.output == 'done'
    assert any(
        isinstance(part, ToolReturnPart) and part.content == 'tool-result'
        for message in result.all_messages()
        for part in message.parts
    )


async def test_dbos_agent_run_in_workflow_rejects_runtime_mcp_toolset(dbos: DBOS):
    with workflow_raises(
        UserError,
        snapshot(
            'MCPToolset cannot be passed to `run(toolsets=...)` at runtime with DBOS, because toolsets that '
            'execute their own tools or resolve dynamically must be registered for durable execution when the '
            'agent is constructed. Pass them to the agent constructor instead. Non-executing toolsets like '
            '`ExternalToolset` can be passed at runtime.'
        ),
    ):
        await simple_dbos_agent.run(
            'Hello',
            toolsets=[MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='runtime_mcp')],
        )


async def test_dbos_agent_run_in_workflow_rejects_runtime_dynamic_toolset(dbos: DBOS):
    with workflow_raises(
        UserError,
        snapshot(
            'DynamicToolset cannot be passed to `run(toolsets=...)` at runtime with DBOS, because toolsets that '
            'execute their own tools or resolve dynamically must be registered for durable execution when the '
            'agent is constructed. Pass them to the agent constructor instead. Non-executing toolsets like '
            '`ExternalToolset` can be passed at runtime.'
        ),
    ):
        await simple_dbos_agent.run(
            'Hello',
            toolsets=[DynamicToolset(lambda _: FunctionToolset(), id='runtime_dynamic')],
        )


async def test_dbos_agent_override_model_in_workflow(allow_model_requests: None, dbos: DBOS):
    # We cannot override the model to a non-DBOS one in a DBOS workflow.
    with workflow_raises(
        UserError,
        snapshot(
            'Non-DBOS model cannot be contextually overridden inside a DBOS workflow, it must be set at agent creation time.'
        ),
    ):
        with simple_dbos_agent.override(model=model):
            pass


async def test_dbos_agent_override_toolsets_in_workflow(allow_model_requests: None, dbos: DBOS):
    # Since DBOS does not automatically wrap the tools in a workflow, and allows dynamic steps, we can override toolsets directly.
    @DBOS.workflow()
    async def run_with_toolsets():
        with simple_dbos_agent.override(toolsets=[FunctionToolset()]):
            pass

    await run_with_toolsets()


async def test_dbos_agent_override_tools_in_workflow(allow_model_requests: None, dbos: DBOS):
    # Since DBOS does not automatically wrap the tools in a workflow, and allows dynamic steps, we can override tools directly.
    @DBOS.workflow()
    async def run_with_tools():
        with simple_dbos_agent.override(tools=[get_weather]):
            result = await simple_dbos_agent.run('What is the capital of Mexico?')
            return result.output

    output = await run_with_tools()
    assert output == snapshot('The capital of Mexico is Mexico City.')


async def test_dbos_agent_override_deps_in_workflow(allow_model_requests: None, dbos: DBOS):
    # This is allowed
    @DBOS.workflow()
    async def run_with_deps():
        with simple_dbos_agent.override(deps=None):
            result = await simple_dbos_agent.run('What is the capital of the country?')
            return result.output

    output = await run_with_deps()
    assert output == snapshot('The capital of Mexico is Mexico City.')


async def test_dbos_model_stream_direct(allow_model_requests: None, dbos: DBOS):
    @DBOS.workflow()
    async def run_model_stream():
        messages: list[ModelMessage] = [ModelRequest.user_text_prompt('What is the capital of Mexico?')]
        async with model_request_stream(complex_dbos_agent.model, messages) as stream:
            async for _ in stream:
                pass

    with workflow_raises(
        AssertionError,
        snapshot(
            'A DBOS model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
        ),
    ):
        await run_model_stream()


@dataclass
class UnserializableDeps:
    client: AsyncClient


unserializable_deps_agent = Agent(model, name='unserializable_deps_agent', deps_type=UnserializableDeps)


@unserializable_deps_agent.tool
async def get_model_name(ctx: RunContext[UnserializableDeps]) -> int:
    return ctx.deps.client.max_redirects  # pragma: lax no cover


async def test_dbos_agent_with_unserializable_deps_type(allow_model_requests: None, dbos: DBOS):
    unserializable_deps_dbos_agent = DBOSAgent(unserializable_deps_agent)
    # Test this raises a serialization error because httpx.AsyncClient is not serializable.
    with pytest.raises(Exception) as exc_info:
        async with AsyncClient() as client:
            # This will trigger the client to be unserializable
            logfire.instrument_httpx(client, capture_all=True)
            await unserializable_deps_dbos_agent.run('What is the model name?', deps=UnserializableDeps(client=client))

    assert str(exc_info.value) == snapshot("cannot pickle '_thread.RLock' object")


# Test dynamic toolsets in an agent with DBOS


@DBOS.step()
def temperature_celsius(city: str) -> float:
    return 21.0


@DBOS.step()
def temperature_fahrenheit(city: str) -> float:
    return 69.8


weather_toolset = FunctionToolset(tools=[temperature_celsius, temperature_fahrenheit])


@weather_toolset.tool
@DBOS.step()
def conditions(ctx: RunContext, city: str) -> str:
    if ctx.run_step % 2 == 0:
        return "It's sunny"  # pragma: lax no cover
    else:
        return "It's raining"


datetime_toolset = FunctionToolset()


@DBOS.step()
def now_func() -> datetime:
    return datetime.now()


datetime_toolset.add_function(now_func, name='now')


@dataclass
class ToggleableDeps:
    active: Literal['weather', 'datetime']

    def toggle(self):
        if self.active == 'weather':
            self.active = 'datetime'
        else:
            self.active = 'weather'


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


dynamic_dbos_agent = DBOSAgent(dynamic_agent)


def test_dynamic_toolset(dbos: DBOS):
    weather_deps = ToggleableDeps('weather')

    result = dynamic_dbos_agent.run_sync('Toggle the toolset', deps=weather_deps)
    assert result.output == snapshot(
        '{"toggle":null,"temperature_celsius":21.0,"temperature_fahrenheit":69.8,"conditions":"It\'s raining"}'
    )

    result = dynamic_dbos_agent.run_sync('Toggle the toolset', deps=weather_deps)
    assert result.output == snapshot(IsStr(regex=r'{"toggle":null,"now":".+?"}'))


# Test human-in-the-loop with DBOS agent
hitl_agent = Agent(
    model,
    name='hitl_agent',
    output_type=[str, DeferredToolRequests],
    instructions='Just call tools without asking for confirmation.',
)


@hitl_agent.tool
@DBOS.step()
def create_file(ctx: RunContext, path: str) -> None:
    raise CallDeferred


@hitl_agent.tool
@DBOS.step()
def delete_file(ctx: RunContext, path: str) -> bool:
    if not ctx.tool_call_approved:
        raise ApprovalRequired
    return True


hitl_dbos_agent = DBOSAgent(hitl_agent)


async def test_dbos_agent_with_hitl_tool(allow_model_requests: None, dbos: DBOS):
    # Main loop for the agent, keep running until we get a final string output.
    @DBOS.workflow()
    async def hitl_main_loop(prompt: str) -> AgentRunResult[str | DeferredToolRequests]:
        messages: list[ModelMessage] = [ModelRequest.user_text_prompt(prompt)]
        deferred_tool_results: DeferredToolResults | None = None
        while True:
            result = await hitl_dbos_agent.run(message_history=messages, deferred_tool_results=deferred_tool_results)
            messages = result.all_messages()

            if isinstance(result.output, DeferredToolRequests):
                deferred_tool_requests = result.output
                # Set deferred_tool_requests as a DBOS workflow event, so the external functions can see it.
                await DBOS.set_event_async('deferred_tool_requests', deferred_tool_requests)

                # Wait for the deferred tool requests to be handled externally.
                deferred_tool_results = await DBOS.recv_async('deferred_tool_results', timeout_seconds=30)
            else:
                return result

    wf_handle = await DBOS.start_workflow_async(hitl_main_loop, 'Delete the file `.env` and create `test.txt`')

    while True:
        await asyncio.sleep(1)
        status = await wf_handle.get_status()
        if status.status == 'SUCCESS':
            break

        assert status.status == 'PENDING'
        # Wait and check if the workflow has set a deferred tool request event.
        deferred_tool_requests = await DBOS.get_event_async(
            wf_handle.workflow_id, 'deferred_tool_requests', timeout_seconds=1
        )
        if deferred_tool_requests is not None:  # pragma: no branch
            results = DeferredToolResults()
            # Approve all calls
            for tool_call in deferred_tool_requests.approvals:
                results.approvals[tool_call.tool_call_id] = True

            for tool_call in deferred_tool_requests.calls:
                results.calls[tool_call.tool_call_id] = 'Success'

            # Signal the workflow with the results.
            await DBOS.send_async(wf_handle.workflow_id, results, topic='deferred_tool_results')

    result = await wf_handle.get_result()
    assert result.output == snapshot('The file `.env` has been deleted and `test.txt` has been created successfully.')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Delete the file `.env` and create `test.txt`',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='Just call tools without asking for confirmation.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='delete_file',
                        args='{"path": ".env"}',
                        tool_call_id='call_jYdIdRZHxZTn5bWCq5jlMrJi',
                    ),
                    ToolCallPart(
                        tool_name='create_file',
                        args='{"path": "test.txt"}',
                        tool_call_id='call_TmlTVWQbzrXCZ4jNsCVNbNqu',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=71,
                    output_tokens=46,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name=IsStr(),
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': IsDatetime(),
                },
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='delete_file',
                        content=True,
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='create_file',
                        content='Success',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='Just call tools without asking for confirmation.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content='The file `.env` has been deleted and `test.txt` has been created successfully.')
                ],
                usage=RequestUsage(
                    input_tokens=133,
                    output_tokens=19,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_dbos_agent_with_hitl_tool_sync(allow_model_requests: None, dbos: DBOS):
    # Main loop for the agent, keep running until we get a final string output.
    @DBOS.workflow()
    def hitl_main_loop_sync(prompt: str) -> AgentRunResult[str | DeferredToolRequests]:
        messages: list[ModelMessage] = [ModelRequest.user_text_prompt(prompt)]
        deferred_tool_results: DeferredToolResults | None = None
        while True:
            result = hitl_dbos_agent.run_sync(message_history=messages, deferred_tool_results=deferred_tool_results)
            messages = result.all_messages()

            if isinstance(result.output, DeferredToolRequests):
                deferred_tool_requests = result.output
                # Set deferred_tool_requests as a DBOS workflow event, so the external functions can see it.
                DBOS.set_event('deferred_tool_requests', deferred_tool_requests)

                # Wait for the deferred tool requests to be handled externally.
                deferred_tool_results = DBOS.recv('deferred_tool_results', timeout_seconds=30)
            else:
                return result

    wf_handle = DBOS.start_workflow(hitl_main_loop_sync, 'Delete the file `.env` and create `test.txt`')

    while True:
        time.sleep(1)
        status = wf_handle.get_status()
        if status.status == 'SUCCESS':
            break

        # Wait and check if the workflow has set a deferred tool request event.
        deferred_tool_requests = DBOS.get_event(wf_handle.workflow_id, 'deferred_tool_requests', timeout_seconds=1)
        if deferred_tool_requests is not None:  # pragma: no branch
            results = DeferredToolResults()
            # Approve all calls
            for tool_call in deferred_tool_requests.approvals:
                results.approvals[tool_call.tool_call_id] = True

            for tool_call in deferred_tool_requests.calls:
                results.calls[tool_call.tool_call_id] = 'Success'

            # Signal the workflow with the results.
            DBOS.send(wf_handle.workflow_id, results, topic='deferred_tool_results')

    result = wf_handle.get_result()
    assert result.output == snapshot('The file `.env` has been deleted and `test.txt` has been created successfully.')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Delete the file `.env` and create `test.txt`',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='Just call tools without asking for confirmation.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='delete_file',
                        args='{"path": ".env"}',
                        tool_call_id='call_jYdIdRZHxZTn5bWCq5jlMrJi',
                    ),
                    ToolCallPart(
                        tool_name='create_file',
                        args='{"path": "test.txt"}',
                        tool_call_id='call_TmlTVWQbzrXCZ4jNsCVNbNqu',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=71,
                    output_tokens=46,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name=IsStr(),
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': IsDatetime(),
                },
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='delete_file',
                        content=True,
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='create_file',
                        content='Success',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='Just call tools without asking for confirmation.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content='The file `.env` has been deleted and `test.txt` has been created successfully.')
                ],
                usage=RequestUsage(
                    input_tokens=133,
                    output_tokens=19,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


# Test model retry

model_retry_agent = Agent(model, name='model_retry_agent')


@model_retry_agent.tool_plain
@DBOS.step()
def get_weather_in_city(city: str) -> str:
    if city != 'Mexico City':
        raise ModelRetry('Did you mean Mexico City?')
    return 'sunny'


model_retry_dbos_agent = DBOSAgent(model_retry_agent)


async def test_dbos_agent_with_model_retry(allow_model_requests: None, dbos: DBOS):
    result = await model_retry_dbos_agent.run('What is the weather in CDMX?')
    assert result.output == snapshot('The weather in Mexico City is currently sunny.')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather in CDMX?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_weather_in_city',
                        args='{"city":"CDMX"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(
                    input_tokens=47,
                    output_tokens=17,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': IsDatetime(),
                },
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Did you mean Mexico City?',
                        tool_name='get_weather_in_city',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_weather_in_city',
                        args='{"city":"Mexico City"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(
                    input_tokens=87,
                    output_tokens=17,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': IsDatetime(),
                },
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather_in_city',
                        content='sunny',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The weather in Mexico City is currently sunny.')],
                usage=RequestUsage(
                    input_tokens=116,
                    output_tokens=10,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


class CustomModelSettings(ModelSettings, total=False):
    custom_setting: str


def return_settings(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(str(agent_info.model_settings))])


model_settings = CustomModelSettings(max_tokens=123, custom_setting='custom_value')
return_settings_model = FunctionModel(return_settings, settings=model_settings)

settings_agent = Agent(return_settings_model, name='settings_agent')
settings_dbos_agent = DBOSAgent(settings_agent)


async def test_custom_model_settings(allow_model_requests: None, dbos: DBOS):
    result = await settings_dbos_agent.run('Give me those settings')
    assert result.output == snapshot("{'max_tokens': 123, 'custom_setting': 'custom_value'}")


def return_mcp_instructions(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(agent_info.instructions or '')])


class _TestDBOSMCPToolset(DBOSMCPToolsetBase[int]):
    @property
    def _cache_tools(self) -> bool:
        return False  # pragma: no cover

    def tool_for_tool_def(self, tool_def: ToolDefinition) -> ToolsetTool[int]:
        raise AssertionError('tool_for_tool_def should not be invoked in this test')  # pragma: no cover


_uninit_instructions_toolset = _TestDBOSMCPToolset(
    MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), include_instructions=True),
    step_name_prefix='coverage_test',
    step_config={},
)


async def test_dbos_mcp_toolset_get_instructions_falls_back_to_step(dbos: DBOS):
    """When the MCP server isn't initialized locally, DBOS wrapper fetches instructions via a step."""
    run_context = RunContext(deps=0, model=TestModel(), usage=RunUsage())

    instructions = await _uninit_instructions_toolset.get_instructions(run_context)
    assert instructions == InstructionPart(content='Be a helpful assistant.', dynamic=False)


# Exercises the `DBOSMCPToolset` wrapper's `get_instructions` step path with a real `MCPToolset`.
mcptoolset_instructions_agent = Agent(
    FunctionModel(return_mcp_instructions),
    name='mcptoolset_instructions_agent',
    toolsets=[
        MCPToolset(
            StdioTransport(command='python', args=['-m', 'tests.mcp_server']),
            include_instructions=True,
            id='mcp',
        )
    ],
)
mcptoolset_instructions_dbos_agent = DBOSAgent(mcptoolset_instructions_agent)


async def test_dbos_mcptoolset_instructions_propagate(dbos: DBOS):
    """`MCPToolset` instructions propagate through the `DBOSMCPToolset` wrapper."""
    result = await mcptoolset_instructions_dbos_agent.run('Use MCP instructions')
    assert result.output == snapshot('Be a helpful assistant.')


def test_dbosify_mcptoolset_dispatches_to_dbosmcptoolset():
    """`DBOSAgent` wraps `MCPToolset` in `DBOSMCPToolset`."""
    from pydantic_ai.durable_exec.dbos._mcp_toolset import DBOSMCPToolset

    toolset = MCPToolset('https://example.com/mcp', id='test_dispatch')
    agent = Agent(model=model, name='dispatch_agent', toolsets=[toolset])
    dbos_agent = DBOSAgent(agent)
    wrapped = next(ts for ts in dbos_agent._toolsets if isinstance(ts, DBOSMCPToolset))  # pyright: ignore[reportPrivateUsage]
    assert wrapped.wrapped is toolset


async def test_dbos_mcptoolset_returns_cached_tool_defs(dbos: DBOS):
    """When the run's tool-defs cache is populated, `DBOSMCPToolset.get_tools` returns from it without invoking the step."""
    from pydantic_ai.durable_exec.dbos._mcp_toolset import DBOSMCPToolset

    inner = MCPToolset('https://example.com/mcp', id='cache_return_test')
    wrapper = DBOSMCPToolset(inner, step_name_prefix='cache_return_test', step_config={})
    run_context = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    run_context._mcp_tool_defs_cache['cache_return_test'] = {  # pyright: ignore[reportPrivateUsage]
        'foo': ToolDefinition(name='foo', parameters_json_schema={'type': 'object'}),
    }

    tools = await wrapper.get_tools(run_context)
    assert list(tools.keys()) == ['foo']
    # Returned ToolsetTool wraps the cached `ToolDefinition` via `tool_for_tool_def` on the wrapped MCPToolset.
    assert tools['foo'].tool_def.name == 'foo'


def _call_mcp_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Two model steps: call an MCP tool on the first request, return text on the second.

    Two model requests means `get_tools` runs twice on the MCP toolset within one run, so the
    run-scoped cache (and whether it schedules a step each time) is exercised.
    """
    tool_returned = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
    if tool_returned:
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('get_weather_forecast', {'location': 'Mexico City'})])


mcp_replay_agent = Agent(
    FunctionModel(_call_mcp_then_finish),
    name='mcp_replay_agent',
    toolsets=[MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='mcp', init_timeout=20)],
)
mcp_replay_dbos_agent = DBOSAgent(mcp_replay_agent)


async def test_dbos_mcp_get_tools_recorded_independently_per_run(allow_model_requests: None, dbos: DBOS):
    """#5875 regression: DBOS `get_tools` step scheduling must depend only on the workflow's own history.

    The run-scoped tool-defs cache collapses the per-request `get_tools` calls to a single recorded
    step within a run (the #4331 win). Crucially the cache lives on the run, not on the process-shared
    wrapper instance: two workflows executed back-to-back in the same worker process each record their
    own `get_tools` step at the front. With the old instance-level cache, run 2 would read run 1's warm
    cache and record ZERO `get_tools` steps — a different recorded step sequence than a cold run, which
    is what breaks replay determinism on recovery (`DBOSUnexpectedStepError`).
    """
    get_tools_step = 'mcp_replay_agent__mcp_server__mcp.get_tools'

    async def run_and_list_steps() -> list[str]:
        wfid = str(uuid.uuid4())
        with SetWorkflowID(wfid):
            result = await mcp_replay_dbos_agent.run('hello')
        assert result.output == 'done'
        return [step['function_name'] for step in await dbos.list_workflow_steps_async(wfid)]

    run1_steps = await run_and_list_steps()
    run2_steps = await run_and_list_steps()

    # Within a run, the run-scoped cache collapses both requests' `get_tools` calls to one step, at the front.
    assert run1_steps.count(get_tools_step) == 1
    assert run1_steps[0] == get_tools_step
    # Run 2 records `get_tools` independently — it does NOT inherit run 1's warm process cache (the #5875 fix).
    assert run2_steps.count(get_tools_step) == 1
    assert run2_steps[0] == get_tools_step


async def test_dbos_mcp_toolset_get_instructions_uses_local_when_initialized(dbos: DBOS):
    """When the wrapped MCP toolset is already initialized, the DBOS wrapper short-circuits and returns the local instructions."""
    run_context = RunContext(deps=0, model=TestModel(), usage=RunUsage())

    # Entering the wrapped toolset populates `_instructions` so the local fast-path returns the part directly.
    async with _uninit_instructions_toolset.wrapped:
        instructions = await _uninit_instructions_toolset.get_instructions(run_context)
    assert instructions == InstructionPart(content='Be a helpful assistant.', dynamic=False)


def test_dbos_mcp_wrapper_visit_and_replace():
    """DBOS MCP wrapper toolsets should not be replaced by visit_and_replace."""
    toolsets = mcptoolset_instructions_dbos_agent._toolsets  # pyright: ignore[reportPrivateUsage]
    dbos_mcp_toolsets = [ts for ts in toolsets if isinstance(ts, DBOSMCPToolset)]
    assert len(dbos_mcp_toolsets) >= 1

    dbos_mcp_toolset = dbos_mcp_toolsets[0]

    # visit_and_replace should return self for DBOS wrappers
    result = dbos_mcp_toolset.visit_and_replace(lambda t: FunctionToolset(id='replaced'))
    assert result is dbos_mcp_toolset


# ==========================================
# DBOSDurability capability tests
# ==========================================


def _durability_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Simple model function for durability tests."""
    for msg in reversed(messages):  # pragma: no branch - first message carries the prompt
        for part in msg.parts:  # pragma: no branch - first part is the UserPromptPart
            if isinstance(part, UserPromptPart):  # pragma: no branch - same reason
                return ModelResponse(parts=[TextPart(content=f'Echo: {part.content}')])
    return ModelResponse(parts=[TextPart(content='no prompt')])  # pragma: no cover


_durability_fn_model = FunctionModel(_durability_model_fn)

# DBOSDurability must be created after DBOS.launch() (in fixture), but since the module-level
# agents are created at import time before DBOS is initialized, we use a fixture-based approach.


async def test_dbos_durability_simple_agent(dbos: DBOS) -> None:
    """DBOSDurability routes model requests through DBOS steps."""
    agent = Agent(_durability_fn_model, name='durability_simple', capabilities=[DBOSDurability()])

    @DBOS.workflow()
    async def run_durable_agent() -> str:
        result = await agent.run('Hello DBOS')
        return result.output

    output = await run_durable_agent()
    assert output == 'Echo: Hello DBOS'


async def test_dbos_durability_auto_wraps_run_as_workflow(dbos: DBOS) -> None:
    """`agent.run` outside any workflow auto-wraps into a DBOS workflow.

    Without DBOSDurability, calling agent.run() directly wouldn't produce any DBOS
    workflow record. With it, steps inside the run get recorded under an auto-spawned
    workflow ID — verified by listing the workflow's steps after the run.
    """
    agent = Agent(_durability_fn_model, name='durability_auto', capabilities=[DBOSDurability()])

    wfid = str(uuid.uuid4())
    with SetWorkflowID(wfid):
        result = await agent.run('Auto-wrapped')

    assert result.output == 'Echo: Auto-wrapped'
    steps = await dbos.list_workflow_steps_async(wfid)
    step_names = [step['function_name'] for step in steps]
    assert 'durability_auto__model.request' in step_names


def test_dbos_durability_auto_wraps_run_sync_as_workflow(dbos: DBOS) -> None:
    """`agent.run_sync` outside any workflow auto-wraps into a DBOS workflow."""
    agent = Agent(_durability_fn_model, name='durability_auto_sync', capabilities=[DBOSDurability()])

    wfid = str(uuid.uuid4())
    with SetWorkflowID(wfid):
        result = agent.run_sync('Sync auto-wrapped')

    assert result.output == 'Echo: Sync auto-wrapped'
    steps = asyncio.get_event_loop().run_until_complete(dbos.list_workflow_steps_async(wfid))
    step_names = [step['function_name'] for step in steps]
    assert 'durability_auto_sync__model.request' in step_names


async def test_dbos_durability_parallel_mode_applies_inside_run(dbos: DBOS) -> None:
    """The configured parallel-execution mode is active inside the auto-wrapped run."""
    from pydantic_ai import tool_manager as _tm

    captured: list[str] = []

    def _capture_mode_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured.append(_tm._parallel_execution_mode_ctx_var.get())  # pyright: ignore[reportPrivateUsage]
        return _durability_model_fn(messages, info)

    capture_model = FunctionModel(_capture_mode_fn)
    agent = Agent(
        capture_model,
        name='durability_parallel',
        capabilities=[DBOSDurability(parallel_execution_mode='sequential')],
    )

    await agent.run('measure mode')
    assert captured == ['sequential']


async def test_dbos_durability_outside_workflow() -> None:
    """DBOSDurability is transparent outside a DBOS workflow.

    `agent.run` and `agent.run_sync` auto-wrap into a workflow, so to exercise the
    truly transparent path we go through `iter`, which can't be cleanly decorated
    with `@DBOS.workflow` and stays as plain code outside any workflow.
    """
    agent = Agent(_durability_fn_model, name='durability_outside', capabilities=[DBOSDurability()])

    async with agent.iter('Hello outside') as run:
        async for _ in run:
            pass
    assert run.result is not None
    assert run.result.output == 'Echo: Hello outside'


async def test_dbos_durability_step_verification(dbos: DBOS) -> None:
    """Verify that model requests become DBOS steps."""
    agent = Agent(_durability_fn_model, name='durability_steps', capabilities=[DBOSDurability()])

    wfid = str(uuid.uuid4())

    @DBOS.workflow()
    async def run_durable_agent() -> str:
        result = await agent.run('verify steps')
        return result.output

    with SetWorkflowID(wfid):
        await run_durable_agent()

    steps = await dbos.list_workflow_steps_async(wfid)
    step_names = [step['function_name'] for step in steps]
    assert 'durability_steps__model.request' in step_names


def test_dbos_durability_requires_agent_name() -> None:
    """DBOSDurability raises UserError when the agent has no name."""
    with pytest.raises(UserError, match='unique `name`'):
        Agent(_durability_fn_model, capabilities=[DBOSDurability()])


def test_dbos_durability_requires_concrete_model() -> None:
    """DBOSDurability raises UserError when the agent has no concrete model."""
    with pytest.raises(UserError, match='concrete `model`'):
        Agent('openai:gpt-4o', name='needs_concrete', defer_model_check=True, capabilities=[DBOSDurability()])


def test_dbos_durability_get_ordering() -> None:
    """DBOSDurability declares innermost ordering."""
    from pydantic_ai.capabilities.abstract import CapabilityOrdering

    durability = DBOSDurability()
    ordering = durability.get_ordering()
    assert ordering == CapabilityOrdering(position='innermost')


def test_dbos_durability_get_serialization_name() -> None:
    """DBOSDurability is not spec-serializable."""
    assert DBOSDurability.get_serialization_name() is None


def test_dbos_durability_idempotent_for_agent() -> None:
    """Binding a second `DBOSDurability` to the same agent doesn't re-wrap `agent.run`.

    Without the idempotency guard, re-binding would stack workflow decorators
    on top of each other.
    """
    agent = Agent(_durability_fn_model, name='dbos_idempotent_test', capabilities=[DBOSDurability()])
    first_run = agent.run

    # Re-binding another DBOSDurability should leave agent.run untouched.
    DBOSDurability().for_agent(agent)
    assert agent.run is first_run


async def _durability_stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    for msg in reversed(messages):  # pragma: no branch - first message carries the prompt
        for part in msg.parts:  # pragma: no branch - first part is the UserPromptPart
            if isinstance(part, UserPromptPart):  # pragma: no branch - same reason
                yield f'Echo: {part.content}'
                return
    yield 'no prompt'  # pragma: no cover


async def test_dbos_durability_streaming_in_workflow(dbos: DBOS) -> None:
    """DBOSDurability routes streaming requests through DBOS steps when event_stream_handler is set."""
    events_received: list[Any] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[Any]) -> None:
        async for event in stream:
            events_received.append(event)

    stream_model = FunctionModel(_durability_model_fn, stream_function=_durability_stream_fn)
    agent = Agent(
        stream_model,
        name='durability_streaming',
        capabilities=[DBOSDurability(event_stream_handler=handler)],
    )

    wfid = str(uuid.uuid4())

    @DBOS.workflow()
    async def run_durable_streaming_agent() -> str:
        result = await agent.run('Hello streaming')
        return result.output

    with SetWorkflowID(wfid):
        output = await run_durable_streaming_agent()

    assert output == 'Echo: Hello streaming'

    steps = await dbos.list_workflow_steps_async(wfid)
    step_names = [step['function_name'] for step in steps]
    assert 'durability_streaming__model.request_stream' in step_names


async def _chunks_stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'Stream'
    yield 'ed '
    yield 'response'


async def test_dbos_durability_process_event_stream_fires_live_inside_step(dbos: DBOS) -> None:
    """ProcessEventStream (outer capability) sees live events emitted inside a DBOS step.

    With in-step chain firing, the capability's handler runs against the real streamed
    response — so multiple PartDeltaEvents come through (one per chunk). If the chain fired
    on the replayed stream outside the step instead, ProcessEventStream would see a single
    synthetic delta with the full text.
    """
    from pydantic_ai.capabilities import ProcessEventStream
    from pydantic_ai.messages import PartDeltaEvent, TextPartDelta

    events_received: list[AgentStreamEvent] = []

    async def collect(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events_received.append(event)

    stream_model = FunctionModel(_durability_model_fn, stream_function=_chunks_stream_fn)
    agent = Agent(
        stream_model,
        name='durability_process_stream',
        capabilities=[ProcessEventStream(collect), DBOSDurability()],
    )

    @DBOS.workflow()
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
    # The 'Stream' / 'ed ' / 'response' chunks: first becomes the PartStartEvent's text,
    # subsequent chunks are deltas. Synthetic replay of the final response would collapse
    # everything into a single delta with the full text.
    assert delta_events == ['ed ', 'response']


async def test_dbos_durability_runtime_handler_receives_buffered_events(dbos: DBOS) -> None:
    """A per-run `event_stream_handler` passed to `agent.run()` inside a DBOS workflow

    receives the events captured inside the step (rather than being silently dropped).
    The buffered replay preserves real granular deltas — the per-run handler sees the
    same multi-chunk stream the construction-time handler would see.
    """
    from pydantic_ai.messages import PartDeltaEvent, TextPartDelta

    events_received: list[AgentStreamEvent] = []

    async def runtime_collect(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events_received.append(event)

    stream_model = FunctionModel(_durability_model_fn, stream_function=_chunks_stream_fn)
    agent = Agent(
        stream_model,
        name='durability_runtime_handler',
        capabilities=[DBOSDurability()],
    )

    @DBOS.workflow()
    async def run_durable_agent() -> str:
        result = await agent.run('Hello', event_stream_handler=runtime_collect)
        return result.output

    output = await run_durable_agent()
    assert output == 'Streamed response'

    # The runtime handler got real granular deltas (one PartDeltaEvent per chunk),
    # not a single synthetic delta with the full text.
    delta_events = [
        e.delta.content_delta
        for e in events_received
        if isinstance(e, PartDeltaEvent) and isinstance(e.delta, TextPartDelta)
    ]
    assert delta_events == ['ed ', 'response']


async def test_dbos_durability_mcp_toolset_wrapping(dbos: DBOS) -> None:
    """DBOSDurability discovers MCPToolset and creates DBOS wrappers."""
    from pydantic_ai.durable_exec.dbos._mcp_toolset import DBOSMCPToolset

    mcp_toolset = MCPToolset(
        StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='my_mcp', init_timeout=20
    )
    agent = Agent(
        _durability_fn_model,
        name='durability_mcp',
        toolsets=[mcp_toolset],
        capabilities=[DBOSDurability()],
    )
    bound = DBOSDurability.from_agent(agent)
    assert bound is not None

    # The capability should have stored a DBOS wrapper keyed by the toolset id
    assert 'my_mcp' in bound._dbos_toolsets_by_id  # pyright: ignore[reportPrivateUsage]
    assert isinstance(bound._dbos_toolsets_by_id['my_mcp'], DBOSMCPToolset)  # pyright: ignore[reportPrivateUsage]


async def test_dbos_durability_get_wrapper_toolset_with_mcp(dbos: DBOS) -> None:
    """DBOSDurability.get_wrapper_toolset replaces MCP toolsets by id."""
    from pydantic_ai.durable_exec.dbos._mcp_toolset import DBOSMCPToolset

    mcp_toolset = MCPToolset(
        StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='swap_mcp', init_timeout=20
    )
    agent = Agent(
        _durability_fn_model,
        name='durability_swap',
        toolsets=[mcp_toolset],
        capabilities=[DBOSDurability()],
    )
    bound = DBOSDurability.from_agent(agent)
    assert bound is not None

    assert 'swap_mcp' in bound._dbos_toolsets_by_id  # pyright: ignore[reportPrivateUsage]

    # get_wrapper_toolset should replace the original MCP toolset with the DBOS wrapper
    replaced = bound.get_wrapper_toolset(mcp_toolset)
    assert replaced is not None
    assert isinstance(replaced, DBOSMCPToolset)


async def test_dbos_durability_allows_runtime_function_toolset(dbos: DBOS) -> None:
    """A `FunctionToolset` added per-run is allowed and executes inline, like on `DBOSAgent`."""

    def call_then_answer(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('runtime_tool', {}, tool_call_id='call-1')])

    agent = Agent(
        FunctionModel(call_then_answer), name='durability_runtime_fn', capabilities=[DBOSDurability()]
    )

    result = await agent.run('Call the runtime tool.', toolsets=[FunctionToolset(tools=[runtime_tool], id='runtime_fn')])
    assert result.output == 'done'


async def test_dbos_durability_rejects_runtime_mcp_toolset(dbos: DBOS) -> None:
    """An `MCPToolset` added per-run is rejected: its I/O steps must be registered at construction."""
    agent = Agent(_durability_fn_model, name='durability_runtime_mcp', capabilities=[DBOSDurability()])

    with pytest.raises(UserError, match=r'MCPToolset cannot be passed to `run\(toolsets=\.\.\.\)` at runtime with DBOS'):
        await agent.run(
            'Hello',
            toolsets=[MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='runtime_mcp')],
        )


def test_dbos_durability_rejects_runtime_dynamic_toolset_sync(dbos: DBOS) -> None:
    """A `DynamicToolset` added per-run is rejected, on `run_sync` as well as `run`."""
    agent = Agent(_durability_fn_model, name='durability_runtime_dynamic', capabilities=[DBOSDurability()])

    with pytest.raises(
        UserError, match=r'DynamicToolset cannot be passed to `run\(toolsets=\.\.\.\)` at runtime with DBOS'
    ):
        agent.run_sync('Hello', toolsets=[DynamicToolset(lambda _: FunctionToolset(), id='runtime_dynamic')])
