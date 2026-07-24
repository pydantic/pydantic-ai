from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Generator, Iterator, Sequence
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
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    RunContext,
    RunUsage,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai._run_context import get_current_run_context
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import MCP, Capability, DynamicCapability
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    ToolFailed,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.models import ModelRequestContext, ModelResolutionContext, create_async_http_client
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RequestUsage, UsageLimits

from .conftest import IsDatetime, IsNow, IsStr

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec._toolset import unwrap_recorded_tool_call_result, wrap_tool_call_result
    from pydantic_ai.durable_exec.dbos import (
        DBOSAgent,  # pyright: ignore[reportDeprecated]
        DBOSDurability,
        DBOSModel,
        StepConfig,
    )
    from pydantic_ai.durable_exec.dbos._dynamic_toolset import dbosify_dynamic_toolset
    from pydantic_ai.durable_exec.dbos._mcp_toolset import DBOSMCPToolset, dbosify_mcp_toolset
    from pydantic_ai.toolsets.external import TOOL_SCHEMA_VALIDATOR

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
from pydantic_ai.capabilities import ProcessEventStream, ResolveModelId, SelectModel, Toolset
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._inline_snapshot import snapshot
from .continuation_utils import ScriptedContinuationModel, StreamSegment, scripted_response

# `DBOSAgent` is deprecated in favor of `capabilities=[DBOSDurability(...)]`.
# These tests exercise the wrapper-agent path on purpose; suppress the warning here
# rather than globally in `pyproject.toml`. The `pytestmark` entry below covers warnings
# emitted *inside* test functions; the `filterwarnings` call below covers warnings emitted
# at module import time (e.g. `simple_dbos_agent = DBOSAgent(...)`).
warnings.filterwarnings('ignore', message='`DBOSAgent` is deprecated', category=PydanticAIDeprecationWarning)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.xdist_group(name='dbos'),
    pytest.mark.filterwarnings('ignore:`DBOSAgent` is deprecated:pydantic_ai._warnings.PydanticAIDeprecationWarning'),
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
simple_dbos_agent = DBOSAgent(simple_agent)  # pyright: ignore[reportDeprecated]


def test_dbos_agent_construction_warns_deprecated() -> None:
    """The `DBOSAgent` deprecation fires at runtime; the module-level filters only suppress it."""
    with pytest.warns(PydanticAIDeprecationWarning, match='`DBOSAgent` is deprecated'):
        DBOSAgent(Agent(TestModel(), name='dbos_agent_deprecation_probe'))  # pyright: ignore[reportDeprecated]


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
complex_dbos_agent = DBOSAgent(complex_agent, event_stream_handler=event_stream_handler)  # pyright: ignore[reportDeprecated]
seq_complex_dbos_agent = DBOSAgent(  # pyright: ignore[reportDeprecated]
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
runtime_handler_stream_dbos_agent = DBOSAgent(runtime_handler_stream_agent)  # pyright: ignore[reportDeprecated]


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
    dbos_agent = DBOSAgent(agent, event_stream_handler=runtime_event_stream_handler)  # pyright: ignore[reportDeprecated]
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
        DBOSAgent(simple_agent)  # pyright: ignore[reportDeprecated]


async def test_agent_without_name():
    with pytest.raises(
        UserError,
        match=re.escape(
            "An agent needs to have a unique `name` in order to be used with DBOS. The name will be used to identify the agent's workflows and steps."
        ),
    ):
        DBOSAgent(Agent())  # pyright: ignore[reportDeprecated]


async def test_agent_without_model():
    with pytest.raises(
        UserError,
        match=re.escape(
            'An agent needs to have a `model` in order to be used with DBOS, it cannot be set at agent run time.'
        ),
    ):
        DBOSAgent(Agent(name='test_agent'))  # pyright: ignore[reportDeprecated]


async def test_toolset_without_id():
    # Note: this is allowed in DBOS because we don't wrap the tools automatically in a workflow. It's up to the user to define the tools as DBOS steps if they want to use them as steps in a workflow.
    DBOSAgent(Agent(model=model, name='test_agent', toolsets=[FunctionToolset()]))  # pyright: ignore[reportDeprecated]


async def test_mcp_toolset_without_id():
    # Unlike `FunctionToolset`, an `MCPToolset` is wrapped in a `DBOSMCPToolset` whose step names and per-run
    # tool-defs cache key both derive from the toolset's `id`; without one, two id-less MCP toolsets would
    # collide on both. Require an `id` up front, like Temporal does for all leaf toolsets.
    with pytest.raises(
        UserError,
        match=re.escape(
            'MCP toolsets need to have a unique `id` in order to be used with DBOS. '
            "The ID will be used to identify the MCP server's steps within the workflow."
        ),
    ):
        DBOSAgent(Agent(model=model, name='test_agent', toolsets=[MCPToolset('https://example.com/mcp')]))  # pyright: ignore[reportDeprecated]


async def test_capability_contributed_toolset_id_from_capability():
    """A capability's `id` flows to its contributed leaf toolset, so a capability combined with a
    local MCP server can be used under DBOS instead of tripping the id-less-MCP guard. An `MCP` with
    no explicit `id` derives one from its URL, which is what lets the contributed leaf pass the guard.

    This isn't a VCR test: it inspects the constructed toolset tree and DBOS registration during local
    agent construction, before any model or MCP request, so there's no network round-trip to record.

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
    # Previously raised `UserError` from the DBOS id-less-MCP guard because the contributed MCP leaf
    # had `id=None`; it now derives `mcp.example.com-api` from the URL, so construction succeeds.
    dbos_agent = DBOSAgent(agent)  # pyright: ignore[reportDeprecated]

    leaves: list[AbstractToolset[object]] = []
    for toolset in dbos_agent.toolsets:
        toolset.apply(leaves.append)
    # The contributed MCP leaf carries the URL-derived id; the `billing` function toolset carries the
    # capability id.
    assert any(isinstance(ts, MCPToolset) and ts.id == 'mcp.example.com-api' for ts in leaves)
    assert any(isinstance(ts, FunctionToolset) and ts.id == 'billing' for ts in leaves)


async def test_capability_contributed_toolsets_with_colliding_derived_id():
    """Two genuinely different MCP servers whose URLs derive the same id would silently collide on the
    per-run tool-defs cache key under DBOS (the second server returning the first's cached tools). The
    DBOS wrapper guards against duplicate leaf ids at construction, telling the user to set explicit ids.

    Both `MCP(url=...)` capabilities leave `cap.id=None` (so the agent-level capability-id uniqueness
    check passes), yet both derive `a.com-api` from their URLs' host + last path segment.

    This isn't a VCR test: the collision is rejected during local `DBOSAgent` construction, before any
    model or MCP request, so there's no network round-trip to record.
    """
    with pytest.raises(
        UserError,
        match=re.escape(
            'MCP toolsets need to have a unique `id` in order to be used with DBOS, '
            "but more than one leaf toolset uses the id 'a.com-api'. "
            "The ID identifies the MCP server's steps within the workflow, so duplicates would collide. "
            'Set a distinct `id` on each `MCPToolset` (or the `Capability`/`MCP` that contributes it) to disambiguate them.'
        ),
    ):
        DBOSAgent(  # pyright: ignore[reportDeprecated]
            Agent(
                model,
                name='colliding_capability_agent',
                capabilities=[MCP(url='https://a.com/api'), MCP(url='https://a.com/v2/api')],
            )
        )


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


async def test_dbos_cancel_suspended_response_runs_in_step(allow_model_requests: None, dbos: DBOS):
    """`DBOSModel.cancel_suspended_response` must run as a DBOS step, not inline in the workflow.

    The provider teardown that cancels a server-side suspended/background job is a raw HTTP call;
    wrapping it as a step makes it durable, retried, and recorded rather than running unrecorded
    inside the workflow.
    """
    cancelled: list[ModelResponse] = []

    class RecordingModel(TestModel):
        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            cancelled.append(response)

    dbos_model = DBOSModel(
        RecordingModel(),
        step_name_prefix='cancel_suspended',
        step_config={},
        get_event_stream_handler=lambda: None,
    )
    response = ModelResponse(parts=[TextPart('paused')], state='suspended')

    wfid = str(uuid.uuid4())

    @DBOS.workflow()
    async def cancel_in_workflow() -> None:
        await dbos_model.cancel_suspended_response(response)

    with SetWorkflowID(wfid):
        await cancel_in_workflow()

    steps = await dbos.list_workflow_steps_async(wfid)
    assert [step['function_name'] for step in steps] == snapshot(['cancel_suspended__model.cancel_suspended_response'])
    assert cancelled == [response]


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
    dbos_agent = DBOSAgent(agent)  # pyright: ignore[reportDeprecated]

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
    dbos_agent = DBOSAgent(agent)  # pyright: ignore[reportDeprecated]

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
    unserializable_deps_dbos_agent = DBOSAgent(unserializable_deps_agent)  # pyright: ignore[reportDeprecated]
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


dynamic_dbos_agent = DBOSAgent(dynamic_agent)  # pyright: ignore[reportDeprecated]


def test_dbos_agent_explicit_run_id(dbos: DBOS):
    """A pre-minted `run_id=` is preserved through DBOSAgent.run_sync."""
    agent = Agent(TestModel(custom_output_text='ok'), name='run_id_dbos_agent')
    dbos_agent = DBOSAgent(agent)  # pyright: ignore[reportDeprecated]

    result = dbos_agent.run_sync('Hello', run_id='run-from-dbos')
    assert result.run_id == 'run-from-dbos'
    assert all(m.run_id == 'run-from-dbos' for m in result.all_messages())


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


hitl_dbos_agent = DBOSAgent(hitl_agent)  # pyright: ignore[reportDeprecated]


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


model_retry_dbos_agent = DBOSAgent(model_retry_agent)  # pyright: ignore[reportDeprecated]


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


tool_failed_agent = Agent(TestModel(call_tools=['failing_tool']), name='tool_failed_agent')


@tool_failed_agent.tool_plain
@DBOS.step()
def failing_tool() -> str:
    raise ToolFailed('Disk full')


tool_failed_dbos_agent = DBOSAgent(tool_failed_agent)  # pyright: ignore[reportDeprecated]


async def test_dbos_agent_with_tool_failed(dbos: DBOS):
    result = await tool_failed_dbos_agent.run('Call the failing tool')
    tool_returns = [
        (part.tool_name, part.content, part.outcome)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]

    assert tool_returns == [('failing_tool', 'Disk full', 'failed')]


class CustomModelSettings(ModelSettings, total=False):
    custom_setting: str


def return_settings(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(str(agent_info.model_settings))])


model_settings = CustomModelSettings(max_tokens=123, custom_setting='custom_value')
return_settings_model = FunctionModel(return_settings, settings=model_settings)

settings_agent = Agent(return_settings_model, name='settings_agent')
settings_dbos_agent = DBOSAgent(settings_agent)  # pyright: ignore[reportDeprecated]


async def test_custom_model_settings(allow_model_requests: None, dbos: DBOS):
    result = await settings_dbos_agent.run('Give me those settings')
    assert result.output == snapshot("{'max_tokens': 123, 'custom_setting': 'custom_value'}")


def return_mcp_instructions(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(agent_info.instructions or '')])


_uninit_instructions_toolset = dbosify_mcp_toolset(
    MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), include_instructions=True),
    step_name_prefix='coverage_test',
    step_config={},
)


async def test_dbos_mcp_toolset_get_instructions_falls_back_to_step(dbos: DBOS):
    """When the MCP server isn't initialized locally, DBOS wrapper fetches instructions via a step."""
    run_context = RunContext(deps=0, model=TestModel(), usage=RunUsage())

    instructions = await _uninit_instructions_toolset.get_instructions(run_context)
    assert instructions == InstructionPart(content='Be a helpful assistant.', dynamic=False)


async def test_dbos_mcp_toolset_get_instructions_uses_step_when_server_warm(dbos: DBOS):
    """A warm in-process MCP server must not short-circuit the `get_instructions` step (#5884).

    Whether the step runs must not depend on process warmth: skipping it when the wrapped
    server already holds instructions locally makes the durable schedule diverge between the
    original execution and replay/recovery in a differently-warm process.

    Not a VCR test: what's asserted is *which side* serves the instructions (the DBOS step vs
    the warm server's local cache) — invisible to a network cassette, since both paths issue
    the same MCP traffic.
    """
    inner = MCPToolset(
        StdioTransport(command='python', args=['-m', 'tests.mcp_server']),
        include_instructions=True,
        id='warm_instructions_test',
    )
    wrapper = dbosify_mcp_toolset(inner, step_name_prefix='warm_instructions_test', step_config={})
    run_context = RunContext(deps=0, model=TestModel(), usage=RunUsage())

    step_calls: list[str] = []
    original_operation = wrapper._get_instructions_operation  # pyright: ignore[reportPrivateUsage]
    assert original_operation is not None

    async def recording_operation(
        ctx: RunContext[object],
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        step_calls.append('step')
        return await original_operation(ctx)

    wrapper._get_instructions_operation = recording_operation  # pyright: ignore[reportPrivateUsage]

    # Hold the wrapped server entered so it can serve instructions locally (warm process).
    async with inner:
        assert await inner.get_instructions(run_context) is not None
        instructions = await wrapper.get_instructions(run_context)

    assert instructions == InstructionPart(content='Be a helpful assistant.', dynamic=False)
    assert step_calls == ['step']


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
mcptoolset_instructions_dbos_agent = DBOSAgent(mcptoolset_instructions_agent)  # pyright: ignore[reportDeprecated]


async def test_dbos_mcptoolset_instructions_propagate(dbos: DBOS):
    """`MCPToolset` instructions propagate through the `DBOSMCPToolset` wrapper."""
    result = await mcptoolset_instructions_dbos_agent.run('Use MCP instructions')
    assert result.output == snapshot('Be a helpful assistant.')


def test_dbosify_mcptoolset_dispatches_to_dbosmcptoolset():
    """`DBOSAgent` wraps `MCPToolset` in `DBOSMCPToolset`."""
    from pydantic_ai.durable_exec.dbos._mcp_toolset import DBOSMCPToolset

    toolset = MCPToolset('https://example.com/mcp', id='test_dispatch')
    agent = Agent(model=model, name='dispatch_agent', toolsets=[toolset])
    dbos_agent = DBOSAgent(agent)  # pyright: ignore[reportDeprecated]
    wrapped = next(ts for ts in dbos_agent._toolsets if isinstance(ts, DBOSMCPToolset))  # pyright: ignore[reportPrivateUsage]
    assert wrapped.wrapped is toolset


async def test_dbos_mcptoolset_returns_cached_tool_defs(dbos: DBOS):
    """When the run's tool-defs cache is populated, `DBOSMCPToolset.get_tools` returns from it without invoking the step."""
    from pydantic_ai.durable_exec.dbos._mcp_toolset import dbosify_mcp_toolset

    inner = MCPToolset('https://example.com/mcp', id='cache_return_test')
    wrapper = dbosify_mcp_toolset(inner, step_name_prefix='cache_return_test', step_config={})
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
mcp_replay_dbos_agent = DBOSAgent(mcp_replay_agent)  # pyright: ignore[reportDeprecated]


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


async def test_dbos_durability_registers_legacy_workflows_opt_in(dbos: DBOS) -> None:
    agent = Agent(
        _durability_fn_model,
        name='legacy_workflow_compat',
        capabilities=[DBOSDurability(register_legacy_workflows=True)],
    )
    durability = DBOSDurability.from_agent(agent)
    assert durability is not None
    assert durability._legacy_run_workflow is not None  # pyright: ignore[reportPrivateUsage]
    assert durability._legacy_run_sync_workflow is not None  # pyright: ignore[reportPrivateUsage]

    result = await durability._legacy_run_workflow('legacy')  # pyright: ignore[reportPrivateUsage]
    assert result.output == 'Echo: legacy'

    without_flag = Agent(_durability_fn_model, name='no_legacy_workflows', capabilities=[DBOSDurability()])
    without_flag_durability = DBOSDurability.from_agent(without_flag)
    assert without_flag_durability is not None
    assert without_flag_durability._legacy_run_workflow is None  # pyright: ignore[reportPrivateUsage]
    assert without_flag_durability._legacy_run_sync_workflow is None  # pyright: ignore[reportPrivateUsage]


async def test_dbos_durability_accepts_legacy_stream_step_shape(dbos: DBOS) -> None:
    response = ModelResponse(parts=[TextPart(content='legacy stream')], model_name='legacy')
    agent = Agent(TestModel(), name='legacy_stream_shape', capabilities=[DBOSDurability()])
    durability = DBOSDurability.from_agent(agent)
    assert durability is not None

    async def legacy_stream_step(*args: Any) -> ModelResponse:
        return response

    durability._request_stream_step = legacy_stream_step  # pyright: ignore[reportPrivateUsage]

    @DBOS.workflow()
    async def run_agent() -> tuple[str, list[str]]:
        async with agent.run_stream('stream') as result:
            chunks = [chunk async for chunk in result.stream_text(debounce_by=None)]
            return await result.get_output(), chunks

    assert await run_agent() == ('legacy stream', ['legacy stream'])


# Module-level like real wrapper-era handlers: `DBOSAgent.run` recorded the handler as a
# workflow input, so it had to be picklable by reference.
_legacy_handler_events: list[tuple[AgentStreamEvent, bool]] = []


async def _legacy_workflow_event_handler(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in stream:
        _legacy_handler_events.append((event, DBOS.step_id is not None))


def test_dbos_durability_legacy_run_sync_workflow(dbos: DBOS) -> None:
    """The opt-in legacy `run_sync` workflow executes the agent like `DBOSAgent.run_sync` did,
    including honoring a recorded per-run `event_stream_handler` input."""
    _legacy_handler_events.clear()
    agent = Agent(
        TestModel(custom_output_text='legacy sync'),
        name='legacy_workflow_sync_compat',
        capabilities=[DBOSDurability(register_legacy_workflows=True)],
    )
    durability = DBOSDurability.from_agent(agent)
    assert durability is not None
    result = durability._legacy_run_sync_workflow('legacy', event_stream_handler=_legacy_workflow_event_handler)  # pyright: ignore[reportPrivateUsage]
    assert result.output == 'legacy sync'
    assert _legacy_handler_events


async def test_dbos_durability_legacy_workflow_matches_wrapper_step_sequence(dbos: DBOS) -> None:
    """A legacy run delivers handler events exactly the way `DBOSAgent` did, so recovery replays.

    Wrapper-era recordings contain only model and MCP steps: model events reached the handler
    live inside the `__model.request_stream` step, and graph-level events through a direct
    workflow-level call that consumed no step. Dispatching graph events through the capability's
    `__event_stream_handler` step would insert step ids the recording doesn't have and fail
    recovery with `DBOSUnexpectedStepError`.
    """
    _legacy_handler_events.clear()

    async def handled_tool() -> str:
        return 'handled'

    agent = Agent(
        TestModel(),
        name='legacy_workflow_handler',
        tools=[handled_tool],
        capabilities=[DBOSDurability(register_legacy_workflows=True)],
    )
    durability = DBOSDurability.from_agent(agent)
    assert durability is not None

    wfid = str(uuid.uuid4())
    with SetWorkflowID(wfid):
        result = await durability._legacy_run_workflow('Hello', event_stream_handler=_legacy_workflow_event_handler)  # pyright: ignore[reportPrivateUsage]
    assert result.output

    events = [event for event, _ in _legacy_handler_events]
    assert events
    assert sum(isinstance(event, FunctionToolCallEvent) for event in events) == 1
    # Model events arrive live inside the model-request step; graph events at workflow level.
    assert all(in_step for event, in_step in _legacy_handler_events if isinstance(event, PartStartEvent))
    assert not any(in_step for event, in_step in _legacy_handler_events if isinstance(event, FunctionToolCallEvent))

    steps = await dbos.list_workflow_steps_async(wfid)
    step_names = [step['function_name'] for step in steps]
    assert 'legacy_workflow_handler__model.request_stream' in step_names
    assert 'legacy_workflow_handler__event_stream_handler' not in step_names


async def test_unwrap_recorded_tool_call_result_handles_both_generations() -> None:
    """Recovery may replay step outputs recorded before control-flow wrapping (raw) or after (wrapped).

    A unit test because a real recovery would need to kill and restart the process mid-workflow;
    the step-recorded value is exactly what this helper receives on replay.
    """

    async def raise_model_retry() -> str:
        raise ModelRetry('again')

    wrapped_result = await wrap_tool_call_result(raise_model_retry())
    with pytest.raises(ModelRetry, match='again'):
        unwrap_recorded_tool_call_result(wrapped_result)

    assert unwrap_recorded_tool_call_result('raw recorded output') == 'raw recorded output'


async def test_dbos_mcp_model_retry_crosses_step_without_engine_retries(
    dbos: DBOS, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ModelRetry` from an MCP tool crosses the step as a value; DBOS must not re-execute the call.

    Previously the raw exception failed the step, so `retries_allowed` re-ran the side-effecting
    MCP call and surfaced `DBOSMaxStepRetriesExceeded` instead of the agent's retry-prompt loop.
    """
    calls = 0
    mcp_toolset = MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='retry_mcp')

    async def raise_model_retry(
        tool_name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
    ) -> Any:
        nonlocal calls
        calls += 1
        raise ModelRetry('try again')

    monkeypatch.setattr(mcp_toolset, 'call_tool', raise_model_retry)
    durable = dbosify_mcp_toolset(
        mcp_toolset,
        step_name_prefix='retry_mcp_agent',
        step_config=StepConfig(retries_allowed=True, max_attempts=3),
    )
    run_context = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    tool = ToolsetTool(
        toolset=durable,
        tool_def=ToolDefinition(name='boom'),
        max_retries=1,
        args_validator=TOOL_SCHEMA_VALIDATOR,
    )

    @DBOS.workflow()
    async def run_workflow() -> int:
        with pytest.raises(ModelRetry, match='try again'):
            await durable.call_tool('boom', {}, run_context, tool)
        return calls

    assert await run_workflow() == 1


async def test_dbos_dynamic_tool_model_retry_crosses_step_without_engine_retries(dbos: DBOS) -> None:
    """`ModelRetry` from a `DynamicToolset` tool crosses the step as a value, like MCP and function tools."""
    calls = 0

    async def raise_model_retry() -> str:
        nonlocal calls
        calls += 1
        raise ModelRetry('try again')

    dynamic = DynamicToolset[None](lambda ctx: FunctionToolset([raise_model_retry]), id='retry_dynamic')
    durable = dbosify_dynamic_toolset(
        dynamic,
        step_name_prefix='retry_dynamic_agent',
        step_config=StepConfig(retries_allowed=True, max_attempts=3),
    )
    run_context = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    tool = ToolsetTool(
        toolset=durable,
        tool_def=ToolDefinition(name='raise_model_retry'),
        max_retries=1,
        args_validator=TOOL_SCHEMA_VALIDATOR,
    )

    @DBOS.workflow()
    async def run_workflow() -> int:
        with pytest.raises(ModelRetry, match='try again'):
            await durable.call_tool('raise_model_retry', {}, run_context, tool)
        return calls

    assert await run_workflow() == 1


async def test_dbos_mcp_step_rejects_enqueue_in_workflow(dbos: DBOS, monkeypatch: pytest.MonkeyPatch) -> None:
    """The MCP step path guards enqueue too: a `process_tool_call=` hook receives the run context."""
    mcp_toolset = MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='enqueue_mcp')

    async def enqueue_call_tool(
        tool_name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
    ) -> Any:
        ctx.enqueue('later')
        return 'done'

    monkeypatch.setattr(mcp_toolset, 'call_tool', enqueue_call_tool)
    durable = dbosify_mcp_toolset(mcp_toolset, step_name_prefix='enqueue_mcp_agent', step_config={})
    run_context = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    tool = ToolsetTool(
        toolset=durable,
        tool_def=ToolDefinition(name='hook'),
        max_retries=1,
        args_validator=TOOL_SCHEMA_VALIDATOR,
    )

    @DBOS.workflow()
    async def run_workflow() -> None:
        await durable.call_tool('hook', {}, run_context, tool)

    with pytest.raises(UserError, match='enqueued messages would be dropped'):
        await run_workflow()

    # Outside a workflow the step degrades to a plain call and enqueueing keeps working.
    outside_context = RunContext(deps=None, model=TestModel(), usage=RunUsage(), pending_messages=[])
    assert await durable.call_tool('hook', {}, outside_context, tool) == 'done'
    assert len(outside_context.pending_messages or []) == 1


async def test_dbos_dynamic_tool_rejects_enqueue_in_workflow(dbos: DBOS) -> None:
    """`ctx.enqueue()` inside a step-wrapped dynamic tool raises instead of silently dropping.

    Recovery replays the recorded step output without re-executing the tool, so in-step
    enqueued messages would be lost. Outside a workflow the step degrades to a plain call
    and enqueueing keeps working.
    """

    async def enqueue(ctx: RunContext[object]) -> str:
        ctx.enqueue('later')
        return 'done'

    agent = Agent(
        TestModel(),
        deps_type=object,
        name='dbos_dynamic_enqueue',
        toolsets=[DynamicToolset(lambda ctx: FunctionToolset([enqueue]), id='enqueue_dynamic')],
        capabilities=[DBOSDurability()],
    )

    @DBOS.workflow()
    async def run_workflow() -> None:
        await agent.run('run')

    with pytest.raises(UserError, match='enqueued messages would be dropped'):
        await run_workflow()

    await agent.run('run')


async def test_dbos_durability_parallel_mode_applies_inside_run(dbos: DBOS) -> None:
    """The configured parallel-execution mode is active for the duration of the run."""
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
    """DBOSDurability is transparent outside a DBOS workflow."""
    agent = Agent(_durability_fn_model, name='durability_outside', capabilities=[DBOSDurability()])

    result = await agent.run('Hello outside')
    assert result.output == 'Echo: Hello outside'


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


def test_dbos_durability_explicit_name_overrides_agent_name_and_supports_unnamed_agent() -> None:
    named_agent = Agent(_durability_fn_model, name='agent-name', capabilities=[DBOSDurability(name='custom')])
    bound = DBOSDurability.from_agent(named_agent)
    assert bound is not None
    assert bound.name == 'custom'

    unnamed_agent = Agent(_durability_fn_model, capabilities=[DBOSDurability(name='unnamed-custom')])
    unnamed_bound = DBOSDurability.from_agent(unnamed_agent)
    assert unnamed_bound is not None
    assert unnamed_bound.name == 'unnamed-custom'


def test_dbos_durability_requires_model() -> None:
    """DBOSDurability raises UserError when the agent has no model at all."""
    with pytest.raises(UserError, match='needs to have a `model`'):
        Agent(name='needs_model', capabilities=[DBOSDurability()])


def _dbos_alt_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='alt-response')])


# A module-level function (not a lambda): passing the instance to `agent.run(model=...)`
# makes it part of the DBOS workflow's pickled arguments.
_dbos_alt_model = FunctionModel(_dbos_alt_model_fn, model_name='alt')


async def test_dbos_durability_runtime_registered_model(dbos: DBOS) -> None:
    """A model registered in `models=` can be selected at run time, by key or instance.

    The `model_id` crosses the step boundary and the step rebuilds the model from the
    registry, so the response is produced by the selected model inside the DBOS step.
    """
    agent = Agent(
        _durability_fn_model,
        name='durability_runtime_registered',
        capabilities=[DBOSDurability(models={'alt': _dbos_alt_model})],
    )

    @DBOS.workflow()
    async def run_agent() -> tuple[str, str]:
        by_key = await agent.run('hello', model='alt')
        by_instance = await agent.run('hello', model=_dbos_alt_model)
        return by_key.output, by_instance.output

    assert await run_agent() == ('alt-response', 'alt-response')


async def test_dbos_durability_per_step_model_selector_uses_selected_model(dbos: DBOS) -> None:
    """A per-step selector's model identity, rather than the run-level model ID, crosses the step boundary."""
    durability = DBOSDurability(models={'primary': _durability_fn_model, 'alt': _dbos_alt_model})
    agent = Agent(
        'primary',
        name='durability_per_step_model_selector',
        capabilities=[SelectModel(lambda ctx: _dbos_alt_model), durability],
    )

    @DBOS.workflow()
    async def run_agent() -> str:
        return (await agent.run('hello')).output

    assert await run_agent() == 'alt-response'


async def test_dbos_durability_per_step_model_selector_preserves_alias(dbos: DBOS) -> None:
    """DBOS persists the selected registration alias instead of the resolved model's ID."""
    durability = DBOSDurability(models={'primary': _durability_fn_model, 'alt': _dbos_alt_model})
    agent = Agent(
        'primary',
        name='durability_per_step_model_selector_alias',
        capabilities=[SelectModel(lambda ctx: 'alt'), durability],
    )

    @DBOS.workflow()
    async def run_agent() -> str:
        return (await agent.run('hello')).output

    assert await run_agent() == 'alt-response'


async def test_dbos_durability_override_registered_model(dbos: DBOS) -> None:
    """A model set via `override(model=...)` round-trips the step boundary like a per-run `model=`."""
    agent = Agent(
        _durability_fn_model,
        name='durability_override_registered',
        capabilities=[DBOSDurability(models={'alt': _dbos_alt_model})],
    )

    @DBOS.workflow()
    async def run_agent() -> str:
        with agent.override(model='alt'):
            result = await agent.run('hello')
        return result.output

    assert await run_agent() == 'alt-response'


async def test_dbos_durability_unrebuildable_runtime_model_errors(dbos: DBOS) -> None:
    """An unregistered instance whose `model_id` can't be fed back through `infer_model` errors helpfully.

    `TestModel()` round-trips as `'test:test'`, which `infer_model` can't rebuild; instead of a
    bare 'Unknown provider' the step points at the `models=` / `ResolveModelId` escape hatches.
    """
    agent = Agent(_durability_fn_model, name='durability_unrebuildable', capabilities=[DBOSDurability()])

    @DBOS.workflow()
    async def run_agent() -> None:
        await agent.run('hello', model=TestModel())

    with pytest.raises(UserError, match='could not be rebuilt'):
        await run_agent()


async def test_dbos_durability_string_default_model(dbos: DBOS) -> None:
    """A string default that `infer_model` can build is registered under its own id as well as `'default'`.

    Passing the same string at run time then resolves to the bound default instance via the
    registry, instead of building a fresh instance that would round-trip as a runtime override.
    """
    agent = Agent('test', name='durability_string_default', capabilities=[DBOSDurability()])

    @DBOS.workflow()
    async def run_agent() -> tuple[str, str]:
        default = await agent.run('hello')
        by_string = await agent.run('hello', model='test')
        return default.output, by_string.output

    assert await run_agent() == ('success (no tool calls)', 'success (no tool calls)')


async def test_dbos_durability_wrapped_default_model(dbos: DBOS) -> None:
    """A `WrapperModel` around the agent's own model still takes the default's fast path.

    Models are identity-matched after stripping wrapper layers (e.g. the `InstrumentedModel`
    an `Instrumentation` capability wraps around the model before the request runs), so a
    wrapped default isn't mistaken for a runtime override that would round-trip through
    `infer_model` and fail to rebuild.
    """
    agent = Agent(_durability_fn_model, name='durability_wrapped_default', capabilities=[DBOSDurability()])

    @DBOS.workflow()
    async def run_agent() -> str:
        result = await agent.run('hello', model=WrapperModel(_durability_fn_model))
        return result.output

    assert await run_agent() == 'Echo: hello'


def _dbos_tenant_resolver(ctx: ModelResolutionContext[str], model_id: str) -> FunctionModel | None:
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


async def test_dbos_durability_resolve_model_id_capability_is_deps_aware(dbos: DBOS) -> None:
    """A deps-aware `ResolveModelId` resolver rebuilds the model with the run's deps inside the step."""
    agent = Agent(
        _durability_fn_model,
        name='durability_tenant',
        deps_type=str,
        capabilities=[ResolveModelId(_dbos_tenant_resolver), DBOSDurability()],
    )

    @DBOS.workflow()
    async def run_agent() -> list[str]:
        outputs = [(await agent.run('hi', model='tenant-model', deps=tenant)).output for tenant in ('acme', 'globex')]
        # A string the resolver doesn't recognize defers to the default `infer_model` flow.
        outputs.append((await agent.run('hi', model='test', deps='acme')).output)
        return outputs

    assert await run_agent() == ['tenant:acme', 'tenant:globex', 'success (no tool calls)']


def _dbos_broken_resolver(ctx: ModelResolutionContext[Any], model_id: str) -> FunctionModel | None:
    if model_id == 'broken-model':
        raise ValueError('resolver exploded')
    return None  # pragma: no cover - only 'broken-model' flows through this test


async def test_dbos_durability_user_resolver_error_propagates(dbos: DBOS) -> None:
    """An exception raised by a user's resolver propagates unchanged.

    Only the `infer_model` backstop gets translated into the could-not-be-rebuilt error
    (see the unrebuildable test above); a resolver's own failure must surface as-is —
    both from run-setup resolution and from re-resolution inside the step — so its
    message isn't misattributed to the durable round-trip.
    """
    agent = Agent(
        _durability_fn_model,
        name='durability_broken_resolver',
        capabilities=[ResolveModelId(_dbos_broken_resolver), DBOSDurability()],
    )

    @DBOS.workflow()
    async def run_agent() -> None:
        await agent.run('hello', model='broken-model')

    with pytest.raises(ValueError, match='resolver exploded'):
        await run_agent()


async def test_dbos_durability_alias_default_model(dbos: DBOS) -> None:
    """An agent whose *default* model is an alias only a `ResolveModelId` capability can resolve.

    `infer_model` can't build `'tenant-model'`, so binding registers no concrete default;
    every request carries the raw alias string across the step boundary and the step
    re-resolves it with the run's deps.
    """
    agent = Agent(
        'tenant-model',
        name='durability_alias_default',
        deps_type=str,
        capabilities=[ResolveModelId(_dbos_tenant_resolver), DBOSDurability()],
    )

    @DBOS.workflow()
    async def run_agent() -> str:
        result = await agent.run('hi', deps='acme')
        return result.output

    assert await run_agent() == 'tenant:acme'


async def test_dbos_durability_per_step_model_selector_alias_resolved_by_capability(dbos: DBOS) -> None:
    """A selector alias that only a `ResolveModelId` capability can resolve crosses the step boundary.

    Unlike a `models=`-registered alias, identity lookup can't recover this one: the selector's
    string itself must be recorded as the request's model ID, or the step would receive the
    resolved model's own ID (`function:tenant-model`), which neither the resolver nor
    `infer_model` can rebuild.
    """
    agent = Agent(
        _durability_fn_model,
        name='durability_selector_capability_alias',
        deps_type=str,
        capabilities=[
            SelectModel(lambda ctx: 'tenant-model'),
            ResolveModelId(_dbos_tenant_resolver),
            DBOSDurability(),
        ],
    )

    @DBOS.workflow()
    async def run_agent() -> str:
        return (await agent.run('hi', deps='acme')).output

    assert await run_agent() == 'tenant:acme'


@dataclass
class _WrapRequestModelCapability(AbstractCapability[str]):
    """Swaps a transparent `WrapperModel` around the request's model in `before_model_request`."""

    async def before_model_request(
        self, ctx: RunContext[str], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        request_context.model = WrapperModel(request_context.model)
        return request_context


async def test_dbos_durability_wrapper_swap_keeps_alias_provenance(dbos: DBOS) -> None:
    """A transparent wrapper swapped in by `before_model_request` doesn't invalidate provenance.

    Unwrapping both sides shows the request still targets the run's model, so the original
    alias string crosses the step boundary and the worker re-resolves it deps-aware.
    """
    agent = Agent(
        'tenant-model',
        name='durability_wrapper_swap_provenance',
        deps_type=str,
        capabilities=[_WrapRequestModelCapability(), ResolveModelId(_dbos_tenant_resolver), DBOSDurability()],
    )

    @DBOS.workflow()
    async def run_agent() -> str:
        return (await agent.run('hi', deps='acme')).output

    assert await run_agent() == 'tenant:acme'


async def test_dbos_durability_allows_instrumented_default_model(dbos: DBOS) -> None:
    """An outer `Instrumentation` capability wraps the model, but the default model is still accepted.

    `_find_model_id` peels any wrapper layers off the request's model before comparing
    instances by identity, so an instrumented run still takes the default's `model_id=None`
    fast path.
    """
    agent = Agent(
        _durability_fn_model,
        name='durability_instrumented_default',
        capabilities=[Instrumentation(settings=InstrumentationSettings()), DBOSDurability()],
    )

    @DBOS.workflow()
    async def run_agent() -> str:
        result = await agent.run('hello')
        return result.output

    assert await run_agent() == 'Echo: hello'


def test_dbos_durability_get_ordering() -> None:
    """DBOSDurability declares innermost ordering."""
    from pydantic_ai.capabilities.abstract import CapabilityOrdering

    durability = DBOSDurability()
    ordering = durability.get_ordering()
    assert ordering == CapabilityOrdering(position='innermost')


def test_dbos_durability_get_serialization_name() -> None:
    """DBOSDurability is not spec-serializable."""
    assert DBOSDurability.get_serialization_name() is None


async def _durability_stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    for msg in reversed(messages):  # pragma: no branch - first message carries the prompt
        for part in msg.parts:  # pragma: no branch - first part is the UserPromptPart
            if isinstance(part, UserPromptPart):  # pragma: no branch - same reason
                yield f'Echo: {part.content}'
                return
    yield 'no prompt'  # pragma: no cover


async def test_dbos_durability_streaming_in_workflow(dbos: DBOS) -> None:
    """`ProcessEventStream` receives captured model events in workflow code."""
    events_in_step: list[tuple[AgentStreamEvent, bool]] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events_in_step.append((event, DBOS.step_id is not None))

    stream_model = FunctionModel(_durability_model_fn, stream_function=_durability_stream_fn)
    agent = Agent(
        stream_model,
        name='durability_streaming',
        capabilities=[ProcessEventStream(handler), DBOSDurability()],
    )

    wfid = str(uuid.uuid4())

    @DBOS.workflow()
    async def run_durable_streaming_agent() -> str:
        result = await agent.run('Hello streaming')
        return result.output

    with SetWorkflowID(wfid):
        output = await run_durable_streaming_agent()

    assert output == 'Echo: Hello streaming'
    model_events_in_step = [
        in_step for event, in_step in events_in_step if isinstance(event, (PartStartEvent, PartDeltaEvent))
    ]
    assert model_events_in_step
    assert not any(model_events_in_step)

    steps = await dbos.list_workflow_steps_async(wfid)
    step_names = [step['function_name'] for step in steps]
    assert 'durability_streaming__model.request_stream' in step_names


async def _chunks_stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'Stream'
    yield 'ed '
    yield 'response'


async def test_dbos_durability_process_event_stream_fires_workflow_side(dbos: DBOS) -> None:
    """ProcessEventStream sees the real captured events replayed in the workflow."""

    events_received: list[AgentStreamEvent] = []

    async def collect(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            assert DBOS.step_id is None
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
    assert delta_events == ['ed ', 'response']


async def test_dbos_durability_buffers_caller_streams(dbos: DBOS) -> None:
    agent = Agent(
        TestModel(custom_output_text='hello world'),
        name='durability_buffered_streams',
        capabilities=[DBOSDurability()],
    )

    @DBOS.workflow()
    async def run_durable_streams() -> tuple[list[str], str, list[str]]:
        async with agent.run_stream('Hello') as stream:
            chunks = [chunk async for chunk in stream.stream_text(debounce_by=None)]
            output = await stream.get_output()

        async with agent.run_stream_events('Hello') as event_stream:
            events = [event async for event in event_stream]
        deltas = [
            event.delta.content_delta
            for event in events
            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta)
        ]
        return chunks, output, deltas

    assert await run_durable_streams() == (['hello ', 'hello world'], 'hello world', ['hello ', 'world'])


def test_dbos_durability_event_stream_handler_step_config(monkeypatch: pytest.MonkeyPatch) -> None:
    registered_steps: list[tuple[str, dict[str, object]]] = []

    def step(*, name: str, **config: object):
        registered_steps.append((name, config))

        def decorator(function: Any) -> Any:
            return function

        return decorator

    monkeypatch.setattr(DBOS, 'step', step)
    config = StepConfig(retries_allowed=True, max_attempts=3)

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in stream:
            pass

    Agent(
        TestModel(),
        name='event_handler_config',
        capabilities=[DBOSDurability(event_stream_handler=handler, event_stream_handler_step_config=config)],
    )
    assert ('event_handler_config__event_stream_handler', config) in registered_steps


async def test_dbos_durability_event_stream_handler(dbos: DBOS) -> None:
    events_in_boundary: list[tuple[AgentStreamEvent, bool]] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events_in_boundary.append((event, DBOS.step_id is not None))

    async def handled_tool() -> str:
        return 'handled'

    durability = DBOSDurability(event_stream_handler=handler)
    agent = Agent(TestModel(), name='durability_handler', tools=[handled_tool], capabilities=[durability])

    wfid = str(uuid.uuid4())

    @DBOS.workflow()
    async def run_durable_agent() -> str:
        return (await agent.run('Hello')).output

    with SetWorkflowID(wfid):
        await run_durable_agent()
    events = [event for event, _ in events_in_boundary]
    assert events
    # Model events run inside the model-request step; tool events run inside a dedicated
    # event-handler step. Either way the handler always runs inside a checkpointed step.
    assert all(in_boundary for _, in_boundary in events_in_boundary)
    assert sum(isinstance(event, FunctionToolCallEvent) for event in events) == 1
    assert sum(isinstance(event, FunctionToolResultEvent) for event in events) == 1
    assert any(isinstance(event, PartStartEvent) for event in events)
    assert any(isinstance(event, FinalResultEvent) for event in events)

    steps = await dbos.list_workflow_steps_async(wfid)
    step_names = [step['function_name'] for step in steps]
    assert 'durability_handler__event_stream_handler' in step_names


async def test_dbos_durability_event_stream_handler_rejects_enqueue(dbos: DBOS) -> None:
    """An `event_stream_handler` that enqueues inside a durable step raises, like a tool would.

    The handler runs inside a durable step for both model events (the model-request step) and
    graph events (the `__event_stream_handler` step); either step's recorded result is replayed
    without re-running it, so an enqueue would be dropped. The handler catches the error on every
    event so the run still completes, exercising both delivery paths.
    """
    enqueue_errors: list[str] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in stream:
            with pytest.raises(UserError, match='enqueued messages would be dropped') as exc_info:
                ctx.enqueue('later')
            # The ambient current context is guarded too, so reading it instead of the argument
            # doesn't bypass the guard.
            ambient = get_current_run_context()
            assert ambient is not None
            with pytest.raises(UserError, match='enqueued messages would be dropped'):
                ambient.enqueue('later')
            enqueue_errors.append(str(exc_info.value))

    async def handled_tool() -> str:
        return 'handled'

    durability = DBOSDurability(event_stream_handler=handler)
    agent = Agent(TestModel(), name='durability_handler_enqueue', tools=[handled_tool], capabilities=[durability])

    @DBOS.workflow()
    async def run_durable_agent() -> str:
        return (await agent.run('Hello')).output

    with SetWorkflowID(str(uuid.uuid4())):
        await run_durable_agent()
    # Guarded on both the model-event (model-request step) and graph-event (dispatch step) paths.
    assert len(enqueue_errors) > 1


async def test_dbos_durability_event_stream_handler_outside_workflow(dbos: DBOS) -> None:
    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    durability = DBOSDurability(event_stream_handler=handler)
    agent = Agent(TestModel(custom_output_text='done'), name='outside_handler', capabilities=[durability])
    with agent.override():
        await agent.run('Hello')
    assert any(isinstance(event, PartStartEvent) for event in events)


def test_dbos_durability_without_handler_does_not_wrap_event_stream() -> None:
    assert DBOSDurability().has_wrap_run_event_stream is False


async def test_dbos_durability_runtime_handler_receives_buffered_events(dbos: DBOS) -> None:
    """A per-run `event_stream_handler` passed to `agent.run()` inside a DBOS workflow

    receives the events captured inside the step (rather than being silently dropped).
    The buffered replay preserves real granular deltas — the per-run handler sees the
    same multi-chunk stream the construction-time handler would see.
    """

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
    assert 'my_mcp' in bound._toolsets_by_id  # pyright: ignore[reportPrivateUsage]
    assert isinstance(bound._toolsets_by_id['my_mcp'], DBOSMCPToolset)  # pyright: ignore[reportPrivateUsage]


async def test_dbos_durability_rejects_idless_mcp_toolset(dbos: DBOS) -> None:
    """An `MCPToolset` without an `id` fails loudly at construction.

    The DBOS step wrapper is swapped in by toolset ID at run time, so without one the
    toolset's I/O would silently run un-checkpointed inside the DBOS workflow and
    re-execute on recovery. Temporal raises the equivalent error for id-less leaves.
    """
    mcp_toolset = MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), init_timeout=20)
    with pytest.raises(UserError, match=r"Toolsets that are 'leaves'.*unique `id`.*DBOS"):
        Agent(
            _durability_fn_model,
            name='durability_idless_mcp',
            toolsets=[mcp_toolset],
            capabilities=[DBOSDurability()],
        )


async def test_dbos_durability_wraps_capability_contributed_mcp_toolset(dbos: DBOS) -> None:
    """MCP toolsets contributed by other capabilities are wrapped as DBOS steps too.

    Durability capabilities are in the `innermost` ordering tier, so `Agent.__init__` binds
    them only after every other capability's contributed toolsets have been extracted into
    `agent.toolsets`. Without that two-phase binding, this toolset would be invisible to
    `for_agent` and run un-checkpointed inside the DBOS workflow.
    """
    from pydantic_ai.durable_exec.dbos._mcp_toolset import DBOSMCPToolset

    mcp_toolset = MCPToolset(
        StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='cap_mcp', init_timeout=20
    )
    agent = Agent(
        _durability_fn_model,
        name='durability_cap_mcp',
        capabilities=[Toolset(mcp_toolset), DBOSDurability()],
    )
    bound = DBOSDurability.from_agent(agent)
    assert bound is not None

    assert 'cap_mcp' in bound._toolsets_by_id  # pyright: ignore[reportPrivateUsage]
    assert isinstance(bound._toolsets_by_id['cap_mcp'], DBOSMCPToolset)  # pyright: ignore[reportPrivateUsage]


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

    assert 'swap_mcp' in bound._toolsets_by_id  # pyright: ignore[reportPrivateUsage]

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

    agent = Agent(FunctionModel(call_then_answer), name='durability_runtime_fn', capabilities=[DBOSDurability()])

    result = await agent.run(
        'Call the runtime tool.', toolsets=[FunctionToolset(tools=[runtime_tool], id='runtime_fn')]
    )
    assert result.output == 'done'


async def test_dbos_durability_dynamic_capability_tool_runs_in_step(dbos: DBOS) -> None:
    """A dynamic capability's tool discovery and calls run in DBOS steps."""
    calls: list[str] = []

    def dynamic_tool() -> str:
        calls.append('called')
        return 'dynamic result'

    def factory(ctx: RunContext[Any]) -> Capability[Any]:
        return Capability(tools=[dynamic_tool])

    agent = Agent(
        TestModel(),
        name='dbos_dynamic_capability',
        capabilities=[DynamicCapability(factory, id='dyn'), DBOSDurability()],
    )

    @DBOS.workflow()
    async def run_agent() -> str:
        return (await agent.run('Call the tool')).output

    wfid = str(uuid.uuid4())
    with SetWorkflowID(wfid):
        output = await run_agent()

    assert output == '{"dynamic_tool":"dynamic result"}'
    assert calls == ['called']
    step_names = [step['function_name'] for step in await dbos.list_workflow_steps_async(wfid)]
    assert 'dbos_dynamic_capability__dynamic_toolset__dyn.get_tools' in step_names
    assert 'dbos_dynamic_capability__dynamic_toolset__dyn.call_tool' in step_names


def test_dbos_durability_dynamic_capability_requires_id(dbos: DBOS) -> None:
    def factory(ctx: RunContext[Any]) -> Capability[Any]:
        return Capability()  # pragma: no cover — construction raises before the factory can run

    with pytest.raises(UserError, match=r"DynamicCapability\(\.\.\., id='user-tools'\)"):
        Agent(
            TestModel(),
            name='dbos_dynamic_capability_no_id',
            capabilities=[DynamicCapability(factory), DBOSDurability()],
        )


def test_dbos_durability_bare_capability_func_requires_explicit_wrapper(dbos: DBOS) -> None:
    """A bare `CapabilityFunc` in `capabilities=` is wrapped in an id-less `DynamicCapability`,
    so under durable execution it raises with a hint to wrap it explicitly."""

    def factory(ctx: RunContext[Any]) -> Capability[Any]:
        return Capability()  # pragma: no cover — construction raises before the factory can run

    with pytest.raises(UserError, match=r'wrap it explicitly'):
        Agent(
            TestModel(),
            name='dbos_bare_capability_func',
            capabilities=[factory, DBOSDurability()],
        )


async def test_dbos_durability_dynamic_capability_mcp_runs_in_steps(dbos: DBOS) -> None:
    def call_then_answer(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('celsius_to_fahrenheit', {'celsius': 0}, tool_call_id='call-1')])

    def factory(ctx: RunContext[Any]) -> Capability[Any]:
        return Capability(
            toolsets=[
                MCPToolset(
                    StdioTransport(command='python', args=['-m', 'tests.mcp_server']),
                    id='dynamic_mcp',
                )
            ]
        )

    agent = Agent(
        FunctionModel(call_then_answer),
        name='dbos_dynamic_mcp',
        capabilities=[DynamicCapability(factory, id='dyn_mcp'), DBOSDurability()],
    )

    @DBOS.workflow()
    async def run_agent() -> str:
        return (await agent.run('Convert zero Celsius to Fahrenheit.')).output

    wfid = str(uuid.uuid4())
    with SetWorkflowID(wfid):
        output = await run_agent()

    assert output == 'done'
    step_names = [step['function_name'] for step in await dbos.list_workflow_steps_async(wfid)]
    assert 'dbos_dynamic_mcp__dynamic_toolset__dyn_mcp.get_tools' in step_names
    assert 'dbos_dynamic_mcp__dynamic_toolset__dyn_mcp.call_tool' in step_names


async def test_dbos_durability_rejects_runtime_mcp_toolset(dbos: DBOS) -> None:
    """An `MCPToolset` added per-run is rejected: its I/O steps must be registered at construction."""
    agent = Agent(_durability_fn_model, name='durability_runtime_mcp', capabilities=[DBOSDurability()])

    @DBOS.workflow()
    async def run_agent() -> None:
        await agent.run(
            'Hello',
            toolsets=[MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='runtime_mcp')],
        )

    with pytest.raises(
        UserError, match=r'MCPToolset cannot be passed to `run\(toolsets=\.\.\.\)` at runtime with DBOS'
    ):
        await run_agent()


def test_dbos_durability_rejects_runtime_dynamic_toolset_sync(dbos: DBOS) -> None:
    """A `DynamicToolset` added per-run is rejected, on `run_sync` as well as `run`."""
    agent = Agent(_durability_fn_model, name='durability_runtime_dynamic', capabilities=[DBOSDurability()])

    @DBOS.workflow()
    def run_agent() -> None:
        agent.run_sync('Hello', toolsets=[DynamicToolset(lambda _: FunctionToolset(), id='runtime_dynamic')])

    with pytest.raises(
        UserError, match=r'DynamicToolset cannot be passed to `run\(toolsets=\.\.\.\)` at runtime with DBOS'
    ):
        run_agent()


async def test_dbos_durability_rejects_runtime_mcp_toolset_in_iter(dbos: DBOS) -> None:
    """`agent.iter(toolsets=...)` inside a user workflow is guarded like `run(toolsets=...)`.

    The rejection lives in run setup (`get_wrapper_toolset`), which every entry point routes
    through so `iter` inside a workflow cannot execute the MCP toolset's I/O un-checkpointed.
    """
    agent = Agent(_durability_fn_model, name='durability_runtime_mcp_iter', capabilities=[DBOSDurability()])

    @DBOS.workflow()
    async def run_agent() -> None:
        async with agent.iter(
            'Hello',
            toolsets=[MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='iter_mcp')],
        ):
            pass  # pragma: no cover — run setup raises before any node runs

    with pytest.raises(
        UserError, match=r'MCPToolset cannot be passed to `run\(toolsets=\.\.\.\)` at runtime with DBOS'
    ):
        await run_agent()


def _per_run_dynamic_factory(ctx: RunContext[Any]) -> FunctionToolset[Any]:
    return FunctionToolset()  # pragma: no cover — rejected before the factory is resolved


async def test_dbos_durability_rejects_per_run_capability_toolset(dbos: DBOS) -> None:
    """An executing toolset contributed by a per-run capability is rejected like `run(toolsets=...)`.

    Construction-time capability toolsets are wrapped by `for_agent` (see the
    capability-contributed test above); a per-run capability's toolset arrives after that
    wrapping has happened, so it would run un-checkpointed inside the workflow. A
    `DynamicToolset` with a module-level factory keeps the workflow input serializable.
    """
    agent = Agent(_durability_fn_model, name='durability_per_run_cap_toolset', capabilities=[DBOSDurability()])

    @DBOS.workflow()
    async def run_agent() -> None:
        await agent.run('Hello', capabilities=[Toolset(DynamicToolset(_per_run_dynamic_factory, id='per_run_dynamic'))])

    with pytest.raises(UserError, match='DynamicToolset cannot be passed'):
        await run_agent()


async def test_dbos_durability_rejects_duplicate_toolset_id(dbos: DBOS) -> None:
    """Two distinct MCP toolsets under one `id` are rejected at binding time.

    The registry maps `id` → step wrapper, so a duplicate would silently replace the first
    entry and route both toolsets' calls through the last one's steps.
    """
    with pytest.raises(UserError, match="Two toolsets have the same `id` 'dup'"):
        Agent(
            _durability_fn_model,
            name='durability_dup_mcp',
            toolsets=[
                MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='dup'),
                MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='dup'),
            ],
            capabilities=[DBOSDurability()],
        )


async def test_dbos_durability_same_toolset_instance_reused(dbos: DBOS) -> None:
    """The same MCP toolset instance appearing twice maps to one wrapper, not an `id` conflict."""
    mcp_toolset = MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='shared_mcp')
    agent = Agent(
        _durability_fn_model,
        name='durability_shared_mcp',
        toolsets=[mcp_toolset, mcp_toolset],
        capabilities=[DBOSDurability()],
    )
    bound = DBOSDurability.from_agent(agent)
    assert bound is not None
    assert list(bound._toolsets_by_id) == ['shared_mcp']  # pyright: ignore[reportPrivateUsage]


# --- Continuation chains (suspended → complete) run one step per segment ---
#
# When a model suspends a turn (Anthropic `pause_turn`, OpenAI background mode), the
# continuation loop in the innermost `model_request`/`model_request_stream` helpers runs
# workflow-side under `DBOSDurability`, dispatching each segment through its own model
# request step. These tests use a scripted model (no cassettes: `FunctionModel` can't emit
# suspended streaming segments, and VCR matchers wouldn't pin the chain shape).


async def test_dbos_durability_continuation_chain_in_workflow(dbos: DBOS) -> None:
    """A suspended → complete chain resolves across per-segment DBOS steps, as one merged response.

    Usage is counted once (a continuation isn't a separate request step), and the workflow
    record shows one model-request step per segment.
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
    agent = Agent(model, name='durability_continuation', capabilities=[DBOSDurability()])

    results: list[AgentRunResult[str]] = []

    @DBOS.workflow()
    async def run_durable_agent() -> str:
        result = await agent.run('go')
        results.append(result)
        return result.output

    wfid = str(uuid.uuid4())
    with SetWorkflowID(wfid):
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
    # Each segment ran in its own durable boundary: two model calls, two steps.
    assert model.request_calls == 2
    steps = await dbos.list_workflow_steps_async(wfid)
    step_names = [step['function_name'] for step in steps]
    assert step_names.count('durability_continuation__model.request') == 2


async def test_dbos_durability_continuation_usage_limit_cancels_suspended(dbos: DBOS) -> None:
    """A usage limit tripped between segments cancels the live suspended job in its own step.

    The continuation loop runs workflow-side and checks the limit as each segment merges;
    the provider teardown of the abandoned server-side job is I/O, so it crosses the
    boundary through the dedicated cancellation step, and the error surfaces to workflow
    code with its real type.
    """
    model = ScriptedContinuationModel(
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
    agent = Agent(model, name='durability_continuation_usage_limit', capabilities=[DBOSDurability()])

    @DBOS.workflow()
    async def run_agent() -> None:
        await agent.run('go', usage_limits=UsageLimits(total_tokens_limit=20))

    wfid = str(uuid.uuid4())
    with SetWorkflowID(wfid):
        with pytest.raises(UsageLimitExceeded, match='total_tokens_limit'):
            await run_agent()

    assert model.request_calls == 2
    # The over-budget merge was still suspended, so the live job was cancelled before raising.
    assert [cancelled.provider_response_id for cancelled in model.cancelled] == ['cont2']
    steps = await dbos.list_workflow_steps_async(wfid)
    step_names = [step['function_name'] for step in steps]
    assert step_names.count('durability_continuation_usage_limit__model.request') == 2
    assert step_names.count('durability_continuation_usage_limit__model.cancel_suspended_response') == 1


async def test_dbos_durability_streaming_continuation_chain_in_workflow(dbos: DBOS) -> None:
    """A streamed suspended → complete chain is stitched across per-segment DBOS steps.

    `ProcessEventStream` receives each captured segment in workflow code, and the
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
        capabilities=[ProcessEventStream(handler), DBOSDurability()],
    )

    results: list[AgentRunResult[str]] = []

    @DBOS.workflow()
    async def run_durable_agent() -> str:
        result = await agent.run('go')
        results.append(result)
        return result.output

    wfid = str(uuid.uuid4())
    with SetWorkflowID(wfid):
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
    steps = await dbos.list_workflow_steps_async(wfid)
    step_names = [step['function_name'] for step in steps]
    assert step_names.count('durability_continuation_stream__model.request_stream') == 2


async def test_dbos_durability_continuation_resume_from_history(dbos: DBOS) -> None:
    """A suspended history tail crosses the first DBOS step boundary.

    The workflow-side continuation loop passes the tail as the last request message.
    """
    model = ScriptedContinuationModel(
        responses=[scripted_response(texts=['is 42.'], provider_response_id='cont2', input_tokens=3, output_tokens=4)]
    )
    agent = Agent(model, name='durability_continuation_resume', capabilities=[DBOSDurability()])

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='go')]),
        scripted_response(
            texts=['The answer '], state='suspended', provider_response_id='cont1', input_tokens=5, output_tokens=2
        ),
    ]

    results: list[AgentRunResult[str]] = []

    @DBOS.workflow()
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
