# pyright: reportDeprecated=false
# `TemporalAgent` (the wrapper-agent path) is deprecated in favor of the
# `TemporalDurability` capability, but this file still exercises both paths in
# parallel for parity. Silenced at file level rather than annotating every
# individual usage.
from __future__ import annotations

import asyncio
import os
import re
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal, cast
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    AbstractToolset,
    Agent,
    AgentRunResultEvent,
    AgentStreamEvent,
    BinaryContent,
    BinaryImage,
    CodeExecutionTool,
    DocumentUrl,
    ExternalToolset,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    FunctionToolset,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    MultiModalContent,
    OutputToolCallEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RequestUsage,
    RetryPromptPart,
    RunContext,
    RunUsage,
    TextContent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturn,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    WebSearchTool,
    WebSearchUserLocation,
)
from pydantic_ai.capabilities import (
    Instrumentation,
    NativeTool,
    ProcessEventStream,
    ProcessHistory,
    ResolveModelId,
    Toolset,
)
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import UploadedFile
from pydantic_ai.models import (
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    ModelResolutionContext,
    create_async_http_client,
    infer_model,
    infer_model_profile,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool
from pydantic_ai.profiles import DEFAULT_PROFILE
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition
from pydantic_ai.usage import UsageLimits
from pydantic_graph import GraphBuilder, StepContext
from pydantic_graph.join import reduce_list_append

from ._inline_snapshot import snapshot
from .continuation_utils import ScriptedContinuationModel, StreamSegment, scripted_response

try:
    import temporalio.api.common.v1
    from temporalio import activity, workflow
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]
    from temporalio.client import Client, WorkflowFailureError, WorkflowHistory
    from temporalio.common import RetryPolicy
    from temporalio.contrib.opentelemetry import TracingInterceptor
    from temporalio.contrib.pydantic import PydanticPayloadConverter, pydantic_data_converter
    from temporalio.converter import DataConverter, DefaultPayloadConverter, PayloadCodec
    from temporalio.exceptions import ApplicationError
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        LogfirePlugin,
        PydanticAIPlugin,
        PydanticAIWorkflow,
        TemporalAgent,
        TemporalDurability,
    )
    from pydantic_ai.durable_exec.temporal._function_toolset import TemporalFunctionToolset
    from pydantic_ai.durable_exec.temporal._mcp_toolset import TemporalMCPToolset
    from pydantic_ai.durable_exec.temporal._model import TemporalModel
    from pydantic_ai.durable_exec.temporal._run_context import TemporalRunContext
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

import sys

if sys.version_info >= (3, 14):
    pytest.skip(
        'temporalio sandbox is incompatible with Python 3.14: '
        'sandbox module state accumulates across validation cycles causing import failures after ~22 workflows '
        '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
        allow_module_level=True,
    )

try:
    import logfire
    from logfire import Logfire
    from logfire._internal.tracer import _ProxyTracer  # pyright: ignore[reportPrivateUsage]
    from logfire.testing import CaptureLogfire
    from opentelemetry.trace import ProxyTracer
except ImportError:  # pragma: lax no cover
    pytest.skip('logfire not installed', allow_module_level=True)

try:
    from fastmcp.client.transports import StdioTransport

    from pydantic_ai.mcp import MCPToolset
except ImportError:  # pragma: lax no cover
    pytest.skip('mcp not installed', allow_module_level=True)

try:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:  # pragma: lax no cover
    pytest.skip('openai not installed', allow_module_level=True)


with workflow.unsafe.imports_passed_through():
    # Workaround for a race condition when running `logfire.info` inside an activity with attributes to serialize and pandas importable:
    # AttributeError: partially initialized module 'pandas' has no attribute '_pandas_parser_CAPI' (most likely due to a circular import)
    try:
        import pandas  # pyright: ignore[reportUnusedImport] # noqa: F401
    except ImportError:  # pragma: lax no cover
        pass

    # https://github.com/temporalio/sdk-python/blob/3244f8bffebee05e0e7efefb1240a75039903dda/tests/test_client.py#L112C1-L113C1
    from mcp.client.session import ClientSession
    from mcp.types import ClientRequest

    from ._inline_snapshot import snapshot

    # Loads `vcr`, which Temporal doesn't like without passing through the import
    from .conftest import IsDatetime, IsInt, IsStr, message

# `TemporalAgent` is deprecated in favor of `capabilities=[TemporalDurability(...)]`.
# These tests exercise the wrapper-agent path on purpose; suppress the warning here
# rather than globally in `pyproject.toml`. The `pytestmark` entry below covers warnings
# emitted *inside* test functions; the `filterwarnings` call below covers warnings emitted
# at module import time (e.g. module-level construction of `TemporalAgent`).
warnings.filterwarnings('ignore', message='`TemporalAgent` is deprecated', category=DeprecationWarning)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.xdist_group(name='temporal'),
    pytest.mark.filterwarnings('ignore:`TemporalAgent` is deprecated:DeprecationWarning'),
]


# We need to use a custom cached HTTP client here as the default one created for OpenAIProvider will be closed automatically
# at the end of each test, but we need this one to live longer.
http_client = create_async_http_client()


# Scoped to `session` rather than `module`: the `http_client` and the module-level agents that
# capture it are constructed at import time, so they must outlive a single module entry. This is a
# sync fixture so it doesn't force AnyIO to reuse a session-level event loop for all Temporal async
# fixtures; the `temporal_env` teardown can make that loop unusable for later tests.
@pytest.fixture(autouse=True, scope='session')
def close_cached_httpx_client() -> Iterator[None]:
    try:
        yield
    finally:
        asyncio.run(http_client.aclose())


# `LogfirePlugin` calls `logfire.instrument_pydantic_ai()`, so we need to make sure this doesn't bleed into other tests.
@pytest.fixture(autouse=True, scope='module')
def uninstrument_pydantic_ai() -> Iterator[None]:
    try:
        yield
    finally:
        Agent.instrument_all(False)


@contextmanager
def workflow_raises(exc_type: type[Exception], exc_message: str) -> Generator[None]:
    """Helper for asserting that a Temporal workflow fails with the expected error."""
    with pytest.raises(WorkflowFailureError) as exc_info:
        yield
    assert isinstance(exc_info.value.__cause__, ApplicationError)
    assert exc_info.value.__cause__.type == exc_type.__name__
    assert exc_info.value.__cause__.message == exc_message


TEMPORAL_PORT = 7243
TASK_QUEUE = 'pydantic-ai-agent-task-queue'
BASE_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)


def _kill_leaked_temporal_server(port: int) -> None:
    """Kill any `temporal-sdk-python-*` dev server still bound to `port`.

    A previous test run that crashed mid-fixture leaves the embedded Temporal
    dev server listening on `port`, which makes the next run fail to bind. The
    leak persists across pytest invocations, so detect-and-kill at fixture entry
    keeps local iterations smooth without requiring a manual `kill` between runs.
    Best-effort: failures here don't propagate, the fixture's own bind attempt
    will surface a real port conflict downstream.
    """
    import signal
    import subprocess

    try:
        result = subprocess.run(
            ['ss', '-tlnpH', f'sport = :{port}'],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):  # pragma: lax no cover - no `ss` or unresponsive
        return

    for line in result.stdout.splitlines():  # pragma: lax no cover - body fires only on a real leak
        if 'temporal-sdk-py' not in line:
            continue
        match = re.search(r'pid=(\d+)', line)
        if not match:
            continue
        pid = int(match.group(1))
        try:
            os.kill(pid, signal.SIGTERM)
        except (PermissionError, ProcessLookupError):
            pass


@pytest.fixture(scope='module')
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    _kill_leaked_temporal_server(TEMPORAL_PORT)
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        port=TEMPORAL_PORT,
        ui=True,
        dev_server_extra_args=['--dynamic-config-value', 'frontend.enableServerVersionCheck=false'],
    ) as env:
        yield env


@pytest.fixture
async def client(temporal_env: WorkflowEnvironment) -> Client:
    return await Client.connect(
        f'localhost:{TEMPORAL_PORT}',
        plugins=[PydanticAIPlugin()],
    )


@pytest.fixture
async def client_with_logfire(temporal_env: WorkflowEnvironment) -> Client:
    return await Client.connect(
        f'localhost:{TEMPORAL_PORT}',
        plugins=[PydanticAIPlugin(), LogfirePlugin()],
    )


# Can't use the `openai_api_key` fixture here because the workflow needs to be defined at the top level of the file.
model = OpenAIChatModel(
    'gpt-4o',
    provider=OpenAIProvider(
        api_key=os.getenv('OPENAI_API_KEY', 'mock-api-key'),
        http_client=http_client,
    ),
)

simple_agent = Agent(model, name='simple_agent')

# This needs to be done before the `TemporalAgent` is bound to the workflow.
simple_temporal_agent = TemporalAgent(simple_agent, activity_config=BASE_ACTIVITY_CONFIG)


@workflow.defn
class SimpleAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_temporal_agent.run(prompt)
        return result.output


async def test_simple_agent_run_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflow],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        output = await client.execute_workflow(
            SimpleAgentWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=SimpleAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
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


async def get_country(ctx: RunContext[Deps]) -> str:
    return ctx.deps.country


class WeatherArgs(BaseModel):
    city: str


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
    name='complex_agent',
)

# This needs to be done before the `TemporalAgent` is bound to the workflow.
complex_temporal_agent = TemporalAgent(
    complex_agent,
    event_stream_handler=event_stream_handler,
    activity_config=BASE_ACTIVITY_CONFIG,
    model_activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=90)),
    toolset_activity_config={
        'country': ActivityConfig(start_to_close_timeout=timedelta(seconds=120)),
    },
    tool_activity_config={
        'country': {
            'get_country': False,
        },
        'mcp': {
            'get_product_name': ActivityConfig(start_to_close_timeout=timedelta(seconds=150)),
        },
        '<agent>': {
            'get_weather': ActivityConfig(start_to_close_timeout=timedelta(seconds=180)),
        },
    },
)


@workflow.defn
class ComplexAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: Deps) -> Response:
        result = await complex_temporal_agent.run(prompt, deps=deps)
        return result.output


@dataclass
class BasicSpan:
    content: str
    children: list[BasicSpan] = field(default_factory=list['BasicSpan'])
    parent_id: int | None = field(repr=False, compare=False, default=None)


async def test_complex_agent_run_in_workflow(
    allow_model_requests: None, client_with_logfire: Client, capfire: CaptureLogfire
):
    async with Worker(
        client_with_logfire,
        task_queue=TASK_QUEUE,
        workflows=[ComplexAgentWorkflow],
        plugins=[AgentPlugin(complex_temporal_agent)],
    ):
        output = await client_with_logfire.execute_workflow(
            ComplexAgentWorkflow.run,
            args=[
                'Tell me: the capital of the country; the weather there; the product name',
                Deps(country='Mexico'),
            ],
            id=ComplexAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            Response(
                answers=[
                    Answer(label='Capital of the country', answer='Mexico City'),
                    Answer(label='Weather in the capital', answer='Sunny'),
                    Answer(label='Product Name', answer='Pydantic AI'),
                ]
            )
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

    assert root_span == snapshot(
        BasicSpan(
            content='StartWorkflow:ComplexAgentWorkflow',
            children=[
                BasicSpan(content='RunWorkflow:ComplexAgentWorkflow'),
                BasicSpan(
                    content='complex_agent run',
                    children=[
                        BasicSpan(
                            content='StartActivity:agent__complex_agent__mcp_server__mcp__get_tools',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__mcp_server__mcp__get_tools',
                                    children=[BasicSpan(content='tools/list')],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__complex_agent__model_request_stream',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__complex_agent__model_request_stream',
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
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=1'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "get_country", "args": "{}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "args_valid": true, "event_kind": "function_tool_call"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=1'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "get_product_name", "args": "{}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "args_valid": true, "event_kind": "function_tool_call"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(content='running tool: get_country'),
                        BasicSpan(
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=1'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "get_country", "content": "Mexico", "tool_call_id": null, "tool_kind": null, "metadata": null, "timestamp": null, "outcome": "success", "part_kind": "tool-return"}, "content": null, "event_kind": "function_tool_result"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='running tool: get_product_name',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__complex_agent__mcp_server__mcp__call_tool',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__complex_agent__mcp_server__mcp__call_tool',
                                            children=[BasicSpan(content='tools/call get_product_name')],
                                        )
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=1'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "get_product_name", "content": "Pydantic AI", "tool_call_id": null, "tool_kind": null, "metadata": null, "timestamp": null, "outcome": "success", "part_kind": "tool-return"}, "content": null, "event_kind": "function_tool_result"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__complex_agent__model_request_stream',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__complex_agent__model_request_stream',
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
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=2'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "get_weather", "args": "{\\"city\\":\\"Mexico City\\"}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "args_valid": true, "event_kind": "function_tool_call"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='running tool: get_weather',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__complex_agent__toolset__<agent>__call_tool',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__complex_agent__toolset__<agent>__call_tool'
                                        )
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=2'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "get_weather", "content": "sunny", "tool_call_id": null, "tool_kind": null, "metadata": null, "timestamp": null, "outcome": "success", "part_kind": "tool-return"}, "content": null, "event_kind": "function_tool_result"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__complex_agent__model_request_stream',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__complex_agent__model_request_stream',
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
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=3'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "final_result", "args": "{\\"answers\\":[{\\"label\\":\\"Capital of the country\\",\\"answer\\":\\"Mexico City\\"},{\\"label\\":\\"Weather in the capital\\",\\"answer\\":\\"Sunny\\"},{\\"label\\":\\"Product Name\\",\\"answer\\":\\"Pydantic AI\\"}]}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "args_valid": true, "event_kind": "output_tool_call"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=3'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "final_result", "content": "Final result processed.", "tool_call_id": null, "tool_kind": null, "metadata": null, "timestamp": null, "outcome": "success", "part_kind": "tool-return"}, "event_kind": "output_tool_result"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
                BasicSpan(content='CompleteWorkflow:ComplexAgentWorkflow'),
            ],
        )
    )


async def test_mcp_tools_cached_across_activities(allow_model_requests: None, client: Client):
    """Verify that MCP tool caching reduces server round-trips across activities.

    The complex agent makes 3 model requests, each preceded by a get_tools activity.
    With the run-scoped tool-defs cache, only the first get_tools activity actually runs
    (opening an MCP connection and calling `tools/list`). Subsequent get_tools calls return
    the run-cached tool definitions without scheduling an activity at all.
    """

    original_send_request = ClientSession.send_request
    methods_called: list[str] = []

    async def tracking_send_request(self_: ClientSession, request: ClientRequest, *args: Any, **kwargs: Any) -> Any:
        methods_called.append(request.root.method)
        return await original_send_request(self_, request, *args, **kwargs)

    with patch.object(ClientSession, 'send_request', tracking_send_request):
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[ComplexAgentWorkflow],
            plugins=[AgentPlugin(complex_temporal_agent)],
        ):
            coro = client.execute_workflow(
                ComplexAgentWorkflow.run,
                args=[
                    'Tell me: the capital of the country; the weather there; the product name',
                    Deps(country='Mexico'),
                ],
                id=f'{ComplexAgentWorkflow.__name__}_cache_test',
                task_queue=TASK_QUEUE,
            )
            output = await coro
        assert output is not None

    # 3 get_tools calls are made, but only 1 results in an actual tools/list MCP request
    assert methods_called.count('tools/list') == 1
    # call_tool should still make a request each time (not cached)
    assert methods_called.count('tools/call') == 1


def _call_mcp_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Two model steps: call an MCP tool on the first request, return text on the second.

    Two model requests means `get_tools` is invoked twice on the MCP toolset within one run,
    so the run-scoped cache (and the activity it does or doesn't schedule each step) is exercised.
    """
    tool_returned = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
    if tool_returned:
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('get_weather_forecast', {'location': 'Mexico City'})])


# A holder lets the replay step swap in a freshly-constructed (cold-process) instance,
# reproducing the worker-restart scenario from #5875.
mcp_replay_holder: dict[str, TemporalAgent[None, str]] = {}


def _make_mcp_replay_agent(cache_tools: bool = True) -> TemporalAgent[None, str]:
    agent = Agent(
        FunctionModel(_call_mcp_then_finish),
        name='mcp_replay_agent',
        toolsets=[
            MCPToolset(
                StdioTransport(command='python', args=['-m', 'tests.mcp_server']),
                id='mcp',
                init_timeout=20,
                cache_tools=cache_tools,
            )
        ],
    )
    return TemporalAgent(agent, activity_config=BASE_ACTIVITY_CONFIG)


mcp_replay_holder['agent'] = _make_mcp_replay_agent()


@workflow.defn
class MCPReplayWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await mcp_replay_holder['agent'].run(prompt)
        return result.output


def _scheduled_get_tools_count(history: WorkflowHistory) -> int:
    return sum(
        1
        for event in history.events
        if event.HasField('activity_task_scheduled_event_attributes')
        and event.activity_task_scheduled_event_attributes.activity_type.name.endswith('__get_tools')
    )


async def test_temporal_mcp_get_tools_replay_deterministic(allow_model_requests: None, client: Client):
    """#5875 regression: `get_tools` activity scheduling must be replay-deterministic.

    The tool-defs cache must not let shared-process cache warmth decide whether a workflow
    emits a `get_tools` activity command — otherwise a history recorded on a warm worker fails
    replay on a cold one (and vice versa) with `TMPRL1100`. Each run must independently record
    exactly one `get_tools` activity: the #4331 within-run win (N calls collapse to one activity)
    without leaking cache state across the replay boundary.
    """
    warm = _make_mcp_replay_agent()
    mcp_replay_holder['agent'] = warm

    histories: list[WorkflowHistory] = []
    # Unsandboxed so the module-level instance (and its cache) is shared across both runs,
    # exactly as a long-running worker process shares it in production — the condition under
    # which #5875 records a warm run with no `get_tools` event.
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MCPReplayWorkflow],
        activities=warm.temporal_activities,
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        for i in range(2):
            wf_id = f'{MCPReplayWorkflow.__name__}_{i}'
            await client.execute_workflow(MCPReplayWorkflow.run, args=['hello'], id=wf_id, task_queue=TASK_QUEUE)
            histories.append(await client.get_workflow_handle(wf_id).fetch_history())
    h1, h2 = histories

    # Within a run, the run-scoped cache collapses the per-step `get_tools` calls to one activity...
    assert _scheduled_get_tools_count(h1) == 1
    # ...and each run records it independently — run 2 does not inherit run 1's warm process cache.
    assert _scheduled_get_tools_count(h2) == 1

    def replayer() -> Replayer:
        return Replayer(
            workflows=[MCPReplayWorkflow],
            workflow_runner=UnsandboxedWorkflowRunner(),
            data_converter=pydantic_data_converter,
        )

    try:
        # Direction 1: cold-recorded history (run 1) replayed after the process cache warmed
        # (the same-process sticky-cache-eviction trigger). Holder still points at the warm instance.
        await replayer().replay_workflow(h1)

        # Direction 2: warm-recorded history (run 2) replayed on a freshly-constructed cold instance
        # (the worker-restart trigger).
        mcp_replay_holder['agent'] = _make_mcp_replay_agent()
        await replayer().replay_workflow(h2)
    finally:
        mcp_replay_holder['agent'] = warm


async def test_temporal_mcp_get_tools_not_cached_when_disabled(allow_model_requests: None, client: Client):
    """With `cache_tools=False`, `get_tools` is scheduled for every model request (no run cache).

    The complementary case to the run-scoped cache: each of the two model requests records its own
    `get_tools` activity, so disabling the cache stays replay-deterministic by always scheduling.
    """
    agent = _make_mcp_replay_agent(cache_tools=False)
    mcp_replay_holder['agent'] = agent
    try:
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[MCPReplayWorkflow],
            activities=agent.temporal_activities,
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            wf_id = f'{MCPReplayWorkflow.__name__}_no_cache'
            await client.execute_workflow(MCPReplayWorkflow.run, args=['hello'], id=wf_id, task_queue=TASK_QUEUE)
            history = await client.get_workflow_handle(wf_id).fetch_history()
        assert _scheduled_get_tools_count(history) == 2
    finally:
        mcp_replay_holder['agent'] = _make_mcp_replay_agent()


async def test_complex_agent_run(allow_model_requests: None):
    events: list[AgentStreamEvent] = []

    async def event_stream_handler(
        ctx: RunContext[Deps],
        stream: AsyncIterable[AgentStreamEvent],
    ):
        async for event in stream:
            events.append(event)

    with complex_temporal_agent.override(deps=Deps(country='Mexico')):
        result = await complex_temporal_agent.run(
            'Tell me: the capital of the country; the weather there; the product name',
            deps=Deps(country='The Netherlands'),
            event_stream_handler=event_stream_handler,
        )
    assert result.output == snapshot(
        Response(
            answers=[
                Answer(label='Capital', answer='The capital of Mexico is Mexico City.'),
                Answer(label='Weather', answer='The weather in Mexico City is currently sunny.'),
                Answer(label='Product Name', answer='The product name is Pydantic AI.'),
            ]
        )
    )
    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='get_country', args='', tool_call_id='call_q2UyBRP7eXNTzAoR8lEhjc9Z'),
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='{}', tool_call_id='call_q2UyBRP7eXNTzAoR8lEhjc9Z')
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='get_country', args='{}', tool_call_id='call_q2UyBRP7eXNTzAoR8lEhjc9Z'),
                next_part_kind='tool-call',
            ),
            PartStartEvent(
                index=1,
                part=ToolCallPart(tool_name='get_product_name', args='', tool_call_id='call_b51ijcpFkDiTQG1bQzsrmtW5'),
                previous_part_kind='tool-call',
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='{}', tool_call_id='call_b51ijcpFkDiTQG1bQzsrmtW5')
            ),
            PartEndEvent(
                index=1,
                part=ToolCallPart(
                    tool_name='get_product_name', args='{}', tool_call_id='call_b51ijcpFkDiTQG1bQzsrmtW5'
                ),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(tool_name='get_country', args='{}', tool_call_id='call_q2UyBRP7eXNTzAoR8lEhjc9Z'),
                args_valid=True,
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name='get_product_name', args='{}', tool_call_id='call_b51ijcpFkDiTQG1bQzsrmtW5'
                ),
                args_valid=True,
            ),
            FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name='get_country',
                    content='Mexico',
                    tool_call_id='call_q2UyBRP7eXNTzAoR8lEhjc9Z',
                    timestamp=IsDatetime(),
                )
            ),
            FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name='get_product_name',
                    content='Pydantic AI',
                    tool_call_id='call_b51ijcpFkDiTQG1bQzsrmtW5',
                    timestamp=IsDatetime(),
                )
            ),
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='get_weather', args='', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv'),
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='{"', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='city', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":"', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='Mexico', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' City', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='"}', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv')
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(
                    tool_name='get_weather', args='{"city":"Mexico City"}', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv'
                ),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name='get_weather', args='{"city":"Mexico City"}', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv'
                ),
                args_valid=True,
            ),
            FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name='get_weather',
                    content='sunny',
                    tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv',
                    timestamp=IsDatetime(),
                )
            ),
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='final_result', args='', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn'),
            ),
            FinalResultEvent(tool_name='final_result', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn'),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='{"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='answers', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":[', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='{"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='label', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='Capital', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='","', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='answer', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='The', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' capital', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' of', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' Mexico', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' is', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' Mexico', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' City', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='."', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='},{"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='label', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='Weather', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='","', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='answer', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='The', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' weather', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' in', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' Mexico', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' City', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' is', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' currently', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' sunny', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='."', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='},{"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='label', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='Product', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' Name', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='","', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='answer', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='The', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' product', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' name', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' is', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' P', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='yd', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='antic', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' AI', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='."', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='}', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=']}', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(
                    tool_name='final_result',
                    args='{"answers":[{"label":"Capital","answer":"The capital of Mexico is Mexico City."},{"label":"Weather","answer":"The weather in Mexico City is currently sunny."},{"label":"Product Name","answer":"The product name is Pydantic AI."}]}',
                    tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn',
                ),
            ),
            OutputToolCallEvent(
                part=ToolCallPart(
                    tool_name='final_result',
                    args='{"answers":[{"label":"Capital","answer":"The capital of Mexico is Mexico City."},{"label":"Weather","answer":"The weather in Mexico City is currently sunny."},{"label":"Product Name","answer":"The product name is Pydantic AI."}]}',
                    tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn',
                ),
                args_valid=True,
            ),
            OutputToolResultEvent(
                part=ToolReturnPart(
                    tool_name='final_result',
                    content='Final result processed.',
                    tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn',
                    timestamp=IsDatetime(),
                )
            ),
        ]
    )


async def test_multiple_agents(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflow, ComplexAgentWorkflow],
        plugins=[AgentPlugin(simple_temporal_agent), AgentPlugin(complex_temporal_agent)],
    ):
        output = await client.execute_workflow(
            SimpleAgentWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=SimpleAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('The capital of Mexico is Mexico City.')

        output = await client.execute_workflow(
            ComplexAgentWorkflow.run,
            args=[
                'Tell me: the capital of the country; the weather there; the product name',
                Deps(country='Mexico'),
            ],
            id=ComplexAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            Response(
                answers=[
                    Answer(label='Capital of the Country', answer='Mexico City'),
                    Answer(label='Weather in Mexico City', answer='Sunny'),
                    Answer(label='Product Name', answer='Pydantic AI'),
                ]
            )
        )


async def test_agent_name_collision(allow_model_requests: None, client: Client):
    with pytest.raises(ValueError, match='More than one activity named agent__simple_agent__event_stream_handler'):
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[SimpleAgentWorkflow],
            plugins=[AgentPlugin(simple_temporal_agent), AgentPlugin(simple_temporal_agent)],
        ):
            pass


async def test_agent_without_name():
    with pytest.raises(
        UserError,
        match=re.escape(
            "An agent needs to have a unique `name` in order to be used with Temporal. The name will be used to identify the agent's activities within the workflow."
        ),
    ):
        TemporalAgent(Agent())


async def test_agent_without_model():
    with pytest.raises(
        UserError,
        match=re.escape(
            "The wrapped agent's `model` or the TemporalAgent's `models` parameter must provide at least one Model instance to be used with Temporal. Models cannot be set at agent run time."
        ),
    ):
        TemporalAgent(Agent(name='test_agent'))


async def test_old_temporalize_toolset_func_compat():
    """Old 6-arg temporalize_toolset_func implementations still work."""
    from pydantic_ai.durable_exec.temporal._toolset import temporalize_toolset

    def old_style_func(
        toolset: Any, prefix: Any, config: Any, tool_config: Any, deps_type: Any, run_context_type: Any
    ) -> Any:
        return temporalize_toolset(toolset, prefix, config, tool_config, deps_type, run_context_type)

    TemporalAgent(
        Agent(model=model, name='old_compat_agent'),
        activity_config=BASE_ACTIVITY_CONFIG,
        temporalize_toolset_func=old_style_func,  # pyright: ignore[reportArgumentType]
    )


async def test_toolset_without_id():
    with pytest.raises(
        UserError,
        match=re.escape(
            "Toolsets that are 'leaves' (i.e. those that implement their own tool listing and calling) need to have a unique `id` in order to be used with Temporal. The ID will be used to identify the toolset's activities within the workflow."
        ),
    ):
        TemporalAgent(Agent(model=model, name='test_agent', toolsets=[FunctionToolset()]))


# --- DynamicToolset / @agent.toolset tests ---


@dataclass
class DynamicToolsetDeps:
    user_name: str


dynamic_toolset_agent = Agent(TestModel(), name='dynamic_toolset_agent', deps_type=DynamicToolsetDeps)


@dynamic_toolset_agent.toolset(id='my_dynamic_tools')
def my_dynamic_toolset(ctx: RunContext[DynamicToolsetDeps]) -> FunctionToolset[DynamicToolsetDeps]:
    toolset = FunctionToolset[DynamicToolsetDeps](id='dynamic_weather')

    @toolset.tool_plain
    def get_dynamic_weather(location: str) -> str:
        """Get the weather for a location."""
        user = ctx.deps.user_name
        return f'Weather in {location} for {user}: sunny.'

    return toolset


dynamic_toolset_temporal_agent = TemporalAgent(
    dynamic_toolset_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class DynamicToolsetAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: DynamicToolsetDeps) -> str:
        result = await dynamic_toolset_temporal_agent.run(prompt, deps=deps)
        return result.output


async def test_dynamic_toolset_in_workflow(client: Client):
    """Test that @agent.toolset works correctly in a Temporal workflow."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DynamicToolsetAgentWorkflow],
        plugins=[AgentPlugin(dynamic_toolset_temporal_agent)],
    ):
        output = await client.execute_workflow(
            DynamicToolsetAgentWorkflow.run,
            args=['Get the weather for London', DynamicToolsetDeps(user_name='Alice')],
            id='test_dynamic_toolset_workflow',
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('{"get_dynamic_weather":"Weather in a for Alice: sunny."}')


async def test_dynamic_toolset_outside_workflow():
    """Test that the dynamic toolset agent works correctly outside of a workflow."""
    result = await dynamic_toolset_temporal_agent.run(
        'Get the weather for Paris', deps=DynamicToolsetDeps(user_name='Bob')
    )
    assert result.output == snapshot('{"get_dynamic_weather":"Weather in a for Bob: sunny."}')


# --- DynamicToolset.get_instructions test (issue #5282) ---
# A dynamic toolset whose resolved toolset implements `get_instructions()` must contribute those
# instructions under `TemporalAgent`, resolved inside an activity like `get_tools`.


def _echo_instructions(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    request = message(messages, ModelRequest, index=-1)
    return ModelResponse(parts=[TextPart(request.instructions or '<no instructions>')])


dynamic_instructions_agent = Agent(FunctionModel(_echo_instructions), name='dynamic_instructions_agent')


@dynamic_instructions_agent.toolset(id='dynamic_instruction_toolset', per_run_step=False)
def dynamic_instruction_toolset(ctx: RunContext[object]) -> AbstractToolset[object]:
    # A toolset that only contributes instructions, no tools.
    return FunctionToolset(instructions='SENTINEL_INSTRUCTION_FROM_DYNAMIC_TOOLSET', id='instruction-only-toolset')


dynamic_instructions_temporal_agent = TemporalAgent(
    dynamic_instructions_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class DynamicInstructionsAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await dynamic_instructions_temporal_agent.run(prompt)
        return result.output


async def test_dynamic_toolset_instructions_in_workflow(allow_model_requests: None, client: Client):
    """A dynamic toolset's `get_instructions()` reaches the model under `TemporalAgent` (issue #5282).

    The model echoes the request's instructions back as its output, so the sentinel in the output
    proves the resolved dynamic toolset's instructions were collected via the new activity.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DynamicInstructionsAgentWorkflow],
        plugins=[AgentPlugin(dynamic_instructions_temporal_agent)],
    ):
        output = await client.execute_workflow(
            DynamicInstructionsAgentWorkflow.run,
            args=['hello'],
            id='test_dynamic_toolset_instructions_workflow',
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('SENTINEL_INSTRUCTION_FROM_DYNAMIC_TOOLSET')


def test_dynamic_toolset_temporal_activities():
    """`TemporalDynamicToolset` collects instructions inside `get_tools`, so it has no separate `get_instructions` activity."""
    activity_names = {
        ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        for activity in dynamic_instructions_temporal_agent.temporal_activities
    }
    prefix = 'agent__dynamic_instructions_agent__dynamic_toolset__dynamic_instruction_toolset'
    assert {f'{prefix}__get_tools', f'{prefix}__call_tool'} <= activity_names
    assert f'{prefix}__get_instructions' not in activity_names


# --- DynamicToolset instructions refresh across run steps (issue #5282 follow-up) ---
# The per-run instructions cache is written by `get_tools` and read by `get_instructions` each
# step; this guards against it serving a stale step-1 value on a later step.


def _echo_instructions_after_tool_call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # First request: call a tool to force a second model-request step.
    # Second request (carrying the tool return): echo the instructions, which by then must
    # reflect the current step — proving the cache is repopulated by `get_tools` each step.
    request = message(messages, ModelRequest, index=-1)
    if any(isinstance(part, ToolReturnPart) for part in request.parts):
        return ModelResponse(parts=[TextPart(request.instructions or '<no instructions>')])
    return ModelResponse(parts=[ToolCallPart('noop', {})])


multi_step_instructions_agent = Agent(
    FunctionModel(_echo_instructions_after_tool_call), name='multi_step_instructions_agent'
)


@multi_step_instructions_agent.toolset(id='multi_step_instruction_toolset')
def multi_step_instruction_toolset(ctx: RunContext[object]) -> AbstractToolset[object]:
    # Instructions encode the run step, so a stale step-1 cached value read at step 2 would
    # surface as the wrong sentinel in the model output.
    toolset = FunctionToolset[object](
        instructions=f'INSTRUCTIONS_FOR_STEP_{ctx.run_step}', id='step-instruction-toolset'
    )

    @toolset.tool_plain
    def noop() -> str:
        return 'noop'

    return toolset


multi_step_instructions_temporal_agent = TemporalAgent(
    multi_step_instructions_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class MultiStepInstructionsAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await multi_step_instructions_temporal_agent.run(prompt)
        return result.output


async def test_dynamic_toolset_instructions_refresh_across_steps_in_workflow(
    allow_model_requests: None, client: Client
):
    """A dynamic toolset's instructions are refreshed each run step under `TemporalAgent` (issue #5282).

    The toolset encodes the run step in its instructions; the model calls a tool on the first request to
    force a second step, then echoes the instructions on the second request. The output being the step-2
    sentinel (not the step-1 one) proves `get_tools` repopulates the per-run instructions cache each step
    rather than serving a stale step-1 value.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiStepInstructionsAgentWorkflow],
        plugins=[AgentPlugin(multi_step_instructions_temporal_agent)],
    ):
        output = await client.execute_workflow(
            MultiStepInstructionsAgentWorkflow.run,
            args=['hello'],
            id='test_dynamic_toolset_instructions_refresh_workflow',
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('INSTRUCTIONS_FOR_STEP_2')


# --- DynamicToolset instructions replay determinism (issue #5282) ---
# The per-run instructions cache lives on a `for_run` copy of the wrapper rather than on the
# process-shared, module-level instance. A history recorded on one worker must replay on a
# freshly-constructed (cold) one, proving the `for_run` override reconstructs identically and
# introduces no `TMPRL1100` nondeterminism.

# A holder lets the replay step swap in a freshly-constructed (cold-process) instance.
dynamic_instructions_replay_holder: dict[str, TemporalAgent[object, str]] = {}


def _make_dynamic_instructions_replay_agent() -> TemporalAgent[object, str]:
    agent = Agent(FunctionModel(_echo_instructions_after_tool_call), name='dynamic_instructions_replay_agent')

    @agent.toolset(id='replay_instruction_toolset')
    def _replay_toolset(ctx: RunContext[object]) -> AbstractToolset[object]:
        toolset = FunctionToolset[object](
            instructions=f'INSTRUCTIONS_FOR_STEP_{ctx.run_step}', id='step-instruction-toolset'
        )

        @toolset.tool_plain
        def noop() -> str:
            return 'noop'

        return toolset

    return TemporalAgent(agent, activity_config=BASE_ACTIVITY_CONFIG)


dynamic_instructions_replay_holder['agent'] = _make_dynamic_instructions_replay_agent()


@workflow.defn
class DynamicInstructionsReplayWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await dynamic_instructions_replay_holder['agent'].run(prompt)
        return result.output


async def test_dynamic_toolset_instructions_replay_deterministic(allow_model_requests: None, client: Client):
    """The per-run `for_run` instructions cache must be replay-deterministic (issue #5282).

    Instructions resolved by `get_tools` are held on a per-run `for_run` copy of the wrapper, not
    on the module-level instance. This records a two-step workflow (instructions differ per step)
    and replays its history on a freshly-constructed cold instance — the worker-restart scenario —
    asserting no nondeterminism, so the `for_run` copy is reconstructed identically on replay.
    """
    warm = _make_dynamic_instructions_replay_agent()
    dynamic_instructions_replay_holder['agent'] = warm

    # Unsandboxed so the module-level instance is shared across the run exactly as a long-running
    # worker process shares it in production.
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DynamicInstructionsReplayWorkflow],
        activities=warm.temporal_activities,
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        wf_id = DynamicInstructionsReplayWorkflow.__name__
        output = await client.execute_workflow(
            DynamicInstructionsReplayWorkflow.run, args=['hello'], id=wf_id, task_queue=TASK_QUEUE
        )
        assert output == snapshot('INSTRUCTIONS_FOR_STEP_2')
        history = await client.get_workflow_handle(wf_id).fetch_history()

    # Warm-recorded history replayed on a freshly-constructed cold instance (worker-restart trigger).
    dynamic_instructions_replay_holder['agent'] = _make_dynamic_instructions_replay_agent()
    try:
        await Replayer(
            workflows=[DynamicInstructionsReplayWorkflow],
            workflow_runner=UnsandboxedWorkflowRunner(),
            data_converter=pydantic_data_converter,
        ).replay_workflow(history)
    finally:
        dynamic_instructions_replay_holder['agent'] = warm


# --- MCP-based DynamicToolset test ---
# Tests that @agent.toolset returning an MCPToolset works with Temporal workflows.
# Uses an HTTP-based MCP server rather than subprocess-based since the subprocess transports
# don't play nicely with Temporal's sandbox.


mcptoolset_dynamic_toolset_agent = Agent(model, name='mcptoolset_dynamic_toolset_agent')


@mcptoolset_dynamic_toolset_agent.toolset(id='mcptoolset_dynamic')
def my_mcptoolset_dynamic_toolset(ctx: RunContext) -> MCPToolset:
    """Dynamic toolset that returns an `MCPToolset` — exercises lifecycle + `TemporalMCPToolset`."""
    return MCPToolset('https://mcp.deepwiki.com/mcp')


mcptoolset_dynamic_toolset_temporal_agent = TemporalAgent(
    mcptoolset_dynamic_toolset_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class MCPToolsetDynamicToolsetAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await mcptoolset_dynamic_toolset_temporal_agent.run(prompt)
        return result.output


async def test_mcptoolset_dynamic_toolset_in_workflow(allow_model_requests: None, client: Client):
    """`@agent.toolset` returning an `MCPToolset` works in a Temporal workflow.

    Verifies the `MCPToolset`/`TemporalMCPToolset` pair handles `DynamicToolset` lifecycle
    (entering/exiting the context manager around each activity invocation).
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MCPToolsetDynamicToolsetAgentWorkflow],
        plugins=[AgentPlugin(mcptoolset_dynamic_toolset_temporal_agent)],
    ):
        output = await client.execute_workflow(
            MCPToolsetDynamicToolsetAgentWorkflow.run,
            args=['Can you tell me about the pydantic/pydantic-ai repo? Keep it short.'],
            id='test_mcptoolset_dynamic_toolset_workflow',
            task_queue=TASK_QUEUE,
        )
        assert 'pydantic' in output.lower() or 'agent' in output.lower()


# Regression test for the workflow-sandbox passthrough list (`_workflow_runner` in
# `durable_exec/temporal/__init__.py`). A `gateway/` model named by string is constructed lazily via
# `infer_model` *inside* the workflow, so the provider's SDK is imported and its client built under
# the `SandboxedWorkflowRunner`. Provider SDKs touch the filesystem/env at construction time, which
# the sandbox forbids unless the SDK module is passed through. Every other test builds its model at
# module scope (outside the sandbox), so this seam was previously uncovered. Construction-only (no
# model request) keeps it deterministic.
@workflow.defn
class ConstructModelInWorkflow:
    @workflow.run
    async def run(self, model_name: str) -> str:
        # We assert only that construction succeeds — no request is made.
        return type(infer_model(model_name)).__name__


@pytest.mark.parametrize(
    ('model_name', 'expected_model_class'),
    [
        # Only `gateway/` providers exercise the sandbox: they import their SDK lazily inside
        # `gateway_provider()`, so the import and client construction run *inside* the workflow. Direct
        # providers (e.g. `anthropic:`) import their SDK at module level, which rides Temporal's
        # transitive passthrough of `pydantic_ai` and never trips — so they give no regression coverage.
        #
        # The reported regression: `gateway/anthropic:` in-workflow tripped the `anthropic` SDK's
        # `Path.home()` access.
        pytest.param('gateway/anthropic:claude-sonnet-4-6', 'AnthropicModel', id='gateway-anthropic'),
        # Canary: OpenAI needs no passthrough today; turns red here (not in a user's workflow) if a
        # future SDK release makes a restricted call (e.g. reads `~/...`) during construction.
        pytest.param('gateway/openai-chat:gpt-5', 'OpenAIChatModel', id='gateway-openai'),
        # Positive coverage of the `google.auth` (+`certifi`) passthrough: `google-genai` lazily
        # imports `google.auth` during construction, which the sandbox flags without it.
        pytest.param('gateway/google-cloud:gemini-2.5-pro', 'GoogleModel', id='gateway-google'),
    ],
)
async def test_model_construction_in_workflow_passes_sandbox(
    model_name: str,
    expected_model_class: str,
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
):
    # Dummy credentials suffice since no request is made. The gateway key must encode a region
    # (`pylf_v<n>_<region>_...`) so the base URL can be inferred.
    monkeypatch.setenv('PYDANTIC_AI_GATEWAY_API_KEY', 'pylf_v1_us_0123456789abcdef')

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ConstructModelInWorkflow],
        # A sandbox violation surfaces as a workflow *task* failure, which Temporal retries forever
        # by default — so a regression would hang rather than fail. Promote any in-workflow exception
        # (e.g. `RestrictedWorkflowAccessError`) to a workflow failure so it surfaces immediately.
        workflow_failure_exception_types=[Exception],
    ):
        # Without the SDK passed through this fails with a `WorkflowFailureError`: under the suite's
        # warnings-as-errors, Temporal's "imported after initial workflow load" becomes a hard error;
        # in production the SDK's restricted `Path.home()`/env access raises `RestrictedWorkflowAccessError`.
        result = await client.execute_workflow(
            ConstructModelInWorkflow.run,
            args=[model_name],
            id=f'construct_model_{re.sub(r"[^a-zA-Z0-9]", "_", model_name)}',
            task_queue=TASK_QUEUE,
        )
    assert result == expected_model_class


# Regression test for the `genai_prices`/`httpx2` passthrough entries in `_workflow_runner`.
# `ModelResponse.cost()` lazily imports genai-prices on first call; inside a workflow that trips the
# sandbox unless those modules are passed through (see #6215).
@workflow.defn
class CalculateCostInWorkflow:
    @workflow.run
    async def run(self) -> float:
        response = ModelResponse(
            parts=[TextPart('ok')],
            usage=RequestUsage(input_tokens=100, output_tokens=10),
            model_name='claude-sonnet-4-5',
            provider_name='anthropic',
        )
        return float(response.cost().total_price)


async def test_response_cost_in_workflow_passes_sandbox(client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CalculateCostInWorkflow],
        workflow_failure_exception_types=[Exception],
    ):
        result = await client.execute_workflow(
            CalculateCostInWorkflow.run,
            id='calculate_cost_in_workflow',
            task_queue=TASK_QUEUE,
        )
    assert result > 0


async def test_temporal_agent():
    assert isinstance(complex_temporal_agent.model, TemporalModel)
    assert complex_temporal_agent.model.wrapped == complex_agent.model

    toolsets = complex_temporal_agent.toolsets
    assert len(toolsets) == 5

    # Empty function toolset for the agent's own tools
    assert isinstance(toolsets[0], FunctionToolset)
    assert toolsets[0].id == '<agent>'
    assert toolsets[0].tools == {}

    # Wrapped function toolset for the agent's own tools
    assert isinstance(toolsets[1], TemporalFunctionToolset)
    assert toolsets[1].id == '<agent>'
    assert isinstance(toolsets[1].wrapped, FunctionToolset)
    assert toolsets[1].wrapped.tools.keys() == {'get_weather'}

    # Wrapped 'country' toolset
    assert isinstance(toolsets[2], TemporalFunctionToolset)
    assert toolsets[2].id == 'country'
    assert toolsets[2].wrapped == complex_agent.toolsets[1]
    assert isinstance(toolsets[2].wrapped, FunctionToolset)
    assert toolsets[2].wrapped.tools.keys() == {'get_country'}

    # Wrapped 'mcp' MCP server
    assert isinstance(toolsets[3], TemporalMCPToolset)
    assert toolsets[3].id == 'mcp'
    assert toolsets[3].wrapped == complex_agent.toolsets[2]

    # Unwrapped 'external' toolset
    assert isinstance(toolsets[4], ExternalToolset)
    assert toolsets[4].id == 'external'
    assert toolsets[4] == complex_agent.toolsets[3]

    assert [
        ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        for activity in complex_temporal_agent.temporal_activities
    ] == snapshot(
        [
            'agent__complex_agent__event_stream_handler',
            'agent__complex_agent__model_request',
            'agent__complex_agent__model_request_stream',
            'agent__complex_agent__model_cancel_suspended_response',
            'agent__complex_agent__toolset__<agent>__call_tool',
            'agent__complex_agent__toolset__country__call_tool',
            'agent__complex_agent__mcp_server__mcp__get_instructions',
            'agent__complex_agent__mcp_server__mcp__get_tools',
            'agent__complex_agent__mcp_server__mcp__call_tool',
        ]
    )


def test_temporal_model_request_activities_capture_deps_type():
    """Both model-request activities must capture the real `deps_type` as the `deps` argument type.

    `temporalio`'s `@activity.defn` freezes a function's type hints into `arg_types` at decoration time for
    payload conversion, so `deps`'s annotation has to be set before decorating. If it's set afterwards (as the
    non-streaming activity used to do), the patch is cosmetic and the activity deserializes `deps` as a raw
    dict instead of the declared deps type.
    """
    model = dynamic_toolset_temporal_agent.model
    assert isinstance(model, TemporalModel)

    # `arg_types[1]` is the `deps` argument's captured type, which drives Temporal's payload conversion.
    deps_type = DynamicToolsetDeps | None
    request_arg_types = ActivityDefinition.must_from_callable(model.request_activity).arg_types  # pyright: ignore[reportUnknownMemberType]
    stream_arg_types = ActivityDefinition.must_from_callable(model.request_stream_activity).arg_types  # pyright: ignore[reportUnknownMemberType]
    assert request_arg_types is not None and request_arg_types[1] == deps_type
    assert stream_arg_types is not None and stream_arg_types[1] == deps_type


def test_temporal_wrapper_visit_and_replace():
    """Temporal wrapper toolsets should not be replaced by visit_and_replace."""
    from pydantic_ai.durable_exec.temporal._function_toolset import TemporalFunctionToolset

    toolsets = complex_temporal_agent._toolsets  # pyright: ignore[reportPrivateUsage]
    temporal_function_toolsets = [ts for ts in toolsets if isinstance(ts, TemporalFunctionToolset)]
    assert len(temporal_function_toolsets) >= 1

    temporal_function_toolset = temporal_function_toolsets[0]

    # visit_and_replace should return self for temporal wrappers
    result = temporal_function_toolset.visit_and_replace(lambda t: FunctionToolset(id='replaced'))
    assert result is temporal_function_toolset


async def test_temporal_agent_run(allow_model_requests: None):
    result = await simple_temporal_agent.run('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is Mexico City.')


def test_temporal_agent_run_sync(allow_model_requests: None):
    result = simple_temporal_agent.run_sync('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is Mexico City.')


async def test_temporal_agent_run_stream(allow_model_requests: None):
    async with simple_temporal_agent.run_stream('What is the capital of Mexico?') as result:
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


async def test_temporal_agent_run_stream_events(allow_model_requests: None):
    async with simple_temporal_agent.run_stream_events('What is the capital of Mexico?') as event_stream:
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


async def test_temporal_agent_iter(allow_model_requests: None):
    output: list[str] = []
    async with simple_temporal_agent.iter('What is the capital of Mexico?') as run:
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


@workflow.defn
class SimpleAgentWorkflowWithRunSync:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = simple_temporal_agent.run_sync(prompt)
        return result.output  # pragma: no cover


async def test_temporal_agent_run_sync_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithRunSync],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot('`agent.run_sync()` cannot be used inside a Temporal workflow. Use `await agent.run()` instead.'),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithRunSync.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithRunSync.__name__,
                task_queue=TASK_QUEUE,
            )


def drop_first_message(msgs: list[ModelMessage]) -> list[ModelMessage]:
    return msgs[1:] if len(msgs) > 1 else msgs


agent_with_sync_history_processor = Agent(
    model, name='agent_with_sync_history_processor', capabilities=[ProcessHistory(drop_first_message)]
)
temporal_agent_with_sync_history_processor = TemporalAgent(
    agent_with_sync_history_processor, activity_config=BASE_ACTIVITY_CONFIG
)


@workflow.defn
class AgentWithSyncHistoryProcessorWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_agent_with_sync_history_processor.run(prompt)
        return result.output


async def test_temporal_agent_with_sync_history_processor(allow_model_requests: None, client: Client):
    """Test that sync history processors work inside Temporal workflows.

    This validates that the _disable_threads ContextVar is properly set
    by TemporalAgent._temporal_overrides(), allowing sync history processors to
    execute without triggering NotImplementedError from anyio.to_thread.run_sync.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWithSyncHistoryProcessorWorkflow],
        plugins=[AgentPlugin(temporal_agent_with_sync_history_processor)],
    ):
        output = await client.execute_workflow(
            AgentWithSyncHistoryProcessorWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=AgentWithSyncHistoryProcessorWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('The capital of Mexico is Mexico City.')


agent_with_sync_instructions = Agent(model, name='agent_with_sync_instructions')


@agent_with_sync_instructions.instructions
def sync_instructions_fn() -> str:
    return 'You are a helpful assistant.'


temporal_agent_with_sync_instructions = TemporalAgent(
    agent_with_sync_instructions, activity_config=BASE_ACTIVITY_CONFIG
)


@workflow.defn
class AgentWithSyncInstructionsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_agent_with_sync_instructions.run(prompt)
        return result.output


async def test_temporal_agent_with_sync_instructions(allow_model_requests: None, client: Client):
    """Test that sync instructions functions work inside Temporal workflows.

    This validates that the _disable_threads ContextVar is properly set
    by TemporalAgent._temporal_overrides(), allowing sync instructions functions to
    execute without triggering NotImplementedError from anyio.to_thread.run_sync.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWithSyncInstructionsWorkflow],
        plugins=[AgentPlugin(temporal_agent_with_sync_instructions)],
    ):
        output = await client.execute_workflow(
            AgentWithSyncInstructionsWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=AgentWithSyncInstructionsWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('The capital of Mexico is Mexico City.')


@workflow.defn
class SimpleAgentWorkflowWithRunStream:
    @workflow.run
    async def run(self, prompt: str) -> str:
        async with simple_temporal_agent.run_stream(prompt) as result:
            pass
        return await result.get_output()  # pragma: no cover


async def test_temporal_agent_run_stream_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithRunStream],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                '`agent.run_stream()` cannot be used inside a Temporal workflow. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithRunStream.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithRunStream.__name__,
                task_queue=TASK_QUEUE,
            )


@workflow.defn
class SimpleAgentWorkflowWithRunStreamEvents:
    @workflow.run
    async def run(self, prompt: str) -> list[AgentStreamEvent | AgentRunResultEvent]:
        async with simple_temporal_agent.run_stream_events(prompt) as event_stream:
            return [event async for event in event_stream]  # pragma: no cover


async def test_temporal_agent_run_stream_events_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithRunStreamEvents],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                '`agent.run_stream_events()` cannot be used inside a Temporal workflow. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithRunStreamEvents.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithRunStreamEvents.__name__,
                task_queue=TASK_QUEUE,
            )


@workflow.defn
class SimpleAgentWorkflowWithIter:
    @workflow.run
    async def run(self, prompt: str) -> str:
        async with simple_temporal_agent.iter(prompt) as run:
            async for _ in run:
                pass
        return 'done'  # pragma: no cover


async def test_temporal_agent_iter_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithIter],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                '`agent.iter()` cannot be used inside a Temporal workflow. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithIter.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithIter.__name__,
                task_queue=TASK_QUEUE,
            )


async def simple_event_stream_handler(
    ctx: RunContext,
    stream: AsyncIterable[AgentStreamEvent],
):
    pass


@workflow.defn
class SimpleAgentWorkflowWithEventStreamHandler:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_temporal_agent.run(prompt, event_stream_handler=simple_event_stream_handler)
        return result.output  # pragma: no cover


async def test_temporal_agent_run_in_workflow_with_event_stream_handler(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithEventStreamHandler],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Event stream handler cannot be set at agent run time inside a Temporal workflow, it must be set at agent creation time.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithEventStreamHandler.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithEventStreamHandler.__name__,
                task_queue=TASK_QUEUE,
            )


# Unregistered model instance for testing error case
unregistered_model = OpenAIChatModel(
    'gpt-4o-mini',
    provider=OpenAIProvider(
        api_key=os.getenv('OPENAI_API_KEY', 'mock-api-key'),
        http_client=http_client,
    ),
)


@workflow.defn
class SimpleAgentWorkflowWithRunModel:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_temporal_agent.run(prompt, model=unregistered_model)
        return result.output  # pragma: no cover


async def test_temporal_agent_run_in_workflow_with_model(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithRunModel],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Arbitrary model instances cannot be used at runtime inside a Temporal workflow. Register the model via `models` or reference a registered model by id.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithRunModel.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithRunModel.__name__,
                task_queue=TASK_QUEUE,
            )


@workflow.defn
class SimpleAgentWorkflowWithRunToolsets:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_temporal_agent.run(prompt, toolsets=[FunctionToolset()])
        return result.output  # pragma: no cover


async def test_temporal_agent_run_in_workflow_with_executing_toolsets(allow_model_requests: None, client: Client):
    # Executing toolsets (here a `FunctionToolset`) can't be added per-run because their activities must
    # be registered with the worker before the workflow runs.
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithRunToolsets],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'FunctionToolset cannot be passed to `run(toolsets=...)` at runtime with Temporal, because '
                'toolsets that execute their own tools or resolve dynamically must be registered for durable '
                'execution when the agent is constructed. Pass them to the agent constructor instead. '
                'Non-executing toolsets like `ExternalToolset` can be passed at runtime.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithRunToolsets.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithRunToolsets.__name__,
                task_queue=TASK_QUEUE,
            )


def request_runtime_external_tool(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('external', {'query': 'runtime'}, tool_call_id='call-1')])


runtime_external_agent = Agent(
    FunctionModel(request_runtime_external_tool),
    name='runtime_external_toolset_agent',
    output_type=[str, DeferredToolRequests],
)
runtime_external_temporal_agent = TemporalAgent(runtime_external_agent, activity_config=BASE_ACTIVITY_CONFIG)

runtime_external_toolset = ExternalToolset(
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


@workflow.defn
class RuntimeExternalToolsetWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str | DeferredToolRequests]:
        return await runtime_external_temporal_agent.run(prompt, toolsets=[runtime_external_toolset])


async def test_temporal_agent_run_in_workflow_with_runtime_external_toolset(allow_model_requests: None, client: Client):
    # Non-executing toolsets like `ExternalToolset` need no durable wrapping, so they can be added per-run.
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RuntimeExternalToolsetWorkflow],
        plugins=[AgentPlugin(runtime_external_temporal_agent)],
    ):
        result = await client.execute_workflow(
            RuntimeExternalToolsetWorkflow.run,
            args=['Call the runtime external tool.'],
            id=RuntimeExternalToolsetWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert result.output == DeferredToolRequests(
            calls=[ToolCallPart('external', {'query': 'runtime'}, tool_call_id='call-1')]
        )


@workflow.defn
class SimpleAgentWorkflowWithOverrideModel:
    @workflow.run
    async def run(self, prompt: str) -> None:
        with simple_temporal_agent.override(model=model):
            pass


async def test_temporal_agent_override_model_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithOverrideModel],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Model cannot be contextually overridden inside a Temporal workflow, it must be set at agent creation time.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithOverrideModel.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithOverrideModel.__name__,
                task_queue=TASK_QUEUE,
            )


@workflow.defn
class SimpleAgentWorkflowWithOverrideToolsets:
    @workflow.run
    async def run(self, prompt: str) -> None:
        with simple_temporal_agent.override(toolsets=[FunctionToolset()]):
            pass


async def test_temporal_agent_override_toolsets_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithOverrideToolsets],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Toolsets cannot be contextually overridden inside a Temporal workflow, they must be set at agent creation time.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithOverrideToolsets.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithOverrideToolsets.__name__,
                task_queue=TASK_QUEUE,
            )


@workflow.defn
class SimpleAgentWorkflowWithOverrideTools:
    @workflow.run
    async def run(self, prompt: str) -> None:
        with simple_temporal_agent.override(tools=[get_weather]):
            pass


async def test_temporal_agent_override_tools_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithOverrideTools],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Tools cannot be contextually overridden inside a Temporal workflow, they must be set at agent creation time.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithOverrideTools.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithOverrideTools.__name__,
                task_queue=TASK_QUEUE,
            )


@workflow.defn
class SimpleAgentWorkflowWithOverrideBuiltinTools:
    @workflow.run
    async def run(self, prompt: str) -> None:
        with simple_temporal_agent.override(native_tools=[WebSearchTool()]):
            pass


async def test_temporal_agent_override_builtin_tools_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithOverrideBuiltinTools],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Native tools cannot be contextually overridden inside a Temporal workflow, they must be set at agent creation time.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithOverrideBuiltinTools.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithOverrideBuiltinTools.__name__,
                task_queue=TASK_QUEUE,
            )


@workflow.defn
class SimpleAgentWorkflowWithOverrideDeps:
    @workflow.run
    async def run(self, prompt: str) -> str:
        with simple_temporal_agent.override(deps=None):
            result = await simple_temporal_agent.run(prompt)
            return result.output


async def test_temporal_agent_override_deps_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithOverrideDeps],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        output = await client.execute_workflow(
            SimpleAgentWorkflowWithOverrideDeps.run,
            args=['What is the capital of Mexico?'],
            id=SimpleAgentWorkflowWithOverrideDeps.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('The capital of Mexico is Mexico City.')


agent_with_sync_tool = Agent(model, name='agent_with_sync_tool', tools=[get_weather])

# This needs to be done before the `TemporalAgent` is bound to the workflow.
temporal_agent_with_sync_tool_activity_disabled = TemporalAgent(
    agent_with_sync_tool,
    activity_config=BASE_ACTIVITY_CONFIG,
    tool_activity_config={
        '<agent>': {
            'get_weather': False,
        },
    },
)


@workflow.defn
class AgentWorkflowWithSyncToolActivityDisabled:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_agent_with_sync_tool_activity_disabled.run(prompt)
        return result.output  # pragma: no cover


async def test_temporal_agent_sync_tool_activity_disabled(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflowWithSyncToolActivityDisabled],
        plugins=[AgentPlugin(temporal_agent_with_sync_tool_activity_disabled)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                "Temporal activity config for tool 'get_weather' has been explicitly set to `False` (activity disabled), but non-async tools are run in threads which are not supported outside of an activity. Make the tool function async instead."
            ),
        ):
            await client.execute_workflow(
                AgentWorkflowWithSyncToolActivityDisabled.run,
                args=['What is the weather in Mexico City?'],
                id=AgentWorkflowWithSyncToolActivityDisabled.__name__,
                task_queue=TASK_QUEUE,
            )


async def test_temporal_agent_mcp_server_activity_disabled(client: Client):
    with pytest.raises(
        UserError,
        match=re.escape(
            "Temporal activity config for MCP tool 'get_product_name' has been explicitly set to `False` (activity disabled), "
            'but MCP tools require the use of IO and so cannot be run outside of an activity.'
        ),
    ):
        TemporalAgent(
            complex_agent,
            tool_activity_config={
                'mcp': {
                    'get_product_name': False,
                },
            },
        )


@workflow.defn
class DirectStreamWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        messages: list[ModelMessage] = [ModelRequest.user_text_prompt(prompt)]
        async with model_request_stream(complex_temporal_agent.model, messages) as stream:
            async for _ in stream:
                pass
        return 'done'  # pragma: no cover


async def test_temporal_model_stream_direct(client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DirectStreamWorkflow],
        plugins=[AgentPlugin(complex_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'A Temporal model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            ),
        ):
            await client.execute_workflow(
                DirectStreamWorkflow.run,
                args=['What is the capital of Mexico?'],
                id=DirectStreamWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )


unserializable_deps_agent = Agent(model, name='unserializable_deps_agent', deps_type=Model)


@unserializable_deps_agent.tool
async def get_model_name(ctx: RunContext[Model]) -> str:
    return ctx.deps.model_name  # pragma: no cover


# This needs to be done before the `TemporalAgent` is bound to the workflow.
unserializable_deps_temporal_agent = TemporalAgent(unserializable_deps_agent, activity_config=BASE_ACTIVITY_CONFIG)


@workflow.defn
class UnserializableDepsAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await unserializable_deps_temporal_agent.run(prompt, deps=unserializable_deps_temporal_agent.model)
        return result.output  # pragma: no cover


async def test_temporal_agent_with_unserializable_deps_type(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[UnserializableDepsAgentWorkflow],
        plugins=[AgentPlugin(unserializable_deps_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                "The `deps` object failed to be serialized. Temporal requires all objects that are passed to activities to be serializable using Pydantic's `TypeAdapter`."
            ),
        ):
            await client.execute_workflow(
                UnserializableDepsAgentWorkflow.run,
                args=['What is the model name?'],
                id=UnserializableDepsAgentWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )


async def test_logfire_plugin(client: Client):
    def setup_logfire(send_to_logfire: bool = True, metrics: Literal[False] | None = None) -> Logfire:
        instance = logfire.configure(local=True, metrics=metrics)
        instance.config.token = 'test'
        instance.config.send_to_logfire = send_to_logfire
        return instance

    plugin = LogfirePlugin(setup_logfire)

    config = client.config()
    config['plugins'] = [plugin]
    new_client = Client(**config)

    interceptor = new_client.config(active_config=True)['interceptors'][0]
    assert isinstance(interceptor, TracingInterceptor)
    if isinstance(interceptor.tracer, ProxyTracer):
        assert interceptor.tracer._instrumenting_module_name == 'temporalio'  # pyright: ignore[reportPrivateUsage] # pragma: lax no cover
    elif isinstance(interceptor.tracer, _ProxyTracer):
        assert interceptor.tracer.instrumenting_module_name == 'temporalio'  # pragma: lax no cover
    else:
        assert False, f'Unexpected tracer type: {type(interceptor.tracer)}'  # pragma: no cover

    new_client = await Client.connect(client.service_client.config.target_host, plugins=[plugin])
    # We can't check if the metrics URL was actually set correctly because it's on a `temporalio.bridge.runtime.Runtime` that we can't read from.
    assert new_client.service_client.config.runtime is not None

    plugin = LogfirePlugin(setup_logfire, metrics=False)
    new_client = await Client.connect(client.service_client.config.target_host, plugins=[plugin])
    assert new_client.service_client.config.runtime is None

    plugin = LogfirePlugin(lambda: setup_logfire(send_to_logfire=False))
    new_client = await Client.connect(client.service_client.config.target_host, plugins=[plugin])
    assert new_client.service_client.config.runtime is None

    plugin = LogfirePlugin(lambda: setup_logfire(metrics=False))
    new_client = await Client.connect(client.service_client.config.target_host, plugins=[plugin])
    assert new_client.service_client.config.runtime is None


hitl_agent = Agent(
    model,
    name='hitl_agent',
    output_type=[str, DeferredToolRequests],
    instructions='Just call tools without asking for confirmation.',
)


@hitl_agent.tool
async def create_file(ctx: RunContext, path: str) -> None:
    raise CallDeferred


@hitl_agent.tool
async def delete_file(ctx: RunContext, path: str) -> bool:
    if not ctx.tool_call_approved:
        raise ApprovalRequired
    return True


hitl_temporal_agent = TemporalAgent(hitl_agent, activity_config=BASE_ACTIVITY_CONFIG)


@workflow.defn
class HitlAgentWorkflow:
    def __init__(self):
        self._status: Literal['running', 'waiting_for_results', 'done'] = 'running'
        self._deferred_tool_requests: DeferredToolRequests | None = None
        self._deferred_tool_results: DeferredToolResults | None = None

    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str | DeferredToolRequests]:
        messages: list[ModelMessage] = [ModelRequest.user_text_prompt(prompt)]
        while True:
            result = await hitl_temporal_agent.run(
                message_history=messages, deferred_tool_results=self._deferred_tool_results
            )
            messages = result.all_messages()

            if isinstance(result.output, DeferredToolRequests):
                self._deferred_tool_requests = result.output
                self._deferred_tool_results = None
                self._status = 'waiting_for_results'

                await workflow.wait_condition(lambda: self._deferred_tool_results is not None)
                self._status = 'running'
            else:
                self._status = 'done'
                return result

    @workflow.query
    def get_status(self) -> Literal['running', 'waiting_for_results', 'done']:
        return self._status

    @workflow.query
    def get_deferred_tool_requests(self) -> DeferredToolRequests | None:
        return self._deferred_tool_requests

    @workflow.signal
    def set_deferred_tool_results(self, results: DeferredToolResults) -> None:
        self._status = 'running'
        self._deferred_tool_requests = None
        self._deferred_tool_results = results


async def test_temporal_agent_with_hitl_tool(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[HitlAgentWorkflow],
        plugins=[AgentPlugin(hitl_temporal_agent)],
    ):
        workflow = await client.start_workflow(
            HitlAgentWorkflow.run,
            args=['Delete the file `.env` and create `test.txt`'],
            id=HitlAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        while True:
            await asyncio.sleep(1)
            status = await workflow.query(HitlAgentWorkflow.get_status)
            if status == 'done':
                break
            elif status == 'waiting_for_results':  # pragma: no branch
                deferred_tool_requests = await workflow.query(HitlAgentWorkflow.get_deferred_tool_requests)
                assert deferred_tool_requests is not None

                results = DeferredToolResults()
                # Approve all calls
                for tool_call in deferred_tool_requests.approvals:
                    results.approvals[tool_call.tool_call_id] = True

                for tool_call in deferred_tool_requests.calls:
                    results.calls[tool_call.tool_call_id] = 'Success'

                await workflow.signal(HitlAgentWorkflow.set_deferred_tool_results, results)

        result = await workflow.result()
        assert result.output == snapshot(
            'The file `.env` has been deleted and `test.txt` has been created successfully.'
        )
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Delete the file `.env` and create `test.txt`',
                            timestamp=IsDatetime(),
                        )
                    ],
                    # NOTE in other tests we check timestamp=IsNow(tz=timezone.utc)
                    # but temporal tests fail when we use IsNow
                    timestamp=IsDatetime(),
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
                    provider_details={'finish_reason': 'tool_calls', 'timestamp': '2025-08-28T22:11:03Z'},
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
                    timestamp=IsDatetime(),
                    instructions='Just call tools without asking for confirmation.',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='The file `.env` has been deleted and `test.txt` has been created successfully.'
                        )
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
                    provider_details={'finish_reason': 'stop', 'timestamp': '2025-08-28T22:11:06Z'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


model_retry_agent = Agent(model, name='model_retry_agent')


@model_retry_agent.tool_plain
def get_weather_in_city(city: str) -> str:
    if city != 'Mexico City':
        raise ModelRetry('Did you mean Mexico City?')
    return 'sunny'


model_retry_temporal_agent = TemporalAgent(model_retry_agent, activity_config=BASE_ACTIVITY_CONFIG)


@workflow.defn
class ModelRetryWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str]:
        result = await model_retry_temporal_agent.run(prompt)
        return result


async def test_temporal_agent_with_model_retry(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ModelRetryWorkflow],
        plugins=[AgentPlugin(model_retry_temporal_agent)],
    ):
        workflow = await client.start_workflow(
            ModelRetryWorkflow.run,
            args=['What is the weather in CDMX?'],
            id=ModelRetryWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        result = await workflow.result()
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
                    timestamp=IsDatetime(),
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
                    provider_details={'finish_reason': 'tool_calls', 'timestamp': '2025-08-28T23:19:50Z'},
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
                    timestamp=IsDatetime(),
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
                    provider_details={'finish_reason': 'tool_calls', 'timestamp': '2025-08-28T23:19:51Z'},
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
                    timestamp=IsDatetime(),
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
                    provider_details={'finish_reason': 'stop', 'timestamp': '2025-08-28T23:19:52Z'},
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

# This needs to be done before the `TemporalAgent` is bound to the workflow.
settings_temporal_agent = TemporalAgent(settings_agent, activity_config=BASE_ACTIVITY_CONFIG)


@workflow.defn
class SettingsAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await settings_temporal_agent.run(prompt)
        return result.output


async def test_custom_model_settings(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SettingsAgentWorkflow],
        plugins=[AgentPlugin(settings_temporal_agent)],
    ):
        output = await client.execute_workflow(
            SettingsAgentWorkflow.run,
            args=['Give me those settings'],
            id=SettingsAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot("{'max_tokens': 123, 'custom_setting': 'custom_value'}")


def return_mcp_instructions(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(agent_info.instructions or '')])


# Exercises the `TemporalMCPToolset` wrapper's `get_instructions` activity path.
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

mcptoolset_instructions_temporal_agent = TemporalAgent(
    mcptoolset_instructions_agent, activity_config=BASE_ACTIVITY_CONFIG
)


@workflow.defn
class MCPToolsetInstructionsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await mcptoolset_instructions_temporal_agent.run(prompt)
        return result.output


async def test_temporal_mcptoolset_instructions_propagate(client: Client):
    """`MCPToolset` instructions propagate through the `TemporalMCPToolset` wrapper."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MCPToolsetInstructionsWorkflow],
        plugins=[AgentPlugin(mcptoolset_instructions_temporal_agent)],
    ):
        output = await client.execute_workflow(
            MCPToolsetInstructionsWorkflow.run,
            args=['Use MCP instructions'],
            id=MCPToolsetInstructionsWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('Be a helpful assistant.')


def test_temporalize_mcptoolset_dispatches_to_temporalmcptoolset():
    """`temporalize_toolset` wraps `MCPToolset` in `TemporalMCPToolset`."""
    toolset = MCPToolset('https://example.com/mcp', id='test_dispatch')
    agent = Agent(model=model, name='dispatch_agent', toolsets=[toolset])
    temporal = TemporalAgent(agent, activity_config=BASE_ACTIVITY_CONFIG)
    wrapped = next(ts for ts in temporal.toolsets if isinstance(ts, TemporalMCPToolset))
    assert wrapped.wrapped is toolset


image_agent = Agent(model, name='image_agent', output_type=BinaryImage)

# This needs to be done before the `TemporalAgent` is bound to the workflow.
image_temporal_agent = TemporalAgent(image_agent, activity_config=BASE_ACTIVITY_CONFIG)


@workflow.defn
class ImageAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> BinaryImage:
        result = await image_temporal_agent.run(prompt)
        return result.output  # pragma: no cover


async def test_image_agent(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ImageAgentWorkflow],
        plugins=[AgentPlugin(image_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot('Image output is not supported with Temporal because of the 2MB payload size limit.'),
        ):
            await client.execute_workflow(
                ImageAgentWorkflow.run,
                args=['Generate an image of an axolotl.'],
                id=ImageAgentWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )


# ============================================================================
# DocumentUrl Serialization Test - Verifies that DocumentUrl with custom
# media_type is properly serialized through Temporal activities
# ============================================================================

document_url_agent = Agent(
    TestModel(custom_output_args={'url': 'https://example.com/doc/12345', 'media_type': 'application/pdf'}),
    name='document_url_agent',
    output_type=DocumentUrl,
)

document_url_temporal_agent = TemporalAgent(document_url_agent, activity_config=BASE_ACTIVITY_CONFIG)


@workflow.defn
class DocumentUrlAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> DocumentUrl:
        result = await document_url_temporal_agent.run(prompt)
        return result.output


async def test_document_url_serialization_preserves_media_type(allow_model_requests: None, client: Client):
    """Test that `DocumentUrl` with custom `media_type` is preserved through Temporal serialization.

    This is a regression test for https://github.com/pydantic/pydantic-ai/issues/3949
    where `DocumentUrl.media_type` (a computed field) was lost during Temporal activity
    serialization because the backing field `_media_type` was excluded from serialization.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DocumentUrlAgentWorkflow],
        plugins=[AgentPlugin(document_url_temporal_agent)],
    ):
        output = await client.execute_workflow(
            DocumentUrlAgentWorkflow.run,
            args=['Return a document'],
            id=DocumentUrlAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            DocumentUrl(url='https://example.com/doc/12345', _media_type='application/pdf', _identifier='eb8998')
        )


# ============================================================================
# UploadedFile Serialization Test - Verifies that UploadedFile with custom
# media_type is properly serialized through Temporal activities
# ============================================================================

uploaded_file_agent = Agent(
    TestModel(
        custom_output_args={
            'file_id': 'file-abc123',
            'provider_name': 'openai',
            'media_type': 'image/png',
            'identifier': 'file-1',
        }
    ),
    name='uploaded_file_agent',
    output_type=UploadedFile,
)

uploaded_file_temporal_agent = TemporalAgent(uploaded_file_agent, activity_config=BASE_ACTIVITY_CONFIG)


@workflow.defn
class UploadedFileAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> UploadedFile:
        result = await uploaded_file_temporal_agent.run(prompt)
        return result.output


async def test_uploaded_file_serialization_preserves_media_type(allow_model_requests: None, client: Client):
    """Test that `UploadedFile` with custom `media_type` is preserved through Temporal serialization."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[UploadedFileAgentWorkflow],
        plugins=[AgentPlugin(uploaded_file_temporal_agent)],
    ):
        output = await client.execute_workflow(
            UploadedFileAgentWorkflow.run,
            args=['Return a file reference'],
            id=UploadedFileAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            UploadedFile(file_id='file-abc123', provider_name='openai', _media_type='image/png', _identifier='file-1')
        )


# Can't use the `openai_api_key` fixture here because the workflow needs to be defined at the top level of the file.
web_search_model = OpenAIResponsesModel(
    'gpt-5',
    provider=OpenAIProvider(
        api_key=os.getenv('OPENAI_API_KEY', 'mock-api-key'),
        http_client=http_client,
    ),
)

web_search_agent = Agent(
    web_search_model,
    name='web_search_agent',
    capabilities=[NativeTool(WebSearchTool(user_location=WebSearchUserLocation(city='Mexico City', country='MX')))],
)

# This needs to be done before the `TemporalAgent` is bound to the workflow.
web_search_temporal_agent = TemporalAgent(
    web_search_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
    model_activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=300)),
)


@workflow.defn
class WebSearchAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await web_search_temporal_agent.run(prompt)
        return result.output


async def test_web_search_agent_run_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WebSearchAgentWorkflow],
        plugins=[AgentPlugin(web_search_temporal_agent)],
    ):
        output = await client.execute_workflow(
            WebSearchAgentWorkflow.run,
            args=['In one sentence, what is the top news story in my country today?'],
            id=WebSearchAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            'Severe floods and landslides across Veracruz, Hidalgo, and Puebla have cut off hundreds of communities and left dozens dead and many missing, prompting a major federal emergency response. ([apnews.com](https://apnews.com/article/5d036e18057361281e984b44402d3b1b?utm_source=openai))'
        )


def test_temporal_run_context_preserves_run_id():
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id='run-123',
    )

    serialized = TemporalRunContext.serialize_run_context(ctx)
    assert serialized['run_id'] == 'run-123'

    reconstructed = TemporalRunContext.deserialize_run_context(serialized, deps=None)
    assert reconstructed.run_id == 'run-123'


def test_temporal_run_context_serializes_metadata():
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id='run-123',
        metadata={'env': 'prod'},
    )

    serialized = TemporalRunContext.serialize_run_context(ctx)
    assert serialized['metadata'] == {'env': 'prod'}

    reconstructed = TemporalRunContext.deserialize_run_context(serialized, deps=None)
    assert reconstructed.metadata == {'env': 'prod'}


def test_temporal_run_context_excludes_agent():
    """agent is not serialized but defaults to None after deserialization."""
    from pydantic_ai.durable_exec.temporal._run_context import deserialize_run_context

    agent = Agent('test', name='test_agent')
    ctx = RunContext(
        deps=None,
        agent=agent,
        model=TestModel(),
        usage=RunUsage(),
        run_id='run-123',
    )

    serialized = TemporalRunContext.serialize_run_context(ctx)
    assert 'agent' not in serialized

    # Without agent — e.g. when _agent was never set on a temporal wrapper
    reconstructed = deserialize_run_context(TemporalRunContext, serialized, deps=None, agent=None)
    assert reconstructed.agent is None

    # With agent — as used by TemporalAgent's wrappers
    reconstructed = deserialize_run_context(TemporalRunContext, serialized, deps=None, agent=agent)
    assert reconstructed.agent is agent
    assert agent.name == 'test_agent'


def test_temporal_run_context_serializes_usage():
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(
            requests=2,
            tool_calls=1,
            input_tokens=123,
            output_tokens=456,
            details={'foo': 1},
        ),
        run_id='run-123',
    )

    serialized = TemporalRunContext.serialize_run_context(ctx)
    assert serialized['usage'] == ctx.usage

    reconstructed = TemporalRunContext.deserialize_run_context(serialized, deps=None)
    assert reconstructed.usage == ctx.usage


def test_temporal_run_context_serializes_usage_limits():
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        usage_limits=UsageLimits(request_limit=7, total_tokens_limit=1000),
        run_id='run-123',
    )

    serialized = TemporalRunContext.serialize_run_context(ctx)
    assert serialized['usage_limits'] == ctx.usage_limits

    reconstructed = TemporalRunContext.deserialize_run_context(serialized, deps=None)
    assert reconstructed.usage_limits == ctx.usage_limits


def test_temporal_run_context_serialization_is_exhaustive():
    """Every `RunContext` field must be consciously categorized for Temporal serialization.

    Guards against silent drift: when a `RunContext` field is added, this test fails until
    the author either includes it in `TemporalRunContext.serialize_run_context` or lists it
    in `intentionally_unserialized` below with a reason. Without that decision a new field
    silently becomes unavailable inside a Temporal activity (the `__getattribute__` guard
    raises `UserError` on access), which is how the deferred-capability fields were missed.
    """
    # Fields deliberately NOT carried across the activity boundary, each with its reason.
    intentionally_unserialized = {
        'deps',  # passed separately to deserialize_run_context
        'agent',  # reattached after deserialize by deserialize_run_context
        'model',  # live Model instance, not serializable
        'tracer',  # live tracer, not serializable
        'tool_manager',  # live ToolManager, not serializable (documented on the field)
        'capabilities',  # live capability objects (toolsets/hooks/callables), not serializable
        'root_capability',  # live capability chain, not serializable; reattached from the bound agent by deserialize_run_context
        'pending_messages',  # live run queue, meaningless outside the running agent
        'messages',  # not currently exposed inside activities
        'prompt',  # not currently exposed inside activities
        'validation_context',  # arbitrary user object, possibly unserializable
        'trace_include_content',  # tracing config, not run state
        'instrumentation_version',  # tracing config, not run state
        'conversation_id',  # not currently exposed inside activities
        'model_settings',  # not currently exposed inside activities
        '_mcp_tool_defs_cache',  # run-local cache read/written in workflow code; never needed inside an activity
    }
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    serialized = set(TemporalRunContext.serialize_run_context(ctx))
    all_fields = set(RunContext.__dataclass_fields__)

    overlap = serialized & intentionally_unserialized
    assert not overlap, f'Fields both serialized and excluded: {overlap}'

    uncategorized = all_fields - (serialized | intentionally_unserialized)
    assert not uncategorized, (
        f'Uncategorized `RunContext` fields: {uncategorized}. Add each to '
        '`TemporalRunContext.serialize_run_context` or to `intentionally_unserialized` (with a reason).'
    )


def _tool_return_metadata_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('analyze_data', {})])
    else:
        return ModelResponse(parts=[TextPart('done')])


_tool_return_metadata_agent = Agent(
    FunctionModel(_tool_return_metadata_model),
    name='tool_return_metadata_agent',
)


@_tool_return_metadata_agent.tool_plain
def analyze_data() -> ToolReturn:
    return ToolReturn(
        return_value='analysis result',
        content='extra content for model',
        metadata={'key': 'value', 'count': 42},
    )


_tool_return_metadata_temporal_agent = TemporalAgent(_tool_return_metadata_agent, activity_config=BASE_ACTIVITY_CONFIG)


@workflow.defn
class ToolReturnMetadataWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[ModelMessage]:
        result = await _tool_return_metadata_temporal_agent.run(prompt)
        return result.all_messages()


async def test_tool_return_metadata_survives_temporal(allow_model_requests: None, client: Client):
    """ToolReturn metadata and content survive Temporal serialization.

    Regression test for https://github.com/pydantic/pydantic-ai/issues/4676
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ToolReturnMetadataWorkflow],
        plugins=[AgentPlugin(_tool_return_metadata_temporal_agent)],
    ):
        messages = await client.execute_workflow(
            ToolReturnMetadataWorkflow.run,
            args=['analyze'],
            id=ToolReturnMetadataWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='analyze', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='analyze_data', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='function:_tool_return_metadata_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='analyze_data',
                        content='analysis result',
                        tool_call_id=IsStr(),
                        metadata={'key': 'value', 'count': 42},
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(content='extra content for model', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=57, output_tokens=3),
                model_name='function:_tool_return_metadata_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


mcptoolset_agent = Agent(
    model,
    name='mcptoolset_agent',
    toolsets=[MCPToolset('https://mcp.deepwiki.com/mcp', id='deepwiki')],
)

mcptoolset_temporal_agent = TemporalAgent(
    mcptoolset_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class MCPToolsetAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await mcptoolset_temporal_agent.run(prompt)
        return result.output


async def test_mcptoolset_in_temporal_workflow(allow_model_requests: None, client: Client):
    """`MCPToolset` works in a Temporal workflow — parallel to `test_fastmcp_toolset`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MCPToolsetAgentWorkflow],
        plugins=[AgentPlugin(mcptoolset_temporal_agent)],
    ):
        output = await client.execute_workflow(
            MCPToolsetAgentWorkflow.run,
            args=['Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short'],
            id=MCPToolsetAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert 'pydantic' in output.lower() or 'agent' in output.lower()


# ============================================================================
# ctx.agent in Temporal activities
# ============================================================================


def _ctx_agent_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('get_agent_name', {})])
    else:
        return ModelResponse(parts=[TextPart('done')])


_ctx_agent_test_agent = Agent(
    FunctionModel(_ctx_agent_model),
    name='ctx_agent_test',
)


@_ctx_agent_test_agent.tool
def get_agent_name(ctx: RunContext) -> str:
    return (ctx.agent.name or 'unnamed') if ctx.agent else 'unknown'


_ctx_agent_temporal_agent = TemporalAgent(_ctx_agent_test_agent, activity_config=BASE_ACTIVITY_CONFIG)


@workflow.defn
class CtxAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[ModelMessage]:
        result = await _ctx_agent_temporal_agent.run(prompt)
        return result.all_messages()


async def test_ctx_agent_in_temporal_activity(allow_model_requests: None, client: Client):
    """ctx.agent is available inside Temporal activities, giving access to agent properties like name."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CtxAgentWorkflow],
        plugins=[AgentPlugin(_ctx_agent_temporal_agent)],
    ):
        messages = await client.execute_workflow(
            CtxAgentWorkflow.run,
            args=['test'],
            id=CtxAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='test', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_agent_name', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='function:_ctx_agent_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_agent_name',
                        content='ctx_agent_test',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=52, output_tokens=3),
                model_name='function:_ctx_agent_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


# ============================================================================
# Beta Graph API Tests - Tests for running pydantic-graph beta API in Temporal
# ============================================================================


@dataclass
class GraphState:
    """State for the graph execution test."""

    values: list[int] = field(default_factory=list[int])


# Create a graph with parallel execution using the beta API
graph_builder = GraphBuilder(
    name='parallel_test_graph',
    state_type=GraphState,
    input_type=int,
    output_type=list[int],
)


@graph_builder.step
async def source(ctx: StepContext[GraphState, None, int]) -> int:
    """Source step that passes through the input value."""
    return ctx.inputs


@graph_builder.step
async def multiply_by_two(ctx: StepContext[GraphState, None, int]) -> int:
    """Multiply input by 2."""
    return ctx.inputs * 2


@graph_builder.step
async def multiply_by_three(ctx: StepContext[GraphState, None, int]) -> int:
    """Multiply input by 3."""
    return ctx.inputs * 3


@graph_builder.step
async def multiply_by_four(ctx: StepContext[GraphState, None, int]) -> int:
    """Multiply input by 4."""
    return ctx.inputs * 4


# Create a join to collect results
result_collector = graph_builder.join(reduce_list_append, initial_factory=list[int])

# Build the graph with parallel edges (broadcast pattern)
graph_builder.add(
    graph_builder.edge_from(graph_builder.start_node).to(source),
    # Broadcast: send value to all three parallel steps
    graph_builder.edge_from(source).to(multiply_by_two, multiply_by_three, multiply_by_four),
    # Collect all results
    graph_builder.edge_from(multiply_by_two, multiply_by_three, multiply_by_four).to(result_collector),
    graph_builder.edge_from(result_collector).to(graph_builder.end_node),
)

parallel_test_graph = graph_builder.build()


@workflow.defn
class ParallelGraphWorkflow:
    """Workflow that executes a graph with parallel task execution."""

    @workflow.run
    async def run(self, input_value: int) -> list[int]:
        """Run the parallel graph workflow.

        Args:
            input_value: The input number to process

        Returns:
            List of results from parallel execution
        """
        result = await parallel_test_graph.run(
            state=GraphState(),
            inputs=input_value,
        )
        return result


async def test_beta_graph_parallel_execution_in_workflow(client: Client):
    """Test that beta graph API with parallel execution works in Temporal workflows.

    This test verifies the fix for the bug where parallel task execution in graphs
    wasn't working properly with Temporal workflows due to GraphTask/GraphTaskRequest
    serialization issues.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ParallelGraphWorkflow],
    ):
        output = await client.execute_workflow(
            ParallelGraphWorkflow.run,
            args=[10],
            id=ParallelGraphWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        # Results can be in any order due to parallel execution
        # 10 * 2 = 20, 10 * 3 = 30, 10 * 4 = 40
        assert sorted(output) == [20, 30, 40]


@workflow.defn
class WorkflowWithAgents(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [simple_temporal_agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_temporal_agent.run(prompt)
        return result.output


@workflow.defn
class WorkflowWithAgentsWithoutPydanticAIWorkflow:
    __pydantic_ai_agents__ = [simple_temporal_agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_temporal_agent.run(prompt)
        return result.output


async def test_passing_agents_through_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WorkflowWithAgents],
    ):
        output = await client.execute_workflow(
            WorkflowWithAgents.run,
            args=['What is the capital of Mexico?'],
            id=WorkflowWithAgents.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('The capital of Mexico is Mexico City.')


async def test_passing_agents_through_workflow_without_pydantic_ai_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WorkflowWithAgentsWithoutPydanticAIWorkflow],
    ):
        output = await client.execute_workflow(
            WorkflowWithAgentsWithoutPydanticAIWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=WorkflowWithAgentsWithoutPydanticAIWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('The capital of Mexico is Mexico City.')


# Multi-Model Support Tests

# Module-level test models for multi-model selection test
test_model_selection_1 = TestModel(custom_output_text='Response from model 1')
test_model_selection_2 = TestModel(custom_output_text='Response from model 2')
test_model_selection_3 = TestModel(custom_output_text='Response from model 3')

# Module-level test models for error test
test_model_error_1 = TestModel()
test_model_error_2 = TestModel()
test_model_error_unregistered = TestModel()

# Module-level temporal agents
agent_selection = Agent(test_model_selection_1, name='multi_model_workflow_test')
multi_model_selection_test_agent = TemporalAgent(
    agent_selection,
    name='multi_model_workflow_test',
    models={
        'model_2': test_model_selection_2,
        'model_3': test_model_selection_3,
    },
    activity_config=BASE_ACTIVITY_CONFIG,
)

agent_error = Agent(test_model_error_1, name='error_test')
multi_model_error_test_agent = TemporalAgent(
    agent_error,
    name='error_test',
    models={'other': test_model_error_2},
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class MultiModelWorkflow:
    @workflow.run
    async def run(self, prompt: str, model_id: str | None = None) -> str:
        result = await multi_model_selection_test_agent.run(prompt, model=model_id)
        return result.output


class _BuiltinToolModel(TestModel):
    SUPPORTED_TOOLS: frozenset[type[AbstractNativeTool]] = frozenset()

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        return cls.SUPPORTED_TOOLS

    def _request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        # Override to skip TestModel._request's builtin tools rejection
        return ModelResponse(parts=[TextPart(self.custom_output_text or '')], model_name=self.model_name)


class _WebSearchOnlyModel(_BuiltinToolModel):
    SUPPORTED_TOOLS = frozenset({WebSearchTool})


class _CodeExecutionOnlyModel(_BuiltinToolModel):
    SUPPORTED_TOOLS = frozenset({CodeExecutionTool})


def _select_builtin_tool(ctx: RunContext[Any]) -> AbstractNativeTool:
    if WebSearchTool in ctx.model.profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS):
        return WebSearchTool()
    return CodeExecutionTool()


web_search_builtin_model = _WebSearchOnlyModel(custom_output_text='search model', model_name='web-search')
code_execution_builtin_model = _CodeExecutionOnlyModel(custom_output_text='code model', model_name='code-exec')

builtin_tool_agent = Agent(
    web_search_builtin_model,
    name='builtin_tool_dynamic_agent',
    capabilities=[NativeTool(_select_builtin_tool)],
)

builtin_tool_temporal_agent = TemporalAgent(
    builtin_tool_agent,
    name='builtin_tool_dynamic_agent',
    models={'code': code_execution_builtin_model},
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class BuiltinToolWorkflow:
    @workflow.run
    async def run(self, prompt: str, model_id: str | None = None) -> str:
        result = await builtin_tool_temporal_agent.run(prompt, model=model_id)
        return result.output


# Model that does NOT support any builtin tools (used as default)
no_builtin_support_model = _BuiltinToolModel(custom_output_text='no builtin support', model_name='no-builtin-test')

# Model that DOES support WebSearchTool (registered as alternate model)
web_search_builtin_override_model = _WebSearchOnlyModel(
    custom_output_text='web search response',
    model_name='web-search-override',
)

# Agent initialized with model that doesn't support builtins, but has builtin tools configured
builtins_in_workflow_agent = Agent(
    no_builtin_support_model,
    capabilities=[NativeTool(WebSearchTool()), Instrumentation(settings=InstrumentationSettings())],
    name='builtins_in_workflow',
)

# TemporalAgent registers an alternate model that DOES support builtins
builtins_in_workflow_temporal_agent = TemporalAgent(
    builtins_in_workflow_agent,
    name='builtins_in_workflow',
    models={'web_search': web_search_builtin_override_model},
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class BuiltinsInWorkflow(PydanticAIWorkflow):
    @workflow.run
    async def run(self, prompt: str, model_id: str | None = None) -> str:
        result = await builtins_in_workflow_temporal_agent.run(prompt, model=model_id)
        return result.output


@workflow.defn
class MultiModelWorkflowUnregistered:
    @workflow.run
    async def run(self, prompt: str) -> str:
        # Try to use an unregistered model
        result = await multi_model_error_test_agent.run(prompt, model=test_model_error_unregistered)
        return result.output  # pragma: no cover


async def test_temporal_agent_multi_model_reserved_id():
    """Test that reserved model IDs raise helpful errors."""
    test_model1 = TestModel()
    test_model2 = TestModel()

    agent = Agent(test_model1, name='reserved_id_test')
    with pytest.raises(UserError, match="Model ID 'default' is reserved"):
        TemporalAgent(
            agent,
            name='reserved_id_test',
            models={'default': test_model2},
        )


async def test_temporal_agent_multi_model_selection_in_workflow(allow_model_requests: None, client: Client):
    """Test selecting different models in a workflow using the model parameter."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModelWorkflow],
        plugins=[AgentPlugin(multi_model_selection_test_agent)],
    ):
        # Test using default model (model_id=None)
        output = await client.execute_workflow(
            MultiModelWorkflow.run,
            args=['Hello', None],
            id='MultiModelWorkflow_default',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Response from model 1'

        # Test selecting second model by ID
        output = await client.execute_workflow(
            MultiModelWorkflow.run,
            args=['Hello', 'model_2'],
            id='MultiModelWorkflow_model2',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Response from model 2'

        # Test selecting third model by ID
        output = await client.execute_workflow(
            MultiModelWorkflow.run,
            args=['Hello', 'model_3'],
            id='MultiModelWorkflow_model3',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Response from model 3'


async def test_temporal_dynamic_builtin_tools_select_by_model(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[BuiltinToolWorkflow],
        plugins=[AgentPlugin(builtin_tool_temporal_agent)],
    ):
        output = await client.execute_workflow(
            BuiltinToolWorkflow.run,
            args=['Hello', None],
            id='BuiltinToolWorkflow_default',
            task_queue=TASK_QUEUE,
        )
        assert output == 'search model'
        assert isinstance(web_search_builtin_model.last_model_request_parameters, ModelRequestParameters)
        assert web_search_builtin_model.last_model_request_parameters.native_tools
        assert isinstance(web_search_builtin_model.last_model_request_parameters.native_tools[0], WebSearchTool)

        output = await client.execute_workflow(
            BuiltinToolWorkflow.run,
            args=['Hello', 'code'],
            id='BuiltinToolWorkflow_code',
            task_queue=TASK_QUEUE,
        )
        assert output == 'code model'
        assert isinstance(code_execution_builtin_model.last_model_request_parameters, ModelRequestParameters)
        assert code_execution_builtin_model.last_model_request_parameters.native_tools
        assert isinstance(
            code_execution_builtin_model.last_model_request_parameters.native_tools[0],
            CodeExecutionTool,
        )


async def test_builtins_in_workflow_with_runtime_model_override(allow_model_requests: None, client: Client):
    """Test that builtin tools work when agent is initialized with a non-supporting model
    but run with a model that does support builtins."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[BuiltinsInWorkflow],
        plugins=[AgentPlugin(builtins_in_workflow_temporal_agent)],
    ):
        # Run with the model that supports WebSearchTool
        result = await client.execute_workflow(
            BuiltinsInWorkflow.run,
            args=['search for something', 'web_search'],
            id='BuiltinsInWorkflow',
            task_queue=TASK_QUEUE,
        )
        assert result == 'web search response'

    # Verify the web search model received the WebSearchTool in its request parameters
    assert isinstance(web_search_builtin_override_model.last_model_request_parameters, ModelRequestParameters)
    assert web_search_builtin_override_model.last_model_request_parameters.native_tools
    assert isinstance(
        web_search_builtin_override_model.last_model_request_parameters.native_tools[0],
        WebSearchTool,
    )


async def test_temporal_agent_multi_model_unregistered_error(allow_model_requests: None, client: Client):
    """Test that using an unregistered model raises a helpful error."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModelWorkflowUnregistered],
        plugins=[AgentPlugin(multi_model_error_test_agent)],
    ):
        with workflow_raises(
            UserError,
            'Arbitrary model instances cannot be used at runtime inside a Temporal workflow. Register the model via `models` or reference a registered model by id.',
        ):
            await client.execute_workflow(
                MultiModelWorkflowUnregistered.run,
                args=['Hello'],
                id='MultiModelWorkflowUnregistered',
                task_queue=TASK_QUEUE,
            )


async def test_temporal_agent_multi_model_outside_workflow():
    """Test that multi-model agents work outside workflows (using wrapped agent behavior).

    Outside a workflow, a TemporalAgent should behave like a regular Agent.
    This includes supporting model selection by registered ID or instance.
    """
    test_model1 = TestModel(custom_output_text='Model 1 response')
    test_model2 = TestModel(custom_output_text='Model 2 response')
    test_model_unregistered = TestModel(custom_output_text='Unregistered model response')

    agent = Agent(test_model1, name='outside_workflow_test')
    temporal_agent = TemporalAgent(
        agent,
        name='outside_workflow_test',
        models={'secondary': test_model2},
    )

    # Outside workflow, should use default model
    result = await temporal_agent.run('Hello')
    assert result.output == 'Model 1 response'

    # Outside workflow, passing a registered model ID should also work
    result = await temporal_agent.run('Hello', model='secondary')
    assert result.output == 'Model 2 response'

    # Passing a registered model instance should also work
    result = await temporal_agent.run('Hello', model=test_model2)
    assert result.output == 'Model 2 response'

    # Passing an unregistered model instance should also work outside workflow
    result = await temporal_agent.run('Hello', model=test_model_unregistered)
    assert result.output == 'Unregistered model response'


async def test_temporal_agent_without_default_model():
    """Test that a TemporalAgent can be created without a default model if models is provided.

    When no model is provided to run(), the first registered model should be used.
    """
    test_model1 = TestModel(custom_output_text='Model 1 response')
    test_model2 = TestModel(custom_output_text='Model 2 response')

    # Agent without a model
    agent = Agent(name='no_default_model_test')
    temporal_agent = TemporalAgent(
        agent,
        name='no_default_model_test',
        models={
            'primary': test_model1,
            'secondary': test_model2,
        },
    )

    # Without a model, should use the first registered model
    result = await temporal_agent.run('Hello')
    assert result.output == 'Model 1 response'

    # Outside workflow, can use registered models by id
    result = await temporal_agent.run('Hello', model='primary')
    assert result.output == 'Model 1 response'

    result = await temporal_agent.run('Hello', model='secondary')
    assert result.output == 'Model 2 response'


# Workflow for testing passing model instances (can't be workflow args, so map by key)
_model_instance_map = {
    'default_instance': test_model_selection_1,
    'model_2_instance': test_model_selection_2,
}


@workflow.defn
class MultiModelWorkflowInstance:
    @workflow.run
    async def run(self, prompt: str, instance_key: str) -> str:
        model_instance = _model_instance_map[instance_key]
        result = await multi_model_selection_test_agent.run(prompt, model=model_instance)
        return result.output


@pytest.mark.parametrize(
    ('model_id', 'expected_output'),
    [
        pytest.param('default', 'Response from model 1', id='default_explicit'),
    ],
)
async def test_temporal_agent_model_selection_by_id(
    allow_model_requests: None, client: Client, model_id: str, expected_output: str
):
    """Test model selection by passing model ID strings."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModelWorkflow],
        plugins=[AgentPlugin(multi_model_selection_test_agent)],
    ):
        output = await client.execute_workflow(
            MultiModelWorkflow.run,
            args=['Hello', model_id],
            id=f'MultiModelWorkflow_{model_id}',
            task_queue=TASK_QUEUE,
        )
        assert output == expected_output


@pytest.mark.parametrize(
    ('instance_key', 'expected_output'),
    [
        pytest.param('default_instance', 'Response from model 1', id='default_instance'),
        pytest.param('model_2_instance', 'Response from model 2', id='registered_instance'),
    ],
)
async def test_temporal_agent_model_selection_by_instance(
    allow_model_requests: None, client: Client, instance_key: str, expected_output: str
):
    """Test model selection by passing model instances."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModelWorkflowInstance],
        plugins=[AgentPlugin(multi_model_selection_test_agent)],
    ):
        output = await client.execute_workflow(
            MultiModelWorkflowInstance.run,
            args=['Hello', instance_key],
            id=f'MultiModelWorkflowInstance_{instance_key}',
            task_queue=TASK_QUEUE,
        )
        assert output == expected_output


def test_temporal_model_profile_for_raw_strings():
    """Test TemporalModel infers model_name, system, and profile from raw strings without constructing providers."""

    default_model = TestModel(custom_output_text='default')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__profile_inference',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # Without using_model, properties come from default
    assert temporal_model.profile == default_model.profile
    assert temporal_model.model_name == default_model.model_name
    assert temporal_model.system == default_model.system

    # With raw string, all properties are inferred correctly
    with temporal_model.using_model('openai:gpt-5'):
        assert temporal_model.model_name == 'gpt-5'
        assert temporal_model.system == 'openai'
        assert temporal_model.profile == infer_model_profile('openai:gpt-5')

    # Anthropic profile inference includes WebSearchTool support
    with temporal_model.using_model('anthropic:claude-sonnet-4-5'):
        assert temporal_model.model_name == 'claude-sonnet-4-5'
        assert temporal_model.system == 'anthropic'
        assert temporal_model.profile == infer_model_profile('anthropic:claude-sonnet-4-5')

    # Registered models work correctly for all properties
    alt_model = TestModel(custom_output_text='alt', model_name='alt-model')
    temporal_model_with_registry = TemporalModel(
        default_model,
        activity_name_prefix='test__profile_registry',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
        models={'alt': alt_model},
    )
    with temporal_model_with_registry.using_model('alt'):
        assert temporal_model_with_registry.model_name == 'alt-model'
        assert temporal_model_with_registry.system == alt_model.system
        assert temporal_model_with_registry.profile == alt_model.profile


async def test_temporal_model_request_outside_workflow():
    """Test that TemporalModel.request() falls back to wrapped model outside a workflow.

    When TemporalModel.request() is called directly (not through TemporalAgent.run())
    and not inside a Temporal workflow, it should delegate to the wrapped model's request method.
    """
    test_model = TestModel(custom_output_text='Direct model response')

    temporal_model = TemporalModel(
        test_model,
        activity_name_prefix='test__direct_request',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # Call request() directly - outside a workflow, this should fall back to super().request()
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
    response = await temporal_model.request(
        messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            native_tools=[],
            output_mode='text',
            allow_text_output=True,
            output_tools=[],
            output_object=None,
        ),
    )

    # Verify response comes from the wrapped TestModel
    assert any(isinstance(part, TextPart) and part.content == 'Direct model response' for part in response.parts)


async def test_temporal_model_cancel_suspended_response_outside_workflow():
    """`TemporalModel.cancel_suspended_response()` falls back to the wrapped model outside a workflow.

    Inside a workflow it runs the provider teardown in the `model_cancel_suspended_response` activity
    (registered in `temporal_activities`) so the raw HTTP call never runs in the workflow sandbox;
    outside a workflow it delegates straight to the wrapped model.
    """
    cancelled: list[ModelResponse] = []

    class RecordingModel(TestModel):
        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            cancelled.append(response)

    temporal_model = TemporalModel(
        RecordingModel(),
        activity_name_prefix='test__direct_cancel',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # The cancel activity is registered alongside the request activities.
    assert [
        ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        for activity in temporal_model.temporal_activities
    ] == snapshot(
        [
            'test__direct_cancel__model_request',
            'test__direct_cancel__model_request_stream',
            'test__direct_cancel__model_cancel_suspended_response',
        ]
    )

    response = ModelResponse(parts=[TextPart('paused')], state='suspended')
    await temporal_model.cancel_suspended_response(response)
    assert cancelled == [response]


# Module-level so the `@workflow.defn` below can bind to it (mirrors `simple_temporal_agent`). The
# activity records into this list; since activities always run outside the workflow sandbox in the
# worker process, the workflow can dispatch the teardown while the assertion still observes it here.
model_cancel_calls: list[ModelResponse] = []


class CancelRecordingModel(TestModel):
    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        model_cancel_calls.append(response)


cancel_temporal_model = TemporalModel(
    CancelRecordingModel(),
    activity_name_prefix='cancel_suspended',
    activity_config=BASE_ACTIVITY_CONFIG,
    deps_type=type(None),
)


@workflow.defn
class CancelSuspendedResponseWorkflow:
    @workflow.run
    async def run(self, response: ModelResponse) -> None:
        # In-workflow, `cancel_suspended_response` must dispatch the provider teardown to the
        # `model_cancel_suspended_response` activity rather than make the raw HTTP call in the sandbox.
        await cancel_temporal_model.cancel_suspended_response(response)


async def test_temporal_model_cancel_suspended_response_in_workflow(client: Client):
    """Inside a workflow, `cancel_suspended_response` tears the server-side job down via an activity.

    Counterpart to `test_temporal_model_cancel_suspended_response_outside_workflow`: it drives the
    in-workflow override -> `workflow.execute_activity` -> activity-body path end to end, proving the
    wrapped model's cancel actually runs and that the `ModelResponse` argument survives serialization
    across both the workflow and activity boundaries.
    """
    model_cancel_calls.clear()
    response = ModelResponse(parts=[TextPart('paused')], state='suspended')
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CancelSuspendedResponseWorkflow],
        activities=cancel_temporal_model.temporal_activities,
    ):
        await client.execute_workflow(
            CancelSuspendedResponseWorkflow.run,
            args=[response],
            id=CancelSuspendedResponseWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    # The teardown ran in the activity worker against the wrapped model, with the response faithfully
    # round-tripped through both serialization boundaries.
    assert model_cancel_calls == [response]


async def test_temporal_model_request_stream_outside_workflow():
    """Test that TemporalModel.request_stream() falls back to wrapped model outside a workflow.

    When TemporalModel.request_stream() is called directly (not through TemporalAgent.run())
    and not inside a Temporal workflow, it should delegate to the wrapped model's request_stream method.
    """
    test_model = TestModel(custom_output_text='Direct stream response')

    temporal_model = TemporalModel(
        test_model,
        activity_name_prefix='test__direct_stream',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # Call request_stream() directly - outside a workflow, this should fall back to super().request_stream()
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
    async with temporal_model.request_stream(
        messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            native_tools=[],
            output_mode='text',
            allow_text_output=True,
            output_tools=[],
            output_object=None,
        ),
    ) as stream:
        # Consume the stream
        async for _ in stream:
            pass

        # Get the final response
        response = stream.get()

    # Verify response comes from the wrapped TestModel
    assert any(isinstance(part, TextPart) and part.content == 'Direct stream response' for part in response.parts)


class CustomPydanticPayloadConverter(PydanticPayloadConverter):
    """A custom payload converter that inherits from PydanticPayloadConverter."""

    pass


class CustomPayloadConverter(DefaultPayloadConverter):
    """A custom payload converter that does not inherit from PydanticPayloadConverter."""

    pass


class MockPayloadCodec(PayloadCodec):
    """A mock payload codec for testing (simulates encryption codec)."""

    async def encode(
        self, payloads: Sequence[temporalio.api.common.v1.Payload]
    ) -> list[temporalio.api.common.v1.Payload]:  # pragma: no cover
        return list(payloads)

    async def decode(
        self, payloads: Sequence[temporalio.api.common.v1.Payload]
    ) -> list[temporalio.api.common.v1.Payload]:  # pragma: no cover
        return list(payloads)


def test_pydantic_ai_plugin_no_converter_returns_pydantic_data_converter() -> None:
    """When no converter is provided, PydanticAIPlugin uses the standard pydantic_data_converter."""
    plugin = PydanticAIPlugin()
    # Create a minimal config without data_converter
    config: dict[str, Any] = {}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'] is pydantic_data_converter


def test_pydantic_ai_plugin_with_pydantic_payload_converter_unchanged() -> None:
    """When converter already uses PydanticPayloadConverter, return it unchanged."""
    plugin = PydanticAIPlugin()
    converter = DataConverter(payload_converter_class=PydanticPayloadConverter)
    config: dict[str, Any] = {'data_converter': converter}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'] is converter


def test_pydantic_ai_plugin_with_custom_pydantic_subclass_unchanged() -> None:
    """When converter uses a subclass of PydanticPayloadConverter, return it unchanged (no warning)."""
    plugin = PydanticAIPlugin()
    converter = DataConverter(payload_converter_class=CustomPydanticPayloadConverter)
    config: dict[str, Any] = {'data_converter': converter}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'] is converter
    assert result['data_converter'].payload_converter_class is CustomPydanticPayloadConverter


def test_pydantic_ai_plugin_with_default_payload_converter_replaced() -> None:
    """When converter uses DefaultPayloadConverter, replace payload_converter_class without warning."""
    plugin = PydanticAIPlugin()
    converter = DataConverter(payload_converter_class=DefaultPayloadConverter)
    config: dict[str, Any] = {'data_converter': converter}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'] is not converter
    assert result['data_converter'].payload_converter_class is PydanticPayloadConverter


def test_pydantic_ai_plugin_preserves_custom_payload_codec() -> None:
    """When converter has a custom payload_codec, preserve it while replacing payload_converter_class."""
    plugin = PydanticAIPlugin()
    codec = MockPayloadCodec()
    converter = DataConverter(
        payload_converter_class=DefaultPayloadConverter,
        payload_codec=codec,
    )
    config: dict[str, Any] = {'data_converter': converter}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'] is not converter
    assert result['data_converter'].payload_converter_class is PydanticPayloadConverter
    assert result['data_converter'].payload_codec is codec


def test_pydantic_ai_plugin_with_non_pydantic_converter_warns() -> None:
    """When converter uses a non-Pydantic payload converter, warn and replace."""
    plugin = PydanticAIPlugin()
    converter = DataConverter(payload_converter_class=CustomPayloadConverter)
    config: dict[str, Any] = {'data_converter': converter}
    with pytest.warns(
        UserWarning,
        match='A non-Pydantic Temporal payload converter was used which has been replaced with PydanticPayloadConverter',
    ):
        result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'].payload_converter_class is PydanticPayloadConverter


def test_pydantic_ai_plugin_with_non_pydantic_converter_preserves_codec() -> None:
    """When converter uses a non-Pydantic payload converter with custom codec, warn but preserve codec."""
    plugin = PydanticAIPlugin()
    codec = MockPayloadCodec()
    converter = DataConverter(
        payload_converter_class=CustomPayloadConverter,
        payload_codec=codec,
    )
    config: dict[str, Any] = {'data_converter': converter}
    with pytest.warns(UserWarning):
        result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'].payload_converter_class is PydanticPayloadConverter
    assert result['data_converter'].payload_codec is codec


def test_temporal_model_profile_with_no_provider_prefix() -> None:
    """Test TemporalModel uses DEFAULT_PROFILE when model string has no inferable provider."""

    default_model = TestModel(custom_output_text='default')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__no_provider_prefix',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # A model string without a provider prefix that can't be inferred returns DEFAULT_PROFILE
    with temporal_model.using_model('some-random-model'):
        assert temporal_model.profile is DEFAULT_PROFILE


def test_temporal_model_profile_with_unknown_provider() -> None:
    """Test TemporalModel uses DEFAULT_PROFILE when provider is unknown."""

    default_model = TestModel(custom_output_text='default')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__unknown_provider',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # An unknown provider should return DEFAULT_PROFILE
    with temporal_model.using_model('unknown-provider:some-model'):
        assert temporal_model.profile is DEFAULT_PROFILE


@pytest.mark.parametrize(
    'model_id',
    [
        'openai:gpt-5',
        'gateway/openai:gpt-5',
    ],
)
def test_temporal_model_prepare_request_with_unregistered_model_string(model_id: str) -> None:
    """Test prepare_request uses inferred profile for unregistered model strings.

    Verifies that the OpenAI json_schema_transformer is applied to function tool
    schemas (adding additionalProperties: false) when using an OpenAI model string,
    both directly and via gateway/.
    """
    default_model = TestModel(custom_output_text='default')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__prepare_request_unregistered',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    tool_def = ToolDefinition(
        name='my_tool',
        description='A test tool',
        parameters_json_schema={
            'type': 'object',
            'properties': {'x': {'type': 'integer'}},
            'required': ['x'],
        },
    )

    model_request_params = ModelRequestParameters(
        function_tools=[tool_def],
        native_tools=[],
        output_mode='text',
        allow_text_output=True,
        output_tools=[],
        output_object=None,
    )

    # With an unregistered model string, prepare_request should use the inferred
    # profile's json_schema_transformer (OpenAI adds additionalProperties: false)
    with temporal_model.using_model(model_id):
        _, params = temporal_model.prepare_request(None, model_request_params)
        assert params.output_mode == 'text'
        assert len(params.function_tools) == 1
        assert params.function_tools[0].parameters_json_schema['additionalProperties'] is False


def test_temporal_model_prepare_messages_with_unregistered_model_string() -> None:
    """`prepare_messages` falls back to `Model.prepare_messages` for unregistered model strings.

    Mirrors `prepare_request`: when `using_model('openai:...')` swaps in a model the
    registry doesn't know, the temporal wrapper has no concrete `Model` instance to
    delegate to, so it must invoke the grandparent `Model.prepare_messages` against
    its own profile-derived behavior.
    """
    default_model = TestModel(custom_output_text='default')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__prepare_messages_unregistered',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='hi')])]
    with temporal_model.using_model('openai:gpt-5'):
        prepared = temporal_model.prepare_messages(messages)
    assert prepared == messages


def test_temporal_model_customize_request_parameters_with_registered_model() -> None:
    """Test customize_request_parameters delegates to the currently active registered model."""

    class _CustomizingTestModel(TestModel):
        def customize_request_parameters(
            self, model_request_parameters: ModelRequestParameters
        ) -> ModelRequestParameters:
            return ModelRequestParameters(output_mode='tool', allow_text_output=False)

    default_model = TestModel(custom_output_text='default')
    alternate_model = _CustomizingTestModel(custom_output_text='alternate')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__customize_registered',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
        models={'alternate': alternate_model},
    )

    with temporal_model.using_model('alternate'):
        customized = temporal_model.customize_request_parameters(ModelRequestParameters())

    assert customized.output_mode == 'tool'
    assert customized.allow_text_output is False


# Tests for BinaryContent and DocumentUrl serialization in Temporal
# This is a regression test for #3702 (BinaryContent) and verifies that FileUrl
# instances (like DocumentUrl) with explicit media_type are properly preserved.


multimodal_content_agent = Agent(TestModel(), name='multimodal_content_agent')


@multimodal_content_agent.tool
def get_multimodal_content(ctx: RunContext) -> list[str | MultiModalContent]:
    """Return a list with text, BinaryContent, and DocumentUrl."""
    return [
        'test',
        BinaryImage(data=b'\x89PNG', media_type='image/png'),
        # URL doesn't hint at media type, so media_type must be specified explicitly
        DocumentUrl(url='https://example.com/doc/12345', media_type='application/pdf'),
    ]


multimodal_content_temporal_agent = TemporalAgent(multimodal_content_agent, activity_config=BASE_ACTIVITY_CONFIG)


@workflow.defn
class MultiModalContentWorkflow:
    @workflow.run
    async def run(self, prompt: list[UserContent]) -> list[ModelMessage]:
        result = await multimodal_content_temporal_agent.run(prompt)
        return result.all_messages()


async def test_multimodal_content_serialization_in_workflow(client: Client):
    """Test that BinaryContent and DocumentUrl survive Temporal serialization.

    This tests both:
    1. Passing BinaryContent and DocumentUrl as input to agent.run (workflow→activity)
    2. Returning BinaryContent and DocumentUrl from a tool (activity→workflow)

    BinaryContent is serialized with base64 encoding. DocumentUrl requires explicit
    media_type since it cannot be inferred from the URL.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModalContentWorkflow],
        plugins=[AgentPlugin(multimodal_content_temporal_agent)],
    ):
        # Pass both BinaryContent and DocumentUrl as input
        prompt: list[str | MultiModalContent] = [
            'Process these files and call the tool',
            BinaryImage(data=b'\x89PNG', media_type='image/png'),
            DocumentUrl(url='https://example.com/doc/12345', media_type='application/pdf'),
        ]
        messages = await client.execute_workflow(
            MultiModalContentWorkflow.run,
            args=[prompt],
            id='test_multimodal_content_serialization',
            task_queue=TASK_QUEUE,
        )
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[
                                'Process these files and call the tool',
                                BinaryImage(data=b'\x89PNG', media_type='image/png', identifier='4effda'),
                                DocumentUrl(
                                    url='https://example.com/doc/12345',
                                    _media_type='application/pdf',
                                    _identifier='eb8998',
                                ),
                            ],
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_multimodal_content',
                            args={},
                            tool_call_id='pyd_ai_tool_call_id__get_multimodal_content',
                        )
                    ],
                    usage=RequestUsage(input_tokens=61, output_tokens=2),
                    model_name='test',
                    timestamp=IsDatetime(),
                    provider_name='test',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_multimodal_content',
                            content=[
                                'test',
                                BinaryImage(data=b'\x89PNG', media_type='image/png', identifier='4effda'),
                                DocumentUrl(
                                    url='https://example.com/doc/12345',
                                    _media_type='application/pdf',
                                    _identifier='eb8998',
                                ),
                            ],
                            tool_call_id='pyd_ai_tool_call_id__get_multimodal_content',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='{"get_multimodal_content":["test",{"data":"iVBORw==","media_type":"image/png","vendor_metadata":null,"kind":"binary","identifier":"4effda"},{"url":"https://example.com/doc/12345","force_download":false,"vendor_metadata":null,"kind":"document-url","media_type":"application/pdf","identifier":"eb8998"}]}'
                        )
                    ],
                    usage=RequestUsage(input_tokens=62, output_tokens=34),
                    model_name='test',
                    timestamp=IsDatetime(),
                    provider_name='test',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

        # Explicitly verify that media_type is preserved through serialization for both
        # BinaryContent and DocumentUrl. This is important because _media_type has compare=False
        # on DocumentUrl, so the snapshot comparison doesn't actually verify it. The media_type
        # cannot be inferred from the URL, so if serialization loses it, accessing media_type
        # would raise an error.
        media_types: list[tuple[str, str]] = []
        for message in messages:
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    for content in part.content:
                        if isinstance(content, (BinaryContent, DocumentUrl)):
                            media_types.append((type(content).__name__, content.media_type))
                elif isinstance(part, ToolReturnPart):
                    for content in part.content_items():
                        if isinstance(content, (BinaryContent, DocumentUrl)):
                            media_types.append((type(content).__name__, content.media_type))
        # Should have 4 items: 2 from user input, 2 from tool return.
        # The image `BinaryContent` round-trips as `BinaryImage`: narrowing is applied during
        # `MultiModalContent` validation, so it now survives the Temporal serialization boundary too.
        assert media_types == [
            ('BinaryImage', 'image/png'),
            ('DocumentUrl', 'application/pdf'),
            ('BinaryImage', 'image/png'),
            ('DocumentUrl', 'application/pdf'),
        ]


nested_multimodal_tool_return_agent = Agent(TestModel(), name='nested_multimodal_tool_return_agent')


@nested_multimodal_tool_return_agent.tool
def get_nested_multimodal_content(ctx: RunContext) -> dict[str, str | MultiModalContent]:
    """Return multimodal content nested inside a mapping."""
    return {
        'caption': 'see attached',
        'attachment': BinaryImage(data=b'\x89PNG', media_type='image/png'),
        'source': DocumentUrl(url='https://example.com/doc/12345', media_type='application/pdf'),
    }


nested_multimodal_tool_return_temporal_agent = TemporalAgent(
    nested_multimodal_tool_return_agent, activity_config=BASE_ACTIVITY_CONFIG
)


@workflow.defn
class NestedMultiModalToolReturnWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[ModelMessage]:
        result = await nested_multimodal_tool_return_temporal_agent.run(prompt)
        return result.all_messages()


async def test_nested_multimodal_tool_return_survives_temporal(client: Client):
    """Nested multimodal values in tool returns survive the Temporal activity boundary."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[NestedMultiModalToolReturnWorkflow],
        plugins=[AgentPlugin(nested_multimodal_tool_return_temporal_agent)],
    ):
        messages = await client.execute_workflow(
            NestedMultiModalToolReturnWorkflow.run,
            args=['inspect attachment'],
            id='test_nested_multimodal_tool_return',
            task_queue=TASK_QUEUE,
        )

    tool_return = next(
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'get_nested_multimodal_content'
    )
    tool_return_content_obj = tool_return.content
    assert isinstance(tool_return_content_obj, dict)
    tool_return_content = cast(dict[str, object], tool_return_content_obj)
    assert tool_return_content['caption'] == 'see attached'

    attachment = tool_return_content['attachment']
    assert isinstance(attachment, BinaryImage)
    assert attachment.media_type == 'image/png'
    assert attachment.data == b'\x89PNG'

    source = tool_return_content['source']
    assert isinstance(source, DocumentUrl)
    assert source.media_type == 'application/pdf'
    assert source.url == 'https://example.com/doc/12345'


async def test_text_content_serialization_in_workflow(client: Client):
    """Test that TextContent is properly serialized in Temporal."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModalContentWorkflow],
        plugins=[AgentPlugin(multimodal_content_temporal_agent)],
    ):
        prompt = [
            'This is a text content test',
            TextContent(content='This should be preserved as TextContent', metadata={'preserved': True}),
        ]
        messages = await client.execute_workflow(
            MultiModalContentWorkflow.run,
            args=[prompt],
            id='test_text_content_serialization',
            task_queue=TASK_QUEUE,
        )
        assert messages[0] == snapshot(
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'This is a text content test',
                            TextContent(
                                content='This should be preserved as TextContent', metadata={'preserved': True}
                            ),
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        )


# ==========================================
# TemporalDurability capability tests
# ==========================================


def _durability_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Simple model function for durability tests that echoes the last user prompt."""
    for msg in reversed(messages):  # pragma: no branch - first message always carries the prompt
        for part in msg.parts:  # pragma: no branch - first part is always the UserPromptPart
            if isinstance(part, UserPromptPart):  # pragma: no branch - same reason
                return ModelResponse(parts=[TextPart(content=f'Echo: {part.content}')])
    return ModelResponse(parts=[TextPart(content='no prompt')])  # pragma: no cover


_durability_fn_model = FunctionModel(_durability_model_fn)

simple_durability = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)
simple_durable_agent = Agent(_durability_fn_model, name='durability_simple_agent', capabilities=[simple_durability])


@workflow.defn
class SimpleDurableAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_durable_agent.run(prompt)
        return result.output


async def test_durability_simple_agent_run_in_workflow(client: Client):
    """TemporalDurability routes model requests through activities."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleDurableAgentWorkflow],
        plugins=[AgentPlugin(simple_durable_agent)],
    ):
        output = await client.execute_workflow(
            SimpleDurableAgentWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=SimpleDurableAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == 'Echo: What is the capital of Mexico?'


# --- Durability with tools ---


def _tool_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Model function that calls `get_country` tool then returns the result."""
    # Check if we already have a tool result
    for msg in reversed(messages):
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(parts=[TextPart(content=f'The country is: {part.content}')])

    # First call: invoke the tool
    if info.function_tools:
        return ModelResponse(parts=[ToolCallPart(tool_name='get_country', args='{}')])

    return ModelResponse(parts=[TextPart(content='no tools')])  # pragma: no cover


durability_country_toolset = FunctionToolset[Deps](tools=[get_country], id='durability_country')

_tool_fn_model = FunctionModel(_tool_model_fn)

complex_durability = TemporalDurability[Deps](deps_type=Deps, activity_config=BASE_ACTIVITY_CONFIG)
complex_durable_agent = Agent(
    _tool_fn_model,
    deps_type=Deps,
    toolsets=[durability_country_toolset],
    capabilities=[complex_durability],
    name='durability_complex_agent',
)


@workflow.defn
class ComplexDurableAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: Deps) -> str:
        result = await complex_durable_agent.run(prompt, deps=deps)
        return result.output


async def test_durability_agent_with_tools_in_workflow(client: Client):
    """TemporalDurability wraps toolsets and routes tool calls through activities."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ComplexDurableAgentWorkflow],
        plugins=[AgentPlugin(complex_durable_agent)],
    ):
        output = await client.execute_workflow(
            ComplexDurableAgentWorkflow.run,
            args=['What country?', Deps(country='France')],
            id=ComplexDurableAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == 'The country is: France'


# --- Durability outside workflow (transparent passthrough) ---


async def test_durability_outside_workflow_is_transparent():
    """TemporalDurability is a no-op outside a workflow — calls pass through to the real model."""
    result = await simple_durable_agent.run('Hello')
    assert result.output == 'Echo: Hello'


# --- Durability wrap_run disables threads ---


_threads_durability = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)
_threads_agent = Agent(_durability_fn_model, name='sync_tool_test', capabilities=[_threads_durability])


@workflow.defn
class ThreadsDurableWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _threads_agent.run(prompt)
        return result.output


async def test_durability_wrap_run_disables_threads(client: Client):
    """wrap_run disables threads when inside a Temporal workflow."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ThreadsDurableWorkflow],
        plugins=[AgentPlugin(_threads_agent)],
    ):
        output = await client.execute_workflow(
            ThreadsDurableWorkflow.run,
            args=['test'],
            id='ThreadsDurableWorkflow',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Echo: test'


# --- Durability validation ---


def test_durability_requires_agent_name():
    """TemporalDurability raises UserError when agent has no name."""
    durability = TemporalDurability()
    with pytest.raises(UserError, match='unique `name`'):
        Agent(_durability_fn_model, capabilities=[durability])


def test_durability_requires_model():
    """TemporalDurability raises UserError when the agent has no model at all."""
    durability = TemporalDurability()
    with pytest.raises(UserError, match='needs to have a `model`'):
        Agent(name='test', capabilities=[durability])


def test_durability_rejects_default_model_key():
    """TemporalDurability raises UserError when 'default' is used in the models dict."""
    with pytest.raises(UserError, match="'default' is reserved"):
        Agent(
            _durability_fn_model,
            name='test',
            capabilities=[TemporalDurability(models={'default': _durability_fn_model})],
        )


def test_durability_rejects_construction_inside_workflow(monkeypatch: pytest.MonkeyPatch):
    """`TemporalDurability.for_agent` rejects construction inside a workflow.

    Activities have to be registered with the worker before the workflow runs, so
    `for_agent` (which discovers and registers activities) must run at module level
    or in worker setup code — not inside `@workflow.run`.
    """
    from temporalio import workflow as _wf

    monkeypatch.setattr(_wf, 'in_workflow', lambda: True)
    with pytest.raises(UserError, match=r'must be constructed outside of a Temporal workflow'):
        Agent(_durability_fn_model, name='test', capabilities=[TemporalDurability()])


def test_durability_image_output_rejected():
    """TemporalDurability rejects image output because of the 2MB payload limit."""
    agent = Agent(_durability_fn_model, name='test', capabilities=[TemporalDurability()])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    with pytest.raises(UserError, match='Image output is not supported'):
        bound._validate_model_request_parameters(  # pyright: ignore[reportPrivateUsage]
            ModelRequestParameters(allow_image_output=True),
        )


# --- Model registry ---


def test_durability_find_model_id_by_identity():
    """_find_model_id matches models by identity."""
    m1 = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(content='hi')]))
    m2 = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(content='hi')]))
    agent = Agent(m1, name='test', capabilities=[TemporalDurability(models={'alt': m2})])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    assert bound._find_model_id(m1) is None  # default → None  # pyright: ignore[reportPrivateUsage]
    assert bound._find_model_id(m2) == 'alt'  # pyright: ignore[reportPrivateUsage]


def test_durability_temporal_activities():
    """temporal_activities returns all registered activities after for_agent."""
    agent = Agent(_durability_fn_model, name='test', capabilities=[TemporalDurability()])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    # 2 base activities (request, request_stream) + 1 for the agent's <agent> FunctionToolset
    assert len(bound.temporal_activities) == 3


def test_durability_temporal_activities_with_toolsets():
    """temporal_activities includes toolset activities for agent's toolsets."""
    agent = Agent(
        _durability_fn_model,
        name='test',
        toolsets=[FunctionToolset(id='test_toolset')],
        capabilities=[TemporalDurability()],
    )
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    # 2 base activities + 1 for <agent> FunctionToolset + 1 for test_toolset
    assert len(bound.temporal_activities) == 4


def test_durability_shared_instance_across_agents():
    """Same TemporalDurability instance can be reused across multiple agents.

    for_agent returns a new bound copy; the original stays pristine.
    """
    durability = TemporalDurability()
    a1 = Agent(_durability_fn_model, name='a1', capabilities=[durability])
    a2 = Agent(_durability_fn_model, name='a2', capabilities=[durability])
    # Original is unbound
    assert durability.name == ''
    assert durability.temporal_activities == []
    # Each agent has its own bound copy
    b1 = TemporalDurability.from_agent(a1)
    b2 = TemporalDurability.from_agent(a2)
    assert b1 is not None and b2 is not None
    assert b1 is not b2
    assert b1.name == 'a1'
    assert b2.name == 'a2'


# --- _find_model_id rejects unregistered models ---


_rt_primary_model = FunctionModel(_durability_model_fn, model_name='primary')
_rt_alt_model = FunctionModel(
    lambda messages, info: ModelResponse(parts=[TextPart(content='alt-response')]),
    model_name='alt',
)
_rt_durability = TemporalDurability(models={'alt': _rt_alt_model}, activity_config=BASE_ACTIVITY_CONFIG)
_rt_agent = Agent(_rt_primary_model, name='runtime_model_test', capabilities=[_rt_durability])


@workflow.defn
class RuntimeModelWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _rt_agent.run(prompt, model=_rt_alt_model)
        return result.output


async def test_durability_runtime_registered_model_is_used(client: Client):
    """agent.run(model=registered_model) routes through the registered model's activity."""
    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[RuntimeModelWorkflow], plugins=[AgentPlugin(_rt_agent)]
    ):
        output = await client.execute_workflow(
            RuntimeModelWorkflow.run,
            args=['ignored'],
            id='RuntimeModelWorkflow',
            task_queue=TASK_QUEUE,
        )
    assert output == 'alt-response'


async def test_durability_resolve_model_id_uses_models_registry():
    """resolve_model_id maps a registered model-id string to its registered Model instance."""
    primary = FunctionModel(_durability_model_fn, model_name='primary')
    alt = FunctionModel(_durability_model_fn, model_name='alt')

    durability = TemporalDurability(models={'alt': alt}, activity_config=BASE_ACTIVITY_CONFIG)
    agent = Agent(primary, name='resolve_registry_test', capabilities=[durability])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    resolution_ctx = ModelResolutionContext(agent=agent, deps=None)

    # String matches a registered model → returns that exact instance.
    assert await bound.resolve_model_id('alt', resolution_ctx) is alt

    # String not in registry → defer (None) so the default `infer_model` flow — or a
    # user's `ResolveModelId` capability — handles it, and so an exception raised by a
    # user resolver is never masked by this capability's backstop.
    assert await bound.resolve_model_id('test', resolution_ctx) is None


async def test_durability_default_string_registered_in_models_becomes_default():
    """A `models=` key equal to the agent's raw default model string supplies the default instance.

    The user explicitly mapped that string to an instance, so binding uses it as `'default'`
    (rather than building an orphaned one via `infer_model`), and run-time resolution of the
    default string returns the same instance — keeping the identity match that gives the
    default the `model_id=None` fast path across the activity boundary.
    """
    custom = FunctionModel(_durability_model_fn, model_name='custom-default')
    durability = TemporalDurability(models={'test': custom}, activity_config=BASE_ACTIVITY_CONFIG)
    agent = Agent('test', name='default_collision_test', capabilities=[durability])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None

    assert await bound.resolve_model_id('test', ModelResolutionContext(agent=agent, deps=None)) is custom
    assert bound._find_model_id(custom) is None  # identity-matches 'default'  # pyright: ignore[reportPrivateUsage]


# --- Deps-aware model resolution via the `ResolveModelId` capability ---


def _tenant_resolver(ctx: ModelResolutionContext[str], model_id: str) -> FunctionModel | None:
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


_tenant_agent = Agent(
    _rt_primary_model,
    name='tenant_resolver_test',
    deps_type=str,
    capabilities=[ResolveModelId(_tenant_resolver), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class TenantModelWorkflow:
    @workflow.run
    async def run(self, tenant: str) -> str:
        result = await _tenant_agent.run('hi', model='tenant-model', deps=tenant)
        return result.output


async def test_durability_resolve_model_id_capability_is_deps_aware(client: Client):
    """A deps-aware `ResolveModelId` resolver rebuilds the model with the run's deps inside the activity.

    The response content is produced by the model *inside* the model-request activity, so it
    proves the activity re-ran the capability chain with the deserialized deps — not just that
    the workflow-side resolution saw them.

    The resolver is deliberately *synchronous*: workflow-side resolution runs before
    `TemporalDurability.wrap_run`'s `disable_threads()` guard is active, so this also pins
    that `ResolveModelId` invokes sync resolvers inline rather than via a thread executor
    (which is unavailable inside the deterministic workflow sandbox and would hang).
    """
    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[TenantModelWorkflow], plugins=[AgentPlugin(_tenant_agent)]
    ):
        for tenant in ('acme', 'globex'):
            output = await client.execute_workflow(
                TenantModelWorkflow.run,
                args=[tenant],
                id=f'TenantModelWorkflow-{tenant}',
                task_queue=TASK_QUEUE,
            )
            assert output == f'tenant:{tenant}'


_alias_default_agent = Agent(
    'tenant-model',
    name='alias_default_test',
    deps_type=str,
    capabilities=[ResolveModelId(_tenant_resolver), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class AliasDefaultWorkflow:
    @workflow.run
    async def run(self, tenant: str) -> str:
        result = await _alias_default_agent.run('hi', deps=tenant)
        return result.output


async def test_durability_alias_default_model(client: Client):
    """An agent whose *default* model is an alias only a `ResolveModelId` capability can resolve.

    `infer_model` can't build `'tenant-model'`, so binding registers no concrete default;
    every request carries the raw alias string across the activity boundary and the
    worker-side chain re-resolves it with the run's deps.
    """
    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[AliasDefaultWorkflow], plugins=[AgentPlugin(_alias_default_agent)]
    ):
        output = await client.execute_workflow(
            AliasDefaultWorkflow.run,
            args=['acme'],
            id='AliasDefaultWorkflow',
            task_queue=TASK_QUEUE,
        )
    assert output == 'tenant:acme'


# --- Outer capability swaps `request_context.model` inside a workflow ---


def _swapped_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='swapped-response')])


_swap_target_registered = FunctionModel(_swapped_model_fn)


class _SwapModelCapability(AbstractCapability[Any]):
    """Outer capability that swaps the request's model to a fresh, unregistered instance."""

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        request_context.model = FunctionModel(_swapped_model_fn)
        return request_context


_swap_model_durability = TemporalDurability(
    # A *different* instance is registered under the same `model_id`, so the swapped
    # instance can only reach the activity via the `model_id` string round-trip.
    models={_swap_target_registered.model_id: _swap_target_registered},
    activity_config=BASE_ACTIVITY_CONFIG,
)
_swap_model_agent = Agent(
    _durability_fn_model,
    name='durability_swap_model_agent',
    capabilities=[_SwapModelCapability(), _swap_model_durability],
)


@workflow.defn
class SwapModelWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _swap_model_agent.run(prompt)
        return result.output


async def test_durability_outer_capability_model_swap_round_trips(client: Client):
    """A model swapped in by an outer capability's `before_model_request` survives the activity boundary.

    Managed-style capabilities sit outside the durability capability and may replace
    `request_context.model` with a freshly-built instance the durability registry has
    never seen. `_find_model_id` must fall back to the instance's `model_id` string
    (rather than erroring), so the activity can rebuild the equivalent model.
    """
    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[SwapModelWorkflow], plugins=[AgentPlugin(_swap_model_agent)]
    ):
        output = await client.execute_workflow(
            SwapModelWorkflow.run,
            args=['ignored'],
            id='SwapModelWorkflow',
            task_queue=TASK_QUEUE,
        )
    assert output == 'swapped-response'


def test_durability_find_model_id_falls_back_to_model_id_string():
    """_find_model_id round-trips runtime-built models via their `model_id` string.

    Pre-registered models (default and `models=` extras) match by identity. Runtime
    models — built from a string via the `resolve_model_id` chain — aren't in the
    registry, so we send their `model_id` string across the activity boundary; the
    worker rebuilds them via the same chain (or default `infer_model`).
    """
    m1 = FunctionModel(_durability_model_fn, model_name='registered')
    m_runtime = FunctionModel(_durability_model_fn, model_name='runtime-built')

    agent = Agent(m1, name='model_round_trip_test', capabilities=[TemporalDurability()])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None

    # Registered default model matches by identity → None
    assert bound._find_model_id(m1) is None  # pyright: ignore[reportPrivateUsage]

    # Unregistered runtime model: round-trip via its model_id string.
    assert bound._find_model_id(m_runtime) == m_runtime.model_id  # pyright: ignore[reportPrivateUsage]


# --- _validate_per_run_capabilities rejects runtime-added classes ---


def test_durability_rejects_runtime_added_capabilities():
    """Per-run capabilities not seen at construction time are rejected.

    Capability instances added via `agent.run(capabilities=[...])` bypass the
    activity-registration step in `for_agent`. The capability detects this by
    snapshotting the bound chain's classes and comparing against `ctx.root_capability`.
    """
    from pydantic_ai._run_context import RunContext
    from pydantic_ai.capabilities.abstract import AbstractCapability
    from pydantic_ai.capabilities.combined import CombinedCapability
    from pydantic_ai.result import RunUsage

    @dataclass
    class _UnregisteredCap(AbstractCapability[None]):
        pass

    durability = TemporalDurability()
    agent = Agent(_durability_fn_model, name='runtime_cap_test', capabilities=[durability])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None

    def make_ctx(root: AbstractCapability[Any]) -> RunContext[None]:
        return RunContext[None](deps=None, model=_durability_fn_model, usage=RunUsage(), root_capability=root)

    bound_chain = agent.root_capability
    runtime_chain = CombinedCapability([*bound_chain.capabilities, _UnregisteredCap()])
    with pytest.raises(UserError, match='Capabilities added per-run inside a Temporal workflow'):
        bound._validate_per_run_capabilities(make_ctx(runtime_chain))  # pyright: ignore[reportPrivateUsage]

    # Sanity: the bound chain alone passes validation.
    bound._validate_per_run_capabilities(make_ctx(bound_chain))  # pyright: ignore[reportPrivateUsage]


def test_durability_skips_per_run_check_when_dynamic_capability_bound():
    """Per-run validation is skipped when the bound chain contains a `DynamicCapability`.

    A `DynamicCapability` resolves to a different capability instance per run, so the
    static-class check would falsely reject any class produced by the factory. Issue
    #5253 tracks proper end-to-end durable support; until then, the check is relaxed.
    """
    from pydantic_ai._run_context import RunContext
    from pydantic_ai.capabilities import DynamicCapability
    from pydantic_ai.capabilities.abstract import AbstractCapability
    from pydantic_ai.capabilities.combined import CombinedCapability
    from pydantic_ai.result import RunUsage

    @dataclass
    class _UnregisteredCap(AbstractCapability[None]):
        pass

    durability = TemporalDurability()

    agent = Agent(
        _durability_fn_model,
        name='dynamic_cap_test',
        capabilities=[durability, lambda ctx: None],
    )
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    assert any(issubclass(cls, DynamicCapability) for cls in bound._bound_capability_classes)  # pyright: ignore[reportPrivateUsage]

    # Even with an unregistered class in the runtime chain, validation passes
    # because a `DynamicCapability` is in the bound chain.
    runtime_chain = CombinedCapability([*agent.root_capability.capabilities, _UnregisteredCap()])
    ctx = RunContext[None](deps=None, model=_durability_fn_model, usage=RunUsage(), root_capability=runtime_chain)
    bound._validate_per_run_capabilities(ctx)  # pyright: ignore[reportPrivateUsage]


# --- get_serialization_name returns None ---


def test_durability_get_serialization_name():
    """TemporalDurability.get_serialization_name() returns None."""
    assert TemporalDurability.get_serialization_name() is None


def test_durability_plugin_requires_durability_capability():
    """`AgentPlugin` raises a clear error when the agent has no `TemporalDurability`."""
    plain_agent = Agent(_durability_fn_model, name='no_cap_agent')
    with pytest.raises(UserError, match='no `TemporalDurability` capability'):
        AgentPlugin(plain_agent)


_pydantic_ai_agents_durable = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)
_pydantic_ai_agents_agent = Agent(
    _durability_fn_model,
    name='pydantic_ai_agents_attr_test',
    capabilities=[_pydantic_ai_agents_durable],
)


@workflow.defn
class _BareAgentWorkflowViaAttribute:
    __pydantic_ai_agents__ = [_pydantic_ai_agents_agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _pydantic_ai_agents_agent.run(prompt)
        return result.output


async def test_pydantic_ai_plugin_discovers_bare_agent_with_durability(client: Client):
    """`PydanticAIPlugin` registers activities from a bare `AbstractAgent` listed in `__pydantic_ai_agents__`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[_BareAgentWorkflowViaAttribute],
    ):
        output = await client.execute_workflow(
            _BareAgentWorkflowViaAttribute.run,
            args=['Discovered'],
            id=_BareAgentWorkflowViaAttribute.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == 'Echo: Discovered'


_missing_cap_agent = Agent(_durability_fn_model, name='no_cap_in_attr')


@workflow.defn
class _MissingCapWorkflow:
    __pydantic_ai_agents__ = [_missing_cap_agent]

    @workflow.run
    async def run(self, prompt: str) -> str:  # pragma: no cover - configure_worker rejects before exec
        result = await _missing_cap_agent.run(prompt)
        return result.output


async def test_pydantic_ai_plugin_rejects_bare_agent_without_durability(client: Client):
    """`PydanticAIPlugin` raises a clear error when an agent in `__pydantic_ai_agents__` lacks `TemporalDurability`."""
    with pytest.raises(UserError, match='no `TemporalDurability` capability'):
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[_MissingCapWorkflow],
        ):
            pass  # pragma: no cover - error raised before reaching here


# --- Toolset without ID raises UserError ---


def test_durability_toolset_without_id_raises():
    """TemporalDurability raises UserError for leaf toolsets without an ID."""
    durability = TemporalDurability()
    with pytest.raises(UserError, match='unique `id`'):
        Agent(
            _durability_fn_model,
            name='no_id_test',
            toolsets=[ExternalToolset(tool_defs=[ToolDefinition(name='ext_tool')])],
            capabilities=[durability],
        )


# --- temporalize returning non-TemporalWrapperToolset (line 294->297 branch) ---


def test_durability_non_temporal_wrapper_toolset_not_in_registry():
    """When temporalize returns a non-TemporalWrapperToolset, it's not added to the registry."""
    agent = Agent(
        _durability_fn_model,
        name='external_ts_test',
        toolsets=[ExternalToolset(tool_defs=[ToolDefinition(name='ext_tool')], id='ext')],
        capabilities=[TemporalDurability()],
    )
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    # ExternalToolset is not wrapped into a TemporalWrapperToolset by the default
    # temporalize_toolset, so 'ext' should not appear in _temporal_toolsets_by_id.
    assert 'ext' not in bound._temporal_toolsets_by_id  # pyright: ignore[reportPrivateUsage]
    # The agent's built-in <agent> FunctionToolset IS wrapped.
    assert '<agent>' in bound._temporal_toolsets_by_id  # pyright: ignore[reportPrivateUsage]


# --- get_wrapper_toolset returns None when no temporal toolsets ---


def test_durability_get_wrapper_toolset_returns_none():
    """get_wrapper_toolset returns None when _temporal_toolsets_by_id is empty."""
    # An unbound capability has an empty registry — `for_agent` is what populates it.
    durability = TemporalDurability()
    assert len(durability._temporal_toolsets_by_id) == 0  # pyright: ignore[reportPrivateUsage]

    dummy_toolset = FunctionToolset[object](id='dummy')
    assert durability.get_wrapper_toolset(dummy_toolset) is None


# --- get_wrapper_toolset swap returns unchanged toolset ---


def test_durability_get_wrapper_toolset_swap_unchanged():
    """get_wrapper_toolset's swap returns a toolset unchanged if its ID is not in the registry."""
    agent = Agent(_durability_fn_model, name='swap_test', capabilities=[TemporalDurability()])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None

    # Create a new toolset not registered with this durability
    unregistered_toolset = FunctionToolset(id='unregistered')
    result = bound.get_wrapper_toolset(unregistered_toolset)
    # The toolset should be returned as-is since its ID is not in the registry
    assert result is unregistered_toolset


# --- Streaming in workflow (event_stream_handler) ---


async def _stream_model_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'Stream'
    yield 'ed '
    yield 'response'


_stream_fn_model = FunctionModel(_durability_model_fn, stream_function=_stream_model_fn)

_stream_events_collected: list[AgentStreamEvent] = []


async def _durability_event_stream_handler(
    ctx: RunContext[object],
    stream: AsyncIterable[AgentStreamEvent],
) -> None:
    async for event in stream:
        _stream_events_collected.append(event)


_stream_durability = TemporalDurability(
    event_stream_handler=_durability_event_stream_handler,
    activity_config=BASE_ACTIVITY_CONFIG,
)
_stream_durable_agent = Agent(
    _stream_fn_model,
    name='durability_stream_agent',
    capabilities=[_stream_durability],
)


@workflow.defn
class StreamDurableAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _stream_durable_agent.run(prompt)
        return result.output


async def test_durability_streaming_in_workflow(client: Client):
    """TemporalDurability routes model requests through streaming activity when event_stream_handler is set."""
    _stream_events_collected.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[StreamDurableAgentWorkflow],
        plugins=[AgentPlugin(_stream_durable_agent)],
    ):
        output = await client.execute_workflow(
            StreamDurableAgentWorkflow.run,
            args=['Hello streaming'],
            id=StreamDurableAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        # The non-streaming FunctionModel function is NOT used for the streaming activity;
        # instead, request_stream_activity uses the stream_function path.
        # The final response is assembled from the streamed chunks.
        assert output == 'Streamed response'


# --- ProcessEventStream capability fires live inside the activity ---

_process_events_collected: list[AgentStreamEvent] = []


async def _process_event_stream_handler(
    ctx: RunContext[object],
    stream: AsyncIterable[AgentStreamEvent],
) -> None:
    async for event in stream:
        _process_events_collected.append(event)


_process_durability = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)
_process_durable_agent = Agent(
    _stream_fn_model,
    name='durability_process_agent',
    capabilities=[
        ProcessEventStream(_process_event_stream_handler),
        _process_durability,
    ],
)


@workflow.defn
class ProcessStreamDurableAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _process_durable_agent.run(prompt)
        return result.output


def test_durability_tool_metadata_disables_activity():
    """Tool metadata={'temporal': False} disables activity wrapping for that tool."""

    async def slow_tool() -> str:
        return 'slow'  # pragma: no cover - registered with toolset; test only verifies wrapping

    toolset = FunctionToolset[object](id='meta_toolset')
    toolset.add_function(slow_tool, metadata={'temporal': False})

    agent = Agent(
        _durability_fn_model,
        name='meta_disable_test',
        toolsets=[toolset],
        capabilities=[TemporalDurability()],
    )
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None

    # Should have wrapped the toolset (capability discovered it at for_agent time);
    # the per-tool skip is applied at call time via resolve_tool_activity_config.
    assert 'meta_toolset' in bound._temporal_toolsets_by_id  # pyright: ignore[reportPrivateUsage]


def test_resolve_tool_activity_config_reads_metadata():
    """Per-tool Temporal config from `tool_def.metadata['temporal']` takes priority."""
    from pydantic_ai.tools import ToolDefinition
    from pydantic_ai.toolsets import ToolsetTool
    from pydantic_ai_slim.pydantic_ai.durable_exec.temporal._toolset import resolve_tool_activity_config

    metadata_config = ActivityConfig(start_to_close_timeout=timedelta(seconds=120))

    fn_toolset = FunctionToolset[None](id='resolve_meta_toolset')

    async def fn_tool() -> str:
        return 'ok'  # pragma: no cover - registered with toolset; test only resolves metadata

    fn_toolset.add_function(fn_tool, metadata={'temporal': metadata_config})
    tool_def = ToolDefinition(name='fn_tool', metadata={'temporal': metadata_config})
    tool = ToolsetTool[None](
        toolset=fn_toolset,
        tool_def=tool_def,
        max_retries=0,
        args_validator=None,  # pyright: ignore[reportArgumentType]
    )

    # Metadata wins over the per-tool dict.
    resolved = resolve_tool_activity_config(tool, 'fn_tool', {'fn_tool': ActivityConfig(summary='from_dict')})
    assert resolved is metadata_config

    # `False` in metadata also wins.
    tool.tool_def.metadata = {'temporal': False}
    assert resolve_tool_activity_config(tool, 'fn_tool', {}) is False

    # Invalid metadata (e.g. a string from a misuse like `metadata={'temporal': '5s'}`)
    # raises `UserError` instead of silently passing the wrong shape to Temporal.
    tool.tool_def.metadata = {'temporal': '5s'}
    with pytest.raises(UserError, match=r"Tool 'fn_tool' has invalid 'temporal' metadata"):
        resolve_tool_activity_config(tool, 'fn_tool', {})


async def test_durability_process_event_stream_fires_live_inside_activity(client: Client):
    """ProcessEventStream (outer capability) sees live events emitted inside the Temporal activity.

    With in-activity chain firing, the capability's handler runs against the real streamed
    response — so multiple PartDeltaEvents come through (one per chunk). If the chain fired
    on the replayed stream outside the activity instead, ProcessEventStream would see a
    single synthetic delta with the full text.
    """
    _process_events_collected.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ProcessStreamDurableAgentWorkflow],
        plugins=[AgentPlugin(_process_durable_agent)],
    ):
        output = await client.execute_workflow(
            ProcessStreamDurableAgentWorkflow.run,
            args=['Hello'],
            id=ProcessStreamDurableAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == 'Streamed response'

    delta_events = [
        e for e in _process_events_collected if isinstance(e, PartDeltaEvent) and isinstance(e.delta, TextPartDelta)
    ]
    text_chunks = [cast(TextPartDelta, e.delta).content_delta for e in delta_events]
    # Live stream: the first chunk becomes the PartStartEvent's text; subsequent chunks
    # are deltas. Synthetic replay would collapse all chunks into a single delta with
    # the full text ('Streamed response').
    assert text_chunks == ['ed ', 'response']


# ==========================================
# TemporalDurability capability — parity with TemporalAgent wrapper tests
# ==========================================
#
# Each test below is the capability-path equivalent of a `TemporalAgent`-based
# test earlier in this file. They assert the same behaviors but use
# `Agent(..., capabilities=[TemporalDurability(...)])` and `AgentPlugin`
# instead of wrapping the agent.


# --- Complex agent: full Logfire span tree ---

# Mirrors the legacy `complex_agent`: the handler is passed to the capability (the
# capability-path equivalent of `TemporalAgent(event_stream_handler=...)`) so the
# resulting Logfire span tree matches the one produced by the wrapper-agent variant.
complex_durability_for_logfire = TemporalDurability[Deps](
    deps_type=Deps,
    event_stream_handler=event_stream_handler,
    activity_config=BASE_ACTIVITY_CONFIG,
    model_activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=90)),
    toolset_activity_config={
        'durability_complex_country': ActivityConfig(start_to_close_timeout=timedelta(seconds=120)),
    },
)
complex_durable_logfire_agent = Agent(
    model,
    deps_type=Deps,
    output_type=Response,
    toolsets=[
        FunctionToolset[Deps](tools=[get_country], id='durability_complex_country'),
        MCPToolset(
            StdioTransport(command='python', args=['-m', 'tests.mcp_server']),
            id='durability_complex_mcp',
            init_timeout=20,
        ),
        ExternalToolset(tool_defs=[ToolDefinition(name='external')], id='durability_complex_external'),
    ],
    tools=[get_weather],
    name='durability_complex_agent_logfire',
    capabilities=[complex_durability_for_logfire],
)


@workflow.defn
class ComplexDurableAgentLogfireWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: Deps) -> Response:
        result = await complex_durable_logfire_agent.run(prompt, deps=deps)
        return result.output


async def test_durability_complex_agent_logfire_span_tree(
    allow_model_requests: None, client_with_logfire: Client, capfire: CaptureLogfire
):
    """Capability-path equivalent of `test_complex_agent_run_in_workflow`.

    Asserts the Logfire span tree shape — span names will use
    `agent__durability_complex_agent_logfire__*` instead of `agent__complex_agent__*`,
    but the structure should otherwise match. Run with `--inline-snapshot=create`
    to populate the expected value on first run; needs a fresh VCR cassette under
    the new test name (record in CI / locally with `--record-mode=once`).
    """
    async with Worker(
        client_with_logfire,
        task_queue=TASK_QUEUE,
        workflows=[ComplexDurableAgentLogfireWorkflow],
        plugins=[AgentPlugin(complex_durable_logfire_agent)],
    ):
        output = await client_with_logfire.execute_workflow(
            ComplexDurableAgentLogfireWorkflow.run,
            args=[
                'Tell me: the capital of the country; the weather there; the product name',
                Deps(country='Mexico'),
            ],
            id=ComplexDurableAgentLogfireWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            Response(
                answers=[
                    Answer(label='Capital of the country', answer='Mexico City'),
                    Answer(label='Weather in the capital', answer='Sunny'),
                    Answer(label='Product Name', answer='Pydantic AI'),
                ]
            )
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

    assert root_span == snapshot(
        BasicSpan(
            content='StartWorkflow:ComplexDurableAgentLogfireWorkflow',
            children=[
                BasicSpan(content='RunWorkflow:ComplexDurableAgentLogfireWorkflow'),
                BasicSpan(
                    content='durability_complex_agent_logfire run',
                    children=[
                        BasicSpan(
                            content='StartActivity:agent__durability_complex_agent_logfire__mcp_server__durability_complex_mcp__get_tools',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__durability_complex_agent_logfire__mcp_server__durability_complex_mcp__get_tools',
                                    children=[BasicSpan(content='tools/list')],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__durability_complex_agent_logfire__model_request_stream',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__durability_complex_agent_logfire__model_request_stream',
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
                                )
                            ],
                        ),
                        BasicSpan(
                            content='running tool: get_country',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__durability_complex_agent_logfire__toolset__durability_complex_country__call_tool',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__durability_complex_agent_logfire__toolset__durability_complex_country__call_tool'
                                        )
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='running tool: get_product_name',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__durability_complex_agent_logfire__mcp_server__durability_complex_mcp__call_tool',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__durability_complex_agent_logfire__mcp_server__durability_complex_mcp__call_tool',
                                            children=[BasicSpan(content='tools/call get_product_name')],
                                        )
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__durability_complex_agent_logfire__model_request_stream',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__durability_complex_agent_logfire__model_request_stream',
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
                                )
                            ],
                        ),
                        BasicSpan(
                            content='running tool: get_weather',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__durability_complex_agent_logfire__toolset__<agent>__call_tool',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__durability_complex_agent_logfire__toolset__<agent>__call_tool'
                                        )
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__durability_complex_agent_logfire__model_request_stream',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__durability_complex_agent_logfire__model_request_stream',
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
                                )
                            ],
                        ),
                    ],
                ),
                BasicSpan(content='CompleteWorkflow:ComplexDurableAgentLogfireWorkflow'),
            ],
        )
    )


# --- Model retry ---


_durability_model_retry_agent = Agent(model, name='durability_model_retry_agent', capabilities=[TemporalDurability()])


@_durability_model_retry_agent.tool_plain
def durability_get_weather_in_city(city: str) -> str:
    if city != 'Mexico City':
        raise ModelRetry('Did you mean Mexico City?')
    return 'sunny'


@workflow.defn
class DurabilityModelRetryWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str]:
        result = await _durability_model_retry_agent.run(prompt)
        return result


async def test_durability_agent_with_model_retry(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_temporal_agent_with_model_retry`.

    Needs a fresh VCR cassette (different test name from the wrapper test);
    record locally with `--record-mode=once`.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityModelRetryWorkflow],
        plugins=[AgentPlugin(_durability_model_retry_agent)],
    ):
        wf = await client.start_workflow(
            DurabilityModelRetryWorkflow.run,
            args=['What is the weather in CDMX?'],
            id=DurabilityModelRetryWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        result = await wf.result()
        assert result.output == snapshot('The weather in Mexico City is currently sunny.')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='What is the weather in CDMX?', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='durability_get_weather_in_city',
                            args='{"city":"CDMX"}',
                            tool_call_id='call_TtLEMpCeAhnG48btCDrw8lhl',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=48,
                        output_tokens=20,
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
                    provider_details={'finish_reason': 'tool_calls', 'timestamp': '2026-05-08T21:37:16Z'},
                    provider_response_id='chatcmpl-DdNAiT49qrYrZOaeeAd39RynAa1g7',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Did you mean Mexico City?',
                            tool_name='durability_get_weather_in_city',
                            tool_call_id='call_TtLEMpCeAhnG48btCDrw8lhl',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='durability_get_weather_in_city',
                            args='{"city":"Mexico City"}',
                            tool_call_id='call_d8k0Vk8dw6eWKFWF8Dj0rCL6',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=93,
                        output_tokens=20,
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
                    provider_details={'finish_reason': 'tool_calls', 'timestamp': '2026-05-08T21:37:17Z'},
                    provider_response_id='chatcmpl-DdNAjt5pJt1nYbeCdbHGbo4ntTKy8',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='durability_get_weather_in_city',
                            content='sunny',
                            tool_call_id='call_d8k0Vk8dw6eWKFWF8Dj0rCL6',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='The weather in Mexico City is currently sunny.')],
                    usage=RequestUsage(
                        input_tokens=127,
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
                    provider_details={'finish_reason': 'stop', 'timestamp': '2026-05-08T21:37:18Z'},
                    provider_response_id='chatcmpl-DdNAkzvAFU1knSut20EiutyMs7PZy',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


# --- Multi-model selection by ID ---

_durability_model_1 = TestModel(custom_output_text='Response from model 1')
_durability_model_2 = TestModel(custom_output_text='Response from model 2')
_durability_model_3 = TestModel(custom_output_text='Response from model 3')

_durability_multi_model_agent = Agent(
    _durability_model_1,
    name='durability_multi_model_agent',
    capabilities=[
        TemporalDurability(
            models={
                'model_2': _durability_model_2,
                'model_3': _durability_model_3,
            },
            activity_config=BASE_ACTIVITY_CONFIG,
        )
    ],
)


@workflow.defn
class DurabilityMultiModelWorkflow:
    @workflow.run
    async def run(self, prompt: str, model_id: str | None = None) -> str:
        result = await _durability_multi_model_agent.run(prompt, model=model_id)
        return result.output


async def test_durability_multi_model_selection_in_workflow(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_temporal_agent_multi_model_selection_in_workflow`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityMultiModelWorkflow],
        plugins=[AgentPlugin(_durability_multi_model_agent)],
    ):
        # Default model (no model arg)
        output = await client.execute_workflow(
            DurabilityMultiModelWorkflow.run,
            args=['Hello', None],
            id='DurabilityMultiModelWorkflow_default',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Response from model 1'

        # Selecting registered second model by ID
        output = await client.execute_workflow(
            DurabilityMultiModelWorkflow.run,
            args=['Hello', 'model_2'],
            id='DurabilityMultiModelWorkflow_model2',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Response from model 2'

        # Selecting registered third model by ID
        output = await client.execute_workflow(
            DurabilityMultiModelWorkflow.run,
            args=['Hello', 'model_3'],
            id='DurabilityMultiModelWorkflow_model3',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Response from model 3'


# --- Model selection by instance ---

_durability_model_instance_map = {
    'default_instance': _durability_model_1,
    'model_2_instance': _durability_model_2,
}


@workflow.defn
class DurabilityMultiModelInstanceWorkflow:
    @workflow.run
    async def run(self, prompt: str, instance_key: str) -> str:
        model_instance = _durability_model_instance_map[instance_key]
        result = await _durability_multi_model_agent.run(prompt, model=model_instance)
        return result.output


@pytest.mark.parametrize(
    ('instance_key', 'expected_output'),
    [
        pytest.param('default_instance', 'Response from model 1', id='default_instance'),
        pytest.param('model_2_instance', 'Response from model 2', id='registered_instance'),
    ],
)
async def test_durability_model_selection_by_instance(
    allow_model_requests: None, client: Client, instance_key: str, expected_output: str
):
    """Capability-path equivalent of `test_temporal_agent_model_selection_by_instance`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityMultiModelInstanceWorkflow],
        plugins=[AgentPlugin(_durability_multi_model_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityMultiModelInstanceWorkflow.run,
            args=['Hello', instance_key],
            id=f'DurabilityMultiModelInstanceWorkflow_{instance_key}',
            task_queue=TASK_QUEUE,
        )
        assert output == expected_output


# --- Web search builtin tool ---

_durability_web_search_agent = Agent(
    web_search_model,
    name='durability_web_search_agent',
    capabilities=[
        NativeTool(WebSearchTool(user_location=WebSearchUserLocation(city='Mexico City', country='MX'))),
        TemporalDurability(
            activity_config=BASE_ACTIVITY_CONFIG,
            model_activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=300)),
        ),
    ],
)


@workflow.defn
class DurabilityWebSearchAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _durability_web_search_agent.run(prompt)
        return result.output


@pytest.mark.filterwarnings(  # TODO (v2): Remove this once we drop the deprecated events
    'ignore:`BuiltinToolCallEvent` is deprecated', 'ignore:`BuiltinToolResultEvent` is deprecated'
)
async def test_durability_web_search_in_workflow(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_web_search_agent_run_in_workflow`.

    Needs a fresh VCR cassette (different test name from the wrapper test);
    record in CI / locally with `--record-mode=once`.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityWebSearchAgentWorkflow],
        plugins=[AgentPlugin(_durability_web_search_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityWebSearchAgentWorkflow.run,
            args=['In one sentence, what is the top news story in my country today?'],
            id=DurabilityWebSearchAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            "Mexico's central bank cut its benchmark interest rate by 25 basis points to 6.50%--effective today, May 8, 2026--signaling the end of its rate‐cut cycle. ([banxico.org.mx](https://www.banxico.org.mx/publicaciones-y-prensa/anuncios-de-las-decisiones-de-politica-monetaria/%7B8A05C722-0A97-4527-2166-0CE802CE6838%7D.pdf?utm_source=openai))"
        )


# --- Dynamic builtin tools select-by-model ---

_durability_builtin_tool_agent = Agent(
    web_search_builtin_model,
    name='durability_builtin_tool_dynamic_agent',
    capabilities=[
        NativeTool(_select_builtin_tool),
        TemporalDurability(
            models={'code': code_execution_builtin_model},
            activity_config=BASE_ACTIVITY_CONFIG,
        ),
    ],
)


@workflow.defn
class DurabilityBuiltinToolWorkflow:
    @workflow.run
    async def run(self, prompt: str, model_id: str | None = None) -> str:
        result = await _durability_builtin_tool_agent.run(prompt, model=model_id)
        return result.output


async def test_durability_dynamic_builtin_tools_select_by_model(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_temporal_dynamic_builtin_tools_select_by_model`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityBuiltinToolWorkflow],
        plugins=[AgentPlugin(_durability_builtin_tool_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityBuiltinToolWorkflow.run,
            args=['Hello', None],
            id='DurabilityBuiltinToolWorkflow_default',
            task_queue=TASK_QUEUE,
        )
        assert output == 'search model'
        assert isinstance(web_search_builtin_model.last_model_request_parameters, ModelRequestParameters)
        assert web_search_builtin_model.last_model_request_parameters.native_tools
        assert isinstance(web_search_builtin_model.last_model_request_parameters.native_tools[0], WebSearchTool)

        output = await client.execute_workflow(
            DurabilityBuiltinToolWorkflow.run,
            args=['Hello', 'code'],
            id='DurabilityBuiltinToolWorkflow_code',
            task_queue=TASK_QUEUE,
        )
        assert output == 'code model'
        assert isinstance(code_execution_builtin_model.last_model_request_parameters, ModelRequestParameters)
        assert code_execution_builtin_model.last_model_request_parameters.native_tools
        assert isinstance(
            code_execution_builtin_model.last_model_request_parameters.native_tools[0],
            CodeExecutionTool,
        )


# --- @agent.toolset returning an MCP toolset ---

_durability_mcp_dynamic_toolset_agent = Agent(
    model,
    name='durability_mcp_dynamic_toolset_agent',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@_durability_mcp_dynamic_toolset_agent.toolset(id='durability_mcp_toolset')
def _durability_my_mcp_dynamic_toolset(ctx: RunContext[object]) -> MCPToolset[object]:
    return MCPToolset('https://mcp.deepwiki.com/mcp')  # pragma: no cover - exercised only by the skipped test below


@workflow.defn
class DurabilityMCPDynamicToolsetAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _durability_mcp_dynamic_toolset_agent.run(prompt)  # pragma: no cover - skipped test
        return result.output  # pragma: no cover - skipped test


@pytest.mark.skip(
    reason=(
        'Pending: replays of this MCP toolset workflow trip the Temporal sandbox with '
        '`Module certifi was imported after initial workflow load`. Issue tracked.'
    )
)
async def test_durability_mcp_dynamic_toolset_in_workflow(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_mcp_dynamic_toolset_in_workflow`.

    Needs a fresh VCR cassette (different test name from the wrapper test);
    record in CI / locally with `--record-mode=once`.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityMCPDynamicToolsetAgentWorkflow],
        plugins=[AgentPlugin(_durability_mcp_dynamic_toolset_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityMCPDynamicToolsetAgentWorkflow.run,
            args=['Can you tell me about the pydantic/pydantic-ai repo? Keep it short.'],
            id='test_durability_mcp_dynamic_toolset_workflow',
            task_queue=TASK_QUEUE,
        )
        # The deepwiki MCP server should return info about the pydantic-ai repo
        assert 'pydantic' in output.lower() or 'agent' in output.lower()


# --- MCPToolset over HTTP ---

_durability_mcptoolset_agent = Agent(
    model,
    name='durability_mcptoolset_agent',
    toolsets=[MCPToolset('https://mcp.deepwiki.com/mcp', id='durability_deepwiki')],
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class DurabilityMCPToolsetAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _durability_mcptoolset_agent.run(prompt)  # pragma: no cover - skipped test
        return result.output  # pragma: no cover - skipped test


@pytest.mark.skip(
    reason=(
        'Pending: replays of this MCP toolset workflow trip the Temporal sandbox with '
        '`Module certifi was imported after initial workflow load`. Issue tracked.'
    )
)
async def test_durability_mcptoolset_in_workflow(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_mcptoolset_in_temporal_workflow`.

    Needs a fresh VCR cassette (different test name from the wrapper test);
    record in CI / locally with `--record-mode=once`.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityMCPToolsetAgentWorkflow],
        plugins=[AgentPlugin(_durability_mcptoolset_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityMCPToolsetAgentWorkflow.run,
            args=['Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short'],
            id=DurabilityMCPToolsetAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot()


# --- @agent.toolset returning a FunctionToolset ---

_durability_dynamic_toolset_agent = Agent(
    TestModel(),
    name='durability_dynamic_toolset_agent',
    deps_type=DynamicToolsetDeps,
    capabilities=[
        TemporalDurability[DynamicToolsetDeps](deps_type=DynamicToolsetDeps, activity_config=BASE_ACTIVITY_CONFIG)
    ],
)


@_durability_dynamic_toolset_agent.toolset(id='durability_my_dynamic_tools')
def _durability_my_dynamic_toolset(ctx: RunContext[DynamicToolsetDeps]) -> FunctionToolset[DynamicToolsetDeps]:
    toolset = FunctionToolset[DynamicToolsetDeps](id='durability_dynamic_weather')

    @toolset.tool_plain
    def get_dynamic_weather(location: str) -> str:
        """Get the weather for a location."""
        user = ctx.deps.user_name
        return f'Weather in {location} for {user}: sunny.'

    return toolset


@workflow.defn
class DurabilityDynamicToolsetAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: DynamicToolsetDeps) -> str:
        result = await _durability_dynamic_toolset_agent.run(prompt, deps=deps)
        return result.output


async def test_durability_dynamic_toolset_in_workflow(client: Client):
    """Capability-path equivalent of `test_dynamic_toolset_in_workflow`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityDynamicToolsetAgentWorkflow],
        plugins=[AgentPlugin(_durability_dynamic_toolset_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityDynamicToolsetAgentWorkflow.run,
            args=['Get the weather for London', DynamicToolsetDeps(user_name='Alice')],
            id='test_durability_dynamic_toolset_workflow',
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('{"get_dynamic_weather":"Weather in a for Alice: sunny."}')


# --- ToolReturn metadata round-trip ---


def _durability_tool_return_metadata_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('durability_analyze_data', {})])
    else:
        return ModelResponse(parts=[TextPart('done')])


_durability_tool_return_metadata_agent = Agent(
    FunctionModel(_durability_tool_return_metadata_model),
    name='durability_tool_return_metadata_agent',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@_durability_tool_return_metadata_agent.tool_plain
def durability_analyze_data() -> ToolReturn:
    return ToolReturn(
        return_value='analysis result',
        content='extra content for model',
        metadata={'key': 'value', 'count': 42},
    )


@workflow.defn
class DurabilityToolReturnMetadataWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[ModelMessage]:
        result = await _durability_tool_return_metadata_agent.run(prompt)
        return result.all_messages()


async def test_durability_tool_return_metadata_survives(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_tool_return_metadata_survives_temporal`.

    Regression test for https://github.com/pydantic/pydantic-ai/issues/4676 — `ToolReturn`
    `metadata` and `content` survive Temporal serialization on the capability path too.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityToolReturnMetadataWorkflow],
        plugins=[AgentPlugin(_durability_tool_return_metadata_agent)],
    ):
        messages = await client.execute_workflow(
            DurabilityToolReturnMetadataWorkflow.run,
            args=['analyze'],
            id=DurabilityToolReturnMetadataWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='analyze', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='durability_analyze_data', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=IsInt(), output_tokens=IsInt()),
                model_name='function:_durability_tool_return_metadata_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='durability_analyze_data',
                        content='analysis result',
                        tool_call_id=IsStr(),
                        metadata={'key': 'value', 'count': 42},
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(content='extra content for model', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=IsInt(), output_tokens=IsInt()),
                model_name='function:_durability_tool_return_metadata_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


# --- Passing image (BinaryImage) input through to a workflow ---

_durability_multimodal_agent = Agent(
    TestModel(),
    name='durability_multimodal_content_agent',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@_durability_multimodal_agent.tool
def _durability_get_multimodal_content(ctx: RunContext[object]) -> list[str | MultiModalContent]:
    """Return a list with text, BinaryContent, and DocumentUrl."""
    return [
        'test',
        BinaryImage(data=b'\x89PNG', media_type='image/png'),
        DocumentUrl(url='https://example.com/doc/12345', media_type='application/pdf'),
    ]


@workflow.defn
class DurabilityMultiModalContentWorkflow:
    @workflow.run
    async def run(self, prompt: list[UserContent]) -> list[ModelMessage]:
        result = await _durability_multimodal_agent.run(prompt)
        return result.all_messages()


async def test_durability_passing_image_to_run(client: Client):
    """Capability-path equivalent of `test_multimodal_content_serialization_in_workflow` — image input.

    Verifies BinaryImage / DocumentUrl survive Temporal serialization both as workflow
    input and as tool return values when running on the TemporalDurability capability path.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityMultiModalContentWorkflow],
        plugins=[AgentPlugin(_durability_multimodal_agent)],
    ):
        prompt: list[str | MultiModalContent] = [
            'Process these files and call the tool',
            BinaryImage(data=b'\x89PNG', media_type='image/png'),
            DocumentUrl(url='https://example.com/doc/12345', media_type='application/pdf'),
        ]
        messages = await client.execute_workflow(
            DurabilityMultiModalContentWorkflow.run,
            args=[prompt],
            id='test_durability_passing_image_to_run',
            task_queue=TASK_QUEUE,
        )

    # media_type is preserved through serialization for both BinaryContent and DocumentUrl.
    media_types: list[tuple[str, str]] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                for content in part.content:
                    if isinstance(content, (BinaryContent, DocumentUrl)):
                        media_types.append((type(content).__name__, content.media_type))
            elif isinstance(part, ToolReturnPart):
                for content in part.content_items():
                    if isinstance(content, (BinaryContent, DocumentUrl)):
                        media_types.append((type(content).__name__, content.media_type))
    # The image `BinaryContent` round-trips as `BinaryImage`: narrowing is applied during
    # validation on the way back across the activity boundary.
    assert media_types == [
        ('BinaryImage', 'image/png'),
        ('DocumentUrl', 'application/pdf'),
        ('BinaryImage', 'image/png'),
        ('DocumentUrl', 'application/pdf'),
    ]


# --- UploadedFile output round-trip ---

_durability_uploaded_file_agent = Agent(
    TestModel(
        custom_output_args={
            'file_id': 'file-abc123',
            'provider_name': 'openai',
            'media_type': 'image/png',
            'identifier': 'file-1',
        }
    ),
    name='durability_uploaded_file_agent',
    output_type=UploadedFile,
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class DurabilityUploadedFileAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> UploadedFile:
        result = await _durability_uploaded_file_agent.run(prompt)
        return result.output


async def test_durability_uploaded_file_serialization_preserves_media_type(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_uploaded_file_serialization_preserves_media_type`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityUploadedFileAgentWorkflow],
        plugins=[AgentPlugin(_durability_uploaded_file_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityUploadedFileAgentWorkflow.run,
            args=['Return a file reference'],
            id=DurabilityUploadedFileAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            UploadedFile(file_id='file-abc123', provider_name='openai', _media_type='image/png', _identifier='file-1')
        )


# --- Toolsets at runtime ---


@workflow.defn
class DurabilityRuntimeFunctionToolsetWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_durable_agent.run(prompt, toolsets=[FunctionToolset()])
        return result.output  # pragma: no cover


async def test_durability_rejects_runtime_executing_toolsets_in_workflow(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_temporal_agent_run_in_workflow_with_executing_toolsets`.

    Executing toolsets can't be added per-run inside a workflow because their activities must
    be registered with the worker before the workflow runs.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityRuntimeFunctionToolsetWorkflow],
        plugins=[AgentPlugin(simple_durable_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'FunctionToolset cannot be passed to `run(toolsets=...)` at runtime with Temporal, because '
                'toolsets that execute their own tools or resolve dynamically must be registered for durable '
                'execution when the agent is constructed. Pass them to the agent constructor instead. '
                'Non-executing toolsets like `ExternalToolset` can be passed at runtime.'
            ),
        ):
            await client.execute_workflow(
                DurabilityRuntimeFunctionToolsetWorkflow.run,
                args=['What is the capital of Mexico?'],
                id=DurabilityRuntimeFunctionToolsetWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )


async def test_durability_allows_runtime_toolsets_outside_workflow(allow_model_requests: None):
    """Outside a workflow the capability is transparent, so per-run executing toolsets are fine."""

    def call_then_answer(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('runtime_tool', {}, tool_call_id='call-1')])

    def runtime_tool() -> str:
        return 'tool-result'

    agent = Agent(
        FunctionModel(call_then_answer),
        name='durability_runtime_outside_workflow',
        capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
    )
    result = await agent.run(
        'Call the runtime tool.', toolsets=[FunctionToolset(tools=[runtime_tool], id='runtime_fn')]
    )
    assert result.output == 'done'


def _durability_request_external_tool(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('external', {'query': 'runtime'}, tool_call_id='call-1')])


_durability_runtime_external_agent = Agent(
    FunctionModel(_durability_request_external_tool),
    name='durability_runtime_external_agent',
    output_type=[str, DeferredToolRequests],
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)

_durability_runtime_external_toolset = ExternalToolset(
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


@workflow.defn
class DurabilityRuntimeExternalToolsetWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> DeferredToolRequests | str:
        result = await _durability_runtime_external_agent.run(prompt, toolsets=[_durability_runtime_external_toolset])
        return result.output


async def test_durability_run_in_workflow_with_runtime_external_toolset(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_temporal_agent_run_in_workflow_with_runtime_external_toolset`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityRuntimeExternalToolsetWorkflow],
        plugins=[AgentPlugin(_durability_runtime_external_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityRuntimeExternalToolsetWorkflow.run,
            args=['Call the runtime external tool.'],
            id=DurabilityRuntimeExternalToolsetWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == DeferredToolRequests(
            calls=[ToolCallPart('external', {'query': 'runtime'}, tool_call_id='call-1')]
        )


# --- Capability-contributed toolsets ---


def _durability_call_where_am_i(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(parts=[TextPart(str(part.content))])
    return ModelResponse(parts=[ToolCallPart('where_am_i', {}, tool_call_id='call-1')])


def where_am_i() -> str:
    return 'activity' if activity.in_activity() else 'workflow'


_durability_cap_toolset_agent = Agent(
    FunctionModel(_durability_call_where_am_i),
    name='durability_cap_toolset_agent',
    capabilities=[
        Toolset(FunctionToolset([where_am_i], id='cap_tools')),
        TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class DurabilityCapabilityToolsetWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _durability_cap_toolset_agent.run(prompt)
        return result.output


async def test_durability_temporalizes_capability_contributed_toolsets(allow_model_requests: None, client: Client):
    """Toolsets contributed by other capabilities run as Temporal activities.

    Durability capabilities are in the `innermost` ordering tier, so `Agent.__init__` binds
    them only after every other capability's contributed toolsets have been extracted into
    `agent.toolsets`. Without that two-phase binding, the `Toolset(...)` capability's tools
    would be invisible to `for_agent` and run unwrapped (non-deterministically) inside the
    workflow instead of in an activity.
    """
    durability = TemporalDurability.from_agent(_durability_cap_toolset_agent)
    assert durability is not None
    assert 'agent__durability_cap_toolset_agent__toolset__cap_tools__call_tool' in [
        ActivityDefinition.must_from_callable(act).name  # pyright: ignore[reportUnknownMemberType]
        for act in durability.temporal_activities
    ]

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityCapabilityToolsetWorkflow],
        plugins=[AgentPlugin(_durability_cap_toolset_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityCapabilityToolsetWorkflow.run,
            args=['Where does the tool run?'],
            id=DurabilityCapabilityToolsetWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == 'activity'


# --- Continuation chains (suspended → complete) resolve inside the activity ---
#
# When a model suspends a turn (Anthropic `pause_turn`, OpenAI background mode), the
# continuation loop lives in the innermost `model_request`/`model_request_stream` helpers,
# so under `TemporalDurability` the whole suspended → complete chain runs inside ONE
# model-request activity. These tests use a scripted model (no cassettes: `FunctionModel`
# can't emit suspended streaming segments, and VCR matchers wouldn't pin the chain shape).


def _workflow_failure_cause(exc: WorkflowFailureError) -> ApplicationError:
    """The innermost `ApplicationError` of a workflow failure (walking through `ActivityError`)."""
    cause: BaseException | None = exc.__cause__
    while cause is not None and not isinstance(cause, ApplicationError):
        cause = cause.__cause__
    assert isinstance(cause, ApplicationError), f'expected ApplicationError in cause chain of {exc!r}'
    return cause


def _scheduled_activity_count(history: WorkflowHistory) -> int:
    return len([e for e in history.events if e.HasField('activity_task_scheduled_event_attributes')])


_continuation_model = ScriptedContinuationModel()
_continuation_agent = Agent(
    _continuation_model,
    name='durability_continuation_agent',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class DurabilityContinuationWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str]:
        return await _continuation_agent.run(prompt)


@workflow.defn
class DurabilityContinuationResumeWorkflow:
    @workflow.run
    async def run(self, messages: list[ModelMessage]) -> AgentRunResult[str]:
        return await _continuation_agent.run(message_history=messages)


@workflow.defn
class DurabilityContinuationUsageLimitWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str]:
        return await _continuation_agent.run(prompt, usage_limits=UsageLimits(total_tokens_limit=20))


async def test_durability_continuation_chain_in_workflow(client: Client):
    """A suspended → complete chain resolves inside one activity, as one merged response.

    Usage is counted once (a continuation isn't a separate request step), and the workflow
    history shows a single scheduled activity for both segments.
    """
    _continuation_model.reset(
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
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityContinuationWorkflow],
        plugins=[AgentPlugin(_continuation_agent)],
    ):
        wf = await client.start_workflow(
            DurabilityContinuationWorkflow.run,
            args=['go'],
            id='DurabilityContinuationWorkflow_chain',
            task_queue=TASK_QUEUE,
        )
        result = await wf.result()
        history = await wf.fetch_history()

    assert result.output == 'The answer is 42.'
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'complete'
    assert [part.content for part in response.parts if isinstance(part, TextPart)] == ['The answer ', 'is 42.']
    usage = result.usage
    assert usage.requests == 1
    assert usage.input_tokens == 8
    assert usage.output_tokens == 6
    # Both segments ran inside the durable boundary: two model calls, one activity.
    assert _continuation_model.request_calls == 2
    assert _scheduled_activity_count(history) == 1


async def test_durability_continuation_resume_from_history(client: Client):
    """A `message_history` ending in a suspended response resumes inside the activity.

    The suspended tail crosses the activity boundary as the last request message and seeds
    the continuation loop there, so the run completes the paused turn instead of starting a
    fresh generation.
    """
    _continuation_model.reset(
        responses=[scripted_response(texts=['is 42.'], provider_response_id='cont2', input_tokens=3, output_tokens=4)]
    )
    history_messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='go')]),
        scripted_response(
            texts=['The answer '], state='suspended', provider_response_id='cont1', input_tokens=5, output_tokens=2
        ),
    ]
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityContinuationResumeWorkflow],
        plugins=[AgentPlugin(_continuation_agent)],
    ):
        wf = await client.start_workflow(
            DurabilityContinuationResumeWorkflow.run,
            args=[history_messages],
            id='DurabilityContinuationWorkflow_resume',
            task_queue=TASK_QUEUE,
        )
        result = await wf.result()
        history = await wf.fetch_history()

    assert result.output == 'The answer is 42.'
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'complete'
    assert [part.content for part in response.parts if isinstance(part, TextPart)] == ['The answer ', 'is 42.']
    usage = result.usage
    assert usage.requests == 1
    assert usage.input_tokens == 8
    assert usage.output_tokens == 6
    # The continuation request ran inside the boundary — the seed wasn't re-generated.
    assert _continuation_model.request_calls == 1
    assert _scheduled_activity_count(history) == 1


async def test_durability_continuation_error_cancels_job_inside_activity(client: Client):
    """A request failure mid-chain cancels the suspended server-side job inside the activity.

    The cancel-on-error policy runs on the real model inside the durable boundary — the
    workflow side never sees the live suspended response.
    """
    _continuation_model.reset(
        responses=[
            scripted_response(
                texts=['The answer '],
                state='suspended',
                provider_response_id='cont1',
                input_tokens=5,
                output_tokens=2,
            ),
            RuntimeError('provider blew up'),
        ]
    )
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityContinuationWorkflow],
        plugins=[AgentPlugin(_continuation_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                DurabilityContinuationWorkflow.run,
                args=['go'],
                id='DurabilityContinuationWorkflow_cancel_on_error',
                task_queue=TASK_QUEUE,
            )

    cause = _workflow_failure_cause(exc_info.value)
    assert cause.type == 'RuntimeError'
    assert cause.message == 'provider blew up'
    assert _continuation_model.request_calls == 2
    assert len(_continuation_model.cancelled) == 1
    assert _continuation_model.cancelled[0].provider_response_id == 'cont1'


async def test_durability_continuation_usage_limit_checked_inside_activity(client: Client):
    """Token limits are enforced mid-chain inside the activity, cancelling the live job.

    `usage`/`usage_limits` cross the activity boundary on the serialized run context (a
    custom `TemporalRunContext` subclass must keep including them), so a runaway
    continuation fails fast without waiting for the workflow-side commit.
    """
    _continuation_model.reset(
        responses=[
            scripted_response(
                texts=['The answer '],
                state='suspended',
                provider_response_id='cont1',
                input_tokens=5,
                output_tokens=2,
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
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityContinuationUsageLimitWorkflow],
        plugins=[AgentPlugin(_continuation_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                DurabilityContinuationUsageLimitWorkflow.run,
                args=['go'],
                id='DurabilityContinuationWorkflow_usage_limit',
                task_queue=TASK_QUEUE,
            )

    cause = _workflow_failure_cause(exc_info.value)
    assert cause.type == UsageLimitExceeded.__name__
    assert 'total_tokens_limit' in cause.message
    assert _continuation_model.request_calls == 2
    # The over-budget merge was still suspended, so the live job was cancelled before raising.
    assert len(_continuation_model.cancelled) == 1
    assert _continuation_model.cancelled[0].provider_response_id == 'cont2'


_continuation_ceiling_model = ScriptedContinuationModel()
_continuation_ceiling_agent = Agent(
    _continuation_ceiling_model,
    name='durability_continuation_ceiling_agent',
    capabilities=[
        TemporalDurability(
            activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=60),
                # More than one attempt allowed, to prove `UnexpectedModelBehavior` is
                # non-retryable rather than merely running out of attempts.
                retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(milliseconds=10)),
            )
        )
    ],
)


@workflow.defn
class DurabilityContinuationCeilingWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str]:
        return await _continuation_ceiling_agent.run(prompt)


async def test_durability_continuation_ceiling_surfaces_unexpected_model_behavior(client: Client):
    """Exceeding the continuation ceiling fails the workflow without activity retries.

    `UnexpectedModelBehavior` is in the activity retry policy's non-retryable error types:
    re-running the whole chain wouldn't fix a model that never leaves `'suspended'`, it
    would only re-incur its cost. The single-attempt call count proves no retry happened.
    """
    _continuation_ceiling_model.reset(
        responses=[
            scripted_response(
                texts=[f'segment {i} '],
                state='suspended',
                provider_response_id=f'cont{i}',
                input_tokens=1,
                output_tokens=1,
            )
            for i in range(1, 12)
        ]
    )
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityContinuationCeilingWorkflow],
        plugins=[AgentPlugin(_continuation_ceiling_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                DurabilityContinuationCeilingWorkflow.run,
                args=['go'],
                id=DurabilityContinuationCeilingWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )

    cause = _workflow_failure_cause(exc_info.value)
    assert cause.type == UnexpectedModelBehavior.__name__
    assert cause.message == snapshot("Model response 'cont11' was suspended more than the maximum of 10 times")
    # 1 initial + 10 continuation requests, from a single activity attempt (no retries).
    assert _continuation_ceiling_model.request_calls == 11
    # Giving up on a still-suspended job cancels it inside the activity so it doesn't leak.
    assert len(_continuation_ceiling_model.cancelled) == 1


# --- Streaming continuation chains inside the activity ---

_continuation_stream_model = ScriptedContinuationModel()
_continuation_stream_events: list[AgentStreamEvent] = []


async def _continuation_event_stream_handler(
    ctx: RunContext[object],
    stream: AsyncIterable[AgentStreamEvent],
) -> None:
    async for event in stream:
        _continuation_stream_events.append(event)


_continuation_stream_agent = Agent(
    _continuation_stream_model,
    name='durability_continuation_stream_agent',
    capabilities=[
        TemporalDurability(
            event_stream_handler=_continuation_event_stream_handler,
            activity_config=BASE_ACTIVITY_CONFIG,
        )
    ],
)


@workflow.defn
class DurabilityContinuationStreamWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str]:
        return await _continuation_stream_agent.run(prompt)


@workflow.defn
class DurabilityContinuationStreamResumeWorkflow:
    @workflow.run
    async def run(self, messages: list[ModelMessage]) -> AgentRunResult[str]:
        return await _continuation_stream_agent.run(message_history=messages)


def _text_part_indices(events: list[AgentStreamEvent]) -> list[tuple[str, int]]:
    return [
        (type(event).__name__, event.index) for event in events if isinstance(event, (PartStartEvent, PartDeltaEvent))
    ]


async def test_durability_streaming_continuation_chain_in_workflow(client: Client):
    """A streamed suspended → complete chain is stitched into one stream inside one activity.

    The `event_stream_handler` fires inside the activity against the live stitched stream:
    the second segment's part indices are offset past the first's (no collision), and the
    final response merges both segments' text with usage summed once.
    """
    _continuation_stream_model.reset(
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
    _continuation_stream_events.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityContinuationStreamWorkflow],
        plugins=[AgentPlugin(_continuation_stream_agent)],
    ):
        wf = await client.start_workflow(
            DurabilityContinuationStreamWorkflow.run,
            args=['go'],
            id='DurabilityContinuationStreamWorkflow_chain',
            task_queue=TASK_QUEUE,
        )
        result = await wf.result()
        history = await wf.fetch_history()

    assert result.output == 'The answer is 42.'
    usage = result.usage
    assert usage.requests == 1
    assert usage.input_tokens == 8
    assert usage.output_tokens == 6
    # The handler saw the stitched stream live inside the activity: the second segment's
    # part continues the index space rather than colliding with the first's.
    assert _text_part_indices(_continuation_stream_events) == snapshot(
        [
            ('PartStartEvent', 0),
            ('PartDeltaEvent', 0),
            ('PartStartEvent', 1),
            ('PartDeltaEvent', 1),
        ]
    )
    assert _continuation_stream_model.request_stream_calls == 2
    assert _scheduled_activity_count(history) == 1


async def test_durability_streaming_continuation_resume_from_history(client: Client):
    """A streamed resume of a suspended `message_history` tail completes inside the activity.

    The suspended tail seeds the stitched stream, so the new segment's part indices start
    after the seed's parts and the final output merges seed and continuation text.
    """
    _continuation_stream_model.reset(
        segments=[
            StreamSegment(
                texts=['is 42.'], state='complete', provider_response_id='cont2', input_tokens=3, output_tokens=4
            ),
        ]
    )
    _continuation_stream_events.clear()
    history_messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='go')]),
        scripted_response(
            texts=['The answer '], state='suspended', provider_response_id='cont1', input_tokens=5, output_tokens=2
        ),
    ]
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityContinuationStreamResumeWorkflow],
        plugins=[AgentPlugin(_continuation_stream_agent)],
    ):
        wf = await client.start_workflow(
            DurabilityContinuationStreamResumeWorkflow.run,
            args=[history_messages],
            id='DurabilityContinuationStreamWorkflow_resume',
            task_queue=TASK_QUEUE,
        )
        result = await wf.result()

    assert result.output == 'The answer is 42.'
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'complete'
    assert [part.content for part in response.parts if isinstance(part, TextPart)] == ['The answer ', 'is 42.']
    # The new segment's part is indexed after the seed's single part.
    assert _text_part_indices(_continuation_stream_events) == snapshot(
        [
            ('PartStartEvent', 1),
            ('PartDeltaEvent', 1),
        ]
    )
    assert _continuation_stream_model.request_stream_calls == 1
