from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterable, AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    AgentStreamEvent,
    BinaryImage,
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
    RetryPromptPart,
    RunContext,
    RunUsage,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
    WebSearchTool,
    WebSearchUserLocation,
)
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.models import Model, ModelRequestParameters, cached_async_http_client
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition
from pydantic_ai.usage import RequestUsage
from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_append

try:
    import temporalio.api.common.v1
    from temporalio import workflow
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]
    from temporalio.client import Client, WorkflowFailureError
    from temporalio.common import RetryPolicy
    from temporalio.contrib.opentelemetry import TracingInterceptor
    from temporalio.contrib.pydantic import PydanticPayloadConverter, pydantic_data_converter
    from temporalio.converter import DataConverter, DefaultPayloadConverter, PayloadCodec
    from temporalio.exceptions import ApplicationError
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        LogfirePlugin,
        PydanticAIPlugin,
        PydanticAIWorkflow,
        TemporalAgent,
    )
    from pydantic_ai.durable_exec.temporal._function_toolset import TemporalFunctionToolset
    from pydantic_ai.durable_exec.temporal._mcp_server import TemporalMCPServer
    from pydantic_ai.durable_exec.temporal._model import TemporalModel
    from pydantic_ai.durable_exec.temporal._run_context import TemporalRunContext
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

try:
    import logfire
    from logfire import Logfire
    from logfire._internal.tracer import _ProxyTracer  # pyright: ignore[reportPrivateUsage]
    from logfire.testing import CaptureLogfire
    from opentelemetry.trace import ProxyTracer
except ImportError:  # pragma: lax no cover
    pytest.skip('logfire not installed', allow_module_level=True)

try:
    from pydantic_ai.mcp import MCPServerStdio, MCPServerStreamableHTTP
except ImportError:  # pragma: lax no cover
    pytest.skip('mcp not installed', allow_module_level=True)

try:
    from pydantic_ai.toolsets.fastmcp import FastMCPToolset
except ImportError:  # pragma: lax no cover
    pytest.skip('fastmcp not installed', allow_module_level=True)

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
    from inline_snapshot import snapshot

    # Loads `vcr`, which Temporal doesn't like without passing through the import
    from .conftest import IsDatetime, IsStr

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.xdist_group(name='temporal'),
]


# We need to use a custom cached HTTP client here as the default one created for OpenAIProvider will be closed automatically
# at the end of each test, but we need this one to live longer.
http_client = cached_async_http_client(provider='temporal')


@pytest.fixture(autouse=True, scope='module')
async def close_cached_httpx_client(anyio_backend: str) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await http_client.aclose()


# `LogfirePlugin` calls `logfire.instrument_pydantic_ai()`, so we need to make sure this doesn't bleed into other tests.
@pytest.fixture(autouse=True, scope='module')
def uninstrument_pydantic_ai() -> Iterator[None]:
    try:
        yield
    finally:
        Agent.instrument_all(False)


@contextmanager
def workflow_raises(exc_type: type[Exception], exc_message: str) -> Iterator[None]:
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


@pytest.fixture(scope='module')
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
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
        MCPServerStdio('python', ['-m', 'tests.mcp_server'], timeout=20, id='mcp'),
        ExternalToolset(tool_defs=[ToolDefinition(name='external')], id='external'),
    ],
    tools=[get_weather],
    event_stream_handler=event_stream_handler,
    name='complex_agent',
)

# This needs to be done before the `TemporalAgent` is bound to the workflow.
complex_temporal_agent = TemporalAgent(
    complex_agent,
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
    children: list[BasicSpan] = field(default_factory=list)
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
                                BasicSpan(content='RunActivity:agent__complex_agent__mcp_server__mcp__get_tools')
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
                                                    content='{"index":0,"part":{"tool_name":"get_country","args":"","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","id":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":null,"event_kind":"part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{}","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"part":{"tool_name":"get_country","args":"{}","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","id":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":"tool-call","event_kind":"part_end"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":1,"part":{"tool_name":"get_product_name","args":"","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","id":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":"tool-call","event_kind":"part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":1,"delta":{"tool_name_delta":null,"args_delta":"{}","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":1,"part":{"tool_name":"get_product_name","args":"{}","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","id":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":null,"event_kind":"part_end"}'
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
                                            content='{"part":{"tool_name":"get_country","args":"{}","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","id":null,"provider_details":null,"part_kind":"tool-call"},"event_kind":"function_tool_call"}'
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
                                            content='{"part":{"tool_name":"get_product_name","args":"{}","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","id":null,"provider_details":null,"part_kind":"tool-call"},"event_kind":"function_tool_call"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='running 2 tools',
                            children=[
                                BasicSpan(content='running tool: get_country'),
                                BasicSpan(
                                    content='StartActivity:agent__complex_agent__event_stream_handler',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__complex_agent__event_stream_handler',
                                            children=[
                                                BasicSpan(content='ctx.run_step=1'),
                                                BasicSpan(
                                                    content=IsStr(
                                                        regex=r'{"result":{"tool_name":"get_country","content":"Mexico","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"content":null,"event_kind":"function_tool_result"}'
                                                    )
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
                                                    content='RunActivity:agent__complex_agent__mcp_server__mcp__call_tool'
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
                                                    content=IsStr(
                                                        regex=r'{"result":{"tool_name":"get_product_name","content":"Pydantic AI","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"content":null,"event_kind":"function_tool_result"}'
                                                    )
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__complex_agent__mcp_server__mcp__get_tools',
                            children=[
                                BasicSpan(content='RunActivity:agent__complex_agent__mcp_server__mcp__get_tools')
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
                                                    content='{"index":0,"part":{"tool_name":"get_weather","args":"","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","id":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":null,"event_kind":"part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"city","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Mexico","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" City","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"}","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"part":{"tool_name":"get_weather","args":"{\\"city\\":\\"Mexico City\\"}","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","id":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":null,"event_kind":"part_end"}'
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
                                            content='{"part":{"tool_name":"get_weather","args":"{\\"city\\":\\"Mexico City\\"}","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","id":null,"provider_details":null,"part_kind":"tool-call"},"event_kind":"function_tool_call"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='running 1 tool',
                            children=[
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
                                                    content=IsStr(
                                                        regex=r'{"result":{"tool_name":"get_weather","content":"sunny","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"content":null,"event_kind":"function_tool_result"}'
                                                    )
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__complex_agent__mcp_server__mcp__get_tools',
                            children=[
                                BasicSpan(content='RunActivity:agent__complex_agent__mcp_server__mcp__get_tools')
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
                                                    content='{"index":0,"part":{"tool_name":"final_result","args":"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","id":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":null,"event_kind":"part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"tool_name":"final_result","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","event_kind":"final_result"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answers","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":[","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Capital","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" of","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" the","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" country","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Mexico","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" City","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"},{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Weather","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" in","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" the","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" capital","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Sunny","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"},{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Product","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" Name","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"P","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"yd","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"antic","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" AI","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"}","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"]}","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index":0,"part":{"tool_name":"final_result","args":"{\\"answers\\":[{\\"label\\":\\"Capital of the country\\",\\"answer\\":\\"Mexico City\\"},{\\"label\\":\\"Weather in the capital\\",\\"answer\\":\\"Sunny\\"},{\\"label\\":\\"Product Name\\",\\"answer\\":\\"Pydantic AI\\"}]}","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","id":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":null,"event_kind":"part_end"}'
                                                ),
                                            ],
                                        )
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
                part=ToolCallPart(tool_name='get_country', args='{}', tool_call_id='call_q2UyBRP7eXNTzAoR8lEhjc9Z')
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(tool_name='get_product_name', args='{}', tool_call_id='call_b51ijcpFkDiTQG1bQzsrmtW5')
            ),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='get_country',
                    content='Mexico',
                    tool_call_id='call_q2UyBRP7eXNTzAoR8lEhjc9Z',
                    timestamp=IsDatetime(),
                )
            ),
            FunctionToolResultEvent(
                result=ToolReturnPart(
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
                )
            ),
            FunctionToolResultEvent(
                result=ToolReturnPart(
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
        match="An agent needs to have a unique `name` in order to be used with Temporal. The name will be used to identify the agent's activities within the workflow.",
    ):
        TemporalAgent(Agent())


async def test_agent_without_model():
    with pytest.raises(
        UserError,
        match="The wrapped agent's `model` or the TemporalAgent's `models` parameter must provide at least one Model instance to be used with Temporal. Models cannot be set at agent run time.",
    ):
        TemporalAgent(Agent(name='test_agent'))


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

    @toolset.tool
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


# --- MCP-based DynamicToolset test ---
# Tests that @agent.toolset with an MCP toolset works with Temporal workflows.
# Uses MCPServerStreamableHTTP (HTTP-based) rather than subprocess-based MCP servers.
# See https://github.com/pydantic/pydantic-ai/issues/2818 for MCPServer subprocess issues with Temporal.


mcp_dynamic_toolset_agent = Agent(model, name='mcp_dynamic_toolset_agent')


@mcp_dynamic_toolset_agent.toolset(id='mcp_toolset')
def my_mcp_dynamic_toolset(ctx: RunContext[None]) -> MCPServerStreamableHTTP:
    """Dynamic toolset that returns an MCP toolset.

    This tests MCP lifecycle management (context manager enter/exit) within DynamicToolset + Temporal.
    """
    return MCPServerStreamableHTTP('https://mcp.deepwiki.com/mcp')


mcp_dynamic_toolset_temporal_agent = TemporalAgent(
    mcp_dynamic_toolset_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class MCPDynamicToolsetAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await mcp_dynamic_toolset_temporal_agent.run(prompt)
        return result.output


async def test_mcp_dynamic_toolset_in_workflow(allow_model_requests: None, client: Client):
    """Test that @agent.toolset with MCPServerStreamableHTTP works in a Temporal workflow.

    This demonstrates MCP lifecycle management (entering/exiting the MCP toolset context manager)
    within a DynamicToolset wrapped by TemporalDynamicToolset.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MCPDynamicToolsetAgentWorkflow],
        plugins=[AgentPlugin(mcp_dynamic_toolset_temporal_agent)],
    ):
        output = await client.execute_workflow(
            MCPDynamicToolsetAgentWorkflow.run,
            args=['Can you tell me about the pydantic/pydantic-ai repo? Keep it short.'],
            id='test_mcp_dynamic_toolset_workflow',
            task_queue=TASK_QUEUE,
        )
        # The deepwiki MCP server should return info about the pydantic-ai repo
        assert 'pydantic' in output.lower() or 'agent' in output.lower()


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
    assert isinstance(toolsets[3], TemporalMCPServer)
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
            'agent__complex_agent__toolset__<agent>__call_tool',
            'agent__complex_agent__toolset__country__call_tool',
            'agent__complex_agent__mcp_server__mcp__get_tools',
            'agent__complex_agent__mcp_server__mcp__call_tool',
        ]
    )


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
    events = [event async for event in simple_temporal_agent.run_stream_events('What is the capital of Mexico?')]
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
    model, name='agent_with_sync_history_processor', history_processors=[drop_first_message]
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
        return [event async for event in simple_temporal_agent.run_stream_events(prompt)]


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
    ctx: RunContext[None],
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


async def test_temporal_agent_run_in_workflow_with_toolsets(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithRunToolsets],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Toolsets cannot be set at agent run time inside a Temporal workflow, it must be set at agent creation time.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithRunToolsets.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithRunToolsets.__name__,
                task_queue=TASK_QUEUE,
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

    interceptor = new_client.config()['interceptors'][0]
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
async def create_file(ctx: RunContext[None], path: str) -> None:
    raise CallDeferred


@hitl_agent.tool
async def delete_file(ctx: RunContext[None], path: str) -> bool:
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
            status = await workflow.query(HitlAgentWorkflow.get_status)  # pyright: ignore[reportUnknownMemberType]
            if status == 'done':
                break
            elif status == 'waiting_for_results':  # pragma: no branch
                deferred_tool_requests = await workflow.query(HitlAgentWorkflow.get_deferred_tool_requests)  # pyright: ignore[reportUnknownMemberType]
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
    builtin_tools=[WebSearchTool(user_location=WebSearchUserLocation(city='Mexico City', country='MX'))],
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


@pytest.mark.filterwarnings(  # TODO (v2): Remove this once we drop the deprecated events
    'ignore:`BuiltinToolCallEvent` is deprecated', 'ignore:`BuiltinToolResultEvent` is deprecated'
)
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


fastmcp_agent = Agent(
    model,
    name='fastmcp_agent',
    toolsets=[FastMCPToolset('https://mcp.deepwiki.com/mcp', id='deepwiki')],
)

# This needs to be done before the `TemporalAgent` is bound to the workflow.
fastmcp_temporal_agent = TemporalAgent(
    fastmcp_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class FastMCPAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await fastmcp_temporal_agent.run(prompt)
        return result.output


async def test_fastmcp_toolset(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[FastMCPAgentWorkflow],
        plugins=[AgentPlugin(fastmcp_temporal_agent)],
    ):
        output = await client.execute_workflow(
            FastMCPAgentWorkflow.run,
            args=['Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short'],
            id=FastMCPAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            'The `pydantic/pydantic-ai` repository is a Python agent framework crafted for developing production-grade Generative AI applications. It emphasizes type safety, model-agnostic design, and extensibility. The framework supports various LLM providers, manages agent workflows using graph-based execution, and ensures structured, reliable LLM outputs. Key packages include core framework components, graph execution engines, evaluation tools, and example applications.'
        )


# ============================================================================
# Beta Graph API Tests - Tests for running pydantic-graph beta API in Temporal
# ============================================================================


@dataclass
class GraphState:
    """State for the graph execution test."""

    values: list[int] = field(default_factory=list)


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
            builtin_tools=[],
            output_mode='text',
            allow_text_output=True,
            output_tools=[],
            output_object=None,
        ),
    )

    # Verify response comes from the wrapped TestModel
    assert any(isinstance(part, TextPart) and part.content == 'Direct model response' for part in response.parts)


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
            builtin_tools=[],
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
