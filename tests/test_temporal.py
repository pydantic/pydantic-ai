from __future__ import annotations

import os
import re
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    HandleResponseEvent,
    ModelMessage,
    ModelRequest,
    PartDeltaEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from pydantic_ai.models import Model, cached_async_http_client
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import DeferredToolset, FunctionToolset

try:
    from temporalio import workflow
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]
    from temporalio.client import Client, WorkflowFailureError
    from temporalio.common import RetryPolicy
    from temporalio.contrib.opentelemetry import TracingInterceptor
    from temporalio.exceptions import ApplicationError
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        LogfirePlugin,
        PydanticAIPlugin,
        TemporalAgent,
    )
    from pydantic_ai.durable_exec.temporal._function_toolset import TemporalFunctionToolset
    from pydantic_ai.durable_exec.temporal._mcp_server import TemporalMCPServer
    from pydantic_ai.durable_exec.temporal._model import TemporalModel
except ImportError:  # pragma: lax no cover
    import pytest

    pytest.skip('temporal not installed', allow_module_level=True)

try:
    import logfire
    from logfire import Logfire
    from logfire._internal.tracer import _ProxyTracer  # pyright: ignore[reportPrivateUsage]
    from logfire.testing import CaptureLogfire
    from opentelemetry.trace import ProxyTracer
except ImportError:  # pragma: lax no cover
    import pytest

    pytest.skip('logfire not installed', allow_module_level=True)

try:
    from pydantic_ai.mcp import MCPServerStdio
except ImportError:  # pragma: lax no cover
    import pytest

    pytest.skip('mcp not installed', allow_module_level=True)

try:
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:  # pragma: lax no cover
    import pytest

    pytest.skip('openai not installed', allow_module_level=True)


with workflow.unsafe.imports_passed_through():
    # Workaround for a race condition when running `logfire.info` inside an activity with attributes to serialize and pandas importable:
    # AttributeError: partially initialized module 'pandas' has no attribute '_pandas_parser_CAPI' (most likely due to a circular import)
    import pandas  # pyright: ignore[reportUnusedImport] # noqa: F401

    # https://github.com/temporalio/sdk-python/blob/3244f8bffebee05e0e7efefb1240a75039903dda/tests/test_client.py#L112C1-L113C1
    import pytest
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
    async with await WorkflowEnvironment.start_local(port=TEMPORAL_PORT, ui=True) as env:  # pyright: ignore[reportUnknownMemberType]
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
model = OpenAIModel(
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
        output = await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
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
    stream: AsyncIterable[AgentStreamEvent | HandleResponseEvent],
):
    logfire.info(f'{ctx.run_step=}')
    async for event in stream:
        logfire.info('event', event=event)


async def get_country(ctx: RunContext[Deps]) -> str:
    return ctx.deps.country


def get_weather(city: str) -> str:
    return 'sunny'


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
        DeferredToolset(tool_defs=[ToolDefinition(name='deferred')], id='deferred'),
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


async def test_complex_agent_run_in_workflow(
    allow_model_requests: None, client_with_logfire: Client, capfire: CaptureLogfire
):
    async with Worker(
        client_with_logfire,
        task_queue=TASK_QUEUE,
        workflows=[ComplexAgentWorkflow],
        plugins=[AgentPlugin(complex_temporal_agent)],
    ):
        output = await client_with_logfire.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
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

    parsed_otel_items: list[str] = []
    for span in exporter.exported_spans_as_dict():
        attributes = span['attributes']
        if event := attributes.get('event'):
            parsed_otel_items.append(event)
        else:
            parsed_otel_items.append(attributes['logfire.msg'])

    assert parsed_otel_items == snapshot(
        [
            'StartWorkflow:ComplexAgentWorkflow',
            'RunWorkflow:ComplexAgentWorkflow',
            'StartActivity:agent__complex_agent__mcp_server__mcp__get_tools',
            'RunActivity:agent__complex_agent__mcp_server__mcp__get_tools',
            'StartActivity:agent__complex_agent__mcp_server__mcp__get_tools',
            'RunActivity:agent__complex_agent__mcp_server__mcp__get_tools',
            'StartActivity:agent__complex_agent__model_request_stream',
            'ctx.run_step=1',
            '{"index":0,"part":{"tool_name":"get_country","args":"","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","part_kind":"tool-call"},"event_kind":"part_start"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{}","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":1,"part":{"tool_name":"get_product_name","args":"","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","part_kind":"tool-call"},"event_kind":"part_start"}',
            '{"index":1,"delta":{"tool_name_delta":null,"args_delta":"{}","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            'RunActivity:agent__complex_agent__model_request_stream',
            'ctx.run_step=1',
            'chat gpt-4o',
            'ctx.run_step=1',
            '{"part":{"tool_name":"get_country","args":"{}","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","part_kind":"tool-call"},"event_kind":"function_tool_call"}',
            '{"part":{"tool_name":"get_product_name","args":"{}","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","part_kind":"tool-call"},"event_kind":"function_tool_call"}',
            'running tool: get_country',
            'StartActivity:agent__complex_agent__mcp_server__mcp__call_tool',
            IsStr(
                regex=r'{"result":{"tool_name":"get_country","content":"Mexico","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"event_kind":"function_tool_result"}'
            ),
            'RunActivity:agent__complex_agent__mcp_server__mcp__call_tool',
            'running tool: get_product_name',
            IsStr(
                regex=r'{"result":{"tool_name":"get_product_name","content":"Pydantic AI","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"event_kind":"function_tool_result"}'
            ),
            'running 2 tools',
            'StartActivity:agent__complex_agent__mcp_server__mcp__get_tools',
            'RunActivity:agent__complex_agent__mcp_server__mcp__get_tools',
            'StartActivity:agent__complex_agent__model_request_stream',
            'ctx.run_step=2',
            '{"index":0,"part":{"tool_name":"get_weather","args":"","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_kind":"tool-call"},"event_kind":"part_start"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"city","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Mexico","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":" City","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"}","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            'RunActivity:agent__complex_agent__model_request_stream',
            'ctx.run_step=2',
            'chat gpt-4o',
            'ctx.run_step=2',
            '{"part":{"tool_name":"get_weather","args":"{\\"city\\":\\"Mexico City\\"}","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_kind":"tool-call"},"event_kind":"function_tool_call"}',
            'StartActivity:agent__complex_agent__toolset__<agent>__call_tool',
            'RunActivity:agent__complex_agent__toolset__<agent>__call_tool',
            'running tool: get_weather',
            IsStr(
                regex=r'{"result":{"tool_name":"get_weather","content":"sunny","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"event_kind":"function_tool_result"}'
            ),
            'running 1 tool',
            'StartActivity:agent__complex_agent__mcp_server__mcp__get_tools',
            'RunActivity:agent__complex_agent__mcp_server__mcp__get_tools',
            'StartActivity:agent__complex_agent__model_request_stream',
            'ctx.run_step=3',
            '{"index":0,"part":{"tool_name":"final_result","args":"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_kind":"tool-call"},"event_kind":"part_start"}',
            '{"tool_name":"final_result","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","event_kind":"final_result"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answers","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":[","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Capital","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":" of","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":" the","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":" country","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Mexico","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":" City","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"},{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Weather","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":" in","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":" the","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":" capital","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Sunny","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"},{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Product","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":" Name","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"P","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"yd","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"antic","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":" AI","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"}","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"]}","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            'RunActivity:agent__complex_agent__model_request_stream',
            'ctx.run_step=3',
            'chat gpt-4o',
            'ctx.run_step=3',
            'complex_agent run',
            'CompleteWorkflow:ComplexAgentWorkflow',
        ]
    )


async def test_complex_agent_run(allow_model_requests: None):
    events: list[AgentStreamEvent | HandleResponseEvent] = []

    async def event_stream_handler(
        ctx: RunContext[Deps],
        stream: AsyncIterable[AgentStreamEvent | HandleResponseEvent],
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
            PartStartEvent(
                index=1,
                part=ToolCallPart(tool_name='get_product_name', args='', tool_call_id='call_b51ijcpFkDiTQG1bQzsrmtW5'),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='{}', tool_call_id='call_b51ijcpFkDiTQG1bQzsrmtW5')
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
        ]
    )


async def test_multiple_agents(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflow, ComplexAgentWorkflow],
        plugins=[AgentPlugin(simple_temporal_agent), AgentPlugin(complex_temporal_agent)],
    ):
        output = await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
            SimpleAgentWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=SimpleAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('The capital of Mexico is Mexico City.')

        output = await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
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
    with pytest.raises(ValueError, match='More than one activity named agent__simple_agent__model_request'):
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
        match='An agent needs to have a `model` in order to be used with Temporal, it cannot be set at agent run time.',
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

    # Unwrapped 'deferred' toolset
    assert isinstance(toolsets[4], DeferredToolset)
    assert toolsets[4].id == 'deferred'
    assert toolsets[4] == complex_agent.toolsets[3]

    assert [
        ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        for activity in complex_temporal_agent.temporal_activities
    ] == snapshot(
        [
            'agent__complex_agent__model_request',
            'agent__complex_agent__model_request_stream',
            'agent__complex_agent__toolset__<agent>__call_tool',
            'agent__complex_agent__toolset__country__call_tool',
            'agent__complex_agent__mcp_server__mcp__get_tools',
            'agent__complex_agent__mcp_server__mcp__call_tool',
        ]
    )


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
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
                SimpleAgentWorkflowWithRunSync.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithRunSync.__name__,
                task_queue=TASK_QUEUE,
            )


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
                '`agent.run_stream()` cannot currently be used inside a Temporal workflow. Set an `event_stream_handler` on the agent and use `agent.run()` instead. Please file an issue if this is not sufficient for your use case.'
            ),
        ):
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
                SimpleAgentWorkflowWithRunStream.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithRunStream.__name__,
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
                '`agent.iter()` cannot currently be used inside a Temporal workflow. Set an `event_stream_handler` on the agent and use `agent.run()` instead. Please file an issue if this is not sufficient for your use case.'
            ),
        ):
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
                SimpleAgentWorkflowWithIter.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithIter.__name__,
                task_queue=TASK_QUEUE,
            )


async def simple_event_stream_handler(
    ctx: RunContext[None],
    stream: AsyncIterable[AgentStreamEvent | HandleResponseEvent],
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
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
                SimpleAgentWorkflowWithEventStreamHandler.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithEventStreamHandler.__name__,
                task_queue=TASK_QUEUE,
            )


@workflow.defn
class SimpleAgentWorkflowWithRunModel:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_temporal_agent.run(prompt, model=model)
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
                'Model cannot be set at agent run time inside a Temporal workflow, it must be set at agent creation time.'
            ),
        ):
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
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
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
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
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
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
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
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
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
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
        output = await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
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
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
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
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
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
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
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
