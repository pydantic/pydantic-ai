from __future__ import annotations

import os
import re
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from datetime import timedelta

from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    HandleResponseEvent,
    PartDeltaEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.toolsets import FunctionToolset

try:
    from temporalio import workflow
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]
    from temporalio.client import Client, WorkflowFailureError
    from temporalio.exceptions import ApplicationError
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.ext.temporal import (
        AgentPlugin,
        LogfirePlugin,
        PydanticAIPlugin,
        TemporalAgent,
        TemporalRunContextWithDeps,
    )
    from pydantic_ai.ext.temporal._function_toolset import TemporalFunctionToolset
    from pydantic_ai.ext.temporal._mcp_server import TemporalMCPServer
    from pydantic_ai.ext.temporal._model import TemporalModel
except ImportError:
    import pytest

    pytest.skip('temporal not installed', allow_module_level=True)

try:
    import logfire
    from logfire.testing import CaptureLogfire
except ImportError:
    import pytest

    pytest.skip('logfire not installed', allow_module_level=True)

try:
    from pydantic_ai.mcp import MCPServerStdio
except ImportError:
    import pytest

    pytest.skip('mcp not installed', allow_module_level=True)

try:
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:
    import pytest

    pytest.skip('openai not installed', allow_module_level=True)


with workflow.unsafe.imports_passed_through():
    # Workaround for a race condition when running `logfire.info` inside an activity with attributes to serialize and pandas importable:
    # AttributeError: partially initialized module 'pandas' has no attribute '_pandas_parser_CAPI' (most likely due to a circular import)
    import pandas  # pyright: ignore[reportUnusedImport] # noqa: F401

    # https://github.com/temporalio/sdk-python/blob/3244f8bffebee05e0e7efefb1240a75039903dda/tests/test_client.py#L112C1-L113C1
    import pytest

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


TEMPORAL_PORT = 7243
TASK_QUEUE = 'pydantic-ai-agent-task-queue'


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
simple_temporal_agent = TemporalAgent(simple_agent)


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


class Deps(TypedDict):
    country: str


async def event_stream_handler(
    ctx: RunContext[Deps],
    stream: AsyncIterable[AgentStreamEvent | HandleResponseEvent],
):
    logfire.info(f'{ctx.run_step=}')
    async for event in stream:
        logfire.info(f'{event=}')


async def get_country(ctx: RunContext[Deps]) -> str:
    return ctx.deps['country']


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
    ],
    tools=[get_weather],
    event_stream_handler=event_stream_handler,
    name='complex_agent',
)

# This needs to be done before the `TemporalAgent` is bound to the workflow.
complex_temporal_agent = TemporalAgent(
    complex_agent,
    activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=60)),
    model_activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=90)),
    toolset_activity_config={
        'country': ActivityConfig(start_to_close_timeout=timedelta(seconds=120)),
    },
    tool_activity_config={
        'country': {
            'get_country': False,
        },
    },
    run_context_type=TemporalRunContextWithDeps,
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

    result = await complex_temporal_agent.run(
        'Tell me: the capital of the country; the weather there; the product name',
        deps=Deps(country='Mexico'),
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
    assert len(toolsets) == 4

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


def test_temporal_agent_run_sync(allow_model_requests: None):
    result = simple_temporal_agent.run_sync('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is Mexico City.')


async def test_temporal_agent_run(allow_model_requests: None):
    result = await simple_temporal_agent.run('What is the capital of Mexico?')
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
        return result.output


async def test_temporal_agent_run_in_workflow_with_event_stream_handler(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithEventStreamHandler],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
                SimpleAgentWorkflowWithEventStreamHandler.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithEventStreamHandler.__name__,
                task_queue=TASK_QUEUE,
            )
        assert isinstance(exc_info.value.__cause__, ApplicationError)
        assert exc_info.value.__cause__.type == UserError.__name__
        assert exc_info.value.__cause__.message == snapshot(
            'Event stream handler cannot be set at agent run time when using Temporal, it must be set at agent creation time.'
        )


@workflow.defn
class SimpleAgentWorkflowWithIterModel:
    @workflow.run
    async def run(self, prompt: str) -> str:
        async with simple_temporal_agent.iter(prompt, model=model) as run:
            assert run.result is not None
            return run.result.output


async def test_temporal_agent_iter_in_workflow_with_model(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithIterModel],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
                SimpleAgentWorkflowWithIterModel.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithIterModel.__name__,
                task_queue=TASK_QUEUE,
            )
        assert isinstance(exc_info.value.__cause__, ApplicationError)
        assert exc_info.value.__cause__.type == UserError.__name__
        assert exc_info.value.__cause__.message == snapshot(
            'Model cannot be set at agent run time when using Temporal, it must be set at agent creation time.'
        )


@workflow.defn
class SimpleAgentWorkflowWithIterToolsets:
    @workflow.run
    async def run(self, prompt: str) -> str:
        async with simple_temporal_agent.iter(prompt, toolsets=[FunctionToolset()]) as run:
            assert run.result is not None
            return run.result.output


async def test_temporal_agent_iter_in_workflow_with_toolsets(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithIterToolsets],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
                SimpleAgentWorkflowWithIterToolsets.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithIterToolsets.__name__,
                task_queue=TASK_QUEUE,
            )
        assert isinstance(exc_info.value.__cause__, ApplicationError)
        assert exc_info.value.__cause__.type == UserError.__name__
        assert exc_info.value.__cause__.message == snapshot(
            'Toolsets cannot be set at agent run time when using Temporal, it must be set at agent creation time.'
        )


@workflow.defn
class SimpleAgentWorkflowWithOverrideModel:
    @workflow.run
    async def run(self, prompt: str) -> str:
        with simple_temporal_agent.override(model=model):
            result = await simple_temporal_agent.run(prompt)
            return result.output


async def test_temporal_agent_override_model_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithOverrideModel],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
                SimpleAgentWorkflowWithOverrideModel.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithOverrideModel.__name__,
                task_queue=TASK_QUEUE,
            )
        assert isinstance(exc_info.value.__cause__, ApplicationError)
        assert exc_info.value.__cause__.type == UserError.__name__
        assert exc_info.value.__cause__.message == snapshot(
            'Model cannot be contextually overridden when using Temporal, it must be set at agent creation time.'
        )


@workflow.defn
class SimpleAgentWorkflowWithOverrideToolsets:
    @workflow.run
    async def run(self, prompt: str) -> str:
        with simple_temporal_agent.override(toolsets=[FunctionToolset()]):
            result = await simple_temporal_agent.run(prompt)
            return result.output


async def test_temporal_agent_override_toolsets_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithOverrideToolsets],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
                SimpleAgentWorkflowWithOverrideToolsets.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithOverrideToolsets.__name__,
                task_queue=TASK_QUEUE,
            )
        assert isinstance(exc_info.value.__cause__, ApplicationError)
        assert exc_info.value.__cause__.type == UserError.__name__
        assert exc_info.value.__cause__.message == snapshot(
            'Toolsets cannot be contextually overridden when using Temporal, they must be set at agent creation time.'
        )


@workflow.defn
class SimpleAgentWorkflowWithOverrideTools:
    @workflow.run
    async def run(self, prompt: str) -> str:
        with simple_temporal_agent.override(tools=[get_weather]):
            result = await simple_temporal_agent.run(prompt)
            return result.output


async def test_temporal_agent_override_tools_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithOverrideTools],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
                SimpleAgentWorkflowWithOverrideTools.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithOverrideTools.__name__,
                task_queue=TASK_QUEUE,
            )
        assert isinstance(exc_info.value.__cause__, ApplicationError)
        assert exc_info.value.__cause__.type == UserError.__name__
        assert exc_info.value.__cause__.message == snapshot(
            'Tools cannot be contextually overridden when using Temporal, they must be set at agent creation time.'
        )


# TODO: f'Temporal activity config for tool {name!r} has been explicitly set to `False` (activity disabled), '
# 'but non-async tools are run in threads which are not supported outside of an activity. Make the tool function async instead.'

# TODO: 'The `deps` object must be a JSON-serializable dictionary in order to be used with Temporal. '
# 'To use a different type, pass a `TemporalRunContext` subclass to `TemporalAgent` with custom `serialize_run_context` and `deserialize_run_context` class methods.'

# TODO: Custom run_context_type

# TODO: tool_activity_config

# TODO: f'Temporal activity config for MCP tool {tool_name!r} has been explicitly set to `False` (activity disabled), '
# 'but MCP tools require the use of IO and so cannot be run outside of an activity.'

# TODO: Custom temporalize_toolset_func

# TODO: raise UserError('Streaming with Temporal requires `Agent` to have an `event_stream_handler` set.')

# TODO: raise UserError('Streaming with Temporal requires `request_stream` to be called with a `run_context`')

# TODO: LogfirePlugin(setup_logfire=<custom>)
# TODO: LogfirePlugin(metrics=False)
