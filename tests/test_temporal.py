from __future__ import annotations

import os
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from datetime import timedelta

from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import AgentStreamEvent, HandleResponseEvent
from pydantic_ai.toolsets import FunctionToolset

try:
    from temporalio import workflow
    from temporalio.client import Client
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
    from .conftest import IsStr

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
]

TEMPORAL_PORT = 7243


@pytest.fixture
async def env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_local(port=TEMPORAL_PORT) as env:  # pyright: ignore[reportUnknownMemberType]
        yield env


@pytest.fixture
async def client(env: WorkflowEnvironment) -> Client:
    return await Client.connect(
        f'localhost:{TEMPORAL_PORT}',
        plugins=[PydanticAIPlugin(), LogfirePlugin()],
    )


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


agent = Agent(
    # Can't use the `openai_api_key` fixture here because the workflow needs to be defined at the top level of the file.
    OpenAIModel('gpt-4o', provider=OpenAIProvider(api_key=os.getenv('OPENAI_API_KEY', 'mock-api-key'))),
    deps_type=Deps,
    output_type=Response,
    toolsets=[
        FunctionToolset[Deps](tools=[get_country], id='country'),
        MCPServerStdio('python', ['-m', 'tests.mcp_server'], timeout=20, id='mcp'),
    ],
    tools=[get_weather],
    event_stream_handler=event_stream_handler,
)

# This needs to be done before the `agent` is bound to the workflow.
temporal_agent = TemporalAgent(
    agent,
    activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=60)),
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
class AgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: Deps) -> Response:
        result = await temporal_agent.run(prompt, deps=deps)
        return result.output


async def test_temporal(allow_model_requests: None, client: Client, capfire: CaptureLogfire):
    task_queue = 'pydantic-ai-agent-task-queue'

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[AgentWorkflow],
        plugins=[AgentPlugin(temporal_agent)],
    ):
        output = await client.execute_workflow(  # pyright: ignore[reportUnknownMemberType]
            AgentWorkflow.run,
            args=[
                'Tell me: the capital of the country; the weather there; the product name',
                Deps(country='Mexico'),
            ],
            id='pydantic-ai-agent-workflow',
            task_queue=task_queue,
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

    parsed_spans: list[str | AgentStreamEvent | HandleResponseEvent] = []
    for span in exporter.exported_spans_as_dict():
        attributes = span['attributes']
        if event := attributes.get('event'):
            parsed_spans.append(event)
        else:
            parsed_spans.append(attributes['logfire.msg'])

    assert parsed_spans == snapshot(
        [
            'StartWorkflow:AgentWorkflow',
            'RunWorkflow:AgentWorkflow',
            'StartActivity:mcp_server__mcp__get_tools',
            'RunActivity:mcp_server__mcp__get_tools',
            'StartActivity:mcp_server__mcp__get_tools',
            'RunActivity:mcp_server__mcp__get_tools',
            'StartActivity:model__openai_gpt-4o__request_stream',
            'ctx.run_step=1',
            '{"index":0,"part":{"tool_name":"get_country","args":"","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","part_kind":"tool-call"},"event_kind":"part_start"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{}","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":1,"part":{"tool_name":"get_product_name","args":"","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","part_kind":"tool-call"},"event_kind":"part_start"}',
            '{"index":1,"delta":{"tool_name_delta":null,"args_delta":"{}","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            'RunActivity:model__openai_gpt-4o__request_stream',
            'ctx.run_step=1',
            'chat gpt-4o',
            'ctx.run_step=1',
            '{"part":{"tool_name":"get_country","args":"{}","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","part_kind":"tool-call"},"event_kind":"function_tool_call"}',
            '{"part":{"tool_name":"get_product_name","args":"{}","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","part_kind":"tool-call"},"event_kind":"function_tool_call"}',
            'running tool: get_country',
            'StartActivity:mcp_server__mcp__call_tool',
            IsStr(
                regex=r'{"result":{"tool_name":"get_country","content":"Mexico","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"event_kind":"function_tool_result"}'
            ),
            'RunActivity:mcp_server__mcp__call_tool',
            'running tool: get_product_name',
            IsStr(
                regex=r'{"result":{"tool_name":"get_product_name","content":"Pydantic AI","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"event_kind":"function_tool_result"}'
            ),
            'running 2 tools',
            'StartActivity:mcp_server__mcp__get_tools',
            'RunActivity:mcp_server__mcp__get_tools',
            'StartActivity:model__openai_gpt-4o__request_stream',
            'ctx.run_step=2',
            '{"index":0,"part":{"tool_name":"get_weather","args":"","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_kind":"tool-call"},"event_kind":"part_start"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"city","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Mexico","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":" City","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            '{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"}","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}',
            'RunActivity:model__openai_gpt-4o__request_stream',
            'ctx.run_step=2',
            'chat gpt-4o',
            'ctx.run_step=2',
            '{"part":{"tool_name":"get_weather","args":"{\\"city\\":\\"Mexico City\\"}","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_kind":"tool-call"},"event_kind":"function_tool_call"}',
            'StartActivity:function_toolset__<agent>__call_tool',
            'RunActivity:function_toolset__<agent>__call_tool',
            'running tool: get_weather',
            IsStr(
                regex=r'{"result":{"tool_name":"get_weather","content":"sunny","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"event_kind":"function_tool_result"}'
            ),
            'running 1 tool',
            'StartActivity:mcp_server__mcp__get_tools',
            'RunActivity:mcp_server__mcp__get_tools',
            'StartActivity:model__openai_gpt-4o__request_stream',
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
            'RunActivity:model__openai_gpt-4o__request_stream',
            'ctx.run_step=3',
            'chat gpt-4o',
            'ctx.run_step=3',
            'self run',
            'CompleteWorkflow:AgentWorkflow',
        ]
    )
