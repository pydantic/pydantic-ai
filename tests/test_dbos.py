from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import pytest
from httpx import AsyncClient
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    RetryPromptPart,
    RunContext,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RequestUsage

from .conftest import IsDatetime, IsNow, IsStr

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSAgent, DBOSMCPServer, DBOSModel

except ImportError:  # pragma: lax no cover
    pytest.skip('DBOS is not installed', allow_module_level=True)

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
    from pydantic_ai.toolsets.fastmcp import FastMCPToolset
except ImportError:  # pragma: lax no cover
    pytest.skip('fastmcp not installed', allow_module_level=True)

try:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:  # pragma: lax no cover
    pytest.skip('openai not installed', allow_module_level=True)

from inline_snapshot import snapshot

from pydantic_ai import ExternalToolset, FunctionToolset
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.xdist_group(name='dbos'),
]

# We need to use a custom cached HTTP client here as the default one created for OpenAIProvider will be closed automatically
# at the end of each test, but we need this one to live longer.
http_client = cached_async_http_client(provider='dbos')


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
def workflow_raises(exc_type: type[Exception], exc_message: str) -> Iterator[None]:
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
        MCPServerStdio('python', ['-m', 'tests.mcp_server'], timeout=20, id='mcp'),
        ExternalToolset(tool_defs=[ToolDefinition(name='external')], id='external'),
    ],
    tools=[get_weather],
    event_stream_handler=event_stream_handler,
    instrument=True,  # Enable instrumentation for testing
    name='complex_agent',
)
complex_dbos_agent = DBOSAgent(complex_agent)
seq_complex_dbos_agent = DBOSAgent(complex_agent, parallel_execution_mode='sequential', name='seq_complex_agent')


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
            'complex_agent__mcp_server__mcp.get_tools',
            'complex_agent__model.request_stream',
            'event_stream_handler',
            'get_weather',
            'event_stream_handler',
            'complex_agent__mcp_server__mcp.get_tools',
            'complex_agent__model.request_stream',
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

    # Assert the root span and its structure matches expected hierarchy
    assert root_span == snapshot(
        BasicSpan(
            content='complex_agent.run',
            children=[
                BasicSpan(
                    content='complex_agent run',
                    children=[
                        BasicSpan(content='complex_agent__mcp_server__mcp.get_tools'),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='complex_agent__model.request_stream',
                                    children=[
                                        BasicSpan(content='ctx.run_step=1'),
                                        BasicSpan(
                                            content='{"index":0,"part":{"tool_name":"get_country","args":"","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":null,"event_kind":"part_start"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{}","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"part":{"tool_name":"get_country","args":"{}","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":"tool-call","event_kind":"part_end"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":1,"part":{"tool_name":"get_product_name","args":"","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":"tool-call","event_kind":"part_start"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":1,"delta":{"tool_name_delta":null,"args_delta":"{}","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":1,"part":{"tool_name":"get_product_name","args":"{}","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":null,"event_kind":"part_end"}'
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
                                    content='{"part":{"tool_name":"get_country","args":"{}","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"event_kind":"function_tool_call"}'
                                ),
                            ],
                        ),
                        BasicSpan(
                            content='event_stream_handler',
                            children=[
                                BasicSpan(content='ctx.run_step=1'),
                                BasicSpan(
                                    content='{"part":{"tool_name":"get_product_name","args":"{}","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"event_kind":"function_tool_call"}'
                                ),
                            ],
                        ),
                        BasicSpan(
                            content='running 2 tools',
                            children=[
                                BasicSpan(content='running tool: get_country'),
                                BasicSpan(
                                    content='running tool: get_product_name',
                                    children=[BasicSpan(content='complex_agent__mcp_server__mcp.call_tool')],
                                ),
                                BasicSpan(
                                    content='event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=1'),
                                        BasicSpan(
                                            content=IsStr(
                                                regex=r'{"result":{"tool_name":"get_country","content":"Mexico","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"content":null,"event_kind":"function_tool_result"}'
                                            )
                                        ),
                                    ],
                                ),
                                BasicSpan(
                                    content='event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=1'),
                                        BasicSpan(
                                            content=IsStr(
                                                regex=r'{"result":{"tool_name":"get_product_name","content":"Pydantic AI","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"content":null,"event_kind":"function_tool_result"}'
                                            )
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        BasicSpan(content='complex_agent__mcp_server__mcp.get_tools'),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='complex_agent__model.request_stream',
                                    children=[
                                        BasicSpan(content='ctx.run_step=2'),
                                        BasicSpan(
                                            content='{"index":0,"part":{"tool_name":"get_weather","args":"","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":null,"event_kind":"part_start"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"city","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Mexico","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" City","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"}","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"part":{"tool_name":"get_weather","args":"{\\"city\\":\\"Mexico City\\"}","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":null,"event_kind":"part_end"}'
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
                                    content='{"part":{"tool_name":"get_weather","args":"{\\"city\\":\\"Mexico City\\"}","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"event_kind":"function_tool_call"}'
                                ),
                            ],
                        ),
                        BasicSpan(
                            content='running 1 tool',
                            children=[
                                BasicSpan(
                                    content='running tool: get_weather', children=[BasicSpan(content='get_weather')]
                                ),
                                BasicSpan(
                                    content='event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=2'),
                                        BasicSpan(
                                            content=IsStr(
                                                regex=r'{"result":{"tool_name":"get_weather","content":"sunny","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"content":null,"event_kind":"function_tool_result"}'
                                            )
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        BasicSpan(content='complex_agent__mcp_server__mcp.get_tools'),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='complex_agent__model.request_stream',
                                    children=[
                                        BasicSpan(content='ctx.run_step=3'),
                                        BasicSpan(
                                            content='{"index":0,"part":{"tool_name":"final_result","args":"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"previous_part_kind":null,"event_kind":"part_start"}'
                                        ),
                                        BasicSpan(
                                            content='{"tool_name":"final_result","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","event_kind":"final_result"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answers","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":[","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Capital","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" of","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" the","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" country","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Mexico","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" City","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"},{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Weather","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" in","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" the","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" capital","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Sunny","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"},{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Product","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" Name","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"P","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"yd","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"antic","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" AI","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"}","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"]}","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","provider_name":null,"provider_details":null,"part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                                        ),
                                        BasicSpan(
                                            content='{"index":0,"part":{"tool_name":"final_result","args":"{\\"answers\\":[{\\"label\\":\\"Capital of the country\\",\\"answer\\":\\"Mexico City\\"},{\\"label\\":\\"Weather in the capital\\",\\"answer\\":\\"Sunny\\"},{\\"label\\":\\"Product Name\\",\\"answer\\":\\"Pydantic AI\\"}]}","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},"next_part_kind":null,"event_kind":"part_end"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                    ],
                )
            ],
        )
    )


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
            'seq_complex_agent__mcp_server__mcp.get_tools',
            'seq_complex_agent__model.request_stream',
            'event_stream_handler',
            'get_weather',
            'event_stream_handler',
            'seq_complex_agent__mcp_server__mcp.get_tools',
            'seq_complex_agent__model.request_stream',
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
        match="An agent needs to have a unique `name` in order to be used with DBOS. The name will be used to identify the agent's workflows and steps.",
    ):
        DBOSAgent(Agent())


async def test_agent_without_model():
    with pytest.raises(
        UserError,
        match='An agent needs to have a `model` in order to be used with DBOS, it cannot be set at agent run time.',
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
    assert isinstance(toolsets[3], DBOSMCPServer)
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
        async for _ in simple_dbos_agent.run_stream_events('What is the capital of Mexico?'):
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
        return [event async for event in simple_dbos_agent.run_stream_events('What is the capital of Mexico?')]

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
        ctx: RunContext[None],
        stream: AsyncIterable[AgentStreamEvent],
    ):
        pass

    with pytest.raises(Exception) as exc_info:
        await simple_dbos_agent.run('What is the capital of Mexico?', event_stream_handler=simple_event_stream_handler)

    assert (
        "local object 'test_dbos_agent_run_in_workflow_with_event_stream_handler.<locals>.simple_event_stream_handler'"
        in str(exc_info.value)
    )


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


@dynamic_agent.toolset  # type: ignore
def toggleable_toolset(ctx: RunContext[ToggleableDeps]) -> FunctionToolset[None]:
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
def create_file(ctx: RunContext[None], path: str) -> None:
    raise CallDeferred


@hitl_agent.tool
@DBOS.step()
def delete_file(ctx: RunContext[None], path: str) -> bool:
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


fastmcp_agent = Agent(
    model,
    name='fastmcp_agent',
    toolsets=[FastMCPToolset('https://mcp.deepwiki.com/mcp', id='deepwiki')],
)

fastmcp_dbos_agent = DBOSAgent(fastmcp_agent)


async def test_fastmcp_toolset(allow_model_requests: None, dbos: DBOS):
    wfid = str(uuid.uuid4())
    with SetWorkflowID(wfid):
        result = await fastmcp_dbos_agent.run(
            'Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short'
        )
    assert result.output == snapshot(
        'The `pydantic/pydantic-ai` repository is a Python agent framework designed for building production-grade Generative AI applications. It emphasizes type-safety, a model-agnostic design, and an ergonomic developer experience. Key features include type-safe agents using Pydantic, support for over 15 LLM providers, structured outputs with automatic validation, comprehensive observability, and production-ready tooling. The framework is structured as a UV workspace monorepo with core and supporting packages for defining and executing complex applications.'
    )

    steps = await dbos.list_workflow_steps_async(wfid)
    assert [step['function_name'] for step in steps] == snapshot(
        [
            'fastmcp_agent__mcp_server__deepwiki.get_tools',
            'fastmcp_agent__model.request',
            'fastmcp_agent__mcp_server__deepwiki.call_tool',
            'fastmcp_agent__mcp_server__deepwiki.get_tools',
            'fastmcp_agent__model.request',
        ]
    )


def test_dbos_mcp_wrapper_visit_and_replace():
    """DBOS MCP wrapper toolsets should not be replaced by visit_and_replace."""
    from pydantic_ai.durable_exec.dbos._fastmcp_toolset import DBOSFastMCPToolset

    toolsets = fastmcp_dbos_agent._toolsets  # pyright: ignore[reportPrivateUsage]
    dbos_mcp_toolsets = [ts for ts in toolsets if isinstance(ts, DBOSFastMCPToolset)]
    assert len(dbos_mcp_toolsets) >= 1

    dbos_mcp_toolset = dbos_mcp_toolsets[0]

    # visit_and_replace should return self for DBOS wrappers
    result = dbos_mcp_toolset.visit_and_replace(lambda t: FunctionToolset(id='replaced'))
    assert result is dbos_mcp_toolset
