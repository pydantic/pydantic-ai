from __future__ import annotations

import os
from collections.abc import AsyncIterable, AsyncIterator, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pytest
from httpx import AsyncClient
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
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
from pydantic_ai.models import cached_async_http_client

from .conftest import IsDatetime, IsStr

try:
    from dbos import DBOS, DBOSConfig

    from pydantic_ai.durable_exec.dbos import DBOSAgent, DBOSMCPServer, DBOSModel
except ImportError:  # pragma: lax no cover
    pytest.skip('DBOS is not installed', allow_module_level=True)

# Check if Postgres is available
try:
    import psycopg

    DBOS_DATABASE_URL = os.environ.get('DBOS_DATABASE_URL', 'postgresql://postgres:dbos@localhost:5432/postgres')
    # Connect to Postgres to see if it's running
    conn = None
    try:
        conn = psycopg.connect(
            DBOS_DATABASE_URL,
        )
        cursor = conn.cursor()
        cursor.execute('SELECT 1;')
        cursor.close()
    except Exception as error:  # pragma: lax no cover
        pytest.skip(f'Postgres not available: {error}', allow_module_level=True)
    finally:
        if conn is not None:  # pragma: lax no cover
            conn.close()
except ImportError:  # pragma: lax no cover
    pytest.skip('psycopg is not installed', allow_module_level=True)

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
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:  # pragma: lax no cover
    pytest.skip('openai not installed', allow_module_level=True)

from inline_snapshot import snapshot

from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import DeferredToolset, FunctionToolset

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


# Our setup calls `logfire.instrument_pydantic_ai()`, so we need to make sure this doesn't bleed into other tests.
@pytest.fixture(autouse=True, scope='module')
def uninstrument_pydantic_ai() -> Iterator[None]:
    try:
        # Set up logfire for the tests.
        logfire.configure(metrics=False)
        logfire.instrument_pydantic_ai()
        yield
    finally:
        Agent.instrument_all(False)


@contextmanager
def workflow_raises(exc_type: type[Exception], exc_message: str) -> Iterator[None]:
    """Helper for asserting that a DBOS workflow fails with the expected error."""
    with pytest.raises(Exception) as exc_info:
        yield
    assert isinstance(exc_info.value, Exception)
    assert str(exc_info.value) == exc_message


DBOS_CONFIG: DBOSConfig = {
    'name': 'pydantic_dbos_tests',
    'database_url': DBOS_DATABASE_URL,
    'system_database_url': DBOS_DATABASE_URL,
    'run_admin_server': False,
}


@pytest.fixture()
def dbos() -> Generator[DBOS, Any, None]:
    dbos = DBOS(config=DBOS_CONFIG)
    DBOS.launch()
    try:
        yield dbos
    finally:
        DBOS.destroy()


model = OpenAIModel(
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


async def event_stream_handler(
    ctx: RunContext[Deps],
    stream: AsyncIterable[AgentStreamEvent | HandleResponseEvent],
):
    logfire.info(f'{ctx.run_step=}')
    async for event in stream:
        logfire.info('event', event=event)


# Do not run it as a step
async def get_country(ctx: RunContext[Deps]) -> str:
    return ctx.deps.country


@DBOS.step()
def get_weather(city: str) -> str:
    return 'sunny'


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
    children: list[BasicSpan] = field(default_factory=list)
    parent_id: int | None = field(repr=False, compare=False, default=None)


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
complex_dbos_agent = DBOSAgent(complex_agent)


async def test_complex_agent_run_in_workflow(allow_model_requests: None, dbos: DBOS, capfire: CaptureLogfire) -> None:
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
                BasicSpan(content='complex_agent__mcp_server__mcp.get_tools'),
                BasicSpan(
                    content='complex_agent__model.request_stream',
                    children=[
                        BasicSpan(content='ctx.run_step=1'),
                        BasicSpan(
                            content='{"index":0,"part":{"tool_name":"get_country","args":"","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","part_kind":"tool-call"},"event_kind":"part_start"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{}","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":1,"part":{"tool_name":"get_product_name","args":"","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","part_kind":"tool-call"},"event_kind":"part_start"}'
                        ),
                        BasicSpan(
                            content='{"index":1,"delta":{"tool_name_delta":null,"args_delta":"{}","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                    ],
                ),
                BasicSpan(content='ctx.run_step=1'),
                BasicSpan(content='ctx.run_step=1'),
                BasicSpan(
                    content='{"part":{"tool_name":"get_country","args":"{}","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","part_kind":"tool-call"},"event_kind":"function_tool_call"}'
                ),
                BasicSpan(
                    content='{"part":{"tool_name":"get_product_name","args":"{}","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","part_kind":"tool-call"},"event_kind":"function_tool_call"}'
                ),
                BasicSpan(
                    content=IsStr(
                        regex=r'{"result":{"tool_name":"get_country","content":"Mexico","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"event_kind":"function_tool_result"}'
                    )
                ),
                BasicSpan(content='complex_agent__mcp_server__mcp.call_tool'),
                BasicSpan(
                    content=IsStr(
                        regex=r'{"result":{"tool_name":"get_product_name","content":"Pydantic AI","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"event_kind":"function_tool_result"}'
                    )
                ),
                BasicSpan(content='complex_agent__mcp_server__mcp.get_tools'),
                BasicSpan(
                    content='complex_agent__model.request_stream',
                    children=[
                        BasicSpan(content='ctx.run_step=2'),
                        BasicSpan(
                            content='{"index":0,"part":{"tool_name":"get_weather","args":"","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_kind":"tool-call"},"event_kind":"part_start"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"city","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Mexico","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" City","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"}","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                    ],
                ),
                BasicSpan(content='ctx.run_step=2'),
                BasicSpan(content='ctx.run_step=2'),
                BasicSpan(
                    content='{"part":{"tool_name":"get_weather","args":"{\\"city\\":\\"Mexico City\\"}","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_kind":"tool-call"},"event_kind":"function_tool_call"}'
                ),
                BasicSpan(content='get_weather'),
                BasicSpan(
                    content=IsStr(
                        regex=r'{"result":{"tool_name":"get_weather","content":"sunny","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"event_kind":"function_tool_result"}'
                    )
                ),
                BasicSpan(content='complex_agent__mcp_server__mcp.get_tools'),
                BasicSpan(
                    content='complex_agent__model.request_stream',
                    children=[
                        BasicSpan(content='ctx.run_step=3'),
                        BasicSpan(
                            content='{"index":0,"part":{"tool_name":"final_result","args":"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_kind":"tool-call"},"event_kind":"part_start"}'
                        ),
                        BasicSpan(
                            content='{"tool_name":"final_result","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","event_kind":"final_result"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answers","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":[","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Capital","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" of","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" the","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" country","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Mexico","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" City","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"},{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Weather","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" in","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" the","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" capital","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Sunny","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"},{\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"label","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"Product","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" Name","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\",\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"answer","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\":\\"","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"P","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"yd","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"antic","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":" AI","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"\\"}","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                        BasicSpan(
                            content='{"index":0,"delta":{"tool_name_delta":null,"args_delta":"]}","tool_call_id":"call_4kc6691zCzjPnOuEtbEGUvz2","part_delta_kind":"tool_call"},"event_kind":"part_delta"}'
                        ),
                    ],
                ),
                BasicSpan(content='ctx.run_step=3'),
                BasicSpan(content='ctx.run_step=3'),
            ],
        )
    )


# Note: since we wrap the agent run in a DBOS workflow, we cannot just use a DBOS agent without DBOS. This test shows we can use a complex agent with DBOS decorated tools. Without DBOS workflows, those steps are just normal functions.
async def test_complex_agent_run(allow_model_requests: None) -> None:
    events: list[AgentStreamEvent | HandleResponseEvent] = []

    async def event_stream_handler(
        ctx: RunContext[Deps],
        stream: AsyncIterable[AgentStreamEvent | HandleResponseEvent],
    ):
        async for event in stream:
            events.append(event)

    with complex_agent.override(deps=Deps(country='Mexico')):
        result = await complex_agent.run(
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

    # Unwrapped 'deferred' toolset
    assert isinstance(toolsets[4], DeferredToolset)
    assert toolsets[4].id == 'deferred'
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
            '`agent.run_stream()` cannot currently be used inside a DBOS workflow. '
            'Set an `event_stream_handler` on the agent and use `agent.run()` instead. '
            'Please file an issue if this is not sufficient for your use case.'
        ),
    ):
        await run_stream_workflow()


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


async def simple_event_stream_handler(
    ctx: RunContext[None],
    stream: AsyncIterable[AgentStreamEvent | HandleResponseEvent],
):
    pass


async def test_dbos_agent_run_in_workflow_with_event_stream_handler(allow_model_requests: None, dbos: DBOS) -> None:
    # DBOS workflow input must be serializable, so we cannot use a function as a dependency.
    # Therefore, we cannot pass in an event stream handler as an argument.
    with workflow_raises(TypeError, snapshot('Serialized data item should not be a function')):
        await simple_dbos_agent.run('What is the capital of Mexico?', event_stream_handler=simple_event_stream_handler)


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


async def test_dbos_agent_sync_tool_activity_disabled():
    # Not a valid test for DBOS
    pass


async def test_dbos_agent_mcp_server_activity_disabled():
    # Not a valid test for DBOS
    pass


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
    return ctx.deps.client.max_redirects  # pragma: no cover


async def test_dbos_agent_with_unserializable_deps_type(allow_model_requests: None, dbos: DBOS):
    unserializable_deps_dbos_agent = DBOSAgent(unserializable_deps_agent)
    # Test this raises a serialization error because httpx.AsyncClient is not serializable.
    with pytest.raises(
        Exception,
        match='object proxy must define __reduce_ex__()',
    ):
        async with AsyncClient() as client:
            # This will trigger the client to be unserializable
            logfire.instrument_httpx(client, capture_all=True)
            await unserializable_deps_dbos_agent.run('What is the model name?', deps=UnserializableDeps(client=client))


async def test_logfire_plugin():
    # Not a valid test for DBOS, as we don't need the LogfirePlugin.
    pass
