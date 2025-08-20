from __future__ import annotations

import os
from collections.abc import AsyncIterable, AsyncIterator, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
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

from .conftest import IsDatetime

try:
    from dbos import DBOS, DBOSConfig

    from pydantic_ai.durable_exec.dbos import DBOSAgent
except ImportError:  # pragma: lax no cover
    import pytest

    pytest.skip('DBOS is not installed', allow_module_level=True)

try:
    import logfire

    # from logfire import Logfire
    # from logfire._internal.tracer import _ProxyTracer
    from logfire.testing import CaptureLogfire
    # from opentelemetry.trace import ProxyTracer
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

import pytest
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


# `LogfirePlugin` calls `logfire.instrument_pydantic_ai()`, so we need to make sure this doesn't bleed into other tests.
@pytest.fixture(autouse=True, scope='module')
def uninstrument_pydantic_ai() -> Iterator[None]:
    try:
        yield
    finally:
        Agent.instrument_all(False)


@contextmanager
def workflow_raises(exc_type: type[Exception], exc_message: str) -> Iterator[None]:
    """Helper for asserting that a DBOS workflow fails with the expected error."""
    with pytest.raises(Exception) as exc_info:
        yield
    assert isinstance(exc_info.value.__cause__, Exception)
    # TODO (Qian): check if this is the right way to check the cause
    assert exc_info.value.__cause__.__class__.__name__ == exc_type.__name__
    assert exc_info.value.__cause__ == exc_message


DBOS_DATABASE_URL = os.environ.get('DBOS_DATABASE_URL', 'postgresql://postgres:dbos@localhost:5432/postgres')
DBOS_CONFIG: DBOSConfig = {
    'name': 'pydantic_dbos_tests',
    'database_url': DBOS_DATABASE_URL,
    'system_database_url': DBOS_DATABASE_URL,
    'run_admin_server': False,
}


@pytest.fixture()
def dbos() -> Generator[DBOS, Any, None]:
    dbos = DBOS(config=DBOS_CONFIG, conductor_key=os.environ.get('DBOS_CONDUCTOR_KEY'))
    DBOS.launch()
    yield dbos
    DBOS.destroy()


async def test_simple_agent_run_in_workflow(allow_model_requests: None, dbos: DBOS, openai_api_key: str) -> None:
    """Test that a simple agent can run in a DBOS workflow."""

    model = OpenAIModel(
        'gpt-4o',
        provider=OpenAIProvider(
            api_key=openai_api_key,
            http_client=http_client,
        ),
    )

    simple_agent = Agent(model, name='simple_agent')
    simple_dbos_agent = DBOSAgent(simple_agent)

    result = await simple_dbos_agent.run('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is Mexico City.')


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


model = OpenAIModel(
    'gpt-4o',
    provider=OpenAIProvider(
        api_key=os.getenv('OPENAI_API_KEY', 'mock-api-key'),
        http_client=http_client,
    ),
)

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
    """Test that a simple agent can run in a DBOS workflow."""

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

    # TODO (Qian): update test to check spans
    assert len(basic_spans_by_id) > 0, 'No spans were exported'
    # Debug: print all spans and their relationships
    # print("All spans:")
    # for span_id, basic_span in basic_spans_by_id.items():
    #     print(f"  {span_id} -> parent: {basic_span.parent_id}, content: {basic_span.content}")

    # root_span = None
    # for basic_span in basic_spans_by_id.values():
    #     if basic_span.parent_id is None:
    #         root_span = basic_span
    #     else:
    #         parent_id = basic_span.parent_id
    #         # Check if parent exists before accessing
    #         if parent_id in basic_spans_by_id:
    #             parent_span = basic_spans_by_id[parent_id]
    #             parent_span.children.append(basic_span)
    #         else:
    #             print(f"Warning: Parent span {parent_id} not found for span with content: {basic_span.content}")
    #             # Treat as root span if parent is missing
    #             if root_span is None:
    #                 root_span = basic_span
    # print(root_span)


# Note (Qian): since we wrap the agent run in a DBOS workflow, we cannot just use a DBOS agent without DBOS. This test shows we can use a complex agent with DBOS decorated tools.
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
