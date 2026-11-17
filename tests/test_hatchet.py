from __future__ import annotations

import os
from collections.abc import AsyncIterable, AsyncIterator, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast

import pytest
from httpx import AsyncClient
from pydantic import BaseModel
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy
from testcontainers.postgres import PostgresContainer

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

from .conftest import IsDatetime, IsStr

try:
    from hatchet_sdk import ClientConfig, Hatchet

    from pydantic_ai.durable_exec.hatchet import HatchetAgent, HatchetMCPServer, HatchetModel
except ImportError:  # pragma: lax no cover
    pytest.skip('Hatchet is not installed', allow_module_level=True)

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
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:  # pragma: lax no cover
    pytest.skip('openai not installed', allow_module_level=True)

from inline_snapshot import snapshot

from pydantic_ai.tools import DeferredToolRequests, ToolDefinition
from pydantic_ai.toolsets import ExternalToolset, FunctionToolset

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.xdist_group(name='hatchet'),
]

# We need to use a custom cached HTTP client here as the default one created for OpenAIProvider will be closed automatically
# at the end of each test, but we need this one to live longer.
http_client = cached_async_http_client(provider='hatchet')


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
    """Helper for asserting that a Hatchet workflow fails with the expected error."""
    with pytest.raises(Exception) as exc_info:
        yield
    assert isinstance(exc_info.value, Exception)
    assert str(exc_info.value) == exc_message


@pytest.fixture(scope='module', autouse=True)
def hatchet_lite_container() -> Generator[DockerContainer, None, None]:
    """Start a Hatchet Lite container for testing."""
    with (
        PostgresContainer().with_bind_ports(5432) as pg,
        DockerContainer('ghcr.io/hatchet-dev/hatchet/hatchet-lite:latest')
        .with_bind_ports(8888, 8888)
        .with_bind_ports(7077, 7077)
        .with_env(
            key='DATABASE_URL',
            value=f'postgresql://{pg.username}:{pg.password}@host.docker.internal:{pg.get_exposed_port(5432)}/{pg.dbname}?sslmode=disable',
        )
        .with_env(key='SERVER_AUTH_COOKIE_DOMAIN', value='localhost')
        .with_env(key='SERVER_AUTH_COOKIE_INSECURE', value='t')
        .with_env(key='SERVER_GRPC_BIND_ADDRESS', value='0.0.0.0')
        .with_env(key='SERVER_GRPC_INSECURE', value='t')
        .with_env(key='SERVER_GRPC_BROADCAST_ADDRESS', value='localhost:7077')
        .with_env(key='SERVER_GRPC_PORT', value='7077')
        .with_env(key='SERVER_URL', value='http://localhost:8888')
        .with_env(key='SERVER_AUTH_SET_EMAIL_VERIFIED', value='t')
        .with_env(key='SERVER_DEFAULT_ENGINE_VERSION', value='V1')
        .with_env(key='SERVER_INTERNAL_CLIENT_INTERNAL_GRPC_BROADCAST_ADDRESS', value='localhost:7077')
        .waiting_for(
            LogMessageWaitStrategy('created tenant 707d0855-80ab-4e1f-a156-f1c4546cbf52').with_startup_timeout(20)
        ) as hatchet,
    ):
        yield hatchet


@pytest.fixture(scope='module')
def hatchet(hatchet_lite_container: DockerContainer) -> Generator[Hatchet, Any, None]:
    exec_result = hatchet_lite_container.exec(
        [
            '/hatchet-admin',
            'token',
            'create',
            '--config',
            '/config',
            '--tenant-id',
            '707d0855-80ab-4e1f-a156-f1c4546cbf52',
        ]
    )

    output = cast(bytes, exec_result.output).decode('utf-8')
    token = [line for line in output.split('\n') if line.strip()][-1]

    import time

    time.sleep(5)

    yield Hatchet(debug=True, config=ClientConfig(token=token))


model = OpenAIChatModel(
    'gpt-4o',
    provider=OpenAIProvider(
        api_key=os.getenv('OPENAI_API_KEY', 'mock-api-key'),
        http_client=http_client,
    ),
)

# Not necessarily need to define it outside of the function. Hatchet just requires workflows to be statically defined so recovery would be able to find those workflows. It's nice to reuse it in multiple tests.
simple_agent = Agent(model, name='simple_agent')


async def test_simple_agent_run_in_workflow(allow_model_requests: None, hatchet: Hatchet, openai_api_key: str) -> None:
    """Test that a simple agent can run in a Hatchet workflow."""
    simple_hatchet_agent = HatchetAgent(simple_agent, hatchet=hatchet)

    output = await simple_hatchet_agent.run('What is the capital of Mexico?')
    assert output.output == snapshot('The capital of Mexico is Mexico City.')


class Deps(BaseModel):
    country: str


# Wrap event_stream_handler as a Hatchet task because it's non-deterministic (uses logfire)
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
    children: list[BasicSpan] = field(default_factory=list)
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


async def test_complex_agent_run_in_workflow(
    allow_model_requests: None, hatchet: Hatchet, capfire: CaptureLogfire
) -> None:
    complex_hatchet_agent = HatchetAgent(complex_agent, hatchet=hatchet)

    # HatchetAgent already wraps the `run` function as a Hatchet workflow, so we can just call it directly.
    result = await complex_hatchet_agent.run(
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
            content='complex_agent run',
            children=[
                BasicSpan(
                    content='chat gpt-4o',
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
                BasicSpan(
                    content='{"part":{"tool_name":"get_country","args":"{}","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","part_kind":"tool-call"},"event_kind":"function_tool_call"}'
                ),
                BasicSpan(content='ctx.run_step=1'),
                BasicSpan(
                    content='{"part":{"tool_name":"get_product_name","args":"{}","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","part_kind":"tool-call"},"event_kind":"function_tool_call"}'
                ),
                BasicSpan(
                    content='running 2 tools',
                    children=[
                        BasicSpan(content='running tool: get_country'),
                        BasicSpan(content='ctx.run_step=1'),
                        BasicSpan(
                            content=IsStr(
                                regex=r'{"result":{"tool_name":"get_country","content":"Mexico","tool_call_id":"call_3rqTYrA6H21AYUaRGP4F66oq","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"event_kind":"function_tool_result"}'
                            )
                        ),
                        BasicSpan(content='running tool: get_product_name'),
                        BasicSpan(content='ctx.run_step=1'),
                        BasicSpan(
                            content=IsStr(
                                regex=r'{"result":{"tool_name":"get_product_name","content":"Pydantic AI","tool_call_id":"call_Xw9XMKBJU48kAAd78WgIswDx","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"event_kind":"function_tool_result"}'
                            )
                        ),
                    ],
                ),
                BasicSpan(
                    content='chat gpt-4o',
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
                BasicSpan(
                    content='{"part":{"tool_name":"get_weather","args":"{\\"city\\":\\"Mexico City\\"}","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","part_kind":"tool-call"},"event_kind":"function_tool_call"}'
                ),
                BasicSpan(
                    content='running 1 tool',
                    children=[
                        BasicSpan(content='running tool: get_weather'),
                        BasicSpan(content='ctx.run_step=2'),
                        BasicSpan(
                            content=IsStr(
                                regex=r'{"result":{"tool_name":"get_weather","content":"sunny","tool_call_id":"call_Vz0Sie91Ap56nH0ThKGrZXT7","metadata":null,"timestamp":".+?","part_kind":"tool-return"},"event_kind":"function_tool_result"}'
                            )
                        ),
                    ],
                ),
                BasicSpan(
                    content='chat gpt-4o',
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
            ],
        )
    )


# Note: since we wrap the agent run in a Hatchet workflow, we cannot just use a Hatchet agent without Hatchet. This test shows we can use a complex agent with Hatchet decorated tools. Without Hatchet workflows, those steps are just normal functions.
async def test_complex_agent_run(allow_model_requests: None) -> None:
    events: list[AgentStreamEvent] = []

    async def event_stream_handler_func(
        ctx: RunContext[Deps],
        stream: AsyncIterable[AgentStreamEvent],
    ):
        async for event in stream:
            events.append(event)

    with complex_agent.override(deps=Deps(country='Mexico')):
        result = await complex_agent.run(
            'Tell me: the capital of the country; the weather there; the product name',
            deps=Deps(country='The Netherlands'),
            event_stream_handler=event_stream_handler_func,
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


async def test_multiple_agents(allow_model_requests: None, hatchet: Hatchet):
    """Test that multiple agents can run in a Hatchet workflow."""
    simple_hatchet_agent = HatchetAgent(simple_agent, hatchet=hatchet)
    complex_hatchet_agent = HatchetAgent(complex_agent, hatchet=hatchet)

    # This is just a smoke test to ensure that multiple agents can run in a Hatchet workflow.
    # We don't need to check the output as it's already tested in the individual agent tests.
    result = await simple_hatchet_agent.run('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is Mexico City.')

    result = await complex_hatchet_agent.run(
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


async def test_agent_name_collision(allow_model_requests: None, hatchet: Hatchet):
    simple_hatchet_agent = HatchetAgent(simple_agent, hatchet=hatchet)
    with pytest.raises(Exception, match='duplicate workflow'):
        HatchetAgent(simple_agent, hatchet=hatchet)


async def test_agent_without_name(hatchet: Hatchet):
    with pytest.raises(
        UserError,
        match="An agent needs to have a unique `name` in order to be used with Hatchet. The name will be used to identify the agent's workflows and tasks.",
    ):
        HatchetAgent(Agent(), hatchet=hatchet)


async def test_agent_without_model(hatchet: Hatchet):
    with pytest.raises(
        UserError,
        match='An agent needs to have a `model` in order to be used with Hatchet, it cannot be set at agent run time.',
    ):
        HatchetAgent(Agent(name='test_agent'), hatchet=hatchet)


async def test_toolset_without_id(hatchet: Hatchet):
    # Note: this is allowed in Hatchet because we don't wrap the tools automatically in a workflow. It's up to the user to define the tools as Hatchet tasks if they want to use them as steps in a workflow.
    HatchetAgent(Agent(model=model, name='test_agent', toolsets=[FunctionToolset()]), hatchet=hatchet)


async def test_hatchet_agent(hatchet: Hatchet):
    complex_hatchet_agent = HatchetAgent(complex_agent, hatchet=hatchet)

    assert isinstance(complex_hatchet_agent.model, HatchetModel)
    assert complex_hatchet_agent.model.wrapped == complex_agent.model

    # Hatchet only wraps the MCP server toolsets. Other toolsets are not wrapped.
    toolsets = complex_hatchet_agent.toolsets
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
    assert isinstance(toolsets[3], HatchetMCPServer)
    assert toolsets[3].id == 'mcp'
    assert toolsets[3].wrapped == complex_agent.toolsets[2]

    # Unwrapped 'external' toolset
    assert isinstance(toolsets[4], ExternalToolset)
    assert toolsets[4].id == 'external'
    assert toolsets[4] == complex_agent.toolsets[3]


async def test_hatchet_agent_run(allow_model_requests: None, hatchet: Hatchet):
    simple_hatchet_agent = HatchetAgent(simple_agent, hatchet=hatchet)

    # Note: this runs as a Hatchet workflow because we automatically wrap the run function.
    result = await simple_hatchet_agent.run('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is Mexico City.')


def test_hatchet_agent_run_sync(allow_model_requests: None, hatchet: Hatchet):
    simple_hatchet_agent = HatchetAgent(simple_agent, hatchet=hatchet)

    # Note: this runs as a Hatchet workflow because we automatically wrap the run_sync function.
    # This is equivalent to test_hatchet_agent_run_sync_in_workflow
    result = simple_hatchet_agent.run_sync('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is Mexico City.')


async def test_hatchet_agent_run_stream(allow_model_requests: None, hatchet: Hatchet):
    simple_hatchet_agent = HatchetAgent(simple_agent, hatchet=hatchet)

    # Run stream is not a Hatchet workflow, so we can use it directly.
    async with simple_hatchet_agent.run_stream('What is the capital of Mexico?') as result:
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


async def test_hatchet_agent_iter(allow_model_requests: None, hatchet: Hatchet):
    simple_hatchet_agent = HatchetAgent(simple_agent, hatchet=hatchet)

    output: list[str] = []
    async with simple_hatchet_agent.iter('What is the capital of Mexico?') as run:
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


async def test_hatchet_agent_run_stream_in_workflow(allow_model_requests: None, hatchet: Hatchet):
    simple_hatchet_agent = HatchetAgent(simple_agent, hatchet=hatchet)

    # In Hatchet, run_stream cannot be used inside a task
    # This test would need to be adapted based on Hatchet's actual behavior
    # For now, we'll skip testing this specific scenario or adapt it
    pass  # TODO: Adapt this test for Hatchet's workflow semantics


async def test_hatchet_agent_override_model_in_workflow(allow_model_requests: None, hatchet: Hatchet):
    simple_hatchet_agent = HatchetAgent(simple_agent, hatchet=hatchet)

    # We cannot override the model to a non-Hatchet one in a Hatchet workflow.
    with workflow_raises(
        UserError,
        snapshot(
            'Non-Hatchet model cannot be contextually overridden inside a Hatchet workflow, it must be set at agent creation time.'
        ),
    ):
        with simple_hatchet_agent.override(model=model):
            pass


async def test_hatchet_model_stream_direct(allow_model_requests: None, hatchet: Hatchet):
    complex_hatchet_agent = HatchetAgent(complex_agent, hatchet=hatchet)

    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('What is the capital of Mexico?')]

    with workflow_raises(
        AssertionError,
        snapshot(
            'A Hatchet model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
        ),
    ):
        async with model_request_stream(complex_hatchet_agent.model, messages) as stream:
            async for _ in stream:
                pass


@dataclass
class UnserializableDeps:
    client: AsyncClient


async def test_hatchet_agent_with_unserializable_deps_type(allow_model_requests: None, hatchet: Hatchet):
    unserializable_deps_agent = Agent(model, name='unserializable_deps_agent', deps_type=UnserializableDeps)

    @unserializable_deps_agent.tool
    async def get_model_name(ctx: RunContext[UnserializableDeps]) -> int:
        return ctx.deps.client.max_redirects  # pragma: lax no cover

    unserializable_deps_hatchet_agent = HatchetAgent(unserializable_deps_agent, hatchet=hatchet)

    # Test this raises a serialization error because httpx.AsyncClient is not serializable.
    with pytest.raises(
        Exception,
        match='object proxy must define __reduce_ex__()',
    ):
        async with AsyncClient() as client:
            # This will trigger the client to be unserializable
            logfire.instrument_httpx(client, capture_all=True)
            await unserializable_deps_hatchet_agent.run(
                'What is the model name?', deps=UnserializableDeps(client=client)
            )


# Test dynamic toolsets in an agent with Hatchet


def temperature_celsius(city: str) -> float:
    return 21.0


def temperature_fahrenheit(city: str) -> float:
    return 69.8


weather_toolset = FunctionToolset(tools=[temperature_celsius, temperature_fahrenheit])


@weather_toolset.tool
def conditions(ctx: RunContext, city: str) -> str:
    if ctx.run_step % 2 == 0:
        return "It's sunny"  # pragma: lax no cover
    else:
        return "It's raining"


datetime_toolset = FunctionToolset()


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


def test_dynamic_toolset(hatchet: Hatchet):
    dynamic_hatchet_agent = HatchetAgent(dynamic_agent, hatchet=hatchet)
    weather_deps = ToggleableDeps('weather')

    result = dynamic_hatchet_agent.run_sync('Toggle the toolset', deps=weather_deps)
    assert result.output == snapshot(
        '{"toggle":null,"temperature_celsius":21.0,"temperature_fahrenheit":69.8,"conditions":"It\'s raining"}'
    )

    result = dynamic_hatchet_agent.run_sync('Toggle the toolset', deps=weather_deps)
    assert result.output == snapshot(IsStr(regex=r'{"toggle":null,"now":".+?"}'))


# Test human-in-the-loop with Hatchet agent
hitl_agent = Agent(
    model,
    name='hitl_agent',
    output_type=[str, DeferredToolRequests],
    instructions='Just call tools without asking for confirmation.',
)


@hitl_agent.tool
def create_file(ctx: RunContext[None], path: str) -> None:
    raise CallDeferred


@hitl_agent.tool
def delete_file(ctx: RunContext[None], path: str) -> bool:
    if not ctx.tool_call_approved:
        raise ApprovalRequired
    return True


# TODO: Implement HITL tests for Hatchet using its event/messaging system
# This will require understanding Hatchet's event handling and message passing


# Test model retry

model_retry_agent = Agent(model, name='model_retry_agent')


@model_retry_agent.tool_plain
def get_weather_in_city(city: str) -> str:
    if city != 'Mexico City':
        raise ModelRetry('Did you mean Mexico City?')
    return 'sunny'


async def test_hatchet_agent_with_model_retry(allow_model_requests: None, hatchet: Hatchet):
    model_retry_hatchet_agent = HatchetAgent(model_retry_agent, hatchet=hatchet)

    result = await model_retry_hatchet_agent.run('What is the weather in CDMX?')
    assert result.output == snapshot('The weather in Mexico City is currently sunny.')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather in CDMX?',
                        timestamp=IsDatetime(),
                    )
                ]
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
                provider_details={'finish_reason': 'tool_calls'},
                provider_response_id=IsStr(),
                finish_reason='tool_call',
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Did you mean Mexico City?',
                        tool_name='get_weather_in_city',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
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
                provider_details={'finish_reason': 'tool_calls'},
                provider_response_id=IsStr(),
                finish_reason='tool_call',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather_in_city',
                        content='sunny',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
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
                provider_details={'finish_reason': 'stop'},
                provider_response_id=IsStr(),
                finish_reason='stop',
            ),
        ]
    )
