"""The capabilities and toolsets a run used, as recorded on its agent run and model request spans.

`pydantic_ai.capabilities` and `pydantic_ai.toolsets` on the agent run span are keyed by the public
class path of the nearest Pydantic AI or Harness class, so a backend can index which classes a run
used by key alone; `pydantic_ai.active_capabilities` on each model request span lists the ids
active on that step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import httpx
import httpx2
import pytest

from pydantic_ai import Agent, ModelMessage, ModelResponse, RunContext, TextPart, ToolCallPart
from pydantic_ai.capabilities import AbstractCapability, Capability, Thinking
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.capabilities.mcp import MCP
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, FunctionToolset

from ._inline_snapshot import snapshot
from .conftest import IsStr, try_import

with try_import() as otel_sdk_installed:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

with try_import() as mcp_installed:
    from fastmcp import FastMCP

    from pydantic_ai._mcp_compat import is_mcp_sdk_v2
    from pydantic_ai.mcp import MCPToolset

MCP_SDK_V2 = mcp_installed() and is_mcp_sdk_v2()

pytestmark = [pytest.mark.skipif(not otel_sdk_installed(), reason='opentelemetry-sdk not installed')]


@dataclass
class BillingGuard(AbstractCapability[Any]):
    """A user-defined capability with no `id`, so the run assigns it one."""


def instrumentation(include_content: bool = True) -> tuple[Instrumentation, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    settings = InstrumentationSettings(include_content=include_content, tracer_provider=provider)
    return Instrumentation(settings=settings), exporter


def run_span_composition(exporter: InMemorySpanExporter) -> dict[str, Any]:
    (span,) = [span for span in exporter.get_finished_spans() if span.name.startswith('invoke_agent')]
    attributes = span.attributes or {}
    schema = json.loads(str(attributes['logfire.json_schema']))['properties']
    assert schema['pydantic_ai.capabilities'] == {'type': 'object'}
    assert schema['pydantic_ai.toolsets'] == {'type': 'object'}
    return {
        'capabilities': json.loads(str(attributes['pydantic_ai.capabilities'])),
        'toolsets': json.loads(str(attributes['pydantic_ai.toolsets'])),
    }


def active_capabilities_per_step(exporter: InMemorySpanExporter) -> list[list[str]]:
    steps: list[list[str]] = []
    for span in exporter.get_finished_spans():
        if span.name.startswith('chat ') and span.attributes:
            active = span.attributes['pydantic_ai.active_capabilities']
            assert isinstance(active, tuple)
            steps.append([str(capability_id) for capability_id in active])
    return steps


def tools() -> FunctionToolset[Any]:
    toolset = FunctionToolset[Any](id='crm')

    @toolset.tool_plain
    def lookup(customer: str) -> str:
        return f'{customer} is a customer.'

    return toolset


@pytest.mark.parametrize('mode', ['run', 'stream', 'iter'])
async def test_run_records_capabilities_and_toolsets(mode: Literal['run', 'stream', 'iter']) -> None:
    """Every way of running records the same composition when the run span closes."""
    capability, exporter = instrumentation()
    agent = Agent(TestModel(), capabilities=[capability, Thinking(), BillingGuard()], toolsets=[tools()])

    if mode == 'run':
        await agent.run('Hello')
    elif mode == 'stream':
        async with agent.run_stream('Hello') as result:
            await result.get_output()
    else:
        async with agent.iter('Hello') as run:
            async for _ in run:
                pass

    assert run_span_composition(exporter) == snapshot(
        {
            'capabilities': {
                'pydantic_ai.capabilities.Instrumentation': [
                    {'id': 'instrumentation', 'defer_loading': False, 'active': True}
                ],
                'pydantic_ai.capabilities.ToolSearch': [{'id': 'tool_search', 'defer_loading': False, 'active': True}],
                'pydantic_ai.capabilities.Thinking': [{'id': 'thinking', 'defer_loading': False, 'active': True}],
                'pydantic_ai.capabilities.AbstractCapability': [
                    {
                        'id': IsStr(regex=r'<billing_guard:[0-9a-f]{6}>'),
                        'user_class': 'tests.test_instrumentation_composition.BillingGuard',
                        'defer_loading': False,
                        'active': True,
                    }
                ],
            },
            'toolsets': {'pydantic_ai.toolsets.FunctionToolset': [{'id': 'crm'}]},
        }
    )
    assert active_capabilities_per_step(exporter) == snapshot(
        [
            ['instrumentation', 'tool_search', 'thinking', IsStr(regex=r'<billing_guard:[0-9a-f]{6}>')],
            ['instrumentation', 'tool_search', 'thinking', IsStr(regex=r'<billing_guard:[0-9a-f]{6}>')],
        ]
    )


async def test_without_content_only_explicit_ids_are_recorded() -> None:
    """An id the run assigned is built from the class name, which may be the user's, so it needs content."""
    capability, exporter = instrumentation(include_content=False)
    agent = Agent(TestModel(), capabilities=[capability, Thinking(), BillingGuard()], toolsets=[tools()])

    await agent.run('Hello')

    assert run_span_composition(exporter) == snapshot(
        {
            'capabilities': {
                'pydantic_ai.capabilities.Instrumentation': [
                    {'id': 'instrumentation', 'defer_loading': False, 'active': True}
                ],
                'pydantic_ai.capabilities.ToolSearch': [{'id': 'tool_search', 'defer_loading': False, 'active': True}],
                'pydantic_ai.capabilities.Thinking': [{'id': 'thinking', 'defer_loading': False, 'active': True}],
                'pydantic_ai.capabilities.AbstractCapability': [{'defer_loading': False, 'active': True}],
            },
            'toolsets': {'pydantic_ai.toolsets.FunctionToolset': [{'id': 'crm'}]},
        }
    )
    assert active_capabilities_per_step(exporter) == snapshot(
        [['instrumentation', 'tool_search', 'thinking'], ['instrumentation', 'tool_search', 'thinking']]
    )


async def test_deferred_capability_loaded_or_not() -> None:
    """A deferred capability the model loaded is active from the next step on; one it never loaded is not."""
    capability, exporter = instrumentation()
    billing = Capability[Any](id='billing', defer_loading=True, description='Billing.', instructions='Bill.')
    support = Capability[Any](id='support', defer_loading=True, description='Support.', instructions='Help.')
    responses = [ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'billing'})])]

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return responses.pop(0) if responses else ModelResponse(parts=[TextPart('Done.')])

    agent = Agent(FunctionModel(model_function), capabilities=[capability, billing, support])

    await agent.run('Hello')

    assert run_span_composition(exporter)['capabilities'] == snapshot(
        {
            'pydantic_ai.capabilities.Instrumentation': [
                {'id': 'instrumentation', 'defer_loading': False, 'active': True}
            ],
            'pydantic_ai.capabilities.ToolSearch': [{'id': 'tool_search', 'defer_loading': False, 'active': True}],
            'pydantic_ai.capabilities.Capability': [
                {'id': 'billing', 'defer_loading': True, 'active': True},
                {'id': 'support', 'defer_loading': True, 'active': False},
            ],
        }
    )
    assert active_capabilities_per_step(exporter) == snapshot(
        [['instrumentation', 'tool_search'], ['instrumentation', 'tool_search', 'billing']]
    )


async def test_dynamic_toolset_is_recorded_as_resolved_on_each_step() -> None:
    """A `DynamicToolset` resolves per step, so the run records every toolset it resolved to."""
    capability, exporter = instrumentation()
    first = FunctionToolset[Any](id='first')

    @first.tool_plain
    def greet() -> str:
        return 'Hello.'

    second = tools()

    def pick(ctx: RunContext[Any]) -> AbstractToolset[Any]:
        return first if ctx.run_step <= 1 else second

    agent = Agent(TestModel(), capabilities=[capability], toolsets=[DynamicToolset(pick, per_run_step=True)])

    await agent.run('Hello')

    assert run_span_composition(exporter)['toolsets'] == snapshot(
        {'pydantic_ai.toolsets.FunctionToolset': [{'id': 'first'}, {'id': 'crm'}]}
    )


@pytest.mark.parametrize('include_content', [True, False])
async def test_url_derived_mcp_id_is_never_recorded(include_content: bool) -> None:
    """An id derived from an MCP server URL is withheld; the class and an explicit id are still recorded."""
    capability, exporter = instrumentation(include_content=include_content)
    derived = MCP('https://mcp.example.test/private/SENTINEL-path', native=True, local=False)
    explicit = MCP('https://other.example.test/mcp', id='docs', native=True, local=False)
    agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('Done.')])),
        capabilities=[capability, derived, explicit],
    )

    await agent.run('Hello')

    assert run_span_composition(exporter)['capabilities']['pydantic_ai.capabilities.MCP'] == snapshot(
        [{'defer_loading': False, 'active': True}, {'id': 'docs', 'defer_loading': False, 'active': True}]
    )
    assert all('mcp.example.test' not in ids for ids in active_capabilities_per_step(exporter))
    assert 'mcp.example.test' not in str([span.attributes for span in exporter.get_finished_spans()])


@pytest.mark.skipif(not mcp_installed(), reason='fastmcp not installed')
@pytest.mark.parametrize('include_content', [True, False])
async def test_url_derived_mcp_toolset_id_is_never_recorded(include_content: bool) -> None:
    """An `MCPToolset` built from a URL without an `id` takes its hostname, which isn't recorded even with content."""
    server = FastMCP('test')

    @server.tool
    def ping() -> str:
        return 'pong'  # pragma: no cover

    app = server.http_app(stateless_http=True, json_response=True)
    client: httpx.AsyncClient | httpx2.AsyncClient
    if MCP_SDK_V2:
        client = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app))
    else:
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
    toolset = MCPToolset('http://testserver/mcp', http_client=client)
    assert toolset.id == 'testserver'
    capability, exporter = instrumentation(include_content=include_content)
    agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('Done.')])),
        capabilities=[capability],
        toolsets=[toolset],
    )
    try:
        async with app.router.lifespan_context(app):
            await agent.run('Hello')
    finally:
        await client.aclose()

    assert run_span_composition(exporter)['toolsets'] == snapshot({'pydantic_ai.mcp.MCPToolset': [{}]})
    assert 'testserver' not in str([span.attributes for span in exporter.get_finished_spans()])
