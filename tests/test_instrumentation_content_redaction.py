"""One sweep over every way content enters an agent, checking none of it reaches telemetry.

`InstrumentationSettings(include_content=False)` is a promise about the whole exported trace, not
about individual attributes, but it has been fixed one channel at a time: the retry prompt
(GHSA-3gh4-cghq-f8v4), then exception events on tool and agent run spans, then the model request
span, then realtime, then the ERROR status description that repeated every exception message. Each
fix was correct and each left another channel open, because nothing checked the promise as a whole.

This module does. Each content channel gets a distinct sentinel, every run happens with content
capture off, and the exported spans are scanned exhaustively -- names, attributes, event
attributes, and status descriptions, from the raw `ReadableSpan`s rather than a dict view that
omits the status and truncates stack traces. A channel that leaks names itself in the failure.

Run metadata is deliberately not a channel here. It comes from the agent definition rather than
from a user, and is expected to appear on spans; content in there is the caller's own doing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import httpx
import httpx2
import pytest
from opentelemetry.context import Context
from pydantic import BaseModel, TypeAdapter

from pydantic_ai import (
    Agent,
    FunctionToolset,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai._instrumentation import (
    ContentPolicy,
    MessageJsonCache,
    has_stale_message_json,
    include_content_ctx,
    model_request_parameters_attributes,
    open_request_policy,
    span_include_content,
)
from pydantic_ai.capabilities import Hooks, LocalWorkspace
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.capabilities.mcp import MCP
from pydantic_ai.direct import model_request
from pydantic_ai.exceptions import ModelHTTPError, ModelRetry, ToolFailed, UnexpectedModelBehavior
from pydantic_ai.messages import NativeToolCallPart, NativeToolReturnPart, UploadedFile
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import CodeExecutionTool, FileSearchTool, MCPServerTool
from pydantic_ai.output import PromptedOutput
from pydantic_ai.tools import Tool, ToolDefinition

from .conftest import try_import

with try_import() as otel_sdk_installed:
    # `opentelemetry-sdk` arrives with the `logfire` extra, so it is not importable in the
    # `pydantic-ai-slim` / `pydantic-evals` install groups.
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

with try_import() as mcp_installed:
    from fastmcp import Client, FastMCP
    from fastmcp.client.transports import StreamableHttpTransport

    from pydantic_ai._mcp_compat import is_mcp_sdk_v2
    from pydantic_ai.mcp import MCPToolset

MCP_SDK_V2 = mcp_installed() and is_mcp_sdk_v2()

pytestmark = [
    pytest.mark.skipif(not otel_sdk_installed(), reason='opentelemetry-sdk not installed'),
]

# Every way content reaches an agent, each with a sentinel that identifies the channel when it leaks.
SECRETS = {
    'user_prompt': 'SENTINEL-user-prompt',
    'instructions': 'SENTINEL-instructions',
    'dynamic_instructions': 'SENTINEL-dynamic-instructions',
    'system_prompt': 'SENTINEL-system-prompt',
    'message_history': 'SENTINEL-message-history',
    'tool_args': 'SENTINEL-tool-args',
    'tool_return': 'SENTINEL-tool-return',
    'tool_retry': 'SENTINEL-tool-retry',
    'tool_failed': 'SENTINEL-tool-failed',
    'tool_exception': 'SENTINEL-tool-exception',
    'output_validator_retry': 'SENTINEL-output-validator-retry',
    'final_output': 'SENTINEL-final-output',
    'model_text': 'SENTINEL-model-text',
    'provider_error_body': 'SENTINEL-provider-error-body',
    'prompted_output_template': 'SENTINEL-prompted-output-template',
}


def _span_text(span: ReadableSpan) -> str:
    """Everything a backend would receive for one span, as text to search.

    Deliberately not `exported_spans_as_dict()`: that view drops the status, whose description
    repeated every exception message, and truncates stack traces.
    """
    parts: list[str] = [span.name, str(span.status.description or '')]
    parts += [f'{key}={value}' for key, value in (span.attributes or {}).items()]
    for event in span.events:
        parts.append(event.name)
        parts += [f'{key}={value}' for key, value in (event.attributes or {}).items()]
    return '\n'.join(parts)


def leaked_channels(spans: Sequence[ReadableSpan]) -> set[str]:
    """The channels whose sentinel appears anywhere in the exported spans."""
    exported = '\n'.join(_span_text(span) for span in spans)
    return {channel for channel, secret in SECRETS.items() if secret in exported}


def redacted_setup(include_content: bool = False) -> tuple[InstrumentationSettings, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return InstrumentationSettings(include_content=include_content, tracer_provider=provider), exporter


async def test_workspace_identity_is_recorded_without_content(tmp_path: Path) -> None:
    settings, exporter = redacted_setup(include_content=False)
    agent = Agent(TestModel(), capabilities=[LocalWorkspace(tmp_path), Instrumentation(settings=settings)])
    await agent.run('go')
    run_spans = [
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get('gen_ai.operation.name') == 'invoke_agent'
    ]
    assert len(run_spans) == 1
    attributes = run_spans[0].attributes
    assert attributes is not None
    assert attributes['pydantic_ai.workspace.provider'] == 'local'
    assert attributes['pydantic_ai.workspace.id'] == str(tmp_path)


def check(exporter: InMemorySpanExporter, include_content: bool, drives: set[str]) -> None:
    """Assert the run exported something, and that it leaked exactly what it should.

    `drives` is the set of channels the scenario actually pushes content through. With capture on it
    is the positive control -- every one of them has to appear, or the scenario is not exercising the
    channel it claims and its redacted run proves nothing. With capture off, nothing may appear.
    """
    spans = exporter.get_finished_spans()
    assert spans, 'no spans exported: the scenario proves nothing about redaction'
    leaked = leaked_channels(spans)
    assert leaked == (drives if include_content else set())


@pytest.mark.parametrize('include_content', [True, False])
def test_native_mcp_credentials_are_masked_but_explicit_id_is_retained(include_content: bool) -> None:
    tool = MCPServerTool(
        id='docs',
        url='https://mcp.example.com?token=SENTINEL-url',
        authorization_token='SENTINEL-token',
        headers={'Authorization': 'SENTINEL-header'},
        description='Docs server',
        allowed_tools=['search'],
    )
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='lookup', metadata={'source': 'user'})],
        native_tools=[
            tool,
            CodeExecutionTool(files=[UploadedFile(file_id='file-123', provider_name='openai')]),
            FileSearchTool(file_store_ids=['store'], instructions='search only these files'),
        ],
    )
    serialized = model_request_parameters_attributes(parameters, include_content=include_content)[
        'model_request_parameters'
    ]
    assert isinstance(serialized, str)
    assert 'SENTINEL-' not in serialized
    exported = json.loads(serialized)
    native_tool = exported['native_tools'][0]
    assert native_tool['id'] == 'docs'
    assert native_tool['url'] == '[REDACTED]'
    assert native_tool['authorization_token'] == '[REDACTED]'
    assert native_tool['headers'] == {'Authorization': '[REDACTED]'}
    assert native_tool['description'] == 'Docs server'
    assert native_tool['allowed_tools'] == ['search']
    assert exported['function_tools'][0]['metadata'] == {'source': 'user'}
    assert exported['native_tools'][1]['files'][0]['file_id'] == 'file-123'
    assert exported['native_tools'][2]['instructions'] == 'search only these files'
    assert tool.authorization_token == 'SENTINEL-token'
    response = ModelResponse(parts=[NativeToolCallPart(tool_name='mcp_server:docs', tool_call_id='call-1')])
    settings, exporter = redacted_setup(include_content)
    with settings.tracer.start_as_current_span('chat') as span:
        settings.handle_messages([], response, span, parameters)
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert 'mcp_server:docs' in str(attributes['gen_ai.output.messages'])


def test_unrelated_request_telemetry_matches_the_existing_serializer() -> None:
    parameters = ModelRequestParameters(
        function_tools=[
            ToolDefinition(name='lookup', toolset_id='user-chosen', capability_id='user-capability', metadata={'x': 1})
        ],
        native_tools=[
            CodeExecutionTool(files=[UploadedFile(file_id='file-123', provider_name='openai')]),
            FileSearchTool(file_store_ids=['store'], instructions='search only these files'),
        ],
    )
    expected = TypeAdapter(ModelRequestParameters).dump_python(parameters, mode='json')
    serialized = model_request_parameters_attributes(parameters)['model_request_parameters']
    assert isinstance(serialized, str)
    assert json.loads(serialized) == expected
    assert expected['function_tools'][0]['metadata'] == {'x': 1}
    assert expected['native_tools'][0]['files'][0]['file_id'] == 'file-123'
    assert expected['native_tools'][1]['instructions'] == 'search only these files'


def test_native_mcp_redaction_survives_serializer_fallback() -> None:
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='lookup', metadata={'opaque': object()})],
        native_tools=[MCPServerTool(id='docs', url='https://mcp.example.test/SENTINEL-url')],
    )
    serialized = model_request_parameters_attributes(parameters)['model_request_parameters']
    assert isinstance(serialized, str)
    assert 'SENTINEL-url' not in serialized
    assert '[REDACTED]' in serialized


def test_model_parameter_customization_preserves_url_identity_provenance() -> None:
    class RebuildsParameters(TestModel):
        def customize_request_parameters(
            self, model_request_parameters: ModelRequestParameters
        ) -> ModelRequestParameters:
            return ModelRequestParameters(native_tools=model_request_parameters.native_tools)

    capability = MCP('https://mcp.example.test/private/SENTINEL-path', native=True, local=False)
    assert capability.id is not None
    tool = next(iter(capability.get_native_tools()))
    assert isinstance(tool, MCPServerTool)
    parameters = ModelRequestParameters(
        native_tools=[tool],
        _telemetry_mcp_url_ids={capability.id},
        _telemetry_mcp_url_toolset_ids={capability.id},
    )
    _, prepared = RebuildsParameters().prepare_request(None, parameters)

    assert prepared._telemetry_mcp_url_ids == {capability.id}  # pyright: ignore[reportPrivateUsage]
    assert prepared._telemetry_mcp_url_toolset_ids == {capability.id}  # pyright: ignore[reportPrivateUsage]
    assert isinstance(prepared.native_tools[0], MCPServerTool)
    assert prepared.native_tools[0].id == capability.id
    serialized = model_request_parameters_attributes(prepared)['model_request_parameters']
    assert isinstance(serialized, str)
    assert 'SENTINEL-path' not in serialized
    assert json.loads(serialized)['native_tools'][0]['id'] == '[REDACTED]'


def test_model_parameter_customization_preserves_explicit_native_id() -> None:
    class RebuildsParameters(TestModel):
        def customize_request_parameters(
            self, model_request_parameters: ModelRequestParameters
        ) -> ModelRequestParameters:
            return ModelRequestParameters(native_tools=model_request_parameters.native_tools)

    url = 'https://mcp.example.test/private/SENTINEL-path'
    identity = MCP(url, native=True, local=False).id
    assert identity is not None
    parameters = ModelRequestParameters(
        native_tools=[MCPServerTool(id=identity, url=url)], _telemetry_mcp_url_ids=set()
    )

    _, prepared = RebuildsParameters().prepare_request(None, parameters)
    serialized = model_request_parameters_attributes(prepared)['model_request_parameters']
    assert isinstance(serialized, str)
    assert prepared._telemetry_mcp_url_ids == set()  # pyright: ignore[reportPrivateUsage]
    assert json.loads(serialized)['native_tools'][0]['id'] == identity
    assert json.loads(serialized)['native_tools'][0]['url'] == '[REDACTED]'


def test_model_parameter_customization_classifies_added_url_derived_native_id() -> None:
    capability = MCP('https://mcp.example.test/private/SENTINEL-added-tool', native=True, local=False)
    assert capability.id is not None
    tool = next(iter(capability.get_native_tools()))
    assert isinstance(tool, MCPServerTool)

    class AddsTool(TestModel):
        def customize_request_parameters(
            self, model_request_parameters: ModelRequestParameters
        ) -> ModelRequestParameters:
            return ModelRequestParameters(
                native_tools=[*model_request_parameters.native_tools, tool], _telemetry_mcp_url_ids=set()
            )

    _, prepared = AddsTool().prepare_request(None, ModelRequestParameters(_telemetry_mcp_url_ids=set()))
    serialized = model_request_parameters_attributes(prepared)['model_request_parameters']
    assert isinstance(serialized, str)
    assert prepared._telemetry_mcp_url_ids == {capability.id}  # pyright: ignore[reportPrivateUsage]
    assert json.loads(serialized)['native_tools'][0]['id'] == '[REDACTED]'
    assert tool.id == capability.id

    settings, exporter = redacted_setup(True)
    response = ModelResponse(parts=[NativeToolCallPart(tool_name=f'mcp_server:{capability.id}', tool_call_id='call-1')])
    with settings.tracer.start_as_current_span('chat') as span:
        settings.handle_messages([], response, span, prepared)
    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert 'SENTINEL-added-tool' not in str(attributes['gen_ai.output.messages'])


def test_url_derived_mcp_tool_call_name_is_masked_in_request_span() -> None:
    capability = MCP('https://mcp.example.test/private/SENTINEL-path', native=True, local=False)
    tool = next(iter(capability.get_native_tools()))
    assert isinstance(tool, MCPServerTool)
    assert capability.id is not None
    parameters = ModelRequestParameters(native_tools=[tool], _telemetry_mcp_url_ids={capability.id})
    tool_name = f'mcp_server:{capability.id}'
    response = ModelResponse(
        parts=[
            NativeToolCallPart(tool_name=tool_name, tool_call_id='call-1', provider_name='openai'),
            NativeToolReturnPart(tool_name=tool_name, tool_call_id='call-1', content={}, provider_name='openai'),
        ]
    )
    settings, exporter = redacted_setup(True)
    with settings.tracer.start_as_current_span('chat') as span:
        span.set_attributes(model_request_parameters_attributes(parameters))
        settings.handle_messages([], response, span, parameters)

    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    assert 'SENTINEL-path' not in str(attributes)
    assert isinstance(response.parts[0], NativeToolCallPart)
    assert response.parts[0].tool_name == tool_name


def test_cached_history_name_projection_is_conservative_across_requests() -> None:
    identity = 'mcp.example.test'
    history = ModelResponse(parts=[NativeToolCallPart(tool_name=f'mcp_server:{identity}', tool_call_id='call-1')])
    settings, exporter = redacted_setup(True)
    cache: MessageJsonCache = {}
    for url_ids in (set[str](), {identity}):
        parameters = ModelRequestParameters(_telemetry_mcp_url_ids=url_ids)
        with settings.tracer.start_as_current_span('chat') as span:
            settings.handle_messages(
                [history], ModelResponse(parts=[TextPart('done')]), span, parameters, message_json_cache=cache
            )

    first, second = exporter.get_finished_spans()
    assert first.attributes is not None and second.attributes is not None
    assert identity not in str(first.attributes['gen_ai.input.messages'])
    assert identity not in str(second.attributes['gen_ai.input.messages'])
    assert not has_stale_message_json(settings, [history], cache)


@pytest.mark.parametrize('include_content', [True, False])
async def test_url_derived_mcp_name_is_masked_if_run_fails_before_model_request(include_content: bool) -> None:
    capability = MCP('https://mcp.example.test/private/SENTINEL-path', native=True, local=False)
    assert capability.id is not None
    tool_name = f'mcp_server:{capability.id}'
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('previous'),
        ModelResponse(parts=[NativeToolCallPart(tool_name=tool_name, tool_call_id='call-1')]),
    ]

    async def fail_before_run(_ctx: RunContext[None]) -> None:
        raise RuntimeError('before model request')

    settings, exporter = redacted_setup(include_content)
    agent = Agent(
        TestModel(), capabilities=[capability, Hooks(before_run=fail_before_run), Instrumentation(settings=settings)]
    )
    with pytest.raises(RuntimeError, match='before model request'):
        await agent.run('go', message_history=history)

    agent_span = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get('gen_ai.operation.name') == 'invoke_agent'
    )
    assert agent_span.attributes is not None
    assert 'SENTINEL-path' not in str(agent_span.attributes['pydantic_ai.all_messages'])
    assert isinstance(history[1].parts[0], NativeToolCallPart)
    assert history[1].parts[0].tool_name == tool_name


@pytest.fixture
def guidance_server_app() -> Any:
    """An MCP server with instructions and one tool, as an HTTP app built outside the event loop."""
    server = FastMCP('test', instructions='server guidance')

    @server.tool
    def ping() -> str:
        return 'pong'  # pragma: no cover

    return server.http_app(stateless_http=True, json_response=True)


@pytest.mark.skipif(not mcp_installed(), reason='fastmcp not installed')
@pytest.mark.parametrize('include_content', [True, False])
async def test_a_url_mcp_toolsets_hostname_derived_id_is_masked(
    include_content: bool, guidance_server_app: Any
) -> None:
    """An `MCPToolset` built from a URL without an `id` takes its hostname, which telemetry masks like other URL-derived ids.

    It reaches spans as the server instructions' id, which is masked; tool definitions don't carry it.
    """
    app = guidance_server_app
    client: httpx.AsyncClient | httpx2.AsyncClient
    if MCP_SDK_V2:
        client = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app))
    else:
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
    toolset = MCPToolset('http://testserver/mcp', http_client=client, include_instructions=True)
    assert toolset.id == 'testserver'

    def respond(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    settings, exporter = redacted_setup(include_content)
    agent = Agent(FunctionModel(respond), toolsets=[toolset], capabilities=[Instrumentation(settings=settings)])
    try:
        async with app.router.lifespan_context(app):
            await agent.run('go')
    finally:
        await client.aclose()

    chat = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get('gen_ai.operation.name') == 'chat'
    )
    assert chat.attributes is not None
    serialized = chat.attributes['model_request_parameters']
    assert isinstance(serialized, str)
    parameters = json.loads(serialized)
    # A hostname-derived `id` isn't stamped on tool definitions at all.
    assert [tool['toolset_id'] for tool in parameters['function_tools']] == [None]
    assert [part['id'] for part in parameters['instruction_parts']] == ['toolset:[REDACTED]']
    if not include_content:
        assert 'testserver' not in serialized


@pytest.mark.skipif(not mcp_installed(), reason='fastmcp not installed')
@pytest.mark.parametrize('local', ['client', 'transport'])
def test_an_mcp_capability_keeps_its_local_toolsets_own_marker(local: str):
    """`MCP(local=...)` with no URL or `id` of its own has no marker to give, so the toolset keeps the one its hostname-derived `id` set."""
    url = 'http://private-tenant.example/mcp'
    capability = MCP[None](local=Client(url) if local == 'client' else StreamableHttpTransport(url))
    # `MCP.local` is typed as every input it accepts; this one was wrapped in an `MCPToolset`.
    local_toolset: Any = capability.local
    assert isinstance(local_toolset, MCPToolset)
    toolset = cast('MCPToolset[None]', local_toolset)
    assert toolset.id == 'private-tenant.example'
    assert toolset._telemetry_url_derived_id == 'private-tenant.example'  # pyright: ignore[reportPrivateUsage]


def test_a_hostname_derived_id_is_masked_only_while_its_derived():
    """Choosing an `id` makes it the caller's label, which isn't masked; clearing it derives and masks the hostname again."""
    toolset = MCPToolset('https://private-tenant.example/mcp')
    assert toolset._telemetry_url_derived_id == 'private-tenant.example'  # pyright: ignore[reportPrivateUsage]
    toolset.id = 'chosen'
    assert toolset._telemetry_url_derived_id is None  # pyright: ignore[reportPrivateUsage]
    toolset.id = None
    assert toolset.id == 'private-tenant.example'
    assert toolset._telemetry_url_derived_id == 'private-tenant.example'  # pyright: ignore[reportPrivateUsage]
    assert MCPToolset('https://private-tenant.example/mcp', id='docs')._telemetry_url_derived_id is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(not mcp_installed(), reason='fastmcp not installed')
@pytest.mark.parametrize('include_content', [True, False])
@pytest.mark.parametrize('native', [True, False])
async def test_only_url_derived_mcp_ids_are_masked(include_content: bool, native: bool) -> None:
    class SupportsMCP(FunctionModel):
        @classmethod
        def supported_native_tools(cls) -> frozenset[type[MCPServerTool]]:
            return frozenset({MCPServerTool})

    server = FastMCP('test', instructions='server guidance')

    @server.tool
    def ping() -> str:
        return 'pong'

    observed_parameters: list[ModelRequestParameters] = []

    def respond(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        observed_parameters.append(info.model_request_parameters)
        return ModelResponse(parts=[TextPart('done')])

    settings, exporter = redacted_setup(include_content)
    # This module's in-process servers are passed as a pre-built `Client`, which keeps fastmcp's default of
    # no handshake deadline. Wrapping the bare server would apply `MCPToolset`'s 5-second `init_timeout`,
    # and a stalled CI worker can exceed it.
    capability = MCP('https://mcp.example.test/private/SENTINEL-path', native=native, local=Client(server))
    assert capability.id is not None
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('previous'),
        ModelResponse(parts=[NativeToolCallPart(tool_name=f'mcp_server:{capability.id}', tool_call_id='call-1')]),
    ]
    await Agent(SupportsMCP(respond), capabilities=[capability, Instrumentation(settings=settings)]).run(
        'go', message_history=history
    )
    agent_span = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get('gen_ai.operation.name') == 'invoke_agent'
    )
    assert agent_span.attributes is not None
    assert 'SENTINEL-path' not in str(agent_span.attributes['pydantic_ai.all_messages'])
    chat = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get('gen_ai.operation.name') == 'chat'
    )
    assert chat.attributes is not None
    serialized = chat.attributes['model_request_parameters']
    assert isinstance(serialized, str)
    assert 'SENTINEL-path' not in serialized
    assert observed_parameters
    # The id is derived from the hostname alone, so the path's token never reaches it.
    assert capability.id == 'mcp.example.test'
    if native:
        assert json.loads(serialized)['native_tools'][0]['id'] == '[REDACTED]'
        assert any(
            tool.id == capability.id for tool in observed_parameters[0].native_tools if isinstance(tool, MCPServerTool)
        )
    else:
        assert json.loads(serialized)['function_tools'][0]['toolset_id'] == '[REDACTED]'
        assert any(tool.toolset_id == capability.id for tool in observed_parameters[0].function_tools)


@pytest.mark.parametrize('include_content', [True, False])
@pytest.mark.parametrize('current_mcp', [True, False])
async def test_resumed_mcp_history_from_another_server_is_masked(include_content: bool, current_mcp: bool) -> None:
    class SupportsMCP(FunctionModel):
        @classmethod
        def supported_native_tools(cls) -> frozenset[type[MCPServerTool]]:
            return frozenset({MCPServerTool})

    old_name = 'mcp_server:old.example-OLD-token'
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('previous'),
        ModelResponse(parts=[NativeToolCallPart(tool_name=old_name, tool_call_id='call-1')]),
    ]
    current = MCP('https://new.example/current', native=True, local=False)

    def respond(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    settings, exporter = redacted_setup(include_content)
    capabilities = (
        [current, Instrumentation(settings=settings)] if current_mcp else [Instrumentation(settings=settings)]
    )
    await Agent(SupportsMCP(respond), capabilities=capabilities).run('go', message_history=history)
    agent_span = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get('gen_ai.operation.name') == 'invoke_agent'
    )
    chat = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get('gen_ai.operation.name') == 'chat'
    )
    assert agent_span.attributes is not None and chat.attributes is not None
    assert 'OLD-token' not in str(agent_span.attributes['pydantic_ai.all_messages'])
    assert 'OLD-token' not in str(chat.attributes['gen_ai.input.messages'])
    assert isinstance(history[1], ModelResponse)
    original_part = history[1].parts[0]
    assert isinstance(original_part, NativeToolCallPart)
    assert original_part.tool_name == old_name


@pytest.mark.skipif(not mcp_installed(), reason='fastmcp not installed')
async def test_explicit_mcp_toolset_id_remains_in_request_telemetry() -> None:
    server = FastMCP('test')

    @server.tool
    def ping() -> str:
        return 'pong'

    def respond(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    settings, exporter = redacted_setup()
    toolset = MCPToolset(Client(server), id='github')
    await Agent(FunctionModel(respond), toolsets=[toolset], capabilities=[Instrumentation(settings=settings)]).run('go')
    chat = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get('gen_ai.operation.name') == 'chat'
    )
    assert chat.attributes is not None
    serialized = chat.attributes['model_request_parameters']
    assert isinstance(serialized, str)
    assert json.loads(serialized)['function_tools'][0]['toolset_id'] == 'github'


async def test_unrelated_toolset_id_matching_mcp_url_id_is_retained() -> None:
    class SupportsMCP(FunctionModel):
        @classmethod
        def supported_native_tools(cls) -> frozenset[type[MCPServerTool]]:
            return frozenset({MCPServerTool})

    capability = MCP('https://mcp.example.test/private/SENTINEL-path', native=True, local=False)
    assert capability.id is not None
    unrelated = FunctionToolset(id=capability.id, instructions='public guidance')

    @unrelated.tool_plain
    def lookup() -> str:
        return 'ok'

    requests = 0

    def respond(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart('lookup', {})])
        return ModelResponse(parts=[TextPart('done')])

    settings, exporter = redacted_setup()
    result = await Agent(
        SupportsMCP(respond), toolsets=[unrelated], capabilities=[capability, Instrumentation(settings=settings)]
    ).run('go')
    assert result.output == 'done'
    chat = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get('gen_ai.operation.name') == 'chat'
    )
    assert chat.attributes is not None
    serialized = chat.attributes['model_request_parameters']
    assert isinstance(serialized, str)
    exported = json.loads(serialized)
    assert exported['native_tools'][0]['id'] == '[REDACTED]'
    assert exported['function_tools'][0]['toolset_id'] == capability.id
    assert capability.id in json.dumps(exported['instruction_parts'])


@pytest.mark.parametrize('last_explicit', [False, True])
async def test_combined_mcp_keeps_the_winning_id_provenance(last_explicit: bool) -> None:
    class SupportsMCP(FunctionModel):
        @classmethod
        def supported_native_tools(cls) -> frozenset[type[MCPServerTool]]:
            return frozenset({MCPServerTool})

    url = 'https://mcp.example.test/private/SENTINEL-path'
    derived = MCP(url, native=True, local=False)
    assert derived.id is not None
    explicit = MCP(url, native=True, local=False, id=derived.id)
    merged = MCP.combine([derived, explicit] if last_explicit else [explicit, derived])

    def respond(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    settings, exporter = redacted_setup()
    await Agent(SupportsMCP(respond), capabilities=[merged, Instrumentation(settings=settings)]).run('go')
    chat = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get('gen_ai.operation.name') == 'chat'
    )
    assert chat.attributes is not None
    serialized = chat.attributes['model_request_parameters']
    assert isinstance(serialized, str)
    assert json.loads(serialized)['native_tools'][0]['id'] == (derived.id if last_explicit else '[REDACTED]')


@pytest.mark.skipif(not mcp_installed(), reason='fastmcp not installed')
async def test_local_mcp_instruction_id_is_masked_without_tools() -> None:
    server = FastMCP('test', instructions='server guidance')
    capability = MCP('https://mcp.example.test/private/SENTINEL-path', native=False, local=Client(server))
    assert capability.id is not None

    def respond(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    settings, exporter = redacted_setup()
    await Agent(FunctionModel(respond), capabilities=[capability, Instrumentation(settings=settings)]).run('go')
    chat = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get('gen_ai.operation.name') == 'chat'
    )
    assert chat.attributes is not None
    serialized = chat.attributes['model_request_parameters']
    assert isinstance(serialized, str)
    exported = json.loads(serialized)
    assert exported['function_tools'] == []
    assert '[REDACTED]' in json.dumps(exported['instruction_parts'])
    assert capability.id not in json.dumps(exported['instruction_parts'])


@pytest.mark.parametrize('explicit_id', [None, 'chosen', 'matching'])
async def test_direct_model_request_masks_only_url_derived_native_mcp_id(explicit_id: str | None) -> None:
    class SupportsMCP(FunctionModel):
        @classmethod
        def supported_native_tools(cls) -> frozenset[type[MCPServerTool]]:
            return frozenset({MCPServerTool})

    url = 'https://mcp.example.test/private/SENTINEL-path'
    capability = MCP(url, native=True, local=False)
    assert capability.id is not None
    caller_id = capability.id if explicit_id == 'matching' else explicit_id
    tool = MCPServerTool(id=caller_id, url=url) if caller_id is not None else next(iter(capability.get_native_tools()))
    assert isinstance(tool, MCPServerTool)

    def respond(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[NativeToolCallPart(tool_name=f'mcp_server:{tool.id}', tool_call_id='call-1')])

    settings, exporter = redacted_setup()
    await model_request(
        SupportsMCP(respond),
        [ModelRequest.user_text_prompt('go')],
        model_request_parameters=ModelRequestParameters(native_tools=[tool]),
        instrument=settings,
    )
    chat = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get('gen_ai.operation.name') == 'chat'
    )
    assert chat.attributes is not None
    serialized = chat.attributes['model_request_parameters']
    assert isinstance(serialized, str)
    assert json.loads(serialized)['native_tools'][0]['id'] == (caller_id or '[REDACTED]')
    assert json.loads(serialized)['native_tools'][0]['url'] == '[REDACTED]'
    assert f'mcp_server:{caller_id or "[REDACTED]"}' in str(chat.attributes['gen_ai.output.messages'])
    if explicit_id != 'matching':
        assert 'SENTINEL-path' not in str(chat.attributes)


@pytest.mark.skipif(not mcp_installed(), reason='fastmcp not installed')
async def test_unrelated_toolset_id_matching_local_mcp_id_is_retained() -> None:
    server = FastMCP('test', instructions='server guidance')

    @server.tool
    def ping() -> str:
        return 'pong'

    capability = MCP('https://mcp.example.test/private/SENTINEL-path', native=False, local=Client(server))
    assert capability.id is not None
    unrelated = FunctionToolset(id=capability.id)

    @unrelated.tool_plain
    def lookup() -> str:
        return 'ok'

    def respond(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    settings, exporter = redacted_setup()
    await Agent(
        FunctionModel(respond), toolsets=[unrelated], capabilities=[capability, Instrumentation(settings=settings)]
    ).run('go')
    chat = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get('gen_ai.operation.name') == 'chat'
    )
    assert chat.attributes is not None
    serialized = chat.attributes['model_request_parameters']
    assert isinstance(serialized, str)
    by_name = {tool['name']: tool for tool in json.loads(serialized)['function_tools']}
    assert by_name['ping']['toolset_id'] == '[REDACTED]'
    assert by_name['lookup']['toolset_id'] == capability.id


class Output(BaseModel):
    answer: str


@pytest.mark.parametrize('include_content', [True, False])
async def test_no_content_reaches_telemetry_on_a_successful_run(include_content: bool) -> None:
    """Prompts, instructions, history, tool arguments and results, and the final output."""
    settings, exporter = redacted_setup(include_content)

    requests = 0

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart('lookup', {'query': SECRETS['tool_args']})])
        return ModelResponse(
            parts=[
                TextPart(SECRETS['model_text']),
                ToolCallPart('final_result', {'answer': SECRETS['final_output']}),
            ]
        )

    agent = Agent(
        FunctionModel(respond),
        instructions=SECRETS['instructions'],
        system_prompt=SECRETS['system_prompt'],
        output_type=Output,
        capabilities=[Instrumentation(settings=settings)],
    )

    @agent.instructions
    def extra_instructions() -> str:
        return SECRETS['dynamic_instructions']

    @agent.tool_plain
    def lookup(query: str) -> str:
        return SECRETS['tool_return']

    result = await agent.run(SECRETS['user_prompt'])
    assert result.output.answer == SECRETS['final_output']
    # The tool has to have run, or the argument and return sentinels prove nothing.
    assert requests == 2

    check(
        exporter,
        include_content,
        {
            'user_prompt',
            'instructions',
            'dynamic_instructions',
            'system_prompt',
            'tool_args',
            'tool_return',
            'model_text',
            'final_output',
        },
    )


@pytest.mark.parametrize('include_content', [True, False])
async def test_no_content_reaches_telemetry_from_message_history(include_content: bool) -> None:
    """Messages the caller carried in from an earlier run."""
    settings, exporter = redacted_setup(include_content)

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(SECRETS['model_text'])])

    agent = Agent(FunctionModel(respond), capabilities=[Instrumentation(settings=settings)])
    history = [
        ModelRequest(parts=[UserPromptPart(content=SECRETS['message_history'])]),
        ModelResponse(parts=[TextPart('earlier reply')]),
    ]
    await agent.run(SECRETS['user_prompt'], message_history=history)

    check(exporter, include_content, {'user_prompt', 'message_history', 'model_text'})


@pytest.mark.parametrize('include_content', [True, False])
async def test_no_content_reaches_telemetry_when_a_tool_retries_to_exhaustion(include_content: bool) -> None:
    """The retry prompt, and the error chained from it that ends the run."""
    settings, exporter = redacted_setup(include_content)

    def call_tool(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('flaky', {'value': SECRETS['tool_args']})])

    agent = Agent(FunctionModel(call_tool), retries=1, capabilities=[Instrumentation(settings=settings)])

    @agent.tool_plain
    def flaky(value: str) -> str:
        raise ModelRetry(SECRETS['tool_retry'])

    with pytest.raises(UnexpectedModelBehavior):
        await agent.run(SECRETS['user_prompt'])

    check(exporter, include_content, {'user_prompt', 'tool_args', 'tool_retry'})


@pytest.mark.parametrize('include_content', [True, False])
async def test_no_content_reaches_telemetry_when_a_tool_fails_or_raises(include_content: bool) -> None:
    """`ToolFailed`'s message, and a plain exception from tool code."""
    settings, exporter = redacted_setup(include_content)

    def call_tools(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('failing_tool', {})])
        return ModelResponse(parts=[ToolCallPart('raising_tool', {})])

    agent = Agent(FunctionModel(call_tools), capabilities=[Instrumentation(settings=settings)])

    @agent.tool_plain
    def failing_tool() -> str:
        raise ToolFailed(SECRETS['tool_failed'])

    @agent.tool_plain
    def raising_tool() -> str:
        raise ValueError(SECRETS['tool_exception'])

    with pytest.raises(ValueError):
        await agent.run(SECRETS['user_prompt'])

    check(exporter, include_content, {'user_prompt', 'tool_failed', 'tool_exception'})


@pytest.mark.parametrize('include_content', [True, False])
async def test_no_content_reaches_telemetry_when_an_output_validator_retries(include_content: bool) -> None:
    """A retry raised outside a tool call, which is the GHSA-3gh4-cghq-f8v4 shape."""
    settings, exporter = redacted_setup(include_content)

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(SECRETS['model_text'])])

    agent = Agent(FunctionModel(respond), retries=1, capabilities=[Instrumentation(settings=settings)])

    @agent.output_validator
    def reject(value: str) -> str:
        raise ModelRetry(SECRETS['output_validator_retry'])

    with pytest.raises(UnexpectedModelBehavior):
        await agent.run(SECRETS['user_prompt'])

    check(exporter, include_content, {'user_prompt', 'model_text', 'output_validator_retry'})


@pytest.mark.parametrize('include_content', [True, False])
async def test_no_content_reaches_telemetry_when_the_provider_errors(include_content: bool) -> None:
    """A provider's error body, which travels in the exception message."""
    settings, exporter = redacted_setup(include_content)

    def fail(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=400, model_name='fn', body=SECRETS['provider_error_body'])

    agent = Agent(FunctionModel(fail), capabilities=[Instrumentation(settings=settings)])

    with pytest.raises(ModelHTTPError):
        await agent.run(SECRETS['user_prompt'])

    check(exporter, include_content, {'user_prompt', 'provider_error_body'})


@pytest.mark.parametrize('include_content', [True, False])
async def test_no_content_reaches_telemetry_through_a_fallback_refresh(include_content: bool) -> None:
    """`FallbackModel` refreshes `model_request_parameters` once it knows which model answered.

    The refresh serializes the *selected* model's parameters, so it can add instruction parts the
    outer request never had -- a prompted-output template, say. It has no access to the settings,
    which is why the span's `include_content` travels in a context variable: inferring it from what
    is already recorded reads an absent `instruction_parts` list as "content was included".
    """
    settings, exporter = redacted_setup(include_content)

    def fail(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=500, model_name='first', body='unavailable')

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('{"answer": "ok"}')])

    # No agent-level instructions, so the outer request carries no instruction parts at all; the
    # prompted output template is added by whichever model is selected.
    agent = Agent(
        FallbackModel(FunctionModel(fail), FunctionModel(respond)),
        output_type=PromptedOutput(Output, template=SECRETS['prompted_output_template'] + ' {schema}'),
        capabilities=[Instrumentation(settings=settings)],
    )
    await agent.run(SECRETS['user_prompt'])

    check(exporter, include_content, {'user_prompt', 'prompted_output_template'})


@pytest.mark.parametrize('include_content', [True, False])
async def test_no_content_reaches_telemetry_from_a_failed_fallback_attempt(include_content: bool) -> None:
    """An error `FallbackModel` fell back from gets its own span even though the run succeeds.

    The error is also kept in the answering response's `failed_attempts`, so the history a later run
    continues from carries it too.
    """
    settings, exporter = redacted_setup(include_content)

    def fail(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=500, model_name='first', body=SECRETS['provider_error_body'])

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(SECRETS['model_text'])])

    agent = Agent(
        FallbackModel(FunctionModel(fail), FunctionModel(respond)), capabilities=[Instrumentation(settings=settings)]
    )
    result = await agent.run(SECRETS['user_prompt'])
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse) and response.failed_attempts
    assert SECRETS['provider_error_body'] in (response.failed_attempts[0].error or '')
    await agent.run('again', message_history=result.all_messages())

    check(exporter, include_content, {'user_prompt', 'model_text', 'provider_error_body'})


@pytest.mark.parametrize('include_content', [True, False])
async def test_no_content_reaches_telemetry_when_request_parameters_cannot_be_serialized(
    include_content: bool,
) -> None:
    """A tool carrying arbitrary metadata, which no schema can serialize.

    `model_request_parameters` then falls back to the request's string representation, which holds
    the instructions as prose rather than in a field, so there is nothing to redact within it. The
    whole attribute has to be withheld instead -- the fallback is a supported shape, since tool
    `metadata` takes any object, not a malformed request.
    """
    settings, exporter = redacted_setup(include_content)

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    def lookup(query: str) -> str:  # pragma: no cover
        """The tool is never called; its definition alone defeats serialization."""
        return 'unused'

    agent = Agent(
        FunctionModel(respond),
        instructions=SECRETS['instructions'],
        tools=[Tool(lookup, metadata={'unserializable': object()})],
        capabilities=[Instrumentation(settings=settings)],
    )
    await agent.run(SECRETS['user_prompt'])

    # Nothing within the fallback can be redacted, so the whole attribute has to be gone -- which is
    # also what proves the run took the fallback rather than serializing cleanly.
    exported_parameters = [
        span for span in exporter.get_finished_spans() if 'model_request_parameters' in (span.attributes or {})
    ]
    assert bool(exported_parameters) is include_content

    check(exporter, include_content, {'user_prompt', 'instructions'})


def test_a_content_policy_from_another_span_is_not_trusted() -> None:
    """Only the span a policy names may be refreshed with content.

    The variable is restored with a plain `set`, which lands only in the `Context` that runs it, so
    an interrupted streamed run finalizing elsewhere can leave the opening `Context` holding the
    finished request's value. Tagging the policy with its span is what stops the next span reading
    that as permission. Driving this through an agent run cannot show it: the run's own request sets
    the policy for its span before the refresh reads it, in the same `Context`, so the guard is
    checked here where it is expressed.
    """
    settings, _ = redacted_setup(include_content=False)
    tracer = settings.tracer
    with tracer.start_as_current_span('chat') as span:
        with tracer.start_as_current_span('finished chat') as other_span:
            pass

        # What a cross-context finalization leaves behind: capture enabled, but for another span.
        include_content_ctx.set(ContentPolicy(other_span, include_content=True, tracer=tracer))
        assert span_include_content(span) is False

        # This span's own policy is honoured, in both directions.
        include_content_ctx.set(ContentPolicy(span, include_content=True, tracer=tracer))
        assert span_include_content(span) is True
        include_content_ctx.set(ContentPolicy(span, include_content=False, tracer=tracer))
        assert span_include_content(span) is False

        # No request open at all.
        include_content_ctx.set(None)
        assert span_include_content(span) is False


def test_a_span_inside_a_request_needs_the_request_still_open() -> None:
    """A span opened inside a request honours its policy while that request's span is open, in its trace.

    A decision model's `decide` span reads the policy of the `chat` span above it, with any number of spans in
    between (a durable engine's step, say), so it cannot ask for the policy's own span the way a `chat` span's
    refresh does. What it can ask still fails closed on a stale value: a finished request's span has ended, and a
    span in another trace is not inside the request at all.
    """
    settings, _ = redacted_setup(include_content=True)
    tracer = settings.tracer
    with tracer.start_as_current_span('chat') as chat_span:
        policy = ContentPolicy(chat_span, include_content=True, tracer=tracer)
        include_content_ctx.set(policy)
        with tracer.start_as_current_span('durable step'):
            assert open_request_policy() is policy

        # The current span is in another trace, so this is not inside the request.
        with tracer.start_as_current_span('elsewhere', context=Context()):
            assert open_request_policy() is None

    # What a cross-context finalization leaves behind: the policy of a request whose span has ended.
    with tracer.start_as_current_span('next request'):
        assert open_request_policy() is None

    include_content_ctx.set(None)
    assert open_request_policy() is None
