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

`ENTRY_POINT_PARAMETERS` ties the sweep to every parameter of the entry points that start a run or a
realtime session. Each one names the channels a scenario drives through it, or is checked to be
recorded nowhere, or says why it needs neither, so a new parameter fails until someone decides.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import httpx2
import pytest
from opentelemetry.context import Context
from pydantic import BaseModel, TypeAdapter, ValidationInfo, field_validator

from pydantic_ai import (
    Agent,
    Conversation,
    DeferredToolRequests,
    DeferredToolResults,
    ExternalToolset,
    FunctionToolset,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    ToolApproved,
    ToolCallPart,
    ToolDenied,
    UserPromptPart,
)
from pydantic_ai._instrumentation import (
    ContentPolicy,
    MessageJsonCache,
    current_model_request_span,
    has_stale_message_json,
    include_content_ctx,
    model_request_parameters_attributes,
    open_request_policy,
    serialize_any,
    span_include_content,
)
from pydantic_ai.capabilities import (
    AbstractCapability,
    Capability,
    Hooks,
    LocalWorkspace,
    open_span,
    record_decision,
    tool_call_span,
)
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.capabilities.mcp import MCP
from pydantic_ai.direct import model_request
from pydantic_ai.exceptions import ModelHTTPError, ModelRetry, ToolFailed, UnexpectedModelBehavior
from pydantic_ai.messages import (
    DataCapabilityPart,
    InstructionCapabilityPart,
    NativeToolCallPart,
    NativeToolReturnPart,
    OperatorSource,
    OperatorText,
    ToolResultSource,
    UploadedFile,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import AbstractNativeTool, CodeExecutionTool, FileSearchTool, MCPServerTool
from pydantic_ai.output import PromptedOutput
from pydantic_ai.realtime import RealtimeModelSettings
from pydantic_ai.realtime.codec import InputTranscript, OutputTranscript, ResponseDone, ToolCall
from pydantic_ai.tools import Tool, ToolDefinition

from .conftest import try_import
from .manifests import assert_classified
from .realtime.test_session import FakeRealtimeConnection, FakeRealtimeModel

with try_import() as otel_sdk_installed:
    # `opentelemetry-sdk` arrives with the `logfire` extra, so it is not importable in the
    # `pydantic-ai-slim` / `pydantic-evals` install groups.
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

with try_import() as mcp_installed:
    from fastmcp import Client, FastMCP
    from fastmcp.client.transports import StreamableHttpTransport

    from pydantic_ai import _mcp_compat
    from pydantic_ai.mcp import MCPToolset

MCP_SDK_V2 = mcp_installed() and _mcp_compat.MCP_SDK_V2

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
    'validated_args': 'SENTINEL-validated-args',
    'tool_return': 'SENTINEL-tool-return',
    'tool_retry': 'SENTINEL-tool-retry',
    'tool_failed': 'SENTINEL-tool-failed',
    'tool_exception': 'SENTINEL-tool-exception',
    'output_validator_retry': 'SENTINEL-output-validator-retry',
    'final_output': 'SENTINEL-final-output',
    'model_text': 'SENTINEL-model-text',
    'provider_error_body': 'SENTINEL-provider-error-body',
    'prompted_output_template': 'SENTINEL-prompted-output-template',
    'capability_instruction': 'SENTINEL-capability-instruction',
    'capability_data': 'SENTINEL-capability-data',
    'capability_description': 'SENTINEL-capability-description',
    'conversation': 'SENTINEL-conversation',
    'spec': 'SENTINEL-spec',
    'toolset_instructions': 'SENTINEL-toolset-instructions',
    'approval_override_args': 'SENTINEL-approval-override-args',
    'denial_message': 'SENTINEL-denial-message',
    'external_tool_result': 'SENTINEL-external-tool-result',
    'model_settings': 'SENTINEL-model-settings',
    'deps': 'SENTINEL-deps',
    'validation_context': 'SENTINEL-validation-context',
    'helper_content': 'SENTINEL-helper-content',
    'decision_content': 'SENTINEL-decision-content',
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
            FileSearchTool(
                file_store_ids=['store'], provider_settings={'xai': {'instructions': 'search only these files'}}
            ),
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
    assert exported['native_tools'][2]['provider_settings']['xai']['instructions'] == 'search only these files'
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
            FileSearchTool(
                file_store_ids=['store'], provider_settings={'xai': {'instructions': 'search only these files'}}
            ),
        ],
    )
    expected = TypeAdapter(ModelRequestParameters).dump_python(parameters, mode='json')
    serialized = model_request_parameters_attributes(parameters)['model_request_parameters']
    assert isinstance(serialized, str)
    assert json.loads(serialized) == expected
    assert expected['function_tools'][0]['metadata'] == {'x': 1}
    assert expected['native_tools'][0]['files'][0]['file_id'] == 'file-123'
    assert expected['native_tools'][1]['provider_settings']['xai']['instructions'] == 'search only these files'


@pytest.mark.usefixtures('restore_native_tool_types')
def test_request_telemetry_serializes_a_native_tool_type_defined_after_the_first_request() -> None:
    """A native tool type defined after the first request serializes through the parameters' schema."""
    model_request_parameters_attributes(ModelRequestParameters(native_tools=[CodeExecutionTool()]))

    @dataclass(kw_only=True)
    class WidgetTool(AbstractNativeTool):
        kind: str = 'widget'
        size: int = 0

    parameters = ModelRequestParameters(native_tools=[WidgetTool(size=3)])
    serialized = model_request_parameters_attributes(parameters)['model_request_parameters']
    assert isinstance(serialized, str)
    assert json.loads(serialized) == TypeAdapter(ModelRequestParameters).dump_python(parameters, mode='json')


@pytest.mark.usefixtures('native_tool_type_without_schema')
def test_request_telemetry_serializes_by_inference_beside_a_native_tool_type_without_a_schema() -> None:
    """A registered native tool type pydantic can't build a schema for leaves the parameters serializing by inference."""
    parameters = ModelRequestParameters(native_tools=[CodeExecutionTool()])
    serialized = model_request_parameters_attributes(parameters)['model_request_parameters']
    assert isinstance(serialized, str)
    assert json.loads(serialized) == serialize_any(parameters)


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
async def test_no_content_reaches_telemetry_from_a_deferred_capability_catalog(include_content: bool) -> None:
    """A deferred capability's description, in the catalog instructions and in the request parameters' catalog."""
    settings, exporter = redacted_setup(include_content)

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(SECRETS['model_text'])])

    capability = Capability[object](
        id='secret', description=lambda: SECRETS['capability_description'], defer_loading=True
    )
    agent = Agent(FunctionModel(respond), capabilities=[capability, Instrumentation(settings=settings)])
    await agent.run(SECRETS['user_prompt'])

    check(exporter, include_content, {'user_prompt', 'model_text', 'capability_description'})


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
async def test_no_content_reaches_telemetry_from_a_conversation(include_content: bool) -> None:
    """The messages of a `Conversation` the caller continues, in place of `message_history`."""
    settings, exporter = redacted_setup(include_content)

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(SECRETS['model_text'])])

    agent = Agent(FunctionModel(respond), capabilities=[Instrumentation(settings=settings)])
    conversation = Conversation(
        messages=[
            ModelRequest(parts=[UserPromptPart(content=SECRETS['conversation'])]),
            ModelResponse(parts=[TextPart('earlier reply')]),
        ]
    )
    await agent.run(SECRETS['user_prompt'], conversation=conversation)

    check(exporter, include_content, {'user_prompt', 'conversation', 'model_text'})


@pytest.mark.parametrize('include_content', [True, False])
async def test_no_content_reaches_telemetry_from_what_a_run_adds(include_content: bool) -> None:
    """Instructions a run adds to the agent's: its own, an agent spec's, and a toolset's."""
    settings, exporter = redacted_setup(include_content)

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(SECRETS['model_text'])])

    agent = Agent(FunctionModel(respond), capabilities=[Instrumentation(settings=settings)])
    await agent.run(
        SECRETS['user_prompt'],
        instructions=SECRETS['instructions'],
        spec={'instructions': SECRETS['spec']},
        toolsets=[FunctionToolset(instructions=SECRETS['toolset_instructions'])],
    )

    check(exporter, include_content, {'user_prompt', 'instructions', 'spec', 'toolset_instructions', 'model_text'})


class AddAccount(AbstractCapability[Any]):
    """A validation hook that adds a value the model never sent, so only the deferred call's `validated_args` holds it."""

    async def after_tool_validate(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> dict[str, Any]:
        return {**args, 'account': SECRETS['validated_args']}


@pytest.mark.parametrize('include_content', [True, False])
async def test_no_content_reaches_telemetry_from_a_deferred_calls_validated_args(include_content: bool) -> None:
    """The run's `DeferredToolRequests` output carries the arguments as a validation hook left them."""
    settings, exporter = redacted_setup(include_content)

    def call_tool(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('transfer', {'amount': 1})])

    agent = Agent(
        FunctionModel(call_tool),
        output_type=[str, DeferredToolRequests],
        capabilities=[AddAccount(), Instrumentation(settings=settings)],
    )

    @agent.tool_plain(requires_approval=True)
    def transfer(amount: int) -> str:  # pragma: no cover
        # Never approved: the run ends on the approval request.
        return str(amount)

    result = await agent.run(SECRETS['user_prompt'])
    assert isinstance(result.output, DeferredToolRequests)

    check(exporter, include_content, {'user_prompt', 'validated_args'})


@pytest.mark.parametrize('include_content', [True, False])
async def test_no_content_reaches_telemetry_from_deferred_tool_results(include_content: bool) -> None:
    """What the caller answers deferred calls with: an approval's arguments, a denial's message, an external result."""
    settings, exporter = redacted_setup(include_content)

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('transfer', {'account': 'model-chosen'}, tool_call_id='approved'),
                    ToolCallPart('delete', {}, tool_call_id='denied'),
                    ToolCallPart('lookup', {}, tool_call_id='external'),
                ]
            )
        return ModelResponse(parts=[TextPart('done')])

    transferred_to: list[str] = []

    def transfer(account: str) -> str:
        transferred_to.append(account)
        return SECRETS['tool_return']

    def delete() -> str:  # pragma: no cover
        # Denied: the call never runs.
        return 'deleted'

    agent = Agent(
        FunctionModel(respond),
        output_type=[str, DeferredToolRequests],
        tools=[Tool(transfer, requires_approval=True), Tool(delete, requires_approval=True)],
        toolsets=[ExternalToolset([ToolDefinition(name='lookup')])],
        capabilities=[Instrumentation(settings=settings)],
    )
    paused = await agent.run(SECRETS['user_prompt'])
    assert isinstance(paused.output, DeferredToolRequests)
    result = await agent.run(
        message_history=paused.all_messages(),
        deferred_tool_results=DeferredToolResults(
            approvals={
                'approved': ToolApproved(override_args={'account': SECRETS['approval_override_args']}),
                'denied': ToolDenied(SECRETS['denial_message']),
            },
            calls={'external': SECRETS['external_tool_result']},
        ),
    )
    assert result.output == 'done'
    # The approved call has to have run with the overriding arguments, or their sentinel proves nothing.
    assert transferred_to == [SECRETS['approval_override_args']]

    check(
        exporter,
        include_content,
        {'user_prompt', 'approval_override_args', 'denial_message', 'external_tool_result', 'tool_return'},
    )


@dataclass(kw_only=True)
class _RedactedNudgePart(InstructionCapabilityPart, namespace='test_redaction', name='nudge'):
    pass


@dataclass(kw_only=True)
class _RedactedPagePart(DataCapabilityPart, namespace='test_redaction', name='page', tag='fetched_page'):
    pass


@dataclass
class _CapabilityPartReporter(AbstractCapability[object]):
    """Enqueues an operator-voice part, which renders on the system role, and a data part framed on the user channel.

    The run it's in makes one request.
    """

    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        ctx.enqueue(
            _RedactedNudgePart(content=OperatorText(SECRETS['capability_instruction']), source=OperatorSource()),
            _RedactedPagePart(content=SECRETS['capability_data'], source=ToolResultSource(tool_name='fetch')),
        )
        return request_context


@pytest.mark.parametrize('include_content', [True, False])
async def test_no_content_reaches_telemetry_from_capability_parts(include_content: bool) -> None:
    """What a capability enqueues, on the system role in the application's voice and framed on the user channel."""
    settings, exporter = redacted_setup(include_content)
    received: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        received.append(messages)
        return ModelResponse(parts=[TextPart(SECRETS['model_text'])])

    agent = Agent(FunctionModel(respond), capabilities=[_CapabilityPartReporter(), Instrumentation(settings=settings)])
    await agent.run(SECRETS['user_prompt'])
    # Both parts have to have reached the model, or their sentinels prove nothing.
    assert SECRETS['capability_instruction'] in str(received) and SECRETS['capability_data'] in str(received)

    check(exporter, include_content, {'user_prompt', 'capability_instruction', 'capability_data', 'model_text'})


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


class _FakeRealtimeSettings(RealtimeModelSettings, total=False):
    """Provider realtime settings with a field for prompt text, as `openai_live_instructions` is."""

    fake_instructions: str


@pytest.mark.parametrize('include_content', [True, False])
@pytest.mark.parametrize('seed', ['message_history', 'conversation'])
async def test_no_content_reaches_telemetry_from_a_realtime_session(
    include_content: bool, seed: Literal['message_history', 'conversation']
) -> None:
    """A realtime session's instructions, settings, toolsets, capabilities and seeded history, and what is said in it.

    The session records the settings it connected with on the session span and on every `chat` span.
    """
    settings, exporter = redacted_setup(include_content)

    def lookup(query: str) -> str:
        return SECRETS['tool_return']

    connection = FakeRealtimeConnection(
        [
            InputTranscript(text=SECRETS['user_prompt'], is_final=True),
            ToolCall(tool_call_id='call-1', tool_name='lookup', args=json.dumps({'query': SECRETS['tool_args']})),
            OutputTranscript(text=SECRETS['model_text'], is_final=True),
            ResponseDone(),
        ]
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=SECRETS[seed])]),
        ModelResponse(parts=[TextPart('earlier reply')]),
    ]
    agent = Agent(capabilities=[Instrumentation(settings=settings)])
    realtime = agent.realtime(
        FakeRealtimeModel(connection),
        instructions=SECRETS['instructions'],
        model_settings=_FakeRealtimeSettings(fake_instructions=SECRETS['model_settings']),
        toolsets=[FunctionToolset([lookup], instructions=SECRETS['toolset_instructions'])],
        capabilities=[Capability(instructions=SECRETS['capability_instruction'])],
        message_history=history if seed == 'message_history' else None,
        conversation=Conversation(messages=history) if seed == 'conversation' else None,
    )
    async with realtime.session() as session:
        _ = [event async for event in session]

    check(
        exporter,
        include_content,
        {
            'user_prompt',
            'tool_args',
            'tool_return',
            'model_text',
            'instructions',
            'model_settings',
            'toolset_instructions',
            'capability_instruction',
            seed,
        },
    )


@pytest.mark.parametrize('include_content', [True, False])
async def test_no_content_reaches_telemetry_from_a_lowered_capabilitys_instructions(include_content: bool) -> None:
    """A lowered capability's instructions reach the model as framed data, and with capture off no span carries them."""
    settings, exporter = redacted_setup(include_content=include_content)
    received: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received.append(str(messages))
        return ModelResponse(parts=[TextPart(SECRETS['model_text'])])

    lowered = Capability[None](id='sentinel_lowered_block', instructions=SECRETS['capability_instruction'])
    agent = Agent(
        FunctionModel(respond),
        deps_type=type(None),
        capabilities=[lowered.with_authority('none'), Instrumentation(settings=settings)],
    )
    await agent.run(SECRETS['user_prompt'])

    assert SECRETS['capability_instruction'] in received[-1]
    check(exporter, include_content, {'user_prompt', 'capability_instruction', 'model_text'})
    exported = '\n'.join(_span_text(span) for span in exporter.get_finished_spans())
    assert ('capability:sentinel_lowered_block' in exported) is include_content


def _record_with_a_context(ctx: RunContext[None]) -> str:
    """What a tool records with the `ctx` helpers, including on its tool call's span."""
    with open_span(ctx, 'example span', content={'example.first': SECRETS['helper_content']}):
        pass
    if ctx.trace_include_content:
        tool_call_span(ctx).set_attribute('example.later', SECRETS['helper_content'])
    record_decision(ctx, 'example decision', content={'example.reason': SECRETS['decision_content']})
    return 'done'


@pytest.mark.parametrize('include_content', [True, False])
@pytest.mark.parametrize('mode', ['run', 'realtime'])
async def test_no_content_reaches_telemetry_from_the_ctx_helpers(
    include_content: bool, mode: Literal['run', 'realtime']
) -> None:
    """The run's content setting reaches the `ctx` helpers, in a run and in a realtime session."""
    settings, exporter = redacted_setup(include_content)

    def lookup(ctx: RunContext[None]) -> str:
        return _record_with_a_context(ctx)

    if mode == 'run':
        agent = Agent(
            TestModel(call_tools=['lookup']), tools=[lookup], capabilities=[Instrumentation(settings=settings)]
        )
        await agent.run('go')
    else:
        connection = FakeRealtimeConnection(
            [ToolCall(tool_call_id='call-1', tool_name='lookup', args='{}'), ResponseDone()]
        )
        realtime = Agent(capabilities=[Instrumentation(settings=settings)]).realtime(
            FakeRealtimeModel(connection), toolsets=[FunctionToolset([lookup])]
        )
        async with realtime.session() as session:
            _ = [event async for event in session]

    check(exporter, include_content, {'helper_content', 'decision_content'})
    # Each channel on its own span, so one that stops recording can't hide behind another that still does.
    attributes = {span.name: dict(span.attributes or {}) for span in exporter.get_finished_spans()}
    helper_content = SECRETS['helper_content'] if include_content else None
    assert attributes['example span'].get('example.first') == helper_content
    assert attributes['execute_tool lookup'].get('example.later') == helper_content
    decision_content = SECRETS['decision_content'] if include_content else None
    assert attributes['example decision'].get('example.reason') == decision_content


@dataclass(frozen=True)
class Swept:
    """A parameter a scenario in this module drives a sentinel through, so the sweep covers it."""

    channels: tuple[str, ...]
    """The `SECRETS` channels that scenario drives through the parameter."""

    reason: str


@dataclass(frozen=True)
class NotExported:
    """A parameter that can carry text, which no span records even with content capture on.

    `probe` runs an agent with a sentinel in the parameter, and checks the sentinel reached the code that reads it.
    `test_a_parameter_no_span_records_exports_nothing_with_content_on` runs it with content capture on.
    """

    probe: Callable[[InstrumentationSettings], Awaitable[None]]
    reason: str


@dataclass(frozen=True)
class ExportedByDesign:
    """An identity or correlation value that spans record whatever the content setting, as the cited doc or test says."""

    reason: str


@dataclass(frozen=True)
class NotContent:
    """A parameter that can't hold user text."""

    reason: str


async def _probe_run_model_settings(settings: InstrumentationSettings) -> None:
    received: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received.append(str(info.model_settings))
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[Instrumentation(settings=settings)])
    await agent.run('go', model_settings={'extra_body': {'note': SECRETS['model_settings']}})
    assert SECRETS['model_settings'] in received[0]


class _ContextValidatedOutput(BaseModel):
    answer: str

    @field_validator('answer')
    @classmethod
    def requires_the_validation_context(cls, answer: str, info: ValidationInfo) -> str:
        if info.context != SECRETS['validation_context']:
            raise ValueError('the validation context did not reach the output validator')  # pragma: no cover
        return answer


async def _probe_run_validation_context(settings: InstrumentationSettings) -> None:
    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('final_result', {'answer': 'done'})])

    agent = Agent(
        FunctionModel(respond),
        output_type=_ContextValidatedOutput,
        validation_context=SECRETS['validation_context'],
        capabilities=[Instrumentation(settings=settings)],
    )
    result = await agent.run('go')
    assert result.output.answer == 'done'


async def _probe_run_deps(settings: InstrumentationSettings) -> None:
    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('read_deps', {})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), deps_type=str, capabilities=[Instrumentation(settings=settings)])
    read: list[str] = []

    @agent.tool
    def read_deps(ctx: RunContext[str]) -> str:
        read.append(ctx.deps)
        return 'read'

    await agent.run('go', deps=SECRETS['deps'])
    assert read == [SECRETS['deps']]


async def _probe_realtime_deps(settings: InstrumentationSettings) -> None:
    agent = Agent(deps_type=str, capabilities=[Instrumentation(settings=settings)])
    read: list[str] = []

    @agent.tool
    def read_deps(ctx: RunContext[str]) -> str:
        read.append(ctx.deps)
        return 'read'

    connection = FakeRealtimeConnection(
        [ToolCall(tool_call_id='call-1', tool_name='read_deps', args='{}'), ResponseDone()]
    )
    async with agent.realtime(FakeRealtimeModel(connection), deps=SECRETS['deps']).session() as session:
        _ = [event async for event in session]
    assert read == [SECRETS['deps']]


EntryPointGroup = Literal['run', 'realtime']

ENTRY_POINTS: dict[EntryPointGroup, tuple[Callable[..., object], ...]] = {
    'run': (
        Agent.__init__,
        Agent.run,
        Agent.run_sync,
        Agent.iter,
        Agent.run_stream,
        Agent.run_stream_sync,
        Agent.run_stream_events,
    ),
    'realtime': (Agent.realtime,),
}
"""The entry points whose parameters `ENTRY_POINT_PARAMETERS` classifies, by the group that keys them."""

_TOOL_DEFINITIONS_BY_DESIGN = (
    'Tool names, descriptions and schemas go into `gen_ai.tool.definitions`, which `build_tool_definitions` in '
    '`pydantic_ai/_instrumentation.py` records whatever the content setting.'
)

ENTRY_POINT_PARAMETERS: dict[tuple[EntryPointGroup, str], Swept | NotExported | ExportedByDesign | NotContent] = {
    ('run', 'model'): ExportedByDesign(
        "The model's name is the run span's `model_name` and each request span's `gen_ai.request.model`, per "
        '`tests/test_logfire.py::test_logfire`.'
    ),
    ('run', 'output_type'): Swept(
        channels=('prompted_output_template',),
        reason='A `PromptedOutput` template, in `test_no_content_reaches_telemetry_through_a_fallback_refresh`. '
        f'Output tools are tools: {_TOOL_DEFINITIONS_BY_DESIGN}',
    ),
    ('run', 'instructions'): Swept(
        channels=('instructions',),
        reason="The agent's in `test_no_content_reaches_telemetry_on_a_successful_run`, a run's in "
        '`test_no_content_reaches_telemetry_from_what_a_run_adds`.',
    ),
    ('run', 'system_prompt'): Swept(
        channels=('system_prompt',), reason='In `test_no_content_reaches_telemetry_on_a_successful_run`.'
    ),
    ('run', 'deps_type'): NotContent('A type, which only checks `deps`.'),
    ('run', 'name'): ExportedByDesign(
        'Names the run span and is its `gen_ai.agent.name`, per `tests/test_logfire.py::test_logfire`.'
    ),
    ('run', 'description'): ExportedByDesign(
        "The run span's `gen_ai.agent.description`, per `tests/test_logfire.py::test_agent_description`; a "
        '`TemplateStr` renders it from `deps`, per `tests/test_logfire.py::test_instrumentation_capability_template_description`.'
    ),
    ('run', 'model_settings'): NotExported(
        _probe_run_model_settings,
        'Only numeric OTel-spec settings become `gen_ai.request.*` attributes (`model_settings_attributes`); text '
        'such as `extra_body` is recorded nowhere. The probe sends the sentinel in `extra_body`.',
    ),
    ('run', 'retries'): NotContent('Retry budgets: an int or `AgentRetries` counts.'),
    ('run', 'validation_context'): NotExported(
        _probe_run_validation_context,
        'Handed to Pydantic validators, which no span records. The probe has an output validator require the sentinel.',
    ),
    ('run', 'tools'): Swept(
        channels=('tool_return',),
        reason='A tool passed in `tools=` returns it in `test_no_content_reaches_telemetry_from_deferred_tool_results`; '
        'the other tool channels go through the same function toolset via `@agent.tool_plain`. '
        f'{_TOOL_DEFINITIONS_BY_DESIGN}',
    ),
    ('run', 'toolsets'): Swept(
        channels=('toolset_instructions',),
        reason=f"A toolset's instructions, in `test_no_content_reaches_telemetry_from_what_a_run_adds`. {_TOOL_DEFINITIONS_BY_DESIGN}",
    ),
    ('run', 'defer_model_check'): NotContent('A bool.'),
    ('run', 'end_strategy'): NotContent('A `Literal` saying when the run stops.'),
    ('run', 'metadata'): ExportedByDesign(
        "The module docstring says why; the run span's `metadata`, per "
        '`tests/test_logfire.py::test_logfire_metadata_values`.'
    ),
    ('run', 'tool_timeout'): NotContent('Seconds.'),
    ('run', 'max_concurrency'): NotContent('A concurrency limit.'),
    ('run', 'capabilities'): Swept(
        channels=('capability_instruction', 'capability_data', 'capability_description'),
        reason='In `test_no_content_reaches_telemetry_from_capability_parts` and '
        '`test_no_content_reaches_telemetry_from_a_deferred_capability_catalog`.',
    ),
    ('run', 'user_prompt'): Swept(channels=('user_prompt',), reason='In every scenario.'),
    ('run', 'conversation'): Swept(
        channels=('conversation',),
        reason='Its messages, in `test_no_content_reaches_telemetry_from_a_conversation`. Its `conversation_id` '
        'is exported by design, like the `conversation_id` parameter.',
    ),
    ('run', 'message_history'): Swept(
        channels=('message_history',), reason='In `test_no_content_reaches_telemetry_from_message_history`.'
    ),
    ('run', 'deferred_tool_results'): Swept(
        channels=('approval_override_args', 'denial_message', 'external_tool_result'),
        reason='In `test_no_content_reaches_telemetry_from_deferred_tool_results`.',
    ),
    ('run', 'conversation_id'): ExportedByDesign(
        "The run span's `gen_ai.conversation.id`, per `docs/message-history.md`."
    ),
    ('run', 'run_id'): ExportedByDesign(
        "The run span's `gen_ai.agent.call.id`, per `tests/test_logfire.py::test_logfire_explicit_run_id`."
    ),
    ('run', 'deps'): NotExported(
        _probe_run_deps,
        'Handed to tools, instructions and hooks, which no span records it from; a `TemplateStr` `description` '
        'rendered from it is the description, exported by design. The probe has a tool read the sentinel.',
    ),
    ('run', 'usage_limits'): NotContent('Request and token counts.'),
    ('run', 'cancellation_token'): NotContent('A handle to cancel the run.'),
    ('run', 'timeout'): NotContent('Seconds.'),
    ('run', 'usage'): NotContent('Token and request counts to start from.'),
    ('run', 'infer_name'): NotContent('A bool.'),
    ('run', 'event_stream_handler'): NotContent(
        'A callable the run hands its events to: it receives content rather than supplying any.'
    ),
    ('run', 'workspace'): ExportedByDesign(
        "Its provider and id are the run span's `pydantic_ai.workspace.*`, per "
        '`test_workspace_identity_is_recorded_without_content`.'
    ),
    ('run', 'spec'): Swept(
        channels=('spec',),
        reason='Its instructions, in `test_no_content_reaches_telemetry_from_what_a_run_adds`. The rest of a spec '
        "sets the agent's other parameters, classified here under their own names.",
    ),
    ('realtime', 'model'): ExportedByDesign(
        "The model's name is the session span's `model_name`, per "
        '`tests/realtime/test_instrumentation.py::test_session_and_tool_spans_with_usage`.'
    ),
    ('realtime', 'deps'): NotExported(
        _probe_realtime_deps, 'Handed to tools, which no span records it from. The probe has a tool read the sentinel.'
    ),
    ('realtime', 'model_settings'): Swept(
        channels=('model_settings',),
        reason='A provider setting holding prompt text, in `test_no_content_reaches_telemetry_from_a_realtime_session`.',
    ),
    ('realtime', 'instructions'): Swept(
        channels=('instructions',), reason='In `test_no_content_reaches_telemetry_from_a_realtime_session`.'
    ),
    ('realtime', 'toolsets'): Swept(
        channels=('toolset_instructions', 'tool_args', 'tool_return'),
        reason="A toolset's instructions, and a call's arguments and result, in "
        f'`test_no_content_reaches_telemetry_from_a_realtime_session`. {_TOOL_DEFINITIONS_BY_DESIGN}',
    ),
    ('realtime', 'capabilities'): Swept(
        channels=('capability_instruction',),
        reason="A capability's instructions, in `test_no_content_reaches_telemetry_from_a_realtime_session`.",
    ),
    ('realtime', 'usage'): NotContent('Token and request counts to start from.'),
    ('realtime', 'usage_limits'): NotContent('Request and token counts.'),
    ('realtime', 'metadata'): ExportedByDesign(
        "The session span's `metadata`, with content capture off too, per "
        '`tests/realtime/test_instrumentation.py::test_include_content_false_redacts_transcript_messages`.'
    ),
    ('realtime', 'conversation_id'): ExportedByDesign(
        "The session span's `gen_ai.conversation.id`, per "
        '`tests/realtime/test_instrumentation.py::test_session_span_sets_conversation_id`.'
    ),
    ('realtime', 'run_id'): ExportedByDesign(
        "The session span's `gen_ai.agent.call.id`, per "
        '`tests/realtime/test_instrumentation.py::test_session_and_tool_spans_with_usage`.'
    ),
    ('realtime', 'conversation'): Swept(
        channels=('conversation',),
        reason='Its messages seed the session, in `test_no_content_reaches_telemetry_from_a_realtime_session`.',
    ),
    ('realtime', 'message_history'): Swept(
        channels=('message_history',),
        reason='Seeds the session, in `test_no_content_reaches_telemetry_from_a_realtime_session`.',
    ),
}
"""How content each entry-point parameter carries is kept out of telemetry, keyed by entry-point group and name."""


@pytest.mark.parametrize('group', list(ENTRY_POINTS))
def test_every_entry_point_parameter_is_classified(group: EntryPointGroup) -> None:
    parameters: set[tuple[EntryPointGroup, str]] = {
        (group, name)
        for entry_point in ENTRY_POINTS[group]
        for name in inspect.signature(entry_point).parameters
        if name != 'self'
    }
    table = {key: entry for key, entry in ENTRY_POINT_PARAMETERS.items() if key[0] == group}
    assert_classified(parameters, table, complete_walk=True, table_name='ENTRY_POINT_PARAMETERS')
    unknown = {
        name: entry.channels
        for (_, name), entry in table.items()
        if isinstance(entry, Swept) and not set(entry.channels) <= SECRETS.keys()
    }
    assert not unknown, f'`Swept` entries name channels `SECRETS` lacks: {unknown}'


@pytest.mark.parametrize(
    'entry',
    [
        pytest.param(entry, id=f'{group}.{name}')
        for (group, name), entry in ENTRY_POINT_PARAMETERS.items()
        if isinstance(entry, NotExported)
    ],
)
async def test_a_parameter_no_span_records_exports_nothing_with_content_on(entry: NotExported) -> None:
    settings, exporter = redacted_setup(include_content=True)
    await entry.probe(settings)
    spans = exporter.get_finished_spans()
    assert spans, 'no spans exported: the probe proves nothing about redaction'
    leaked = leaked_channels(spans)
    assert not leaked, f'{sorted(leaked)} reached telemetry: classify the parameter as `Swept` and sweep it'


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


def test_current_model_request_span_requires_the_owned_span() -> None:
    """A model records request attributes on the open `chat` span only while it's the current span.

    Not on a span opened inside the request, which is the current span instead, nor on a finished request's span.
    """
    settings, _ = redacted_setup(include_content=False)
    tracer = settings.tracer
    previous = include_content_ctx.get()
    try:
        include_content_ctx.set(None)
        assert current_model_request_span() is None
        with tracer.start_as_current_span('chat') as chat_span:
            include_content_ctx.set(ContentPolicy(chat_span, include_content=False, tracer=tracer))
            assert current_model_request_span() is chat_span
            with tracer.start_as_current_span('nested span'):
                assert current_model_request_span() is None
        assert current_model_request_span() is None
    finally:
        include_content_ctx.set(previous)


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
