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
from dataclasses import dataclass
from typing import Literal

import pytest
from pydantic import BaseModel, JsonValue

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
    _instrumentation,
    native_tools,
)
from pydantic_ai._instrumentation import (
    ContentPolicy,
    include_content_ctx,
    mcp_telemetry_sensitive_ids,
    model_request_parameters_attributes,
    span_include_content,
)
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.capabilities.mcp import MCP
from pydantic_ai.capabilities.wrapper import WrapperCapability
from pydantic_ai.exceptions import ModelHTTPError, ModelRetry, UnexpectedModelBehavior
from pydantic_ai.messages import InstructionPart, NativeToolCallPart, NativeToolReturnPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import AbstractNativeTool, MCPServerTool
from pydantic_ai.output import PromptedOutput
from pydantic_ai.tools import RunContext, Tool, ToolDefinition

from .conftest import try_import

with try_import() as otel_sdk_installed:
    # `opentelemetry-sdk` arrives with the `logfire` extra, so it is not importable in the
    # `pydantic-ai-slim` / `pydantic-evals` install groups.
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

with try_import() as mcp_installed:
    from fastmcp import FastMCP

    from pydantic_ai.mcp import MCPToolset  # Preload the SDK version outside async tests.

pytestmark = [
    pytest.mark.anyio,
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
def test_mcp_server_credentials_never_reach_request_parameters_attribute(include_content: bool) -> None:
    """The telemetry field must mask secrets even when content capture is enabled."""
    tool = MCPServerTool(
        id='docs',
        url='https://mcp.example.com?access_token=SENTINEL-mcp-url-token',
        authorization_token='SENTINEL-mcp-token',
        description='Docs MCP server',
        allowed_tools=['search'],
        headers={'X-Api-Key': 'SENTINEL-mcp-key', 'Authorization': 'SENTINEL-mcp-auth'},
        optional=True,
    )
    parameters = ModelRequestParameters(native_tools=[tool])

    serialized = model_request_parameters_attributes(parameters, include_content=include_content)[
        'model_request_parameters'
    ]

    assert isinstance(serialized, str)
    assert 'SENTINEL-mcp-token' not in serialized
    assert 'SENTINEL-mcp-url-token' not in serialized
    assert 'SENTINEL-mcp-key' not in serialized
    assert 'SENTINEL-mcp-auth' not in serialized
    assert '"authorization_token":"[REDACTED]"' in serialized
    assert '"url":"[REDACTED]"' in serialized
    assert '"X-Api-Key":"[REDACTED]"' in serialized
    assert '"Authorization":"[REDACTED]"' in serialized
    assert '"description":"Docs MCP server"' in serialized
    assert '"allowed_tools":["search"]' in serialized
    assert '"optional":true' in serialized
    assert tool.authorization_token == 'SENTINEL-mcp-token'
    assert tool.url == 'https://mcp.example.com?access_token=SENTINEL-mcp-url-token'
    assert tool.headers == {'X-Api-Key': 'SENTINEL-mcp-key', 'Authorization': 'SENTINEL-mcp-auth'}


@pytest.mark.parametrize('include_content', [True, False])
def test_mcp_credentials_are_masked_when_request_serialization_uses_string_fallback(include_content: bool) -> None:
    class Unserializable:
        def __repr__(self) -> str:
            return 'SENTINEL-string-fallback'

    tool = MCPServerTool(
        id='caller-provided-id',
        url='https://mcp.example.test/private/SENTINEL-fallback-url',
        authorization_token='SENTINEL-fallback-token',
        headers={'X-Api-Key': 'SENTINEL-fallback-key'},
    )
    parameters = ModelRequestParameters(
        native_tools=[tool],
        function_tools=[ToolDefinition(name='lookup', metadata={'unserializable': Unserializable()})],
    )

    attributes = model_request_parameters_attributes(parameters, include_content=include_content)

    exported = str(attributes)
    assert 'SENTINEL-fallback-url' not in exported
    assert 'SENTINEL-fallback-token' not in exported
    assert 'SENTINEL-fallback-key' not in exported
    assert tool.id == 'caller-provided-id'
    assert tool.url == 'https://mcp.example.test/private/SENTINEL-fallback-url'
    assert tool.authorization_token == 'SENTINEL-fallback-token'
    assert tool.headers == {'X-Api-Key': 'SENTINEL-fallback-key'}
    if include_content:
        serialized = attributes['model_request_parameters']
        assert isinstance(serialized, str)
        assert 'SENTINEL-string-fallback' in serialized
        assert 'caller-provided-id' in serialized
        assert '[REDACTED]' in serialized


def test_unknown_native_tool_extensions_are_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrecognized native tool types and their extension fields remain visible."""
    monkeypatch.setattr(native_tools, 'NATIVE_TOOL_TYPES', native_tools.NATIVE_TOOL_TYPES.copy())

    @dataclass(kw_only=True)
    class FutureNativeTool(AbstractNativeTool):
        kind: str = 'future_native_tool'
        extension: str = 'SENTINEL-native-extension'

    tool = FutureNativeTool()
    parameters = ModelRequestParameters(native_tools=[tool])

    serialized = model_request_parameters_attributes(parameters)['model_request_parameters']

    assert isinstance(serialized, str)
    assert json.loads(serialized)['native_tools'][0]['extension'] == 'SENTINEL-native-extension'


@pytest.mark.parametrize('include_content', [True, False])
def test_tool_metadata_is_preserved_in_request_telemetry(include_content: bool) -> None:
    metadata: dict[str, object] = {'meta': {'authorization_token': 'SENTINEL-mcp-meta-token'}}
    tool = ToolDefinition(name='lookup', metadata=metadata)
    parameters = ModelRequestParameters(function_tools=[tool])

    serialized = model_request_parameters_attributes(parameters, include_content=include_content)[
        'model_request_parameters'
    ]

    assert isinstance(serialized, str)
    assert 'SENTINEL-mcp-meta-token' in serialized
    assert '"name":"lookup"' in serialized
    assert tool.metadata is metadata


def test_mcp_telemetry_identity_collector_walks_wrappers_and_combined_capabilities() -> None:
    capability = MCP('https://mcp.example.test/private/SENTINEL-nested-mcp-id', native=True, local=False)
    wrapper = WrapperCapability(capability)
    combined = CombinedCapability([capability, wrapper])

    assert mcp_telemetry_sensitive_ids([combined, wrapper, capability]) == {capability.id}


def test_mcp_empty_id_uses_derived_identity_and_nonempty_id_is_preserved() -> None:
    url = 'https://mcp.example.test/private/SENTINEL-empty-id'
    empty_id_capability = MCP(url, id='', native=True, local=False)
    derived_tool = next(iter(empty_id_capability.get_native_tools()))
    assert isinstance(derived_tool, MCPServerTool)
    assert derived_tool.id != ''
    derived_ids = mcp_telemetry_sensitive_ids([empty_id_capability])
    assert derived_ids == {derived_tool.id}

    derived_parameters = ModelRequestParameters(native_tools=[derived_tool], telemetry_sensitive_ids=derived_ids)
    derived_serialized = model_request_parameters_attributes(derived_parameters)['model_request_parameters']
    assert isinstance(derived_serialized, str)
    assert json.loads(derived_serialized)['native_tools'][0]['id'] == '[REDACTED]'

    explicit_capability = MCP(url, id='caller-provided-id', native=True, local=False)
    explicit_tool = next(iter(explicit_capability.get_native_tools()))
    assert isinstance(explicit_tool, MCPServerTool)
    assert mcp_telemetry_sensitive_ids([explicit_capability]) == set()
    explicit_parameters = ModelRequestParameters(
        native_tools=[explicit_tool], telemetry_sensitive_ids=mcp_telemetry_sensitive_ids([explicit_capability])
    )
    explicit_serialized = model_request_parameters_attributes(explicit_parameters)['model_request_parameters']
    assert isinstance(explicit_serialized, str)
    assert json.loads(explicit_serialized)['native_tools'][0]['id'] == 'caller-provided-id'


@pytest.mark.parametrize('include_content', [True, False])
async def test_mcp_server_url_derived_id_is_redacted_from_instrumented_model(include_content: bool) -> None:
    capability = MCP('https://mcp.example.com/private/SENTINEL-mcp-path-token', native=True, local=False)
    tool = next(iter(capability.get_native_tools()))
    assert isinstance(tool, MCPServerTool)
    parameters = ModelRequestParameters(
        native_tools=[tool], telemetry_sensitive_ids=mcp_telemetry_sensitive_ids([capability])
    )
    tool_name = f'mcp_server:{tool.id}'
    call = NativeToolCallPart(tool_name=tool_name, tool_call_id='call-1', provider_name='openai')
    returned = NativeToolReturnPart(tool_name=tool_name, tool_call_id='call-1', content={}, provider_name='openai')

    def respond(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[call, returned])

    settings, exporter = redacted_setup(include_content)
    model = InstrumentedModel(FunctionModel(respond), settings)
    await model.request([ModelRequest(parts=[UserPromptPart('go')])], None, parameters)

    assert 'SENTINEL-mcp-path-token' in tool.id  # The provider request still uses the derived ID.
    assert call.tool_name == returned.tool_name == tool_name
    spans = exporter.get_finished_spans()
    assert spans
    exported = '\n'.join(_span_text(span) for span in spans)
    assert 'SENTINEL-mcp-path-token' not in exported
    attributes = spans[0].attributes
    assert attributes is not None
    request_parameters = attributes['model_request_parameters']
    output_messages = attributes['gen_ai.output.messages']
    assert isinstance(request_parameters, str)
    assert isinstance(output_messages, str)
    assert '"id":"[REDACTED]"' in request_parameters
    assert output_messages.count('mcp_server:[REDACTED]') == 2


@pytest.mark.parametrize('serialized', [{'instruction_parts': [None]}, {'instruction_parts': 'unstructured'}])
@pytest.mark.parametrize('include_content', [True, False])
def test_malformed_instruction_shape_only_withheld_when_content_is_disabled(
    monkeypatch: pytest.MonkeyPatch, serialized: JsonValue, include_content: bool
) -> None:
    parameters = ModelRequestParameters()

    def serialize(_value: object) -> JsonValue:
        return serialized

    monkeypatch.setattr(_instrumentation, 'serialize_any', serialize)

    attributes = model_request_parameters_attributes(parameters, include_content=include_content)

    if include_content:
        exported = attributes['model_request_parameters']
        assert isinstance(exported, str)
        assert json.loads(exported) == serialized
    else:
        assert attributes == {}


@pytest.mark.skipif(not mcp_installed(), reason='fastmcp not installed')
@pytest.mark.parametrize('include_content', [True, False])
@pytest.mark.parametrize('mode', ['fallback', 'local'])
async def test_mcp_url_derived_identity_never_reaches_request_telemetry(include_content: bool, mode: str) -> None:
    class NoNative(FunctionModel):
        @classmethod
        def supported_native_tools(cls) -> frozenset[type[MCPServerTool]]:
            return frozenset()

    observed_parameters: list[ModelRequestParameters] = []

    def respond(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        observed_parameters.append(info.model_request_parameters)
        return ModelResponse(parts=[TextPart('done')])

    server = FastMCP('test', instructions='server guidance')

    @server.tool
    def ping() -> str:
        return 'pong'

    settings, exporter = redacted_setup(include_content)
    capability = MCP(
        'https://mcp.example.test/private/SENTINEL-mcp-path-token', native=mode == 'fallback', local=server
    )
    assert isinstance(capability.local, MCPToolset)
    agent = Agent(NoNative(respond), capabilities=[capability, Instrumentation(settings=settings)])
    await agent.run('go')

    chat = next(
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get('gen_ai.operation.name') == 'chat'
    )
    attributes = chat.attributes
    assert attributes is not None
    serialized = attributes['model_request_parameters']
    assert isinstance(serialized, str)
    assert 'SENTINEL-mcp-path-token' not in serialized
    assert json.loads(serialized)['function_tools']
    assert capability.id is not None and 'SENTINEL-mcp-path-token' in capability.id
    assert len(observed_parameters) == 1
    assert any(tool.capability_id == capability.id for tool in observed_parameters[0].function_tools)


@pytest.mark.parametrize('version', [1, 5])
async def test_direct_instrumented_model_preserves_provider_authored_arguments(version: Literal[1, 5]) -> None:
    capability = MCP(
        'https://mcp.example.test/private/SENTINEL-mcp-path-token',
        id='SENTINEL-caller-mcp-id',
        native=True,
        local=False,
    )
    tool = next(iter(capability.get_native_tools()))
    assert isinstance(tool, MCPServerTool)
    parameters = ModelRequestParameters(native_tools=[tool])
    tool_name = f'mcp_server:{tool.id}'
    call = NativeToolCallPart(tool_name=tool_name, tool_call_id='call-1', provider_name='openai')
    returned = NativeToolReturnPart(tool_name=tool_name, tool_call_id='call-1', content={}, provider_name='openai')

    def respond(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[call, returned, ToolCallPart('lookup', {'id': tool.id})])

    settings, exporter = redacted_setup(include_content=True)
    settings.version = version
    model = InstrumentedModel(FunctionModel(respond), settings)
    await model.request([ModelRequest(parts=[UserPromptPart('go')])], None, parameters)

    spans = exporter.get_finished_spans()
    chat = next(span for span in spans if 'model_request_parameters' in (span.attributes or {}))
    assert chat.attributes is not None
    assert tool.id in str(chat.attributes['model_request_parameters'])
    assert tool.id in '\n'.join(_span_text(span) for span in spans)
    if version == 5:  # Version 1 events omit native tool parts.
        assert tool_name in '\n'.join(_span_text(span) for span in spans)
    assert call.tool_name == returned.tool_name == tool_name


@pytest.mark.parametrize('version', [1, 5])
async def test_native_tool_capability_preserves_model_text_and_run_metadata(version: Literal[1, 5]) -> None:
    class SupportsMCP(FunctionModel):
        @classmethod
        def supported_native_tools(cls) -> frozenset[type[MCPServerTool]]:
            return frozenset({MCPServerTool})

    url = 'https://mcp.example.test/private/SENTINEL-mcp-path-token'
    tool = MCPServerTool(id='SENTINEL-caller-mcp-id', url=url)
    capability = MCP(url, native=tool, local=False)
    tool_name = f'mcp_server:{tool.id}'
    call = NativeToolCallPart(tool_name=tool_name, tool_call_id='history-call', provider_name='openai')
    returned = NativeToolReturnPart(
        tool_name=tool_name, tool_call_id='history-call', content={}, provider_name='openai'
    )

    def respond(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(f'used {tool.id}')])

    settings, exporter = redacted_setup(include_content=True)
    settings.version = version
    agent = Agent(SupportsMCP(respond), capabilities=[capability, Instrumentation(settings=settings)])
    history = [ModelRequest(parts=[UserPromptPart('previous request')]), ModelResponse(parts=[call, returned])]
    await agent.run('go', message_history=history, metadata={'mcp_id': tool.id})

    spans = exporter.get_finished_spans()
    chat = next(span for span in spans if 'model_request_parameters' in (span.attributes or {}))
    run = next(span for span in spans if span.attributes and 'metadata' in span.attributes)
    assert chat.attributes is not None
    assert run.attributes is not None
    assert tool.id in str(chat.attributes['model_request_parameters'])
    assert tool.id in str(run.attributes['metadata'])
    assert tool.id in '\n'.join(_span_text(span) for span in spans)
    assert 'SENTINEL-mcp-path-token' not in '\n'.join(_span_text(span) for span in spans)
    if version == 5:
        assert tool_name in '\n'.join(_span_text(span) for span in spans)
    assert call.tool_name == returned.tool_name == tool_name


@pytest.mark.parametrize('version', [1, 5])
def test_short_mcp_id_preserves_ordinary_content(version: Literal[1, 5]) -> None:
    capability = MCP('https://id', native=True, local=False)
    assert capability.id == 'id'
    tool = next(iter(capability.get_native_tools()))
    assert isinstance(tool, MCPServerTool)
    parameters = ModelRequestParameters(
        native_tools=[tool], telemetry_sensitive_ids=mcp_telemetry_sensitive_ids([capability])
    )
    response = ModelResponse(parts=[TextPart('The id is 42.')])
    settings, exporter = redacted_setup(include_content=True)
    settings.version = version

    with settings.tracer.start_as_current_span('chat') as span:
        span.set_attributes(model_request_parameters_attributes(parameters))
        settings.handle_messages(
            [ModelRequest(parts=[UserPromptPart('What is my id?')])], response, 'openai', span, parameters
        )

    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    serialized = attributes['model_request_parameters']
    assert isinstance(serialized, str)
    assert json.loads(serialized)['native_tools'][0]['id'] == '[REDACTED]'
    exported = str(attributes)
    assert 'What is my id?' in exported
    assert 'The id is 42.' in exported


def test_v1_response_telemetry_does_not_repeat_system_instructions_as_choice() -> None:
    settings, exporter = redacted_setup(include_content=True)
    settings.version = 1
    parameters = ModelRequestParameters(instruction_parts=[InstructionPart(content='system guidance')])

    with settings.tracer.start_as_current_span('chat') as span:
        settings.handle_messages([], ModelResponse(parts=[TextPart('done')]), 'function', span, parameters)

    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes is not None
    events = attributes['events']
    assert isinstance(events, str)
    assert events.count('gen_ai.choice') == 1


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
async def test_no_content_reaches_telemetry_when_a_tool_raises(include_content: bool) -> None:
    """A plain exception from tool code."""
    settings, exporter = redacted_setup(include_content)

    def call_tools(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('raising_tool', {})])

    agent = Agent(FunctionModel(call_tools), capabilities=[Instrumentation(settings=settings)])

    @agent.tool_plain
    def raising_tool() -> str:
        raise ValueError(SECRETS['tool_exception'])

    with pytest.raises(ValueError):
        await agent.run(SECRETS['user_prompt'])

    check(exporter, include_content, {'user_prompt', 'tool_exception'})


@pytest.mark.parametrize('include_content', [True, False])
async def test_no_content_reaches_telemetry_when_before_run_fails(include_content: bool) -> None:
    """Derived MCP labels in history are redacted even when before_run fails."""
    settings, exporter = redacted_setup(include_content)
    model = TestModel()
    capability = MCP('https://mcp.example.test/private/SENTINEL-before-run-mcp-id', native=True, local=False)
    tool = next(iter(capability.get_native_tools()))
    assert isinstance(tool, MCPServerTool)
    tool_name = f'mcp_server:{tool.id}'
    call = NativeToolCallPart(tool_name=tool_name, tool_call_id='history-call', provider_name='openai')
    returned = NativeToolReturnPart(
        tool_name=tool_name, tool_call_id='history-call', content={}, provider_name='openai'
    )

    @dataclass
    class FailingCapability(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            raise ValueError(SECRETS['tool_exception'])

    agent = Agent(
        model,
        capabilities=[capability, FailingCapability(), Instrumentation(settings=settings)],
    )
    with pytest.raises(ValueError, match=SECRETS['tool_exception']):
        await agent.run(
            SECRETS['user_prompt'],
            message_history=[
                ModelRequest(parts=[UserPromptPart('old request')]),
                ModelResponse(parts=[call, returned]),
            ],
        )

    assert model.last_model_request_parameters is None
    assert call.tool_name == returned.tool_name == tool_name
    exported = '\n'.join(_span_text(span) for span in exporter.get_finished_spans())
    assert 'SENTINEL-before-run-mcp-id' not in exported
    assert 'mcp_server:[REDACTED]' in exported
    check(exporter, include_content, {'tool_exception'})


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
async def test_request_parameter_string_fallback_respects_content_setting(include_content: bool) -> None:
    """Unsupported metadata uses the string fallback only when content capture is enabled."""
    settings, exporter = redacted_setup(include_content)

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    class Unserializable:
        def __repr__(self) -> str:
            return 'SENTINEL-string-fallback'

    def lookup(query: str) -> str:  # pragma: no cover
        return 'unused'

    agent = Agent(
        FunctionModel(respond),
        instructions=SECRETS['instructions'],
        tools=[Tool(lookup, metadata={'unserializable': Unserializable()})],
        capabilities=[Instrumentation(settings=settings)],
    )
    await agent.run(SECRETS['user_prompt'])

    exported_parameters = [
        span.attributes['model_request_parameters']
        for span in exporter.get_finished_spans()
        if span.attributes and 'model_request_parameters' in span.attributes
    ]
    if include_content:
        assert exported_parameters
        assert 'SENTINEL-string-fallback' in str(exported_parameters)
    else:
        assert 'SENTINEL-string-fallback' not in str(exported_parameters)

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
    with settings.tracer.start_as_current_span('chat') as span:
        own_span_id = span.get_span_context().span_id

        # What a cross-context finalization leaves behind: capture enabled, but for another span.
        include_content_ctx.set(ContentPolicy(span_id=own_span_id ^ 1, include_content=True))
        assert span_include_content(span) is False

        # This span's own policy is honoured, in both directions.
        include_content_ctx.set(ContentPolicy(span_id=own_span_id, include_content=True))
        assert span_include_content(span) is True
        include_content_ctx.set(ContentPolicy(span_id=own_span_id, include_content=False))
        assert span_include_content(span) is False

        # No request open at all.
        include_content_ctx.set(None)
        assert span_include_content(span) is False
