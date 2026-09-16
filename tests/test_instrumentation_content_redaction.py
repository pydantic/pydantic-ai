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
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.exceptions import ModelHTTPError, ModelRetry, UnexpectedModelBehavior
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings

from ._inline_snapshot import snapshot
from .conftest import try_import

with try_import() as otel_sdk_installed:
    # `opentelemetry-sdk` arrives with the `logfire` extra, so it is not importable in the
    # `pydantic-ai-slim` / `pydantic-evals` install groups.
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

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
    'run_metadata': 'SENTINEL-run-metadata',
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


def redacted_setup() -> tuple[InstrumentationSettings, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return InstrumentationSettings(include_content=False, tracer_provider=provider), exporter


class Output(BaseModel):
    answer: str


async def test_no_content_reaches_telemetry_on_a_successful_run() -> None:
    """Prompts, instructions, history, tool arguments and results, and the final output."""
    settings, exporter = redacted_setup()

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

    history = [
        ModelRequest(parts=[UserPromptPart(content=SECRETS['message_history'])]),
        ModelResponse(parts=[TextPart('earlier reply')]),
    ]
    result = await agent.run(SECRETS['user_prompt'], message_history=history)
    assert result.output.answer == SECRETS['final_output']
    # The tool has to have run, or the argument and return sentinels prove nothing.
    assert requests == 2

    assert leaked_channels(exporter.get_finished_spans()) == snapshot(set())


async def test_no_content_reaches_telemetry_when_a_tool_retries_to_exhaustion() -> None:
    """The retry prompt, and the error chained from it that ends the run."""
    settings, exporter = redacted_setup()

    def call_tool(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('flaky', {'value': SECRETS['tool_args']})])

    agent = Agent(FunctionModel(call_tool), retries=1, capabilities=[Instrumentation(settings=settings)])

    @agent.tool_plain
    def flaky(value: str) -> str:
        raise ModelRetry(SECRETS['tool_retry'])

    with pytest.raises(UnexpectedModelBehavior):
        await agent.run(SECRETS['user_prompt'])

    assert leaked_channels(exporter.get_finished_spans()) == snapshot(set())


async def test_no_content_reaches_telemetry_when_a_tool_raises() -> None:
    """A plain exception from tool code."""
    settings, exporter = redacted_setup()

    def call_tool(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('raising_tool', {})])

    agent = Agent(FunctionModel(call_tool), capabilities=[Instrumentation(settings=settings)])

    @agent.tool_plain
    def raising_tool() -> str:
        raise ValueError(SECRETS['tool_exception'])

    with pytest.raises(ValueError):
        await agent.run(SECRETS['user_prompt'])

    assert leaked_channels(exporter.get_finished_spans()) == snapshot(set())


async def test_no_content_reaches_telemetry_when_an_output_validator_retries() -> None:
    """A retry raised outside a tool call, which is the GHSA-3gh4-cghq-f8v4 shape."""
    settings, exporter = redacted_setup()

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(SECRETS['model_text'])])

    agent = Agent(FunctionModel(respond), retries=1, capabilities=[Instrumentation(settings=settings)])

    @agent.output_validator
    def reject(value: str) -> str:
        raise ModelRetry(SECRETS['output_validator_retry'])

    with pytest.raises(UnexpectedModelBehavior):
        await agent.run(SECRETS['user_prompt'])

    assert leaked_channels(exporter.get_finished_spans()) == snapshot(set())


async def test_no_content_reaches_telemetry_when_the_provider_errors() -> None:
    """A provider's error body, which travels in the exception message."""
    settings, exporter = redacted_setup()

    def fail(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=400, model_name='fn', body=SECRETS['provider_error_body'])

    agent = Agent(FunctionModel(fail), capabilities=[Instrumentation(settings=settings)])

    with pytest.raises(ModelHTTPError):
        await agent.run(SECRETS['user_prompt'])

    assert leaked_channels(exporter.get_finished_spans()) == snapshot(set())


async def test_no_content_reaches_telemetry_from_run_metadata() -> None:
    """Metadata a caller attaches to the run."""
    settings, exporter = redacted_setup()

    def respond(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[Instrumentation(settings=settings)])
    metadata: dict[str, Any] = {'note': SECRETS['run_metadata']}
    await agent.run(SECRETS['user_prompt'], metadata=metadata)

    assert leaked_channels(exporter.get_finished_spans()) == snapshot(set())
