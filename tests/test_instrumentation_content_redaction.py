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

from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai._instrumentation import ContentPolicy, include_content_ctx, span_include_content
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.exceptions import ModelHTTPError, ModelRetry, ToolFailed, UnexpectedModelBehavior
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.output import PromptedOutput
from pydantic_ai.tools import Tool

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
