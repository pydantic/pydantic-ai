"""Tests for the `OutputGuard` capability."""

from __future__ import annotations

import asyncio

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NoOpTracer, Tracer
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.capabilities import CapabilityOrdering, Instrumentation
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness import GuardResult, OutputBlocked, OutputGuard
from pydantic_ai_harness.guardrails import GuardrailError

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _recording_tracer() -> tuple[Tracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer('test'), exporter


def _run_ctx(
    *,
    partial_output: bool = False,
    trace_include_content: bool = False,
    tracer: Tracer | None = None,
) -> RunContext[object]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        partial_output=partial_output,
        trace_include_content=trace_include_content,
        tracer=tracer if tracer is not None else NoOpTracer(),
    )


_TEXT_OUTPUT_CONTEXT = OutputContext(mode='text', output_type=str, object_def=None, has_function=False)


def _only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f'expected exactly one span, got {[s.name for s in spans]}'
    return spans[0]


class TestOutputGuard:
    """Integration tests for the `OutputGuard` capability driven through `Agent.run`."""

    async def test_allows_safe_output(self):
        agent = Agent(
            TestModel(custom_output_text='harmless reply'),
            capabilities=[OutputGuard(guard=lambda out: 'SSN' not in str(out))],
        )
        assert (await agent.run('hello')).output == 'harmless reply'

    async def test_guard_result_allow(self):
        agent = Agent(
            TestModel(custom_output_text='harmless reply'),
            capabilities=[OutputGuard(guard=lambda _: GuardResult.allow())],
        )
        assert (await agent.run('hello')).output == 'harmless reply'

    async def test_blocks_with_default_message(self):
        agent = Agent(
            TestModel(custom_output_text='leaks SSN 123-45-6789'),
            capabilities=[OutputGuard(guard=lambda out: 'SSN' not in str(out))],
        )
        with pytest.raises(OutputBlocked, match='Output blocked by output guardrail.'):
            await agent.run('hello')

    async def test_blocks_with_custom_message(self):
        agent = Agent(
            TestModel(custom_output_text='leaks SSN'),
            capabilities=[OutputGuard(guard=lambda _: GuardResult.block('contains SSN'))],
        )
        with pytest.raises(OutputBlocked, match='contains SSN'):
            await agent.run('hello')

    async def test_replace_substitutes_output(self):
        def guard(output: object) -> GuardResult:
            return GuardResult.replace(str(output).replace('SSN', '[redacted]'))

        agent = Agent(
            TestModel(custom_output_text='leaks SSN here'),
            capabilities=[OutputGuard(guard=guard)],
        )
        result = await agent.run('hello')
        assert result.output == 'leaks [redacted] here'

    async def test_retry_triggers_model_retry(self):
        attempts: list[object] = []

        def guard(output: object) -> GuardResult:
            attempts.append(output)
            if len(attempts) == 1:
                return GuardResult.retry('Try again without personal data.')
            return GuardResult.allow()

        agent = Agent(TestModel(custom_output_text='answer'), capabilities=[OutputGuard(guard=guard)])
        result = await agent.run('hello')

        assert result.output == 'answer'
        assert len(attempts) == 2

    async def test_async_guard_awaited(self):
        async def guard(output: object) -> bool:
            await asyncio.sleep(0)
            return 'bad' not in str(output)

        agent = Agent(TestModel(custom_output_text='ok reply'), capabilities=[OutputGuard(guard=guard)])
        assert (await agent.run('prompt')).output == 'ok reply'

        agent_bad = Agent(TestModel(custom_output_text='bad reply'), capabilities=[OutputGuard(guard=guard)])
        with pytest.raises(OutputBlocked):
            await agent_bad.run('prompt')

    async def test_raising_propagates(self):
        def guard(_: object) -> bool:
            raise RuntimeError('guard exploded')

        agent = Agent(TestModel(custom_output_text='anything'), capabilities=[OutputGuard(guard=guard)])
        with pytest.raises(RuntimeError, match='guard exploded'):
            await agent.run('hello')

    async def test_guard_receives_run_context(self):
        seen: list[object] = []

        def guard(ctx: RunContext[object], output: object) -> bool:
            seen.append(ctx.prompt)
            return 'SSN' not in str(output)

        agent = Agent(TestModel(custom_output_text='harmless reply'), capabilities=[OutputGuard(guard=guard)])
        result = await agent.run('hello')
        assert result.output == 'harmless reply'
        assert seen == ['hello']

    async def test_receives_structured_output_unchanged(self):
        """For typed outputs the guard gets the model instance, not a stringified form."""

        class Answer(BaseModel):
            reply: str
            internal_url: str

        seen: list[object] = []

        def guard(output: object) -> bool:
            seen.append(output)
            assert isinstance(output, Answer)
            return 'internal.example.com' not in output.internal_url

        agent = Agent(
            TestModel(custom_output_args={'reply': 'hi', 'internal_url': 'https://public.example.com/x'}),
            output_type=Answer,
            capabilities=[OutputGuard(guard=guard)],
        )
        result = await agent.run('hello')
        assert isinstance(result.output, Answer)
        assert seen == [result.output]

        agent_bad = Agent(
            TestModel(custom_output_args={'reply': 'hi', 'internal_url': 'https://internal.example.com/x'}),
            output_type=Answer,
            capabilities=[OutputGuard(guard=guard)],
        )
        with pytest.raises(OutputBlocked):
            await agent_bad.run('hello')

    def test_output_blocked_is_guardrail_error(self):
        assert issubclass(OutputBlocked, GuardrailError)


class TestOutputGuardDirect:
    """Direct `after_output_process` tests for lifecycle behaviour hard to isolate via `Agent`."""

    async def test_partial_output_is_skipped(self):
        called: list[object] = []

        def guard(output: object) -> bool:  # pragma: no cover - must not run on partial output
            called.append(output)
            return False

        og = OutputGuard(guard=guard)
        out = await og.after_output_process(
            _run_ctx(partial_output=True), output_context=_TEXT_OUTPUT_CONTEXT, output='partial'
        )
        assert out == 'partial'
        assert called == []


class TestOutputGuardStreaming:
    """`OutputGuard` under `run_stream()`."""

    async def test_allows_under_run_stream(self):
        agent = Agent(
            TestModel(custom_output_text='clean reply'),
            capabilities=[OutputGuard(guard=lambda out: 'SSN' not in str(out))],
        )
        async with agent.run_stream('hello') as result:
            assert (await result.get_output()) == 'clean reply'

    async def test_block_under_run_stream(self):
        agent = Agent(
            TestModel(custom_output_text='leaks SSN'),
            capabilities=[OutputGuard(guard=lambda _: GuardResult.block('contains SSN'))],
        )
        with pytest.raises(OutputBlocked, match='contains SSN'):
            async with agent.run_stream('hello') as result:
                await result.get_output()

    async def test_retry_under_run_stream_is_unsupported(self):
        """pydantic-ai does not support output retries while streaming."""
        agent = Agent(
            TestModel(custom_output_text='answer'),
            capabilities=[OutputGuard(guard=lambda _: GuardResult.retry('redo'))],
        )
        with pytest.raises(UnexpectedModelBehavior):
            async with agent.run_stream('hello') as result:
                await result.get_output()


class TestOutputGuardTracing:
    """Spans emitted on block and redaction."""

    async def test_block_span_includes_message_when_enabled(self):
        tracer, exporter = _recording_tracer()
        og = OutputGuard(guard=lambda _: GuardResult.block('contains SSN'))

        with pytest.raises(OutputBlocked):
            await og.after_output_process(
                _run_ctx(tracer=tracer, trace_include_content=True),
                output_context=_TEXT_OUTPUT_CONTEXT,
                output='leaks SSN',
            )

        span = _only_span(exporter)
        assert span.name == 'guardrail blocked output'
        assert dict(span.attributes or {}) == {
            'guardrail.direction': 'output',
            'guardrail.action': 'block',
            'guardrail.message': 'contains SSN',
        }

    async def test_block_span_omits_message_by_default(self):
        """The refusal message may quote sensitive output, so it is gated like redactions are."""
        tracer, exporter = _recording_tracer()
        og = OutputGuard(guard=lambda _: GuardResult.block('contains SSN'))

        with pytest.raises(OutputBlocked):
            await og.after_output_process(
                _run_ctx(tracer=tracer), output_context=_TEXT_OUTPUT_CONTEXT, output='leaks SSN'
            )

        span = _only_span(exporter)
        assert span.name == 'guardrail blocked output'
        assert dict(span.attributes or {}) == {'guardrail.direction': 'output', 'guardrail.action': 'block'}


class TestOutputGuardOrdering:
    """`get_ordering()` placement: outermost but wrapped by `Instrumentation`."""

    def test_declares_outermost_inside_instrumentation(self):
        ordering = OutputGuard(guard=lambda _: True).get_ordering()
        assert ordering == CapabilityOrdering(position='outermost', wrapped_by=[Instrumentation])

    async def test_redaction_span_includes_content_when_enabled(self):
        tracer, exporter = _recording_tracer()
        og = OutputGuard(guard=lambda _: GuardResult.replace('clean'))

        out = await og.after_output_process(
            _run_ctx(tracer=tracer, trace_include_content=True),
            output_context=_TEXT_OUTPUT_CONTEXT,
            output='dirty',
        )
        assert out == 'clean'

        span = _only_span(exporter)
        assert span.name == 'guardrail redacted output'
        assert dict(span.attributes or {}) == {
            'guardrail.direction': 'output',
            'guardrail.action': 'replace',
            'guardrail.original': 'dirty',
            'guardrail.replacement': 'clean',
        }

    async def test_redaction_span_omits_content_by_default(self):
        tracer, exporter = _recording_tracer()
        og = OutputGuard(guard=lambda _: GuardResult.replace('clean'))

        await og.after_output_process(_run_ctx(tracer=tracer), output_context=_TEXT_OUTPUT_CONTEXT, output='dirty')

        span = _only_span(exporter)
        assert span.name == 'guardrail redacted output'
        assert dict(span.attributes or {}) == {'guardrail.direction': 'output', 'guardrail.action': 'replace'}
