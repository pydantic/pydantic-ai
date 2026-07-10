"""Tests for the `InputGuard` capability."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NoOpTracer, Tracer
from pydantic_ai import Agent
from pydantic_ai.capabilities import CapabilityOrdering
from pydantic_ai.exceptions import SkipModelRequest, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness import GuardResult, InputBlocked, InputGuard
from pydantic_ai_harness.guardrails._capability import _extract_prompt  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _recording_tracer() -> tuple[Tracer, InMemorySpanExporter]:
    """A real OTel tracer that records finished spans into an in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer('test'), exporter


def _build_ctx_and_req(
    run_step: int = 1,
    prompt: str | None = 'hello world',
    *,
    messages: list[ModelMessage] | None = None,
    trace_include_content: bool = False,
    tracer: Tracer | None = None,
) -> tuple[RunContext[object], ModelRequestContext]:
    model = TestModel()
    if messages is None:
        messages = [ModelRequest(parts=[UserPromptPart(content=prompt)])] if prompt is not None else []
    req_ctx = ModelRequestContext(
        model=model,
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    run_ctx: RunContext[object] = RunContext(
        deps=None,
        model=model,
        usage=RunUsage(),
        prompt=prompt,
        messages=messages,
        run_step=run_step,
        trace_include_content=trace_include_content,
        tracer=tracer if tracer is not None else NoOpTracer(),
    )
    return run_ctx, req_ctx


class TestInputGuard:
    """Integration tests for the `InputGuard` capability driven through `Agent.run`."""

    async def test_allows_when_safe(self):
        calls: list[str] = []

        def guard(prompt: str) -> bool:
            calls.append(prompt)
            return True

        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[InputGuard(guard=guard)])
        result = await agent.run('hello')

        assert result.output == 'ok'
        assert calls == ['hello']

    async def test_guard_result_allow(self):
        agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[InputGuard(guard=lambda _: GuardResult.allow())],
        )
        assert (await agent.run('hello')).output == 'ok'

    async def test_block_uses_default_message(self):
        agent = Agent(
            TestModel(custom_output_text='would be model output'),
            capabilities=[InputGuard(guard=lambda _: False)],
        )
        result = await agent.run('hello')

        assert result.output == 'Request blocked by input guardrail.'

    async def test_block_with_custom_message(self):
        agent = Agent(
            TestModel(custom_output_text='would be model output'),
            capabilities=[InputGuard(guard=lambda _: GuardResult.block('nope'))],
        )
        assert (await agent.run('hello')).output == 'nope'

    async def test_block_without_message_uses_default(self):
        agent = Agent(
            TestModel(custom_output_text='would be model output'),
            capabilities=[InputGuard(guard=lambda _: GuardResult.block())],
        )
        assert (await agent.run('hello')).output == 'Request blocked by input guardrail.'

    async def test_async_guard_awaited(self):
        async def guard(prompt: str) -> bool:
            await asyncio.sleep(0)
            return 'safe' in prompt

        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[InputGuard(guard=guard)])

        assert (await agent.run('safe message')).output == 'ok'
        assert (await agent.run('bad message')).output == 'Request blocked by input guardrail.'

    async def test_raising_propagates(self):
        def guard(_: str) -> bool:
            raise InputBlocked('policy violation')

        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[InputGuard(guard=guard)])
        with pytest.raises(InputBlocked, match='policy violation'):
            await agent.run('anything')

    async def test_guard_receives_run_context(self):
        seen: list[object] = []

        def guard(ctx: RunContext[object], prompt: str) -> bool:
            seen.append(ctx.prompt)
            return True

        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[InputGuard(guard=guard)])
        result = await agent.run('hello')

        assert result.output == 'ok'
        assert seen == ['hello']

    async def test_retry_action_is_a_usage_error(self):
        agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[InputGuard(guard=lambda _: GuardResult.retry('redo'))],
        )
        with pytest.raises(UserError, match='cannot return GuardResult.retry'):
            await agent.run('hello')

    async def test_runs_once_across_tool_loop(self):
        """End-to-end: guard fires once even when the model makes multiple tool calls."""
        calls: list[str] = []

        def guard(prompt: str) -> bool:
            calls.append(prompt)
            return True

        model = TestModel(call_tools='all', custom_output_text='done')
        agent = Agent(model, capabilities=[InputGuard(guard=guard)])

        @agent.tool_plain
        def ping() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'pong'

        result = await agent.run('hello')
        assert result.output == 'done'
        assert calls == ['hello']


class TestInputGuardRedaction:
    """Tests for `GuardResult.replace()` rewriting the prompt sent to the model."""

    async def test_replace_rewrites_prompt_for_handler(self):
        run_ctx, req_ctx = _build_ctx_and_req()
        sentinel = ModelResponse(parts=[TextPart(content='direct')])
        seen: list[object] = []

        async def handler(received: ModelRequestContext) -> ModelResponse:
            part = received.messages[0].parts[0]
            assert isinstance(part, UserPromptPart)
            seen.append(part.content)
            return sentinel

        ig = InputGuard(guard=lambda _: GuardResult.replace('[redacted]'))
        out = await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)

        assert out is sentinel
        assert seen == ['[redacted]']
        part = req_ctx.messages[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert part.content == '[redacted]'

    async def test_replace_via_agent_run(self):
        def guard(prompt: str) -> GuardResult:
            return GuardResult.replace(prompt.replace('secret', '***'))

        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[InputGuard(guard=guard)])
        result = await agent.run('my secret value')

        assert result.output == 'ok'
        part = list(result.all_messages())[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert part.content == 'my *** value'

    async def test_replace_with_non_str_is_a_usage_error(self):
        agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[InputGuard(guard=lambda _: GuardResult.replace(123))],
        )
        with pytest.raises(UserError, match='must provide replacement prompt text'):
            await agent.run('hello')

    async def test_replace_without_a_user_prompt_is_a_usage_error(self):
        # `ctx.prompt` is set so the guard runs, but the request carries no `UserPromptPart`.
        run_ctx, req_ctx = _build_ctx_and_req(messages=[ModelResponse(parts=[TextPart(content='no user prompt here')])])

        ig = InputGuard(guard=lambda _: GuardResult.replace('[redacted]'))
        with pytest.raises(UserError, match='could not find a user prompt'):
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=_unreachable_handler)


class TestInputGuardSequential:
    """Direct `wrap_model_request` tests for sequential mode."""

    async def test_runs_guard_then_handler(self):
        run_ctx, req_ctx = _build_ctx_and_req()
        sentinel = ModelResponse(parts=[TextPart(content='direct')])
        calls: list[str] = []

        def guard(prompt: str) -> bool:
            calls.append(prompt)
            return True

        async def handler(_: Any) -> ModelResponse:
            return sentinel

        ig = InputGuard(guard=guard, parallel=False)
        out = await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)
        assert out is sentinel
        assert calls == ['hello world']

    async def test_skipped_when_prompt_missing(self):
        run_ctx, req_ctx = _build_ctx_and_req(prompt=None)
        sentinel = ModelResponse(parts=[TextPart(content='direct')])
        called: list[str] = []

        def guard(prompt: str) -> bool:  # pragma: no cover - should not be called
            called.append(prompt)
            return True

        async def handler(_: Any) -> ModelResponse:
            return sentinel

        ig = InputGuard(guard=guard, parallel=False)
        out = await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)
        assert out is sentinel
        assert called == []

    async def test_skips_guard_on_subsequent_steps(self):
        run_ctx, req_ctx = _build_ctx_and_req(run_step=2)
        sentinel = ModelResponse(parts=[TextPart(content='direct')])
        called: list[str] = []

        def guard(prompt: str) -> bool:  # pragma: no cover - should not be called after step 1
            called.append(prompt)
            return False

        async def handler(_: Any) -> ModelResponse:
            return sentinel

        ig = InputGuard(guard=guard, parallel=False)
        out = await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)
        assert out is sentinel
        assert called == []


class TestInputGuardParallel:
    """Tests for `InputGuard(parallel=True)` exercising the race between guard and handler."""

    async def test_allows_handler_to_return(self):
        run_ctx, req_ctx = _build_ctx_and_req()
        sentinel = ModelResponse(parts=[TextPart(content='from handler')])

        async def handler(_: Any) -> ModelResponse:
            return sentinel

        guard = InputGuard(guard=lambda _: True, parallel=True)
        out = await guard.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)
        assert out is sentinel

    async def test_trips_and_cancels_handler(self):
        run_ctx, req_ctx = _build_ctx_and_req()
        handler_cancelled = asyncio.Event()
        handler_started = asyncio.Event()

        async def slow_handler(_: Any) -> ModelResponse:
            handler_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                handler_cancelled.set()
                raise
            return ModelResponse(parts=[TextPart(content='should never')])  # pragma: no cover

        async def guard(_: str) -> GuardResult:
            await handler_started.wait()
            return GuardResult.block('blocked!')

        ig = InputGuard(guard=guard, parallel=True)
        with pytest.raises(SkipModelRequest) as exc_info:
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=slow_handler)

        assert exc_info.value.response.parts[0] == TextPart(content='blocked!')
        await asyncio.sleep(0)
        assert handler_cancelled.is_set()

    async def test_guard_raises_propagates(self):
        run_ctx, req_ctx = _build_ctx_and_req()

        async def slow_handler(_: Any) -> ModelResponse:
            await asyncio.sleep(10)
            return ModelResponse(parts=[TextPart(content='never')])  # pragma: no cover

        async def guard(_: str) -> bool:
            raise InputBlocked('hard policy failure')

        ig = InputGuard(guard=guard, parallel=True)
        with pytest.raises(InputBlocked, match='hard policy failure'):
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=slow_handler)

    async def test_handler_finishes_before_guard(self):
        """Handler completes first; guard still has to be awaited for a verdict."""
        run_ctx, req_ctx = _build_ctx_and_req()
        sentinel = ModelResponse(parts=[TextPart(content='from handler')])
        release_guard = asyncio.Event()

        async def fast_handler(_: Any) -> ModelResponse:
            return sentinel

        async def slow_guard(_: str) -> bool:
            await release_guard.wait()
            return True

        async def runner() -> ModelResponse:
            ig = InputGuard(guard=slow_guard, parallel=True)
            return await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=fast_handler)

        task = asyncio.create_task(runner())
        for _ in range(3):
            await asyncio.sleep(0)
        release_guard.set()
        assert await task is sentinel

    async def test_handler_finishes_then_guard_trips(self):
        """Handler returns first, then the guard trips - `SkipModelRequest` still wins."""
        run_ctx, req_ctx = _build_ctx_and_req()
        release_guard = asyncio.Event()

        async def fast_handler(_: Any) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='from handler')])

        async def slow_guard(_: str) -> GuardResult:
            await release_guard.wait()
            return GuardResult.block('late trip')

        async def runner() -> ModelResponse:
            ig = InputGuard(guard=slow_guard, parallel=True)
            return await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=fast_handler)

        task = asyncio.create_task(runner())
        for _ in range(3):
            await asyncio.sleep(0)
        release_guard.set()
        with pytest.raises(SkipModelRequest) as exc_info:
            await task
        assert exc_info.value.response.parts[0] == TextPart(content='late trip')

    async def test_handler_raises_while_guard_runs(self):
        """When the handler raises, `finally` cancels the still-running guard."""
        run_ctx, req_ctx = _build_ctx_and_req()
        guard_cancelled = asyncio.Event()

        async def failing_handler(_: Any) -> ModelResponse:
            raise RuntimeError('model boom')

        async def slow_guard(_: str) -> bool:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                guard_cancelled.set()
                raise
            return True  # pragma: no cover

        ig = InputGuard(guard=slow_guard, parallel=True)
        with pytest.raises(RuntimeError, match='model boom'):
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=failing_handler)
        await asyncio.sleep(0)
        assert guard_cancelled.is_set()

    async def test_skipped_when_prompt_missing(self):
        run_ctx, req_ctx = _build_ctx_and_req(prompt=None)
        sentinel = ModelResponse(parts=[TextPart(content='direct')])
        called: list[str] = []

        def guard(prompt: str) -> bool:  # pragma: no cover - should never be called
            called.append(prompt)
            return False

        async def handler(_: Any) -> ModelResponse:
            return sentinel

        ig = InputGuard(guard=guard, parallel=True)
        out = await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)
        assert out is sentinel
        assert called == []

    async def test_skips_guard_on_subsequent_steps(self):
        run_ctx, req_ctx = _build_ctx_and_req(run_step=2)
        sentinel = ModelResponse(parts=[TextPart(content='direct')])
        called: list[str] = []

        def guard(prompt: str) -> bool:  # pragma: no cover - should not be called after step 1
            called.append(prompt)
            return False

        async def handler(_: Any) -> ModelResponse:
            return sentinel

        ig = InputGuard(guard=guard, parallel=True)
        out = await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)
        assert out is sentinel
        assert called == []

    async def test_replace_under_parallel_is_a_usage_error(self):
        run_ctx, req_ctx = _build_ctx_and_req()

        async def handler(_: Any) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='from handler')])

        ig = InputGuard(guard=lambda _: GuardResult.replace('[redacted]'), parallel=True)
        with pytest.raises(UserError, match='incompatible with GuardResult.replace'):
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)

    async def test_no_dangling_tasks_when_handler_raises(self):
        """`finally` must drain cancelled tasks so they don't outlive the call."""
        run_ctx, req_ctx = _build_ctx_and_req()

        async def failing_handler(_: Any) -> ModelResponse:
            raise RuntimeError('handler boom')

        async def slow_guard(_: str) -> bool:
            await asyncio.sleep(10)
            return True  # pragma: no cover

        current = asyncio.current_task()
        before = {t for t in asyncio.all_tasks() if t is not current}

        ig = InputGuard(guard=slow_guard, parallel=True)
        with pytest.raises(RuntimeError, match='handler boom'):
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=failing_handler)

        leftover = {t for t in asyncio.all_tasks() if t is not current} - before
        assert leftover == set(), f'guard/handler tasks must be drained, got dangling: {leftover}'

    async def test_no_dangling_tasks_on_outer_cancellation(self):
        """If the caller cancels the run mid-flight, the `finally` block must still drain
        guard and handler tasks instead of leaking them.

        Regression guard for a reviewed concern about whether `asyncio.shield` was needed
        around the cleanup `gather`. It isn't — the outer cancel is consumed by the
        `asyncio.wait` above, so the subsequent `cancel()` + `gather()` in `finally`
        complete without being re-cancelled.
        """
        run_ctx, req_ctx = _build_ctx_and_req()

        async def slow_handler(_: Any) -> ModelResponse:
            await asyncio.sleep(10)
            return ModelResponse(parts=[TextPart(content='never')])  # pragma: no cover

        async def slow_guard(_: str) -> bool:
            await asyncio.sleep(10)
            return True  # pragma: no cover

        current = asyncio.current_task()
        before = {t for t in asyncio.all_tasks() if t is not current}

        ig = InputGuard(guard=slow_guard, parallel=True)

        async def runner() -> ModelResponse:
            return await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=slow_handler)

        runner_task = asyncio.create_task(runner())
        for _ in range(5):
            await asyncio.sleep(0)
        runner_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner_task
        for _ in range(5):
            await asyncio.sleep(0)

        leftover = {t for t in asyncio.all_tasks() if t is not current and not t.done()} - before
        assert leftover == set(), f'guard/handler tasks must be drained on outer cancel, got: {leftover}'


class TestInputGuardTracing:
    """Spans emitted on block and redaction."""

    async def test_block_span_includes_message_when_enabled(self):
        tracer, exporter = _recording_tracer()
        run_ctx, req_ctx = _build_ctx_and_req(tracer=tracer, trace_include_content=True)

        ig = InputGuard(guard=lambda _: GuardResult.block('nope'))
        with pytest.raises(SkipModelRequest):
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=_unreachable_handler)

        span = _only_span(exporter)
        assert span.name == 'guardrail blocked input'
        assert dict(span.attributes or {}) == {
            'guardrail.direction': 'input',
            'guardrail.action': 'block',
            'guardrail.message': 'nope',
        }

    async def test_block_span_omits_message_by_default(self):
        """The refusal message may quote sensitive content, so it is gated like redactions are."""
        tracer, exporter = _recording_tracer()
        run_ctx, req_ctx = _build_ctx_and_req(tracer=tracer)

        ig = InputGuard(guard=lambda _: GuardResult.block('nope'))
        with pytest.raises(SkipModelRequest):
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=_unreachable_handler)

        span = _only_span(exporter)
        assert span.name == 'guardrail blocked input'
        assert dict(span.attributes or {}) == {'guardrail.direction': 'input', 'guardrail.action': 'block'}

    async def test_redaction_span_includes_content_when_enabled(self):
        tracer, exporter = _recording_tracer()
        run_ctx, req_ctx = _build_ctx_and_req(tracer=tracer, trace_include_content=True)

        ig = InputGuard(guard=lambda _: GuardResult.replace('[redacted]'))
        await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=_sentinel_handler)

        span = _only_span(exporter)
        assert span.name == 'guardrail redacted input'
        assert dict(span.attributes or {}) == {
            'guardrail.direction': 'input',
            'guardrail.action': 'replace',
            'guardrail.original': 'hello world',
            'guardrail.replacement': '[redacted]',
        }

    async def test_redaction_span_omits_content_by_default(self):
        tracer, exporter = _recording_tracer()
        run_ctx, req_ctx = _build_ctx_and_req(tracer=tracer)

        ig = InputGuard(guard=lambda _: GuardResult.replace('[redacted]'))
        await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=_sentinel_handler)

        span = _only_span(exporter)
        assert span.name == 'guardrail redacted input'
        assert dict(span.attributes or {}) == {'guardrail.direction': 'input', 'guardrail.action': 'replace'}


class TestInputGuardStreaming:
    """`InputGuard` behaves the same under `run_stream()` as under `run()`."""

    async def test_parallel_guard_under_run_stream(self):
        """Regression guard for a reviewed deadlock concern.

        A pydantic-ai-correctness review suspected `parallel=True` could deadlock
        under `run_stream()`, because the streaming model handler parks until the
        caller drains the stream while the parallel branch awaits that same
        handler task. It does not — the graph hands off on `stream_ready`, which
        the handler sets before parking. The `wait_for` makes a reintroduced hang
        fail fast here instead of stalling CI.
        """
        agent = Agent(
            TestModel(custom_output_text='streamed'),
            capabilities=[InputGuard(guard=lambda _: True, parallel=True)],
        )

        async def run() -> str:
            async with agent.run_stream('hello') as result:
                return await result.get_output()

        assert await asyncio.wait_for(run(), timeout=10) == 'streamed'

    async def test_block_under_run_stream(self):
        agent = Agent(
            TestModel(custom_output_text='would be model output'),
            capabilities=[InputGuard(guard=lambda _: GuardResult.block('nope'))],
        )
        async with agent.run_stream('hello') as result:
            assert (await result.get_output()) == 'nope'


class TestExtractPrompt:
    """Unit tests for the `_extract_prompt` helper."""

    def test_from_messages(self):
        class _Ctx:
            prompt = None

        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='first')]),
            ModelResponse(parts=[TextPart(content='assistant')]),
            ModelRequest(parts=[UserPromptPart(content='second')]),
        ]
        assert _extract_prompt(_Ctx(), messages) == 'second'  # pyright: ignore[reportArgumentType]

    def test_stringifies_non_str_prompt(self):
        class _Ctx:
            prompt = ['multimodal', 'content']

        assert _extract_prompt(_Ctx(), []) == str(['multimodal', 'content'])  # pyright: ignore[reportArgumentType]

    def test_stringifies_non_str_message_part(self):
        class _Ctx:
            prompt = None

        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=['multi'])])]
        assert _extract_prompt(_Ctx(), messages) == str(['multi'])  # pyright: ignore[reportArgumentType]

    def test_returns_none_when_no_user_prompt_part(self):
        class _Ctx:
            prompt = None

        messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(content='only model parts here')])]
        assert _extract_prompt(_Ctx(), messages) is None  # pyright: ignore[reportArgumentType]


class TestInputGuardOrdering:
    """`get_ordering()` placement: innermost so message-morphing capabilities run first."""

    def test_declares_innermost(self):
        ordering = InputGuard(guard=lambda _: True).get_ordering()
        assert ordering == CapabilityOrdering(position='innermost')


class TestGuardResult:
    """`GuardResult.__post_init__` rejects field combinations the four-outcome contract does not allow."""

    def test_allow_rejects_message(self):
        with pytest.raises(UserError, match="'allow'"):
            GuardResult(action='allow', message='oops')

    def test_allow_rejects_replacement(self):
        with pytest.raises(UserError, match="'allow'"):
            GuardResult(action='allow', replacement='oops')

    def test_replace_requires_replacement(self):
        with pytest.raises(UserError, match="'replace'"):
            GuardResult(action='replace')

    def test_retry_requires_message(self):
        with pytest.raises(UserError, match="'retry'"):
            GuardResult(action='retry')

    def test_block_without_message_is_valid(self):
        """A `block` with no message is fine: the default kicks in at the use site."""
        result = GuardResult(action='block')
        assert result.message is None

    def test_is_frozen(self):
        result = GuardResult.allow()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.message = 'mutated'  # pyright: ignore[reportAttributeAccessIssue]


async def _unreachable_handler(_: ModelRequestContext) -> ModelResponse:  # pragma: no cover - never awaited
    raise AssertionError('handler should not be called')


async def _sentinel_handler(_: ModelRequestContext) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='ok')])


def _only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f'expected exactly one span, got {[s.name for s in spans]}'
    return spans[0]
