"""Tests for the `InputGuardrail` capability."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Sequence
from typing import Any

import anyio
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NoOpTracer, Tracer

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Hooks, WrapModelRequestHandler
from pydantic_ai.exceptions import ModelRetry, SkipModelRequest, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    BinaryContent,
    FinalResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartEndEvent,
    PartStartEvent,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import BackgroundTools, GuardrailResult, InputBlocked, InputGuardrail
from pydantic_ai_harness.guardrails._capability import _extract_prompt  # pyright: ignore[reportPrivateUsage]

READINESS_WAIT_TIMEOUT = 10


def _recording_tracer() -> tuple[Tracer, InMemorySpanExporter]:
    """A real OTel tracer that records finished spans into an in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer('test'), exporter


def _recording_model() -> tuple[list[object], FunctionModel]:
    """A model that records the prompt content it was handed, so a rewritten prompt is observable."""
    seen: list[object] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        part = messages[0].parts[0]
        assert isinstance(part, UserPromptPart)
        seen.append(part.content)
        return ModelResponse(parts=[TextPart(content='ok')])

    return seen, FunctionModel(respond)


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


class TestInputGuardrail:
    """Integration tests for the `InputGuardrail` capability driven through `Agent.run`."""

    async def test_allows_when_safe(self):
        calls: list[str] = []

        def guard(prompt: str) -> bool:
            calls.append(prompt)
            return True

        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[InputGuardrail(guard=guard)])
        result = await agent.run('hello')

        assert result.output == 'ok'
        assert calls == ['hello']

    @pytest.mark.parametrize('metadata', [None, 'tag', [], {'origin': 'caller'}])
    async def test_singleton_textcontent_user_prompt_keeps_legacy_guardrail_behavior(self, metadata: object):
        prompt: list[TextContent] = [TextContent('ordinary user content', metadata=metadata)]
        seen: list[str] = []

        agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[InputGuardrail(guard=lambda text: seen.append(text) or True)],
        )
        await agent.run(prompt)

        assert seen == [str(prompt)]

        replacing_agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[InputGuardrail(guard=lambda _: GuardrailResult.replace('[redacted]'))],
        )
        with pytest.raises(UserError, match='multimodal prompt'):
            await replacing_agent.run(prompt)

    async def test_extracts_singleton_textcontent_from_resumed_prompt(self):
        guard_calls: list[str] = []
        model_prompts: list[list[object]] = []
        model_calls = 0
        completion = TextContent(
            'background answer',
            metadata={'origin': 'pydantic-ai-harness.background_tools.result.v1', 'tool_call_id': 'task-1'},
        )

        def guard(prompt: str) -> bool:
            guard_calls.append(prompt)
            return True

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal model_calls
            model_calls += 1
            model_prompts.append(
                [part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)]
            )
            if model_calls == 1:
                return ModelResponse(parts=[ToolCallPart('queue_completion', '{}', tool_call_id='tc1')])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(respond), capabilities=[InputGuardrail(guard=guard)])

        @agent.tool
        def queue_completion(ctx: RunContext[object]) -> str:
            ctx.enqueue(completion)
            return 'queued'

        first = await agent.run('initial')
        assert guard_calls == ['initial']

        resumed = await agent.run(None, message_history=first.all_messages()[:-1])
        assert resumed.output == 'done'
        assert guard_calls == ['initial', 'background answer']
        assert model_prompts[1] == ['initial', [completion]]

        await agent.run(None, message_history=first.all_messages())
        assert guard_calls == ['initial', 'background answer']

    async def test_redacts_background_completion_from_pending_tool_history(self):
        guard_observations: list[tuple[object, int, str]] = []
        model_user_prompts: list[object] = []
        completion = TextContent(
            'background answer',
            metadata={
                'origin': 'pydantic-ai-harness.background_tools.result.v1',
                'tool_call_id': 'task-1',
            },
        )

        def guard(ctx: RunContext[object], prompt: str) -> GuardrailResult:
            guard_observations.append((ctx.prompt, ctx.run_step, prompt))
            return GuardrailResult.replace('[redacted]')

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            model_user_prompts.extend(
                part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)
            )
            return ModelResponse(parts=[TextPart('done')])

        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=[completion])]),
            ModelResponse(parts=[ToolCallPart(tool_name='some_tool', args='{}', tool_call_id='tc1')]),
            ModelRequest(parts=[ToolReturnPart(tool_name='some_tool', content='result', tool_call_id='tc1')]),
        ]
        agent = Agent(FunctionModel(respond), capabilities=[InputGuardrail(guard=guard)])
        result = await agent.run(None, message_history=history)

        assert result.output == 'done'
        assert guard_observations == [(None, 1, 'background answer')]
        assert model_user_prompts == [
            [
                TextContent(
                    '[redacted]',
                    metadata={
                        'origin': 'pydantic-ai-harness.background_tools.result.v1',
                        'tool_call_id': 'task-1',
                    },
                )
            ]
        ]

    async def test_background_completion_with_document_uses_multimodal_prompt_rules(self):
        completion = TextContent(
            'background answer',
            metadata={
                'origin': 'pydantic-ai-harness.background_tools.result.v1',
                'tool_call_id': 'task-1',
            },
        )
        document = BinaryContent(data=b'background report\n', media_type='text/plain')
        content: list[TextContent | BinaryContent] = [completion, document]
        history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=content)])]
        seen: list[str] = []

        allowing_agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[InputGuardrail(guard=lambda prompt: seen.append(prompt) or True)],
        )
        result = await allowing_agent.run(None, message_history=history)

        assert result.output == 'ok'
        assert seen == [str(content)]

        replacing_agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[InputGuardrail(guard=lambda _: GuardrailResult.replace('[redacted]'))],
        )
        with pytest.raises(UserError, match='multimodal prompt'):
            await replacing_agent.run(None, message_history=history)

        prompt = history[-1].parts[0]
        assert isinstance(prompt, UserPromptPart)
        assert prompt.content == content

    async def test_guard_result_allow(self):
        agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[InputGuardrail(guard=lambda _: GuardrailResult.allow())],
        )
        assert (await agent.run('hello')).output == 'ok'

    async def test_block_uses_default_message(self):
        agent = Agent(
            TestModel(custom_output_text='would be model output'),
            capabilities=[InputGuardrail(guard=lambda _: False)],
        )
        result = await agent.run('hello')

        assert result.output == 'Request blocked by input guardrail.'

    async def test_block_with_custom_message(self):
        agent = Agent(
            TestModel(custom_output_text='would be model output'),
            capabilities=[InputGuardrail(guard=lambda _: GuardrailResult.block('nope'))],
        )
        assert (await agent.run('hello')).output == 'nope'

    async def test_block_without_message_uses_default(self):
        agent = Agent(
            TestModel(custom_output_text='would be model output'),
            capabilities=[InputGuardrail(guard=lambda _: GuardrailResult.block())],
        )
        assert (await agent.run('hello')).output == 'Request blocked by input guardrail.'

    async def test_async_guard_awaited(self):
        async def guard(prompt: str) -> bool:
            await asyncio.sleep(0)
            return 'safe' in prompt

        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[InputGuardrail(guard=guard)])

        assert (await agent.run('safe message')).output == 'ok'
        assert (await agent.run('bad message')).output == 'Request blocked by input guardrail.'

    async def test_raising_propagates(self):
        def guard(_: str) -> bool:
            raise InputBlocked('policy violation')

        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[InputGuardrail(guard=guard)])
        with pytest.raises(InputBlocked, match='policy violation'):
            await agent.run('anything')

    async def test_guard_receives_run_context(self):
        seen: list[object] = []

        def guard(ctx: RunContext[object], prompt: str) -> bool:
            seen.append(ctx.prompt)
            return True

        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[InputGuardrail(guard=guard)])
        result = await agent.run('hello')

        assert result.output == 'ok'
        assert seen == ['hello']

    async def test_retry_action_is_a_usage_error(self):
        agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[InputGuardrail(guard=lambda _: GuardrailResult.retry('redo'))],
        )
        with pytest.raises(UserError, match=r'cannot return GuardrailResult.retry'):
            await agent.run('hello')

    async def test_runs_once_across_tool_loop(self):
        """End-to-end: guard fires once even when the model makes multiple tool calls."""
        calls: list[str] = []

        def guard(prompt: str) -> bool:
            calls.append(prompt)
            return True

        model = TestModel(call_tools='all', custom_output_text='done')
        agent = Agent(model, capabilities=[InputGuardrail(guard=guard)])

        @agent.tool_plain
        def ping() -> str:
            return 'pong'

        result = await agent.run('hello')
        assert result.output == 'done'
        assert calls == ['hello']


class TestInputGuardrailRedaction:
    """Tests for `GuardrailResult.replace()` rewriting the prompt sent to the model."""

    async def test_replace_rewrites_prompt_for_handler(self):
        run_ctx, req_ctx = _build_ctx_and_req()
        sentinel = ModelResponse(parts=[TextPart(content='direct')])
        seen: list[object] = []

        async def handler(received: ModelRequestContext) -> ModelResponse:
            part = received.messages[0].parts[0]
            assert isinstance(part, UserPromptPart)
            seen.append(part.content)
            return sentinel

        ig = InputGuardrail(guard=lambda _: GuardrailResult.replace('[redacted]'))
        out = await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)

        assert out is sentinel
        assert seen == ['[redacted]']
        part = req_ctx.messages[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert part.content == '[redacted]'

    async def test_replace_via_agent_run(self):
        def guard(prompt: str) -> GuardrailResult:
            return GuardrailResult.replace(prompt.replace('secret', '***'))

        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[InputGuardrail(guard=guard)])
        result = await agent.run('my secret value')

        assert result.output == 'ok'
        part = list(result.all_messages())[0].parts[0]
        assert isinstance(part, UserPromptPart)
        assert part.content == 'my *** value'

    async def test_replace_background_completion_preserves_metadata(self):
        guard_calls: list[str] = []
        model_completions: list[tuple[str, object]] = []

        def guard(prompt: str) -> bool | GuardrailResult:
            guard_calls.append(prompt)
            if prompt.startswith('<tool_result'):
                return GuardrailResult.replace('[redacted]')
            return True

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            completions = [
                item
                for message in messages
                for part in message.parts
                if isinstance(part, UserPromptPart) and isinstance(part.content, list)
                for item in part.content
                if isinstance(item, TextContent)
                and item.metadata.get('origin') == 'pydantic-ai-harness.background_tools.result.v1'
            ]
            if completions:
                model_completions.append((completions[-1].content, completions[-1].metadata))
                return ModelResponse(parts=[TextPart('done')])
            return ModelResponse(parts=[ToolCallPart('lookup', '{}', tool_call_id='tool-1')])

        agent = Agent(FunctionModel(respond), capabilities=[BackgroundTools(), InputGuardrail(guard=guard)])

        @agent.tool_plain(metadata={'background': True})
        async def lookup() -> str:
            return 'answer'

        completion = '<tool_result tool_name="lookup" tool_call_id="tool-1">\nanswer\n</tool_result>'
        first = await agent.run('go')
        assert first.output == 'done'
        assert guard_calls == ['go']
        assert model_completions == [
            (
                completion,
                {'origin': 'pydantic-ai-harness.background_tools.result.v1', 'tool_call_id': 'tool-1'},
            )
        ]

        resumed = await agent.run(None, message_history=first.all_messages()[:-1])

        assert resumed.output == 'done'
        assert guard_calls == ['go', completion]
        assert model_completions[-1] == (
            '[redacted]',
            {'origin': 'pydantic-ai-harness.background_tools.result.v1', 'tool_call_id': 'tool-1'},
        )

    async def test_replace_with_non_str_is_a_usage_error(self):
        agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[InputGuardrail(guard=lambda _: GuardrailResult.replace(123))],
        )
        with pytest.raises(UserError, match='must provide replacement prompt text'):
            await agent.run('hello')

    async def test_replace_on_a_multimodal_prompt_is_a_usage_error(self):
        """Writing the replacement over the part would send the model text alone and drop the image."""
        seen, model = _recording_model()
        agent = Agent(model, capabilities=[InputGuardrail(guard=lambda _: GuardrailResult.replace('[redacted]'))])

        with pytest.raises(UserError, match='multimodal prompt'):
            await agent.run(['describe this', BinaryContent(data=b'\x89PNG', media_type='image/png')])

        assert seen == []

    async def test_a_multimodal_prompt_the_guard_allows_reaches_the_model_intact(self):
        seen, model = _recording_model()
        image = BinaryContent(data=b'\x89PNG', media_type='image/png')
        agent = Agent(model, capabilities=[InputGuardrail(guard=lambda _: True)])

        await agent.run(['describe this', image])

        assert seen == [['describe this', image]]

    async def test_replace_without_a_user_prompt_is_a_usage_error(self):
        # `ctx.prompt` is set so the guard runs, but the request carries no `UserPromptPart`.
        run_ctx, req_ctx = _build_ctx_and_req(messages=[ModelResponse(parts=[TextPart(content='no user prompt here')])])

        ig = InputGuardrail(guard=lambda _: GuardrailResult.replace('[redacted]'))
        with pytest.raises(UserError, match='could not find a user prompt'):
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=_unreachable_handler)


class TestInputGuardrailSequential:
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

        ig = InputGuardrail(guard=guard, parallel=False)
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

        ig = InputGuardrail(guard=guard, parallel=False)
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

        ig = InputGuardrail(guard=guard, parallel=False)
        out = await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)
        assert out is sentinel
        assert called == []


class TestInputGuardrailParallel:
    """Tests for `InputGuardrail(parallel=True)` exercising the race between guard and handler."""

    async def test_allows_handler_to_return(self):
        run_ctx, req_ctx = _build_ctx_and_req()
        sentinel = ModelResponse(parts=[TextPart(content='from handler')])

        async def handler(_: Any) -> ModelResponse:
            return sentinel

        guard = InputGuardrail(guard=lambda _: True, parallel=True)
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

        async def guard(_: str) -> GuardrailResult:
            await handler_started.wait()
            return GuardrailResult.block('blocked!')

        ig = InputGuardrail(guard=guard, parallel=True)
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

        ig = InputGuardrail(guard=guard, parallel=True)
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
            ig = InputGuardrail(guard=slow_guard, parallel=True)
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

        async def slow_guard(_: str) -> GuardrailResult:
            await release_guard.wait()
            return GuardrailResult.block('late trip')

        async def runner() -> ModelResponse:
            ig = InputGuardrail(guard=slow_guard, parallel=True)
            return await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=fast_handler)

        task = asyncio.create_task(runner())
        for _ in range(3):
            await asyncio.sleep(0)
        release_guard.set()
        with pytest.raises(SkipModelRequest) as exc_info:
            await task
        assert exc_info.value.response.parts[0] == TextPart(content='late trip')

    @pytest.mark.parametrize(
        ('verdict', 'raised'),
        [
            pytest.param(True, RuntimeError, id='allow'),
            pytest.param(GuardrailResult.block('late trip'), SkipModelRequest, id='block'),
        ],
    )
    async def test_handler_raises_then_guard_decides(self, verdict: bool | GuardrailResult, raised: type[Exception]):
        """The handler raises before the guard decides: the error waits for the verdict, and a block wins."""
        run_ctx, req_ctx = _build_ctx_and_req()
        handler_task: asyncio.Future[asyncio.Task[Any] | None] = asyncio.get_running_loop().create_future()
        decided: list[bool] = []

        async def failing_handler(_: Any) -> ModelResponse:
            handler_task.set_result(asyncio.current_task())
            raise RuntimeError('model boom')

        async def slow_guard(_: str) -> bool | GuardrailResult:
            # Decide once the handler's task finished raising, giving `wrap_model_request` the first chance to act.
            task = await handler_task
            assert task is not None
            await asyncio.wait({task})
            decided.append(True)
            return verdict

        ig = InputGuardrail(guard=slow_guard, parallel=True)
        with pytest.raises(raised):
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=failing_handler)
        assert decided == [True]

    async def test_block_after_a_retried_first_request_blocks_the_run(self):
        """A `ModelRetry` on the first response, raised before the guard decides, does not skip the verdict."""
        response_task: asyncio.Future[asyncio.Task[Any] | None] = asyncio.get_running_loop().create_future()
        hooks = Hooks[None]()

        @hooks.on.after_model_request
        async def retry_response(
            ctx: RunContext[None], request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            response_task.set_result(asyncio.current_task())
            raise ModelRetry('try again')

        async def guard(_: str) -> GuardrailResult:
            # Decide once the response's task finished raising, giving `wrap_model_request` the first chance to act.
            task = await response_task
            assert task is not None
            await asyncio.wait({task})
            return GuardrailResult.block('blocked')

        agent = Agent(
            TestModel(), deps_type=type(None), capabilities=[InputGuardrail[None](guard=guard, parallel=True), hooks]
        )

        result = await agent.run('go')

        assert result.output == 'blocked'

    async def test_skipped_when_prompt_missing(self):
        run_ctx, req_ctx = _build_ctx_and_req(prompt=None)
        sentinel = ModelResponse(parts=[TextPart(content='direct')])
        called: list[str] = []

        def guard(prompt: str) -> bool:  # pragma: no cover - should never be called
            called.append(prompt)
            return False

        async def handler(_: Any) -> ModelResponse:
            return sentinel

        ig = InputGuardrail(guard=guard, parallel=True)
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

        ig = InputGuardrail(guard=guard, parallel=True)
        out = await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)
        assert out is sentinel
        assert called == []

    async def test_replace_under_parallel_is_a_usage_error(self):
        run_ctx, req_ctx = _build_ctx_and_req()

        async def handler(_: Any) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='from handler')])

        ig = InputGuardrail(guard=lambda _: GuardrailResult.replace('[redacted]'), parallel=True)
        with pytest.raises(UserError, match=r'incompatible with GuardrailResult.replace'):
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=handler)

    async def test_no_dangling_tasks_when_guard_blocks(self):
        """`finally` must cancel and drain the handler a block interrupts so it doesn't outlive the call."""
        run_ctx, req_ctx = _build_ctx_and_req()
        handler_started = asyncio.Event()

        async def slow_handler(_: Any) -> ModelResponse:
            handler_started.set()
            await asyncio.Event().wait()
            return ModelResponse(parts=[TextPart(content='never')])  # pragma: no cover

        async def guard(_: str) -> GuardrailResult:
            await handler_started.wait()
            return GuardrailResult.block('blocked!')

        current = asyncio.current_task()
        before = {t for t in asyncio.all_tasks() if t is not current}

        ig = InputGuardrail(guard=guard, parallel=True)
        with pytest.raises(SkipModelRequest):
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=slow_handler)

        leftover = {t for t in asyncio.all_tasks() if t is not current} - before
        assert leftover == set(), f'guard/handler tasks must be drained, got dangling: {leftover}'

    async def test_no_dangling_tasks_on_outer_cancellation(self):
        """If the caller cancels the run mid-flight, the `finally` block must still drain
        guard and handler tasks instead of leaking them.

        A single raw cancel is consumed by the `asyncio.wait` above the `finally`, so the
        `cancel()` + `gather()` drain completes even without a shield. The shield matters
        when the cancellation arrives through an already-cancelled anyio scope, covered by
        `test_teardown_survives_cancelled_scope`.
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

        ig = InputGuardrail(guard=slow_guard, parallel=True)

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

    async def test_teardown_survives_cancelled_scope(self):
        """The `finally` drain is shielded, so cancellation delivered through an
        already-cancelled anyio scope (run cancellation, `anyio.fail_after`) cannot
        abandon the guard and handler tasks mid-unwind (#559).

        In a cancelled anyio scope every unshielded await is re-cancelled, so without the
        shield `wrap_model_request` returns while the guard is still unwinding. Sequencing
        is event-driven: the runner must stay blocked in the drain until the guard's
        cancellation handler is released, and the handler must have fully unwound by the
        time the runner completes.
        """
        run_ctx, req_ctx = _build_ctx_and_req()
        started = asyncio.Event()
        cancel_seen = asyncio.Event()
        release = asyncio.Event()
        unwound = asyncio.Event()

        async def slow_handler(_: Any) -> ModelResponse:
            started.set()
            await asyncio.Event().wait()
            return ModelResponse(parts=[TextPart(content='never')])  # pragma: no cover

        async def slow_guard(_: str) -> bool:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancel_seen.set()
                await release.wait()
                unwound.set()
                raise
            return True  # pragma: no cover

        ig = InputGuardrail(guard=slow_guard, parallel=True)
        scope = anyio.CancelScope()

        async def runner() -> None:
            with scope:
                await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=slow_handler)

        task = asyncio.create_task(runner())
        await started.wait()
        scope.cancel()
        await cancel_seen.wait()
        for _ in range(5):
            await asyncio.sleep(0)
        assert not task.done(), 'the drain must wait for the guard to finish unwinding'
        release.set()
        await task
        assert scope.cancelled_caught
        assert unwound.is_set()


class TestInputGuardrailTracing:
    """Spans emitted on block and redaction."""

    async def test_block_span_includes_message_when_enabled(self):
        tracer, exporter = _recording_tracer()
        run_ctx, req_ctx = _build_ctx_and_req(tracer=tracer, trace_include_content=True)

        ig = InputGuardrail(guard=lambda _: GuardrailResult.block('nope'))
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

        ig = InputGuardrail(guard=lambda _: GuardrailResult.block('nope'))
        with pytest.raises(SkipModelRequest):
            await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=_unreachable_handler)

        span = _only_span(exporter)
        assert span.name == 'guardrail blocked input'
        assert dict(span.attributes or {}) == {'guardrail.direction': 'input', 'guardrail.action': 'block'}

    async def test_redaction_span_includes_content_when_enabled(self):
        tracer, exporter = _recording_tracer()
        run_ctx, req_ctx = _build_ctx_and_req(tracer=tracer, trace_include_content=True)

        ig = InputGuardrail(guard=lambda _: GuardrailResult.replace('[redacted]'))
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

        ig = InputGuardrail(guard=lambda _: GuardrailResult.replace('[redacted]'))
        await ig.wrap_model_request(run_ctx, request_context=req_ctx, handler=_sentinel_handler)

        span = _only_span(exporter)
        assert span.name == 'guardrail redacted input'
        assert dict(span.attributes or {}) == {'guardrail.direction': 'input', 'guardrail.action': 'replace'}


class TestInputGuardrailStreaming:
    """`InputGuardrail` behaves the same under `run_stream()` as under `run()`."""

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
            capabilities=[InputGuardrail(guard=lambda _: True, parallel=True)],
        )

        async def run() -> str:
            async with agent.run_stream('hello') as result:
                return await result.get_output()

        assert await asyncio.wait_for(run(), timeout=10) == 'streamed'

    async def test_block_under_run_stream(self):
        agent = Agent(
            TestModel(custom_output_text='would be model output'),
            capabilities=[InputGuardrail(guard=lambda _: GuardrailResult.block('nope'))],
        )
        async with agent.run_stream('hello') as result:
            assert (await result.get_output()) == 'nope'


@dataclasses.dataclass
class _StalledModel:
    """A streaming model that sends one chunk, then stalls until its stream is closed."""

    first_chunk_sent: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    closed: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)

    async def stream(self, _messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        try:
            yield 'leaked '
            self.first_chunk_sent.set()
            await asyncio.Event().wait()
            yield 'never'  # pragma: no cover
        finally:
            self.closed.set()

    def agent(self, guard: InputGuardrail[Any], *capabilities: AbstractCapability[Any]) -> Agent[None, str]:
        return Agent(FunctionModel(stream_function=self.stream), capabilities=[guard, *capabilities])

    def blocking_guard(self) -> InputGuardrail[Any]:
        """A parallel guard that blocks once the first chunk has reached the caller."""

        async def guard(_prompt: str) -> GuardrailResult:
            await self.first_chunk_sent.wait()
            return GuardrailResult.block('nope')

        return InputGuardrail(guard=guard, parallel=True)


@dataclasses.dataclass
class _BlockObserver(AbstractCapability[Any]):
    """Sets `blocked` once the guard it wraps has blocked the request."""

    blocked: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)

    async def wrap_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        try:
            return await handler(request_context)
        except SkipModelRequest:
            self.blocked.set()
            raise


def _text_events(events: Sequence[AgentStreamEvent | AgentRunResultEvent[str]]) -> list[object]:
    """Reduce text stream events to what the caller saw."""
    seen: list[object] = []
    for event in events:
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            seen.append(('start', event.part.content))
        elif isinstance(event, PartEndEvent) and isinstance(event.part, TextPart):
            seen.append(('end', event.part.content))
        elif isinstance(event, FinalResultEvent):
            seen.append('final result')
        else:
            assert isinstance(event, AgentRunResultEvent)
            seen.append(('output', event.result.output))
    return seen


class TestInputGuardrailParallelStreaming:
    """A parallel block cancels a streamed model request and ends the run with the block message.

    Regression tests for https://github.com/pydantic/pydantic-ai/issues/9279: the stream used to
    deliver every model chunk and then raise `SkipModelRequest`. The model stalls after its first
    chunk, so a stream that is not cancelled hangs until `READINESS_WAIT_TIMEOUT`.
    """

    async def test_block_cancels_run_stream_events(self):
        model = _StalledModel()
        agent = model.agent(model.blocking_guard())
        current = asyncio.current_task()
        before = {t for t in asyncio.all_tasks() if t is not current}

        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            async with agent.run_stream_events('hello') as stream:
                events = [event async for event in stream]

        # Chunks sent before the verdict still reach the caller; the block message then replaces them.
        assert _text_events(events) == [
            ('start', 'leaked '),
            'final result',
            ('start', 'nope'),
            'final result',
            ('end', 'nope'),
            ('output', 'nope'),
        ]
        assert model.closed.is_set()
        assert stream.result is not None
        assert stream.result.all_messages()[-1].parts == [TextPart(content='nope')]
        assert {t for t in asyncio.all_tasks() if t is not current and not t.done()} - before == set()

    @pytest.mark.parametrize('delta', [True, False])
    async def test_block_cancels_run_stream(self, delta: bool):
        model = _StalledModel()
        agent = model.agent(model.blocking_guard())

        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            async with agent.run_stream('hello') as result:
                texts = [text async for text in result.stream_text(delta=delta, debounce_by=None)]
                assert await result.get_output() == 'nope'

        # The block message replaces the text streamed so far, rather than being appended to it.
        assert texts == ['leaked ', 'nope']
        assert model.closed.is_set()
        assert result.all_messages()[-1].parts == [TextPart(content='nope')]

    async def test_block_cancels_node_stream(self):
        model = _StalledModel()
        agent = model.agent(model.blocking_guard())
        events: list[AgentStreamEvent] = []

        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            async with agent.iter('hello') as run:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as stream:
                            events.extend([event async for event in stream])

        assert _text_events(events) == [
            ('start', 'leaked '),
            'final result',
            ('start', 'nope'),
            'final result',
            ('end', 'nope'),
        ]
        assert model.closed.is_set()
        assert run.result is not None
        assert run.result.output == 'nope'

    async def test_block_before_the_first_chunk_skips_the_model_stream(self):
        """A block that lands after the stream opened but before the caller pulled from it."""
        model = _StalledModel()
        release_guard = asyncio.Event()

        async def guard(_prompt: str) -> GuardrailResult:
            await release_guard.wait()
            return GuardrailResult.block('nope')

        observer = _BlockObserver()
        agent = model.agent(InputGuardrail(guard=guard, parallel=True), observer)
        events: list[AgentStreamEvent] = []

        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            async with agent.iter('hello') as run:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as stream:
                            release_guard.set()
                            await observer.blocked.wait()
                            events.extend([event async for event in stream])

        assert _text_events(events) == [('start', 'nope'), 'final result', ('end', 'nope')]
        assert not model.first_chunk_sent.is_set()
        assert run.result is not None
        assert run.result.output == 'nope'

    async def test_guard_error_mid_stream_raises_from_the_stream(self):
        """A guard error is raised where the stream was cut, so the cut-short output is never settled on."""
        model = _StalledModel()

        async def guard(_prompt: str) -> bool:
            await model.first_chunk_sent.wait()
            raise InputBlocked('hard policy failure')

        agent = model.agent(InputGuardrail(guard=guard, parallel=True))
        outputs: list[str] = []

        with anyio.fail_after(READINESS_WAIT_TIMEOUT), pytest.raises(InputBlocked, match='hard policy failure'):
            async with agent.run_stream('hello') as result:
                outputs.append(await result.get_output())

        assert outputs == []
        assert model.closed.is_set()

    async def test_allow_streams_every_chunk(self):
        guard_done = asyncio.Event()

        async def stream_function(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
            yield 'one '
            await guard_done.wait()
            yield 'two '
            yield 'three'

        async def guard(_prompt: str) -> bool:
            guard_done.set()
            return True

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            capabilities=[InputGuardrail(guard=guard, parallel=True)],
        )

        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            async with agent.run_stream('hello') as result:
                deltas = [text async for text in result.stream_text(delta=True, debounce_by=None)]
                assert await result.get_output() == 'one two three'

        assert deltas == ['one ', 'two ', 'three']

    async def test_block_after_the_whole_response_streamed_still_raises(self):
        """A block that arrives after the caller received the whole response is not swallowed.

        `run_stream()` has already settled on the model's output by then, so the block cannot
        replace it and still surfaces as `SkipModelRequest`, as it did before the fix.
        """
        release_guard = asyncio.Event()

        async def guard(_prompt: str) -> GuardrailResult:
            await release_guard.wait()
            return GuardrailResult.block('nope')

        observer = _BlockObserver()
        capabilities: list[AbstractCapability[Any]] = [InputGuardrail(guard=guard, parallel=True), observer]
        agent = Agent(TestModel(custom_output_text='model output'), capabilities=capabilities)

        with anyio.fail_after(READINESS_WAIT_TIMEOUT), pytest.raises(SkipModelRequest):
            async with agent.run_stream('hello') as result:
                assert await result.get_output() == 'model output'
                release_guard.set()
                await observer.blocked.wait()


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


class TestInputGuardrailOrdering:
    """`get_ordering()` placement: innermost so message-morphing capabilities run first."""

    def test_declares_innermost(self):
        ordering = InputGuardrail(guard=lambda _: True).get_ordering()
        assert ordering == CapabilityOrdering(position='innermost')


class TestGuardrailResult:
    """`GuardrailResult.__post_init__` rejects field combinations the four-outcome contract does not allow."""

    def test_allow_rejects_message(self):
        with pytest.raises(UserError, match="'allow'"):
            GuardrailResult(action='allow', message='oops')

    def test_allow_rejects_replacement(self):
        with pytest.raises(UserError, match="'allow'"):
            GuardrailResult(action='allow', replacement='oops')

    def test_replace_requires_replacement(self):
        with pytest.raises(UserError, match="'replace'"):
            GuardrailResult(action='replace')

    def test_retry_requires_message(self):
        with pytest.raises(UserError, match="'retry'"):
            GuardrailResult(action='retry')

    def test_block_without_message_is_valid(self):
        """A `block` with no message is fine: the default kicks in at the use site."""
        result = GuardrailResult(action='block')
        assert result.message is None

    def test_is_frozen(self):
        result = GuardrailResult.allow()
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
