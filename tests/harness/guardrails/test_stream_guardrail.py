"""Tests for the `StreamGuardrail` capability."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from inline_snapshot import snapshot
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai.capabilities import CapabilityOrdering, Instrumentation, ProcessEventStream
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    ModelResponseEndEvent,
    NativeToolCallPart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.realtime import RealtimeModel, RealtimeModelProfile, RealtimeModelSettings
from pydantic_ai.realtime.codec import RealtimeConnection
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import GuardrailResult, OutputBlocked, StreamGuardrail
from pydantic_ai_harness.guardrails import StreamGuardrailPart
from tests.harness._recording_durability import RecordingDurability


async def _stream(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls | DeltaThinkingCalls]:
    last = messages[-1]
    if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
        yield 'Sunny'
        return
    yield {0: DeltaThinkingPart(content='look ')}
    yield {0: DeltaThinkingPart(content='it up')}
    yield 'Call '
    yield 'SSN 123'
    yield {2: DeltaToolCall(name='weather', json_args='{"city": ', tool_call_id='call-1')}
    yield {2: DeltaToolCall(json_args='"Paris"}')}


def _agent(guardrail: StreamGuardrail[object]) -> Agent[object, str]:
    agent = Agent(FunctionModel(stream_function=_stream), capabilities=[guardrail])

    @agent.tool_plain
    def weather(city: str) -> str:
        return f'{city}: sunny'

    return agent


def _describe(event: AgentStreamEvent | AgentRunResultEvent[str]) -> str:
    if isinstance(event, PartStartEvent | PartEndEvent):
        part = event.part
        content = part.args if isinstance(part, ToolCallPart) else getattr(part, 'content', None)
        return f'{type(event).__name__} {event.index} {part.part_kind} {content!r}'
    if isinstance(event, PartDeltaEvent):
        return f'PartDeltaEvent {event.index}'
    return type(event).__name__


class _RealtimeModel(RealtimeModel):
    @property
    def model_name(self) -> str:
        return 'fake-realtime'  # pragma: no cover

    @property
    def system(self) -> str:
        return 'fake'  # pragma: no cover

    @property
    def profile(self) -> RealtimeModelProfile:
        return RealtimeModelProfile()

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[RealtimeConnection]:  # pragma: no cover
        raise NotImplementedError
        yield


def _redact_ssn(part: StreamGuardrailPart) -> GuardrailResult:
    if isinstance(part, TextPart) and 'SSN' in part.content:
        return GuardrailResult.replace(replace(part, content='[redacted]'))
    return GuardrailResult.allow()


class TestStreamGuardrail:
    """`StreamGuardrail` driven through `Agent` runs and its event stream hook."""

    async def test_each_part_reaches_the_caller_only_after_the_guard_saw_it_complete(self):
        log: list[str] = []

        def guard(part: StreamGuardrailPart) -> bool:
            log.append(f'guard {part.part_kind}')
            return True

        async with _agent(StreamGuardrail(guard=guard)).run_stream_events('weather?') as events:
            async for event in events:
                log.append(_describe(event))

        assert log == snapshot(
            [
                'ModelResponseStartEvent',
                'guard thinking',
                "PartStartEvent 0 thinking 'look '",
                'PartDeltaEvent 0',
                "PartEndEvent 0 thinking 'look it up'",
                'guard text',
                "PartStartEvent 1 text 'Call '",
                'FinalResultEvent',
                'PartDeltaEvent 1',
                "PartEndEvent 1 text 'Call SSN 123'",
                'guard tool-call',
                'PartStartEvent 2 tool-call \'{"city": "Paris"}\'',
                'PartEndEvent 2 tool-call \'{"city": "Paris"}\'',
                'ModelResponseEndEvent',
                'FunctionToolCallEvent',
                'FunctionToolResultEvent',
                'ModelResponseStartEvent',
                'guard text',
                "PartStartEvent 0 text 'Sunny'",
                'FinalResultEvent',
                "PartEndEvent 0 text 'Sunny'",
                'ModelResponseEndEvent',
                'AgentRunResultEvent',
            ]
        )

    async def test_replace_swaps_the_part_and_drops_its_deltas(self):
        async with _agent(StreamGuardrail(guard=_redact_ssn)).run_stream_events('weather?') as events:
            described = [_describe(event) async for event in events]

        assert described == snapshot(
            [
                'ModelResponseStartEvent',
                "PartStartEvent 0 thinking 'look '",
                'PartDeltaEvent 0',
                "PartEndEvent 0 thinking 'look it up'",
                "PartStartEvent 1 text '[redacted]'",
                'FinalResultEvent',
                "PartEndEvent 1 text '[redacted]'",
                'PartStartEvent 2 tool-call \'{"city": "Paris"}\'',
                'PartEndEvent 2 tool-call \'{"city": "Paris"}\'',
                'ModelResponseEndEvent',
                'FunctionToolCallEvent',
                'FunctionToolResultEvent',
                'ModelResponseStartEvent',
                "PartStartEvent 0 text 'Sunny'",
                'FinalResultEvent',
                "PartEndEvent 0 text 'Sunny'",
                'ModelResponseEndEvent',
                'AgentRunResultEvent',
            ]
        )

    async def test_a_response_end_carries_the_parts_as_screened(self):
        """The response a `ModelResponseEndEvent` carries doesn't leak what the guard replaced."""
        async with _agent(StreamGuardrail(guard=_redact_ssn)).run_stream_events('weather?') as events:
            ends = [event async for event in events if isinstance(event, ModelResponseEndEvent)]

        assert [event.response.parts if event.response else None for event in ends] == snapshot(
            [
                [
                    ThinkingPart(content='look it up'),
                    TextPart(content='[redacted]'),
                    ToolCallPart(tool_name='weather', args='{"city": "Paris"}', tool_call_id='call-1'),
                ],
                [TextPart(content='Sunny')],
            ]
        )

    async def test_replace_leaves_the_run_output_unchanged(self):
        texts: list[str] = []

        async def handler(ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in events:
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    texts.append(event.part.content)

        agent = Agent(TestModel(custom_output_text='leaks SSN here'), capabilities=[StreamGuardrail(guard=_redact_ssn)])
        result = await agent.run('hello', event_stream_handler=handler)

        assert texts == ['[redacted]']
        assert result.output == 'leaks SSN here'

    async def test_run_stream_raises_a_block_but_streams_text_a_replace_did_not_reach(self):
        blocking = Agent(TestModel(custom_output_text='hi'), capabilities=[StreamGuardrail(guard=lambda part: False)])
        with pytest.raises(OutputBlocked):
            async with blocking.run_stream('hello'):
                pass  # pragma: no cover

        redacting = Agent(
            TestModel(custom_output_text='leaks SSN here'), capabilities=[StreamGuardrail(guard=_redact_ssn)]
        )
        async with redacting.run_stream('hello') as result:
            assert [text async for text in result.stream_text(debounce_by=None)] == ['leaks SSN here']

    async def test_process_event_stream_sees_events_before_the_guard(self):
        texts: list[str] = []

        async def handler(ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in events:
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    texts.append(event.part.content)
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    texts.append(event.delta.content_delta)

        agent = Agent(
            TestModel(custom_output_text='leaks SSN here'),
            capabilities=[ProcessEventStream(handler), StreamGuardrail(guard=_redact_ssn)],
        )
        await agent.run('hello')

        assert ''.join(texts) == 'leaks SSN here'

    async def test_block_raises_before_the_part_is_released(self):
        seen: list[AgentStreamEvent | AgentRunResultEvent[str]] = []

        def guard(part: StreamGuardrailPart) -> GuardrailResult:
            if isinstance(part, TextPart):
                return GuardrailResult.block('no SSNs')
            return GuardrailResult.allow()

        with pytest.raises(OutputBlocked, match='no SSNs'):
            async with _agent(StreamGuardrail(guard=guard)).run_stream_events('weather?') as events:
                async for event in events:
                    seen.append(event)

        assert [_describe(event) for event in seen] == snapshot(
            [
                'ModelResponseStartEvent',
                "PartStartEvent 0 thinking 'look '",
                'PartDeltaEvent 0',
                "PartEndEvent 0 thinking 'look it up'",
            ]
        )

    async def test_block_under_run_uses_the_default_message(self):
        agent = Agent(TestModel(custom_output_text='hi'), capabilities=[StreamGuardrail(guard=lambda part: False)])
        with pytest.raises(OutputBlocked, match=r'Streamed part blocked by stream guardrail\.'):
            await agent.run('hello')

    async def test_guard_chain_sees_the_replacement_and_run_context(self):
        seen: list[str] = []

        def check(ctx: RunContext[object], part: StreamGuardrailPart) -> bool:
            assert ctx.prompt == 'hello'
            assert isinstance(part, TextPart)
            seen.append(part.content)
            return True

        agent = Agent(
            TestModel(custom_output_text='leaks SSN here'), capabilities=[StreamGuardrail(guard=[_redact_ssn, check])]
        )
        await agent.run('hello')

        assert seen == ['[redacted]']

    @pytest.mark.parametrize('verdict', [GuardrailResult.retry('again'), GuardrailResult.approve()])
    async def test_retry_and_approve_are_refused(self, verdict: GuardrailResult):
        agent = Agent(TestModel(custom_output_text='hi'), capabilities=[StreamGuardrail(guard=lambda part: verdict)])
        with pytest.raises(UserError, match=f'cannot return GuardrailResult.{verdict.action}'):
            await agent.run('hello')

    @pytest.mark.parametrize('replacement', ['plain text', ThinkingPart(content='hi')])
    async def test_replacement_must_be_the_same_part_type(self, replacement: object):
        agent = Agent(
            TestModel(custom_output_text='hi'),
            capabilities=[StreamGuardrail(guard=lambda part: GuardrailResult.replace(replacement))],
        )
        with pytest.raises(UserError, match='must provide a TextPart to replace the streamed one; guard at position 0'):
            await agent.run('hello')

    def test_is_excluded_from_agent_specs(self):
        assert StreamGuardrail.get_serialization_name() is None

    def test_declares_outermost_inside_instrumentation(self):
        ordering = StreamGuardrail(guard=lambda part: True).get_ordering()
        assert ordering == CapabilityOrdering(position='outermost', wrapped_by=[Instrumentation])

    @pytest.mark.parametrize('include_content', [True, False])
    async def test_block_and_redaction_spans_gate_content(self, include_content: bool):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        ctx = RunContext(
            deps=None,
            model=TestModel(),
            usage=RunUsage(),
            tracer=provider.get_tracer('test'),
            trace_include_content=include_content,
        )

        def guard(part: StreamGuardrailPart) -> GuardrailResult:
            assert isinstance(part, TextPart)
            if part.content == 'secret':
                return GuardrailResult.replace(TextPart(content='[redacted]'))
            return GuardrailResult.block('no SSNs')

        async def events() -> AsyncIterator[AgentStreamEvent]:
            yield PartStartEvent(index=0, part=TextPart(content='secret'))
            yield PartEndEvent(index=0, part=TextPart(content='secret'))
            yield PartStartEvent(index=1, part=TextPart(content='SSN'))
            yield PartEndEvent(index=1, part=TextPart(content='SSN'))

        with pytest.raises(OutputBlocked):
            async for _ in StreamGuardrail(guard=guard).wrap_run_event_stream(ctx, stream=events()):
                pass

        content = {'guardrail.replacement': str(TextPart(content='[redacted]'))} if include_content else {}
        message = {'guardrail.message': 'no SSNs'} if include_content else {}
        assert [(span.name, dict(span.attributes or {})) for span in exporter.get_finished_spans()] == [
            ('guardrail redacted stream', {'guardrail.direction': 'stream', 'guardrail.action': 'replace', **content}),
            ('guardrail blocked stream', {'guardrail.direction': 'stream', 'guardrail.action': 'block', **message}),
        ]

    async def test_tool_call_args_streamed_after_another_part_started_are_screened(self):
        async def interleaved(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            yield {0: DeltaToolCall(name='weather', json_args='{"city": ', tool_call_id='call-1')}
            yield {1: DeltaToolCall(name='weather', json_args='{"city": "Rome"}', tool_call_id='call-2')}
            yield {0: DeltaToolCall(json_args='"SSN 123"}')}

        screened: list[object] = []

        def guard(part: StreamGuardrailPart) -> GuardrailResult:
            assert isinstance(part, ToolCallPart)
            screened.append(part.args)
            return GuardrailResult.block('no SSNs') if 'SSN' in str(part.args) else GuardrailResult.allow()

        agent = Agent(FunctionModel(stream_function=interleaved), capabilities=[StreamGuardrail(guard=guard)])
        seen: list[AgentStreamEvent | AgentRunResultEvent[str]] = []
        with pytest.raises(OutputBlocked, match='no SSNs'):
            async with agent.run_stream_events('weather?') as events:
                async for event in events:
                    seen.append(event)

        # Only the response's start, which carries no content, reached the caller.
        assert [type(event).__name__ for event in seen] == ['ModelResponseStartEvent']
        assert screened == ['{"city": "SSN 123"}']

    async def test_a_closed_part_streamed_again_is_screened_again(self):
        async def events() -> AsyncIterator[AgentStreamEvent]:
            yield PartStartEvent(index=0, part=TextPart(content='Hi'))
            yield PartEndEvent(index=0, part=TextPart(content='Hi'))
            yield PartStartEvent(index=1, part=TextPart(content='second'))
            yield PartStartEvent(index=0, part=TextPart(content='Hi'))
            yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' SSN'))
            yield PartEndEvent(index=0, part=TextPart(content='Hi SSN'))
            yield PartEndEvent(index=1, part=TextPart(content='second'))

        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        guardrail = StreamGuardrail(guard=_redact_ssn)
        released = [event async for event in guardrail.wrap_run_event_stream(ctx, stream=events())]

        assert [_describe(event) for event in released] == snapshot(
            [
                "PartStartEvent 0 text 'Hi'",
                "PartEndEvent 0 text 'Hi'",
                "PartStartEvent 1 text 'second'",
                "PartStartEvent 0 text '[redacted]'",
                "PartEndEvent 0 text '[redacted]'",
                "PartEndEvent 1 text 'second'",
            ]
        )

    async def test_a_verdict_on_a_restarted_part_leaves_its_earlier_screened_events_alone(self):
        call = ToolCallPart(tool_name='weather', args='{"city": "Paris"}')

        async def events() -> AsyncIterator[AgentStreamEvent]:
            yield PartStartEvent(index=1, part=call)
            yield PartEndEvent(index=1, part=call)
            yield PartStartEvent(index=0, part=TextPart(content='fine'))
            yield PartEndEvent(index=0, part=TextPart(content='fine'))
            yield PartStartEvent(index=0, part=TextPart(content='abandoned'))
            yield PartStartEvent(index=0, part=TextPart(content='SSN'))
            yield PartEndEvent(index=0, part=TextPart(content='SSN'))

        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        guardrail = StreamGuardrail(guard=_redact_ssn)
        released = [event async for event in guardrail.wrap_run_event_stream(ctx, stream=events())]

        assert [_describe(event) for event in released] == snapshot(
            [
                'PartStartEvent 1 tool-call \'{"city": "Paris"}\'',
                'PartEndEvent 1 tool-call \'{"city": "Paris"}\'',
                "PartStartEvent 0 text 'fine'",
                "PartEndEvent 0 text 'fine'",
                "PartStartEvent 0 text '[redacted]'",
                "PartEndEvent 0 text '[redacted]'",
            ]
        )

    async def test_an_allowed_tool_call_is_released_as_its_screened_final_part(self):
        async def events() -> AsyncIterator[AgentStreamEvent]:
            yield PartStartEvent(index=0, part=ToolCallPart(tool_name='weather', args={'city': 'SSN 123'}))
            yield PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'city': 'Paris'}))
            yield PartEndEvent(index=0, part=ToolCallPart(tool_name='weather', args={'city': 'Paris'}))

        def guard(part: StreamGuardrailPart) -> bool:
            return 'SSN' not in str(part)

        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        released = [event async for event in StreamGuardrail(guard=guard).wrap_run_event_stream(ctx, stream=events())]

        assert [_describe(event) for event in released] == snapshot(
            ["PartStartEvent 0 tool-call {'city': 'Paris'}", "PartEndEvent 0 tool-call {'city': 'Paris'}"]
        )

    async def test_refuses_durable_execution(self):
        agent = Agent(
            TestModel(custom_output_text='hi'),
            name='streamed',
            capabilities=[StreamGuardrail(guard=lambda part: True), RecordingDurability()],
        )
        with pytest.raises(UserError, match='StreamGuardrail cannot run under durable execution'):
            await agent.run('hello')

    async def test_refuses_a_realtime_session(self):
        agent = Agent(TestModel(), capabilities=[StreamGuardrail(guard=lambda part: True)])
        with pytest.raises(UserError, match='StreamGuardrail cannot run in a realtime session'):
            async with agent.realtime(_RealtimeModel()).session():
                pass  # pragma: no cover

    async def test_a_blocked_tool_call_span_names_the_tool(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), tracer=provider.get_tracer('test'))
        call = ToolCallPart(tool_name='weather', args='{"city": "Paris"}')

        async def events() -> AsyncIterator[AgentStreamEvent]:
            yield PartStartEvent(index=0, part=call)
            yield PartEndEvent(index=0, part=call)

        with pytest.raises(OutputBlocked):
            async for _ in StreamGuardrail(guard=lambda part: False).wrap_run_event_stream(ctx, stream=events()):
                pass  # pragma: no cover

        assert [dict(span.attributes or {}) for span in exporter.get_finished_spans()] == [
            {'guardrail.direction': 'stream', 'guardrail.action': 'block', 'guardrail.tool': 'weather'}
        ]

    async def test_other_parts_pass_through_unscreened(self):
        async def events() -> AsyncIterator[AgentStreamEvent]:
            yield PartStartEvent(index=0, part=NativeToolCallPart(tool_name='web_search'))

        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        guardrail = StreamGuardrail(guard=lambda part: False)
        released = [event async for event in guardrail.wrap_run_event_stream(ctx, stream=events())]

        assert [_describe(event) for event in released] == ['PartStartEvent 0 builtin-tool-call None']

    async def test_tool_call_replace_is_refused(self):
        def guard(part: StreamGuardrailPart) -> GuardrailResult:
            return GuardrailResult.replace(part)

        with pytest.raises(UserError, match=r'cannot return GuardrailResult.replace\(\) for a tool call'):
            async with _agent(StreamGuardrail(guard=guard)).run_stream_events('weather?') as events:
                async for _ in events:
                    pass

    @pytest.mark.parametrize(
        'stream',
        [
            pytest.param(
                [
                    PartStartEvent(index=0, part=TextPart(content='abandoned')),
                    PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' secret')),
                    PartStartEvent(index=0, part=TextPart(content='kept')),
                    PartEndEvent(index=0, part=TextPart(content='kept')),
                ],
                id='restarted-while-streaming',
            ),
            pytest.param(
                [
                    PartStartEvent(index=0, part=ToolCallPart(tool_name='weather', args='{"city": "SSN"}')),
                    PartEndEvent(index=0, part=ToolCallPart(tool_name='weather', args='{"city": "SSN"}')),
                    PartStartEvent(index=0, part=TextPart(content='kept')),
                    PartEndEvent(index=0, part=TextPart(content='kept')),
                ],
                id='restarted-while-held',
            ),
            pytest.param(
                [
                    PartStartEvent(index=0, part=TextPart(content='kept')),
                    PartEndEvent(index=0, part=TextPart(content='kept')),
                    PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' secret')),
                    PartStartEvent(index=1, part=TextPart(content='never ends')),
                    PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' secret')),
                ],
                id='late-delta-and-incomplete-part',
            ),
        ],
    )
    async def test_unscreened_content_is_dropped(self, stream: list[AgentStreamEvent]):
        screened: list[str] = []

        def guard(part: StreamGuardrailPart) -> bool:
            screened.append(str(part))
            return True

        async def events() -> AsyncIterator[AgentStreamEvent]:
            for event in stream:
                yield event

        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        released = [event async for event in StreamGuardrail(guard=guard).wrap_run_event_stream(ctx, stream=events())]

        assert [_describe(event) for event in released] == [
            "PartStartEvent 0 text 'kept'",
            "PartEndEvent 0 text 'kept'",
        ]
        assert screened == [str(TextPart(content='kept'))]
