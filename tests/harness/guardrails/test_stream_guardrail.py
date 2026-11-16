"""Tests for the `StreamGuardrail` capability."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import replace

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import GuardrailResult, OutputBlocked, StreamGuardrail
from pydantic_ai_harness.guardrails import StreamGuardrailPart


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
        content = part.args if isinstance(part, ToolCallPart) else getattr(part, 'content')
        return f'{type(event).__name__} {event.index} {part.part_kind} {content!r}'
    if isinstance(event, PartDeltaEvent):
        return f'PartDeltaEvent {event.index}'
    return type(event).__name__


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
                'PartStartEvent 2 tool-call \'{"city": \'',
                'PartDeltaEvent 2',
                'PartEndEvent 2 tool-call \'{"city": "Paris"}\'',
                'FunctionToolCallEvent',
                'FunctionToolResultEvent',
                'guard text',
                "PartStartEvent 0 text 'Sunny'",
                'FinalResultEvent',
                "PartEndEvent 0 text 'Sunny'",
                'AgentRunResultEvent',
            ]
        )

    async def test_replace_swaps_the_part_and_drops_its_deltas(self):
        async with _agent(StreamGuardrail(guard=_redact_ssn)).run_stream_events('weather?') as events:
            described = [_describe(event) async for event in events]

        assert described == snapshot(
            [
                "PartStartEvent 0 thinking 'look '",
                'PartDeltaEvent 0',
                "PartEndEvent 0 thinking 'look it up'",
                "PartStartEvent 1 text '[redacted]'",
                'FinalResultEvent',
                "PartEndEvent 1 text '[redacted]'",
                'PartStartEvent 2 tool-call \'{"city": \'',
                'PartDeltaEvent 2',
                'PartEndEvent 2 tool-call \'{"city": "Paris"}\'',
                'FunctionToolCallEvent',
                'FunctionToolResultEvent',
                "PartStartEvent 0 text 'Sunny'",
                'FinalResultEvent',
                "PartEndEvent 0 text 'Sunny'",
                'AgentRunResultEvent',
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
            ["PartStartEvent 0 thinking 'look '", 'PartDeltaEvent 0', "PartEndEvent 0 thinking 'look it up'"]
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

    async def test_replacement_must_be_the_same_part_type(self):
        agent = Agent(
            TestModel(custom_output_text='hi'),
            capabilities=[StreamGuardrail(guard=lambda part: GuardrailResult.replace('plain text'))],
        )
        with pytest.raises(UserError, match='must provide a TextPart to replace the streamed one; guard at position 0'):
            await agent.run('hello')

    def test_is_excluded_from_agent_specs(self):
        assert StreamGuardrail.get_serialization_name() is None

    async def test_a_part_left_incomplete_is_dropped_unscreened(self):
        screened: list[str] = []

        def guard(part: StreamGuardrailPart) -> bool:
            assert isinstance(part, TextPart)
            screened.append(part.content)
            return True

        async def events() -> AsyncIterator[AgentStreamEvent]:
            yield PartStartEvent(index=0, part=TextPart(content='abandoned'))
            yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' secret'))
            yield PartStartEvent(index=0, part=TextPart(content='kept'))
            yield PartEndEvent(index=0, part=TextPart(content='kept'))

        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        released = [event async for event in StreamGuardrail(guard=guard).wrap_run_event_stream(ctx, stream=events())]

        assert [_describe(event) for event in released] == snapshot(
            ["PartStartEvent 0 text 'kept'", "PartEndEvent 0 text 'kept'"]
        )
        assert screened == ['kept']
