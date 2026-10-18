"""Compaction lifecycle event tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pydantic
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, CompactionEndEvent, CompactionStartEvent, Hooks, on_event
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.compaction import (
    FallbackCompaction,
    SlidingWindowCompaction,
    TieredCompaction,
)


def _history(*values: str) -> list[ModelMessage]:
    return [ModelRequest(parts=[UserPromptPart(value)]) for value in values]


@dataclass
class _CancelFirst(AbstractCapability[Any]):
    events: list[CompactionStartEvent | CompactionEndEvent]
    cancelled: bool = False

    @on_event(CompactionStartEvent)
    async def cancel_first(self, ctx: RunContext[Any], event: CompactionStartEvent) -> None:
        self.events.append(event)
        if not self.cancelled:
            event.cancel('activity in progress')
            self.cancelled = True

    @on_event(CompactionEndEvent)
    async def record_end(self, ctx: RunContext[Any], event: CompactionEndEvent) -> None:
        self.events.append(event)


async def test_cancelled_attempt_retries_and_then_emits_end() -> None:
    events: list[CompactionStartEvent | CompactionEndEvent] = []
    listener = _CancelFirst(events)
    compaction = SlidingWindowCompaction(
        max_messages=1,
        keep_messages=1,
        preserve_first_user_message=False,
        receipts=True,
    )
    agent = Agent(TestModel(custom_output_text='ok'), capabilities=[compaction, listener])

    first = await agent.run('first trigger', message_history=_history('old one', 'old two'))
    assert len(events) == 1
    before = events[0]
    assert isinstance(before, CompactionStartEvent)
    assert before.strategy == 'sliding_window'
    assert before.messages_before == 2
    assert before.tokens_before is not None
    assert before.cancelled
    assert 'History before this point' not in first.all_messages_json().decode()

    second = await agent.run('second trigger', message_history=first.all_messages())
    assert [type(event) for event in events] == [CompactionStartEvent, CompactionStartEvent, CompactionEndEvent]
    end = events[-1]
    retry_before = events[-2]
    assert isinstance(end, CompactionEndEvent)
    assert isinstance(retry_before, CompactionStartEvent)
    assert end.messages_before == retry_before.messages_before
    assert end.messages_after < end.messages_before
    assert end.tokens_before == retry_before.tokens_before
    dropped_messages = end.messages_before - end.messages_after + 1  # the receipt itself adds one message
    assert f'({dropped_messages} messages,' in second.all_messages_json().decode()


async def test_hooks_capability_can_cancel_compaction() -> None:
    hooks = Hooks[Any]()

    @hooks.on.event(CompactionStartEvent)
    async def hold(ctx: RunContext[Any], event: CompactionStartEvent) -> None:
        event.cancel('application is mid-activity')

    history = _history('old one', 'old two')
    result = await Agent(
        TestModel(custom_output_text='ok'),
        capabilities=[SlidingWindowCompaction(max_messages=1, keep_messages=1), hooks],
    ).run('trigger', message_history=history)

    serialized = result.all_messages_json().decode()
    assert 'old one' in serialized
    assert 'old two' in serialized


async def test_agent_listener_can_cancel_compaction() -> None:
    """The docs' application-side path: `@agent.on_event`, no capability of its own."""
    agent = Agent(
        TestModel(custom_output_text='ok'),
        capabilities=[SlidingWindowCompaction(max_messages=1, keep_messages=1)],
    )
    seen: list[str] = []

    @agent.on_event(CompactionStartEvent)
    async def hold(ctx: RunContext[None], event: CompactionStartEvent) -> None:
        seen.append(event.strategy)
        event.cancel('application is mid-activity')

    result = await agent.run('trigger', message_history=_history('old one', 'old two'))

    assert seen == ['sliding_window']
    serialized = result.all_messages_json().decode()
    assert 'old one' in serialized
    assert 'old two' in serialized


@dataclass
class _RecordAll(AbstractCapability[Any]):
    events: list[CompactionStartEvent | CompactionEndEvent]

    @on_event(CompactionStartEvent, CompactionEndEvent)
    async def record(self, ctx: RunContext[Any], event: CompactionStartEvent | CompactionEndEvent) -> None:
        self.events.append(event)


def _ten_per_char(text: str) -> int:
    return len(text) * 10


async def test_tier_events_measure_with_the_configured_tokenizer() -> None:
    events: list[CompactionStartEvent | CompactionEndEvent] = []
    tiered = TieredCompaction(
        tiers=[SlidingWindowCompaction(max_messages=1, keep_messages=1, preserve_first_user_message=False)],
        target_tokens=1,
        tokenizer=_ten_per_char,
    )
    agent = Agent(TestModel(custom_output_text='ok'), capabilities=[tiered, _RecordAll(events)])

    await agent.run('trigger', message_history=_history('old one', 'old two'))

    assert [(type(event).__name__, event.strategy) for event in events] == [
        ('CompactionStartEvent', 'tiered'),
        ('CompactionStartEvent', 'sliding_window'),
        ('CompactionEndEvent', 'sliding_window'),
        ('CompactionEndEvent', 'tiered'),
    ]
    tiered_start, tier_start, tier_end, tiered_end = events
    # The outer event is measured with the configured tokenizer; the per-tier events must
    # agree with it rather than fall back to the four-characters-per-token heuristic.
    assert tier_start.tokens_before == tiered_start.tokens_before
    assert isinstance(tier_end, CompactionEndEvent)
    assert isinstance(tiered_end, CompactionEndEvent)
    assert tier_end.tokens_after == tiered_end.tokens_after
    assert tiered_start.tokens_before is not None and tiered_start.tokens_before % 10 == 0


async def test_fallback_events_measure_with_the_configured_tokenizer() -> None:
    events: list[CompactionStartEvent | CompactionEndEvent] = []
    fallback = FallbackCompaction(
        [SlidingWindowCompaction(max_messages=1, keep_messages=1, preserve_first_user_message=False)],
        max_tokens=1,
        tokenizer=_ten_per_char,
    )
    agent = Agent(TestModel(custom_output_text='ok'), capabilities=[fallback, _RecordAll(events)])

    await agent.run('trigger', message_history=_history('old one', 'old two'))

    assert [(type(event).__name__, event.strategy) for event in events] == [
        ('CompactionStartEvent', 'fallback'),
        ('CompactionStartEvent', 'sliding_window'),
        ('CompactionEndEvent', 'sliding_window'),
        ('CompactionEndEvent', 'fallback'),
    ]
    fallback_start, candidate_start, candidate_end, fallback_end = events
    assert candidate_start.tokens_before == fallback_start.tokens_before
    assert isinstance(candidate_end, CompactionEndEvent)
    assert isinstance(fallback_end, CompactionEndEvent)
    assert candidate_end.tokens_after == fallback_end.tokens_after
    assert fallback_start.tokens_before is not None and fallback_start.tokens_before % 10 == 0


class _InPlaceStrategy:
    strategy_id = 'in_place'

    async def compact(self, messages: list[ModelMessage], ctx: RunContext[Any]) -> list[ModelMessage]:
        messages.pop(0)
        return messages


async def test_in_place_strategy_reports_its_end_event() -> None:
    events: list[CompactionStartEvent | CompactionEndEvent] = []
    tiered = TieredCompaction(tiers=[_InPlaceStrategy()], target_tokens=1)
    agent = Agent(TestModel(custom_output_text='ok'), capabilities=[tiered, _RecordAll(events)])

    result = await agent.run('trigger', message_history=_history('old one', 'old two'))

    assert [(type(event).__name__, event.strategy) for event in events] == [
        ('CompactionStartEvent', 'tiered'),
        ('CompactionStartEvent', 'in_place'),
        ('CompactionEndEvent', 'in_place'),
        ('CompactionEndEvent', 'tiered'),
    ]
    for end in events[2:]:
        assert isinstance(end, CompactionEndEvent)
        assert (end.messages_before, end.messages_after) == (2, 1)
    assert 'old one' not in result.all_messages_json().decode()


@dataclass
class _FailingStrategy:
    calls: list[str]

    async def compact(self, messages: list[ModelMessage], ctx: RunContext[Any]) -> list[ModelMessage]:
        self.calls.append('failed')
        raise RuntimeError('failed')


@dataclass
class _DroppingStrategy:
    calls: list[str]

    async def compact(self, messages: list[ModelMessage], ctx: RunContext[Any]) -> list[ModelMessage]:
        self.calls.append('dropped')
        return messages[1:]


async def test_fallback_cancellation_does_not_advance_but_failure_does() -> None:
    calls: list[str] = []
    fallback = FallbackCompaction(
        [_FailingStrategy(calls), _DroppingStrategy(calls)],
        fallback_on=(RuntimeError,),
    )
    cancelled = False

    @dataclass
    class Listener(AbstractCapability[Any]):
        @on_event(CompactionStartEvent)
        async def cancel_failure(self, ctx: RunContext[Any], event: CompactionStartEvent) -> None:
            nonlocal cancelled
            if event.strategy == 'failing_strategy' and not cancelled:
                event.cancel()
                cancelled = True

    tiered = TieredCompaction(tiers=[fallback], target_tokens=1)
    agent = Agent(TestModel(custom_output_text='ok'), capabilities=[tiered, Listener()])
    first = await agent.run('trigger', message_history=_history('x' * 40, 'y' * 40))
    assert calls == []

    await agent.run('retry', message_history=first.all_messages())
    assert calls == ['failed', 'dropped']


def test_events_serialize() -> None:
    adapter = pydantic.TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    events: list[AgentStreamEvent] = [
        CompactionStartEvent(strategy='sliding', messages_before=8, tokens_before=120),
        CompactionEndEvent(
            strategy='sliding',
            messages_before=8,
            messages_after=3,
            tokens_before=120,
            tokens_after=40,
        ),
    ]

    assert [adapter.dump_python(event, exclude_none=True) for event in events] == snapshot(
        [
            {
                'strategy': 'sliding',
                'messages_before': 8,
                'tokens_before': 120,
                'cancelled': False,
                'event_kind': 'capability',
                'kind': 'compaction.start',
            },
            {
                'strategy': 'sliding',
                'messages_before': 8,
                'messages_after': 3,
                'tokens_before': 120,
                'tokens_after': 40,
                'event_kind': 'capability',
                'kind': 'compaction.end',
            },
        ]
    )
