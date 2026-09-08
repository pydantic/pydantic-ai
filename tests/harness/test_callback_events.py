"""Capability events replacing deprecated constructor callbacks."""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

import pydantic
import pytest
from inline_snapshot import snapshot
from pydantic_ai import Agent, CapabilityEvent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.compaction import ContextUsageEvent, ReportContextUsage
from pydantic_ai_harness.planning import (
    InMemoryPlanStore,
    PlanCreatedEvent,
    PlanEventEmitter,
    PlanItem,
    Planning,
)
from pydantic_ai_harness.spend import SpendLimits, SpendRecordedEvent
from pydantic_ai_harness.system_reminders import Reminder, ReminderFiredEvent, SystemReminders

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_spend_event_and_legacy_callback() -> None:
    events: list[SpendRecordedEvent] = []
    legacy: list[Any] = []
    hooks = Hooks[Any]()

    @hooks.on.event(SpendRecordedEvent)
    async def record(ctx: RunContext[Any], event: SpendRecordedEvent) -> None:
        events.append(event)

    with pytest.warns(HarnessDeprecationWarning, match=r'SpendLimits\.on_spend.*SpendRecordedEvent.*on_event'):
        capability = SpendLimits[Any](on_spend=legacy.append, price=lambda response: Decimal('1.25'))
    await Agent(TestModel(), capabilities=[capability, hooks]).run('go')

    assert len(legacy) == 1
    assert len(events) == 1
    assert events[0].usd == Decimal('1.25')
    assert events[0].usage == legacy[0].usage


async def test_context_usage_event_and_legacy_callback() -> None:
    events: list[ContextUsageEvent] = []
    legacy: list[Any] = []
    hooks = Hooks[Any]()

    @hooks.on.event(ContextUsageEvent)
    async def record(ctx: RunContext[Any], event: ContextUsageEvent) -> None:
        events.append(event)

    with pytest.warns(HarnessDeprecationWarning, match=r'ReportContextUsage\.on_usage.*ContextUsageEvent.*on_event'):
        capability = ReportContextUsage(on_usage=legacy.append, context_window=1_000)
    await Agent(TestModel(), capabilities=[capability, hooks]).run('go')

    assert len(legacy) == 1
    assert events == [
        ContextUsageEvent(
            used_tokens=legacy[0].used_tokens,
            window_tokens=1_000,
            resolved=True,
            fraction=legacy[0].fraction,
            capability_id='report_context_usage',
        )
    ]


async def test_reminder_event_and_legacy_callback() -> None:
    events: list[ReminderFiredEvent] = []
    legacy: list[str] = []
    hooks = Hooks[Any]()

    @hooks.on.event(ReminderFiredEvent)
    async def record(ctx: RunContext[Any], event: ReminderFiredEvent) -> None:
        events.append(event)

    with pytest.warns(HarnessDeprecationWarning, match=r'SystemReminders\.on_fire.*ReminderFiredEvent.*on_event'):
        capability = SystemReminders(reminders=[Reminder('focus', tag=None)], on_fire=legacy.append)
    await Agent(TestModel(), capabilities=[capability, hooks]).run('go')

    assert legacy == ['focus']
    assert [event.text for event in events] == ['focus']


def _has_tool_return(messages: list[ModelMessage]) -> bool:
    return any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)


async def _add_then_text(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
    if not _has_tool_return(messages):
        yield {0: DeltaToolCall(name='add_task', json_args='{"content":"ship"}', tool_call_id='call_1')}
    else:
        yield 'done'


async def test_plan_event_and_legacy_emitter() -> None:
    events: list[PlanCreatedEvent] = []
    legacy: list[Any] = []
    hooks = Hooks[Any]()

    @hooks.on.event(PlanCreatedEvent)
    async def record(ctx: RunContext[Any], event: PlanCreatedEvent) -> None:
        events.append(event)

    with pytest.warns(HarnessDeprecationWarning, match=r'PlanEventEmitter.*on_event.*PlanCreatedEvent'):
        emitter = PlanEventEmitter()
    emitter.on_created(legacy.append)
    with pytest.warns(HarnessDeprecationWarning, match=r'event_emitter.*on_event.*PlanCreatedEvent'):
        store = InMemoryPlanStore(event_emitter=emitter)

    await Agent(FunctionModel(stream_function=_add_then_text), capabilities=[Planning(store=store), hooks]).run('go')

    assert len(legacy) == 1
    assert [event.item.content for event in events] == ['ship']


def test_new_construction_paths_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error', HarnessDeprecationWarning)
        SpendLimits()
        ReportContextUsage()
        SystemReminders(reminders=[Reminder('focus')])
        InMemoryPlanStore()


def test_callback_replacement_events_serialize() -> None:
    events: list[CapabilityEvent] = [
        SpendRecordedEvent(
            model='test',
            usage=RequestUsage(input_tokens=2, output_tokens=1),
            usd=Decimal('0.50'),
            priced=True,
            budgets=(),
        ),
        ContextUsageEvent(used_tokens=25, window_tokens=100, resolved=True, fraction=0.25),
        ReminderFiredEvent(text='focus'),
        PlanCreatedEvent(item=PlanItem(id='step-1', content='ship')),
    ]
    assert [
        pydantic.TypeAdapter(type(event)).dump_python(event, exclude_none=True, mode='json') for event in events
    ] == snapshot(
        [
            {
                'model': 'test',
                'usage': {
                    'input_tokens': 2,
                    'cache_write_tokens': 0,
                    'cache_read_tokens': 0,
                    'output_tokens': 1,
                    'input_audio_tokens': 0,
                    'cache_audio_read_tokens': 0,
                    'output_audio_tokens': 0,
                    'details': {},
                },
                'usd': '0.50',
                'priced': True,
                'budgets': [],
                'event_kind': 'capability',
                'kind': 'spend_limits.recorded',
            },
            {
                'used_tokens': 25,
                'window_tokens': 100,
                'resolved': True,
                'fraction': 0.25,
                'event_kind': 'capability',
                'kind': 'report_context_usage.measured',
            },
            {'text': 'focus', 'event_kind': 'capability', 'kind': 'system_reminders.fired'},
            {
                'item': {
                    'id': 'step-1',
                    'content': 'ship',
                    'status': 'pending',
                    'active_form': '',
                    'depends_on': [],
                },
                'event_kind': 'capability',
                'kind': 'planning.created',
            },
        ]
    )
