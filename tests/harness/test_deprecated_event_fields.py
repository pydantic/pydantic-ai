"""Renamed `CapabilityEvent` fields keep their old names for attribute access and on load."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from pydantic_ai import AgentStreamEvent, CapabilityEvent
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.code_mode import (
    SpeculativeCallLaunchedEvent,
    SpeculativeCallMissedEvent,
    SpeculativeCodeUpdateEvent,
)
from pydantic_ai_harness.filesystem import FilesSearchedEvent
from pydantic_ai_harness.planning import (
    PlanCompletedEvent,
    PlanCreatedEvent,
    PlanDeletedEvent,
    PlanItem,
    PlanStatusChangedEvent,
    PlanUpdatedEvent,
)
from pydantic_ai_harness.shell import CommandFinishedEvent
from pydantic_ai_harness.subagents import DelegationEndEvent

_EVENTS = TypeAdapter[AgentStreamEvent](AgentStreamEvent)
_ITEM = PlanItem(id='step-1', content='ship')
_PREVIOUS = PlanItem(id='step-1', content='draft')


@pytest.mark.parametrize(
    ('event', 'old', 'new'),
    [
        (
            CommandFinishedEvent(pid=1, output_path='o', status_path='s', exit_code=0, truncated=False, line_count=3),
            'total_lines',
            'line_count',
        ),
        (
            SpeculativeCodeUpdateEvent(code='x = 1\n', closed_statement_count=1),
            'closed_statements',
            'closed_statement_count',
        ),
        (
            SpeculativeCallLaunchedEvent(
                launch_id='l1',
                sandbox_function_name='grep',
                wrapped_tool_name='grep',
                arguments={},
                line_start=1,
                line_end=1,
            ),
            'sandbox_function',
            'sandbox_function_name',
        ),
        (
            SpeculativeCallMissedEvent(sandbox_function_name='grep', wrapped_tool_name='grep', nested_tool_call_id='c'),
            'sandbox_function',
            'sandbox_function_name',
        ),
        (
            FilesSearchedEvent(
                path='.', root_dir='/w', pattern='x', search_kind='grep', match_count=1, truncated=False
            ),
            'search',
            'search_kind',
        ),
        *(
            (event_type(item=_ITEM, previous_item=_PREVIOUS), 'previous_state', 'previous_item')
            for event_type in (
                PlanCreatedEvent,
                PlanUpdatedEvent,
                PlanStatusChangedEvent,
                PlanCompletedEvent,
                PlanDeletedEvent,
            )
        ),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, CapabilityEvent) else None,
)
def test_renamed_field(event: CapabilityEvent, old: str, new: str) -> None:
    payload = _EVENTS.dump_python(event, mode='json')
    assert old not in payload
    assert _EVENTS.validate_python(payload) == event

    payload[old] = payload.pop(new)
    assert _EVENTS.validate_python(payload) == event

    with pytest.warns(HarnessDeprecationWarning, match=f'`{old}` is deprecated, use `{new}` instead'):
        assert getattr(event, old) == getattr(event, new)


def test_delegation_duration_seconds() -> None:
    event = DelegationEndEvent(
        agent_name='worker', outcome='ok', output='done', truncated=False, usage=None, duration_ms=1500
    )
    payload = _EVENTS.dump_python(event, mode='json')
    assert 'duration_seconds' not in payload
    assert _EVENTS.validate_python(payload) == event

    assert _EVENTS.validate_python({**payload, 'duration_seconds': 9}) == event
    del payload['duration_ms']
    assert _EVENTS.validate_python({**payload, 'duration_seconds': 1.5}) == event

    with pytest.warns(HarnessDeprecationWarning, match='`duration_seconds` is deprecated, use `duration_ms` instead'):
        assert event.duration_seconds == 1.5  # pyright: ignore[reportDeprecated]
