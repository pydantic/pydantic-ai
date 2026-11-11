"""Tests for the names `pydantic_ai_harness.subagents` events had before the event naming pass."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

import pydantic_ai_harness.subagents as subagents
from pydantic_ai.messages import CapabilityEvent
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.subagents import (
    AgentCreatedEvent,
    ChildMessageEvent,
    DelegationFinishedEvent,
    DelegationStartedEvent,
    ParentMessageEvent,
    SubAgentMessageEvent,
    TaskAnsweredEvent,
    TaskCancelRequestedEvent,
    TaskForceCancelRequestedEvent,
    TaskQuestionAskedEvent,
    TaskResumedEvent,
    TaskRetriedEvent,
    TaskSteeredEvent,
)

_EVENT = TypeAdapter[CapabilityEvent](CapabilityEvent)


def _serialized_as(event: CapabilityEvent, new_key: str, old_key: str) -> dict[str, object]:
    """`event` as it was serialized before `old_key` was renamed to `new_key`."""
    payload: dict[str, object] = _EVENT.dump_python(event)
    assert old_key not in payload
    return {old_key if key == new_key else key: value for key, value in payload.items()}


RENAMED: dict[str, type[SubAgentMessageEvent]] = {
    'SubAgentMessage': SubAgentMessageEvent,
    'ParentMessage': ParentMessageEvent,
    'ChildMessage': ChildMessageEvent,
    'TaskFollowUpEvent': TaskResumedEvent,
    'TaskMessageEvent': TaskSteeredEvent,
    'TaskAnswerEvent': TaskAnsweredEvent,
    'TaskCancelRequestEvent': TaskCancelRequestedEvent,
    'TaskForceCancelEvent': TaskForceCancelRequestedEvent,
    'TaskQuestionEvent': TaskQuestionAskedEvent,
    'TaskRetryEvent': TaskRetriedEvent,
}


@pytest.mark.parametrize('old_name', RENAMED)
def test_old_class_names_warn_and_resolve(old_name: str) -> None:
    new_class = RENAMED[old_name]
    message = (
        rf'`pydantic_ai_harness\.subagents\.{old_name}` has been renamed to '
        rf'`pydantic_ai_harness\.subagents\.{new_class.__name__}`\.'
    )
    with pytest.warns(HarnessDeprecationWarning, match=message):
        assert getattr(subagents, old_name) is new_class
    assert new_class.__name__ in subagents.__all__
    assert old_name not in subagents.__all__


def test_agent_created_event_name_is_agent_name() -> None:
    event = AgentCreatedEvent(agent_name='poet', description='d', instructions='i', model=None)
    with pytest.warns(HarnessDeprecationWarning, match='`name` is deprecated, use `agent_name` instead.'):
        assert event.name == 'poet'  # pyright: ignore[reportDeprecated]

    assert _EVENT.validate_python(_EVENT.dump_python(event)) == event
    assert _EVENT.validate_python(_serialized_as(event, 'agent_name', 'name')) == event


@pytest.mark.parametrize(
    'event',
    [
        DelegationStartedEvent(
            agent_name='worker',
            task='t',
            truncated=False,
            model=None,
            task_id='child',
            parent_task_id='parent',
            inherits_tools=False,
        ),
        DelegationFinishedEvent(
            agent_name='worker',
            outcome='ok',
            output='o',
            truncated=False,
            usage=None,
            task_id='child',
            parent_task_id='parent',
            duration_seconds=1.0,
        ),
    ],
)
def test_delegation_event_parent_id_is_parent_task_id(event: DelegationStartedEvent | DelegationFinishedEvent) -> None:
    with pytest.warns(HarnessDeprecationWarning, match='`parent_id` is deprecated, use `parent_task_id` instead.'):
        assert event.parent_id == 'parent'  # pyright: ignore[reportDeprecated]

    assert _EVENT.validate_python(_EVENT.dump_python(event)) == event
    assert _EVENT.validate_python(_serialized_as(event, 'parent_task_id', 'parent_id')) == event
