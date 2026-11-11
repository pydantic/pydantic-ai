"""Harness `CapabilityEvent` class and field names: the deprecated aliases, and the wire kinds and keys a rename must keep."""

from __future__ import annotations

import re
from types import ModuleType

import pytest
from pydantic import TypeAdapter

import pydantic_ai_harness.filesystem as filesystem
import pydantic_ai_harness.shell as shell
import pydantic_ai_harness.step_persistence as step_persistence
import pydantic_ai_harness.subagents as subagents
from pydantic_ai import AgentStreamEvent, CapabilityEvent
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.code_mode import (
    SpeculativeCallLaunchedEvent,
    SpeculativeCallMissedEvent,
    SpeculativeCodeUpdateEvent,
)
from pydantic_ai_harness.planning import (
    PlanCompletedEvent,
    PlanCreatedEvent,
    PlanDeletedEvent,
    PlanItem,
    PlanStatusChangedEvent,
    PlanUpdatedEvent,
)


@pytest.mark.parametrize(
    ('module', 'old_name', 'new_name', 'kind'),
    [
        (filesystem, 'FileChangeRequestEvent', 'FileChangeRequestedEvent', 'file_system.file_change_request'),
        (step_persistence, 'SnapshotSaved', 'SnapshotSavedEvent', 'step_persistence.snapshot_saved'),
        (subagents, 'DelegationStartEvent', 'DelegationStartedEvent', 'sub_agents.delegation_start'),
        (subagents, 'DelegationEndEvent', 'DelegationFinishedEvent', 'sub_agents.delegation_end'),
    ],
)
def test_renamed_event_keeps_its_old_name_and_kind(module: ModuleType, old_name: str, new_name: str, kind: str) -> None:
    event_type = getattr(module, new_name)
    message = (
        f'`{module.__name__}.{old_name}` has been renamed to `{module.__name__}.{new_name}`. '
        'Update your imports; this deprecated alias will be removed in a future release.'
    )
    with pytest.warns(HarnessDeprecationWarning, match=re.escape(message)):
        assert getattr(module, old_name) is event_type
    assert event_type.kind == kind


def test_derived_kinds_are_unchanged() -> None:
    kinds: dict[type[CapabilityEvent], str] = {
        filesystem.DirectoryCreatedEvent: 'file_system.directory_created',
        filesystem.DirectoryListedEvent: 'file_system.directory_listed',
        filesystem.FileEditedEvent: 'file_system.file_edited',
        filesystem.FileReadEvent: 'file_system.file_read',
        filesystem.FileWrittenEvent: 'file_system.file_written',
        filesystem.FilesSearchedEvent: 'file_system.files_searched',
        shell.CommandFinishedEvent: 'shell.command_finished',
        shell.CommandOutputEvent: 'shell.command_output',
        shell.CommandStartedEvent: 'shell.command_started',
        subagents.AgentCreatedEvent: 'sub_agents.agent_created',
        subagents.TaskAnsweredEvent: 'sub_agents.task_answer',
        subagents.TaskAssignedEvent: 'sub_agents.task_assigned',
        subagents.TaskBackgroundedEvent: 'sub_agents.task_backgrounded',
        subagents.TaskCancelRequestedEvent: 'sub_agents.task_cancel_request',
        subagents.TaskCancelledEvent: 'sub_agents.task_cancelled',
        subagents.TaskCompletedEvent: 'sub_agents.task_completed',
        subagents.TaskFailedEvent: 'sub_agents.task_failed',
        subagents.TaskResumedEvent: 'sub_agents.task_follow_up',
        subagents.TaskForceCancelRequestedEvent: 'sub_agents.task_force_cancel',
        subagents.TaskSteeredEvent: 'sub_agents.task_message',
        subagents.TaskProgressEvent: 'sub_agents.task_progress',
        subagents.TaskQuestionAskedEvent: 'sub_agents.task_question',
        subagents.TaskRetriedEvent: 'sub_agents.task_retry',
        subagents.TaskStartedEvent: 'sub_agents.task_started',
    }
    assert {event_type: event_type.kind for event_type in kinds} == kinds
    assert step_persistence.SnapshotSavedEvent.kind == f'{step_persistence.STEP_PERSISTENCE_EVENTS}.snapshot_saved'


_EVENTS = TypeAdapter[AgentStreamEvent](AgentStreamEvent)
_ITEM = PlanItem(id='step-1', content='ship')
_PREVIOUS = PlanItem(id='step-1', content='draft')


@pytest.mark.parametrize(
    ('event', 'old', 'new'),
    [
        (
            shell.CommandFinishedEvent(
                pid=1, output_path='o', status_path='s', exit_code=0, truncated=False, line_count=3
            ),
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
            filesystem.FilesSearchedEvent(
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
    event = subagents.DelegationFinishedEvent(
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
