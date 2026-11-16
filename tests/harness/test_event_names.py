"""Harness `CapabilityEvent` class and field names: the wire kinds and keys a rename must keep."""

from __future__ import annotations

from types import ModuleType

import pytest
from pydantic import TypeAdapter

import pydantic_ai_harness.file_system as file_system
import pydantic_ai_harness.shell as shell
import pydantic_ai_harness.step_persistence as step_persistence
import pydantic_ai_harness.sub_agents as sub_agents
from pydantic_ai import AgentStreamEvent, CapabilityEvent
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
        (file_system, 'FileChangeRequestEvent', 'FileChangeRequestedEvent', 'file_system.file_change_request'),
        (step_persistence, 'SnapshotSaved', 'SnapshotSavedEvent', 'step_persistence.snapshot_saved'),
        (sub_agents, 'DelegationStartEvent', 'DelegationStartedEvent', 'sub_agents.delegation_start'),
        (sub_agents, 'DelegationEndEvent', 'DelegationFinishedEvent', 'sub_agents.delegation_end'),
        (sub_agents, 'SubAgentMessage', 'SubAgentMessageEvent', ''),
        (sub_agents, 'ParentMessage', 'ParentMessageEvent', ''),
        (sub_agents, 'ChildMessage', 'ChildMessageEvent', ''),
        (sub_agents, 'TaskFollowUpEvent', 'TaskResumedEvent', 'sub_agents.task_follow_up'),
        (sub_agents, 'TaskMessageEvent', 'TaskSteeredEvent', 'sub_agents.task_message'),
        (sub_agents, 'TaskAnswerEvent', 'TaskAnsweredEvent', 'sub_agents.task_answer'),
        (sub_agents, 'TaskCancelRequestEvent', 'TaskCancelRequestedEvent', 'sub_agents.task_cancel_request'),
        (sub_agents, 'TaskForceCancelEvent', 'TaskForceCancelRequestedEvent', 'sub_agents.task_force_cancel'),
        (sub_agents, 'TaskQuestionEvent', 'TaskQuestionAskedEvent', 'sub_agents.task_question'),
        (sub_agents, 'TaskRetryEvent', 'TaskRetriedEvent', 'sub_agents.task_retry'),
    ],
)
def test_renamed_event_keeps_its_old_kind(module: ModuleType, old_name: str, new_name: str, kind: str) -> None:
    """The old class name is gone, but the wire kind is unchanged. An abstract event has no kind, which is the empty string."""
    assert getattr(module, new_name).kind == kind
    assert new_name in module.__all__
    assert not hasattr(module, old_name)


def test_derived_kinds_are_unchanged() -> None:
    kinds: dict[type[CapabilityEvent], str] = {
        file_system.DirectoryCreatedEvent: 'file_system.directory_created',
        file_system.DirectoryListedEvent: 'file_system.directory_listed',
        file_system.FileEditedEvent: 'file_system.file_edited',
        file_system.FileReadEvent: 'file_system.file_read',
        file_system.FileWrittenEvent: 'file_system.file_written',
        file_system.FilesSearchedEvent: 'file_system.files_searched',
        shell.CommandFinishedEvent: 'shell.command_finished',
        shell.CommandOutputEvent: 'shell.command_output',
        shell.CommandStartedEvent: 'shell.command_started',
        sub_agents.AgentCreatedEvent: 'sub_agents.agent_created',
        sub_agents.TaskAssignedEvent: 'sub_agents.task_assigned',
        sub_agents.TaskBackgroundedEvent: 'sub_agents.task_backgrounded',
        sub_agents.TaskCancelledEvent: 'sub_agents.task_cancelled',
        sub_agents.TaskCompletedEvent: 'sub_agents.task_completed',
        sub_agents.TaskFailedEvent: 'sub_agents.task_failed',
        sub_agents.TaskProgressEvent: 'sub_agents.task_progress',
        sub_agents.TaskStartedEvent: 'sub_agents.task_started',
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
            file_system.FilesSearchedEvent(
                path='.', root_dir='/w', pattern='x', search_kind='grep', match_count=1, truncated=False
            ),
            'search',
            'search_kind',
        ),
        (
            sub_agents.AgentCreatedEvent(agent_name='poet', description='d', instructions='i', model=None),
            'name',
            'agent_name',
        ),
        (
            sub_agents.DelegationStartedEvent(
                agent_name='worker',
                task='t',
                truncated=False,
                model=None,
                parent_task_id='parent',
            ),
            'parent_id',
            'parent_task_id',
        ),
        (
            sub_agents.DelegationFinishedEvent(
                agent_name='worker',
                outcome='ok',
                output='done',
                truncated=False,
                usage=None,
                parent_task_id='parent',
                duration_ms=1500,
            ),
            'parent_id',
            'parent_task_id',
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

    assert not hasattr(event, old)


def test_delegation_duration_seconds_payload_still_loads() -> None:
    event = sub_agents.DelegationFinishedEvent(
        agent_name='worker', outcome='ok', output='done', truncated=False, usage=None, duration_ms=1500
    )
    payload = _EVENTS.dump_python(event, mode='json')
    assert 'duration_seconds' not in payload
    assert _EVENTS.validate_python(payload) == event

    assert _EVENTS.validate_python({**payload, 'duration_seconds': 9}) == event
    del payload['duration_ms']
    assert _EVENTS.validate_python({**payload, 'duration_seconds': 1.5}) == event
    assert not hasattr(event, 'duration_seconds')
