"""Harness `CapabilityEvent` class names: the deprecated aliases, and the wire kinds a rename must keep."""

from __future__ import annotations

from types import ModuleType

import pytest

import pydantic_ai_harness.compaction as compaction
import pydantic_ai_harness.filesystem as filesystem
import pydantic_ai_harness.shell as shell
import pydantic_ai_harness.step_persistence as step_persistence
import pydantic_ai_harness.subagents as subagents
from pydantic_ai import CapabilityEvent
from pydantic_ai_harness import HarnessDeprecationWarning


@pytest.mark.parametrize(
    ('module', 'old_name', 'new_name', 'kind'),
    [
        (compaction, 'ContextUsageEvent', 'ContextUsageMeasuredEvent', 'report_context_usage.measured'),
        (filesystem, 'FileChangeRequestEvent', 'FileChangeRequestedEvent', 'file_system.file_change_request'),
        (step_persistence, 'SnapshotSaved', 'SnapshotSavedEvent', 'step_persistence.snapshot_saved'),
        (subagents, 'DelegationStartEvent', 'DelegationStartedEvent', 'sub_agents.delegation_start'),
        (subagents, 'DelegationEndEvent', 'DelegationFinishedEvent', 'sub_agents.delegation_end'),
    ],
)
def test_renamed_event_keeps_its_old_name_and_kind(module: ModuleType, old_name: str, new_name: str, kind: str) -> None:
    event_type = getattr(module, new_name)
    with pytest.warns(HarnessDeprecationWarning, match=f'renamed to `{module.__name__}.{new_name}`'):
        assert getattr(module, old_name) is event_type
    assert event_type.kind == kind


def test_filesystem_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match='has no attribute'):
        _ = filesystem.DoesNotExist


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
        subagents.TaskAnswerEvent: 'sub_agents.task_answer',
        subagents.TaskAssignedEvent: 'sub_agents.task_assigned',
        subagents.TaskBackgroundedEvent: 'sub_agents.task_backgrounded',
        subagents.TaskCancelRequestEvent: 'sub_agents.task_cancel_request',
        subagents.TaskCancelledEvent: 'sub_agents.task_cancelled',
        subagents.TaskCompletedEvent: 'sub_agents.task_completed',
        subagents.TaskFailedEvent: 'sub_agents.task_failed',
        subagents.TaskFollowUpEvent: 'sub_agents.task_follow_up',
        subagents.TaskForceCancelEvent: 'sub_agents.task_force_cancel',
        subagents.TaskMessageEvent: 'sub_agents.task_message',
        subagents.TaskProgressEvent: 'sub_agents.task_progress',
        subagents.TaskQuestionEvent: 'sub_agents.task_question',
        subagents.TaskRetryEvent: 'sub_agents.task_retry',
        subagents.TaskStartedEvent: 'sub_agents.task_started',
    }
    assert {event_type: event_type.kind for event_type in kinds} == kinds
    assert step_persistence.SnapshotSavedEvent.kind == f'{step_persistence.STEP_PERSISTENCE_EVENTS}.snapshot_saved'
