"""Deprecated import location for `pydantic_ai_harness.sub_agents`.

Renamed to the snake_case of its capability, `SubAgents`. Importing from here still works but emits a
`HarnessDeprecationWarning`. Import from `pydantic_ai_harness.sub_agents` instead.
"""

import warnings
from typing import TYPE_CHECKING

from pydantic_ai_harness._warn import HarnessDeprecationWarning, warn_class_renamed, warn_module_renamed
from pydantic_ai_harness.sub_agents import (
    MAX_EVENT_TEXT_CHARS,
    SELF_AGENT_NAME,
    SUB_AGENTS_EVENTS,
    AgentCreatedEvent,
    AgentOverride,
    BackgroundReason,
    ChildMessageEvent,
    DelegationFinishedEvent,
    DelegationOutcome,
    DelegationReports,
    DelegationStartedEvent,
    DelegationTask,
    DelegationTaskEvent,
    DelegationTasks,
    DelegationWorkspaceRequest,
    DynamicAgentFactory,
    DynamicAgents,
    DynamicAgentSpec,
    InMemoryTransport,
    ModelOption,
    ParentMessageEvent,
    Recipient,
    RemoteSubAgent,
    SavedTaskProblem,
    SavedTasks,
    SubAgent,
    SubAgentMessageEvent,
    SubAgents,
    SubAgentToolset,
    SubAgentTransport,
    SubAgentWorker,
    TaskAnsweredEvent,
    TaskAssignedEvent,
    TaskBackgroundedEvent,
    TaskCancelledEvent,
    TaskCancelRequestedEvent,
    TaskCompletedEvent,
    TaskControl,
    TaskFailedEvent,
    TaskForceCancelRequestedEvent,
    TaskProgressEvent,
    TaskQuestionAskedEvent,
    TaskRequestEvent,
    TaskResumedEvent,
    TaskRetriedEvent,
    TaskSettledEvent,
    TaskStartedEvent,
    TaskSteeredEvent,
    ToolResolver,
    WorkspaceResolver,
    deserialize_message,
    is_task_report,
    serialize_message,
)

if TYPE_CHECKING:
    from pydantic_ai_harness.sub_agents._effort import MINIMUM_EFFORT_FLOOR, clamp_effort

warn_module_renamed('subagents', 'sub_agents')

__all__ = [
    'MAX_EVENT_TEXT_CHARS',
    'MINIMUM_EFFORT_FLOOR',
    'SELF_AGENT_NAME',
    'SUB_AGENTS_EVENTS',
    'AgentCreatedEvent',
    'AgentOverride',
    'BackgroundReason',
    'ChildMessageEvent',
    'DelegationFinishedEvent',
    'DelegationOutcome',
    'DelegationStartedEvent',
    'DelegationReports',
    'DelegationTask',
    'DelegationTaskEvent',
    'DelegationWorkspaceRequest',
    'DelegationTasks',
    'DynamicAgentFactory',
    'DynamicAgentSpec',
    'DynamicAgents',
    'InMemoryTransport',
    'ModelOption',
    'ParentMessageEvent',
    'Recipient',
    'RemoteSubAgent',
    'SavedTaskProblem',
    'SavedTasks',
    'SubAgent',
    'SubAgentMessageEvent',
    'SubAgentToolset',
    'SubAgentTransport',
    'SubAgentWorker',
    'SubAgents',
    'TaskAnsweredEvent',
    'TaskAssignedEvent',
    'TaskBackgroundedEvent',
    'TaskCancelRequestedEvent',
    'TaskCancelledEvent',
    'TaskCompletedEvent',
    'TaskControl',
    'TaskFailedEvent',
    'TaskForceCancelRequestedEvent',
    'TaskProgressEvent',
    'TaskQuestionAskedEvent',
    'TaskRequestEvent',
    'TaskResumedEvent',
    'TaskRetriedEvent',
    'TaskSettledEvent',
    'TaskStartedEvent',
    'TaskSteeredEvent',
    'ToolResolver',
    'WorkspaceResolver',
    'clamp_effort',
    'deserialize_message',
    'is_task_report',
    'serialize_message',
]


_RENAMED = {
    'DelegationEndEvent': DelegationFinishedEvent,
    'DelegationStartEvent': DelegationStartedEvent,
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


def __getattr__(name: str) -> object:
    renamed = _RENAMED.get(name)
    if renamed is not None:
        warn_class_renamed(name, renamed.__name__, __name__, new_module='pydantic_ai_harness.sub_agents')
        return renamed
    if name in {'MINIMUM_EFFORT_FLOOR', 'clamp_effort'}:
        from pydantic_ai_harness.sub_agents import _effort

        warnings.warn(
            f'`pydantic_ai_harness.subagents.{name}` is deprecated because `SubAgents` no longer imposes a '
            'minimum thinking effort. Pass `AgentOverride(effort=...)` for a disk agent that needs an explicit '
            'level, or apply an application-specific policy outside the capability. This export will be removed '
            'in a future release.',
            category=HarnessDeprecationWarning,
            stacklevel=2,
        )
        return getattr(_effort, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
