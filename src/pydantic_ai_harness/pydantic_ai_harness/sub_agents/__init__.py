"""Sub-agent capability: delegate self-contained tasks to named child agents.

Task control (`SubAgents(task_control=TaskControl())`), remote agents, and agents the model creates are a port of
Vstorm's [subagents-pydantic-ai](https://github.com/vstorm-co/subagents-pydantic-ai), MIT licensed,
copyright (c) 2026 Vstorm.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from pydantic_ai_harness._task_report import is_task_report
from pydantic_ai_harness._warn import HarnessDeprecationWarning, warn_class_renamed
from pydantic_ai_harness.sub_agents._capability import SubAgents, ToolResolver
from pydantic_ai_harness.sub_agents._control import (
    DynamicAgentFactory,
    DynamicAgents,
    DynamicAgentSpec,
    RemoteSubAgent,
    TaskControl,
)
from pydantic_ai_harness.sub_agents._disk import AgentOverride
from pydantic_ai_harness.sub_agents._events import (
    MAX_EVENT_TEXT_CHARS,
    SUB_AGENTS_EVENTS,
    AgentCreatedEvent,
    BackgroundReason,
    ChildMessageEvent,
    DelegationFinishedEvent,
    DelegationOutcome,
    DelegationStartedEvent,
    ParentMessageEvent,
    Recipient,
    SubAgentMessageEvent,
    TaskAnsweredEvent,
    TaskAssignedEvent,
    TaskBackgroundedEvent,
    TaskCancelledEvent,
    TaskCancelRequestedEvent,
    TaskCompletedEvent,
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
)
from pydantic_ai_harness.sub_agents._models import ModelOption
from pydantic_ai_harness.sub_agents._tasks import (
    DelegationReports,
    DelegationTask,
    DelegationTaskEvent,
    DelegationTasks,
    SavedTaskProblem,
    SavedTasks,
)
from pydantic_ai_harness.sub_agents._toolset import (
    SELF_AGENT_NAME,
    DelegationWorkspaceRequest,
    SubAgent,
    SubAgentToolset,
    WorkspaceResolver,
)
from pydantic_ai_harness.sub_agents._transport import (
    InMemoryTransport,
    SubAgentTransport,
    deserialize_message,
    serialize_message,
)
from pydantic_ai_harness.sub_agents._worker import SubAgentWorker

if TYPE_CHECKING:
    from pydantic_ai_harness.sub_agents._effort import MINIMUM_EFFORT_FLOOR, clamp_effort

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
        warn_class_renamed(name, renamed.__name__, __name__)
        return renamed
    if name in {'MINIMUM_EFFORT_FLOOR', 'clamp_effort'}:
        from pydantic_ai_harness.sub_agents import _effort

        warnings.warn(
            f'`pydantic_ai_harness.sub_agents.{name}` is deprecated because `SubAgents` no longer imposes a '
            'minimum thinking effort. Pass `AgentOverride(effort=...)` for a disk agent that needs an explicit '
            'level, or apply an application-specific policy outside the capability. This export will be removed '
            'in a future release.',
            category=HarnessDeprecationWarning,
            stacklevel=2,
        )
        return getattr(_effort, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
