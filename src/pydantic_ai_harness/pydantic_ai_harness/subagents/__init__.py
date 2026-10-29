"""Sub-agent capability: delegate self-contained tasks to named child agents.

Task control (`SubAgents(task_control=TaskControl())`), remote agents, and agents the model creates are a port of
Vstorm's [subagents-pydantic-ai](https://github.com/vstorm-co/subagents-pydantic-ai), MIT licensed,
copyright (c) 2026 Vstorm.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from pydantic_ai_harness._task_report import is_task_report
from pydantic_ai_harness._warn import HarnessDeprecationWarning
from pydantic_ai_harness.subagents._capability import SubAgents, ToolResolver
from pydantic_ai_harness.subagents._control import (
    DynamicAgentFactory,
    DynamicAgents,
    DynamicAgentSpec,
    RemoteSubAgent,
    TaskControl,
)
from pydantic_ai_harness.subagents._disk import AgentOverride
from pydantic_ai_harness.subagents._events import (
    MAX_EVENT_TEXT_CHARS,
    SUB_AGENTS_EVENTS,
    AgentCreatedEvent,
    BackgroundReason,
    ChildMessage,
    DelegationEndEvent,
    DelegationOutcome,
    DelegationStartEvent,
    ParentMessage,
    Recipient,
    SubAgentMessage,
    TaskAnswerEvent,
    TaskAssignedEvent,
    TaskBackgroundedEvent,
    TaskCancelledEvent,
    TaskCancelRequestEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskFollowUpEvent,
    TaskForceCancelEvent,
    TaskMessageEvent,
    TaskProgressEvent,
    TaskQuestionEvent,
    TaskRequestEvent,
    TaskRetryEvent,
    TaskSettledEvent,
    TaskStartedEvent,
)
from pydantic_ai_harness.subagents._models import ModelOption
from pydantic_ai_harness.subagents._tasks import (
    DelegationReports,
    DelegationTask,
    DelegationTaskEvent,
    DelegationTasks,
)
from pydantic_ai_harness.subagents._toolset import (
    DelegationWorkspaceRequest,
    SubAgent,
    SubAgentToolset,
    WorkspaceResolver,
)
from pydantic_ai_harness.subagents._transport import (
    InMemoryTransport,
    SubAgentTransport,
    deserialize_message,
    serialize_message,
)
from pydantic_ai_harness.subagents._worker import SubAgentWorker

if TYPE_CHECKING:
    from pydantic_ai_harness.subagents._effort import MINIMUM_EFFORT_FLOOR, clamp_effort

__all__ = [
    'MAX_EVENT_TEXT_CHARS',
    'MINIMUM_EFFORT_FLOOR',
    'SUB_AGENTS_EVENTS',
    'AgentCreatedEvent',
    'AgentOverride',
    'BackgroundReason',
    'ChildMessage',
    'DelegationEndEvent',
    'DelegationOutcome',
    'DelegationStartEvent',
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
    'ParentMessage',
    'Recipient',
    'RemoteSubAgent',
    'SubAgent',
    'SubAgentMessage',
    'SubAgentToolset',
    'SubAgentTransport',
    'SubAgentWorker',
    'SubAgents',
    'TaskAnswerEvent',
    'TaskAssignedEvent',
    'TaskBackgroundedEvent',
    'TaskCancelRequestEvent',
    'TaskCancelledEvent',
    'TaskCompletedEvent',
    'TaskControl',
    'TaskFailedEvent',
    'TaskFollowUpEvent',
    'TaskForceCancelEvent',
    'TaskMessageEvent',
    'TaskProgressEvent',
    'TaskQuestionEvent',
    'TaskRequestEvent',
    'TaskRetryEvent',
    'TaskSettledEvent',
    'TaskStartedEvent',
    'ToolResolver',
    'WorkspaceResolver',
    'clamp_effort',
    'deserialize_message',
    'is_task_report',
    'serialize_message',
]


def __getattr__(name: str) -> object:
    if name in {'MINIMUM_EFFORT_FLOOR', 'clamp_effort'}:
        from pydantic_ai_harness.subagents import _effort

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
