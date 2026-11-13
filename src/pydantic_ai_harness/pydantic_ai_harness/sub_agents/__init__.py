"""Sub-agent capability: delegate self-contained tasks to named child agents.

Task control (`SubAgents(task_control=TaskControl())`), remote agents, and agents the model creates are a port of
Vstorm's [subagents-pydantic-ai](https://github.com/vstorm-co/subagents-pydantic-ai), MIT licensed,
copyright (c) 2026 Vstorm.
"""

from __future__ import annotations

from pydantic_ai_harness._task_report import is_task_report
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

__all__ = [
    'MAX_EVENT_TEXT_CHARS',
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
    'deserialize_message',
    'is_task_report',
    'serialize_message',
]
