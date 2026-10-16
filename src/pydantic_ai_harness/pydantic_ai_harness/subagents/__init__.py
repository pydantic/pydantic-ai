"""Sub-agent capability: delegate self-contained tasks to named child agents."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from pydantic_ai_harness._warn import HarnessDeprecationWarning
from pydantic_ai_harness.subagents._capability import SubAgents, ToolResolver
from pydantic_ai_harness.subagents._disk import AgentOverride
from pydantic_ai_harness.subagents._events import (
    MAX_EVENT_TEXT_CHARS,
    SUB_AGENTS_EVENTS,
    DelegationEndEvent,
    DelegationOutcome,
    DelegationStartEvent,
)
from pydantic_ai_harness.subagents._models import ModelOption
from pydantic_ai_harness.subagents._tasks import DelegationReports, DelegationTask, DelegationTaskEvent, DelegationTasks
from pydantic_ai_harness.subagents._toolset import SubAgent, SubAgentToolset

if TYPE_CHECKING:
    from pydantic_ai_harness.subagents._effort import MINIMUM_EFFORT_FLOOR, clamp_effort

__all__ = [
    'MAX_EVENT_TEXT_CHARS',
    'MINIMUM_EFFORT_FLOOR',
    'SUB_AGENTS_EVENTS',
    'AgentOverride',
    'DelegationEndEvent',
    'DelegationOutcome',
    'DelegationStartEvent',
    'DelegationReports',
    'DelegationTask',
    'DelegationTaskEvent',
    'DelegationTasks',
    'ModelOption',
    'SubAgent',
    'SubAgentToolset',
    'SubAgents',
    'ToolResolver',
    'clamp_effort',
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
