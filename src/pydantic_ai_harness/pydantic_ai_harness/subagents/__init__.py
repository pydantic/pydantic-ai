"""Sub-agent capability: delegate self-contained tasks to named child agents."""

from pydantic_ai_harness.subagents._capability import SubAgents, ToolResolver
from pydantic_ai_harness.subagents._disk import AgentOverride
from pydantic_ai_harness.subagents._effort import MINIMUM_EFFORT_FLOOR, clamp_effort
from pydantic_ai_harness.subagents._events import (
    MAX_EVENT_TEXT_CHARS,
    SUB_AGENTS_EVENTS,
    DelegationEndEvent,
    DelegationOutcome,
    DelegationStartEvent,
)
from pydantic_ai_harness.subagents._models import ModelOption
from pydantic_ai_harness.subagents._toolset import SubAgent, SubAgentToolset

__all__ = [
    'MAX_EVENT_TEXT_CHARS',
    'MINIMUM_EFFORT_FLOOR',
    'SUB_AGENTS_EVENTS',
    'AgentOverride',
    'DelegationEndEvent',
    'DelegationOutcome',
    'DelegationStartEvent',
    'ModelOption',
    'SubAgent',
    'SubAgentToolset',
    'SubAgents',
    'ToolResolver',
    'clamp_effort',
]
