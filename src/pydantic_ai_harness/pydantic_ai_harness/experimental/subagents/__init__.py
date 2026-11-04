"""Deprecated import location for `pydantic_ai_harness.subagents`.

This capability graduated out of `experimental`; importing from here still works but
emits a `DeprecationWarning`. Import from `pydantic_ai_harness.subagents` instead.
"""

from pydantic_ai_harness.experimental._warn import warn_moved
from pydantic_ai_harness.subagents import AgentOverride, ModelOption, SubAgent, SubAgents, SubAgentToolset, ToolResolver
from pydantic_ai_harness.subagents._effort import MINIMUM_EFFORT_FLOOR, clamp_effort

warn_moved('subagents', 'subagents')

__all__ = [
    'MINIMUM_EFFORT_FLOOR',
    'AgentOverride',
    'ModelOption',
    'SubAgent',
    'SubAgentToolset',
    'SubAgents',
    'ToolResolver',
    'clamp_effort',
]
