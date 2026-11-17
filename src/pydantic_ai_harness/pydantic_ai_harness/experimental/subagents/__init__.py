"""Deprecated import location for `pydantic_ai_harness.sub_agents`.

This capability graduated out of `experimental`; importing from here still works but
emits a `DeprecationWarning`. Import from `pydantic_ai_harness.sub_agents` instead.
"""

from pydantic_ai_harness.experimental._warn import warn_moved
from pydantic_ai_harness.sub_agents import (
    AgentOverride,
    ModelOption,
    SubAgent,
    SubAgents,
    SubAgentToolset,
    ToolResolver,
)
from pydantic_ai_harness.sub_agents._effort import MINIMUM_EFFORT_FLOOR, clamp_effort

warn_moved('subagents', 'sub_agents')

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
