"""Sub-agent capability: delegate self-contained tasks to named child agents."""

from pydantic_ai_harness.experimental._warn import warn_experimental
from pydantic_ai_harness.experimental.subagents._capability import SubAgents, ToolResolver
from pydantic_ai_harness.experimental.subagents._disk import AgentOverride
from pydantic_ai_harness.experimental.subagents._effort import MINIMUM_EFFORT_FLOOR, clamp_effort
from pydantic_ai_harness.experimental.subagents._toolset import SubAgent, SubAgentToolset

warn_experimental('subagents')

__all__ = [
    'MINIMUM_EFFORT_FLOOR',
    'AgentOverride',
    'SubAgent',
    'SubAgentToolset',
    'SubAgents',
    'ToolResolver',
    'clamp_effort',
]
