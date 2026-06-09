"""Sub-agent capability: delegate self-contained tasks to named child agents."""

from pydantic_ai_harness.experimental._warn import warn_experimental
from pydantic_ai_harness.experimental.subagents._capability import SubAgents
from pydantic_ai_harness.experimental.subagents._toolset import SubAgentToolset

warn_experimental('subagents')

__all__ = ['SubAgentToolset', 'SubAgents']
