"""Logfire-backed capabilities: drive agent configuration from Logfire managed variables.

The Agent Control contract -- [`AgentConfig`][logfire.agent_control.AgentConfig] and the models,
stored JSON schema, and apply semantics around it -- lives in `logfire.agent_control`, where every
framework adapter and the Logfire UI share one copy of it. Import it from there;
[`AgentControl`][pydantic_ai_harness.logfire.AgentControl] is what connects it to a Pydantic AI agent.
"""

from pydantic_ai_harness.logfire._agent_control import AgentControl
from pydantic_ai_harness.logfire._managed_prompt import ManagedPrompt
from pydantic_ai_harness.logfire._managed_variable import resolution_reason

__all__ = [
    'AgentControl',
    'ManagedPrompt',
    'resolution_reason',
]
