"""Logfire-backed capabilities: drive agent configuration from Logfire managed variables.

[`AgentControl`][pydantic_ai_harness.logfire.AgentControl] connects a Pydantic AI agent to one
Logfire variable holding an [`AgentConfig`][pydantic_ai_harness.logfire.AgentConfig] -- the Agent
Control contract every SDK and the Logfire UI share. The contract's models are exported here for
publishing a value from code rather than from the UI.
"""

from pydantic_ai_harness.logfire._agent_control import AgentControl
from pydantic_ai_harness.logfire._contract import (
    AgentConfig,
    AgentConfigSettings,
    InstructionBlock,
    OnUnmatched,
    ParameterOverride,
    ToolDefinitionOverride,
)
from pydantic_ai_harness.logfire._managed_prompt import ManagedPrompt
from pydantic_ai_harness.logfire._managed_variable import resolution_reason

__all__ = [
    'AgentConfig',
    'AgentConfigSettings',
    'AgentControl',
    'InstructionBlock',
    'ManagedPrompt',
    'OnUnmatched',
    'ParameterOverride',
    'ToolDefinitionOverride',
    'resolution_reason',
]
