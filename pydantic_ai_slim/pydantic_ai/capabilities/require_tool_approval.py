"""Capability that requires approval before executing selected tools."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.tools import ToolSelector
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset

from .abstract import AbstractCapability


@dataclass
class RequireToolApproval(AbstractCapability[AgentDepsT]):
    """Capability that requires approval before executing selected tools.

    When a tool matching the selector is called, execution is paused and
    an [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] exception
    is raised, allowing the caller to approve or deny the call.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import RequireToolApproval

    # Require approval for all tools:
    agent = Agent('openai:gpt-5', capabilities=[RequireToolApproval()])

    # Require approval only for specific tools:
    agent = Agent('openai:gpt-5', capabilities=[RequireToolApproval(tools=['delete', 'modify'])])

    # Require approval for tools with specific metadata:
    agent = Agent('openai:gpt-5', capabilities=[RequireToolApproval(tools={'dangerous': True})])
    ```
    """

    tools: ToolSelector[AgentDepsT] = 'all'

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'RequireToolApproval'

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return ApprovalRequiredToolset(toolset, tools=self.tools)
