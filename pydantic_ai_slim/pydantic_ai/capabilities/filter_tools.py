"""Capability that filters which tools are visible to the model."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.tools import ToolSelector
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.filtered import FilteredToolset

from .abstract import AbstractCapability


@dataclass
class FilterTools(AbstractCapability[AgentDepsT]):
    """Capability that filters which tools are visible to the model.

    Only tools matching the [`ToolSelector`][pydantic_ai.tools.ToolSelector] are
    included; non-matching tools are hidden entirely.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import FilterTools

    # Only expose specific tools:
    agent = Agent('openai:gpt-5', capabilities=[FilterTools(tools=['search', 'fetch'])])

    # Filter by metadata:
    agent = Agent('openai:gpt-5', capabilities=[FilterTools(tools={'public': True})])
    ```
    """

    tools: ToolSelector[AgentDepsT] = 'all'

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'FilterTools'

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return FilteredToolset(toolset, tools=self.tools)
