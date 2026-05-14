"""Capability that marks selected tools for deferred loading."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.tools import ToolSelector
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.deferred_loading import DeferredLoadingToolset

from .abstract import AbstractCapability


@dataclass
class DeferToolLoading(AbstractCapability[AgentDepsT]):
    """Capability that marks selected tools for deferred loading.

    Deferred tools are hidden from the model until discovered via tool search,
    reducing the number of tools the model sees at once.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import DeferToolLoading

    # Defer all tools:
    agent = Agent('openai:gpt-5', capabilities=[DeferToolLoading()])

    # Defer only specific tools:
    agent = Agent('openai:gpt-5', capabilities=[DeferToolLoading(tools=['rarely_used'])])
    ```
    """

    tools: ToolSelector[AgentDepsT] = 'all'

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'DeferToolLoading'

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return DeferredLoadingToolset(toolset, tools=self.tools)
