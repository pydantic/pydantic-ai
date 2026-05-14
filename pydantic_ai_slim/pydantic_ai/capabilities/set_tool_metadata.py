"""Capability that merges metadata key-value pairs onto selected tools."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.tools import ToolDefinition, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.prepared import PreparedToolset

from .abstract import AbstractCapability


@dataclass
class SetToolMetadata(AbstractCapability[AgentDepsT]):
    """Capability that merges metadata key-value pairs onto selected tools.

    Metadata can be provided either as an explicit dict or as keyword arguments:

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import SetToolMetadata

    # Using keyword arguments:
    agent = Agent('openai:gpt-5', capabilities=[SetToolMetadata(code_mode=True)])

    # Using an explicit dict:
    agent = Agent('openai:gpt-5', capabilities=[SetToolMetadata({'code_mode': True})])

    # Scoped to specific tools:
    agent = Agent('openai:gpt-5', capabilities=[SetToolMetadata(tools=['search'], code_mode=True)])
    ```
    """

    tools: ToolSelector[AgentDepsT] = 'all'
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

    def __init__(
        self,
        metadata: dict[str, Any] | None = None,
        *,
        tools: ToolSelector[AgentDepsT] = 'all',
        **kwargs: Any,
    ) -> None:
        if metadata is not None and kwargs:
            raise TypeError("Cannot specify both a 'metadata' dict and keyword metadata arguments.")
        self.tools = tools
        self.metadata = metadata if metadata is not None else kwargs

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'SetToolMetadata'

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        selector = self.tools
        metadata = self.metadata

        async def _set_metadata(ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            resolved: list[ToolDefinition] = []
            for td in tool_defs:
                if await matches_tool_selector(selector, ctx, td):
                    td = replace(td, metadata={**(td.metadata or {}), **metadata})
                resolved.append(td)
            return resolved

        return PreparedToolset(toolset, _set_metadata)
