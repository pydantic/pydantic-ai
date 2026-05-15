from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai._spec import CapabilitySpec
from pydantic_ai.tools import AgentDepsT, ToolSelector
from pydantic_ai.toolsets import AbstractToolset, AgentToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.prefixed import PrefixedToolset

from .abstract import AbstractCapability
from .wrapper import WrapperCapability


@dataclass(init=False)
class PrefixTools(WrapperCapability[AgentDepsT]):
    """A capability that prefixes tool names.

    When `wrapped` is provided, only the wrapped capability's tools are prefixed;
    other agent tools are unaffected. When `wrapped` is omitted, the capability
    uses [`get_wrapper_toolset`][pydantic_ai.capabilities.AbstractCapability.get_wrapper_toolset]
    to prefix tools from the agent's assembled toolset.

    The `tools` parameter controls which tools are prefixed. Tools that don't
    match the selector pass through with their original names.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import PrefixTools, Toolset
    from pydantic_ai.toolsets import FunctionToolset

    toolset = FunctionToolset()

    # Prefix all tools from a specific toolset:
    agent = Agent(
        'openai:gpt-5',
        capabilities=[
            PrefixTools(
                wrapped=Toolset(toolset),
                prefix='ns',
            ),
        ],
    )

    # Prefix all agent tools (standalone mode):
    agent = Agent(
        'openai:gpt-5',
        capabilities=[PrefixTools(prefix='ns')],
    )

    # Prefix only specific tools:
    agent = Agent(
        'openai:gpt-5',
        capabilities=[PrefixTools(prefix='ns', tools=['search', 'fetch'])],
    )
    ```
    """

    prefix: str
    tools: ToolSelector[AgentDepsT]

    def __init__(
        self,
        *,
        prefix: str,
        wrapped: AbstractCapability[AgentDepsT] | None = None,
        tools: ToolSelector[AgentDepsT] = 'all',
    ) -> None:
        self.wrapped = wrapped
        self.prefix = prefix
        self.tools = tools

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'PrefixTools'

    @classmethod
    def from_spec(
        cls, *, prefix: str, capability: CapabilitySpec | None = None, tools: Any = 'all'
    ) -> PrefixTools[Any]:
        """Create from spec with an optional nested capability specification.

        Args:
            prefix: The prefix to add to tool names (e.g. `'mcp'` turns `'search'` into `'mcp_search'`).
            capability: An optional capability spec (same format as entries in the `capabilities` list).
            tools: A [`ToolSelector`][pydantic_ai.tools.ToolSelector] specifying which tools to prefix.
        """
        wrapped = None
        if capability is not None:
            from pydantic_ai.agent.spec import load_capability_from_nested_spec

            wrapped = load_capability_from_nested_spec(capability)
        return cls(wrapped=wrapped, prefix=prefix, tools=tools)

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        if self.wrapped is None:
            return super().get_toolset()
        toolset = super().get_toolset()
        if toolset is None:
            return None
        if isinstance(toolset, AbstractToolset):
            return PrefixedToolset(toolset, prefix=self.prefix, tools=self.tools)
        # ToolsetFunc callable — wrap in DynamicToolset so PrefixedToolset can delegate
        return PrefixedToolset(DynamicToolset[AgentDepsT](toolset_func=toolset), prefix=self.prefix, tools=self.tools)

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        if self.wrapped is not None:
            return super().get_wrapper_toolset(toolset)
        return PrefixedToolset(toolset, prefix=self.prefix, tools=self.tools)
