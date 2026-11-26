from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai._spec import CapabilitySpec
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, AgentToolset, ToolsetTool
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.prefixed import PrefixedToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset

from .wrapper import WrapperCapability


@dataclass
class PrefixTools(WrapperCapability[AgentDepsT]):
    """A capability that wraps another capability and prefixes its tool names.

    Only the wrapped capability's tools are prefixed; other agent tools are unaffected.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import PrefixTools, Toolset
    from pydantic_ai.toolsets import FunctionToolset

    toolset = FunctionToolset()

    agent = Agent(
        'openai:gpt-5',
        capabilities=[
            PrefixTools(
                wrapped=Toolset(toolset),
                prefix='ns',
            ),
        ],
    )
    ```
    """

    prefix: str

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'PrefixTools'

    @classmethod
    def from_spec(cls, *, prefix: str, capability: CapabilitySpec) -> PrefixTools[Any]:
        """Create from spec with a nested capability specification.

        Args:
            prefix: The prefix to add to tool names (e.g. `'mcp'` turns `'search'` into `'mcp_search'`).
            capability: A capability spec (same format as entries in the `capabilities` list).
        """
        from pydantic_ai.agent.spec import load_capability_from_nested_spec

        wrapped = load_capability_from_nested_spec(capability)
        return cls(wrapped=wrapped, prefix=prefix)

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        toolset = super().get_toolset()
        if toolset is None:
            return None
        if isinstance(toolset, AbstractToolset):
            # Pyright can't narrow Callable type aliases out of unions after isinstance check
            return PrefixedToolset(toolset, prefix=self.prefix)  # pyright: ignore[reportUnknownArgumentType]
        # ToolsetFunc callable — wrap in DynamicToolset so PrefixedToolset can delegate
        return PrefixedToolset(DynamicToolset[AgentDepsT](toolset_func=toolset), prefix=self.prefix)

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        wrapped = super().get_wrapper_toolset(toolset)
        if wrapped is None:
            return None
        return _PrefixedAddedToolsToolset(wrapped=wrapped, inner=toolset, prefix=self.prefix)


@dataclass
class _PrefixedAddedToolsToolset(WrapperToolset[AgentDepsT]):
    """Prefixes only the tools added by a capability's wrapper toolset, leaving passed-through tools unprefixed."""

    inner: AbstractToolset[AgentDepsT]
    prefix: str
    _added_names: set[str] = field(default_factory=set, init=False, repr=False)

    @property
    def tool_name_conflict_hint(self) -> str:
        return 'Change the `prefix` attribute to avoid name conflicts.'

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        inner_names = set(await self.inner.get_tools(ctx))
        wrapped_tools = await super().get_tools(ctx)
        added_names: set[str] = set()
        result: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in wrapped_tools.items():
            if name not in inner_names:
                added_names.add(name)
                new_name = f'{self.prefix}_{name}'
                result[new_name] = replace(
                    tool,
                    toolset=self,
                    tool_def=replace(tool.tool_def, name=new_name),
                )
            else:
                result[name] = tool
        self._added_names = added_names
        return result

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        prefix_with_sep = f'{self.prefix}_'
        if name.startswith(prefix_with_sep):
            original_name = name.removeprefix(prefix_with_sep)
            if original_name in self._added_names:
                ctx = replace(ctx, tool_name=original_name)
                tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name))
                return await super().call_tool(original_name, tool_args, ctx, tool)
        return await super().call_tool(name, tool_args, ctx, tool)
