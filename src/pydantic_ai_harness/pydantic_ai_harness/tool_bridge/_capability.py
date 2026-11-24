"""Capability that keeps the model's tool list fixed while the tools behind it change."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

from pydantic_ai import AbstractToolset
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, ToolSearch, ToolSearchFunc
from pydantic_ai.tools import AgentDepsT, ToolSelector
from pydantic_ai_harness.tool_bridge._toolset import ToolBridgeToolset


@dataclass
class ToolBridge(AbstractCapability[AgentDepsT]):
    """Capability that puts the agent's tools behind `tool_search`, `tool_describe` and `tool_call`.

    The model sees only those three tools, whose definitions never change, so the tool definitions
    at the head of the prompt stay cached however many tools sit behind them and however they change
    mid-run. The model finds a tool with `tool_search`, reads its argument schema with
    `tool_describe`, and runs it with `tool_call`, passing the arguments as a JSON string.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import ToolBridge

    agent = Agent('openai:gpt-5', capabilities=[ToolBridge()])
    ```
    """

    tools: ToolSelector[AgentDepsT] = 'all'
    """Which tools to bridge; the rest stay visible to the model as normal tools.

    - `'all'` (default): every eligible tool.
    - `Sequence[str]`: the tools with these names.
    - `dict[str, Any]`: the tools whose metadata contains these entries, such as those `SetToolMetadata` tags.
    - Callable `(ctx, tool_def) -> bool | Awaitable[bool]`: the tools it returns `True` for.
    """

    _: KW_ONLY

    search: ToolSearchFunc[AgentDepsT] | None = None
    """Ranks the bridged tools for `tool_search`'s queries: `(ctx, queries, tools) -> names`, sync or async.

    `None` (default) ranks tools by how many of the queries' words appear in their name or description.
    """

    max_results: int = 10
    """Maximum number of `tool_search` results."""

    max_retries: int | None = None
    """Retries for each dispatcher tool, spent on unknown tool names, invalid arguments and bridged tools' retries.

    `None` (default) uses the agent's tool retry budget, as `Agent(retries={'tools': N})` sets it.
    """

    id: str | None = 'tool_bridge'
    """An agent has one bridge, so two of them merge into one.

    Two with distinct ids aren't supported: the outer one refuses the inner one's reserved tool names.
    """

    def get_ordering(self) -> CapabilityOrdering:
        """Sit just inside `ToolSearch` and outside every other wrapper.

        Wrappers such as `SetToolMetadata` and `PrefixTools` then see the bridged tools whatever order the
        capabilities are listed in.
        """
        return CapabilityOrdering(position='outermost', wrapped_by=[ToolSearch])

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Wrap the agent's tools in the bridge."""
        return ToolBridgeToolset(
            wrapped=toolset,
            tools=self.tools,
            search=self.search,
            max_results=self.max_results,
            max_retries=self.max_retries,
        )
