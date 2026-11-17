"""Capability that keeps the model's tool list fixed while the tools behind it change."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

from pydantic_ai import AbstractToolset
from pydantic_ai.capabilities import AbstractCapability, ToolSearchFunc
from pydantic_ai.tools import AgentDepsT, ToolSelector
from pydantic_ai.toolsets._tool_search import keywords_search_fn
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
    - Callable `(ctx, tool_def) -> bool | Awaitable[bool]`: the tools it returns `True` for.
    """

    _: KW_ONLY

    search: ToolSearchFunc[AgentDepsT] = keywords_search_fn
    """Ranks the bridged tools for a `tool_search` query: `(ctx, queries, tools) -> names`, sync or async.

    Defaults to the keyword matching [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] uses.
    """

    max_results: int = 10
    """Maximum number of `tool_search` results."""

    max_retries: int = 3
    """Retries for each dispatcher tool, spent on unknown tool names and invalid arguments."""

    id: str | None = 'tool_bridge'
    """An agent has one bridge, so two of them merge into one by default."""

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Wrap the agent's tools in the bridge."""
        return ToolBridgeToolset(
            wrapped=toolset,
            tools=self.tools,
            search=self.search,
            max_results=self.max_results,
            max_retries=self.max_retries,
        )
