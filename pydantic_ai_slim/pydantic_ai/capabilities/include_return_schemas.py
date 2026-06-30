"""Capability that enables return schemas on selected tools."""

from __future__ import annotations

from dataclasses import dataclass, replace

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.tools import ToolDefinition, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.prepared import PreparedToolset

from .abstract import AbstractCapability


@dataclass
class IncludeToolReturnSchemas(AbstractCapability[AgentDepsT]):
    """Capability that includes return schemas for selected tools.

    When added to an agent's capabilities, this sets
    [`include_return_schema`][pydantic_ai.tools.ToolDefinition.include_return_schema]
    to `True` on matching tool definitions, causing the model to receive
    return type information for those tools.

    For models that natively support return schemas (e.g. Google Gemini), the
    schema is passed as a structured field.  For other models, it is injected
    into the tool description as JSON text.

    Per-tool overrides (`Tool(..., include_return_schema=False)`) take
    precedence — this capability only sets the flag on tools that haven't
    explicitly opted out.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import IncludeToolReturnSchemas

    agent = Agent('openai:gpt-5', capabilities=[IncludeToolReturnSchemas()])
    ```
    """

    tools: ToolSelector[AgentDepsT] = 'all'
    """Which tools should have their return schemas included.

    - `'all'` (default): every tool gets its return schema included.
    - `Sequence[str]`: only tools whose names are listed.
    - `dict[str, Any]`: matches tools whose metadata deeply includes the specified key-value pairs.
    - Callable `(ctx, tool_def) -> bool`: custom sync or async predicate.
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'IncludeToolReturnSchemas'

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        selector = self.tools

        async def _include_return_schemas(
            ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
        ) -> list[ToolDefinition]:
            resolved: list[ToolDefinition] = []
            for td in tool_defs:
                # Only set the flag on tools that haven't explicitly opted in or out
                if td.include_return_schema is None and await matches_tool_selector(selector, ctx, td):
                    td = replace(td, include_return_schema=True)
                resolved.append(td)
            return resolved

        return PreparedToolset(toolset, _include_return_schemas)
