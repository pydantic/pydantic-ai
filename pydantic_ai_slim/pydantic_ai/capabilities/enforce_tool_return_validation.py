"""Capability that enforces return validation on selected tools."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.tools import ToolDefinition, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.prepared import PreparedToolset

from .abstract import AbstractCapability


@dataclass
class EnforceToolReturnValidation(AbstractCapability[AgentDepsT]):
    """Capability that enforces return validation for selected tools.

    When added to an agent's capabilities, this sets
    [`validate_return`][pydantic_ai.tools.ToolDefinition.validate_return]
    to `True` on matching tool definitions.

    Per-tool overrides (`Tool(..., validate_return=False)`) take
    precedence — this capability only sets the flag on tools that haven't
    explicitly opted in or out.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import EnforceToolReturnValidation

    agent = Agent('openai:gpt-5', capabilities=[EnforceToolReturnValidation()])
    ```
    """

    tools: ToolSelector[AgentDepsT] = 'all'
    """Which tools should have their return values validated.

    - `'all'` (default): every tool gets validated.
    - `Sequence[str]`: only tools whose names are listed.
    - `dict[str, Any]`: matches tools whose metadata deeply includes the specified key-value pairs.
    - Callable `(ctx, tool_def) -> bool`: custom sync or async predicate.
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'EnforceToolReturnValidation'

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        selector = self.tools

        async def _enforce_return_validation(
            ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
        ) -> list[ToolDefinition]:
            resolved: list[ToolDefinition] = []
            for td in tool_defs:
                match = await matches_tool_selector(selector, ctx, td)
                if not td.validate_return and match:
                    td = replace(td, validate_return=True)
                resolved.append(td)
            return resolved

        return PreparedToolset(toolset, _enforce_return_validation)
