from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition, ToolSelector, ToolsPrepareFunc, matches_tool_selector
from .abstract import AbstractToolset
from .prepared import PreparedToolset


@dataclass(init=False)
class DeferredLoadingToolset(PreparedToolset[AgentDepsT]):
    """A toolset that marks tools for deferred loading, hiding them from the model until discovered via tool search.

    See [toolset docs](../toolsets.md#deferred-loading) for more information.
    """

    prepare_func: ToolsPrepareFunc[AgentDepsT] = field(init=False, repr=False)
    tools: ToolSelector[AgentDepsT]

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        tools: ToolSelector[AgentDepsT] = 'all',
        tool_names: frozenset[str] | Sequence[str] | None = None,
    ):
        if tool_names is not None:
            warnings.warn(
                "'tool_names' is deprecated, use 'tools' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if tools != 'all':
                raise TypeError("Cannot specify both 'tools' and 'tool_names'.")
            # Convert frozenset/sequence to list for ToolSelector (Sequence[str])
            self.tools: ToolSelector[AgentDepsT] = list(tool_names)
        else:
            self.tools = tools

        selector = self.tools

        async def _mark_deferred(ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [
                replace(td, defer_loading=True) if await matches_tool_selector(selector, ctx, td) else td
                for td in tool_defs
            ]

        self.wrapped = wrapped
        self.prepare_func = _mark_deferred
