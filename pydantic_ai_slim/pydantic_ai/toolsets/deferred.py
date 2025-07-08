from __future__ import annotations

from dataclasses import replace
from typing import Any

from .._run_context import AgentDepsT, RunContext
from ..messages import ToolCallPart
from ..tools import ToolDefinition
from . import AbstractToolset
from ._run import RunToolset


class DeferredToolset(AbstractToolset[AgentDepsT]):
    """A toolset that holds deferred tools.

    See [`ToolDefinition.kind`][pydantic_ai.tools.ToolDefinition.kind] for more information about deferred tools.
    """

    _tool_defs: list[ToolDefinition]

    def __init__(self, tool_defs: list[ToolDefinition]):
        self._tool_defs = tool_defs

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        return RunToolset(self, ctx)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return [replace(tool_def, kind='deferred') for tool_def in self._tool_defs]

    def _max_retries_for_tool(self, name: str) -> int:
        raise NotImplementedError('Deferred tools cannot be retried')

    async def call_tool(self, call: ToolCallPart, ctx: RunContext[AgentDepsT], allow_partial: bool = False) -> Any:
        raise NotImplementedError('Deferred tools cannot be called')
