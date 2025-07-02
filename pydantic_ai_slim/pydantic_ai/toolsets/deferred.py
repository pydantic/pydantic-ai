from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic_core import SchemaValidator

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from . import AbstractToolset
from ._run import RunToolset


class DeferredToolset(AbstractToolset[AgentDepsT]):
    """A toolset that holds deferred tool."""

    _tool_defs: list[ToolDefinition]

    def __init__(self, tool_defs: list[ToolDefinition]):
        self._tool_defs = tool_defs

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        return RunToolset(self, ctx)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return [replace(tool_def, kind='deferred') for tool_def in self._tool_defs]

    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        raise NotImplementedError('Deferred tools cannot be validated')

    def _max_retries_for_tool(self, name: str) -> int:
        raise NotImplementedError('Deferred tools cannot be retried')

    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        raise NotImplementedError('Deferred tools cannot be called')
