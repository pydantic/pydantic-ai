from __future__ import annotations

from dataclasses import replace
from typing import Any

from pydantic_core import SchemaValidator

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition
from .base import BaseToolset


class DeferredToolset(BaseToolset[AgentDepsT]):
    """A toolset that holds deferred tools.

    See [`ToolDefinition.kind`][pydantic_ai.tools.ToolDefinition.kind] for more information about deferred tools.
    """

    _tool_defs: list[ToolDefinition]

    def __init__(self, tool_defs: list[ToolDefinition]):
        self._tool_defs = [replace(tool_def, kind='deferred') for tool_def in tool_defs]

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return self._tool_defs

    def _max_retries_for_tool(self, name: str) -> int:
        raise NotImplementedError('Deferred tools cannot be retried')

    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        raise NotImplementedError('Deferred tools cannot be validated')

    def _call_tool(self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any]) -> Any:
        raise NotImplementedError('Deferred tools cannot be called')
