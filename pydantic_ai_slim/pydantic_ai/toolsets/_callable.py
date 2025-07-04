from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic_core import SchemaValidator

from .._run_context import AgentDepsT, RunContext
from ..messages import ToolCallPart
from . import AbstractToolset

if TYPE_CHECKING:
    pass


class CallableToolset(AbstractToolset[AgentDepsT], ABC):
    """A toolset that implements tool args validation and tool calling."""

    @abstractmethod
    def _get_tool_args_validator(self, ctx: RunContext[Any], name: str) -> SchemaValidator:
        raise NotImplementedError()

    @abstractmethod
    async def _call_tool(self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any]) -> Any:
        raise NotImplementedError()

    async def call_tool(self, call: ToolCallPart, ctx: RunContext[AgentDepsT], allow_partial: bool = False) -> Any:
        ctx = replace(ctx, tool_name=call.tool_name, tool_call_id=call.tool_call_id)

        pyd_allow_partial: Literal['off', 'trailing-strings'] = 'trailing-strings' if allow_partial else 'off'
        validator = self._get_tool_args_validator(ctx, call.tool_name)
        if isinstance(call.args, str):
            args_dict = validator.validate_json(call.args or '{}', allow_partial=pyd_allow_partial)
        else:
            args_dict = validator.validate_python(call.args or {}, allow_partial=pyd_allow_partial)
        return await self._call_tool(ctx, call.tool_name, args_dict)
