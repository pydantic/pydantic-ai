from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pydantic

from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.tools import AgentBuiltinTool, AgentDepsT

from .abstract import AbstractCapability

_BUILTIN_TOOL_ADAPTER = pydantic.TypeAdapter(AbstractBuiltinTool)


@dataclass
class BuiltinTool(AbstractCapability[AgentDepsT]):
    """A capability that registers a builtin tool with the agent.

    Wraps a single [`AgentBuiltinTool`][pydantic_ai.tools.AgentBuiltinTool] — either a static
    [`AbstractBuiltinTool`][pydantic_ai.builtin_tools.AbstractBuiltinTool] instance or a callable
    that dynamically produces one.

    When `builtin_tools` is passed to [`Agent.__init__`][pydantic_ai.Agent.__init__], each item is
    automatically wrapped in a `BuiltinTool` capability.
    """

    tool: AgentBuiltinTool[AgentDepsT]

    def get_builtin_tools(self) -> Sequence[AgentBuiltinTool[AgentDepsT]]:
        return [self.tool]

    @classmethod
    def from_spec(cls, tool: AbstractBuiltinTool | None = None, **kwargs: Any) -> BuiltinTool[Any]:
        """Create from spec.

        Supports two YAML forms:

        - Flat: `{BuiltinTool: {kind: web_search, search_context_size: high}}`
        - Explicit: `{BuiltinTool: {tool: {kind: web_search}}}`
        """
        if tool is not None:
            validated = _BUILTIN_TOOL_ADAPTER.validate_python(tool)
        elif kwargs:
            validated = _BUILTIN_TOOL_ADAPTER.validate_python(kwargs)
        else:
            raise TypeError(
                '`BuiltinTool.from_spec()` requires either a `tool` argument or keyword arguments'
                ' specifying the builtin tool type (e.g. `kind="web_search"`)'
            )
        return cls(tool=validated)
