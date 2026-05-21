from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pydantic

from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.tools import AgentDepsT, AgentNativeTool

from .abstract import AbstractCapability

_NATIVE_TOOL_ADAPTER = pydantic.TypeAdapter(AbstractNativeTool)


@dataclass
class NativeTool(AbstractCapability[AgentDepsT]):
    """A capability that registers a native tool with the agent.

    Wraps a single [`AgentNativeTool`][pydantic_ai.tools.AgentNativeTool] — either a static
    [`AbstractNativeTool`][pydantic_ai.native_tools.AbstractNativeTool] instance or a callable
    that dynamically produces one.

    Equivalent to passing the tool through `Agent(capabilities=[NativeTool(my_tool)])`. For
    provider-adaptive use (with a local fallback), see [`NativeOrLocalTool`][pydantic_ai.capabilities.NativeOrLocalTool]
    or its subclasses like [`WebSearch`][pydantic_ai.capabilities.WebSearch].
    """

    tool: AgentNativeTool[AgentDepsT]

    def get_native_tools(self) -> Sequence[AgentNativeTool[AgentDepsT]]:
        return [self.tool]

    @classmethod
    def from_spec(cls, tool: AbstractNativeTool | None = None, **kwargs: Any) -> NativeTool[Any]:
        """Create from spec.

        Supports two YAML forms:

        - Flat: `{NativeTool: {kind: web_search, search_context_size: high}}`
        - Explicit: `{NativeTool: {tool: {kind: web_search}}}`
        """
        if tool is not None:
            validated = _NATIVE_TOOL_ADAPTER.validate_python(tool)
        elif kwargs:
            validated = _NATIVE_TOOL_ADAPTER.validate_python(kwargs)
        else:
            raise TypeError(
                '`NativeTool.from_spec()` requires either a `tool` argument or keyword arguments'
                ' specifying the native tool type (e.g. `kind="web_search"`)'
            )
        return cls(tool=validated)
