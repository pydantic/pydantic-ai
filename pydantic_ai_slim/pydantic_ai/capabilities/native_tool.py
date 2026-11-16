from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import pydantic

from pydantic_ai.native_tools import NATIVE_TOOL_TYPES, AbstractNativeTool
from pydantic_ai.tools import AgentDepsT, AgentNativeTool

from .abstract import AbstractCapability

_BUILT_IN_NATIVE_TOOL_ADAPTER = pydantic.TypeAdapter(AbstractNativeTool)


@lru_cache(maxsize=1)
def _native_tool_adapter(tool_types: tuple[type[AbstractNativeTool], ...]) -> pydantic.TypeAdapter[AbstractNativeTool]:
    """Build the adapter from `NATIVE_TOOL_TYPES`, whose current contents `tool_types` passes as the cache key.

    A native tool type registers when its class is defined, which can be after this module is imported, while the
    `AgentSpec` schema reads the registry each time it's generated: an adapter built at import would reject the flat
    form of a type the schema accepts. The registry changes only when a class is defined, so the cache keeps only the
    adapter for its current contents.

    A registered type pydantic can't build a schema for fails the whole union. The built-in tools then load through
    the adapter built at import.
    """
    try:
        return pydantic.TypeAdapter(AbstractNativeTool)
    except pydantic.PydanticUserError:
        return _BUILT_IN_NATIVE_TOOL_ADAPTER


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

        `kind` names a type in [`NATIVE_TOOL_TYPES`][pydantic_ai.native_tools.NATIVE_TOOL_TYPES] as it stands when the
        spec loads, so a subclass defined after import loads too. A subclass that keeps a built-in `kind` replaces the
        built-in tool in the registry, so specs with that `kind` load the subclass.
        """
        adapter = _native_tool_adapter(tuple(NATIVE_TOOL_TYPES.values()))
        if tool is not None:
            validated = adapter.validate_python(tool)
        elif kwargs:
            validated = adapter.validate_python(kwargs)
        else:
            raise TypeError(
                '`NativeTool.from_spec()` requires either a `tool` argument or keyword arguments'
                ' specifying the native tool type (e.g. `kind="web_search"`)'
            )
        # Validation doesn't warn about moved fields, but a spec's author can act on the warning, so build the tool
        # again outside it. A tool passed in already warned when it was built. No user code is on the stack here, so
        # the warning names `dataclasses.py` as its location; the message names the tool and the field to change.
        if not isinstance(tool, AbstractNativeTool):
            validated = replace(validated)
        return cls(tool=validated)
