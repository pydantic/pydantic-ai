from __future__ import annotations

from typing import Any, Callable

from pydantic_ai.mcp import MCPServer
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset

from ._function_toolset import temporalize_function_toolset
from ._mcp_server import temporalize_mcp_server
from ._settings import TemporalSettings


def temporalize_toolset(toolset: AbstractToolset, settings: TemporalSettings | None) -> list[Callable[..., Any]]:
    """Temporalize a toolset.

    Args:
        toolset: The toolset to temporalize.
        settings: The temporal settings to use.
    """
    if isinstance(toolset, FunctionToolset):
        return temporalize_function_toolset(toolset, settings)
    elif isinstance(toolset, MCPServer):
        return temporalize_mcp_server(toolset, settings)
    else:
        return []
