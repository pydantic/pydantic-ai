"""Reach the connection a hosted MCP capability builds, under its tool prefix and filters."""

from __future__ import annotations

from typing import TypeVar

from pydantic_ai.mcp import MCPToolset
from pydantic_ai.toolsets import AbstractToolset

DepsT = TypeVar('DepsT')


def only_leaf(toolset: AbstractToolset[DepsT]) -> AbstractToolset[DepsT]:
    """The one toolset that lists and calls the tools: the MCP connection, or the per-run factory before a run."""
    leaves: list[AbstractToolset[DepsT]] = []
    toolset.apply(leaves.append)
    [leaf] = leaves
    return leaf


def mcp_connection(toolset: AbstractToolset[DepsT]) -> MCPToolset[DepsT]:
    connection = only_leaf(toolset)
    assert isinstance(connection, MCPToolset)
    return connection
