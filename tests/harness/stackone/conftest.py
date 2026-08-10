"""Shared fixtures for the StackOne capability tests."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

# `mcp` and `fastmcp` are gated on the `stackone` extra, so slim CI runs (no extras)
# can't import these modules; ignore them at collection. A conditional expression rather
# than an `if`: no single environment can take both arms of an install-dependent branch.
collect_ignore = (
    ['test_capability.py', 'test_toolset.py']
    if importlib.util.find_spec('mcp') is None or importlib.util.find_spec('fastmcp') is None
    else []
)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def run_context() -> RunContext[None]:
    """Minimal `RunContext` for invoking toolset methods directly in tests."""
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


def _fastmcp_server(name: str) -> FastMCP:
    """Create a test FastMCP server after resolving its settings forward references."""
    from mcp.server.fastmcp.server import FastMCP, Settings

    # The `mcp` SDK leaves the `lifespan` annotation on `Settings` as an unresolved forward reference,
    # and `pydantic-settings` >= 2.15 warns about that at construction, which this suite's `error`
    # warnings policy escalates. Drop the rebuild once `mcp` resolves the annotation itself.
    Settings.model_rebuild()
    return FastMCP(name)


@pytest.fixture
def stackone_server() -> FastMCP:
    """In-process stand-in for StackOne's MCP endpoint in `individual` tool mode."""
    server = _fastmcp_server('stackone-fake')

    @server.tool()
    def bamboohr_list_employees(limit: int = 10) -> list[dict[str, str]]:
        """List employees from BambooHR."""
        return [{'id': '1', 'name': 'Ada'}, {'id': '2', 'name': 'Grace'}][:limit]

    @server.tool()
    def bamboohr_create_employee(name: str) -> dict[str, str]:
        """Create an employee in BambooHR."""
        return {'id': '3', 'name': name}

    @server.tool()
    def bamboohr_export_employees(lines: int = 1, separator: str = '\n') -> str:
        """Export employee records."""
        return separator.join(f'employee-{index}' for index in range(lines))

    return server


@pytest.fixture
def search_execute_server() -> FastMCP:
    """In-process stand-in for StackOne's MCP endpoint in `search_execute` tool mode."""
    server = _fastmcp_server('stackone-fake-search')

    @server.tool()
    def bamboohr_search_actions(query: str, top_k: int = 10) -> list[dict[str, str]]:
        """Search available actions from a natural language query."""
        return [{'action_id': 'bamboohr_list_employees', 'description': 'List employees'}]

    @server.tool()
    def bamboohr_execute_action(action_id: str) -> dict[str, str]:
        """Execute an action by its id."""
        return {'action_id': action_id, 'status': 'ok'}

    return server
