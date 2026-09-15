from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import CombinedToolset, FunctionToolset, PreparedToolset, ToolsetTool

pytestmark = pytest.mark.anyio


async def test_combined_toolset_dispatches_with_live_prepared_tool_definition():
    received_tool_defs: list[ToolDefinition] = []

    class SpyToolset(FunctionToolset[None]):
        async def call_tool(
            self,
            name: str,
            tool_args: dict[str, Any],
            ctx: RunContext[None],
            tool: ToolsetTool[None],
        ) -> Any:
            received_tool_defs.append(tool.tool_def)
            return await super().call_tool(name, tool_args, ctx, tool)

    spy = SpyToolset()

    @spy.tool_plain
    def my_tool(x: int) -> int:
        return x

    async def prepare(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        return [
            replace(
                tool_def,
                metadata={'dispatch': 'prepared'},
                strict=True,
            )
            for tool_def in tool_defs
        ]

    prepared = PreparedToolset(CombinedToolset([spy]), prepare)
    agent: Agent[None, str] = Agent(
        model=TestModel(call_tools=['my_tool']),
        deps_type=type(None),
        toolsets=[prepared],
    )

    await agent.run('Call my_tool.')

    assert len(received_tool_defs) == 1
    assert received_tool_defs[0].metadata == {'dispatch': 'prepared'}
    assert received_tool_defs[0].strict is True
