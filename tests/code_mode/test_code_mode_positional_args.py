"""Tests that document positional-arg mapping risks in CodeModeToolset."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_core import SchemaValidator, core_schema

from pydantic_ai._run_context import RunContext

try:
    from pydantic_ai.runtime import monty  # pyright: ignore[reportUnusedImport] # noqa: F401
except ImportError:  # pragma: lax no cover
    pytest.skip('pydantic-monty is not installed', allow_module_level=True)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

pytestmark = pytest.mark.anyio


class MisorderedParamsToolset(AbstractToolset[None]):
    @property
    def id(self) -> str | None:
        return None  # pragma: no cover

    async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
        tool_def = ToolDefinition(
            name='swap_tool',
            description='Return inputs in a map.',
            parameters_json_schema={
                'type': 'object',
                'properties': {'b': {'type': 'integer'}, 'a': {'type': 'integer'}},
                'required': ['a', 'b'],
            },
        )
        return {
            tool_def.name: ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=0,
                args_validator=SchemaValidator(core_schema.any_schema()),
            )
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
    ) -> dict[str, int | None]:
        return {'a': tool_args.get('a'), 'b': tool_args.get('b')}
