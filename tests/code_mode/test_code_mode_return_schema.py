"""Code mode return schema signature tests."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_core import SchemaValidator, core_schema

from pydantic_ai._run_context import RunContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.code_mode import CodeModeToolset

from .conftest import build_run_context

pytestmark = pytest.mark.anyio


class ReturnSchemaToolset(AbstractToolset[None]):
    @property
    def id(self) -> str | None:
        return None  # pragma: no cover

    async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
        tool_def = ToolDefinition(
            name='get_data',
            description='Fetch data with pagination.',
            parameters_json_schema={'type': 'object', 'properties': {}},
            return_schema={
                'type': 'object',
                'properties': {
                    'items': {'type': 'array', 'items': {'type': 'string'}},
                    'next_cursor': {'type': 'string'},
                },
                'required': ['items', 'next_cursor'],
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
    ) -> dict[str, Any]:
        return {'items': [], 'next_cursor': ''}


async def test_code_mode_includes_return_schema_signature():
    code_mode = CodeModeToolset(wrapped=ReturnSchemaToolset())
    tools = await code_mode.get_tools(build_run_context())
    description = tools['run_code'].tool_def.description or ''

    assert 'class GetDataReturn(TypedDict):' in description
    assert 'def get_data() -> GetDataReturn' in description
