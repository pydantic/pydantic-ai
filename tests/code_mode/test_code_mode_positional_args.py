"""Tests that document positional-arg mapping risks in CodeModeToolset."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_core import SchemaValidator, core_schema

from pydantic_ai._run_context import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


def build_run_context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


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


@pytest.mark.xfail(
    reason=(
        'CodeModeToolset maps positional args to JSON schema property order, '
        'which can differ from actual parameter order for external tools.'
    )
)
async def test_code_mode_positional_args_respects_signature_order():
    """Positional args should map to the tool signature, not schema property order."""
    code_mode = CodeModeToolset(wrapped=MisorderedParamsToolset())
    run_context = build_run_context()

    tools = await code_mode.get_tools(run_context)
    result = await code_mode.call_tool(
        'run_code',
        {'code': 'swap_tool(1, 2)'},
        run_context,
        tools['run_code'],
    )

    assert result == {'a': 1, 'b': 2}
