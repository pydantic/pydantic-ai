from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeVar

import pytest
from inline_snapshot import snapshot

from pydantic_ai._run_context import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolset import FunctionToolset
from pydantic_ai.usage import Usage

pytestmark = pytest.mark.anyio

T = TypeVar('T')


def build_run_context(deps: T) -> RunContext[T]:
    return RunContext(deps=deps, model=TestModel(), usage=Usage(), prompt=None, messages=[], run_step=0)


async def test_function_toolset_prepare_for_run():
    @dataclass
    class PrefixDeps:
        prefix: str | None = None

    context = build_run_context(PrefixDeps())
    toolset = FunctionToolset[PrefixDeps]()

    async def prepare_add_prefix(ctx: RunContext[PrefixDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
        if ctx.deps.prefix is None:
            return tool_def

        return replace(tool_def, name=f'{ctx.deps.prefix}_{tool_def.name}')

    @toolset.tool(prepare=prepare_add_prefix)
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    assert toolset.tool_names == snapshot(['add'])
    assert toolset.tool_defs == snapshot(
        [
            ToolDefinition(
                name='add',
                description='Add two numbers',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            )
        ]
    )
    assert await toolset.call_tool(context, 'add', {'a': 1, 'b': 2}) == 3

    no_prefix_context = build_run_context(PrefixDeps())
    no_prefix_toolset = await toolset.prepare_for_run(no_prefix_context)
    assert no_prefix_toolset.tool_names == toolset.tool_names
    assert no_prefix_toolset.tool_defs == toolset.tool_defs
    assert await no_prefix_toolset.call_tool(no_prefix_context, 'add', {'a': 1, 'b': 2}) == 3

    foo_context = build_run_context(PrefixDeps(prefix='foo'))
    foo_toolset = await toolset.prepare_for_run(foo_context)
    assert foo_toolset.tool_names == snapshot(['foo_add'])
    assert foo_toolset.tool_defs == snapshot(
        [
            ToolDefinition(
                name='foo_add',
                description='Add two numbers',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            )
        ]
    )
    assert await foo_toolset.call_tool(foo_context, 'add', {'a': 1, 'b': 2}) == 3

    @toolset.tool
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b  # pragma: lax no cover

    assert foo_toolset.tool_names == snapshot(['foo_add'])

    bar_context = build_run_context(PrefixDeps(prefix='bar'))
    bar_toolset = await toolset.prepare_for_run(bar_context)
    assert bar_toolset.tool_names == snapshot(['bar_add', 'subtract'])
    assert bar_toolset.tool_defs == snapshot(
        [
            ToolDefinition(
                name='bar_add',
                description='Add two numbers',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='subtract',
                description='Subtract two numbers',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            ),
        ]
    )
    assert await bar_toolset.call_tool(bar_context, 'add', {'a': 1, 'b': 2}) == 3

    bar_foo_toolset = await foo_toolset.prepare_for_run(bar_context)
    assert bar_foo_toolset == bar_toolset
