"""Importable helpers for Prefect tool-cache regression tests.

Tools used in Prefect cache-key tests must live in a normal module (not the test
module / `__main__`). cloudpickle serializes `__main__` callables by value,
including referenced globals, so a counter mutation would fork the task's own
source hash and accidentally mask cache-miss bugs (#6903 footnote, #6907).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic_ai import FunctionToolset, RunContext, ToolDefinition

SIDE_EFFECT_COUNTER = Path(tempfile.gettempdir()) / 'pydantic_ai_prefect_tool_side_effect_runs.txt'
side_effect_toolset = FunctionToolset[None](id='prefect-cache-side-effect')

_echo_invocations: list[tuple[object, object]] = []
echo_context_toolset = FunctionToolset[None](id='prefect-cache-echo-context')

prepare_calls: list[str] = []
prepare_toolset = FunctionToolset[None](id='prefect-cache-prepare')


@side_effect_toolset.tool
async def side_effect(ctx: RunContext[None]) -> str:
    n = int(SIDE_EFFECT_COUNTER.read_text() or '0') if SIDE_EFFECT_COUNTER.exists() else 0
    SIDE_EFFECT_COUNTER.write_text(str(n + 1))
    return 'ok'


@echo_context_toolset.tool
async def echo_context(ctx: RunContext[None]) -> str:
    """Answer depends only on RunContext fields the cache key must include."""
    _echo_invocations.append((ctx.prompt, ctx.metadata))
    return f'prompt={ctx.prompt!r} metadata={ctx.metadata!r}'


async def _counting_prepare(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
    prepare_calls.append(tool_def.name)
    return tool_def


async def _rename_prepare(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
    prepare_calls.append(tool_def.name)
    return ToolDefinition(
        name='exposed_name',
        description=tool_def.description,
        parameters_json_schema=tool_def.parameters_json_schema,
        timeout=tool_def.timeout,
    )


@prepare_toolset.tool(prepare=_counting_prepare)
async def alpha(ctx: RunContext[None]) -> str:
    return 'alpha-ok'


@prepare_toolset.tool(prepare=_counting_prepare)
async def beta(ctx: RunContext[None]) -> str:
    return 'beta-ok'


renamed_toolset = FunctionToolset[None](id='prefect-cache-renamed')


@renamed_toolset.tool(prepare=_rename_prepare)
async def raw_name(ctx: RunContext[None]) -> str:
    return 'renamed-ok'


def reset_echo_invocations() -> None:
    _echo_invocations.clear()


def echo_invocations() -> list[tuple[object, object]]:
    return list(_echo_invocations)


def reset_prepare_calls() -> None:
    prepare_calls.clear()
