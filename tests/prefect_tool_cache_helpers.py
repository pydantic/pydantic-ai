"""Importable helpers for Prefect tool-cache regression tests.

Tools used in Prefect cache-key tests must live in a normal module (not the test
module / `__main__`). cloudpickle serializes `__main__` callables by value,
including referenced globals, so a counter mutation would fork the task's own
source hash and accidentally mask cache-miss bugs (#6903 footnote, #6907).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic_ai import FunctionToolset, RunContext

SIDE_EFFECT_COUNTER = Path(tempfile.gettempdir()) / 'pydantic_ai_prefect_tool_side_effect_runs.txt'
side_effect_toolset = FunctionToolset[None](id='prefect-cache-side-effect')

_echo_invocations: list[tuple[object, object]] = []
echo_context_toolset = FunctionToolset[None](id='prefect-cache-echo-context')


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


def reset_echo_invocations() -> None:
    _echo_invocations.clear()


def echo_invocations() -> list[tuple[object, object]]:
    return list(_echo_invocations)
