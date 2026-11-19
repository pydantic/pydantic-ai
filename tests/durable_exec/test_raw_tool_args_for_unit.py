"""Which function tool calls [`raw_tool_args_for_unit`][pydantic_ai.durable_exec.raw_tool_args_for_unit] lets a durable unit validate from the model's arguments.

These run in-process rather than through an engine: every engine's model response crosses JSON before its tools
run, so a value JSON would change only reaches a unit in arguments built workflow-side, like an approval's
`override_args`. The engines' own suites cover the arguments a model sends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter

from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec import ToolsetCallToolParams, raw_tool_args_for_unit
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset


@dataclass(frozen=True)
class Case:
    id: str
    args: str | dict[str, Any] | None
    sent: bool
    """Whether the call's own arguments are returned for the unit to validate."""
    changed_by_hook: bool = False
    dump_reproduces: bool = False
    """Whether the unit's load of the dumped arguments validates back to them."""


CASES = [
    Case('json-string', '{"pair": [1.5, "x"], "labels": {"1": 2}}', sent=True),
    Case('json-object', {'pair': [1.5, 'x'], 'labels': {'1': 2}}, sent=True),
    Case('dump-reproduces', {'pair': [1.5, 'x']}, sent=False, dump_reproduces=True),
    Case('no-arguments', None, sent=False),
    Case('tuple', {'pair': (1.5, 'x')}, sent=False),
    Case('infinite-float', {'pair': [math.inf, 'x']}, sent=False),
    Case('integer-key', {'labels': {1: 2}}, sent=False),
    Case('changed-by-hook', {'pair': [1.5, 'x']}, sent=False, changed_by_hook=True),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_raw_tool_args_for_unit(case: Case):
    calls: list[tuple[RunContext[None], dict[str, Any]]] = []
    toolset = FunctionToolset[None]()

    @toolset.tool
    def record(ctx: RunContext[None], pair: tuple[float, str] = (0.0, ''), labels: dict[int, int] | None = None) -> str:
        calls.append((ctx, {'pair': pair, 'labels': labels}))
        return 'recorded'

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('record', case.args)])

    await Agent(FunctionModel(model), deps_type=type(None), toolsets=[toolset]).run('go')
    [(ctx, tool_args)] = calls
    if case.changed_by_hook:
        tool_args = {**tool_args, 'pair': (9.0, 'y')}
    tool = (await toolset.get_tools(ctx))['record']

    # A dump the unit reads back as the defaults stands in for one that loses the arguments.
    loaded_tool_args: dict[str, Any] = {}
    if case.dump_reproduces:
        loaded_tool_args = TypeAdapter(dict[str, Any]).dump_python(tool_args, mode='json')

    params = ToolsetCallToolParams('record', tool_args=tool_args, ctx=ctx, tool=tool)
    assert raw_tool_args_for_unit(toolset, params, loaded_tool_args=loaded_tool_args) == (
        case.args if case.sent else None
    )


class Filters(BaseModel):
    max_price: float


async def test_check_that_raises_keeps_the_dump():
    """A hook that drops a single model argument's key makes validating the dump raise `KeyError`, not `ValidationError`.

    The check leaves the call on today's path, so the unit raises it there instead of the workflow.
    """
    calls: list[RunContext[None]] = []
    toolset = FunctionToolset[None]()

    @toolset.tool
    def lookup(ctx: RunContext[None], filters: Filters) -> str:
        calls.append(ctx)
        return 'found'

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('lookup', {'max_price': 10})])

    await Agent(FunctionModel(model), deps_type=type(None), toolsets=[toolset]).run('go')
    [ctx] = calls
    tool = (await toolset.get_tools(ctx))['lookup']

    params = ToolsetCallToolParams('lookup', tool_args={}, ctx=ctx, tool=tool)
    assert raw_tool_args_for_unit(toolset, params, loaded_tool_args={}) is None
