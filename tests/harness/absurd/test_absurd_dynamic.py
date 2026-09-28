"""`DynamicToolset` under `AbsurdDurability`.

`pydantic-ai-absurd` checkpoints function and MCP toolsets only, so a construction-time
`DynamicToolset` runs as-is inside a task: its resolution and tool calls are not steps, and a replay
runs them again. Adding one per run is still rejected, as it is for the other executing kinds.
"""

from __future__ import annotations

import pytest

pytest.importorskip('absurd_sdk')

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import DynamicToolset, FunctionToolset
from pydantic_ai_harness.absurd import AbsurdDurability

from ._helpers import FakeAsyncTaskContext, absurd_task_context


def _greet_then_done_model() -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='greet', args={'name': 'ada'})])

    return FunctionModel(fn, model_name='fn')


def _dynamic_toolset(tool_calls: dict[str, int], *, id: str | None) -> DynamicToolset[object]:
    def build(ctx: RunContext[object]) -> FunctionToolset[object]:
        inner: FunctionToolset[object] = FunctionToolset(id='inner')

        @inner.tool_plain
        def greet(name: str) -> str:
            tool_calls['n'] += 1
            return f'hi {name}'

        return inner

    return DynamicToolset(build, id=id)


class TestDynamicToolset:
    @pytest.mark.parametrize('toolset_id', ['dyn', None])
    async def test_runs_uncheckpointed_inside_a_task(self, toolset_id: str | None) -> None:
        tool_calls = {'n': 0}
        agent = Agent(
            _greet_then_done_model(),
            name='d',
            toolsets=[_dynamic_toolset(tool_calls, id=toolset_id)],
            capabilities=[AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('greet ada')
        assert sorted(ctx.stored) == ['d__model.request', 'd__model.request#2']

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('greet ada')

        # The model responses replay from their checkpoints; the dynamic tool runs again.
        assert first.output == second.output == 'done'
        assert tool_calls['n'] == 2
        assert replay.invoked == []

    async def test_runtime_dynamic_toolset_rejected_inside_task(self) -> None:
        tool_calls = {'n': 0}
        agent = Agent(_greet_then_done_model(), name='d', capabilities=[AbsurdDurability()])

        with absurd_task_context(FakeAsyncTaskContext()):
            with pytest.raises(UserError, match='cannot be added at runtime with Absurd'):
                await agent.run('greet ada', toolsets=[_dynamic_toolset(tool_calls, id='late')])
        assert tool_calls['n'] == 0
