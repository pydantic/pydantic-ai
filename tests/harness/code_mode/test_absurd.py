"""Composition test: `CodeMode` under `AbsurdDurability`.

`AbsurdDurability` is transparent outside a task, so a passing run alone proves nothing about
durability. The test runs inside a `FakeAsyncTaskContext` and drives a full `run_code` execution
whose generated code calls an underlying function tool, then asserts the inner tool call is
checkpointed as its own step and served from the checkpoint on replay.

Both extras can be absent on a CI leg, so the module `importorskip`s `absurd_sdk` and
`pydantic_monty` (CodeMode's sandbox).
"""

from __future__ import annotations

import pytest

pytest.importorskip('absurd_sdk')
pytest.importorskip('pydantic_monty')

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness import CodeMode
from pydantic_ai_harness.absurd import AbsurdDurability
from tests.harness.absurd._helpers import (
    FakeAsyncTaskContext,
    absurd_task_context,
)

pytestmark = pytest.mark.anyio


def _code_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """Emit one `run_code` call whose code invokes `search`, then answer from its return."""
    for message in messages:
        if isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
                return ModelResponse(parts=[TextPart(content=f'done: {part.content}')])
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='run_code',
                args={'code': "result = await search(query='x')\nresult"},
                tool_call_id='tc1',
            )
        ]
    )


class TestCodeModeUnderAbsurd:
    async def test_inner_sandboxed_tool_call_is_checkpointed(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def search(query: str) -> str:
            calls['n'] += 1
            return f'results for {query}'

        agent = Agent(
            FunctionModel(_code_model),
            name='composed',
            toolsets=[toolset],
            capabilities=[CodeMode(), AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('go')

        assert first.output == 'done: results for x'
        assert calls['n'] == 1
        # CodeMode dispatches each sandbox tool call through its wrapped toolset, which
        # AbsurdDurability (the innermost capability) already swapped to the durable wrapper.
        # So the `search` call made from inside `run_code` is checkpointed as its own step,
        # the same as a direct tool call would be.
        tool_step = 'composed__function_toolset__tools.call_tool:search'
        assert tool_step in ctx.stored
        assert tool_step in ctx.invoked
        assert 'composed__model.request' in ctx.stored

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('go')

        # The `run_code` body is plain task-body Python and re-runs on replay, so it calls
        # `search` again, but the durable step short-circuits to the stored result: the tool
        # function does not run a second time and the model is not re-called.
        assert second.output == 'done: results for x'
        assert calls['n'] == 1
        assert replay.invoked == []
