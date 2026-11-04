"""`AbsurdDurability` on a real Absurd worker: a retried attempt is served from checkpoints."""

from __future__ import annotations

import pytest

pytest.importorskip('absurd_sdk')

from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue

from pydantic_ai import Agent, ModelMessage, ModelResponse
from pydantic_ai.messages import TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.absurd import AbsurdDurability


async def test_worker_retry_replays_postgres_checkpoints(absurd: AsyncAbsurd) -> None:
    """A retried attempt serves the model and tool calls from the first attempt's checkpoints."""
    calls = {'attempts': 0, 'model': 0, 'tool': 0}
    toolset = FunctionToolset[object](id='billing')

    @toolset.tool_plain
    def charge_card(amount: int) -> str:
        calls['tool'] += 1
        return f'charged {amount}'

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls['model'] += 1
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='charge_card', args={'amount': 42})])

    agent = Agent[object, str](
        FunctionModel(model, model_name='fn'),
        name='checkout',
        toolsets=[toolset],
        capabilities=[AbsurdDurability()],
    )

    @absurd.register_task(name='checkout')
    async def checkout(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:
        calls['attempts'] += 1
        result = await agent.run('charge it')
        if calls['attempts'] == 1:
            raise RuntimeError('simulated worker crash after checkpoints were committed')
        return {'output': result.output}

    spawned = await absurd.spawn(
        'checkout',
        None,
        max_attempts=2,
        retry_strategy={'kind': 'fixed', 'base_seconds': 0},
    )
    await absurd.work_batch(batch_size=1)
    await absurd.work_batch(batch_size=1)
    result = await absurd.fetch_task_result(spawned['task_id'])

    assert result is not None
    assert result.state == 'completed'
    assert result.result == {'output': 'done'}
    assert calls == {'attempts': 2, 'model': 2, 'tool': 1}
