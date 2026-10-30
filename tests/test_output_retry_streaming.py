from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

import pytest

from pydantic_ai import Agent, EndStrategy, RunContext, Tool, ToolOutput
from pydantic_ai.capabilities import Hooks, OutputRetryContext
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel


@dataclass(frozen=True)
class Case:
    end_strategy: EndStrategy
    expected_retries: tuple[tuple[str, int], ...] = ()


CASES: tuple[Case, ...] = (
    Case('early'),
    Case('graceful'),
    Case('exhaustive', (('bad', 1),)),
)


@pytest.mark.parametrize('case', [pytest.param(case, id=case.end_strategy) for case in CASES])
@pytest.mark.parametrize('method', ['run', 'run_stream'])
async def test_final_result_notifies_accepted_output_tool_retries(
    case: Case, method: Literal['run', 'run_stream']
) -> None:
    received: list[OutputRetryContext] = []
    calls: list[str] = []

    def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('request')
        return ModelResponse(
            parts=[ToolCallPart('good', {'response': 42}, 'good'), ToolCallPart('bad', {'response': 'invalid'}, 'bad')]
        )

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[dict[int, DeltaToolCall]]:
        calls.append('request')
        yield {
            0: DeltaToolCall(name='good', json_args='{"response":42}', tool_call_id='good'),
            1: DeltaToolCall(name='bad', json_args='{"response":"invalid"}', tool_call_id='bad'),
        }

    def record(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:
        assert ctx.retry == 1
        assert retry_context.source == 'output_tool'
        assert retry_context.tool_call_id == 'bad'
        received.append(retry_context)

    agent = Agent(
        FunctionModel(response, stream_function=stream),
        deps_type=object,
        output_type=[ToolOutput(int, name='good'), ToolOutput(int, name='bad', max_retries=1)],
        end_strategy=case.end_strategy,
        capabilities=[Hooks(output_retry=record)],
    )
    if method == 'run':
        result = await agent.run('Answer.')
        assert result.output == 42
        messages = result.all_messages()
    else:
        async with agent.run_stream('Answer.') as streamed:
            assert await streamed.get_output() == 42
            messages = streamed.all_messages()

    assert tuple((retry.tool_name, retry.retry.attempt) for retry in received) == case.expected_retries
    assert (
        tuple(
            (part.tool_name, part.retry.attempt)
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.retry is not None
        )
        == case.expected_retries
    )
    assert calls == ['request']


async def test_function_tool_error_does_not_notify_uncharged_output_retries() -> None:
    received: list[OutputRetryContext] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[dict[int, DeltaToolCall]]:
        yield {
            0: DeltaToolCall(name='good', json_args='{"response":42}', tool_call_id='good'),
            1: DeltaToolCall(name='bad', json_args='{"response":"invalid"}', tool_call_id='bad'),
            2: DeltaToolCall(name='explode', json_args='{}', tool_call_id='explode'),
        }

    def explode() -> int:
        raise RuntimeError('tool failed')

    def record(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:
        received.append(retry_context)

    agent = Agent(
        FunctionModel(stream_function=stream),
        deps_type=object,
        output_type=[ToolOutput(int, name='good'), ToolOutput(int, name='bad', max_retries=1)],
        tools=[Tool(explode, sequential=True)],
        end_strategy='exhaustive',
        capabilities=[Hooks(output_retry=record)],
    )
    with pytest.raises(RuntimeError, match='tool failed'):
        async with agent.run_stream('Answer.') as streamed:
            await streamed.get_output()
    assert received == []
