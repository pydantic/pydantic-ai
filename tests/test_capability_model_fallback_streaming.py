from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import Literal

import pytest

from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent, ModelRetry, RunContext
from pydantic_ai.capabilities import ModelFallback, OutputRetryEvent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel


@pytest.fixture
def streaming_agent() -> tuple[Agent[object, str], list[str]]:
    calls: list[str] = []

    async def primary(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        calls.append('primary')
        yield 'draft'

    async def backup(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        calls.append('backup')
        yield 'ready'

    agent = Agent(
        deps_type=object,
        capabilities=[
            ModelFallback(
                FunctionModel(stream_function=primary), FunctionModel(stream_function=backup), output_retries=1
            )
        ],
    )

    @agent.output_validator
    def validate(ctx: RunContext[object], value: str) -> str:
        if value == 'draft' and not ctx.partial_output:
            raise ModelRetry('Finish the draft.')
        return value

    return agent, calls


class TestStreaming:
    @pytest.mark.parametrize('method', ['handler', 'events', 'iter'])
    async def test_graph_streaming_can_advance(
        self, streaming_agent: tuple[Agent[object, str], list[str]], method: Literal['handler', 'events', 'iter']
    ) -> None:
        agent, calls = streaming_agent
        events: list[AgentStreamEvent] = []
        if method == 'handler':

            async def collect(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
                events.extend([event async for event in stream])

            result = await agent.run('Answer.', event_stream_handler=collect)
            assert result.output == 'ready'
        elif method == 'events':
            output: str | None = None
            async with agent.run_stream_events('Answer.') as stream:
                async for event in stream:
                    if isinstance(event, AgentRunResultEvent):
                        output = event.result.output
                    else:
                        events.append(event)
            assert output == 'ready'
        else:
            async with agent.iter('Answer.') as run:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as stream:
                            events.extend([event async for event in stream])
                assert run.result is not None
                assert run.result.output == 'ready'
        assert calls == ['primary', 'backup']
        if method != 'iter':
            retries = [event for event in events if isinstance(event, OutputRetryEvent)]
            assert len(retries) == 1

    async def test_delivered_final_stream_is_not_replaced(
        self, streaming_agent: tuple[Agent[object, str], list[str]]
    ) -> None:
        agent, calls = streaming_agent
        delivered: list[str] = []
        with pytest.raises(UnexpectedModelBehavior):
            async with agent.run_stream('Answer.') as result:
                async for value in result.stream_output(debounce_by=None):
                    delivered.append(value)
        assert 'draft' in delivered
        assert calls == ['primary']
