"""Tests for `RefineOutput`, driven through `Agent(..., capabilities=[RefineOutput(...)])`."""

from __future__ import annotations

from collections.abc import AsyncIterable

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, AgentRunResult, DeferredToolRequests, PendingModelResponse, RunContext
from pydantic_ai.capabilities import AbstractCapability, Hooks
from pydantic_ai.messages import (
    AgentStreamEvent,
    FinalResultEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import DEFAULT_REFINE_OUTPUT_INSTRUCTIONS, RefineOutput, SummarizingCompaction
from tests.harness._deferred_tools import BackgroundModel

from ._drafting_agent import numbered_model, reviews


class TestRefineOutput:
    async def test_reviews_the_output_once_by_default(self) -> None:
        agent = Agent(numbered_model(), system_prompt='Be brief.', capabilities=[RefineOutput()])

        result = await agent.run('Explain the change')

        assert result.output == 'answer 2'
        assert reviews(result.new_messages()) == [(DEFAULT_REFINE_OUTPUT_INSTRUCTIONS, 1)]

    async def test_reviews_for_the_configured_rounds(self) -> None:
        agent = Agent(numbered_model(), capabilities=[RefineOutput(rounds=2, instructions='Check the figures.')])

        result = await agent.run('Explain the change')

        assert result.output == 'answer 3'
        assert reviews(result.new_messages()) == [('Check the figures.', 1), ('Check the figures.', 2)]

    async def test_each_run_starts_with_no_rounds_used(self) -> None:
        agent = Agent(numbered_model(), capabilities=[RefineOutput()])
        first = await agent.run('Explain the change')

        second = await agent.run('And the tests?', message_history=first.all_messages())

        assert second.output == 'answer 4'
        assert reviews(second.new_messages()) == [(DEFAULT_REFINE_OUTPUT_INSTRUCTIONS, 1)]

    async def test_reviews_structured_output(self) -> None:
        class Total(BaseModel):
            value: int

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            value = sum(isinstance(message, ModelResponse) for message in messages) + 1
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'value': value})])

        agent = Agent(FunctionModel(model), output_type=Total, capabilities=[RefineOutput()])

        result = await agent.run('Add it up')

        assert result.output == Total(value=2)

    async def test_each_of_several_counts_its_own_rounds(self) -> None:
        agent = Agent(
            numbered_model(),
            capabilities=[
                RefineOutput(instructions='Check facts.'),
                RefineOutput(rounds=2, instructions='Check tone.'),
            ],
        )

        result = await agent.run('Explain the change')

        assert result.output == 'answer 3'
        assert sorted(reviews(result.new_messages())) == [('Check facts.', 1), ('Check tone.', 1), ('Check tone.', 2)]

    async def test_compaction_that_drops_the_review_does_not_restart_the_rounds(self) -> None:
        review_requests = 0
        tool_calls = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal review_requests, tool_calls
            review_requests += 'Review.' in str(messages[-1])
            if not review_requests:
                return ModelResponse(parts=[TextPart('draft')])
            if tool_calls < 3:
                tool_calls += 1
                return ModelResponse(parts=[ToolCallPart('look', {})])
            return ModelResponse(parts=[TextPart('revised')])

        def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart('The agent looked.')])

        compaction = SummarizingCompaction(model=FunctionModel(summarize), max_messages=6, keep_messages=4)
        agent = Agent(FunctionModel(model), capabilities=[RefineOutput(instructions='Review.'), compaction])

        @agent.tool_plain
        def look() -> str:
            return 'seen'

        result = await agent.run('Explain the change')

        assert result.output == 'revised'
        assert review_requests == 1
        assert reviews(result.all_messages()) == []

    async def test_does_not_review_a_run_paused_for_deferred_tools(self) -> None:
        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart('delete_file', {'path': 'notes.txt'})])

        agent = Agent(FunctionModel(model), output_type=[str, DeferredToolRequests], capabilities=[RefineOutput()])

        @agent.tool_plain(requires_approval=True)
        def delete_file(path: str) -> str:
            raise AssertionError('the call is never approved')

        result = await agent.run('Delete notes.txt')

        assert isinstance(result.output, DeferredToolRequests)
        assert reviews(result.new_messages()) == []

    async def test_does_not_review_a_run_paused_for_a_background_response(self) -> None:
        requests: list[None] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            requests.append(None)
            return ModelResponse(parts=[], provider_response_id='job-1', state='suspended')

        agent = Agent(BackgroundModel(model), output_type=[str, PendingModelResponse], capabilities=[RefineOutput()])

        result = await agent.run('Explain the change')

        assert isinstance(result.output, PendingModelResponse)
        assert len(requests) == 1

    @pytest.mark.parametrize('instances', [1, 2])
    async def test_run_stream_ends_at_the_first_output_unreviewed(self, instances: int) -> None:
        hooks = Hooks[object]()
        finished: list[str] = []

        @hooks.on.after_run
        async def record(ctx: RunContext[object], *, result: AgentRunResult[str]) -> AgentRunResult[str]:
            finished.append(result.output)
            return result

        capabilities: list[AbstractCapability[object]] = [hooks, *(RefineOutput() for _ in range(instances))]
        agent = Agent(numbered_model(), capabilities=capabilities)

        async with agent.run_stream('Explain the change') as stream:
            assert await stream.get_output() == 'answer 1'

        assert finished == ['answer 1']
        assert reviews(stream.all_messages()) == []

    async def test_reviews_a_run_streamed_through_an_event_handler(self) -> None:
        agent = Agent(numbered_model(), capabilities=[RefineOutput()])
        answers = 0

        async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            nonlocal answers
            async for event in stream:
                answers += isinstance(event, FinalResultEvent)

        result = await agent.run('Explain the change', event_stream_handler=handler)

        assert result.output == 'answer 2'
        assert answers == 2

    def test_rejects_fewer_than_one_round(self) -> None:
        with pytest.raises(ValueError, match='`rounds` must be at least 1, got 0'):
            RefineOutput(rounds=0)
