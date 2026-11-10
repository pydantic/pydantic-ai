"""Tests for `RefineOutput`, driven through `Agent(..., capabilities=[RefineOutput(...)])`."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Sequence
from itertools import count

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    AgentStreamEvent,
    FinalResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai_harness import DEFAULT_REFINE_OUTPUT_INSTRUCTIONS, RefineOutput

ORIGIN = 'pydantic-ai-harness.refine_output.v1'


def _numbered() -> FunctionModel:
    """A model that answers `'answer 1'`, `'answer 2'`, and so on, streamed or not."""
    number = count(1)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(f'answer {next(number)}')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield f'answer {next(number)}'

    return FunctionModel(model, stream_function=stream)


def _reviews(messages: Sequence[ModelMessage]) -> list[tuple[str, object]]:
    """The content and round of each review instruction in `messages`."""
    return [
        (part.content, part.metadata['round'])
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart) and part.metadata and part.metadata['origin'] == ORIGIN
    ]


class TestRefineOutput:
    async def test_reviews_the_output_once_by_default(self) -> None:
        agent = Agent(_numbered(), system_prompt='Be brief.', capabilities=[RefineOutput()])

        result = await agent.run('Explain the change')

        assert result.output == 'answer 2'
        assert _reviews(result.new_messages()) == [(DEFAULT_REFINE_OUTPUT_INSTRUCTIONS, 1)]

    async def test_reviews_for_the_configured_rounds(self) -> None:
        agent = Agent(_numbered(), capabilities=[RefineOutput(rounds=2, instructions='Check the figures.')])

        result = await agent.run('Explain the change')

        assert result.output == 'answer 3'
        assert _reviews(result.new_messages()) == [('Check the figures.', 1), ('Check the figures.', 2)]

    async def test_each_run_starts_with_no_rounds_used(self) -> None:
        agent = Agent(_numbered(), capabilities=[RefineOutput()])
        first = await agent.run('Explain the change')

        second = await agent.run('And the tests?', message_history=first.all_messages())

        assert second.output == 'answer 4'
        assert _reviews(second.new_messages()) == [(DEFAULT_REFINE_OUTPUT_INSTRUCTIONS, 1)]

    async def test_reviews_structured_output(self) -> None:
        class Total(BaseModel):
            value: int

        number = count(1)

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'value': next(number)})])

        agent = Agent(FunctionModel(model), output_type=Total, capabilities=[RefineOutput()])

        result = await agent.run('Add it up')

        assert result.output == Total(value=2)

    async def test_several_share_the_round_count(self) -> None:
        agent = Agent(
            _numbered(),
            capabilities=[
                RefineOutput(instructions='Check facts.'),
                RefineOutput(rounds=2, instructions='Check tone.'),
            ],
        )

        result = await agent.run('Explain the change')

        assert result.output == 'answer 3'
        assert sorted(_reviews(result.new_messages())) == [('Check facts.', 1), ('Check tone.', 1), ('Check tone.', 2)]

    async def test_does_not_review_a_run_paused_for_deferred_tools(self) -> None:
        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart('delete_file', {'path': 'notes.txt'})])

        agent = Agent(FunctionModel(model), output_type=[str, DeferredToolRequests], capabilities=[RefineOutput()])

        @agent.tool_plain(requires_approval=True)
        def delete_file(path: str) -> str:
            raise AssertionError('the call is never approved')

        result = await agent.run('Delete notes.txt')

        assert isinstance(result.output, DeferredToolRequests)
        assert _reviews(result.new_messages()) == []

    async def test_run_stream_ends_at_the_first_output(self) -> None:
        agent = Agent(_numbered(), capabilities=[RefineOutput()])

        async with agent.run_stream('Explain the change') as stream:
            assert await stream.get_output() == 'answer 1'

    async def test_reviews_a_run_streamed_through_an_event_handler(self) -> None:
        agent = Agent(_numbered(), capabilities=[RefineOutput()])
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
