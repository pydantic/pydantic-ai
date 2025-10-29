"""Tests for output validators with partial parameter support."""

from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel


def run_sync_no_partial_calls():
    partial_calls: list[str] = []
    final_calls: list[str] = []

    agent = Agent(TestModel(seed=0))

    @agent.output_validator
    def validate_output(ctx: RunContext[None], output: str, partial: bool) -> str:
        if partial:
            partial_calls.append(output)
            return output
        else:
            final_calls.append(output)
            return output

    agent.run_sync('test')
    assert len(partial_calls) == 0, 'Should have no partial validation calls during sync run'
    assert len(final_calls) == 1, 'Should have one final validation call'


async def test_allow_partial_streaming_text():
    partial_calls: list[str] = []
    final_calls: list[str] = []

    agent = Agent(TestModel(seed=0))

    @agent.output_validator
    def validate_output(ctx: RunContext[None], output: str, partial: bool) -> str:
        if partial:
            partial_calls.append(output)
        else:
            final_calls.append(output)
        return output

    async with agent.run_stream('test') as result:
        async for message, last in result.stream_responses(debounce_by=0.01):
            await result.validate_response_output(
                message,
                allow_partial=not last,
            )

    assert len(partial_calls) > 0, 'Should have received partial validation calls during streaming'
    assert len(final_calls) == 1, 'Should have one final validation call'


async def test_allow_partial_streaming_structured_output():
    class OutputType(BaseModel):
        value: str

    partial_calls: list[OutputType] = []
    final_calls: list[OutputType] = []

    agent = Agent(TestModel(seed=0), output_type=OutputType)

    @agent.output_validator
    def validate_output(ctx: RunContext[None], output: OutputType, partial: bool) -> OutputType:
        if partial:
            partial_calls.append(output)
        else:
            final_calls.append(output)
        return output

    async with agent.run_stream('test') as result:
        async for message, last in result.stream_responses(debounce_by=0.01):
            await result.validate_response_output(
                message,
                allow_partial=not last,
            )

    assert len(partial_calls) > 0, 'Should have received partial validation calls during streaming'
    assert len(final_calls) == 1, 'Should have one final validation call'
