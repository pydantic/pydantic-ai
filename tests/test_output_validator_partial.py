"""Tests for output validators with partial parameter support."""

from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

TEST_OUTPUT = 'a' * 100


def test_run_sync():
    agent = Agent(TestModel(custom_output_text=TEST_OUTPUT))

    @agent.output_validator
    def validate_output(ctx: RunContext[None], output: str, partial: bool) -> str:
        if not partial and output != TEST_OUTPUT:
            raise ValueError('Output is not correct')
        return output

    agent.run_sync('test')


async def test_allow_partial_streaming_text():
    agent = Agent(TestModel(custom_output_text=TEST_OUTPUT))

    @agent.output_validator
    def validate_output(ctx: RunContext[None], output: str, partial: bool) -> str:
        if not partial and output != TEST_OUTPUT:
            raise ValueError('Output is not correct')
        return output

    async with agent.run_stream('test') as result:
        async for message, last in result.stream_responses(debounce_by=0.01):
            await result.validate_response_output(
                message,
                allow_partial=not last,
            )


async def test_allow_partial_streaming_structured_output():
    class OutputType(BaseModel):
        value: str

    agent = Agent(TestModel(custom_output_args=OutputType(value=TEST_OUTPUT)), output_type=OutputType)

    @agent.output_validator
    def validate_output(ctx: RunContext[None], output: OutputType, partial: bool) -> OutputType:
        if not partial and output.value != TEST_OUTPUT:
            raise ValueError('Output is not correct')
        return output

    async with agent.run_stream('test') as result:
        async for message, last in result.stream_responses(debounce_by=0.01):
            await result.validate_response_output(
                message,
                allow_partial=not last,
            )
