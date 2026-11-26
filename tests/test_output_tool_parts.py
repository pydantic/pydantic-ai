"""Output tool call and return parts carry `tool_kind='output'`, so code can tell them from function tools' without their name."""

from __future__ import annotations

import warnings
from typing import Literal

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry, RunContext, ToolOutput
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import DEFAULT_OUTPUT_TOOL_NAME, is_output_tool_part


class Report(BaseModel):
    title: str


def tool_parts(messages: list[ModelMessage]) -> list[tuple[str, str, str | None, str]]:
    """Each tool call and return part as `(part_kind, tool_name, tool_kind, outcome)`."""
    return [
        (part.part_kind, part.tool_name, part.tool_kind, part.outcome if isinstance(part, ToolReturnPart) else '')
        for message in messages
        for part in message.parts
        if isinstance(part, ToolCallPart | ToolReturnPart)
    ]


@pytest.mark.parametrize('route', ['run', 'run_stream'])
@pytest.mark.parametrize('output_tool_name', [None, 'submit_report'])
async def test_output_tool_parts_are_marked(route: Literal['run', 'run_stream'], output_tool_name: str | None):
    output_type = ToolOutput(Report, name=output_tool_name) if output_tool_name else Report
    agent = Agent(TestModel(), output_type=output_type)

    @agent.tool_plain
    def get_answer() -> str:
        return '42'

    if route == 'run':
        messages = (await agent.run('Write the report.')).all_messages()
    else:
        async with agent.run_stream('Write the report.') as result:
            await result.get_output()
        messages = result.all_messages()

    name = output_tool_name or DEFAULT_OUTPUT_TOOL_NAME
    assert tool_parts(messages) == [
        ('tool-call', 'get_answer', None, ''),
        ('tool-return', 'get_answer', None, 'success'),
        ('tool-call', name, 'output', ''),
        ('tool-return', name, 'output', 'success'),
    ]
    assert [part.part_kind for message in messages for part in message.parts if is_output_tool_part(part)] == [
        'tool-call',
        'tool-return',
    ]


async def test_output_tool_parts_survive_serialization():
    result = await Agent(TestModel(), output_type=ToolOutput(Report, name='submit_report')).run('Write the report.')
    messages = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(result.all_messages()))
    assert [part.tool_kind for message in messages for part in message.parts if is_output_tool_part(part)] == [
        'output',
        'output',
    ]


async def test_output_tool_retries_are_marked():
    """A retry for invalid arguments and one an output validator asks for both carry the output kind."""
    calls = 0

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        args = [{'wrong': 'field'}, {'title': 'draft'}, {'title': 'final'}][calls - 1]
        return ModelResponse(parts=[ToolCallPart('submit_report', args, tool_call_id=f'call_{calls}')])

    agent = Agent(FunctionModel(model), output_type=ToolOutput(Report, name='submit_report'), retries={'output': 2})

    @agent.output_validator
    def not_a_draft(ctx: RunContext[None], output: Report) -> Report:
        if output.title == 'draft':
            raise ModelRetry('Not a draft, please.')
        return output

    result = await agent.run('Write the report.')
    assert tool_parts(result.all_messages()) == [
        ('tool-call', 'submit_report', 'output', ''),
        ('tool-return', 'submit_report', 'output', 'retried'),
        ('tool-call', 'submit_report', 'output', ''),
        ('tool-return', 'submit_report', 'output', 'retried'),
        ('tool-call', 'submit_report', 'output', ''),
        ('tool-return', 'submit_report', 'output', 'success'),
    ]


async def test_skipped_output_tool_call_is_marked():
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('submit_report', {'title': 'first'}, tool_call_id='call_1'),
                ToolCallPart('submit_report', {'title': 'second'}, tool_call_id='call_2'),
            ]
        )

    agent = Agent(FunctionModel(model), output_type=ToolOutput(Report, name='submit_report'))
    result = await agent.run('Write the report.')
    assert result.output == Report(title='first')
    assert tool_parts(result.all_messages()) == [
        ('tool-call', 'submit_report', 'output', ''),
        ('tool-call', 'submit_report', 'output', ''),
        ('tool-return', 'submit_report', 'output', 'success'),
        ('tool-return', 'submit_report', 'output', 'success'),
    ]


async def test_streamed_output_tool_call_is_marked_from_its_first_event():
    async def stream_model(messages: list[ModelMessage], info: AgentInfo):
        yield {0: DeltaToolCall(name='submit_report', json_args='{"title": ', tool_call_id='call_1')}
        yield {0: DeltaToolCall(json_args='"done"}')}

    agent = Agent(FunctionModel(stream_function=stream_model), output_type=ToolOutput(Report, name='submit_report'))
    async with agent.run_stream('Write the report.') as result:
        responses = [response async for response in result.stream_response(debounce_by=None)]
    first_call = next(part for response in responses for part in response.parts if isinstance(part, ToolCallPart))
    assert first_call.tool_kind == 'output'


def test_is_output_tool_part_falls_back_to_the_default_name_for_unmarked_parts():
    """Histories stored before output tool parts were marked are recognized by the default output tool name."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        legacy_retry = RetryPromptPart('Fix it.', tool_name=DEFAULT_OUTPUT_TOOL_NAME, tool_call_id='call_1')  # pyright: ignore[reportDeprecated]
    assert is_output_tool_part(ToolCallPart(DEFAULT_OUTPUT_TOOL_NAME, {'title': 'x'}, 'call_1'))
    assert is_output_tool_part(ToolReturnPart(DEFAULT_OUTPUT_TOOL_NAME, 'Final result processed.', 'call_1'))
    assert is_output_tool_part(legacy_retry)
    # An unmarked part of a custom-named output tool can't be told from a function tool's.
    assert not is_output_tool_part(ToolCallPart('submit_report', {'title': 'x'}, 'call_1'))
    # A marked part goes by its kind, not its name.
    assert not is_output_tool_part(ToolCallPart(DEFAULT_OUTPUT_TOOL_NAME, {}, 'call_1', tool_kind='inventory.lookup'))
    assert is_output_tool_part(ToolCallPart('submit_report', {}, 'call_1', tool_kind='output'))
    assert not is_output_tool_part(TextPart('final_result'))
    assert not is_output_tool_part(UserPromptPart('final_result'))
