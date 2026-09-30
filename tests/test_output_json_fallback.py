import json
from collections.abc import AsyncIterator
from contextlib import nullcontext
from datetime import date
from typing import Annotated

import pytest
from inline_snapshot import snapshot
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, Strict, ValidationInfo

from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel


class StrictEvent(BaseModel):
    model_config = ConfigDict(strict=True)
    day: date


@pytest.mark.parametrize(
    'output_type,response,expected',
    [
        (Annotated[tuple[int, int], Strict()], [1, 2], (1, 2)),
        (Annotated[date, Strict()], '2026-09-30', date(2026, 9, 30)),
        (list[StrictEvent], [{'day': '2026-09-30'}], [StrictEvent(day=date(2026, 9, 30))]),
    ],
)
@pytest.mark.parametrize('json_args,stringified_response', [(True, False), (True, True), (False, True)])
def test_strict_json_output(
    output_type: type[object],
    response: object,
    *,
    expected: object,
    json_args: bool,
    stringified_response: bool,
):
    def respond(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args = {'response': json.dumps(response) if stringified_response else response}
        return ModelResponse([ToolCallPart(info.output_tools[0].name, json.dumps(args) if json_args else args)])

    agent = Agent(FunctionModel(respond), output_type=output_type, retries=0)
    assert agent.run_sync('Test').output == expected


@pytest.mark.parametrize('json_args', [False, True])
def test_json_string_fallback_validation_context(json_args: bool):
    modes: list[str] = []

    def check_context(value: int, info: ValidationInfo) -> int:
        assert info.context == {'tenant': 'acme'}
        modes.append(info.mode)
        return value

    def respond(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args = {'response': '[1, 2]'}
        return ModelResponse([ToolCallPart(info.output_tools[0].name, json.dumps(args) if json_args else args)])

    agent = Agent(
        FunctionModel(respond),
        output_type=list[Annotated[int, AfterValidator(check_context)]],
        validation_context={'tenant': 'acme'},
        retries=0,
    )
    assert agent.run_sync('Test').output == [1, 2]
    assert modes == ['json', 'json']


@pytest.mark.parametrize('valid_json', [False, True])
async def test_partial_defaulted_output_fallback(valid_json: bool):
    async def stream(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        assert info.output_tools is not None
        args = json.dumps({'response': '[1, 2]' if valid_json else 'not-json'})
        yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args=args)}

    agent = Agent(
        FunctionModel(stream_function=stream),
        output_type=Annotated[list[int], Field(default_factory=list)],
    )

    expectation = nullcontext() if valid_json else pytest.raises(UnexpectedModelBehavior, match='retries')
    with expectation:
        async with agent.run_stream('Test') as result:
            assert [output async for output in result.stream_output(debounce_by=None)] == snapshot([[1, 2], [1, 2]])


class Person(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize('stringified_response', [False, True])
async def test_partial_list_output(stringified_response: bool):
    response = [{'name': 'First', 'age': 1}, {'name': 'Second', 'age': 2}]
    args = json.dumps({'response': json.dumps(response) if stringified_response else response})
    split = args.index('Second') + 3

    async def stream(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        assert info.output_tools is not None
        yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args=args[:split])}
        yield {0: DeltaToolCall(json_args=args[split:])}

    agent = Agent(FunctionModel(stream_function=stream), output_type=list[Person])
    async with agent.run_stream('Test') as result:
        assert [output async for output in result.stream_output(debounce_by=None)] == snapshot(
            [
                [Person(name='First', age=1)],
                [Person(name='First', age=1), Person(name='Second', age=2)],
                [Person(name='First', age=1), Person(name='Second', age=2)],
            ]
        )
