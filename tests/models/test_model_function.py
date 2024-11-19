import json
import re
from collections.abc import AsyncIterator
from dataclasses import asdict
from datetime import timezone

import pydantic_core
import pytest
from dirty_equals import IsStr
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, CallContext, ModelRetry
from pydantic_ai.messages import (
    Message,
    ModelAnyResponse,
    ModelStructuredResponse,
    ModelTextResponse,
    SystemPrompt,
    ToolCall,
    ToolReturn,
    UserPrompt,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import Cost
from tests.conftest import IsNow

pytestmark = pytest.mark.anyio


async def return_last(messages: list[Message], _: AgentInfo) -> ModelAnyResponse:
    last = messages[-1]
    response = asdict(last)
    response.pop('timestamp', None)
    response['message_count'] = len(messages)
    return ModelTextResponse(' '.join(f'{k}={v!r}' for k, v in response.items()))


def test_simple():
    agent = Agent(FunctionModel(return_last))
    result = agent.run_sync('Hello')
    assert result.data == snapshot("content='Hello' role='user' message_count=1")
    assert result.all_messages() == snapshot(
        [
            UserPrompt(
                content='Hello',
                timestamp=IsNow(tz=timezone.utc),
                role='user',
            ),
            ModelTextResponse(
                content="content='Hello' role='user' message_count=1",
                timestamp=IsNow(tz=timezone.utc),
                role='model-text-response',
            ),
        ]
    )

    result2 = agent.run_sync('World', message_history=result.all_messages())
    assert result2.data == snapshot("content='World' role='user' message_count=3")
    assert result2.all_messages() == snapshot(
        [
            UserPrompt(
                content='Hello',
                timestamp=IsNow(tz=timezone.utc),
                role='user',
            ),
            ModelTextResponse(
                content="content='Hello' role='user' message_count=1",
                timestamp=IsNow(tz=timezone.utc),
                role='model-text-response',
            ),
            UserPrompt(
                content='World',
                timestamp=IsNow(tz=timezone.utc),
                role='user',
            ),
            ModelTextResponse(
                content="content='World' role='user' message_count=3",
                timestamp=IsNow(tz=timezone.utc),
                role='model-text-response',
            ),
        ]
    )


async def weather_model(messages: list[Message], info: AgentInfo) -> ModelAnyResponse:  # pragma: no cover
    assert info.allow_text_result
    assert info.retrievers.keys() == {'get_location', 'get_weather'}
    last = messages[-1]
    if last.role == 'user':
        return ModelStructuredResponse(
            calls=[
                ToolCall.from_json(
                    'get_location',
                    json.dumps({'location_description': last.content}),
                )
            ]
        )
    elif last.role == 'tool-return':
        if last.tool_name == 'get_location':
            return ModelStructuredResponse(calls=[ToolCall.from_json('get_weather', last.model_response_str())])
        elif last.tool_name == 'get_weather':
            location_name = next(m.content for m in messages if m.role == 'user')
            return ModelTextResponse(f'{last.content} in {location_name}')

    raise ValueError(f'Unexpected message: {last}')


weather_agent: Agent[None, str] = Agent(FunctionModel(weather_model))


@weather_agent.retriever_plain
async def get_location(location_description: str) -> str:
    if location_description == 'London':
        lat_lng = {'lat': 51, 'lng': 0}
    else:
        lat_lng = {'lat': 0, 'lng': 0}
    return json.dumps(lat_lng)


@weather_agent.retriever
async def get_weather(_: CallContext[None], lat: int, lng: int):
    if (lat, lng) == (51, 0):
        # it always rains in London
        return 'Raining'
    else:
        return 'Sunny'


def test_weather():
    result = weather_agent.run_sync('London')
    assert result.data == 'Raining in London'
    assert result.all_messages() == snapshot(
        [
            UserPrompt(
                content='London',
                timestamp=IsNow(tz=timezone.utc),
                role='user',
            ),
            ModelStructuredResponse(
                calls=[ToolCall.from_json('get_location', '{"location_description": "London"}')],
                timestamp=IsNow(tz=timezone.utc),
                role='model-structured-response',
            ),
            ToolReturn(
                tool_name='get_location',
                content='{"lat": 51, "lng": 0}',
                timestamp=IsNow(tz=timezone.utc),
                role='tool-return',
            ),
            ModelStructuredResponse(
                calls=[
                    ToolCall.from_json(
                        'get_weather',
                        '{"lat": 51, "lng": 0}',
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                role='model-structured-response',
            ),
            ToolReturn(
                tool_name='get_weather',
                content='Raining',
                timestamp=IsNow(tz=timezone.utc),
                role='tool-return',
            ),
            ModelTextResponse(
                content='Raining in London',
                timestamp=IsNow(tz=timezone.utc),
                role='model-text-response',
            ),
        ]
    )

    result = weather_agent.run_sync('Ipswich')
    assert result.data == 'Sunny in Ipswich'


async def call_function_model(messages: list[Message], _: AgentInfo) -> ModelAnyResponse:  # pragma: no cover
    last = messages[-1]
    if last.role == 'user':
        if last.content.startswith('{'):
            details = json.loads(last.content)
            return ModelStructuredResponse(
                calls=[
                    ToolCall.from_json(
                        details['function'],
                        json.dumps(details['arguments']),
                    )
                ]
            )
    elif last.role == 'tool-return':
        return ModelTextResponse(pydantic_core.to_json(last).decode())

    raise ValueError(f'Unexpected message: {last}')


var_args_agent = Agent(FunctionModel(call_function_model), deps_type=int)


@var_args_agent.retriever
def get_var_args(ctx: CallContext[int], *args: int):
    assert ctx.deps == 123
    return json.dumps({'args': args})


def test_var_args():
    result = var_args_agent.run_sync('{"function": "get_var_args", "arguments": {"args": [1, 2, 3]}}', deps=123)
    response_data = json.loads(result.data)
    # Can't parse ISO timestamps with trailing 'Z' in older versions of python:
    response_data['timestamp'] = re.sub('Z$', '+00:00', response_data['timestamp'])
    assert response_data == snapshot(
        {
            'tool_name': 'get_var_args',
            'content': '{"args": [1, 2, 3]}',
            'tool_id': None,
            'timestamp': IsStr() & IsNow(iso_string=True, tz=timezone.utc),
            'role': 'tool-return',
        }
    )


async def call_retriever(messages: list[Message], info: AgentInfo) -> ModelAnyResponse:
    if len(messages) == 1:
        assert len(info.retrievers) == 1
        retriever_id = next(iter(info.retrievers.keys()))
        return ModelStructuredResponse(calls=[ToolCall.from_json(retriever_id, '{}')])
    else:
        return ModelTextResponse('final response')


def test_deps_none():
    agent = Agent(FunctionModel(call_retriever))

    @agent.retriever
    async def get_none(ctx: CallContext[None]):
        nonlocal called

        called = True
        assert ctx.deps is None
        return ''

    called = False
    agent.run_sync('Hello')
    assert called

    called = False
    agent.run_sync('Hello')
    assert called


def test_deps_init():
    def get_check_foobar(ctx: CallContext[tuple[str, str]]) -> str:
        nonlocal called

        called = True
        assert ctx.deps == ('foo', 'bar')
        return ''

    agent = Agent(FunctionModel(call_retriever), deps_type=tuple[str, str])
    agent.retriever(get_check_foobar)
    called = False
    agent.run_sync('Hello', deps=('foo', 'bar'))
    assert called


def test_model_arg():
    agent = Agent()
    result = agent.run_sync('Hello', model=FunctionModel(return_last))
    assert result.data == snapshot("content='Hello' role='user' message_count=1")

    with pytest.raises(RuntimeError, match='`model` must be set either when creating the agent or when calling it.'):
        agent.run_sync('Hello')


agent_all = Agent()


@agent_all.retriever
async def foo(_: CallContext[None], x: int) -> str:
    return str(x + 1)


@agent_all.retriever(retries=3)
def bar(ctx, x: int) -> str:  # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
    return str(x + 2)


@agent_all.retriever_plain
async def baz(x: int) -> str:
    return str(x + 3)


@agent_all.retriever_plain(retries=1)
def qux(x: int) -> str:
    return str(x + 4)


@agent_all.retriever_plain  # pyright: ignore[reportUnknownArgumentType]
def quz(x) -> str:  # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
    return str(x)  # pyright: ignore[reportUnknownArgumentType]


@agent_all.system_prompt
def spam() -> str:
    return 'foobar'


def test_register_all():
    async def f(messages: list[Message], info: AgentInfo) -> ModelAnyResponse:
        return ModelTextResponse(
            f'messages={len(messages)} allow_text_result={info.allow_text_result} retrievers={len(info.retrievers)}'
        )

    result = agent_all.run_sync('Hello', model=FunctionModel(f))
    assert result.data == snapshot('messages=2 allow_text_result=True retrievers=5')


def test_call_all():
    result = agent_all.run_sync('Hello', model=TestModel())
    assert result.data == snapshot('{"foo":"1","bar":"2","baz":"3","qux":"4","quz":"a"}')
    assert result.all_messages() == snapshot(
        [
            SystemPrompt(content='foobar'),
            UserPrompt(content='Hello', timestamp=IsNow(tz=timezone.utc)),
            ModelStructuredResponse(
                calls=[
                    ToolCall.from_object('foo', {'x': 0}),
                    ToolCall.from_object('bar', {'x': 0}),
                    ToolCall.from_object('baz', {'x': 0}),
                    ToolCall.from_object('qux', {'x': 0}),
                    ToolCall.from_object('quz', {'x': 'a'}),
                ],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ToolReturn(tool_name='foo', content='1', timestamp=IsNow(tz=timezone.utc)),
            ToolReturn(tool_name='bar', content='2', timestamp=IsNow(tz=timezone.utc)),
            ToolReturn(tool_name='baz', content='3', timestamp=IsNow(tz=timezone.utc)),
            ToolReturn(tool_name='qux', content='4', timestamp=IsNow(tz=timezone.utc)),
            ToolReturn(tool_name='quz', content='a', timestamp=IsNow(tz=timezone.utc)),
            ModelTextResponse(
                content='{"foo":"1","bar":"2","baz":"3","qux":"4","quz":"a"}', timestamp=IsNow(tz=timezone.utc)
            ),
        ]
    )


def test_retry_str():
    call_count = 0

    async def try_again(messages: list[Message], _: AgentInfo) -> ModelAnyResponse:
        nonlocal call_count
        call_count += 1

        return ModelTextResponse(str(call_count))

    agent = Agent(FunctionModel(try_again))

    @agent.result_validator
    async def validate_result(r: str) -> str:
        if r == '1':
            raise ModelRetry('Try again')
        else:
            return r

    result = agent.run_sync('')
    assert result.data == snapshot('2')


def test_retry_result_type():
    call_count = 0

    async def try_again(messages: list[Message], _: AgentInfo) -> ModelAnyResponse:
        nonlocal call_count
        call_count += 1

        return ModelStructuredResponse(calls=[ToolCall.from_object('final_result', {'x': call_count})])

    class Foo(BaseModel):
        x: int

    agent = Agent(FunctionModel(try_again), result_type=Foo)

    @agent.result_validator
    async def validate_result(r: Foo) -> Foo:
        if r.x == 1:
            raise ModelRetry('Try again')
        else:
            return r

    result = agent.run_sync('')
    assert result.data == snapshot(Foo(x=2))


async def stream_text_function(_messages: list[Message], _: AgentInfo) -> AsyncIterator[str]:
    yield 'hello '
    yield 'world'


async def test_stream_text():
    agent = Agent(FunctionModel(stream_function=stream_text_function))
    async with agent.run_stream('') as result:
        assert await result.get_data() == snapshot('hello world')
        assert result.all_messages() == snapshot(
            [
                UserPrompt(content='', timestamp=IsNow(tz=timezone.utc)),
                ModelTextResponse(content='hello world', timestamp=IsNow(tz=timezone.utc)),
            ]
        )
        assert result.cost() == snapshot(Cost())


class Foo(BaseModel):
    x: int


async def test_stream_structure():
    async def stream_structured_function(
        _messages: list[Message], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls]:
        assert agent_info.result_tools is not None
        assert len(agent_info.result_tools) == 1
        name = agent_info.result_tools[0].name
        yield {0: DeltaToolCall(name=name)}
        yield {0: DeltaToolCall(json_args='{"x": ')}
        yield {0: DeltaToolCall(json_args='1}')}

    agent = Agent(FunctionModel(stream_function=stream_structured_function), result_type=Foo)
    async with agent.run_stream('') as result:
        assert await result.get_data() == snapshot(Foo(x=1))
        assert result.cost() == snapshot(Cost())


async def test_pass_neither():
    with pytest.raises(TypeError, match='Either `function` or `stream_function` must be provided'):
        FunctionModel()  # pyright: ignore[reportCallIssue]


async def test_pass_both():
    Agent(FunctionModel(return_last, stream_function=stream_text_function))


async def stream_text_function_empty(_messages: list[Message], _: AgentInfo) -> AsyncIterator[str]:
    if False:
        yield 'hello '


async def test_return_empty():
    agent = Agent(FunctionModel(stream_function=stream_text_function_empty))
    with pytest.raises(ValueError, match='Stream function must return at least one item'):
        async with agent.run_stream(''):
            pass
