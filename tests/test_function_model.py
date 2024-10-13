import json
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pydantic_core
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, CallContext
from pydantic_ai.messages import (
    LLMMessage,
    LLMResponse,
    LLMToolCalls,
    Message,
    SystemPrompt,
    ToolCall,
    ToolReturn,
    UserPrompt,
)
from pydantic_ai.models.function import FunctionModel, ToolDescription
from pydantic_ai.models.test import TestModel

if TYPE_CHECKING:

    def IsNow(*args: Any, **kwargs: Any) -> datetime: ...
else:
    from dirty_equals import IsNow


def return_last(
    messages: list[Message], _allow_text_result: bool, _retrievers: dict[str, ToolDescription]
) -> LLMMessage:
    last = messages[-1]
    response = asdict(last)
    response.pop('timestamp', None)
    response['message_count'] = len(messages)
    return LLMResponse(' '.join(f'{k}={v!r}' for k, v in response.items()))


def test_simple():
    agent = Agent(FunctionModel(return_last), deps=None)
    result = agent.run_sync('Hello')
    assert result.response == snapshot("content='Hello' role='user' message_count=1")
    assert result.message_history == snapshot(
        [
            UserPrompt(
                content='Hello',
                timestamp=IsNow(),
                role='user',
            ),
            LLMResponse(
                content="content='Hello' role='user' message_count=1",
                timestamp=IsNow(),
                role='llm-response',
            ),
        ]
    )

    result2 = agent.run_sync('World', message_history=result.message_history)
    assert result2.response == snapshot("content='World' role='user' message_count=3")
    assert result2.message_history == snapshot(
        [
            UserPrompt(
                content='Hello',
                timestamp=IsNow(),
                role='user',
            ),
            LLMResponse(
                content="content='Hello' role='user' message_count=1",
                timestamp=IsNow(),
                role='llm-response',
            ),
            UserPrompt(
                content='World',
                timestamp=IsNow(),
                role='user',
            ),
            LLMResponse(
                content="content='World' role='user' message_count=3",
                timestamp=IsNow(),
                role='llm-response',
            ),
        ]
    )


def whether_model(
    messages: list[Message], allow_text_result: bool, retrievers: dict[str, ToolDescription]
) -> LLMMessage:  # pragma: no cover
    assert allow_text_result
    assert retrievers.keys() == {'get_location', 'get_whether'}
    last = messages[-1]
    if last.role == 'user':
        return LLMToolCalls(
            calls=[
                ToolCall(
                    tool_name='get_location',
                    arguments=json.dumps({'location_description': last.content}),
                )
            ]
        )
    elif last.role == 'tool-return':
        if last.tool_name == 'get_location':
            return LLMToolCalls(calls=[ToolCall(tool_name='get_whether', arguments=last.content)])
        elif last.tool_name == 'get_whether':
            location_name = next(m.content for m in messages if m.role == 'user')
            return LLMResponse(f'{last.content} in {location_name}')

    raise ValueError(f'Unexpected message: {last}')


weather_agent: Agent[None, str] = Agent(FunctionModel(whether_model))


@weather_agent.retriever_plain
async def get_location(location_description: str) -> str:
    if location_description == 'London':
        lat_lng = {'lat': 51, 'lng': 0}
    else:
        lat_lng = {'lat': 0, 'lng': 0}
    return json.dumps(lat_lng)


@weather_agent.retriever_context
async def get_whether(_: CallContext[None], lat: int, lng: int):
    if (lat, lng) == (51, 0):
        # it always rains in London
        return 'Raining'
    else:
        return 'Sunny'


def test_whether():
    result = weather_agent.run_sync('London')
    assert result.response == 'Raining in London'
    assert result.message_history == snapshot(
        [
            UserPrompt(
                content='London',
                timestamp=IsNow(),
                role='user',
            ),
            LLMToolCalls(
                calls=[ToolCall(tool_name='get_location', arguments='{"location_description": "London"}')],
                timestamp=IsNow(),
                role='llm-tool-calls',
            ),
            ToolReturn(
                tool_name='get_location', content='{"lat": 51, "lng": 0}', timestamp=IsNow(), role='tool-return'
            ),
            LLMToolCalls(
                calls=[
                    ToolCall(
                        tool_name='get_whether',
                        arguments='{"lat": 51, "lng": 0}',
                    )
                ],
                timestamp=IsNow(),
                role='llm-tool-calls',
            ),
            ToolReturn(
                tool_name='get_whether',
                content='Raining',
                timestamp=IsNow(),
                role='tool-return',
            ),
            LLMResponse(
                content='Raining in London',
                timestamp=IsNow(),
                role='llm-response',
            ),
        ]
    )

    result = weather_agent.run_sync('Ipswich')
    assert result.response == 'Sunny in Ipswich'


def call_function_model(
    messages: list[Message], _allow_text_result: bool, _tools: dict[str, ToolDescription]
) -> LLMMessage:  # pragma: no cover
    last = messages[-1]
    if last.role == 'user':
        if last.content.startswith('{'):
            details = json.loads(last.content)
            return LLMToolCalls(
                calls=[
                    ToolCall(
                        tool_name=details['function'],
                        arguments=json.dumps(details['arguments']),
                    )
                ]
            )
    elif last.role == 'tool-return':
        return LLMResponse(pydantic_core.to_json(last).decode())

    raise ValueError(f'Unexpected message: {last}')


var_args_agent = Agent(FunctionModel(call_function_model), deps=123)


@var_args_agent.retriever_context
def get_var_args(ctx: CallContext[int], *args: int):
    assert ctx.deps == 123
    return json.dumps({'args': args})


def test_var_args():
    result = var_args_agent.run_sync('{"function": "get_var_args", "arguments": {"args": [1, 2, 3]}}')
    response_data = json.loads(result.response)
    assert response_data == snapshot(
        {
            'tool_name': 'get_var_args',
            'content': '{"args": [1, 2, 3]}',
            'tool_id': None,
            'timestamp': IsNow(iso_string=True),
            'role': 'tool-return',
        }
    )


def call_retriever(
    messages: list[Message], _allow_text_result: bool, retrievers: dict[str, ToolDescription]
) -> LLMMessage:
    if len(messages) == 1:
        assert len(retrievers) == 1
        retriever_id = next(iter(retrievers.keys()))
        return LLMToolCalls(calls=[ToolCall(tool_name=retriever_id, arguments='{}')])
    else:
        return LLMResponse('final response')


def test_deps_none():
    agent = Agent(FunctionModel(call_retriever), deps=None)

    @agent.retriever_context
    async def get_none(ctx: CallContext[None]):  # pyright: ignore[reportUnusedFunction]
        nonlocal called

        called = True
        assert ctx.deps is None
        return ''

    called = False
    agent.run_sync('Hello')
    assert called

    called = False
    agent.run_sync('Hello', deps=None)
    assert called


def test_deps_init():
    def get_check_foobar(ctx: CallContext[tuple[str, str]]) -> str:
        nonlocal called

        called = True
        assert ctx.deps == ('foo', 'bar')
        return ''

    agent = Agent(FunctionModel(call_retriever), deps=('foo', 'bar'))
    agent.retriever_context(get_check_foobar)
    called = False
    agent.run_sync('Hello')
    assert called

    agent: Agent[tuple[str, str], str] = Agent(FunctionModel(call_retriever))
    agent.retriever_context(get_check_foobar)
    called = False
    agent.run_sync('Hello', deps=('foo', 'bar'))
    assert called


def test_result_schema_tuple():
    def return_tuple(_: list[Message], __: bool, retrievers: dict[str, ToolDescription]) -> LLMMessage:
        assert len(retrievers) == 1
        retriever_key = next(iter(retrievers.keys()))
        tuple_json = '{"response": ["foo", "bar"]}'
        return LLMToolCalls(calls=[ToolCall(tool_name=retriever_key, arguments=tuple_json)])

    agent = Agent(FunctionModel(return_tuple), deps=None, result_type=tuple[str, str])

    result = agent.run_sync('Hello')
    assert result.response == ('foo', 'bar')


def test_result_schema_pydantic_model():
    class Foo(BaseModel):
        a: int
        b: str

    def return_tuple(_: list[Message], __: bool, retrievers: dict[str, ToolDescription]) -> LLMMessage:
        assert len(retrievers) == 1
        retriever_key = next(iter(retrievers.keys()))
        tuple_json = '{"a": 1, "b": "foo"}'
        return LLMToolCalls(calls=[ToolCall(tool_name=retriever_key, arguments=tuple_json)])

    agent = Agent(FunctionModel(return_tuple), deps=None, result_type=Foo)

    result = agent.run_sync('Hello')
    assert isinstance(result.response, Foo)
    assert result.response.model_dump() == {'a': 1, 'b': 'foo'}


def test_model_arg():
    agent = Agent(deps=None)
    result = agent.run_sync('Hello', model=FunctionModel(return_last))
    assert result.response == snapshot("content='Hello' role='user' message_count=1")

    with pytest.raises(RuntimeError, match='`model` must be set either when creating the agent or when calling it.'):
        agent.run_sync('Hello')


agent_all = Agent(deps=None)


@agent_all.retriever_context
async def foo(_: CallContext[None], x: int) -> str:
    return str(x + 1)


@agent_all.retriever_context(retries=3)
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
    def f(messages: list[Message], allow_text_result: bool, retrievers: dict[str, ToolDescription]) -> LLMMessage:
        return LLMResponse(
            f'messages={len(messages)} allow_text_result={allow_text_result} retrievers={len(retrievers)}'
        )

    result = agent_all.run_sync('Hello', model=FunctionModel(f))
    assert result.response == snapshot('messages=2 allow_text_result=True retrievers=5')


def test_call_all():
    result = agent_all.run_sync('Hello', model=TestModel())
    assert result.response == snapshot('Final response')
    assert result.message_history == snapshot(
        [
            SystemPrompt(content='foobar'),
            UserPrompt(content='Hello', timestamp=IsNow()),
            LLMToolCalls(
                calls=[
                    ToolCall(tool_name='foo', arguments='{"x": 0}'),
                    ToolCall(tool_name='bar', arguments='{"x": 0}'),
                    ToolCall(tool_name='baz', arguments='{"x": 0}'),
                    ToolCall(tool_name='qux', arguments='{"x": 0}'),
                    ToolCall(tool_name='quz', arguments='{"x": "a"}'),
                ],
                timestamp=IsNow(),
            ),
            ToolReturn(tool_name='foo', content='1', timestamp=IsNow()),
            ToolReturn(tool_name='bar', content='2', timestamp=IsNow()),
            ToolReturn(tool_name='baz', content='3', timestamp=IsNow()),
            ToolReturn(tool_name='qux', content='4', timestamp=IsNow()),
            ToolReturn(tool_name='quz', content='a', timestamp=IsNow()),
            LLMResponse(content='Final response', timestamp=IsNow()),
        ]
    )


async def do_foobar(foo: int, bar: str) -> str:
    """
    Do foobar stuff, a lot.

    Args:
        foo: The foo thing.
        bar: The bar thing.
    """
    return f'{foo} {bar}'


def test_docstring():
    def f(_messages: list[Message], _allow_text_result: bool, retrievers: dict[str, ToolDescription]) -> LLMMessage:
        assert len(retrievers) == 1
        r = next(iter(retrievers.values()))
        return LLMResponse(json.dumps(r.json_schema))

    agent = Agent(FunctionModel(f), deps=None)
    agent.retriever_plain(do_foobar)

    result = agent.run_sync('Hello')
    json_schema = json.loads(result.response)
    assert json_schema == snapshot(
        {
            'description': 'Do foobar stuff, a lot.',
            'additionalProperties': False,
            'properties': {
                'foo': {'description': 'The foo thing.', 'title': 'Foo', 'type': 'integer'},
                'bar': {'description': 'The bar thing.', 'title': 'Bar', 'type': 'string'},
            },
            'required': ['foo', 'bar'],
            'type': 'object',
        }
    )
    # description should be the first key
    assert next(iter(json_schema)) == 'description'
