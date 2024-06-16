import json
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pydantic_core
from inline_snapshot import snapshot

from pydantic_ai import Agent, CallInfo
from pydantic_ai.messages import (
    FunctionCall,
    FunctionResponse,
    LLMFunctionCalls,
    LLMMessage,
    LLMResponse,
    Message,
    UserPrompt,
)
from pydantic_ai.models.function import FunctionModel, Tool

if TYPE_CHECKING:

    def IsNow(*args: Any, **kwargs: Any) -> datetime: ...
else:
    from dirty_equals import IsNow


def return_last(messages: list[Message], _allow_plain_message: bool, _tools: dict[str, Tool]) -> LLMMessage:
    last = messages[-1]
    response = asdict(last)
    response.pop('timestamp', None)
    response['message_count'] = len(messages)
    return LLMResponse(' '.join(f'{k}={v!r}' for k, v in response.items()))


def test_simple():
    agent = Agent(FunctionModel(return_last))
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


def whether_model(messages: list[Message], allow_plain_message: bool, tools: dict[str, Tool]) -> LLMMessage:
    assert allow_plain_message
    assert tools.keys() == {'get_location', 'get_whether'}
    last = messages[-1]
    if last.role == 'user':
        return LLMFunctionCalls(
            calls=[
                FunctionCall(
                    function_id='1',
                    function_name='get_location',
                    arguments=json.dumps({'location_description': last.content}),
                )
            ]
        )
    elif last.role == 'function-response':
        if last.function_name == 'get_location':
            return LLMFunctionCalls(
                calls=[FunctionCall(function_id='2', function_name='get_whether', arguments=last.content)]
            )
        elif last.function_name == 'get_whether':
            location_name = next(m.content for m in messages if m.role == 'user')
            return LLMResponse(f'{last.content} in {location_name}')

    raise ValueError(f'Unexpected message: {last}')


whether_agent = Agent(FunctionModel(whether_model))


@whether_agent.retriever
async def get_location(_: CallInfo[None], location_description: str) -> str:
    if location_description == 'London':
        lat_lng = {'lat': 51, 'lng': 0}
    else:
        lat_lng = {'lat': 0, 'lng': 0}
    return json.dumps(lat_lng)


@whether_agent.retriever
async def get_whether(_: CallInfo[None], lat: int, lng: int):
    if (lat, lng) == (51, 0):
        # it always rains in London
        return 'Raining'
    else:
        return 'Sunny'


def test_whether():
    result = whether_agent.run_sync('London')
    assert result.response == 'Raining in London'
    assert result.message_history == snapshot(
        [
            UserPrompt(
                content='London',
                timestamp=IsNow(),
                role='user',
            ),
            LLMFunctionCalls(
                calls=[
                    FunctionCall(
                        function_id='1',
                        function_name='get_location',
                        arguments='{"location_description": "London"}',
                    )
                ],
                timestamp=IsNow(),
                role='llm-function-calls',
            ),
            FunctionResponse(
                function_id='1',
                function_name='get_location',
                content='{"lat": 51, "lng": 0}',
                timestamp=IsNow(),
                role='function-response',
            ),
            LLMFunctionCalls(
                calls=[
                    FunctionCall(
                        function_id='2',
                        function_name='get_whether',
                        arguments='{"lat": 51, "lng": 0}',
                    )
                ],
                timestamp=IsNow(),
                role='llm-function-calls',
            ),
            FunctionResponse(
                function_id='2',
                function_name='get_whether',
                content='Raining',
                timestamp=IsNow(),
                role='function-response',
            ),
            LLMResponse(
                content='Raining in London',
                timestamp=IsNow(),
                role='llm-response',
            ),
        ]
    )

    result = whether_agent.run_sync('Ipswich')
    assert result.response == 'Sunny in Ipswich'


def call_function_model(messages: list[Message], allow_plain_message: bool, tools: dict[str, Tool]) -> LLMMessage:
    last = messages[-1]
    if last.role == 'user':
        if last.content.startswith('{'):
            details = json.loads(last.content)
            return LLMFunctionCalls(
                calls=[
                    FunctionCall(
                        function_id='1',
                        function_name=details['function'],
                        arguments=json.dumps(details['arguments']),
                    )
                ]
            )
    elif last.role == 'function-response':
        return LLMResponse(pydantic_core.to_json(last).decode())

    raise ValueError(f'Unexpected message: {last}')


var_args_agent = Agent(FunctionModel(call_function_model))


@var_args_agent.retriever
def get_var_args(_: CallInfo[None], *args: int):
    return json.dumps({'args': args})


def test_var_args():
    result = var_args_agent.run_sync('{"function": "get_var_args", "arguments": {"args": [1, 2, 3]}}')
    response_data = json.loads(result.response)
    assert response_data == snapshot(
        {
            'function_id': '1',
            'function_name': 'get_var_args',
            'content': '{"args": [1, 2, 3]}',
            'timestamp': IsNow(iso_string=True),
            'role': 'function-response',
        }
    )
