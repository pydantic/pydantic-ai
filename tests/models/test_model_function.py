import json
import re
from collections.abc import AsyncIterator
from dataclasses import asdict
from datetime import timezone

import pydantic_core
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    RunContext,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import RunUsage
from pydantic_ai.usage import RequestUsage

from ..conftest import IsNow, IsStr

pytestmark = pytest.mark.anyio


def hello(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('hello world')])  # pragma: no cover


async def stream_hello(_messages: list[ModelMessage], _agent_info: AgentInfo) -> AsyncIterator[str]:
    yield 'hello '  # pragma: no cover
    yield 'world'  # pragma: no cover


def test_init() -> None:
    m = FunctionModel(function=hello)
    assert m.model_name == 'function:hello:'

    m1 = FunctionModel(stream_function=stream_hello)
    assert m1.model_name == 'function::stream_hello'

    m2 = FunctionModel(function=hello, stream_function=stream_hello)
    assert m2.model_name == 'function:hello:stream_hello'


async def return_last(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
    last = messages[-1].parts[-1]
    response = asdict(last)
    response.pop('timestamp', None)
    response['message_count'] = len(messages)
    return ModelResponse(parts=[TextPart(' '.join(f'{k}={v!r}' for k, v in response.items()))])


def test_simple():
    agent = Agent(FunctionModel(return_last))
    result = agent.run_sync('Hello')
    assert result.output == snapshot("content='Hello' part_kind='user-prompt' message_count=1")
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content="content='Hello' part_kind='user-prompt' message_count=1")],
                usage=RequestUsage(input_tokens=51, output_tokens=3),
                model_name='function:return_last:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )

    result2 = agent.run_sync('World', message_history=result.all_messages())
    assert result2.output == snapshot("content='World' part_kind='user-prompt' message_count=3")
    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content="content='Hello' part_kind='user-prompt' message_count=1")],
                usage=RequestUsage(input_tokens=51, output_tokens=3),
                model_name='function:return_last:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='World', timestamp=IsNow(tz=timezone.utc))],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content="content='World' part_kind='user-prompt' message_count=3")],
                usage=RequestUsage(input_tokens=52, output_tokens=6),
                model_name='function:return_last:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def weather_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: lax no cover
    assert info.allow_text_output
    assert {t.name for t in info.function_tools} == {'get_location', 'get_weather'}
    last = messages[-1].parts[-1]
    if isinstance(last, UserPromptPart):
        return ModelResponse(parts=[ToolCallPart('get_location', json.dumps({'location_description': last.content}))])
    elif isinstance(last, ToolReturnPart):
        if last.tool_name == 'get_location':
            return ModelResponse(parts=[ToolCallPart('get_weather', last.model_response_str())])
        elif last.tool_name == 'get_weather':
            location_name: str | None = None
            for m in messages:
                location_name = next(
                    (
                        item
                        for item in (part.content for part in m.parts if isinstance(part, UserPromptPart))
                        if isinstance(item, str)
                    ),
                    None,
                )
                if location_name is not None:
                    break

            assert location_name is not None
            return ModelResponse(parts=[TextPart(f'{last.content} in {location_name}')])

    raise ValueError(f'Unexpected message: {last}')


weather_agent: Agent[None, str] = Agent(FunctionModel(weather_model))


@weather_agent.tool_plain
async def get_location(location_description: str) -> str:
    if location_description == 'London':
        lat_lng = {'lat': 51, 'lng': 0}
    else:
        lat_lng = {'lat': 0, 'lng': 0}
    return json.dumps(lat_lng)


@weather_agent.tool
async def get_weather(_: RunContext[None], lat: int, lng: int):
    if (lat, lng) == (51, 0):
        # it always rains in London
        return 'Raining'
    else:
        return 'Sunny'


def test_weather():
    result = weather_agent.run_sync('London')
    assert result.output == 'Raining in London'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='London', timestamp=IsNow(tz=timezone.utc))],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location', args='{"location_description": "London"}', tool_call_id=IsStr()
                    )
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=5),
                model_name='function:weather_model:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        timestamp=IsNow(tz=timezone.utc),
                        tool_call_id=IsStr(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_weather', args='{"lat": 51, "lng": 0}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=56, output_tokens=11),
                model_name='function:weather_model:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather',
                        content='Raining',
                        timestamp=IsNow(tz=timezone.utc),
                        tool_call_id=IsStr(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Raining in London')],
                usage=RequestUsage(input_tokens=57, output_tokens=14),
                model_name='function:weather_model:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )

    result = weather_agent.run_sync('Ipswich')
    assert result.output == 'Sunny in Ipswich'


async def call_function_model(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:  # pragma: lax no cover
    last = messages[-1].parts[-1]
    if isinstance(last, UserPromptPart):
        if isinstance(last.content, str) and last.content.startswith('{'):
            details = json.loads(last.content)
            return ModelResponse(parts=[ToolCallPart(details['function'], json.dumps(details['arguments']))])
    elif isinstance(last, ToolReturnPart):
        return ModelResponse(parts=[TextPart(pydantic_core.to_json(last).decode())])

    raise ValueError(f'Unexpected message: {last}')


var_args_agent = Agent(FunctionModel(call_function_model), deps_type=int)


@var_args_agent.tool
def get_var_args(ctx: RunContext[int], *args: int):
    assert ctx.deps == 123
    return json.dumps({'args': args})


def test_var_args():
    result = var_args_agent.run_sync('{"function": "get_var_args", "arguments": {"args": [1, 2, 3]}}', deps=123)
    response_data = json.loads(result.output)
    # Can't parse ISO timestamps with trailing 'Z' in older versions of python:
    response_data['timestamp'] = re.sub('Z$', '+00:00', response_data['timestamp'])
    assert response_data == snapshot(
        {
            'tool_name': 'get_var_args',
            'content': '{"args": [1, 2, 3]}',
            'tool_call_id': IsStr(),
            'metadata': None,
            'timestamp': IsStr() & IsNow(iso_string=True, tz=timezone.utc),  # type: ignore[reportUnknownMemberType]
            'part_kind': 'tool-return',
        }
    )


async def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        assert len(info.function_tools) == 1
        tool_name = info.function_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name, '{}')])
    else:
        return ModelResponse(parts=[TextPart('final response')])


def test_deps_none():
    agent = Agent(FunctionModel(call_tool))

    @agent.tool
    async def get_none(ctx: RunContext[None]):
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
    def get_check_foobar(ctx: RunContext[tuple[str, str]]) -> str:
        nonlocal called

        called = True
        assert ctx.deps == ('foo', 'bar')
        return ''

    agent = Agent(FunctionModel(call_tool), deps_type=tuple[str, str])
    agent.tool(get_check_foobar)
    called = False
    agent.run_sync('Hello', deps=('foo', 'bar'))
    assert called


def test_model_arg():
    agent = Agent()
    result = agent.run_sync('Hello', model=FunctionModel(return_last))
    assert result.output == snapshot("content='Hello' part_kind='user-prompt' message_count=1")

    with pytest.raises(RuntimeError, match='`model` must either be set on the agent or included when calling it.'):
        agent.run_sync('Hello')


agent_all = Agent()


@agent_all.tool
async def foo(_: RunContext[None], x: int) -> str:
    return str(x + 1)


@agent_all.tool(retries=3)
def bar(ctx, x: int) -> str:  # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
    return str(x + 2)


@agent_all.tool_plain
async def baz(x: int) -> str:
    return str(x + 3)


@agent_all.tool_plain(retries=1)
def qux(x: int) -> str:
    return str(x + 4)


@agent_all.tool_plain  # pyright: ignore[reportUnknownArgumentType]
def quz(x) -> str:  # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
    return str(x)  # pyright: ignore[reportUnknownArgumentType]


@agent_all.system_prompt
def spam() -> str:
    return 'foobar'


def test_register_all():
    async def f(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart(
                    f'messages={len(messages)} allow_text_output={info.allow_text_output} tools={len(info.function_tools)}'
                )
            ],
        )

    result = agent_all.run_sync('Hello', model=FunctionModel(f))
    assert result.output == snapshot('messages=1 allow_text_output=True tools=5')


def test_call_all():
    result = agent_all.run_sync('Hello', model=TestModel())
    assert result.output == snapshot('{"foo":"1","bar":"2","baz":"3","qux":"4","quz":"a"}')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='foobar', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='foo', args={'x': 0}, tool_call_id=IsStr()),
                    ToolCallPart(tool_name='bar', args={'x': 0}, tool_call_id=IsStr()),
                    ToolCallPart(tool_name='baz', args={'x': 0}, tool_call_id=IsStr()),
                    ToolCallPart(tool_name='qux', args={'x': 0}, tool_call_id=IsStr()),
                    ToolCallPart(tool_name='quz', args={'x': 'a'}, tool_call_id=IsStr()),
                ],
                usage=RequestUsage(input_tokens=52, output_tokens=21),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='foo', content='1', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                    ),
                    ToolReturnPart(
                        tool_name='bar', content='2', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                    ),
                    ToolReturnPart(
                        tool_name='baz', content='3', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                    ),
                    ToolReturnPart(
                        tool_name='qux', content='4', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                    ),
                    ToolReturnPart(
                        tool_name='quz', content='a', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                    ),
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"foo":"1","bar":"2","baz":"3","qux":"4","quz":"a"}')],
                usage=RequestUsage(input_tokens=57, output_tokens=33),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


def test_retry_str():
    call_count = 0

    async def try_again(msgs_: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1

        return ModelResponse(parts=[TextPart(str(call_count))])

    agent = Agent(FunctionModel(try_again))

    @agent.output_validator
    async def validate_output(o: str) -> str:
        if o == '1':
            raise ModelRetry('Try again')
        else:
            return o

    result = agent.run_sync('')
    assert result.output == snapshot('2')


def test_retry_result_type():
    call_count = 0

    async def try_again(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1

        return ModelResponse(parts=[ToolCallPart('final_result', {'x': call_count})])

    class Foo(BaseModel):
        x: int

    agent = Agent(FunctionModel(try_again), output_type=Foo)

    @agent.output_validator
    async def validate_output(o: Foo) -> Foo:
        if o.x == 1:
            raise ModelRetry('Try again')
        else:
            return o

    result = agent.run_sync('')
    assert result.output == snapshot(Foo(x=2))


async def stream_text_function(_messages: list[ModelMessage], _: AgentInfo) -> AsyncIterator[str]:
    yield 'hello '
    yield 'world'


async def test_stream_text():
    agent = Agent(FunctionModel(stream_function=stream_text_function))
    async with agent.run_stream('') as result:
        assert await result.get_output() == snapshot('hello world')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='', timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='hello world')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function::stream_text_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
            ]
        )
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=50, output_tokens=2))


class Foo(BaseModel):
    x: int


async def test_stream_structure():
    async def stream_structured_function(
        _messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls]:
        assert agent_info.output_tools is not None
        assert len(agent_info.output_tools) == 1
        name = agent_info.output_tools[0].name
        # Args don't typically come before the tool name, but it's technically possible and this ensures test coverage
        yield {0: DeltaToolCall(json_args='{"x": ')}
        yield {0: DeltaToolCall(name=name)}
        yield {0: DeltaToolCall(json_args='1}')}

    agent = Agent(FunctionModel(stream_function=stream_structured_function), output_type=Foo)
    async with agent.run_stream('') as result:
        assert await result.get_output() == snapshot(Foo(x=1))
        assert result.usage() == snapshot(
            RunUsage(
                requests=1,
                input_tokens=50,
                output_tokens=4,
            )
        )


async def test_pass_neither():
    with pytest.raises(TypeError, match='Either `function` or `stream_function` must be provided'):
        FunctionModel()  # pyright: ignore[reportCallIssue]


async def test_pass_both():
    Agent(FunctionModel(return_last, stream_function=stream_text_function))


async def stream_text_function_empty(_messages: list[ModelMessage], _: AgentInfo) -> AsyncIterator[str]:
    if False:
        yield 'hello '


async def test_return_empty():
    agent = Agent(FunctionModel(stream_function=stream_text_function_empty))
    with pytest.raises(ValueError, match='Stream function must return at least one item'):
        async with agent.run_stream(''):
            pass
