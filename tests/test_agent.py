import json
import re
import sys
from datetime import timezone
from typing import Any, Callable, Union

import httpx
import pytest
from dirty_equals import IsJson
from inline_snapshot import snapshot
from pydantic import BaseModel, field_validator
from pydantic_core import to_json

from pydantic_ai import Agent, ModelRetry, RunContext, UnexpectedModelBehavior, UserError, capture_run_messages
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import Usage
from pydantic_ai.tools import ToolDefinition

from .conftest import IsNow, IsStr, TestEnv

pytestmark = pytest.mark.anyio


def test_result_tuple():
    def return_tuple(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.result_tools is not None
        args_json = '{"response": ["foo", "bar"]}'
        return ModelResponse(parts=[ToolCallPart(info.result_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_tuple), result_type=tuple[str, str])

    result = agent.run_sync('Hello')
    assert result.data == ('foo', 'bar')


class Foo(BaseModel):
    a: int
    b: str


def test_result_pydantic_model():
    def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.result_tools is not None
        args_json = '{"a": 1, "b": "foo"}'
        return ModelResponse(parts=[ToolCallPart(info.result_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_model), result_type=Foo)

    result = agent.run_sync('Hello')
    assert isinstance(result.data, Foo)
    assert result.data.model_dump() == {'a': 1, 'b': 'foo'}


def test_result_pydantic_model_retry():
    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.result_tools is not None
        if len(messages) == 1:
            args_json = '{"a": "wrong", "b": "foo"}'
        else:
            args_json = '{"a": 42, "b": "foo"}'
        return ModelResponse(parts=[ToolCallPart(info.result_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_model), result_type=Foo)

    assert agent.name is None

    result = agent.run_sync('Hello')
    assert agent.name == 'agent'
    assert isinstance(result.data, Foo)
    assert result.data.model_dump() == {'a': 42, 'b': 'foo'}
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args='{"a": "wrong", "b": "foo"}')],
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        tool_name='final_result',
                        content=[
                            {
                                'type': 'int_parsing',
                                'loc': ('a',),
                                'msg': 'Input should be a valid integer, unable to parse string as an integer',
                                'input': 'wrong',
                            }
                        ],
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args='{"a": 42, "b": "foo"}')],
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result', content='Final result processed.', timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
        ]
    )
    assert result.all_messages_json().startswith(b'[{"parts":[{"content":"Hello",')


def test_result_pydantic_model_validation_error():
    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.result_tools is not None
        if len(messages) == 1:
            args_json = '{"a": 1, "b": "foo"}'
        else:
            args_json = '{"a": 1, "b": "bar"}'
        return ModelResponse(parts=[ToolCallPart(info.result_tools[0].name, args_json)])

    class Bar(BaseModel):
        a: int
        b: str

        @field_validator('b')
        def check_b(cls, v: str) -> str:
            if v == 'foo':
                raise ValueError('must not be foo')
            return v

    agent = Agent(FunctionModel(return_model), result_type=Bar)

    result = agent.run_sync('Hello')
    assert isinstance(result.data, Bar)
    assert result.data.model_dump() == snapshot({'a': 1, 'b': 'bar'})
    messages_part_kinds = [(m.kind, [p.part_kind for p in m.parts]) for m in result.all_messages()]
    assert messages_part_kinds == snapshot(
        [
            ('request', ['user-prompt']),
            ('response', ['tool-call']),
            ('request', ['retry-prompt']),
            ('response', ['tool-call']),
            ('request', ['tool-return']),
        ]
    )

    user_retry = result.all_messages()[2]
    assert isinstance(user_retry, ModelRequest)
    retry_prompt = user_retry.parts[0]
    assert isinstance(retry_prompt, RetryPromptPart)
    assert retry_prompt.model_response() == snapshot("""\
1 validation errors: [
  {
    "type": "value_error",
    "loc": [
      "b"
    ],
    "msg": "Value error, must not be foo",
    "input": "foo"
  }
]

Fix the errors and try again.""")


def test_result_validator():
    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.result_tools is not None
        if len(messages) == 1:
            args_json = '{"a": 41, "b": "foo"}'
        else:
            args_json = '{"a": 42, "b": "foo"}'
        return ModelResponse(parts=[ToolCallPart(info.result_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_model), result_type=Foo)

    @agent.result_validator
    def validate_result(ctx: RunContext[None], r: Foo) -> Foo:
        assert ctx.tool_name == 'final_result'
        if r.a == 42:
            return r
        else:
            raise ModelRetry('"a" should be 42')

    result = agent.run_sync('Hello')
    assert isinstance(result.data, Foo)
    assert result.data.model_dump() == {'a': 42, 'b': 'foo'}
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args='{"a": 41, "b": "foo"}')],
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='"a" should be 42', tool_name='final_result', timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args='{"a": 42, "b": "foo"}')],
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result', content='Final result processed.', timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
        ]
    )


def test_plain_response_then_tuple():
    call_index = 0

    def return_tuple(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_index

        assert info.result_tools is not None
        call_index += 1
        if call_index == 1:
            return ModelResponse(parts=[TextPart('hello')])
        else:
            args_json = '{"response": ["foo", "bar"]}'
            return ModelResponse(parts=[ToolCallPart(info.result_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_tuple), result_type=tuple[str, str])

    result = agent.run_sync('Hello')
    assert result.data == ('foo', 'bar')
    assert call_index == 2
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='hello')],
                model_name='function:return_tuple:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Plain text responses are not permitted, please call one of the functions instead.',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args='{"response": ["foo", "bar"]}')],
                model_name='function:return_tuple:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result', content='Final result processed.', timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
        ]
    )
    assert result._result_tool_name == 'final_result'  # pyright: ignore[reportPrivateUsage]
    assert result.all_messages(result_tool_return_content='foobar')[-1] == snapshot(
        ModelRequest(
            parts=[ToolReturnPart(tool_name='final_result', content='foobar', timestamp=IsNow(tz=timezone.utc))]
        )
    )
    assert result.all_messages()[-1] == snapshot(
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='final_result', content='Final result processed.', timestamp=IsNow(tz=timezone.utc)
                )
            ]
        )
    )


def test_result_tool_return_content_str_return():
    agent = Agent('test')

    result = agent.run_sync('Hello')
    assert result.data == 'success (no tool calls)'

    msg = re.escape('Cannot set result tool return content when the return type is `str`.')
    with pytest.raises(ValueError, match=msg):
        result.all_messages(result_tool_return_content='foobar')


def test_result_tool_return_content_no_tool():
    agent = Agent('test', result_type=int)

    result = agent.run_sync('Hello')
    assert result.data == 0
    result._result_tool_name = 'wrong'  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(LookupError, match=re.escape("No tool call found with tool name 'wrong'.")):
        result.all_messages(result_tool_return_content='foobar')


def test_response_tuple():
    m = TestModel()

    agent = Agent(m, result_type=tuple[str, str])
    assert agent._result_schema.allow_text_result is False  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]

    result = agent.run_sync('Hello')
    assert result.data == snapshot(('a', 'a'))

    assert m.last_model_request_parameters is not None
    assert m.last_model_request_parameters.function_tools == snapshot([])
    assert m.last_model_request_parameters.allow_text_result is False

    assert m.last_model_request_parameters.result_tools is not None
    assert len(m.last_model_request_parameters.result_tools) == 1
    assert m.last_model_request_parameters.result_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'properties': {
                        'response': {
                            'maxItems': 2,
                            'minItems': 2,
                            'prefixItems': [{'type': 'string'}, {'type': 'string'}],
                            'title': 'Response',
                            'type': 'array',
                        }
                    },
                    'required': ['response'],
                    'type': 'object',
                },
                outer_typed_dict_key='response',
            )
        ]
    )


@pytest.mark.parametrize(
    'input_union_callable',
    [lambda: Union[str, Foo], lambda: Union[Foo, str], lambda: str | Foo, lambda: Foo | str],
    ids=['Union[str, Foo]', 'Union[Foo, str]', 'str | Foo', 'Foo | str'],
)
def test_response_union_allow_str(input_union_callable: Callable[[], Any]):
    try:
        union = input_union_callable()
    except TypeError:
        pytest.skip('Python version does not support `|` syntax for unions')

    m = TestModel()
    agent: Agent[None, Union[str, Foo]] = Agent(m, result_type=union)

    got_tool_call_name = 'unset'

    @agent.result_validator
    def validate_result(ctx: RunContext[None], r: Any) -> Any:
        nonlocal got_tool_call_name
        got_tool_call_name = ctx.tool_name
        return r

    assert agent._result_schema.allow_text_result is True  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]

    result = agent.run_sync('Hello')
    assert result.data == snapshot('success (no tool calls)')
    assert got_tool_call_name == snapshot(None)

    assert m.last_model_request_parameters is not None
    assert m.last_model_request_parameters.function_tools == snapshot([])
    assert m.last_model_request_parameters.allow_text_result is True

    assert m.last_model_request_parameters.result_tools is not None
    assert len(m.last_model_request_parameters.result_tools) == 1

    assert m.last_model_request_parameters.result_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'properties': {
                        'a': {'title': 'A', 'type': 'integer'},
                        'b': {'title': 'B', 'type': 'string'},
                    },
                    'required': ['a', 'b'],
                    'title': 'Foo',
                    'type': 'object',
                },
            )
        ]
    )


# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
@pytest.mark.parametrize(
    'union_code',
    [
        pytest.param('ResultType = Union[Foo, Bar]'),
        pytest.param('ResultType = Foo | Bar', marks=pytest.mark.skipif(sys.version_info < (3, 10), reason='3.10+')),
        pytest.param(
            'ResultType: TypeAlias = Foo | Bar',
            marks=pytest.mark.skipif(sys.version_info < (3, 10), reason='Python 3.10+'),
        ),
        pytest.param(
            'type ResultType = Foo | Bar', marks=pytest.mark.skipif(sys.version_info < (3, 12), reason='3.12+')
        ),
    ],
)
def test_response_multiple_return_tools(create_module: Callable[[str], Any], union_code: str):
    module_code = f'''
from pydantic import BaseModel
from typing import Union
from typing_extensions import TypeAlias

class Foo(BaseModel):
    a: int
    b: str


class Bar(BaseModel):
    """This is a bar model."""

    b: str

{union_code}
    '''

    mod = create_module(module_code)

    m = TestModel()
    agent = Agent(m, result_type=mod.ResultType)
    got_tool_call_name = 'unset'

    @agent.result_validator
    def validate_result(ctx: RunContext[None], r: Any) -> Any:
        nonlocal got_tool_call_name
        got_tool_call_name = ctx.tool_name
        return r

    result = agent.run_sync('Hello')
    assert result.data == mod.Foo(a=0, b='a')
    assert got_tool_call_name == snapshot('final_result_Foo')

    assert m.last_model_request_parameters is not None
    assert m.last_model_request_parameters.function_tools == snapshot([])
    assert m.last_model_request_parameters.allow_text_result is False

    assert m.last_model_request_parameters.result_tools is not None
    assert len(m.last_model_request_parameters.result_tools) == 2

    assert m.last_model_request_parameters.result_tools == snapshot(
        [
            ToolDefinition(
                name='final_result_Foo',
                description='Foo: The final response which ends this conversation',
                parameters_json_schema={
                    'properties': {
                        'a': {'title': 'A', 'type': 'integer'},
                        'b': {'title': 'B', 'type': 'string'},
                    },
                    'required': ['a', 'b'],
                    'title': 'Foo',
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='final_result_Bar',
                description='This is a bar model.',
                parameters_json_schema={
                    'properties': {'b': {'title': 'B', 'type': 'string'}},
                    'required': ['b'],
                    'title': 'Bar',
                    'type': 'object',
                },
            ),
        ]
    )

    result = agent.run_sync('Hello', model=TestModel(seed=1))
    assert result.data == mod.Bar(b='b')
    assert got_tool_call_name == snapshot('final_result_Bar')


def test_run_with_history_new():
    m = TestModel()

    agent = Agent(m, system_prompt='Foobar')

    @agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    result1 = agent.run_sync('Hello')
    assert result1.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar'),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'})],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc))]
            ),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')], model_name='test', timestamp=IsNow(tz=timezone.utc)
            ),
        ]
    )

    # if we pass new_messages, system prompt is inserted before the message_history messages
    result2 = agent.run_sync('Hello again', message_history=result1.new_messages())
    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar'),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'})],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc))]
            ),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')], model_name='test', timestamp=IsNow(tz=timezone.utc)
            ),
            ModelRequest(parts=[UserPromptPart(content='Hello again', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')], model_name='test', timestamp=IsNow(tz=timezone.utc)
            ),
        ]
    )
    assert result2._new_message_index == snapshot(4)  # pyright: ignore[reportPrivateUsage]
    assert result2.data == snapshot('{"ret_a":"a-apple"}')
    assert result2._result_tool_name == snapshot(None)  # pyright: ignore[reportPrivateUsage]
    assert result2.usage() == snapshot(
        Usage(requests=1, request_tokens=55, response_tokens=13, total_tokens=68, details=None)
    )
    new_msg_part_kinds = [(m.kind, [p.part_kind for p in m.parts]) for m in result2.all_messages()]
    assert new_msg_part_kinds == snapshot(
        [
            ('request', ['system-prompt', 'user-prompt']),
            ('response', ['tool-call']),
            ('request', ['tool-return']),
            ('response', ['text']),
            ('request', ['user-prompt']),
            ('response', ['text']),
        ]
    )
    assert result2.new_messages_json().startswith(b'[{"parts":[{"content":"Hello again",')

    # if we pass all_messages, system prompt is NOT inserted before the message_history messages,
    # so only one system prompt
    result3 = agent.run_sync('Hello again', message_history=result1.all_messages())
    # same as result2 except for datetimes
    assert result3.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar'),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'})],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc))]
            ),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')], model_name='test', timestamp=IsNow(tz=timezone.utc)
            ),
            ModelRequest(parts=[UserPromptPart(content='Hello again', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')], model_name='test', timestamp=IsNow(tz=timezone.utc)
            ),
        ]
    )
    assert result3._new_message_index == snapshot(4)  # pyright: ignore[reportPrivateUsage]
    assert result3.data == snapshot('{"ret_a":"a-apple"}')
    assert result3._result_tool_name == snapshot(None)  # pyright: ignore[reportPrivateUsage]
    assert result3.usage() == snapshot(
        Usage(requests=1, request_tokens=55, response_tokens=13, total_tokens=68, details=None)
    )


def test_run_with_history_new_structured():
    m = TestModel()

    class Response(BaseModel):
        a: int

    agent = Agent(m, system_prompt='Foobar', result_type=Response)

    @agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    result1 = agent.run_sync('Hello')
    assert result1.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar'),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'})],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc))]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'a': 0},
                        tool_call_id=None,
                    )
                ],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result', content='Final result processed.', timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
        ]
    )

    result2 = agent.run_sync('Hello again', message_history=result1.new_messages())
    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar'),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'})],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc))],
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args={'a': 0})],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                ],
            ),
            # second call, notice no repeated system prompt
            ModelRequest(
                parts=[
                    UserPromptPart(content='Hello again', timestamp=IsNow(tz=timezone.utc)),
                ],
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args={'a': 0})],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                ]
            ),
        ]
    )
    assert result2.data == snapshot(Response(a=0))
    assert result2._new_message_index == snapshot(5)  # pyright: ignore[reportPrivateUsage]
    assert result2._result_tool_name == snapshot('final_result')  # pyright: ignore[reportPrivateUsage]
    assert result2.usage() == snapshot(
        Usage(requests=1, request_tokens=59, response_tokens=13, total_tokens=72, details=None)
    )
    new_msg_part_kinds = [(m.kind, [p.part_kind for p in m.parts]) for m in result2.all_messages()]
    assert new_msg_part_kinds == snapshot(
        [
            ('request', ['system-prompt', 'user-prompt']),
            ('response', ['tool-call']),
            ('request', ['tool-return']),
            ('response', ['tool-call']),
            ('request', ['tool-return']),
            ('request', ['user-prompt']),
            ('response', ['tool-call']),
            ('request', ['tool-return']),
        ]
    )
    assert result2.new_messages_json().startswith(b'[{"parts":[{"content":"Hello again",')


def test_empty_tool_calls():
    def empty(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[])

    agent = Agent(FunctionModel(empty))

    with pytest.raises(UnexpectedModelBehavior, match='Received empty model response'):
        agent.run_sync('Hello')


def test_unknown_tool():
    def empty(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('foobar', '{}')])

    agent = Agent(FunctionModel(empty))

    with capture_run_messages() as messages:
        with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum retries \(1\) for result validation'):
            agent.run_sync('Hello')
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args='{}')],
                model_name='function:empty:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content="Unknown tool name: 'foobar'. No tools available.", timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args='{}')],
                model_name='function:empty:',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


def test_unknown_tool_fix():
    def empty(m: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if len(m) > 1:
            return ModelResponse(parts=[TextPart('success')])
        else:
            return ModelResponse(parts=[ToolCallPart('foobar', '{}')])

    agent = Agent(FunctionModel(empty))

    result = agent.run_sync('Hello')
    assert result.data == 'success'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args='{}')],
                model_name='function:empty:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content="Unknown tool name: 'foobar'. No tools available.", timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='success')],
                model_name='function:empty:',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


def test_model_requests_blocked(env: TestEnv):
    env.set('GEMINI_API_KEY', 'foobar')
    agent = Agent('google-gla:gemini-1.5-flash', result_type=tuple[str, str], defer_model_check=True)

    with pytest.raises(RuntimeError, match='Model requests are not allowed, since ALLOW_MODEL_REQUESTS is False'):
        agent.run_sync('Hello')


def test_override_model(env: TestEnv):
    env.set('GEMINI_API_KEY', 'foobar')
    agent = Agent('google-gla:gemini-1.5-flash', result_type=tuple[int, str], defer_model_check=True)

    with agent.override(model='test'):
        result = agent.run_sync('Hello')
        assert result.data == snapshot((0, 'a'))


def test_override_model_no_model():
    agent = Agent()

    with pytest.raises(UserError, match=r'`model` must be set either.+Even when `override\(model=...\)` is customiz'):
        with agent.override(model='test'):
            agent.run_sync('Hello')


def test_run_sync_multiple():
    agent = Agent('test')

    @agent.tool_plain
    async def make_request() -> str:
        async with httpx.AsyncClient() as client:
            # use this as I suspect it's about the fastest globally available endpoint
            try:
                response = await client.get('https://cloudflare.com/cdn-cgi/trace')
            except httpx.ConnectError:  # pragma: no cover
                pytest.skip('offline')
            else:
                return str(response.status_code)

    for _ in range(2):
        result = agent.run_sync('Hello')
        assert result.data == '{"make_request":"200"}'


async def test_agent_name():
    my_agent = Agent('test')

    assert my_agent.name is None

    await my_agent.run('Hello', infer_name=False)
    assert my_agent.name is None

    await my_agent.run('Hello')
    assert my_agent.name == 'my_agent'


async def test_agent_name_already_set():
    my_agent = Agent('test', name='fig_tree')

    assert my_agent.name == 'fig_tree'

    await my_agent.run('Hello')
    assert my_agent.name == 'fig_tree'


async def test_agent_name_changes():
    my_agent = Agent('test')

    await my_agent.run('Hello')
    assert my_agent.name == 'my_agent'

    new_agent = my_agent
    del my_agent

    await new_agent.run('Hello')
    assert new_agent.name == 'my_agent'


def test_name_from_global(create_module: Callable[[str], Any]):
    module_code = """
from pydantic_ai import Agent

my_agent = Agent('test')

def foo():
    result = my_agent.run_sync('Hello')
    return result.data
"""

    mod = create_module(module_code)

    assert mod.my_agent.name is None
    assert mod.foo() == snapshot('success (no tool calls)')
    assert mod.my_agent.name == 'my_agent'


class TestMultipleToolCalls:
    """Tests for scenarios where multiple tool calls are made in a single response."""

    pytestmark = pytest.mark.usefixtures('set_event_loop')

    class ResultType(BaseModel):
        """Result type used by all tests."""

        value: str

    def test_early_strategy_stops_after_first_final_result(self):
        """Test that 'early' strategy stops processing regular tools after first final result."""
        tool_called = []

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.result_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('final_result', {'value': 'final'}),
                    ToolCallPart('regular_tool', {'x': 1}),
                    ToolCallPart('another_tool', {'y': 2}),
                ]
            )

        agent = Agent(FunctionModel(return_model), result_type=self.ResultType, end_strategy='early')

        @agent.tool_plain
        def regular_tool(x: int) -> int:  # pragma: no cover
            """A regular tool that should not be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:  # pragma: no cover
            """Another tool that should not be called."""
            tool_called.append('another_tool')
            return y

        result = agent.run_sync('test early strategy')
        messages = result.all_messages()

        # Verify no tools were called after final result
        assert tool_called == []

        # Verify we got tool returns for all calls
        assert messages[-1].parts == snapshot(
            [
                ToolReturnPart(
                    tool_name='final_result', content='Final result processed.', timestamp=IsNow(tz=timezone.utc)
                ),
                ToolReturnPart(
                    tool_name='regular_tool',
                    content='Tool not executed - a final result was already processed.',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ToolReturnPart(
                    tool_name='another_tool',
                    content='Tool not executed - a final result was already processed.',
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ]
        )

    def test_early_strategy_uses_first_final_result(self):
        """Test that 'early' strategy uses the first final result and ignores subsequent ones."""

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.result_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('final_result', {'value': 'first'}),
                    ToolCallPart('final_result', {'value': 'second'}),
                ]
            )

        agent = Agent(FunctionModel(return_model), result_type=self.ResultType, end_strategy='early')
        result = agent.run_sync('test multiple final results')

        # Verify the result came from the first final tool
        assert result.data.value == 'first'

        # Verify we got appropriate tool returns
        assert result.new_messages()[-1].parts == snapshot(
            [
                ToolReturnPart(
                    tool_name='final_result', content='Final result processed.', timestamp=IsNow(tz=timezone.utc)
                ),
                ToolReturnPart(
                    tool_name='final_result',
                    content='Result tool not used - a final result was already processed.',
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ]
        )

    def test_exhaustive_strategy_executes_all_tools(self):
        """Test that 'exhaustive' strategy executes all tools while using first final result."""
        tool_called: list[str] = []

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.result_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('regular_tool', {'x': 42}),
                    ToolCallPart('final_result', {'value': 'first'}),
                    ToolCallPart('another_tool', {'y': 2}),
                    ToolCallPart('final_result', {'value': 'second'}),
                    ToolCallPart('unknown_tool', {'value': '???'}),
                ]
            )

        agent = Agent(FunctionModel(return_model), result_type=self.ResultType, end_strategy='exhaustive')

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            """A regular tool that should be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:
            """Another tool that should be called."""
            tool_called.append('another_tool')
            return y

        result = agent.run_sync('test exhaustive strategy')

        # Verify the result came from the first final tool
        assert result.data.value == 'first'

        # Verify all regular tools were called
        assert sorted(tool_called) == sorted(['regular_tool', 'another_tool'])

        # Verify we got tool returns in the correct order
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test exhaustive strategy', timestamp=IsNow(tz=timezone.utc))]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='regular_tool', args={'x': 42}),
                        ToolCallPart(tool_name='final_result', args={'value': 'first'}),
                        ToolCallPart(tool_name='another_tool', args={'y': 2}),
                        ToolCallPart(tool_name='final_result', args={'value': 'second'}),
                        ToolCallPart(tool_name='unknown_tool', args={'value': '???'}),
                    ],
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Result tool not used - a final result was already processed.',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        RetryPromptPart(
                            content="Unknown tool name: 'unknown_tool'. Available tools: regular_tool, another_tool, final_result",
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(tool_name='regular_tool', content=42, timestamp=IsNow(tz=timezone.utc)),
                        ToolReturnPart(tool_name='another_tool', content=2, timestamp=IsNow(tz=timezone.utc)),
                    ]
                ),
            ]
        )

    def test_early_strategy_with_final_result_in_middle(self):
        """Test that 'early' strategy stops at first final result, regardless of position."""
        tool_called = []

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.result_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('regular_tool', {'x': 1}),
                    ToolCallPart('final_result', {'value': 'final'}),
                    ToolCallPart('another_tool', {'y': 2}),
                    ToolCallPart('unknown_tool', {'value': '???'}),
                ]
            )

        agent = Agent(FunctionModel(return_model), result_type=self.ResultType, end_strategy='early')

        @agent.tool_plain
        def regular_tool(x: int) -> int:  # pragma: no cover
            """A regular tool that should not be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:  # pragma: no cover
            """A tool that should not be called."""
            tool_called.append('another_tool')
            return y

        result = agent.run_sync('test early strategy with final result in middle')

        # Verify no tools were called
        assert tool_called == []

        # Verify we got appropriate tool returns
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test early strategy with final result in middle', timestamp=IsNow(tz=timezone.utc)
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='regular_tool', args={'x': 1}),
                        ToolCallPart(tool_name='final_result', args={'value': 'final'}),
                        ToolCallPart(tool_name='another_tool', args={'y': 2}),
                        ToolCallPart(tool_name='unknown_tool', args={'value': '???'}),
                    ],
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content='Tool not executed - a final result was already processed.',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool',
                            content='Tool not executed - a final result was already processed.',
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        RetryPromptPart(
                            content="Unknown tool name: 'unknown_tool'. Available tools: regular_tool, another_tool, final_result",
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ]
                ),
            ]
        )

    def test_early_strategy_does_not_apply_to_tool_calls_without_final_tool(self):
        """Test that 'early' strategy does not apply to tool calls without final tool."""
        tool_called = []
        agent = Agent(TestModel(), result_type=self.ResultType, end_strategy='early')

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            """A regular tool that should be called."""
            tool_called.append('regular_tool')
            return x

        result = agent.run_sync('test early strategy with regular tool calls')
        assert tool_called == ['regular_tool']

        tool_returns = [m for m in result.all_messages() if isinstance(m, ToolReturnPart)]
        assert tool_returns == snapshot([])

    def test_multiple_final_result_are_validated_correctly(self):
        """Tests that if multiple final results are returned, but one fails validation, the other is used."""

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.result_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('final_result', {'bad_value': 'first'}, tool_call_id='first'),
                    ToolCallPart('final_result', {'value': 'second'}, tool_call_id='second'),
                ]
            )

        agent = Agent(FunctionModel(return_model), result_type=self.ResultType, end_strategy='early')
        result = agent.run_sync('test multiple final results')

        # Verify the result came from the second final tool
        assert result.data.value == 'second'

        # Verify we got appropriate tool returns
        assert result.new_messages()[-1].parts == snapshot(
            [
                ToolReturnPart(
                    tool_name='final_result',
                    tool_call_id='first',
                    content='Result tool not used - result failed validation.',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ToolReturnPart(
                    tool_name='final_result',
                    content='Final result processed.',
                    timestamp=IsNow(tz=timezone.utc),
                    tool_call_id='second',
                ),
            ]
        )


async def test_model_settings_override() -> None:
    def return_settings(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(to_json(info.model_settings).decode())])

    my_agent = Agent(FunctionModel(return_settings))
    assert (await my_agent.run('Hello')).data == IsJson(None)
    assert (await my_agent.run('Hello', model_settings={'temperature': 0.5})).data == IsJson({'temperature': 0.5})

    my_agent = Agent(FunctionModel(return_settings), model_settings={'temperature': 0.1})
    assert (await my_agent.run('Hello')).data == IsJson({'temperature': 0.1})
    assert (await my_agent.run('Hello', model_settings={'temperature': 0.5})).data == IsJson({'temperature': 0.5})


async def test_empty_text_part():
    def return_empty_text(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.result_tools is not None
        args_json = '{"response": ["foo", "bar"]}'
        return ModelResponse(parts=[TextPart(''), ToolCallPart(info.result_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_empty_text), result_type=tuple[str, str])

    result = await agent.run('Hello')
    assert result.data == ('foo', 'bar')


def test_heterogeneous_responses_non_streaming() -> None:
    """Indicates that tool calls are prioritized over text in heterogeneous responses."""

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.result_tools is not None
        parts: list[ModelResponsePart] = []
        if len(messages) == 1:
            parts = [
                TextPart(content='foo'),
                ToolCallPart('get_location', {'loc_name': 'London'}),
            ]
        else:
            parts = [TextPart(content='final response')]
        return ModelResponse(parts=parts)

    agent = Agent(FunctionModel(return_model))

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = agent.run_sync('Hello')
    assert result.data == 'final response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(content='foo'),
                    ToolCallPart(
                        tool_name='get_location',
                        args={'loc_name': 'London'},
                    ),
                ],
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location', content='{"lat": 51, "lng": 0}', timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


def test_last_run_messages() -> None:
    agent = Agent('test')

    with pytest.raises(AttributeError, match='The `last_run_messages` attribute has been removed,'):
        agent.last_run_messages  # pyright: ignore[reportDeprecated]


def test_nested_capture_run_messages() -> None:
    agent = Agent('test')

    with capture_run_messages() as messages1:
        assert messages1 == []
        with capture_run_messages() as messages2:
            assert messages2 == []
            assert messages1 is messages2
            result = agent.run_sync('Hello')
            assert result.data == 'success (no tool calls)'

    assert messages1 == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')], model_name='test', timestamp=IsNow(tz=timezone.utc)
            ),
        ]
    )
    assert messages1 == messages2


def test_double_capture_run_messages() -> None:
    agent = Agent('test')

    with capture_run_messages() as messages:
        assert messages == []
        result = agent.run_sync('Hello')
        assert result.data == 'success (no tool calls)'
        result2 = agent.run_sync('Hello 2')
        assert result2.data == 'success (no tool calls)'
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')], model_name='test', timestamp=IsNow(tz=timezone.utc)
            ),
        ]
    )


def test_dynamic_false_no_reevaluate():
    """When dynamic is false (default), the system prompt is not reevaluated
    i.e: SystemPromptPart(
            content="A",       <--- Remains the same when `message_history` is passed.
        part_kind='system-prompt')
    """
    agent = Agent('test', system_prompt='Foobar')

    dynamic_value = 'A'

    @agent.system_prompt
    async def func() -> str:
        return dynamic_value

    res = agent.run_sync('Hello')

    assert res.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar', part_kind='system-prompt'),
                    SystemPromptPart(content=dynamic_value, part_kind='system-prompt'),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc), part_kind='user-prompt'),
                ],
                kind='request',
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)', part_kind='text')],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                kind='response',
            ),
        ]
    )

    dynamic_value = 'B'

    res_two = agent.run_sync('World', message_history=res.all_messages())

    assert res_two.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar', part_kind='system-prompt'),
                    SystemPromptPart(
                        content='A',  # Remains the same
                        part_kind='system-prompt',
                    ),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc), part_kind='user-prompt'),
                ],
                kind='request',
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)', part_kind='text')],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                kind='response',
            ),
            ModelRequest(
                parts=[UserPromptPart(content='World', timestamp=IsNow(tz=timezone.utc), part_kind='user-prompt')],
                kind='request',
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)', part_kind='text')],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                kind='response',
            ),
        ]
    )


def test_dynamic_true_reevaluate_system_prompt():
    """When dynamic is true, the system prompt is reevaluated
    i.e: SystemPromptPart(
            content="B",       <--- Updated value
        part_kind='system-prompt')
    """
    agent = Agent('test', system_prompt='Foobar')

    dynamic_value = 'A'

    @agent.system_prompt(dynamic=True)
    async def func():
        return dynamic_value

    res = agent.run_sync('Hello')

    assert res.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar', part_kind='system-prompt'),
                    SystemPromptPart(
                        content=dynamic_value,
                        part_kind='system-prompt',
                        dynamic_ref=func.__qualname__,
                    ),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc), part_kind='user-prompt'),
                ],
                kind='request',
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)', part_kind='text')],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                kind='response',
            ),
        ]
    )

    dynamic_value = 'B'

    res_two = agent.run_sync('World', message_history=res.all_messages())

    assert res_two.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar', part_kind='system-prompt'),
                    SystemPromptPart(
                        content='B',
                        part_kind='system-prompt',
                        dynamic_ref=func.__qualname__,
                    ),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc), part_kind='user-prompt'),
                ],
                kind='request',
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)', part_kind='text')],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                kind='response',
            ),
            ModelRequest(
                parts=[UserPromptPart(content='World', timestamp=IsNow(tz=timezone.utc), part_kind='user-prompt')],
                kind='request',
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)', part_kind='text')],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                kind='response',
            ),
        ]
    )


def test_capture_run_messages_tool_agent() -> None:
    agent_outer = Agent('test')
    agent_inner = Agent(TestModel(custom_result_text='inner agent result'))

    @agent_outer.tool_plain
    async def foobar(x: str) -> str:
        result_ = await agent_inner.run(x)
        return result_.data

    with capture_run_messages() as messages:
        result = agent_outer.run_sync('foobar')

    assert result.data == snapshot('{"foobar":"inner agent result"}')
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='foobar', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args={'x': 'a'})],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='foobar', content='inner agent result', timestamp=IsNow(tz=timezone.utc))
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='{"foobar":"inner agent result"}')],
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


class Bar(BaseModel):
    c: int
    d: str


def test_custom_result_type_sync() -> None:
    agent = Agent('test', result_type=Foo)

    assert agent.run_sync('Hello').data == snapshot(Foo(a=0, b='a'))
    assert agent.run_sync('Hello', result_type=Bar).data == snapshot(Bar(c=0, d='a'))
    assert agent.run_sync('Hello', result_type=str).data == snapshot('success (no tool calls)')
    assert agent.run_sync('Hello', result_type=int).data == snapshot(0)


async def test_custom_result_type_async() -> None:
    agent = Agent('test')

    result = await agent.run('Hello')
    assert result.data == snapshot('success (no tool calls)')

    result = await agent.run('Hello', result_type=Foo)
    assert result.data == snapshot(Foo(a=0, b='a'))
    result = await agent.run('Hello', result_type=int)
    assert result.data == snapshot(0)


def test_custom_result_type_invalid() -> None:
    agent = Agent('test')

    @agent.result_validator
    def validate_result(ctx: RunContext[None], r: Any) -> Any:  # pragma: no cover
        return r

    with pytest.raises(UserError, match='Cannot set a custom run `result_type` when the agent has result validators'):
        agent.run_sync('Hello', result_type=int)


def test_binary_content_all_messages_json():
    agent = Agent('test')

    result = agent.run_sync(['Hello', BinaryContent(data=b'Hello', media_type='text/plain')])
    assert json.loads(result.all_messages_json()) == snapshot(
        [
            {
                'parts': [
                    {
                        'content': ['Hello', {'data': 'SGVsbG8=', 'media_type': 'text/plain', 'kind': 'binary'}],
                        'timestamp': IsStr(),
                        'part_kind': 'user-prompt',
                    }
                ],
                'kind': 'request',
            },
            {
                'parts': [{'content': 'success (no tool calls)', 'part_kind': 'text'}],
                'model_name': 'test',
                'timestamp': IsStr(),
                'kind': 'response',
            },
        ]
    )
