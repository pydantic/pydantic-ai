import json
import re
import sys
from dataclasses import dataclass
from datetime import timezone
from typing import Any, Callable, Union

import httpx
import pytest
from dirty_equals import IsJson
from inline_snapshot import snapshot
from pydantic import BaseModel, TypeAdapter, field_validator
from pydantic_core import to_json
from typing_extensions import Self

from pydantic_ai import Agent, ModelRetry, RunContext, UnexpectedModelBehavior, UserError, capture_run_messages
from pydantic_ai._output import (
    NativeOutput,
    NativeOutputSchema,
    OutputSpec,
    PromptedOutput,
    TextOutput,
    TextOutputSchema,
    ToolOutputSchema,
)
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import StructuredDict, ToolOutput
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.result import Usage
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.combined import CombinedToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.toolsets.prefixed import PrefixedToolset

from .conftest import IsDatetime, IsNow, IsStr, TestEnv

pytestmark = pytest.mark.anyio


def test_result_tuple():
    def return_tuple(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"response": ["foo", "bar"]}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_tuple), output_type=tuple[str, str])

    result = agent.run_sync('Hello')
    assert result.output == ('foo', 'bar')


class Foo(BaseModel):
    a: int
    b: str


def test_result_pydantic_model():
    def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"a": 1, "b": "foo"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_model), output_type=Foo)

    result = agent.run_sync('Hello')
    assert isinstance(result.output, Foo)
    assert result.output.model_dump() == {'a': 1, 'b': 'foo'}


def test_result_pydantic_model_retry():
    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        if len(messages) == 1:
            args_json = '{"a": "wrong", "b": "foo"}'
        else:
            args_json = '{"a": 42, "b": "foo"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_model), output_type=Foo)

    assert agent.name is None

    result = agent.run_sync('Hello')
    assert agent.name == 'agent'
    assert isinstance(result.output, Foo)
    assert result.output.model_dump() == {'a': 42, 'b': 'foo'}
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args='{"a": "wrong", "b": "foo"}', tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=51, response_tokens=7, total_tokens=58),
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
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args='{"a": 42, "b": "foo"}', tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=87, response_tokens=14, total_tokens=101),
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
        ]
    )
    assert result.all_messages_json().startswith(b'[{"parts":[{"content":"Hello",')


def test_result_pydantic_model_validation_error():
    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        if len(messages) == 1:
            args_json = '{"a": 1, "b": "foo"}'
        else:
            args_json = '{"a": 1, "b": "bar"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    class Bar(BaseModel):
        a: int
        b: str

        @field_validator('b')
        def check_b(cls, v: str) -> str:
            if v == 'foo':
                raise ValueError('must not be foo')
            return v

    agent = Agent(FunctionModel(return_model), output_type=Bar)

    result = agent.run_sync('Hello')
    assert isinstance(result.output, Bar)
    assert result.output.model_dump() == snapshot({'a': 1, 'b': 'bar'})
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


def test_output_validator():
    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        if len(messages) == 1:
            args_json = '{"a": 41, "b": "foo"}'
        else:
            args_json = '{"a": 42, "b": "foo"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_model), output_type=Foo)

    @agent.output_validator
    def validate_output(ctx: RunContext[None], o: Foo) -> Foo:
        assert ctx.tool_name == 'final_result'
        if o.a == 42:
            return o
        else:
            raise ModelRetry('"a" should be 42')

    result = agent.run_sync('Hello')
    assert isinstance(result.output, Foo)
    assert result.output.model_dump() == {'a': 42, 'b': 'foo'}
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args='{"a": 41, "b": "foo"}', tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=51, response_tokens=7, total_tokens=58),
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='"a" should be 42',
                        tool_name='final_result',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args='{"a": 42, "b": "foo"}', tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=63, response_tokens=14, total_tokens=77),
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
        ]
    )


def test_plain_response_then_tuple():
    call_index = 0

    def return_tuple(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_index

        assert info.output_tools is not None
        call_index += 1
        if call_index == 1:
            return ModelResponse(parts=[TextPart('hello')])
        else:
            args_json = '{"response": ["foo", "bar"]}'
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_tuple), output_type=ToolOutput(tuple[str, str]))

    result = agent.run_sync('Hello')
    assert result.output == ('foo', 'bar')
    assert call_index == 2
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='hello')],
                usage=Usage(requests=1, request_tokens=51, response_tokens=1, total_tokens=52),
                model_name='function:return_tuple:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Plain text responses are not permitted, please include your response in a tool call',
                        timestamp=IsNow(tz=timezone.utc),
                        tool_call_id=IsStr(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='final_result', args='{"response": ["foo", "bar"]}', tool_call_id=IsStr())
                ],
                usage=Usage(requests=1, request_tokens=74, response_tokens=8, total_tokens=82),
                model_name='function:return_tuple:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
        ]
    )
    assert result._output_tool_name == 'final_result'  # pyright: ignore[reportPrivateUsage]
    assert result.all_messages(output_tool_return_content='foobar')[-1] == snapshot(
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='final_result', content='foobar', tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                )
            ]
        )
    )
    assert result.all_messages()[-1] == snapshot(
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='final_result',
                    content='Final result processed.',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                )
            ]
        )
    )


def test_output_tool_return_content_str_return():
    agent = Agent('test')

    result = agent.run_sync('Hello')
    assert result.output == 'success (no tool calls)'

    msg = re.escape('Cannot set output tool return content when the return type is `str`.')
    with pytest.raises(ValueError, match=msg):
        result.all_messages(output_tool_return_content='foobar')


def test_output_tool_return_content_no_tool():
    agent = Agent('test', output_type=int)

    result = agent.run_sync('Hello')
    assert result.output == 0
    result._output_tool_name = 'wrong'  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(LookupError, match=re.escape("No tool call found with tool name 'wrong'.")):
        result.all_messages(output_tool_return_content='foobar')


def test_response_tuple():
    m = TestModel()

    agent = Agent(m, output_type=tuple[str, str])
    assert isinstance(agent._output_schema, ToolOutputSchema)  # pyright: ignore[reportPrivateUsage]

    result = agent.run_sync('Hello')
    assert result.output == snapshot(('a', 'a'))

    assert m.last_model_request_parameters is not None
    assert m.last_model_request_parameters.function_tools == snapshot([])
    assert m.last_model_request_parameters.allow_text_output is False

    assert m.last_model_request_parameters.output_tools is not None
    assert len(m.last_model_request_parameters.output_tools) == 1
    assert m.last_model_request_parameters.output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {
                        'response': {
                            'maxItems': 2,
                            'minItems': 2,
                            'prefixItems': [{'type': 'string'}, {'type': 'string'}],
                            'type': 'array',
                        }
                    },
                    'required': ['response'],
                    'type': 'object',
                },
                outer_typed_dict_key='response',
                kind='output',
            )
        ]
    )


def upcase(text: str) -> str:
    return text.upper()


@pytest.mark.parametrize(
    'input_union_callable',
    [
        lambda: Union[str, Foo],
        lambda: Union[Foo, str],
        lambda: str | Foo,
        lambda: Foo | str,
        lambda: [Foo, str],
        lambda: [TextOutput(upcase), ToolOutput(Foo)],
    ],
    ids=[
        'Union[str, Foo]',
        'Union[Foo, str]',
        'str | Foo',
        'Foo | str',
        '[Foo, str]',
        '[TextOutput(upcase), ToolOutput(Foo)]',
    ],
)
def test_response_union_allow_str(input_union_callable: Callable[[], Any]):
    try:
        union = input_union_callable()
    except TypeError:  # pragma: lax no cover
        pytest.skip('Python version does not support `|` syntax for unions')

    m = TestModel()
    agent: Agent[None, Union[str, Foo]] = Agent(m, output_type=union)

    got_tool_call_name = 'unset'

    @agent.output_validator
    def validate_output(ctx: RunContext[None], o: Any) -> Any:
        nonlocal got_tool_call_name
        got_tool_call_name = ctx.tool_name
        return o

    assert isinstance(agent._output_schema, TextOutputSchema)  # pyright: ignore[reportPrivateUsage]

    result = agent.run_sync('Hello')
    assert isinstance(result.output, str)
    assert result.output.lower() == snapshot('success (no tool calls)')
    assert got_tool_call_name == snapshot(None)

    assert m.last_model_request_parameters is not None
    assert m.last_model_request_parameters.function_tools == snapshot([])
    assert m.last_model_request_parameters.allow_text_output is True

    assert m.last_model_request_parameters.output_tools is not None
    assert len(m.last_model_request_parameters.output_tools) == 1

    assert m.last_model_request_parameters.output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'properties': {
                        'a': {'type': 'integer'},
                        'b': {'type': 'string'},
                    },
                    'required': ['a', 'b'],
                    'title': 'Foo',
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
@pytest.mark.parametrize(
    'union_code',
    [
        pytest.param('OutputType = Union[Foo, Bar]'),
        pytest.param('OutputType = [Foo, Bar]'),
        pytest.param('OutputType = [ToolOutput(Foo), ToolOutput(Bar)]'),
        pytest.param('OutputType = Foo | Bar', marks=pytest.mark.skipif(sys.version_info < (3, 10), reason='3.10+')),
        pytest.param(
            'OutputType: TypeAlias = Foo | Bar',
            marks=pytest.mark.skipif(sys.version_info < (3, 10), reason='Python 3.10+'),
        ),
        pytest.param(
            'type OutputType = Foo | Bar', marks=pytest.mark.skipif(sys.version_info < (3, 12), reason='3.12+')
        ),
    ],
)
def test_response_multiple_return_tools(create_module: Callable[[str], Any], union_code: str):
    module_code = f'''
from pydantic import BaseModel
from typing import Union
from typing_extensions import TypeAlias
from pydantic_ai import ToolOutput

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
    agent = Agent(m, output_type=mod.OutputType)
    got_tool_call_name = 'unset'

    @agent.output_validator
    def validate_output(ctx: RunContext[None], o: Any) -> Any:
        nonlocal got_tool_call_name
        got_tool_call_name = ctx.tool_name
        return o

    result = agent.run_sync('Hello')
    assert result.output == mod.Foo(a=0, b='a')
    assert got_tool_call_name == snapshot('final_result_Foo')

    assert m.last_model_request_parameters is not None
    assert m.last_model_request_parameters.function_tools == snapshot([])
    assert m.last_model_request_parameters.allow_text_output is False

    assert m.last_model_request_parameters.output_tools is not None
    assert len(m.last_model_request_parameters.output_tools) == 2

    assert m.last_model_request_parameters.output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result_Foo',
                description='Foo: The final response which ends this conversation',
                parameters_json_schema={
                    'properties': {
                        'a': {'type': 'integer'},
                        'b': {'type': 'string'},
                    },
                    'required': ['a', 'b'],
                    'title': 'Foo',
                    'type': 'object',
                },
                kind='output',
            ),
            ToolDefinition(
                name='final_result_Bar',
                description='This is a bar model.',
                parameters_json_schema={
                    'properties': {'b': {'type': 'string'}},
                    'required': ['b'],
                    'title': 'Bar',
                    'type': 'object',
                },
                kind='output',
            ),
        ]
    )

    result = agent.run_sync('Hello', model=TestModel(seed=1))
    assert result.output == mod.Bar(b='b')
    assert got_tool_call_name == snapshot('final_result_Bar')


def test_output_type_with_two_descriptions():
    class MyOutput(BaseModel):
        """Description from docstring"""

        valid: bool

    m = TestModel()
    agent = Agent(m, output_type=ToolOutput(MyOutput, description='Description from ToolOutput'))
    result = agent.run_sync('Hello')
    assert result.output == snapshot(MyOutput(valid=False))
    assert m.last_model_request_parameters is not None
    assert m.last_model_request_parameters.output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='Description from ToolOutput. Description from docstring',
                parameters_json_schema={
                    'properties': {'valid': {'type': 'boolean'}},
                    'required': ['valid'],
                    'title': 'MyOutput',
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


def test_output_type_tool_output_union():
    class Foo(BaseModel):
        a: int
        b: str

    class Bar(BaseModel):
        c: bool

    m = TestModel()
    marker: ToolOutput[Union[Foo, Bar]] = ToolOutput(Union[Foo, Bar], strict=False)  # type: ignore
    agent = Agent(m, output_type=marker)
    result = agent.run_sync('Hello')
    assert result.output == snapshot(Foo(a=0, b='a'))
    assert m.last_model_request_parameters is not None
    assert m.last_model_request_parameters.output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    '$defs': {
                        'Bar': {
                            'properties': {'c': {'type': 'boolean'}},
                            'required': ['c'],
                            'title': 'Bar',
                            'type': 'object',
                        },
                        'Foo': {
                            'properties': {'a': {'type': 'integer'}, 'b': {'type': 'string'}},
                            'required': ['a', 'b'],
                            'title': 'Foo',
                            'type': 'object',
                        },
                    },
                    'additionalProperties': False,
                    'properties': {'response': {'anyOf': [{'$ref': '#/$defs/Foo'}, {'$ref': '#/$defs/Bar'}]}},
                    'required': ['response'],
                    'type': 'object',
                },
                outer_typed_dict_key='response',
                strict=False,
                kind='output',
            )
        ]
    )


def test_output_type_function():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(city: str) -> Weather:
        return Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=get_weather)
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


def test_output_type_function_with_run_context():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(ctx: RunContext[None], city: str) -> Weather:
        assert ctx is not None
        return Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=get_weather)
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


def test_output_type_bound_instance_method():
    class Weather(BaseModel):
        temperature: float
        description: str

        def get_weather(self, city: str) -> Self:
            return self

    weather = Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=weather.get_weather)
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


def test_output_type_bound_instance_method_with_run_context():
    class Weather(BaseModel):
        temperature: float
        description: str

        def get_weather(self, ctx: RunContext[None], city: str) -> Self:
            assert ctx is not None
            return self

    weather = Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=weather.get_weather)
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


def test_output_type_function_with_retry():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(city: str) -> Weather:
        if city != 'Mexico City':
            raise ModelRetry('City not found, I only know Mexico City')
        return Weather(temperature=28.7, description='sunny')

    def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        if len(messages) == 1:
            args_json = '{"city": "New York City"}'
        else:
            args_json = '{"city": "Mexico City"}'

        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=get_weather)
    result = agent.run_sync('New York City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='New York City',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city": "New York City"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=Usage(requests=1, request_tokens=53, response_tokens=7, total_tokens=60),
                model_name='function:call_tool:',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='City not found, I only know Mexico City',
                        tool_name='final_result',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city": "Mexico City"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=Usage(requests=1, request_tokens=68, response_tokens=13, total_tokens=81),
                model_name='function:call_tool:',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )


def test_output_type_text_output_function_with_retry():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(ctx: RunContext[None], city: str) -> Weather:
        assert ctx is not None
        if city != 'Mexico City':
            raise ModelRetry('City not found, I only know Mexico City')
        return Weather(temperature=28.7, description='sunny')

    def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        if len(messages) == 1:
            city = 'New York City'
        else:
            city = 'Mexico City'

        return ModelResponse(parts=[TextPart(content=city)])

    agent = Agent(FunctionModel(call_tool), output_type=TextOutput(get_weather))
    result = agent.run_sync('New York City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='New York City',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='New York City')],
                usage=Usage(requests=1, request_tokens=53, response_tokens=3, total_tokens=56),
                model_name='function:call_tool:',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='City not found, I only know Mexico City',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='Mexico City')],
                usage=Usage(requests=1, request_tokens=70, response_tokens=5, total_tokens=75),
                model_name='function:call_tool:',
                timestamp=IsDatetime(),
            ),
        ]
    )


@pytest.mark.parametrize(
    'output_type',
    [[str, str], [str, TextOutput(upcase)], [TextOutput(upcase), TextOutput(upcase)]],
)
def test_output_type_multiple_text_output(output_type: OutputSpec[str]):
    with pytest.raises(UserError, match='Only one `str` or `TextOutput` is allowed.'):
        Agent('test', output_type=output_type)


def test_output_type_text_output_invalid():
    def int_func(x: int) -> str:
        return str(int)  # pragma: no cover

    with pytest.raises(UserError, match='TextOutput must take a function taking a `str`'):
        output_type: TextOutput[str] = TextOutput(int_func)  # type: ignore
        Agent('test', output_type=output_type)


def test_output_type_async_function():
    class Weather(BaseModel):
        temperature: float
        description: str

    async def get_weather(city: str) -> Weather:
        return Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=get_weather)
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


def test_output_type_function_with_custom_tool_name():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(city: str) -> Weather:
        return Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=ToolOutput(get_weather, name='get_weather'))
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='get_weather',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


def test_output_type_function_or_model():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(city: str) -> Weather:
        return Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=[get_weather, Weather])
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result_get_weather',
                description='get_weather: The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            ),
            ToolDefinition(
                name='final_result_Weather',
                description='Weather: The final response which ends this conversation',
                parameters_json_schema={
                    'properties': {'temperature': {'type': 'number'}, 'description': {'type': 'string'}},
                    'required': ['temperature', 'description'],
                    'title': 'Weather',
                    'type': 'object',
                },
                kind='output',
            ),
        ]
    )


def test_output_type_text_output_function():
    def say_world(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='world')])

    agent = Agent(FunctionModel(say_world), output_type=TextOutput(upcase))
    result = agent.run_sync('hello')
    assert result.output == snapshot('WORLD')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='hello',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=Usage(requests=1, request_tokens=51, response_tokens=1, total_tokens=52),
                model_name='function:say_world:',
                timestamp=IsDatetime(),
            ),
        ]
    )


def test_output_type_handoff_to_agent():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(city: str) -> Weather:
        return Weather(temperature=28.7, description='sunny')

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=get_weather)

    handoff_result = None

    async def handoff(city: str) -> Weather:
        result = await agent.run(f'Get me the weather in {city}')
        nonlocal handoff_result
        handoff_result = result
        return result.output

    def call_handoff_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    supervisor_agent = Agent(FunctionModel(call_handoff_tool), output_type=handoff)

    result = supervisor_agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Mexico City',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city": "Mexico City"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=Usage(requests=1, request_tokens=52, response_tokens=6, total_tokens=58),
                model_name='function:call_handoff_tool:',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )
    assert handoff_result is not None
    assert handoff_result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Get me the weather in Mexico City',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city": "Mexico City"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=Usage(requests=1, request_tokens=57, response_tokens=6, total_tokens=63),
                model_name='function:call_tool:',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )


def test_output_type_multiple_custom_tools():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(city: str) -> Weather:
        return Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(
        FunctionModel(call_tool),
        output_type=[
            ToolOutput(get_weather, name='get_weather'),
            ToolOutput(Weather, name='return_weather'),
        ],
    )
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='get_weather',
                description='get_weather: The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            ),
            ToolDefinition(
                name='return_weather',
                description='Weather: The final response which ends this conversation',
                parameters_json_schema={
                    'properties': {'temperature': {'type': 'number'}, 'description': {'type': 'string'}},
                    'required': ['temperature', 'description'],
                    'title': 'Weather',
                    'type': 'object',
                },
                kind='output',
            ),
        ]
    )


def test_output_type_structured_dict():
    PersonDict = StructuredDict(
        {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'age': {'type': 'integer'},
            },
            'required': ['name', 'age'],
        },
        name='Person',
        description='A person',
    )
    AnimalDict = StructuredDict(
        {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'species': {'type': 'string'},
            },
            'required': ['name', 'species'],
        },
        name='Animal',
        description='An animal',
    )

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"name": "John Doe", "age": 30}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(
        FunctionModel(call_tool),
        output_type=[PersonDict, AnimalDict],
    )

    result = agent.run_sync('Generate a person')

    assert result.output == snapshot({'name': 'John Doe', 'age': 30})
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result_Person',
                parameters_json_schema={
                    'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}},
                    'required': ['name', 'age'],
                    'title': 'Person',
                    'type': 'object',
                },
                description='A person',
                kind='output',
            ),
            ToolDefinition(
                name='final_result_Animal',
                parameters_json_schema={
                    'properties': {'name': {'type': 'string'}, 'species': {'type': 'string'}},
                    'required': ['name', 'species'],
                    'title': 'Animal',
                    'type': 'object',
                },
                description='An animal',
                kind='output',
            ),
        ]
    )


def test_default_structured_output_mode():
    def hello(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='hello')])  # pragma: no cover

    tool_model = FunctionModel(hello, profile=ModelProfile(default_structured_output_mode='tool'))
    native_model = FunctionModel(
        hello,
        profile=ModelProfile(supports_json_schema_output=True, default_structured_output_mode='native'),
    )
    prompted_model = FunctionModel(
        hello,
        profile=ModelProfile(default_structured_output_mode='prompted'),
    )

    class Foo(BaseModel):
        bar: str

    tool_agent = Agent(tool_model, output_type=Foo)
    assert tool_agent._output_schema.mode == 'tool'  # type: ignore[reportPrivateUsage]

    native_agent = Agent(native_model, output_type=Foo)
    assert native_agent._output_schema.mode == 'native'  # type: ignore[reportPrivateUsage]

    prompted_agent = Agent(prompted_model, output_type=Foo)
    assert prompted_agent._output_schema.mode == 'prompted'  # type: ignore[reportPrivateUsage]


def test_prompted_output():
    def return_city_location(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        text = CityLocation(city='Mexico City', country='Mexico').model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    m = FunctionModel(return_city_location)

    class CityLocation(BaseModel):
        """Description from docstring."""

        city: str
        country: str

    agent = Agent(
        m,
        output_type=PromptedOutput(CityLocation, name='City & Country', description='Description from PromptedOutput'),
    )

    result = agent.run_sync('What is the capital of Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "City & Country", "type": "object", "description": "Description from PromptedOutput. Description from docstring."}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[TextPart(content='{"city":"Mexico City","country":"Mexico"}')],
                usage=Usage(requests=1, request_tokens=56, response_tokens=7, total_tokens=63),
                model_name='function:return_city_location:',
                timestamp=IsDatetime(),
            ),
        ]
    )


def test_prompted_output_with_template():
    def return_foo(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        text = Foo(bar='baz').model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    m = FunctionModel(return_foo)

    class Foo(BaseModel):
        bar: str

    agent = Agent(m, output_type=PromptedOutput(Foo, template='Gimme some JSON:'))

    result = agent.run_sync('What is the capital of Mexico?')
    assert result.output == snapshot(Foo(bar='baz'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="""\
Gimme some JSON:

{"properties": {"bar": {"type": "string"}}, "required": ["bar"], "title": "Foo", "type": "object"}\
""",
            ),
            ModelResponse(
                parts=[TextPart(content='{"bar":"baz"}')],
                usage=Usage(requests=1, request_tokens=56, response_tokens=4, total_tokens=60),
                model_name='function:return_foo:',
                timestamp=IsDatetime(),
            ),
        ]
    )


def test_prompted_output_with_defs():
    class Foo(BaseModel):
        """Foo description"""

        foo: str

    class Bar(BaseModel):
        """Bar description"""

        bar: str

    class Baz(BaseModel):
        """Baz description"""

        baz: str

    class FooBar(BaseModel):
        """FooBar description"""

        foo: Foo
        bar: Bar

    class FooBaz(BaseModel):
        """FooBaz description"""

        foo: Foo
        baz: Baz

    def return_foo_bar(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        text = '{"result": {"kind": "FooBar", "data": {"foo": {"foo": "foo"}, "bar": {"bar": "bar"}}}}'
        return ModelResponse(parts=[TextPart(content=text)])

    m = FunctionModel(return_foo_bar)

    agent = Agent(
        m,
        output_type=PromptedOutput(
            [FooBar, FooBaz], name='FooBar or FooBaz', description='FooBar or FooBaz description'
        ),
    )

    result = agent.run_sync('What is foo?')
    assert result.output == snapshot(FooBar(foo=Foo(foo='foo'), bar=Bar(bar='bar')))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is foo?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"type": "object", "properties": {"result": {"anyOf": [{"type": "object", "properties": {"kind": {"type": "string", "const": "FooBar"}, "data": {"properties": {"foo": {"$ref": "#/$defs/Foo"}, "bar": {"$ref": "#/$defs/Bar"}}, "required": ["foo", "bar"], "type": "object"}}, "required": ["kind", "data"], "additionalProperties": false, "title": "FooBar", "description": "FooBar description"}, {"type": "object", "properties": {"kind": {"type": "string", "const": "FooBaz"}, "data": {"properties": {"foo": {"$ref": "#/$defs/Foo"}, "baz": {"$ref": "#/$defs/Baz"}}, "required": ["foo", "baz"], "type": "object"}}, "required": ["kind", "data"], "additionalProperties": false, "title": "FooBaz", "description": "FooBaz description"}]}}, "required": ["result"], "additionalProperties": false, "$defs": {"Bar": {"description": "Bar description", "properties": {"bar": {"type": "string"}}, "required": ["bar"], "title": "Bar", "type": "object"}, "Foo": {"description": "Foo description", "properties": {"foo": {"type": "string"}}, "required": ["foo"], "title": "Foo", "type": "object"}, "Baz": {"description": "Baz description", "properties": {"baz": {"type": "string"}}, "required": ["baz"], "title": "Baz", "type": "object"}}, "title": "FooBar or FooBaz", "description": "FooBaz description"}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"result": {"kind": "FooBar", "data": {"foo": {"foo": "foo"}, "bar": {"bar": "bar"}}}}'
                    )
                ],
                usage=Usage(requests=1, request_tokens=53, response_tokens=17, total_tokens=70),
                model_name='function:return_foo_bar:',
                timestamp=IsDatetime(),
            ),
        ]
    )


def test_native_output():
    def return_city_location(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            text = '{"city": "Mexico City"}'
        else:
            text = '{"city": "Mexico City", "country": "Mexico"}'
        return ModelResponse(parts=[TextPart(content=text)])

    m = FunctionModel(return_city_location, profile=ModelProfile(supports_json_schema_output=True))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(
        m,
        output_type=NativeOutput(CityLocation),
    )

    result = agent.run_sync('What is the capital of Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of Mexico?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Mexico City"}')],
                usage=Usage(requests=1, request_tokens=56, response_tokens=5, total_tokens=61),
                model_name='function:return_city_location:',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content=[
                            {
                                'type': 'missing',
                                'loc': ('country',),
                                'msg': 'Field required',
                                'input': {'city': 'Mexico City'},
                            }
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Mexico City", "country": "Mexico"}')],
                usage=Usage(requests=1, request_tokens=85, response_tokens=12, total_tokens=97),
                model_name='function:return_city_location:',
                timestamp=IsDatetime(),
            ),
        ]
    )


def test_native_output_strict_mode():
    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(output_type=NativeOutput(CityLocation, strict=True))
    output_schema = agent._output_schema  # pyright: ignore[reportPrivateUsage]
    assert isinstance(output_schema, NativeOutputSchema)
    assert output_schema.object_def.strict


def test_prompted_output_function_with_retry():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(city: str) -> Weather:
        if city != 'Mexico City':
            raise ModelRetry('City not found, I only know Mexico City')
        return Weather(temperature=28.7, description='sunny')

    def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        if len(messages) == 1:
            args_json = '{"city": "New York City"}'
        else:
            args_json = '{"city": "Mexico City"}'

        return ModelResponse(parts=[TextPart(content=args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=PromptedOutput(get_weather))
    result = agent.run_sync('New York City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='New York City',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"additionalProperties": false, "properties": {"city": {"type": "string"}}, "required": ["city"], "type": "object", "title": "get_weather"}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "New York City"}')],
                usage=Usage(requests=1, request_tokens=53, response_tokens=6, total_tokens=59),
                model_name='function:call_tool:',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='City not found, I only know Mexico City',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Mexico City"}')],
                usage=Usage(requests=1, request_tokens=70, response_tokens=11, total_tokens=81),
                model_name='function:call_tool:',
                timestamp=IsDatetime(),
            ),
        ]
    )


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
                    SystemPromptPart(content='Foobar', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=52, response_tokens=5, total_tokens=57),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ret_a', content='a-apple', tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')],
                usage=Usage(requests=1, request_tokens=53, response_tokens=9, total_tokens=62),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )

    # if we pass new_messages, system prompt is inserted before the message_history messages
    result2 = agent.run_sync('Hello again', message_history=result1.new_messages())
    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=52, response_tokens=5, total_tokens=57),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ret_a', content='a-apple', tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')],
                usage=Usage(requests=1, request_tokens=53, response_tokens=9, total_tokens=62),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(parts=[UserPromptPart(content='Hello again', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')],
                usage=Usage(requests=1, request_tokens=55, response_tokens=13, total_tokens=68),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )
    assert result2._new_message_index == snapshot(4)  # pyright: ignore[reportPrivateUsage]
    assert result2.output == snapshot('{"ret_a":"a-apple"}')
    assert result2._output_tool_name == snapshot(None)  # pyright: ignore[reportPrivateUsage]
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
                    SystemPromptPart(content='Foobar', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=52, response_tokens=5, total_tokens=57),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ret_a', content='a-apple', tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')],
                usage=Usage(requests=1, request_tokens=53, response_tokens=9, total_tokens=62),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(parts=[UserPromptPart(content='Hello again', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')],
                usage=Usage(requests=1, request_tokens=55, response_tokens=13, total_tokens=68),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )
    assert result3._new_message_index == snapshot(4)  # pyright: ignore[reportPrivateUsage]
    assert result3.output == snapshot('{"ret_a":"a-apple"}')
    assert result3._output_tool_name == snapshot(None)  # pyright: ignore[reportPrivateUsage]
    assert result3.usage() == snapshot(
        Usage(requests=1, request_tokens=55, response_tokens=13, total_tokens=68, details=None)
    )


def test_run_with_history_new_structured():
    m = TestModel()

    class Response(BaseModel):
        a: int

    agent = Agent(m, system_prompt='Foobar', output_type=Response)

    @agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    result1 = agent.run_sync('Hello')
    assert result1.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=52, response_tokens=5, total_tokens=57),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ret_a', content='a-apple', tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'a': 0},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=Usage(requests=1, request_tokens=53, response_tokens=9, total_tokens=62),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
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
                    SystemPromptPart(content='Foobar', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=52, response_tokens=5, total_tokens=57),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ret_a', content='a-apple', tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ],
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args={'a': 0}, tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=53, response_tokens=9, total_tokens=62),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
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
                parts=[ToolCallPart(tool_name='final_result', args={'a': 0}, tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=59, response_tokens=13, total_tokens=72),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                ]
            ),
        ]
    )
    assert result2.output == snapshot(Response(a=0))
    assert result2._new_message_index == snapshot(5)  # pyright: ignore[reportPrivateUsage]
    assert result2._output_tool_name == snapshot('final_result')  # pyright: ignore[reportPrivateUsage]
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


def test_run_with_history_and_no_user_prompt():
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Hello')], instructions='Original instructions'),
    ]

    m = TestModel()
    agent = Agent(m, instructions='New instructions')

    result = agent.run_sync(message_history=messages)
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='New instructions',
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=Usage(requests=1, request_tokens=51, response_tokens=4, total_tokens=55),
                model_name='test',
                timestamp=IsDatetime(),
            ),
        ]
    )


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
        with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum retries \(1\) for output validation'):
            agent.run_sync('Hello')
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args='{}', tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=51, response_tokens=2, total_tokens=53),
                model_name='function:empty:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        tool_name='foobar',
                        content="Unknown tool name: 'foobar'. No tools available.",
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args='{}', tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=65, response_tokens=4, total_tokens=69),
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
    assert result.output == 'success'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args='{}', tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=51, response_tokens=2, total_tokens=53),
                model_name='function:empty:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        tool_name='foobar',
                        content="Unknown tool name: 'foobar'. No tools available.",
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='success')],
                usage=Usage(requests=1, request_tokens=65, response_tokens=3, total_tokens=68),
                model_name='function:empty:',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


def test_model_requests_blocked(env: TestEnv):
    try:
        env.set('GEMINI_API_KEY', 'foobar')
        agent = Agent('google-gla:gemini-1.5-flash', output_type=tuple[str, str], defer_model_check=True)

        with pytest.raises(RuntimeError, match='Model requests are not allowed, since ALLOW_MODEL_REQUESTS is False'):
            agent.run_sync('Hello')
    except ImportError:  # pragma: lax no cover
        pytest.skip('google-genai not installed')


def test_override_model(env: TestEnv):
    env.set('GEMINI_API_KEY', 'foobar')
    agent = Agent('google-gla:gemini-1.5-flash', output_type=tuple[int, str], defer_model_check=True)

    with agent.override(model='test'):
        result = agent.run_sync('Hello')
        assert result.output == snapshot((0, 'a'))


def test_set_model(env: TestEnv):
    env.set('GEMINI_API_KEY', 'foobar')
    agent = Agent(output_type=tuple[int, str])

    agent.model = 'test'

    result = agent.run_sync('Hello')
    assert result.output == snapshot((0, 'a'))


def test_override_model_no_model():
    agent = Agent()

    with pytest.raises(UserError, match=r'`model` must either be set.+Even when `override\(model=...\)` is customiz'):
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
        assert result.output == '{"make_request":"200"}'


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
    return result.output
"""

    mod = create_module(module_code)

    assert mod.my_agent.name is None
    assert mod.foo() == snapshot('success (no tool calls)')
    assert mod.my_agent.name == 'my_agent'


class TestMultipleToolCalls:
    """Tests for scenarios where multiple tool calls are made in a single response."""

    class OutputType(BaseModel):
        """Result type used by all tests."""

        value: str

    def test_early_strategy_stops_after_first_final_result(self):
        """Test that 'early' strategy stops processing regular tools after first final result."""
        tool_called = []

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('final_result', {'value': 'final'}),
                    ToolCallPart('regular_tool', {'x': 1}),
                    ToolCallPart('another_tool', {'y': 2}),
                ],
            )

        agent = Agent(FunctionModel(return_model), output_type=self.OutputType, end_strategy='early')

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
                    tool_name='final_result',
                    content='Final result processed.',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ToolReturnPart(
                    tool_name='regular_tool',
                    content='Tool not executed - a final result was already processed.',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ToolReturnPart(
                    tool_name='another_tool',
                    content='Tool not executed - a final result was already processed.',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ]
        )

    def test_early_strategy_uses_first_final_result(self):
        """Test that 'early' strategy uses the first final result and ignores subsequent ones."""

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('final_result', {'value': 'first'}),
                    ToolCallPart('final_result', {'value': 'second'}),
                ],
            )

        agent = Agent(FunctionModel(return_model), output_type=self.OutputType, end_strategy='early')
        result = agent.run_sync('test multiple final results')

        # Verify the result came from the first final tool
        assert result.output.value == 'first'

        # Verify we got appropriate tool returns
        assert result.new_messages()[-1].parts == snapshot(
            [
                ToolReturnPart(
                    tool_name='final_result',
                    content='Final result processed.',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ToolReturnPart(
                    tool_name='final_result',
                    content='Output tool not used - a final result was already processed.',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ]
        )

    def test_exhaustive_strategy_executes_all_tools(self):
        """Test that 'exhaustive' strategy executes all tools while using first final result."""
        tool_called: list[str] = []

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('regular_tool', {'x': 42}),
                    ToolCallPart('final_result', {'value': 'first'}),
                    ToolCallPart('another_tool', {'y': 2}),
                    ToolCallPart('final_result', {'value': 'second'}),
                    ToolCallPart('unknown_tool', {'value': '???'}),
                ],
            )

        agent = Agent(FunctionModel(return_model), output_type=self.OutputType, end_strategy='exhaustive')

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
        assert result.output.value == 'first'

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
                        ToolCallPart(tool_name='regular_tool', args={'x': 42}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args={'value': 'first'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='another_tool', args={'y': 2}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args={'value': 'second'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='unknown_tool', args={'value': '???'}, tool_call_id=IsStr()),
                    ],
                    usage=Usage(requests=1, request_tokens=53, response_tokens=23, total_tokens=76),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Output tool not used - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content=42,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool', content=2, tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                        ),
                        RetryPromptPart(
                            content="Unknown tool name: 'unknown_tool'. Available tools: 'final_result', 'regular_tool', 'another_tool'",
                            tool_name='unknown_tool',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        ),
                    ]
                ),
            ]
        )

    def test_early_strategy_with_final_result_in_middle(self):
        """Test that 'early' strategy stops at first final result, regardless of position."""
        tool_called = []

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('regular_tool', {'x': 1}),
                    ToolCallPart('final_result', {'value': 'final'}),
                    ToolCallPart('another_tool', {'y': 2}),
                    ToolCallPart('unknown_tool', {'value': '???'}),
                ],
            )

        agent = Agent(FunctionModel(return_model), output_type=self.OutputType, end_strategy='early')

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
                        ToolCallPart(tool_name='regular_tool', args={'x': 1}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args={'value': 'final'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='another_tool', args={'y': 2}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='unknown_tool', args={'value': '???'}, tool_call_id=IsStr()),
                    ],
                    usage=Usage(requests=1, request_tokens=58, response_tokens=18, total_tokens=76),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        RetryPromptPart(
                            tool_name='unknown_tool',
                            content="Unknown tool name: 'unknown_tool'. Available tools: 'final_result', 'regular_tool', 'another_tool'",
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                    ]
                ),
            ]
        )

    def test_early_strategy_does_not_apply_to_tool_calls_without_final_tool(self):
        """Test that 'early' strategy does not apply to tool calls without final tool."""
        tool_called = []
        agent = Agent(TestModel(), output_type=self.OutputType, end_strategy='early')

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
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('final_result', {'bad_value': 'first'}, tool_call_id='first'),
                    ToolCallPart('final_result', {'value': 'second'}, tool_call_id='second'),
                ],
            )

        agent = Agent(FunctionModel(return_model), output_type=self.OutputType, end_strategy='early')
        result = agent.run_sync('test multiple final results')

        # Verify the result came from the second final tool
        assert result.output.value == 'second'

        # Verify we got appropriate tool returns
        assert result.new_messages()[-1].parts == snapshot(
            [
                RetryPromptPart(
                    content=[
                        {'type': 'missing', 'loc': ('value',), 'msg': 'Field required', 'input': {'bad_value': 'first'}}
                    ],
                    tool_name='final_result',
                    tool_call_id='first',
                    timestamp=IsDatetime(),
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
    assert (await my_agent.run('Hello')).output == IsJson(None)
    assert (await my_agent.run('Hello', model_settings={'temperature': 0.5})).output == IsJson({'temperature': 0.5})

    my_agent = Agent(FunctionModel(return_settings), model_settings={'temperature': 0.1})
    assert (await my_agent.run('Hello')).output == IsJson({'temperature': 0.1})
    assert (await my_agent.run('Hello', model_settings={'temperature': 0.5})).output == IsJson({'temperature': 0.5})


async def test_empty_text_part():
    def return_empty_text(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"response": ["foo", "bar"]}'
        return ModelResponse(
            parts=[
                TextPart(''),
                ToolCallPart(info.output_tools[0].name, args_json),
            ],
        )

    agent = Agent(FunctionModel(return_empty_text), output_type=tuple[str, str])

    result = await agent.run('Hello')
    assert result.output == ('foo', 'bar')


def test_heterogeneous_responses_non_streaming() -> None:
    """Indicates that tool calls are prioritized over text in heterogeneous responses."""

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        parts: list[ModelResponsePart] = []
        if len(messages) == 1:
            parts = [TextPart(content='foo'), ToolCallPart('get_location', {'loc_name': 'London'})]
        else:
            parts = [TextPart(content='final response')]
        return ModelResponse(parts=parts)

    agent = Agent(FunctionModel(return_model))

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')  # pragma: no cover

    result = agent.run_sync('Hello')
    assert result.output == 'final response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    TextPart(content='foo'),
                    ToolCallPart(tool_name='get_location', args={'loc_name': 'London'}, tool_call_id=IsStr()),
                ],
                usage=Usage(requests=1, request_tokens=51, response_tokens=6, total_tokens=57),
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                usage=Usage(requests=1, request_tokens=56, response_tokens=8, total_tokens=64),
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
            assert result.output == 'success (no tool calls)'

    assert messages1 == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=Usage(requests=1, request_tokens=51, response_tokens=4, total_tokens=55),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )
    assert messages1 == messages2


def test_double_capture_run_messages() -> None:
    agent = Agent('test')

    with capture_run_messages() as messages:
        assert messages == []
        result = agent.run_sync('Hello')
        assert result.output == 'success (no tool calls)'
        result2 = agent.run_sync('Hello 2')
        assert result2.output == 'success (no tool calls)'
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=Usage(requests=1, request_tokens=51, response_tokens=4, total_tokens=55),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


def test_capture_run_messages_with_user_exception_does_not_contain_internal_errors() -> None:
    """Test that user exceptions within capture_run_messages context have clean stack traces."""
    agent = Agent('test')

    try:
        with capture_run_messages():
            agent.run_sync('Hello')
            raise ZeroDivisionError('division by zero')
    except Exception as e:
        assert e.__context__ is None


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
                    SystemPromptPart(content='Foobar', part_kind='system-prompt', timestamp=IsNow(tz=timezone.utc)),
                    SystemPromptPart(
                        content=dynamic_value, part_kind='system-prompt', timestamp=IsNow(tz=timezone.utc)
                    ),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc), part_kind='user-prompt'),
                ],
                kind='request',
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)', part_kind='text')],
                usage=Usage(requests=1, request_tokens=53, response_tokens=4, total_tokens=57),
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
                    SystemPromptPart(content='Foobar', part_kind='system-prompt', timestamp=IsNow(tz=timezone.utc)),
                    SystemPromptPart(
                        content='A',  # Remains the same
                        part_kind='system-prompt',
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc), part_kind='user-prompt'),
                ],
                kind='request',
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)', part_kind='text')],
                usage=Usage(requests=1, request_tokens=53, response_tokens=4, total_tokens=57),
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
                usage=Usage(requests=1, request_tokens=54, response_tokens=8, total_tokens=62),
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
                    SystemPromptPart(content='Foobar', part_kind='system-prompt', timestamp=IsNow(tz=timezone.utc)),
                    SystemPromptPart(
                        content=dynamic_value,
                        part_kind='system-prompt',
                        dynamic_ref=func.__qualname__,
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc), part_kind='user-prompt'),
                ],
                kind='request',
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)', part_kind='text')],
                usage=Usage(requests=1, request_tokens=53, response_tokens=4, total_tokens=57),
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
                    SystemPromptPart(content='Foobar', part_kind='system-prompt', timestamp=IsNow(tz=timezone.utc)),
                    SystemPromptPart(
                        content='B',
                        part_kind='system-prompt',
                        dynamic_ref=func.__qualname__,
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc), part_kind='user-prompt'),
                ],
                kind='request',
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)', part_kind='text')],
                usage=Usage(requests=1, request_tokens=53, response_tokens=4, total_tokens=57),
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
                usage=Usage(requests=1, request_tokens=54, response_tokens=8, total_tokens=62),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                kind='response',
            ),
        ]
    )


def test_capture_run_messages_tool_agent() -> None:
    agent_outer = Agent('test')
    agent_inner = Agent(TestModel(custom_output_text='inner agent result'))

    @agent_outer.tool_plain
    async def foobar(x: str) -> str:
        result_ = await agent_inner.run(x)
        return result_.output

    with capture_run_messages() as messages:
        result = agent_outer.run_sync('foobar')

    assert result.output == snapshot('{"foobar":"inner agent result"}')
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='foobar', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=51, response_tokens=5, total_tokens=56),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='foobar',
                        content='inner agent result',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='{"foobar":"inner agent result"}')],
                usage=Usage(requests=1, request_tokens=54, response_tokens=11, total_tokens=65),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


class Bar(BaseModel):
    c: int
    d: str


def test_custom_output_type_sync() -> None:
    agent = Agent('test', output_type=Foo)

    assert agent.run_sync('Hello').output == snapshot(Foo(a=0, b='a'))
    assert agent.run_sync('Hello', output_type=Bar).output == snapshot(Bar(c=0, d='a'))
    assert agent.run_sync('Hello', output_type=str).output == snapshot('success (no tool calls)')
    assert agent.run_sync('Hello', output_type=int).output == snapshot(0)


async def test_custom_output_type_async() -> None:
    agent = Agent('test')

    result = await agent.run('Hello')
    assert result.output == snapshot('success (no tool calls)')

    result = await agent.run('Hello', output_type=Foo)
    assert result.output == snapshot(Foo(a=0, b='a'))
    result = await agent.run('Hello', output_type=int)
    assert result.output == snapshot(0)


def test_custom_output_type_invalid() -> None:
    agent = Agent('test')

    @agent.output_validator
    def validate_output(ctx: RunContext[None], o: Any) -> Any:  # pragma: no cover
        return o

    with pytest.raises(UserError, match='Cannot set a custom run `output_type` when the agent has output validators'):
        agent.run_sync('Hello', output_type=int)


def test_binary_content_all_messages_json():
    agent = Agent('test')

    content = BinaryContent(data=b'Hello', media_type='text/plain')
    result = agent.run_sync(['Hello', content])

    serialized = result.all_messages_json()
    assert json.loads(serialized) == snapshot(
        [
            {
                'parts': [
                    {
                        'content': [
                            'Hello',
                            {
                                'data': 'SGVsbG8=',
                                'media_type': 'text/plain',
                                'vendor_metadata': None,
                                'kind': 'binary',
                                'identifier': None,
                            },
                        ],
                        'timestamp': IsStr(),
                        'part_kind': 'user-prompt',
                    }
                ],
                'instructions': None,
                'kind': 'request',
            },
            {
                'parts': [{'content': 'success (no tool calls)', 'part_kind': 'text'}],
                'usage': {
                    'requests': 1,
                    'request_tokens': 56,
                    'response_tokens': 4,
                    'total_tokens': 60,
                    'details': None,
                },
                'model_name': 'test',
                'vendor_id': None,
                'timestamp': IsStr(),
                'kind': 'response',
                'vendor_details': None,
            },
        ]
    )

    # We also need to be able to round trip the serialized messages.
    messages = ModelMessagesTypeAdapter.validate_json(serialized)
    assert messages == result.all_messages()


def test_tool_return_part_binary_content_serialization():
    """Test that ToolReturnPart can properly serialize BinaryContent."""
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178\x00\x00\x00\x00IEND\xaeB`\x82'
    binary_content = BinaryContent(png_data, media_type='image/png', identifier='image_id_1')

    tool_return = ToolReturnPart(tool_name='test_tool', content=binary_content, tool_call_id='test_call_123')

    response_str = tool_return.model_response_str()

    assert '"kind":"binary"' in response_str
    assert '"media_type":"image/png"' in response_str
    assert '"data":"' in response_str
    assert '"identifier":"image_id_1"' in response_str

    response_obj = tool_return.model_response_object()
    assert response_obj['return_value']['kind'] == 'binary'
    assert response_obj['return_value']['media_type'] == 'image/png'
    assert response_obj['return_value']['identifier'] == 'image_id_1'
    assert 'data' in response_obj['return_value']


def test_tool_returning_binary_content_directly():
    """Test that a tool returning BinaryContent directly works correctly."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('get_image', {})])
        else:
            return ModelResponse(parts=[TextPart('Image received')])

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def get_image() -> BinaryContent:
        """Return a simple image."""
        png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178\x00\x00\x00\x00IEND\xaeB`\x82'
        return BinaryContent(png_data, media_type='image/png')

    # This should work without the serialization error
    result = agent.run_sync('Get an image')
    assert result.output == 'Image received'


def test_tool_returning_binary_content_with_identifier():
    """Test that a tool returning BinaryContent directly works correctly."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('get_image', {})])
        else:
            return ModelResponse(parts=[TextPart('Image received')])

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def get_image() -> BinaryContent:
        """Return a simple image."""
        png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178\x00\x00\x00\x00IEND\xaeB`\x82'
        return BinaryContent(png_data, media_type='image/png', identifier='image_id_1')

    # This should work without the serialization error
    result = agent.run_sync('Get an image')
    assert result.all_messages()[2] == snapshot(
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='get_image',
                    content='See file image_id_1',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
                UserPromptPart(
                    content=[
                        'This is file image_id_1:',
                        BinaryContent(
                            data=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178\x00\x00\x00\x00IEND\xaeB`\x82',
                            media_type='image/png',
                            identifier='image_id_1',
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ]
        )
    )


def test_instructions_raise_error_when_system_prompt_is_set():
    agent = Agent('test', instructions='An instructions!')

    @agent.system_prompt
    def system_prompt() -> str:
        return 'A system prompt!'

    result = agent.run_sync('Hello')
    assert result.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[
                SystemPromptPart(content='A system prompt!', timestamp=IsNow(tz=timezone.utc)),
                UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
            ],
            instructions='An instructions!',
        )
    )


def test_instructions_raise_error_when_instructions_is_set():
    agent = Agent('test', system_prompt='A system prompt!')

    @agent.instructions
    def instructions() -> str:
        return 'An instructions!'

    @agent.instructions
    def empty_instructions() -> str:
        return ''

    result = agent.run_sync('Hello')
    assert result.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[
                SystemPromptPart(content='A system prompt!', timestamp=IsNow(tz=timezone.utc)),
                UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
            ],
            instructions='An instructions!',
        )
    )


def test_instructions_both_instructions_and_system_prompt_are_set():
    agent = Agent('test', instructions='An instructions!', system_prompt='A system prompt!')
    result = agent.run_sync('Hello')
    assert result.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[
                SystemPromptPart(content='A system prompt!', timestamp=IsNow(tz=timezone.utc)),
                UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
            ],
            instructions='An instructions!',
        )
    )


def test_instructions_decorator_without_parenthesis():
    agent = Agent('test')

    @agent.instructions
    def instructions() -> str:
        return 'You are a helpful assistant.'

    result = agent.run_sync('Hello')
    assert result.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
            instructions='You are a helpful assistant.',
        )
    )


def test_instructions_decorator_with_parenthesis():
    agent = Agent('test')

    @agent.instructions()
    def instructions_2() -> str:
        return 'You are a helpful assistant.'

    result = agent.run_sync('Hello')
    assert result.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
            instructions='You are a helpful assistant.',
        )
    )


def test_instructions_with_message_history():
    agent = Agent('test', instructions='You are a helpful assistant.')
    result = agent.run_sync(
        'Hello',
        message_history=[ModelRequest(parts=[SystemPromptPart(content='You are a helpful assistant')])],
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[SystemPromptPart(content='You are a helpful assistant', timestamp=IsNow(tz=timezone.utc))]
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=Usage(requests=1, request_tokens=56, response_tokens=4, total_tokens=60),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


def test_instructions_parameter_with_sequence():
    def instructions() -> str:
        return 'You are a potato.'

    def empty_instructions() -> str:
        return ''

    agent = Agent('test', instructions=('You are a helpful assistant.', empty_instructions, instructions))
    result = agent.run_sync('Hello')
    assert result.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
            instructions="""\
You are a helpful assistant.

You are a potato.\
""",
        )
    )


def test_empty_final_response():
    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[TextPart('foo'), ToolCallPart('my_tool', {'x': 1})])
        elif len(messages) == 3:
            return ModelResponse(parts=[TextPart('bar'), ToolCallPart('my_tool', {'x': 2})])
        else:
            return ModelResponse(parts=[])

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def my_tool(x: int) -> int:
        return x * 2

    result = agent.run_sync('Hello')
    assert result.output == 'bar'

    assert result.new_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    TextPart(content='foo'),
                    ToolCallPart(tool_name='my_tool', args={'x': 1}, tool_call_id=IsStr()),
                ],
                usage=Usage(requests=1, request_tokens=51, response_tokens=5, total_tokens=56),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_tool', content=2, tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(content='bar'),
                    ToolCallPart(tool_name='my_tool', args={'x': 2}, tool_call_id=IsStr()),
                ],
                usage=Usage(requests=1, request_tokens=52, response_tokens=10, total_tokens=62),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_tool', content=4, tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[],
                usage=Usage(requests=1, request_tokens=53, response_tokens=10, total_tokens=63),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


def test_agent_run_result_serialization() -> None:
    agent = Agent('test', output_type=Foo)
    result = agent.run_sync('Hello')

    # Check that dump_json doesn't raise an error
    adapter = TypeAdapter(AgentRunResult[Foo])
    serialized_data = adapter.dump_json(result)

    # Check that we can load the data back
    deserialized_result = adapter.validate_json(serialized_data)
    assert deserialized_result == result


def test_agent_repr() -> None:
    agent = Agent()
    assert repr(agent) == snapshot(
        "Agent(model=None, name=None, end_strategy='early', model_settings=None, output_type=<class 'str'>, instrument=None)"
    )


def test_tool_call_with_validation_value_error_serializable():
    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('foo_tool', {'bar': 0})])
        elif len(messages) == 3:
            return ModelResponse(parts=[ToolCallPart('foo_tool', {'bar': 1})])
        else:
            return ModelResponse(parts=[TextPart('Tool returned 1')])

    agent = Agent(FunctionModel(llm))

    class Foo(BaseModel):
        bar: int

        @field_validator('bar')
        def validate_bar(cls, v: int) -> int:
            if v == 0:
                raise ValueError('bar cannot be 0')
            return v

    @agent.tool_plain
    def foo_tool(foo: Foo) -> int:
        return foo.bar

    result = agent.run_sync('Hello')
    assert json.loads(result.all_messages_json())[2] == snapshot(
        {
            'parts': [
                {
                    'content': [
                        {'type': 'value_error', 'loc': ['bar'], 'msg': 'Value error, bar cannot be 0', 'input': 0}
                    ],
                    'tool_name': 'foo_tool',
                    'tool_call_id': IsStr(),
                    'timestamp': IsStr(),
                    'part_kind': 'retry-prompt',
                }
            ],
            'instructions': None,
            'kind': 'request',
        }
    )


def test_unsupported_output_mode():
    class Foo(BaseModel):
        bar: str

    def hello(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('hello')])  # pragma: no cover

    model = FunctionModel(hello, profile=ModelProfile(supports_tools=False, supports_json_schema_output=False))

    agent = Agent(model, output_type=NativeOutput(Foo))

    with pytest.raises(UserError, match='Native structured output is not supported by the model.'):
        agent.run_sync('Hello')

    agent = Agent(model, output_type=ToolOutput(Foo))

    with pytest.raises(UserError, match='Output tools are not supported by the model.'):
        agent.run_sync('Hello')


def test_multimodal_tool_response():
    """Test ToolReturn with custom content and tool return."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[TextPart('Starting analysis'), ToolCallPart('analyze_data', {})])
        else:
            return ModelResponse(
                parts=[
                    TextPart('Analysis completed'),
                ]
            )

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def analyze_data() -> ToolReturn:
        return ToolReturn(
            return_value='Data analysis completed successfully',
            content=[
                'Here are the analysis results:',
                ImageUrl('https://example.com/chart.jpg'),
                'The chart shows positive trends.',
            ],
            metadata={'foo': 'bar'},
        )

    result = agent.run_sync('Please analyze the data')

    # Verify final output
    assert result.output == 'Analysis completed'

    # Verify message history contains the expected parts

    # Verify the complete message structure using snapshot
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Please analyze the data', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    TextPart(content='Starting analysis'),
                    ToolCallPart(
                        tool_name='analyze_data',
                        args={},
                        tool_call_id=IsStr(),
                    ),
                ],
                usage=Usage(requests=1, request_tokens=54, response_tokens=4, total_tokens=58),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='analyze_data',
                        content='Data analysis completed successfully',
                        tool_call_id=IsStr(),
                        metadata={'foo': 'bar'},
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                    UserPromptPart(
                        content=[
                            'Here are the analysis results:',
                            ImageUrl(url='https://example.com/chart.jpg'),
                            'The chart shows positive trends.',
                        ],
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='Analysis completed')],
                usage=Usage(requests=1, request_tokens=70, response_tokens=6, total_tokens=76),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


def test_plain_tool_response():
    """Test ToolReturn with custom content and tool return."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[TextPart('Starting analysis'), ToolCallPart('analyze_data', {})])
        else:
            return ModelResponse(
                parts=[
                    TextPart('Analysis completed'),
                ]
            )

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def analyze_data() -> ToolReturn:
        return ToolReturn(
            return_value='Data analysis completed successfully',
            metadata={'foo': 'bar'},
        )

    result = agent.run_sync('Please analyze the data')

    # Verify final output
    assert result.output == 'Analysis completed'

    # Verify message history contains the expected parts

    # Verify the complete message structure using snapshot
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Please analyze the data', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    TextPart(content='Starting analysis'),
                    ToolCallPart(
                        tool_name='analyze_data',
                        args={},
                        tool_call_id=IsStr(),
                    ),
                ],
                usage=Usage(requests=1, request_tokens=54, response_tokens=4, total_tokens=58),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='analyze_data',
                        content='Data analysis completed successfully',
                        tool_call_id=IsStr(),
                        metadata={'foo': 'bar'},
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='Analysis completed')],
                usage=Usage(requests=1, request_tokens=58, response_tokens=6, total_tokens=64),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


def test_many_multimodal_tool_response():
    """Test ToolReturn with custom content and tool return."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[TextPart('Starting analysis'), ToolCallPart('analyze_data', {})])
        else:
            return ModelResponse(  # pragma: no cover
                parts=[
                    TextPart('Analysis completed'),
                ]
            )

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def analyze_data() -> list[Any]:
        return [
            ToolReturn(
                return_value='Data analysis completed successfully',
                content=[
                    'Here are the analysis results:',
                    ImageUrl('https://example.com/chart.jpg'),
                    'The chart shows positive trends.',
                ],
                metadata={'foo': 'bar'},
            ),
            'Something else',
        ]

    with pytest.raises(
        UserError,
        match="The return value of tool 'analyze_data' contains invalid nested `ToolReturn` objects. `ToolReturn` should be used directly.",
    ):
        agent.run_sync('Please analyze the data')


def test_multimodal_tool_response_nested():
    """Test ToolReturn with custom content and tool return."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[TextPart('Starting analysis'), ToolCallPart('analyze_data', {})])
        else:
            return ModelResponse(  # pragma: no cover
                parts=[
                    TextPart('Analysis completed'),
                ]
            )

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def analyze_data() -> ToolReturn:
        return ToolReturn(
            return_value=ImageUrl('https://example.com/chart.jpg'),
            content=[
                'Here are the analysis results:',
                ImageUrl('https://example.com/chart.jpg'),
                'The chart shows positive trends.',
            ],
            metadata={'foo': 'bar'},
        )

    with pytest.raises(
        UserError,
        match="The `return_value` of tool 'analyze_data' contains invalid nested `MultiModalContentTypes` objects. Please use `content` instead.",
    ):
        agent.run_sync('Please analyze the data')


def test_deprecated_kwargs_validation_agent_init():
    """Test that invalid kwargs raise UserError in Agent constructor."""
    with pytest.raises(UserError, match='Unknown keyword arguments: `usage_limits`'):
        Agent('test', usage_limits='invalid')  # type: ignore[call-arg]

    with pytest.raises(UserError, match='Unknown keyword arguments: `invalid_kwarg`'):
        Agent('test', invalid_kwarg='value')  # type: ignore[call-arg]

    with pytest.raises(UserError, match='Unknown keyword arguments: `foo`, `bar`'):
        Agent('test', foo='value1', bar='value2')  # type: ignore[call-arg]


def test_deprecated_kwargs_validation_agent_run():
    """Test that invalid kwargs raise UserError in Agent.run method."""
    agent = Agent('test')

    with pytest.raises(UserError, match='Unknown keyword arguments: `invalid_kwarg`'):
        agent.run_sync('test', invalid_kwarg='value')  # type: ignore[call-arg]

    with pytest.raises(UserError, match='Unknown keyword arguments: `foo`, `bar`'):
        agent.run_sync('test', foo='value1', bar='value2')  # type: ignore[call-arg]


def test_deprecated_kwargs_still_work():
    """Test that valid deprecated kwargs still work with warnings."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        agent = Agent('test', result_type=str)  # type: ignore[call-arg]
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert '`result_type` is deprecated' in str(w[0].message)
        assert agent.output_type is str

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        agent = Agent('test', result_tool_name='test_tool')  # type: ignore[call-arg]
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert '`result_tool_name` is deprecated' in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        agent = Agent('test', result_tool_description='test description')  # type: ignore[call-arg]
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert '`result_tool_description` is deprecated' in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        agent = Agent('test', result_retries=3)  # type: ignore[call-arg]
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert '`result_retries` is deprecated' in str(w[0].message)

    try:
        from pydantic_ai.mcp import MCPServerStdio

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            agent = Agent('test', mcp_servers=[MCPServerStdio('python', ['-m', 'tests.mcp_server'])])  # type: ignore[call-arg]
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert '`mcp_servers` is deprecated' in str(w[0].message)
    except ImportError:
        pass


def test_deprecated_kwargs_mixed_valid_invalid():
    """Test that mix of valid deprecated and invalid kwargs raises error for invalid ones."""
    import warnings

    with pytest.raises(UserError, match='Unknown keyword arguments: `usage_limits`'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)  # Ignore the deprecation warning for result_type
            Agent('test', result_type=str, usage_limits='invalid')  # type: ignore[call-arg]

    with pytest.raises(UserError, match='Unknown keyword arguments: `foo`, `bar`'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)  # Ignore the deprecation warning for result_tool_name
            Agent('test', result_tool_name='test', foo='value1', bar='value2')  # type: ignore[call-arg]


def test_override_toolsets():
    foo_toolset = FunctionToolset()

    @foo_toolset.tool
    def foo() -> str:
        return 'Hello from foo'

    available_tools: list[list[str]] = []

    async def prepare_tools(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        nonlocal available_tools
        available_tools.append([tool_def.name for tool_def in tool_defs])
        return tool_defs

    agent = Agent('test', toolsets=[foo_toolset], prepare_tools=prepare_tools)

    @agent.tool_plain
    def baz() -> str:
        return 'Hello from baz'

    result = agent.run_sync('Hello')
    assert available_tools[-1] == snapshot(['baz', 'foo'])
    assert result.output == snapshot('{"baz":"Hello from baz","foo":"Hello from foo"}')

    bar_toolset = FunctionToolset()

    @bar_toolset.tool
    def bar() -> str:
        return 'Hello from bar'

    with agent.override(toolsets=[bar_toolset]):
        result = agent.run_sync('Hello')
    assert available_tools[-1] == snapshot(['baz', 'bar'])
    assert result.output == snapshot('{"baz":"Hello from baz","bar":"Hello from bar"}')

    with agent.override(toolsets=[]):
        result = agent.run_sync('Hello')
    assert available_tools[-1] == snapshot(['baz'])
    assert result.output == snapshot('{"baz":"Hello from baz"}')

    result = agent.run_sync('Hello', toolsets=[bar_toolset])
    assert available_tools[-1] == snapshot(['baz', 'foo', 'bar'])
    assert result.output == snapshot('{"baz":"Hello from baz","foo":"Hello from foo","bar":"Hello from bar"}')

    with agent.override(toolsets=[]):
        result = agent.run_sync('Hello', toolsets=[bar_toolset])
    assert available_tools[-1] == snapshot(['baz'])
    assert result.output == snapshot('{"baz":"Hello from baz"}')


def test_adding_tools_during_run():
    toolset = FunctionToolset()

    def foo() -> str:
        return 'Hello from foo'

    @toolset.tool
    def add_foo_tool() -> str:
        toolset.add_function(foo)
        return 'foo tool added'

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('add_foo_tool')])
        elif len(messages) == 3:
            return ModelResponse(parts=[ToolCallPart('foo')])
        else:
            return ModelResponse(parts=[TextPart('Done')])

    agent = Agent(FunctionModel(respond), toolsets=[toolset])
    result = agent.run_sync('Add the foo tool and run it')
    assert result.output == snapshot('Done')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Add the foo tool and run it',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='add_foo_tool', tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=57, response_tokens=2, total_tokens=59),
                model_name='function:respond:',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='add_foo_tool',
                        content='foo tool added',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foo', tool_call_id=IsStr())],
                usage=Usage(requests=1, request_tokens=60, response_tokens=4, total_tokens=64),
                model_name='function:respond:',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='foo',
                        content='Hello from foo',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='Done')],
                usage=Usage(requests=1, request_tokens=63, response_tokens=5, total_tokens=68),
                model_name='function:respond:',
                timestamp=IsDatetime(),
            ),
        ]
    )


def test_prepare_output_tools():
    @dataclass
    class AgentDeps:
        plan_presented: bool = False

    async def present_plan(ctx: RunContext[AgentDeps], plan: str) -> str:
        """
        Present the plan to the user.
        """
        ctx.deps.plan_presented = True
        return plan

    async def run_sql(ctx: RunContext[AgentDeps], purpose: str, query: str) -> str:
        """
        Run an SQL query.
        """
        return 'SQL query executed successfully'

    async def only_if_plan_presented(
        ctx: RunContext[AgentDeps], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return tool_defs if ctx.deps.plan_presented else []

    agent = Agent(
        model='test',
        deps_type=AgentDeps,
        tools=[present_plan],
        output_type=[ToolOutput(run_sql, name='run_sql')],
        prepare_output_tools=only_if_plan_presented,
    )

    result = agent.run_sync('Hello', deps=AgentDeps())
    assert result.output == snapshot('SQL query executed successfully')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='present_plan',
                        args={'plan': 'a'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=Usage(requests=1, request_tokens=51, response_tokens=5, total_tokens=56),
                model_name='test',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='present_plan',
                        content='a',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='run_sql',
                        args={'purpose': 'a', 'query': 'a'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=Usage(requests=1, request_tokens=52, response_tokens=12, total_tokens=64),
                model_name='test',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='run_sql',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )


async def test_context_manager():
    try:
        from pydantic_ai.mcp import MCPServerStdio
    except ImportError:  # pragma: lax no cover
        pytest.skip('mcp is not installed')

    server1 = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    server2 = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    toolset = CombinedToolset([server1, PrefixedToolset(server2, 'prefix')])
    agent = Agent('test', toolsets=[toolset])

    async with agent:
        assert server1.is_running
        assert server2.is_running

        async with agent:
            assert server1.is_running
            assert server2.is_running


def test_set_mcp_sampling_model():
    try:
        from pydantic_ai.mcp import MCPServerStdio
    except ImportError:  # pragma: lax no cover
        pytest.skip('mcp is not installed')

    test_model = TestModel()
    server1 = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    server2 = MCPServerStdio('python', ['-m', 'tests.mcp_server'], sampling_model=test_model)
    toolset = CombinedToolset([server1, PrefixedToolset(server2, 'prefix')])
    agent = Agent(None, toolsets=[toolset])

    with pytest.raises(UserError, match='No sampling model provided and no model set on the agent.'):
        agent.set_mcp_sampling_model()
    assert server1.sampling_model is None
    assert server2.sampling_model is test_model

    agent.model = test_model
    agent.set_mcp_sampling_model()
    assert server1.sampling_model is test_model
    assert server2.sampling_model is test_model

    function_model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('Hello')]))
    with agent.override(model=function_model):
        agent.set_mcp_sampling_model()
        assert server1.sampling_model is function_model
        assert server2.sampling_model is function_model

    function_model2 = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('Goodbye')]))
    agent.set_mcp_sampling_model(function_model2)
    assert server1.sampling_model is function_model2
    assert server2.sampling_model is function_model2
