from typing import Union

from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import (
    ArgsJson,
    LLMMessage,
    LLMResponse,
    LLMToolCalls,
    Message,
    RetryPrompt,
    ToolCall,
    UserPrompt,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from tests.conftest import IsNow


def test_result_tuple():
    def return_tuple(_: list[Message], info: AgentInfo) -> LLMMessage:
        assert info.result_tool is not None
        args_json = '{"response": ["foo", "bar"]}'
        return LLMToolCalls(calls=[ToolCall.from_json(info.result_tool.name, args_json)])

    agent = Agent(FunctionModel(return_tuple), deps=None, result_type=tuple[str, str])

    result = agent.run_sync('Hello')
    assert result.response == ('foo', 'bar')


class Foo(BaseModel):
    a: int
    b: str


def test_result_pydantic_model():
    def return_model(_: list[Message], info: AgentInfo) -> LLMMessage:
        assert info.result_tool is not None
        args_json = '{"a": 1, "b": "foo"}'
        return LLMToolCalls(calls=[ToolCall.from_json(info.result_tool.name, args_json)])

    agent = Agent(FunctionModel(return_model), deps=None, result_type=Foo)

    result = agent.run_sync('Hello')
    assert isinstance(result.response, Foo)
    assert result.response.model_dump() == {'a': 1, 'b': 'foo'}


def test_result_pydantic_model_retry():
    def return_model(messages: list[Message], info: AgentInfo) -> LLMMessage:
        assert info.result_tool is not None
        if len(messages) == 1:
            args_json = '{"a": "wrong", "b": "foo"}'
        else:
            args_json = '{"a": 42, "b": "foo"}'
        return LLMToolCalls(calls=[ToolCall.from_json(info.result_tool.name, args_json)])

    agent = Agent(FunctionModel(return_model), deps=None, result_type=Foo)

    result = agent.run_sync('Hello')
    assert isinstance(result.response, Foo)
    assert result.response.model_dump() == {'a': 42, 'b': 'foo'}
    assert result.message_history == snapshot(
        [
            UserPrompt(content='Hello', timestamp=IsNow()),
            LLMToolCalls(
                calls=[ToolCall.from_json('final_result', '{"a": "wrong", "b": "foo"}')],
                timestamp=IsNow(),
            ),
            RetryPrompt(
                tool_name='final_result',
                content=[
                    {
                        'type': 'int_parsing',
                        'loc': ('a',),
                        'msg': 'Input should be a valid integer, unable to parse string as an integer',
                        'input': 'wrong',
                    }
                ],
                timestamp=IsNow(),
            ),
            LLMToolCalls(
                calls=[ToolCall.from_json('final_result', '{"a": 42, "b": "foo"}')],
                timestamp=IsNow(),
            ),
        ]
    )


def test_result_validator():
    def return_model(messages: list[Message], info: AgentInfo) -> LLMMessage:
        assert info.result_tool is not None
        if len(messages) == 1:
            args_json = '{"a": 41, "b": "foo"}'
        else:
            args_json = '{"a": 42, "b": "foo"}'
        return LLMToolCalls(calls=[ToolCall.from_json(info.result_tool.name, args_json)])

    agent = Agent(FunctionModel(return_model), deps=None, result_type=Foo)

    @agent.result_validator
    def validate_result(r: Foo) -> Foo:
        if r.a == 42:
            return r
        else:
            raise ModelRetry('"a" should be 42')

    result = agent.run_sync('Hello')
    assert isinstance(result.response, Foo)
    assert result.response.model_dump() == {'a': 42, 'b': 'foo'}
    assert result.message_history == snapshot(
        [
            UserPrompt(content='Hello', timestamp=IsNow()),
            LLMToolCalls(calls=[ToolCall.from_json('final_result', '{"a": 41, "b": "foo"}')], timestamp=IsNow()),
            RetryPrompt(tool_name='final_result', content='"a" should be 42', timestamp=IsNow()),
            LLMToolCalls(calls=[ToolCall.from_json('final_result', '{"a": 42, "b": "foo"}')], timestamp=IsNow()),
        ]
    )


def test_plain_response():
    call_index = 0

    def return_tuple(_: list[Message], info: AgentInfo) -> LLMMessage:
        nonlocal call_index

        assert info.result_tool is not None
        call_index += 1
        if call_index == 1:
            return LLMResponse(content='hello')
        else:
            args_json = '{"response": ["foo", "bar"]}'
            return LLMToolCalls(calls=[ToolCall.from_json(info.result_tool.name, args_json)])

    agent = Agent(FunctionModel(return_tuple), deps=None, result_type=tuple[str, str])

    result = agent.run_sync('Hello')
    assert result.response == ('foo', 'bar')
    assert call_index == 2
    assert result.message_history == snapshot(
        [
            UserPrompt(content='Hello', timestamp=IsNow()),
            LLMResponse(content='hello', timestamp=IsNow()),
            RetryPrompt(
                content='Plain text responses are not permitted, please call one of the functions instead.',
                timestamp=IsNow(),
            ),
            LLMToolCalls(
                calls=[ToolCall(tool_name='final_result', args=ArgsJson(args_json='{"response": ["foo", "bar"]}'))],
                timestamp=IsNow(),
            ),
        ]
    )


def test_response_tuple():
    m = TestModel()

    agent = Agent(m, deps=None, result_type=tuple[str, str])
    assert agent._result_schema.allow_text_result is False  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]

    result = agent.run_sync('Hello')
    assert result.response == snapshot(('b', 'b'))

    assert m.agent_model_retrievers == snapshot({})
    assert m.agent_model_allow_text_result is False

    assert m.agent_model_result_tool is not None

    # to match the protocol, we just extract the attributes we care about
    fields = 'name', 'description', 'json_schema', 'outer_typed_dict_key'
    agent_model_result_tool = {f: getattr(m.agent_model_result_tool, f) for f in fields}
    assert agent_model_result_tool == snapshot(
        {
            'name': 'final_result',
            'description': 'The final response which ends this conversation',
            'json_schema': {
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
            'outer_typed_dict_key': 'response',
        }
    )


def test_response_union_allow_str():
    m = TestModel()
    agent: Agent[None, Union[str, Foo]] = Agent(
        m,
        result_type=Union[str, Foo],  # pyright: ignore[reportArgumentType]
    )
    assert agent._result_schema.allow_text_result is True  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]

    result = agent.run_sync('Hello')
    assert result.response == snapshot('{}')

    assert m.agent_model_retrievers == snapshot({})
    assert m.agent_model_allow_text_result is True

    assert m.agent_model_result_tool is not None

    # to match the protocol, we just extract the attributes we care about
    fields = 'name', 'description', 'json_schema', 'outer_typed_dict_key'
    agent_model_result_tool = {f: getattr(m.agent_model_result_tool, f) for f in fields}
    assert agent_model_result_tool == snapshot(
        {
            'name': 'final_result',
            'description': 'The final response which ends this conversation',
            'json_schema': {
                '$defs': {
                    'Foo': {
                        'properties': {
                            'a': {'title': 'A', 'type': 'integer'},
                            'b': {'title': 'B', 'type': 'string'},
                        },
                        'required': ['a', 'b'],
                        'title': 'Foo',
                        'type': 'object',
                    }
                },
                'properties': {'response': {'$ref': '#/$defs/Foo'}},
                'required': ['response'],
                'type': 'object',
            },
            'outer_typed_dict_key': 'response',
        }
    )


class Bar(BaseModel):
    b: str


def test_response_multiple_return_tools():
    m = TestModel()
    agent: Agent[None, Union[Foo, Bar]] = Agent(
        m,
        result_type=Union[Foo, Bar],  # pyright: ignore[reportArgumentType]
    )

    result = agent.run_sync('Hello')
    assert result.response == Foo(a=1, b='b')

    assert m.agent_model_retrievers == snapshot({})
    assert m.agent_model_allow_text_result is False

    assert m.agent_model_result_tool is not None

    # to match the protocol, we just extract the attributes we care about
    fields = 'name', 'description', 'json_schema', 'outer_typed_dict_key'
    agent_model_result_tool = {f: getattr(m.agent_model_result_tool, f) for f in fields}
    assert agent_model_result_tool == snapshot(
        {
            'name': 'final_result',
            'description': 'The final response which ends this conversation',
            'json_schema': {
                '$defs': {
                    'Bar': {
                        'properties': {'b': {'title': 'B', 'type': 'string'}},
                        'required': ['b'],
                        'title': 'Bar',
                        'type': 'object',
                    },
                    'Foo': {
                        'properties': {
                            'a': {'title': 'A', 'type': 'integer'},
                            'b': {'title': 'B', 'type': 'string'},
                        },
                        'required': ['a', 'b'],
                        'title': 'Foo',
                        'type': 'object',
                    },
                },
                'properties': {
                    'response': {
                        'anyOf': [{'$ref': '#/$defs/Foo'}, {'$ref': '#/$defs/Bar'}],
                        'title': 'Response',
                    }
                },
                'required': ['response'],
                'type': 'object',
            },
            'outer_typed_dict_key': 'response',
        }
    )
