from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, Retry
from pydantic_ai.messages import LLMMessage, LLMToolCalls, Message, ToolCall, ToolRetry, UserPrompt
from pydantic_ai.models.function import AgentInfo, FunctionModel
from tests.utils import IsNow


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
            ToolRetry(
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
            raise Retry('"a" should be 42')

    result = agent.run_sync('Hello')
    assert isinstance(result.response, Foo)
    assert result.response.model_dump() == {'a': 42, 'b': 'foo'}
    assert result.message_history == snapshot(
        [
            UserPrompt(content='Hello', timestamp=IsNow()),
            LLMToolCalls(calls=[ToolCall.from_json('final_result', '{"a": 41, "b": "foo"}')], timestamp=IsNow()),
            ToolRetry(tool_name='final_result', content='"a" should be 42', timestamp=IsNow()),
            LLMToolCalls(calls=[ToolCall.from_json('final_result', '{"a": 42, "b": "foo"}')], timestamp=IsNow()),
        ]
    )
