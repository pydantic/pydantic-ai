import sys
from datetime import timezone
from typing import Any, Callable, Union

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, CallContext, ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    ArgsJson,
    ArgsObject,
    Message,
    ModelAnyResponse,
    ModelStructuredResponse,
    ModelTextResponse,
    RetryPrompt,
    SystemPrompt,
    ToolCall,
    ToolReturn,
    UserPrompt,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import Cost, RunResult
from tests.conftest import IsNow, TestEnv


def test_result_tuple():
    def return_tuple(_: list[Message], info: AgentInfo) -> ModelAnyResponse:
        assert info.result_tools is not None
        args_json = '{"response": ["foo", "bar"]}'
        return ModelStructuredResponse(calls=[ToolCall.from_json(info.result_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_tuple), result_type=tuple[str, str])

    result = agent.run_sync('Hello')
    assert result.data == ('foo', 'bar')


class Foo(BaseModel):
    a: int
    b: str


def test_result_pydantic_model():
    def return_model(_: list[Message], info: AgentInfo) -> ModelAnyResponse:
        assert info.result_tools is not None
        args_json = '{"a": 1, "b": "foo"}'
        return ModelStructuredResponse(calls=[ToolCall.from_json(info.result_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_model), result_type=Foo)

    result = agent.run_sync('Hello')
    assert isinstance(result.data, Foo)
    assert result.data.model_dump() == {'a': 1, 'b': 'foo'}


def test_result_pydantic_model_retry():
    def return_model(messages: list[Message], info: AgentInfo) -> ModelAnyResponse:
        assert info.result_tools is not None
        if len(messages) == 1:
            args_json = '{"a": "wrong", "b": "foo"}'
        else:
            args_json = '{"a": 42, "b": "foo"}'
        return ModelStructuredResponse(calls=[ToolCall.from_json(info.result_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_model), result_type=Foo)

    result = agent.run_sync('Hello')
    assert isinstance(result.data, Foo)
    assert result.data.model_dump() == {'a': 42, 'b': 'foo'}
    assert result.all_messages() == snapshot(
        [
            UserPrompt(content='Hello', timestamp=IsNow(tz=timezone.utc)),
            ModelStructuredResponse(
                calls=[ToolCall.from_json('final_result', '{"a": "wrong", "b": "foo"}')],
                timestamp=IsNow(tz=timezone.utc),
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
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelStructuredResponse(
                calls=[ToolCall.from_json('final_result', '{"a": 42, "b": "foo"}')],
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )
    assert result.all_messages_json().startswith(b'[{"content":"Hello"')


def test_result_validator():
    def return_model(messages: list[Message], info: AgentInfo) -> ModelAnyResponse:
        assert info.result_tools is not None
        if len(messages) == 1:
            args_json = '{"a": 41, "b": "foo"}'
        else:
            args_json = '{"a": 42, "b": "foo"}'
        return ModelStructuredResponse(calls=[ToolCall.from_json(info.result_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_model), result_type=Foo)

    @agent.result_validator
    def validate_result(ctx: CallContext[None], r: Foo) -> Foo:
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
            UserPrompt(content='Hello', timestamp=IsNow(tz=timezone.utc)),
            ModelStructuredResponse(
                calls=[ToolCall.from_json('final_result', '{"a": 41, "b": "foo"}')], timestamp=IsNow(tz=timezone.utc)
            ),
            RetryPrompt(tool_name='final_result', content='"a" should be 42', timestamp=IsNow(tz=timezone.utc)),
            ModelStructuredResponse(
                calls=[ToolCall.from_json('final_result', '{"a": 42, "b": "foo"}')], timestamp=IsNow(tz=timezone.utc)
            ),
        ]
    )


def test_plain_response():
    call_index = 0

    def return_tuple(_: list[Message], info: AgentInfo) -> ModelAnyResponse:
        nonlocal call_index

        assert info.result_tools is not None
        call_index += 1
        if call_index == 1:
            return ModelTextResponse(content='hello')
        else:
            args_json = '{"response": ["foo", "bar"]}'
            return ModelStructuredResponse(calls=[ToolCall.from_json(info.result_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_tuple), result_type=tuple[str, str])

    result = agent.run_sync('Hello')
    assert result.data == ('foo', 'bar')
    assert call_index == 2
    assert result.all_messages() == snapshot(
        [
            UserPrompt(content='Hello', timestamp=IsNow(tz=timezone.utc)),
            ModelTextResponse(content='hello', timestamp=IsNow(tz=timezone.utc)),
            RetryPrompt(
                content='Plain text responses are not permitted, please call one of the functions instead.',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelStructuredResponse(
                calls=[ToolCall(tool_name='final_result', args=ArgsJson(args_json='{"response": ["foo", "bar"]}'))],
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


def test_response_tuple():
    m = TestModel()

    agent = Agent(m, result_type=tuple[str, str])
    assert agent._result_schema.allow_text_result is False  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]

    result = agent.run_sync('Hello')
    assert result.data == snapshot(('a', 'a'))

    assert m.agent_model_retrievers == snapshot({})
    assert m.agent_model_allow_text_result is False

    assert m.agent_model_result_tools is not None
    assert len(m.agent_model_result_tools) == 1

    # to match the protocol, we just extract the attributes we care about
    fields = 'name', 'description', 'json_schema', 'outer_typed_dict_key'
    agent_model_result_tool = {f: getattr(m.agent_model_result_tools[0], f) for f in fields}
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


@pytest.mark.parametrize(
    'input_union_callable',
    [lambda: Union[str, Foo], lambda: Union[Foo, str], lambda: str | Foo, lambda: Foo | str],
    ids=['Union[str, Foo]', 'Union[Foo, str]', 'str | Foo', 'Foo | str'],
)
def test_response_union_allow_str(input_union_callable: Callable[[], Any]):
    try:
        union = input_union_callable()
    except TypeError:
        raise pytest.skip('Python version does not support `|` syntax for unions')

    m = TestModel()
    agent: Agent[None, Union[str, Foo]] = Agent(m, result_type=union)

    got_tool_call_name = 'unset'

    @agent.result_validator
    def validate_result(ctx: CallContext[None], r: Any) -> Any:
        nonlocal got_tool_call_name
        got_tool_call_name = ctx.tool_name
        return r

    assert agent._result_schema.allow_text_result is True  # pyright: ignore[reportPrivateUsage,reportOptionalMemberAccess]

    result = agent.run_sync('Hello')
    assert result.data == snapshot('success (no retriever calls)')
    assert got_tool_call_name == snapshot(None)

    assert m.agent_model_retrievers == snapshot({})
    assert m.agent_model_allow_text_result is True

    assert m.agent_model_result_tools is not None
    assert len(m.agent_model_result_tools) == 1

    # to match the protocol, we just extract the attributes we care about
    fields = 'name', 'description', 'json_schema', 'outer_typed_dict_key'
    agent_model_result_tool = {f: getattr(m.agent_model_result_tools[0], f) for f in fields}
    assert agent_model_result_tool == snapshot(
        {
            'name': 'final_result',
            'description': 'The final response which ends this conversation',
            'json_schema': {
                'properties': {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'string'}},
                'title': 'Foo',
                'required': ['a', 'b'],
                'type': 'object',
            },
            'outer_typed_dict_key': None,
        }
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
    def validate_result(ctx: CallContext[None], r: Any) -> Any:
        nonlocal got_tool_call_name
        got_tool_call_name = ctx.tool_name
        return r

    result = agent.run_sync('Hello')
    assert result.data == mod.Foo(a=0, b='a')
    assert got_tool_call_name == snapshot('final_result_Foo')

    assert m.agent_model_retrievers == snapshot({})
    assert m.agent_model_allow_text_result is False

    assert m.agent_model_result_tools is not None
    assert len(m.agent_model_result_tools) == 2

    # to match the protocol, we just extract the attributes we care about
    fields = 'name', 'description', 'json_schema', 'outer_typed_dict_key'
    agent_model_result_tools = [{f: getattr(t, f) for f in fields} for t in m.agent_model_result_tools]
    assert agent_model_result_tools == snapshot(
        [
            {
                'name': 'final_result_Foo',
                'description': 'Foo: The final response which ends this conversation',
                'json_schema': {
                    'properties': {
                        'a': {'title': 'A', 'type': 'integer'},
                        'b': {'title': 'B', 'type': 'string'},
                    },
                    'required': ['a', 'b'],
                    'title': 'Foo',
                    'type': 'object',
                },
                'outer_typed_dict_key': None,
            },
            {
                'name': 'final_result_Bar',
                'description': 'This is a bar model.',
                'json_schema': {
                    'properties': {'b': {'title': 'B', 'type': 'string'}},
                    'required': ['b'],
                    'title': 'Bar',
                    'type': 'object',
                },
                'outer_typed_dict_key': None,
            },
        ]
    )

    result = agent.run_sync('Hello', model=TestModel(seed=1))
    assert result.data == mod.Bar(b='b')
    assert got_tool_call_name == snapshot('final_result_Bar')


def test_run_with_history_new():
    m = TestModel()

    agent = Agent(m, system_prompt='Foobar')

    @agent.retriever_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    result1 = agent.run_sync('Hello')
    assert result1.new_messages() == snapshot(
        [
            UserPrompt(content='Hello', timestamp=IsNow(tz=timezone.utc)),
            ModelStructuredResponse(
                calls=[ToolCall(tool_name='ret_a', args=ArgsObject(args_object={'x': 'a'}))],
                timestamp=IsNow(tz=timezone.utc),
            ),
            ToolReturn(tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc)),
            ModelTextResponse(content='{"ret_a":"a-apple"}', timestamp=IsNow(tz=timezone.utc)),
        ]
    )

    # if we pass new_messages, system prompt is inserted before the message_history messages
    result2 = agent.run_sync('Hello again', message_history=result1.new_messages())
    assert result2 == snapshot(
        RunResult(
            data='{"ret_a":"a-apple"}',
            _all_messages=[
                SystemPrompt(content='Foobar'),
                UserPrompt(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ModelStructuredResponse(
                    calls=[ToolCall(tool_name='ret_a', args=ArgsObject(args_object={'x': 'a'}))],
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ToolReturn(tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc)),
                ModelTextResponse(content='{"ret_a":"a-apple"}', timestamp=IsNow(tz=timezone.utc)),
                # second call, notice no repeated system prompt
                UserPrompt(content='Hello again', timestamp=IsNow(tz=timezone.utc)),
                ModelStructuredResponse(
                    calls=[ToolCall(tool_name='ret_a', args=ArgsObject(args_object={'x': 'a'}))],
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ToolReturn(tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc)),
                ModelTextResponse(content='{"ret_a":"a-apple"}', timestamp=IsNow(tz=timezone.utc)),
            ],
            _new_message_index=5,
            _cost=Cost(),
        )
    )
    new_msg_roles = [msg.role for msg in result2.new_messages()]
    assert new_msg_roles == snapshot(['user', 'model-structured-response', 'tool-return', 'model-text-response'])
    assert result2.new_messages_json().startswith(b'[{"content":"Hello again",')

    # if we pass all_messages, system prompt is NOT inserted before the message_history messages,
    # so only one system prompt
    result3 = agent.run_sync('Hello again', message_history=result1.all_messages())
    # same as result2 except for datetimes
    assert result3 == snapshot(
        RunResult(
            data='{"ret_a":"a-apple"}',
            _all_messages=[
                SystemPrompt(content='Foobar'),
                UserPrompt(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ModelStructuredResponse(
                    calls=[ToolCall(tool_name='ret_a', args=ArgsObject(args_object={'x': 'a'}))],
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ToolReturn(tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc)),
                ModelTextResponse(content='{"ret_a":"a-apple"}', timestamp=IsNow(tz=timezone.utc)),
                # second call, notice no repeated system prompt
                UserPrompt(content='Hello again', timestamp=IsNow(tz=timezone.utc)),
                ModelStructuredResponse(
                    calls=[ToolCall(tool_name='ret_a', args=ArgsObject(args_object={'x': 'a'}))],
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ToolReturn(tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc)),
                ModelTextResponse(content='{"ret_a":"a-apple"}', timestamp=IsNow(tz=timezone.utc)),
            ],
            _new_message_index=5,
            _cost=Cost(),
        )
    )


def test_empty_tool_calls():
    def empty(_: list[Message], _info: AgentInfo) -> ModelAnyResponse:
        return ModelStructuredResponse(calls=[])

    agent = Agent(FunctionModel(empty))

    with pytest.raises(UnexpectedModelBehavior, match='Received empty tool call message'):
        agent.run_sync('Hello')


def test_unknown_retriever():
    def empty(_: list[Message], _info: AgentInfo) -> ModelAnyResponse:
        return ModelStructuredResponse(calls=[ToolCall.from_json('foobar', '{}')])

    agent = Agent(FunctionModel(empty))

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum retries \(1\) for result validation'):
        agent.run_sync('Hello')
    assert agent.last_run_messages == snapshot(
        [
            UserPrompt(content='Hello', timestamp=IsNow(tz=timezone.utc)),
            ModelStructuredResponse(
                calls=[ToolCall(tool_name='foobar', args=ArgsJson(args_json='{}'))], timestamp=IsNow(tz=timezone.utc)
            ),
            RetryPrompt(content="Unknown tool name: 'foobar'. No tools available.", timestamp=IsNow(tz=timezone.utc)),
            ModelStructuredResponse(
                calls=[ToolCall(tool_name='foobar', args=ArgsJson(args_json='{}'))], timestamp=IsNow(tz=timezone.utc)
            ),
        ]
    )


def test_unknown_retriever_fix():
    def empty(m: list[Message], _info: AgentInfo) -> ModelAnyResponse:
        if len(m) > 1:
            return ModelTextResponse(content='success')
        else:
            return ModelStructuredResponse(calls=[ToolCall.from_json('foobar', '{}')])

    agent = Agent(FunctionModel(empty))

    result = agent.run_sync('Hello')
    assert result.data == 'success'
    assert result.all_messages() == snapshot(
        [
            UserPrompt(content='Hello', timestamp=IsNow(tz=timezone.utc)),
            ModelStructuredResponse(
                calls=[ToolCall(tool_name='foobar', args=ArgsJson(args_json='{}'))], timestamp=IsNow(tz=timezone.utc)
            ),
            RetryPrompt(content="Unknown tool name: 'foobar'. No tools available.", timestamp=IsNow(tz=timezone.utc)),
            ModelTextResponse(content='success', timestamp=IsNow(tz=timezone.utc)),
        ]
    )


def test_model_requests_blocked(env: TestEnv):
    env.set('GEMINI_API_KEY', 'foobar')
    agent = Agent('gemini-1.5-flash', result_type=tuple[str, str], defer_model_check=True)

    with pytest.raises(RuntimeError, match='Model requests are not allowed, since ALLOW_MODEL_REQUESTS is False'):
        agent.run_sync('Hello')


def test_override_model(env: TestEnv):
    env.set('GEMINI_API_KEY', 'foobar')
    agent = Agent('gemini-1.5-flash', result_type=tuple[int, str], defer_model_check=True)

    with agent.override_model('test'):
        result = agent.run_sync('Hello')
        assert result.data == snapshot((0, 'a'))


def test_override_model_no_model():
    agent = Agent()

    with pytest.raises(UserError, match=r'`model` must be set either.+Even when `override_model\(\)` is customizing'):
        with agent.override_model('test'):
            agent.run_sync('Hello')
