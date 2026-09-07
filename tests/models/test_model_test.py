"""This module contains tests for the testing module."""

from __future__ import annotations as _annotations

import asyncio
import dataclasses
import re
from datetime import timezone
from typing import Annotated, Any, Literal

import pytest
from annotated_types import Ge, Gt, Le, Lt, MaxLen, MinLen
from anyio import Event
from pydantic import BaseModel, Field

from pydantic_ai import (
    Agent,
    AudioUrl,
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    RetryPromptPart,
    RunContext,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ToolAvailabilityDeltaPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel, _chars, _JsonSchemaTestData  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage, RunUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsNow, IsStr


def test_response_metadata_consistent_between_run_and_run_stream():
    """Regression test for #6062: TestModel response metadata should not depend on run mode."""
    agent = Agent(model=TestModel())

    run_result = agent.run_sync('hello')

    stream_result = agent.run_stream_sync('hello')
    list(stream_result.stream_text())

    run_responses = [message for message in run_result.all_messages() if isinstance(message, ModelResponse)]
    stream_responses = [message for message in stream_result.all_messages() if isinstance(message, ModelResponse)]

    expected_responses = snapshot(
        [
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )
    assert run_responses == expected_responses
    assert stream_responses == expected_responses


def test_call_one():
    agent = Agent()
    calls: list[str] = []

    @agent.tool_plain
    async def ret_a(x: str) -> str:
        calls.append('a')
        return f'{x}-a'

    @agent.tool_plain
    async def ret_b(x: str) -> str:  # pragma: no cover
        calls.append('b')
        return f'{x}-b'

    result = agent.run_sync('x', model=TestModel(call_tools=['ret_a']))
    assert result.output == snapshot('{"ret_a":"a-a"}')
    assert calls == ['a']


def test_call_hidden_tool_has_clear_error() -> None:
    agent = Agent(TestModel(call_tools=['hidden']))

    @agent.tool_plain(defer_loading=True)
    def hidden() -> str:  # pragma: no cover
        return 'hidden'

    with pytest.raises(
        UserError,
        match=r"Tool 'hidden' has visibility 'withheld'.*revealed.*before `TestModel` can call it",
    ):
        agent.run_sync('call hidden')


def test_call_unknown_tool_has_clear_error() -> None:
    agent = Agent(TestModel(call_tools=['missing']))

    with pytest.raises(UserError, match=r"TestModel was configured to call unknown tool 'missing'"):
        agent.run_sync('call missing')


def _native_addition_agent(call_tools: list[str] | Literal['all']) -> Agent:
    """An agent on a delta-native profile: reveals stay in history as `ToolAvailabilityDeltaPart`s
    and the revealed tool's visibility resolves to `'via_history'` rather than `'visible'`."""
    profile = ModelProfile(tool_deferral_mode='standalone', tool_addition_mode='with_definitions')
    agent = Agent(TestModel(profile=profile, call_tools=call_tools))

    @agent.tool_plain
    def revealer() -> ToolReturn:
        return ToolReturn(return_value='revealed', tools=['hidden'])

    @agent.tool_plain(defer_loading=True)
    def hidden() -> str:  # pragma: no cover
        return 'hidden'

    return agent


def test_native_tool_addition_profile_runs_without_crashing() -> None:
    """The delta part a native-addition profile keeps in history must not blow up usage
    estimation — it is legitimately present, not a skipped-`prepare_messages` violation."""
    result = _native_addition_agent('all').run_sync('go')

    assert result.output == snapshot(
        '{"revealer":"revealed","search_tools":{"discovered_tools":[],"message":"No matching tools found. The tools you need may not be available."}}'
    )
    assert any(
        isinstance(part, ToolAvailabilityDeltaPart)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


async def test_delta_part_without_native_profile_still_raises() -> None:
    """On the default profile (no native addition channel), a delta part in a direct
    `Model.request()` history still means `prepare_messages` was skipped — keep teaching that."""
    model = TestModel()
    messages: list[Any] = [
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['hidden'])]),
    ]
    with pytest.raises(UserError, match=r'Call `model.prepare_messages\(messages\)` first'):
        await model.request(messages, None, ModelRequestParameters())


def test_revealed_via_history_tool_is_callable_in_named_mode() -> None:
    """A revealed tool whose definition travels via history is callable; replaying its history
    must not raise the misleading 'must be revealed' error on later steps."""
    history = _native_addition_agent('all').run_sync('go').all_messages()

    result = _native_addition_agent(['hidden']).run_sync('continue', message_history=history)
    assert result.output == snapshot(
        '{"revealer":"revealed","search_tools":{"discovered_tools":[],"message":"No matching tools found. The tools you need may not be available."}}'
    )


def test_custom_output_text():
    agent = Agent()
    result = agent.run_sync('x', model=TestModel(custom_output_text='custom'))
    assert result.output == snapshot('custom')
    agent = Agent(output_type=tuple[str, str])
    with pytest.raises(AssertionError, match=re.escape('Plain response not allowed, but `custom_output_text` is set.')):
        agent.run_sync('x', model=TestModel(custom_output_text='custom'))


def test_custom_output_args():
    agent = Agent(output_type=tuple[str, str])
    result = agent.run_sync('x', model=TestModel(custom_output_args=['a', 'b']))
    assert result.output == ('a', 'b')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='x',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'response': ['a', 'b']},
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                    )
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=7),
                model_name='test',
                provider_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_custom_output_args_model():
    class Foo(BaseModel):
        foo: str
        bar: int

    agent = Agent(output_type=Foo)
    result = agent.run_sync('x', model=TestModel(custom_output_args={'foo': 'a', 'bar': 1}))
    assert result.output == Foo(foo='a', bar=1)
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='x',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'foo': 'a', 'bar': 1},
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                    )
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=6),
                model_name='test',
                provider_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_output_type():
    agent = Agent(output_type=tuple[str, str])
    result = agent.run_sync('x', model=TestModel())
    assert result.output == ('a', 'a')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='x',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'response': ['a', 'a']},
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                    )
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=7),
                model_name='test',
                provider_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_tool_retry():
    agent = Agent()
    call_count = 0

    @agent.tool_plain
    async def my_ret(x: int) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ModelRetry('First call failed')
        else:
            return str(x + 1)

    result = agent.run_sync('Hello', model=TestModel())
    assert call_count == 2
    assert result.output == snapshot('{"my_ret":"1"}')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_ret', args={'x': 0}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='test',
                provider_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='First call failed',
                        tool_name='my_ret',
                        timestamp=IsNow(tz=timezone.utc),
                        tool_call_id=IsStr(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_ret', args={'x': 0}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=61, output_tokens=8),
                model_name='test',
                provider_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_ret', content='1', tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"my_ret":"1"}')],
                usage=RequestUsage(input_tokens=62, output_tokens=12),
                model_name='test',
                provider_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_output_tool_retry_error_handled():
    class OutputModel(BaseModel):
        x: int
        y: str

    agent = Agent('test', output_type=OutputModel, retries={'tools': 2, 'output': 2})

    call_count = 0

    @agent.output_validator
    def validate_output(ctx: RunContext, output: OutputModel) -> OutputModel:
        nonlocal call_count
        call_count += 1
        raise ModelRetry('Fail')

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum output retries \(2\)'):
        agent.run_sync('Hello', model=TestModel())

    assert call_count == 3


@dataclasses.dataclass
class AgentRunDeps:
    run_id: int


@pytest.mark.anyio
async def test_multiple_concurrent_tool_retries():
    class OutputModel(BaseModel):
        x: int
        y: str

    agent = Agent('test', deps_type=AgentRunDeps, output_type=OutputModel, retries={'tools': 2, 'output': 2})
    retried_run_ids = set[int]()
    event = Event()

    run_ids = list(range(5))  # fire off 5 run ids that will all retry the tool before they finish

    @agent.tool
    async def tool_that_must_be_retried(ctx: RunContext[AgentRunDeps]) -> None:
        if ctx.deps.run_id not in retried_run_ids:
            retried_run_ids.add(ctx.deps.run_id)
            raise ModelRetry('Fail')
        # Won't branch if all runs happen very quickly.
        if len(retried_run_ids) == len(run_ids):  # pragma: no branch
            event.set()
        await event.wait()  # ensure a retry is done by all runs before any of them finish their flow
        return None

    await asyncio.gather(*[agent.run('Hello', model=TestModel(), deps=AgentRunDeps(run_id)) for run_id in run_ids])


def test_output_tool_retry_error_handled_with_custom_args():
    class ResultModel(BaseModel):
        x: int
        y: str

    agent = Agent('test', output_type=ResultModel, retries={'tools': 2, 'output': 2})

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum output retries \(2\)'):
        agent.run_sync('Hello', model=TestModel(custom_output_args={'foo': 'a', 'bar': 1}))


def test_json_schema_test_data():
    class NestedModel(BaseModel):
        foo: str
        bar: int

    class TestModel(BaseModel):
        my_str: str
        my_str_long: Annotated[str, MinLen(10)]
        my_str_short: Annotated[str, MaxLen(1)]
        my_int: int
        my_int_gt: Annotated[int, Gt(5)]
        my_int_ge: Annotated[int, Ge(5)]
        my_int_lt: Annotated[int, Lt(-5)]
        my_int_le: Annotated[int, Le(-5)]
        my_int_range: Annotated[int, Gt(5), Lt(15)]
        my_float: float
        my_float_gt: Annotated[float, Gt(5.0)]
        my_float_lt: Annotated[float, Lt(-5.0)]
        my_bool: bool
        my_bytes: bytes
        my_fixed_tuple: tuple[int, str]
        my_var_tuple: tuple[int, ...]
        my_list: list[str]
        my_dict: dict[str, int]
        my_set: set[str]
        my_set_min_len: Annotated[set[str], MinLen(5)]
        my_list_min_len: Annotated[list[str], MinLen(5)]
        my_lit_int: Literal[1]
        my_lit_ints: Literal[1, 2, 3]
        my_lit_str: Literal['a']
        my_lit_strs: Literal['a', 'b', 'c']
        my_any: Any
        nested: NestedModel
        union: int | list[int]
        optional: str | None
        with_example: int = Field(json_schema_extra={'examples': [1234]})
        max_len_zero: Annotated[str, MaxLen(0)]
        is_null: None
        not_required: str = 'default'

    json_schema = TestModel.model_json_schema()
    data = _JsonSchemaTestData(json_schema).generate()
    assert data == snapshot(
        {
            'my_str': 'a',
            'my_str_long': 'aaaaaaaaaa',
            'my_str_short': 'a',
            'my_int': 0,
            'my_int_gt': 6,
            'my_int_ge': 5,
            'my_int_lt': -6,
            'my_int_le': -5,
            'my_int_range': 6,
            'my_float': 0.0,
            'my_float_gt': 6.0,
            'my_float_lt': -6.0,
            'my_bool': False,
            'my_bytes': 'a',
            'my_fixed_tuple': [0, 'a'],
            'my_var_tuple': [0],
            'my_list': ['a'],
            'my_dict': {'additionalProperty': 0},
            'my_set': ['a'],
            'my_set_min_len': ['b', 'c', 'd', 'e', 'f'],
            'my_list_min_len': ['g', 'g', 'g', 'g', 'g'],
            'my_lit_int': 1,
            'my_lit_ints': 1,
            'my_lit_str': 'a',
            'my_lit_strs': 'a',
            'my_any': 'g',
            'union': 6,
            'optional': 'g',
            'with_example': 1234,
            'max_len_zero': '',
            'is_null': None,
            'nested': {'foo': 'g', 'bar': 6},
        }
    )
    TestModel.model_validate(data)


def test_json_schema_test_data_additional():
    class TestModel(BaseModel, extra='allow'):
        x: int
        additional_property: str = Field(alias='additionalProperty')

    json_schema = TestModel.model_json_schema()
    data = _JsonSchemaTestData(json_schema).generate()
    assert data == snapshot({'x': 0, 'additionalProperty': 'a', 'additionalProperty_': 'a'})
    TestModel.model_validate(data)


def test_json_schema_test_data_equal_inclusive_bounds():
    class TestModel(BaseModel):
        my_int_eq: Annotated[int, Ge(7), Le(7)]
        my_float_eq: Annotated[float, Ge(7.5), Le(7.5)]

    json_schema = TestModel.model_json_schema()
    data = _JsonSchemaTestData(json_schema).generate()
    assert data == snapshot({'my_int_eq': 7, 'my_float_eq': 7.5})
    TestModel.model_validate(data)


def test_json_schema_test_data_narrow_exclusive_bounds():
    """Narrow ranges with exclusive bounds must not crash or produce out-of-range values."""

    class TestModel(BaseModel):
        probability: Annotated[float, Ge(0), Lt(1)]
        strict_fraction: Annotated[float, Gt(0), Lt(1)]
        rate: Annotated[float, Gt(0.0), Le(1.0)]
        only_one: Annotated[int, Gt(0), Lt(2)]

    json_schema = TestModel.model_json_schema()
    for seed in range(3):
        data = _JsonSchemaTestData(json_schema, seed=seed).generate()
        TestModel.model_validate(data)
    assert _JsonSchemaTestData(json_schema).generate() == snapshot(
        {'probability': 0.0, 'strict_fraction': 0.5, 'rate': 0.5, 'only_one': 1}
    )


def test_json_schema_number_uses_strictest_bounds():
    class TestModel(BaseModel):
        lower: Annotated[float, Field(ge=0, gt=0.5, le=1)]
        upper: Annotated[float, Field(ge=0, le=2, lt=0.5)]

    json_schema = TestModel.model_json_schema()
    for seed in range(3):
        data = _JsonSchemaTestData(json_schema, seed=seed).generate()
        TestModel.model_validate(data)
    assert _JsonSchemaTestData(json_schema).generate() == snapshot({'lower': 0.75, 'upper': 0.0})


def test_narrow_exclusive_bounds_tool_args():
    """An agent tool with `Field(ge=0, lt=1)`-style parameters runs without errors."""

    agent = Agent()
    calls: list[dict[str, Any]] = []

    @agent.tool_plain
    def set_sampling(temperature: Annotated[float, Field(ge=0, lt=1)]) -> str:
        calls.append({'temperature': temperature})
        return 'ok'

    agent.run_sync('hello', model=TestModel())
    assert calls == snapshot([{'temperature': 0.0}])


def test_json_schema_number_generation():
    assert _JsonSchemaTestData({'type': 'number', 'exclusiveMinimum': 0}).generate() == 1.0
    assert _JsonSchemaTestData({'type': 'number', 'exclusiveMaximum': 0}).generate() == -1.0
    assert _JsonSchemaTestData({'type': 'number'}).generate() == 0.0


def test_chars_wrap():
    class TestModel(BaseModel):
        a: Annotated[set[str], MinLen(4)]

    json_schema = TestModel.model_json_schema()
    data = _JsonSchemaTestData(json_schema, seed=len(_chars) - 2).generate()
    assert data == snapshot({'a': ['}', '~', 'aa', 'ab']})


def test_prefix_unique():
    json_schema = {
        'type': 'array',
        'uniqueItems': True,
        'prefixItems': [{'type': 'string'}, {'type': 'string'}],
    }
    data = _JsonSchemaTestData(json_schema).generate()
    assert data == snapshot(['a', 'b'])


def test_max_items():
    json_schema = {
        'type': 'array',
        'items': {'type': 'string'},
        'maxItems': 0,
    }
    data = _JsonSchemaTestData(json_schema).generate()
    assert data == snapshot([])


@pytest.mark.parametrize('const', ['', False, 0, None])
def test_json_schema_test_data_falsy_const(const: Any) -> None:
    schema = {
        'type': 'object',
        'required': ['value'],
        'properties': {'value': {'const': const}},
    }

    assert _JsonSchemaTestData(schema).generate() == {'value': const}


def test_falsy_const_tool_args() -> None:
    """Regression test for #7629: falsy JSON Schema `const` values must be generated as-is."""

    agent = Agent()
    calls: list[dict[str, Any]] = []

    @agent.tool_plain
    def my_tool(empty: Literal[''], flag: Literal[False], zero: Literal[0]) -> str:
        calls.append({'empty': empty, 'flag': flag, 'zero': zero})
        return 'ok'

    agent.run_sync('hello', model=TestModel())
    assert calls == snapshot([{'empty': '', 'flag': False, 'zero': 0}])


@pytest.mark.parametrize(
    'content',
    [
        AudioUrl(url='https://example.com'),
        ImageUrl(url='https://example.com'),
        VideoUrl(url='https://example.com'),
        BinaryContent(data=b'', media_type='image/png'),
    ],
)
def test_different_content_input(content: AudioUrl | VideoUrl | ImageUrl | BinaryContent):
    agent = Agent()
    result = agent.run_sync(['x', content], model=TestModel(custom_output_text='custom'))
    assert result.output == snapshot('custom')
    assert result.usage == snapshot(RunUsage(requests=1, input_tokens=51, output_tokens=1))


def test_int_inclusive_upper_bound_reachable():
    """Plain inclusive integer ranges include their ceiling without changing other ranges."""

    class MyOutput(BaseModel):
        integer: Annotated[int, Field(ge=2, le=5)]
        integer_float_bounds: Annotated[int, Field(ge=2.0, le=5.0)]
        exclusive_minimum: Annotated[int, Field(ge=2, gt=1, le=5)]
        exclusive_integer: Annotated[int, Field(ge=2, le=5, lt=5)]
        number: Annotated[float, Field(ge=2, le=5)]

    agent = Agent(output_type=MyOutput)
    outputs = [agent.run_sync('hello', model=TestModel(seed=seed)).output for seed in range(4)]

    assert [
        (
            output.integer,
            output.integer_float_bounds,
            output.exclusive_minimum,
            output.exclusive_integer,
            output.number,
        )
        for output in outputs
    ] == snapshot([(2, 2, 2, 2, 2.0), (3, 3, 3, 3, 3.0), (4, 4, 4, 4, 4.0), (5, 5, 2, 2, 2.0)])

    def generated_values(minimum: float, maximum: float, seeds: list[int]) -> list[Any]:
        schema = {
            'type': 'object',
            'required': ['value'],
            'properties': {'value': {'type': 'integer', 'minimum': minimum, 'maximum': maximum}},
        }
        return [_JsonSchemaTestData(schema, seed=seed).generate()['value'] for seed in seeds]

    assert generated_values(2.5, 5.5, list(range(4))) == [2.5, 3.5, 4.5, 2.5]
    assert generated_values(2.0, 5.5, list(range(5))) == [2.0, 3.0, 4.0, 5.0, 2.5]
    assert generated_values(0.0, 1e20, [10**20]) == [10**20]
