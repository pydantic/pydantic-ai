"""This module contains tests for the testing module."""

from __future__ import annotations as _annotations

import asyncio
import dataclasses
import json
import re
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address, IPv6Address
from typing import Annotated, Any, Literal
from uuid import UUID

import pytest
from annotated_types import Ge, Gt, Le, Lt, MaxLen, MinLen
from anyio import Event
from jsonschema import FormatChecker
from jsonschema.validators import validator_for
from pydantic import AnyUrl, BaseModel, Discriminator, EmailStr, Field, SecretStr, Tag, TypeAdapter, ValidationError
from typing_extensions import TypeForm

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    AudioUrl,
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    NativeOutput,
    PromptedOutput,
    RetryInfo,
    RunContext,
    StructuredDict,
    TextPart,
    Tool,
    ToolCallPart,
    ToolOutput,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai._json_schema import JsonSchema
from pydantic_ai._json_schema_keywords import Subschema, iter_subschemas
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
                model_id='test:test',
                timestamp=IsNow(tz=UTC),
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
                        timestamp=IsNow(tz=UTC),
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
                model_id='test:test',
                provider_name='test',
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                        timestamp=IsNow(tz=UTC),
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
                        timestamp=IsNow(tz=UTC),
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
                model_id='test:test',
                provider_name='test',
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                        timestamp=IsNow(tz=UTC),
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
                        timestamp=IsNow(tz=UTC),
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
                model_id='test:test',
                provider_name='test',
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                        timestamp=IsNow(tz=UTC),
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
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=UTC))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_ret', args={'x': 0}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='test',
                model_id='test:test',
                provider_name='test',
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        content='First call failed',
                        tool_name='my_ret',
                        timestamp=IsNow(tz=UTC),
                        tool_call_id=IsStr(),
                        outcome='retried',
                        retry=RetryInfo(cause='model_retry', attempt=1, max_retries=1),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_ret', args={'x': 0}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=57, output_tokens=8),
                model_name='test',
                model_id='test:test',
                provider_name='test',
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name='my_ret', content='1', tool_call_id=IsStr(), timestamp=IsNow(tz=UTC))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"my_ret":"1"}')],
                usage=RequestUsage(input_tokens=58, output_tokens=12),
                model_name='test',
                model_id='test:test',
                provider_name='test',
                timestamp=IsNow(tz=UTC),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


class StructuredOutput(BaseModel):
    string: str
    number: int


@pytest.mark.parametrize('output_type', [NativeOutput, PromptedOutput])
@pytest.mark.parametrize('stream', [False, True])
async def test_structured_output(
    output_type: type[NativeOutput[StructuredOutput] | PromptedOutput[StructuredOutput]], stream: bool
):
    agent = Agent(TestModel(), output_type=output_type(StructuredOutput))
    if stream:
        async with agent.run_stream('x') as result:
            output = await result.get_output()
    else:
        output = (await agent.run('x')).output
    assert output == StructuredOutput(string='a', number=0)


async def test_structured_output_custom_output_text():
    model = TestModel(custom_output_text='{"string": "Paris", "number": 1}')
    result = await Agent(model, output_type=NativeOutput(StructuredOutput)).run('x')
    assert result.output == StructuredOutput(string='Paris', number=1)


async def test_structured_output_custom_output_args():
    model = TestModel(custom_output_args={'string': 'Paris', 'number': 1})
    with pytest.raises(UserError, match='`custom_output_args` is for tool output'):
        await Agent(model, output_type=NativeOutput(StructuredOutput)).run('x')


async def test_structured_output_non_object_type():
    result = await Agent(TestModel(), output_type=NativeOutput(tuple[str, int])).run('x')
    assert result.output == ('a', 0)


async def test_structured_output_custom_profile():
    model = TestModel(profile=ModelProfile())
    with pytest.raises(UserError, match='Native structured output is not supported by this model'):
        await Agent(model, output_type=NativeOutput(StructuredOutput)).run('x')


async def test_custom_profile_keeps_tool_output():
    profile = ModelProfile(thinking_enabled_by_default=True, forced_tool_choice_disables_thinking=True)
    model = TestModel(profile=profile, custom_output_args={'string': 'Paris', 'number': 1})
    result = await Agent(model, output_type=StructuredOutput).run('x')
    assert isinstance(result.response.parts[0], ToolCallPart)


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


def test_json_schema_test_data_min_properties():
    """Properties added for `minProperties` get numbered names, so many of them take linear time and memory.

    A name a property already has is skipped, and a closed object gets only the properties it lists.
    """
    data = _JsonSchemaTestData({'type': 'object', 'minProperties': 1000}).generate()
    assert len(data) == 1000
    assert max(map(len, data)) == len('additionalProperty_1000')

    schema: JsonSchema = {
        'type': 'object',
        'properties': {'additionalProperty_1': {'type': 'integer'}},
        'required': ['additionalProperty_1'],
        'minProperties': 2,
    }
    assert list(_JsonSchemaTestData(schema).generate()) == ['additionalProperty_1', 'additionalProperty_2']

    closed: JsonSchema = {
        'type': 'object',
        'properties': {'a': {'type': 'integer'}},
        'additionalProperties': False,
        'minProperties': 2,
    }
    assert list(_JsonSchemaTestData(closed).generate()) == ['a']


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


@pytest.mark.parametrize('type_', ['foo', []])
def test_gen_unknown_type(type_: str | list[str]):
    """A hand-written tool schema can name a `type` no Pydantic model emits, so this calls the generator directly.

    An empty `type` list names no type to generate from.
    """
    with pytest.raises(NotImplementedError, match=re.escape(f'Unknown type: {type_}')):
        _JsonSchemaTestData({'type': type_}).generate()


def test_json_schema_number_uses_strictest_bounds():
    class TestModel(BaseModel):
        lower: Annotated[float, Field(ge=0, gt=0.5, le=1)]
        upper: Annotated[float, Field(ge=0, le=2, lt=0.5)]

    json_schema = TestModel.model_json_schema()
    for seed in range(3):
        data = _JsonSchemaTestData(json_schema, seed=seed).generate()
        TestModel.model_validate(data)
    assert _JsonSchemaTestData(json_schema).generate() == snapshot({'lower': 0.75, 'upper': 0.0})


def test_int_uses_strictest_bounds():
    class MyOutput(BaseModel):
        lower: Annotated[int, Field(ge=0, gt=5)]
        upper: Annotated[int, Field(le=10, lt=-5)]
        both: Annotated[int, Field(ge=0, gt=5, le=100, lt=20)]

    agent = Agent(output_type=MyOutput)
    outputs = [agent.run_sync('hello', model=TestModel(seed=seed)).output for seed in range(3)]
    assert [(output.lower, output.upper, output.both) for output in outputs] == snapshot(
        [(6, -6, 6), (7, -7, 7), (8, -8, 8)]
    )


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


def test_json_pointer_ref_tool_args() -> None:
    """A local JSON-pointer `$ref` is followed like a `$defs` one when generating tool arguments.

    The MCP TypeScript SDK sends this shape for a zod v3 tool `{ from: Address, to: Address }`: the reused
    subschema points at its first occurrence instead of `$defs`.
    """
    address = {
        'type': 'object',
        'properties': {'street': {'type': 'string'}, 'city': {'type': 'string'}},
        'required': ['street', 'city'],
        'additionalProperties': False,
    }
    calls: list[dict[str, Any]] = []

    def ship_order(**kwargs: Any) -> str:
        calls.append(kwargs)
        return 'shipped'

    tool = Tool.from_schema(
        ship_order,
        name='ship_order',
        description=None,
        json_schema={
            'type': 'object',
            'properties': {'from': address, 'to': {'$ref': '#/properties/from'}},
            'required': ['from', 'to'],
            'additionalProperties': False,
        },
    )

    Agent(TestModel(), tools=[tool]).run_sync('hello')
    assert calls == snapshot([{'from': {'street': 'a', 'city': 'a'}, 'to': {'street': 'a', 'city': 'a'}}])


def test_list_form_items_tool_args() -> None:
    """A tool schema spelling a tuple as a draft-7 `items` list gets one generated value per element, up to `maxItems`.

    `zod-to-json-schema`, which the MCP TypeScript SDK uses for zod v3 tool schemas, emits this shape.
    When the schema also has `prefixItems`, those are generated instead of the list.
    """
    calls: list[dict[str, Any]] = []

    def pair_tool(**kwargs: Any) -> str:
        calls.append(kwargs)
        return 'ok'

    schema = {
        'type': 'object',
        'properties': {
            'pair': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': [{'type': 'string'}, {'type': 'integer'}]},
            'head': {'type': 'array', 'maxItems': 1, 'items': [{'type': 'string'}, {'type': 'integer'}]},
            'prefixed': {
                'type': 'array',
                'prefixItems': [{'type': 'string'}],
                'items': [{'type': 'integer'}, {'type': 'integer'}],
            },
        },
        'required': ['pair', 'head', 'prefixed'],
    }
    tool = Tool.from_schema(pair_tool, name='pair_tool', description='Takes a pair.', json_schema=schema)
    Agent(TestModel(), tools=[tool]).run_sync('hello')
    assert calls == snapshot([{'pair': ['a', 0], 'head': ['a'], 'prefixed': ['a']}])


def test_prefix_items_max_items_tool_args() -> None:
    """`prefixItems` longer than `maxItems` is cut to `maxItems`, but never below `minItems`."""
    calls: list[dict[str, Any]] = []

    def head_tool(**kwargs: Any) -> str:
        calls.append(kwargs)
        return 'ok'

    schema = {
        'type': 'object',
        'properties': {
            'head': {'type': 'array', 'maxItems': 1, 'prefixItems': [{'type': 'string'}, {'type': 'integer'}]},
            'floor': {
                'type': 'array',
                'minItems': 2,
                'maxItems': 1,
                'prefixItems': [{'type': 'string'}, {'type': 'integer'}],
            },
        },
        'required': ['head', 'floor'],
    }
    tool = Tool.from_schema(head_tool, name='head_tool', description='Takes a head.', json_schema=schema)
    Agent(TestModel(), tools=[tool]).run_sync('hello')
    assert calls == snapshot([{'head': ['a'], 'floor': ['a', 0]}])


def test_boolean_schema_tool_args() -> None:
    """A `true` subschema, which constrains nothing, is generated like a schema without a `type`."""
    calls: list[dict[str, Any]] = []

    def any_tool(**kwargs: Any) -> str:
        calls.append(kwargs)
        return 'ok'

    schema = {
        'type': 'object',
        'properties': {'anything': True, 'pair': {'type': 'array', 'items': [True, {'type': 'integer'}]}},
        'required': ['anything', 'pair'],
    }
    tool = Tool.from_schema(any_tool, name='any_tool', description='Takes anything.', json_schema=schema)
    Agent(TestModel(), tools=[tool]).run_sync('hello')
    assert calls == snapshot([{'anything': 'a', 'pair': ['a', 0]}])


def test_int_hand_written_bounds_tool_args() -> None:
    """Integer arguments are integers inside the unusual bounds a hand-written tool schema can carry.

    Pydantic never emits a fractional, infinite, null or draft-4 boolean bound for an `int` field.
    """
    calls: list[dict[str, Any]] = []

    def bounded_tool(**kwargs: Any) -> str:
        calls.append(kwargs)
        return 'ok'

    properties = {
        'collapsed': {'type': 'integer', 'minimum': 1, 'maximum': 1, 'exclusiveMinimum': 0.5},
        'narrow': {'type': 'integer', 'minimum': 0, 'maximum': 1, 'exclusiveMinimum': 0.5},
        'exclusive': {'type': 'integer', 'exclusiveMinimum': 0.5},
        'inclusive_lower': {'type': 'integer', 'minimum': 2.5},
        'inclusive_upper': {'type': 'integer', 'maximum': -2.5},
        'unbounded': {'type': 'integer', 'minimum': 0, 'maximum': float('inf')},
        'null_bound': {'type': 'integer', 'minimum': None, 'maximum': 3},
        'draft4_exclusive': {'type': 'integer', 'minimum': -5, 'maximum': 0, 'exclusiveMinimum': False},
        'huge': {'type': 'integer', 'minimum': 0, 'maximum': 10**400},
    }
    schema = {'type': 'object', 'properties': properties, 'required': list(properties)}
    tool = Tool.from_schema(
        bounded_tool, name='bounded_tool', description='Takes bounded integers.', json_schema=schema
    )
    Agent(TestModel(), tools=[tool]).run_sync('hello')
    assert calls == snapshot(
        [
            {
                'collapsed': 1,
                'narrow': 1,
                'exclusive': 1,
                'inclusive_lower': 3,
                'inclusive_upper': -3,
                'unbounded': 0,
                'null_bound': 3,
                'draft4_exclusive': -5,
                'huge': 0,
            }
        ]
    )


@pytest.mark.parametrize(
    'seed,expected',
    [
        (0, snapshot({'nullable': 'a', 'null_first': None, 'multi': 3, 'list': [0], 'obj': {'x': False}})),
        (1, snapshot({'nullable': None, 'null_first': 'b', 'multi': 'bb', 'list': None, 'obj': None})),
    ],
)
def test_tool_type_list_args(seed: int, expected: dict[str, object]) -> None:
    """The seed picks one member of a JSON Schema `type` list, as it picks an `anyOf` member.

    The picked type keeps the sibling keywords (`minimum`, `minLength`, `items`, `properties`), and type lists
    nested under `items` and an object property are generated the same way. `TestModel` makes no provider
    request, so there is nothing to record.
    """
    calls: list[dict[str, object]] = []

    def lookup(**kwargs: object) -> str:
        calls.append(kwargs)
        return 'ok'

    schema: dict[str, object] = {
        'type': 'object',
        'properties': {
            'nullable': {'type': ['string', 'null']},
            'null_first': {'type': ['null', 'string']},
            'multi': {'type': ['integer', 'string'], 'minimum': 3, 'minLength': 2},
            'list': {'type': ['array', 'null'], 'minItems': 1, 'items': {'type': ['integer', 'null']}},
            'obj': {
                'type': ['object', 'null'],
                'properties': {'x': {'type': ['boolean', 'null']}},
                'required': ['x'],
            },
        },
        'required': ['nullable', 'null_first', 'multi', 'list', 'obj'],
    }
    tool = Tool.from_schema(lookup, name='lookup', description='Look something up.', json_schema=schema)
    Agent(TestModel(seed=seed), tools=[tool]).run_sync('hello')

    assert calls == [expected]


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
    """Plain inclusive integer ranges reach their ceiling, and the ranges here with an exclusive key don't."""

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
    ] == snapshot([(2, 2, 2, 2, 2.0), (3, 3, 3, 3, 3.0), (4, 4, 4, 2, 4.0), (5, 5, 2, 3, 2.0)])

    def generated_values(minimum: float, maximum: float, seeds: list[int]) -> list[Any]:
        schema = {
            'type': 'object',
            'required': ['value'],
            'properties': {'value': {'type': 'integer', 'minimum': minimum, 'maximum': maximum}},
        }
        return [_JsonSchemaTestData(schema, seed=seed).generate()['value'] for seed in seeds]

    assert generated_values(2.5, 5.5, list(range(4))) == [3, 4, 5, 3]
    assert generated_values(2.0, 5.5, list(range(5))) == [2, 3, 4, 5, 2]
    assert generated_values(0.0, 1e20, [10**20]) == [10**20]


class Cat(BaseModel):
    kind: Literal['cat']
    name: str


class Dog(BaseModel):
    kind: Literal['dog']
    breed: str


Discriminated = Annotated[Cat | Dog, Field(discriminator='kind')]


def test_tool_output_oneof_discriminated_union():
    outputs = [
        Agent(model=TestModel(seed=seed), output_type=ToolOutput(Discriminated)).run_sync('hello').output
        for seed in (0, 1)
    ]

    assert outputs == snapshot([Cat(kind='cat', name='a'), Dog(kind='dog', breed='b')])


def test_function_tool_oneof_discriminated_union_arg():
    received: list[Cat | Dog] = []

    agent = Agent(model=TestModel())

    @agent.tool_plain
    def pet_name(pet: Discriminated) -> str:
        received.append(pet)
        return pet.kind

    agent.run_sync('hello')

    assert received == snapshot([Cat(kind='cat', name='a')])


class DefaultedCat(BaseModel):
    kind: Literal['cat'] = 'cat'
    color: Literal['black']


class DefaultedDog(BaseModel):
    kind: Literal['dog'] = 'dog'
    breed: str


def test_tool_output_oneof_discriminated_union_defaulted_tag():
    """Pydantic leaves a defaulted tag out of `required`, but validation needs it to pick the member."""
    agent = Agent(
        model=TestModel(),
        output_type=ToolOutput(Annotated[DefaultedCat | DefaultedDog, Field(discriminator='kind')]),
    )

    result = agent.run_sync('hello')

    assert result.output == snapshot(DefaultedCat(color='black'))


class WhiteCat(BaseModel):
    kind: Literal['cat'] = 'cat'
    color: Literal['white']


def cat_color(cat: DefaultedCat | WhiteCat | dict[str, str]) -> str:
    return cat['color'] if isinstance(cat, dict) else cat.color


CatsByColor = Annotated[DefaultedCat | WhiteCat, Field(discriminator='color')]
CatsByCallable = Annotated[
    Annotated[DefaultedCat, Tag('black')] | Annotated[WhiteCat, Tag('white')], Discriminator(cat_color)
]


def test_tool_output_oneof_nested_discriminated_union_defaulted_tag():
    """The outer tag must reach the member the inner union picks."""
    agent = Agent(
        model=TestModel(),
        output_type=ToolOutput(Annotated[CatsByColor | DefaultedDog, Field(discriminator='kind')]),
    )

    result = agent.run_sync('hello')

    assert result.output == snapshot(DefaultedCat(color='black'))


def test_tool_output_oneof_nested_callable_discriminator_defaulted_tag():
    """A callable `Discriminator` emits `oneOf` with no `discriminator` keyword, but still passes the outer tag on."""
    agent = Agent(
        model=TestModel(),
        output_type=ToolOutput(Annotated[CatsByCallable | DefaultedDog, Field(discriminator='kind')]),
    )

    result = agent.run_sync('hello')

    assert result.output == snapshot(DefaultedCat(color='black'))


@pytest.mark.parametrize(
    'discriminator', [pytest.param({}, id='absent'), pytest.param({'discriminator': 'value'}, id='swagger-2-string')]
)
def test_structured_dict_oneof_without_discriminator(discriminator: dict[str, str]):
    """A `oneOf` without an OpenAPI `discriminator` object picks a member like `anyOf`."""
    agent = Agent(
        model=TestModel(),
        output_type=StructuredDict(
            {
                'type': 'object',
                'properties': {'value': {'oneOf': [{'type': 'integer'}, {'type': 'string'}], **discriminator}},
                'required': ['value'],
            }
        ),
    )

    result = agent.run_sync('hello')

    assert result.output == snapshot({'value': 0})


def test_structured_dict_oneof_boolean_member():
    """A boolean subschema in `oneOf` has no structure to generate from, so `TestModel` writes a character."""
    agent = Agent(
        model=TestModel(),
        output_type=StructuredDict(
            {
                'type': 'object',
                'properties': {'value': {'oneOf': [True, {'type': 'integer'}]}},
                'required': ['value'],
            }
        ),
    )

    result = agent.run_sync('hello')

    assert result.output == snapshot({'value': 'a'})


def test_structured_dict_oneof_beside_type():
    """A `oneOf` beside a `type` only narrows it, so the `type` drives generation."""
    agent = Agent(
        model=TestModel(),
        output_type=StructuredDict(
            {
                'type': 'object',
                'properties': {'a': {'type': 'integer'}, 'b': {'type': 'string'}, 'c': {'type': 'string'}},
                'required': ['a'],
                'oneOf': [{'required': ['b']}, {'required': ['c']}],
            }
        ),
    )

    result = agent.run_sync('hello')

    assert result.output == snapshot({'a': 0})


def _get_city(city: str, country: str) -> str:
    return f'{city}, {country}'


def _get_weather(city: str, days: int) -> str:
    return f'sunny in {city} for {days} days'


class _City(BaseModel):
    city: str
    country: str


@dataclasses.dataclass(frozen=True)
class UsageCase:
    id: str
    response_usages: list[RequestUsage]
    total: RunUsage
    tools: tuple[Callable[..., str], ...] = ()
    output_type: type[str] | type[_City] = str
    custom_output_text: str | None = None


USAGE_CASES = [
    UsageCase(
        id='text',
        response_usages=snapshot([RequestUsage(input_tokens=51, output_tokens=4)]),
        total=snapshot(RunUsage(input_tokens=51, output_tokens=4, requests=1)),
    ),
    UsageCase(
        id='one_tool',
        tools=(_get_city,),
        response_usages=snapshot(
            [RequestUsage(input_tokens=51, output_tokens=7), RequestUsage(input_tokens=53, output_tokens=12)]
        ),
        total=snapshot(RunUsage(input_tokens=104, output_tokens=19, requests=2, tool_calls=1)),
    ),
    UsageCase(
        id='two_tools',
        tools=(_get_city, _get_weather),
        response_usages=snapshot(
            [RequestUsage(input_tokens=51, output_tokens=13), RequestUsage(input_tokens=59, output_tokens=25)]
        ),
        total=snapshot(RunUsage(input_tokens=110, output_tokens=38, requests=2, tool_calls=2)),
    ),
    UsageCase(
        id='output_tool',
        output_type=_City,
        response_usages=snapshot([RequestUsage(input_tokens=51, output_tokens=7)]),
        total=snapshot(RunUsage(input_tokens=51, output_tokens=7, requests=1)),
    ),
    UsageCase(
        id='text_with_delimiters',
        custom_output_text='Paris, France.',
        response_usages=snapshot([RequestUsage(input_tokens=51, output_tokens=3)]),
        total=snapshot(RunUsage(input_tokens=51, output_tokens=3, requests=1)),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in USAGE_CASES])
async def test_usage_consistent_across_run_modes(case: UsageCase):
    """Regression test for #11854: streamed `TestModel` usage equals `run()` usage, per response and in total."""
    agent = Agent(TestModel(custom_output_text=case.custom_output_text), output_type=case.output_type, tools=case.tools)
    usages: dict[str, tuple[list[RequestUsage], RunUsage]] = {}

    run_result = await agent.run('hello')
    usages['run'] = ([m.usage for m in run_result.all_messages() if isinstance(m, ModelResponse)], run_result.usage)

    async with agent.run_stream('hello') as stream_result:
        await stream_result.get_output()
    usages['run_stream'] = (
        [m.usage for m in stream_result.all_messages() if isinstance(m, ModelResponse)],
        stream_result.usage,
    )

    async with agent.run_stream_events('hello') as event_stream:
        async for event in event_stream:
            if isinstance(event, AgentRunResultEvent):
                usages['run_stream_events'] = (
                    [m.usage for m in event.result.all_messages() if isinstance(m, ModelResponse)],
                    event.result.usage,
                )

    async with agent.iter('hello') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for _ in request_stream:
                        pass
    assert agent_run.result is not None
    usages['iter_node_stream'] = (
        [m.usage for m in agent_run.result.all_messages() if isinstance(m, ModelResponse)],
        agent_run.result.usage,
    )

    run_response_usages, run_total = usages['run']
    assert run_response_usages == case.response_usages
    assert run_total == case.total
    assert usages == dict.fromkeys(['run', 'run_stream', 'run_stream_events', 'iter_node_stream'], usages['run'])


class _Inner(BaseModel):
    x: int


class _Tree(BaseModel):
    children: list[_Tree] = []


class _Color(Enum):
    RED = 'r'
    BLUE = 'b'


TEST_MODEL_TYPES: dict[str, TypeForm[object]] = {
    'int': int,
    'int_multiple_of': Annotated[int, Field(multiple_of=7)],
    'int_multiple_of_bounded': Annotated[int, Field(multiple_of=7, ge=10, le=30)],
    'int_exclusive_bounds': Annotated[int, Field(gt=3, lt=6)],
    'float_multiple_of': Annotated[float, Field(multiple_of=0.5, gt=1)],
    'float_exclusive_bounds': Annotated[float, Field(gt=0.1, lt=0.2)],
    'decimal': Decimal,
    'str_length': Annotated[str, Field(min_length=2, max_length=3)],
    'str_pattern': Annotated[str, Field(pattern=r'^\d{3}$')],
    'email': EmailStr,
    'url': AnyUrl,
    'uuid': UUID,
    'datetime': datetime,
    'date': date,
    'time': time,
    'timedelta': timedelta,
    'ipv4': IPv4Address,
    'ipv6': IPv6Address,
    'bytes': bytes,
    'secret': SecretStr,
    'dict_min_length': Annotated[dict[str, int], Field(min_length=2)],
    'dict_max_length_0': Annotated[dict[str, int], Field(max_length=0)],
    'list_length': Annotated[list[int], Field(min_length=2, max_length=3)],
    'set': set[int],
    'frozenset_min_length': Annotated[frozenset[int], Field(min_length=2)],
    'tuple': tuple[int, str],
    'tuple_variadic': tuple[int, ...],
    'literal': Literal['a', 'b'],
    'enum': _Color,
    'optional': int | None,
    'union': int | str,
    'model': _Inner,
    'recursive_model': _Tree,
    'list_of_models': list[_Inner],
    'bool': bool,
    'none': None,
}
"""Field types whose Pydantic schemas `TestModel` has to generate valid arguments for."""

_LEAF: JsonSchema = {'type': 'string', 'minLength': 1}

TEST_MODEL_SCHEMAS: dict[str, JsonSchema] = {
    'allOf': {'allOf': [{'type': 'integer', 'minimum': 3}]},
    'allOf-ref': {'allOf': [{'$ref': '#/$defs/Leaf'}], 'description': 'A leaf'},
    'allOf-properties': {'allOf': [{'type': 'object', 'properties': {'a': _LEAF}, 'required': ['a']}]},
    'allOf-boolean': {'allOf': [True, {'type': 'integer'}]},
    'allOf-ref-to-boolean': {'allOf': [{'$ref': '#/$defs/Flag'}, {'type': 'integer'}]},
    'allOf-ref-to-ref': {'type': 'string', 'allOf': [{'$ref': '#/$defs/Alias'}]},
    'allOf-anyOf': {'type': 'string', 'allOf': [{'anyOf': [{'type': 'string'}, {'type': 'integer'}]}]},
    'allOf-repeated-bound': {'allOf': [{'type': 'integer', 'minimum': -10}, {'type': 'integer', 'minimum': 0}]},
    'allOf-repeated-property': {
        'allOf': [
            {'type': 'object', 'properties': {'a': {'type': 'integer', 'minimum': -10}}, 'required': ['a']},
            {'properties': {'a': {'type': 'integer', 'minimum': 0}}},
        ]
    },
    'minProperties': {'type': 'object', 'minProperties': 1},
    'minProperties-optional': {'type': 'object', 'properties': {'a': _LEAF}, 'minProperties': 1},
    'minProperties-declared-name': {
        'type': 'object',
        'properties': {'additionalProperty_1': {'$ref': '#/$defs/Count'}},
        'minProperties': 1,
    },
    'maxProperties': {'type': 'object', 'additionalProperties': True, 'maxProperties': 0},
    'multipleOf-within-exclusive': {'type': 'integer', 'multipleOf': 5, 'exclusiveMinimum': 5, 'exclusiveMaximum': 11},
    'multipleOf-below-maximum': {'type': 'integer', 'multipleOf': 5, 'maximum': 8},
    'multipleOf-fractional-integer': {'type': 'integer', 'multipleOf': 2.5, 'minimum': 0, 'maximum': 10},
    'multipleOf-fractional-integer-off-the-adjacent-multiples': {
        'type': 'integer',
        'multipleOf': 1.5,
        'minimum': 1,
        'maximum': 4,
    },
    'multipleOf-integer-too-large-for-a-float': {'type': 'integer', 'multipleOf': 3, 'minimum': 10**400},
    'multipleOf-fractional-number': {'type': 'number', 'multipleOf': 0.5, 'minimum': 0.3},
    'format-longer-than-maxLength': {'type': 'string', 'format': 'uri', 'maxLength': 3},
    'not': {'type': 'string', 'not': {'const': 'a'}},
}
"""Hand-written schemas Pydantic doesn't generate, for the keywords `TestModel` handles."""

_TEST_MODEL_DEFS: JsonSchema = {
    'Leaf': _LEAF,
    'Flag': True,
    'Alias': {'$ref': '#/$defs/TwoCharacters'},
    'TwoCharacters': {'minLength': 2},
    'Count': {'type': 'integer'},
}
"""Definitions beside `TEST_MODEL_SCHEMAS`."""

UNSUPPORTED_KEYWORDS = {
    'pattern': 'Generating a string that matches a regular expression needs a regex-inverting dependency.',
    'not': 'Only the schema itself drives generation, so a value can match what `not` rejects.',
}
"""Keywords `TestModel` doesn't generate for, so a schema using one can get an argument it rejects."""

PYDANTIC_STRICTER = {
    'decimal': "`Decimal`'s schema accepts any string, so `TestModel` can generate one that isn't a number.",
}
"""Types whose schema accepts an argument their Pydantic validation rejects, which an entry expects."""

_FORMATS = FormatChecker(formats=('date', 'email', 'ipv4', 'ipv6', 'uuid'))
"""The formats `jsonschema` checks with the standard library alone, so the result doesn't depend on what's installed."""


def _arguments_schema(field: JsonSchema, defs: JsonSchema) -> JsonSchema:
    """A tool's arguments schema, with `field` as its one required property."""
    return {'type': 'object', 'properties': {'field': field}, 'required': ['field'], '$defs': defs}


def _test_model_schemas() -> Iterator[tuple[str, JsonSchema, TypeAdapter[object] | None]]:
    for name, field_type in TEST_MODEL_TYPES.items():
        adapter = TypeAdapter[object](field_type)
        field_schema = adapter.json_schema()
        defs = field_schema.pop('$defs', {})
        yield name, _arguments_schema(field_schema, defs), adapter
    for name, field_schema in TEST_MODEL_SCHEMAS.items():
        yield name, _arguments_schema(field_schema, _TEST_MODEL_DEFS), None


def _keywords(schema: Subschema) -> Iterator[str]:
    """Every keyword in `schema` and its subschemas."""
    if isinstance(schema, dict):
        yield from schema
        for subschema in iter_subschemas(schema):
            yield from _keywords(subschema)


@pytest.mark.parametrize(
    ('name', 'schema', 'adapter'), [pytest.param(*case, id=case[0]) for case in _test_model_schemas()]
)
def test_test_model_arguments_validate(name: str, schema: JsonSchema, adapter: TypeAdapter[object] | None):
    """`TestModel`'s arguments for each seed validate against their schema, and against the Pydantic type behind it.

    A schema using an `UNSUPPORTED_KEYWORDS` keyword, or a `PYDANTIC_STRICTER` type, is expected to get an invalid
    argument for some seed, so the entry has to go once `TestModel` generates valid ones.
    """
    invalid: dict[int, list[str]] = {}
    for seed in range(3):
        arguments = _JsonSchemaTestData(schema, seed).generate()
        errors = [
            error.message for error in validator_for(schema)(schema, format_checker=_FORMATS).iter_errors(arguments)
        ]
        if adapter is not None:
            try:
                # Tool arguments arrive as JSON, so they're validated in JSON mode.
                adapter.validate_json(json.dumps(arguments['field']))
            except ValidationError as e:
                errors.extend(error['msg'] for error in e.errors())
        if errors:
            invalid[seed] = errors
    if expected := sorted(UNSUPPORTED_KEYWORDS.keys() & set(_keywords(schema))) or PYDANTIC_STRICTER.get(name):
        assert invalid, f'`TestModel` now generates valid arguments for `{name}`: remove the entry for {expected}'
    else:
        assert not invalid, f'`TestModel` generates invalid arguments for `{name}`: {invalid}'


@pytest.mark.parametrize(
    'schema',
    [
        pytest.param(_arguments_schema({'allOf': [{'$ref': '#/$defs/Missing'}]}, {}), id='allOf-missing-ref'),
        pytest.param(
            _arguments_schema({'$ref': '#/$defs/Loop'}, {'Loop': {'allOf': [{'$ref': '#/$defs/Loop'}]}}),
            id='allOf-cycle',
        ),
        pytest.param(
            _arguments_schema({'type': 'number', 'multipleOf': 3e-308, 'minimum': 1e10}, {}), id='multipleOf-overflows'
        ),
        pytest.param(_arguments_schema({'type': 'string', 'allOf': True}, {}), id='allOf-not-a-list'),
        pytest.param(
            _arguments_schema({'type': 'string', 'allOf': [{'allOf': None}]}, {}), id='nested-allOf-not-a-list'
        ),
        pytest.param(_arguments_schema({'type': 'string', 'allOf': [{'$ref': 5}]}, {}), id='allOf-ref-not-a-string'),
        pytest.param(
            {**_arguments_schema({'type': 'string', 'allOf': [{'$ref': '#/$defs/Leaf'}]}, {}), '$defs': []},
            id='defs-not-an-object',
        ),
        pytest.param(
            _arguments_schema({'type': 'string', 'allOf': [{'properties': 5, 'required': 5}]}, {}),
            id='allOf-member-properties-not-an-object',
        ),
        pytest.param(_arguments_schema({'type': 'string', 'format': ['email']}, {}), id='format-not-a-string'),
        pytest.param(
            _arguments_schema({'type': 'string', 'format': 'email', 'maxLength': '3'}, {}),
            id='format-length-not-a-number',
        ),
        pytest.param(_arguments_schema({'type': 'object', 'minProperties': '1'}, {}), id='minProperties-not-a-number'),
        pytest.param(
            _arguments_schema({'type': 'object', 'additionalProperties': True, 'maxProperties': '0'}, {}),
            id='maxProperties-not-a-number',
        ),
        pytest.param(
            _arguments_schema({'type': 'integer', 'multipleOf': 2.5, 'minimum': 10**400}, {}),
            id='multipleOf-fractional-step-for-an-integer-too-large-for-a-float',
        ),
        pytest.param(
            _arguments_schema({'type': 'integer', 'multipleOf': float('inf'), 'minimum': 1}, {}),
            id='multipleOf-infinite',
        ),
        pytest.param(
            _arguments_schema(
                {'$ref': '#/$defs/Filter'},
                {
                    'Filter': {
                        'type': 'object',
                        'properties': {
                            'and': {'type': 'array', 'items': {'$ref': '#/$defs/Filter'}},
                            'eq': {'type': 'string'},
                        },
                        'minProperties': 1,
                    }
                },
            ),
            id='minProperties-recursive',
        ),
    ],
)
def test_test_model_generates_for_schemas_it_cant_satisfy(schema: JsonSchema):
    """`TestModel` still generates arguments where `allOf`, `multipleOf`, `format` or an object size bound can't be
    honored, as it did when it ignored them, rather than raising."""
    assert 'field' in _JsonSchemaTestData(schema).generate()


@pytest.mark.parametrize(
    ('fmt', 'seed', 'expected'),
    [
        pytest.param('uuid', -1, 'ffffffff-ffff-ffff-ffff-ffffffffffff', id='uuid-negative'),
        pytest.param('duration', -1, 'PT1S', id='duration-negative'),
        pytest.param('date', 10**7, '9405-01-28', id='date-past-9999'),
        pytest.param('date-time', -(10**7), '4641-12-04T00:00:00Z', id='date-time-before-year-1'),
    ],
)
def test_test_model_generates_a_format_for_any_seed(fmt: str, seed: int, expected: str):
    """A seed outside what a format holds, like a negative UUID or a date past 9999, still gets a value of it.

    `jsonschema` checks neither `duration` nor `date-time` without extra packages, so the values are pinned here.
    """
    schema = _arguments_schema({'type': 'string', 'format': fmt}, {})
    assert _JsonSchemaTestData(schema, seed=seed).generate() == {'field': expected}
