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
from inline_snapshot import snapshot
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
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel, _chars, _JsonSchemaTestData  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.usage import RequestUsage, RunUsage

from ..conftest import IsNow, IsStr


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


def test_custom_output_text():
    agent = Agent()
    result = agent.run_sync('x', model=TestModel(custom_output_text='custom'))
    assert result.output == snapshot('custom')
    agent = Agent(output_type=tuple[str, str])
    with pytest.raises(AssertionError, match='Plain response not allowed, but `custom_output_text` is set.'):
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
                run_id=IsStr(),
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
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
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
                run_id=IsStr(),
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
                run_id=IsStr(),
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
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
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
                run_id=IsStr(),
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
                run_id=IsStr(),
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
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
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
                run_id=IsStr(),
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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_ret', args={'x': 0}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_ret', args={'x': 0}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=61, output_tokens=8),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_ret', content='1', tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"my_ret":"1"}')],
                usage=RequestUsage(input_tokens=62, output_tokens=12),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


def test_output_tool_retry_error_handled():
    class OutputModel(BaseModel):
        x: int
        y: str

    agent = Agent('test', output_type=OutputModel, retries=2)

    call_count = 0

    @agent.output_validator
    def validate_output(ctx: RunContext[None], output: OutputModel) -> OutputModel:
        nonlocal call_count
        call_count += 1
        raise ModelRetry('Fail')

    with pytest.raises(UnexpectedModelBehavior, match=re.escape('Exceeded maximum retries (2) for output validation')):
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

    agent = Agent('test', deps_type=AgentRunDeps, output_type=OutputModel, retries=2)
    retried_run_ids = set[int]()
    event = Event()

    run_ids = list(range(5))  # fire off 5 run ids that will all retry the tool before they finish

    @agent.tool
    async def tool_that_must_be_retried(ctx: RunContext[AgentRunDeps]) -> None:
        if ctx.deps.run_id not in retried_run_ids:
            retried_run_ids.add(ctx.deps.run_id)
            raise ModelRetry('Fail')
        if len(retried_run_ids) == len(run_ids):  # pragma: no branch  # won't branch if all runs happen very quickly
            event.set()
        await event.wait()  # ensure a retry is done by all runs before any of them finish their flow
        return None

    await asyncio.gather(*[agent.run('Hello', model=TestModel(), deps=AgentRunDeps(run_id)) for run_id in run_ids])


def test_output_tool_retry_error_handled_with_custom_args():
    class ResultModel(BaseModel):
        x: int
        y: str

    agent = Agent('test', output_type=ResultModel, retries=2)

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum retries \(2\) for output validation'):
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
    assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=51, output_tokens=1))
