from __future__ import annotations as _annotations

import asyncio
import contextvars
import os
from collections.abc import AsyncIterator
from importlib.metadata import distributions

import pytest
from inline_snapshot import snapshot

from pydantic_ai import UserError
from pydantic_ai._utils import UNSET, PeekableAsyncStream, check_object_json_schema, group_by_temporal, run_in_executor

from .models.mock_async_stream import MockAsyncStream

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    'interval,expected',
    [
        (None, snapshot([[1], [2], [3]])),
        (0, snapshot([[1], [2], [3]])),
        (0.02, snapshot([[1], [2], [3]])),
        (0.04, snapshot([[1, 2], [3]])),
        (0.1, snapshot([[1, 2, 3]])),
    ],
)
async def test_group_by_temporal(interval: float | None, expected: list[list[int]]):
    async def yield_groups() -> AsyncIterator[int]:
        yield 1
        await asyncio.sleep(0.02)
        yield 2
        await asyncio.sleep(0.02)
        yield 3
        await asyncio.sleep(0.02)

    async with group_by_temporal(yield_groups(), soft_max_interval=interval) as groups_iter:
        groups: list[list[int]] = [g async for g in groups_iter]
        assert groups == expected


def test_check_object_json_schema():
    object_schema = {'type': 'object', 'properties': {'a': {'type': 'string'}}}
    assert check_object_json_schema(object_schema) == object_schema

    assert check_object_json_schema(
        {
            '$defs': {
                'JsonModel': {
                    'properties': {
                        'type': {'title': 'Type', 'type': 'string'},
                        'items': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
                    },
                    'required': ['type', 'items'],
                    'title': 'JsonModel',
                    'type': 'object',
                }
            },
            '$ref': '#/$defs/JsonModel',
        }
    ) == {
        'properties': {
            'items': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
            'type': {'title': 'Type', 'type': 'string'},
        },
        'required': ['type', 'items'],
        'title': 'JsonModel',
        'type': 'object',
    }

    # Can't remove the recursive ref here:
    assert check_object_json_schema(
        {
            '$defs': {
                'JsonModel': {
                    'properties': {
                        'type': {'title': 'Type', 'type': 'string'},
                        'items': {'anyOf': [{'$ref': '#/$defs/JsonModel'}, {'type': 'null'}]},
                    },
                    'required': ['type', 'items'],
                    'title': 'JsonModel',
                    'type': 'object',
                }
            },
            '$ref': '#/$defs/JsonModel',
        }
    ) == {
        '$defs': {
            'JsonModel': {
                'properties': {
                    'items': {'anyOf': [{'$ref': '#/$defs/JsonModel'}, {'type': 'null'}]},
                    'type': {'title': 'Type', 'type': 'string'},
                },
                'required': ['type', 'items'],
                'title': 'JsonModel',
                'type': 'object',
            }
        },
        '$ref': '#/$defs/JsonModel',
    }

    array_schema = {'type': 'array', 'items': {'type': 'string'}}
    with pytest.raises(UserError, match='^Schema must be an object$'):
        check_object_json_schema(array_schema)


@pytest.mark.parametrize('peek_first', [True, False])
@pytest.mark.anyio
async def test_peekable_async_stream(peek_first: bool):
    async_stream = MockAsyncStream(iter([1, 2, 3]))
    peekable_async_stream = PeekableAsyncStream(async_stream)

    items: list[int] = []

    # We need to both peek before starting the stream, and not, to achieve full coverage
    if peek_first:
        assert not await peekable_async_stream.is_exhausted()
        assert await peekable_async_stream.peek() == 1

    async for item in peekable_async_stream:
        items.append(item)

        # The next line is included mostly for the sake of achieving coverage
        assert await peekable_async_stream.peek() == (item + 1 if item < 3 else UNSET)

    assert await peekable_async_stream.is_exhausted()
    assert await peekable_async_stream.peek() is UNSET
    assert items == [1, 2, 3]


def test_package_versions(capsys: pytest.CaptureFixture[str]):
    if os.getenv('CI'):
        with capsys.disabled():  # pragma: lax no cover
            print('\npackage versions:')
            packages = sorted((package.metadata['Name'], package.version) for package in distributions())
            for name, version in packages:
                print(f'{name:30} {version}')


async def test_run_in_executor_with_contextvars() -> None:
    ctx_var = contextvars.ContextVar('test_var', default='default')
    ctx_var.set('original_value')

    result = await run_in_executor(ctx_var.get)
    assert result == ctx_var.get()

    ctx_var.set('new_value')
    result = await run_in_executor(ctx_var.get)
    assert result == ctx_var.get()

    # show that the old version did not work
    old_result = asyncio.get_running_loop().run_in_executor(None, ctx_var.get)
    assert old_result != ctx_var.get()
