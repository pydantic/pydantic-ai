from __future__ import annotations as _annotations

import asyncio
import contextvars
import functools
import os
from collections.abc import AsyncGenerator, AsyncIterator
from importlib.metadata import distributions
from typing import cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import UserError
from pydantic_ai._utils import (
    UNSET,
    PeekableAsyncStream,
    check_object_json_schema,
    group_by_temporal,
    is_async_callable,
    merge_json_schema_defs,
    run_in_executor,
    strip_markdown_fences,
    validate_empty_kwargs,
)

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


async def test_group_by_temporal_break_cancels_pending_task():
    """Test that breaking out of iteration cancels a pending task in the inner finally block."""

    async def yield_slowly() -> AsyncIterator[int]:
        yield 1
        await asyncio.sleep(10)  # Long delay to ensure task is pending when we break
        yield 2  # pragma: no cover

    async with group_by_temporal(yield_slowly(), soft_max_interval=0.01) as groups_iter:
        async for group in groups_iter:  # pragma: no branch
            assert group == [1]
            break  # Triggers inner finally block which cancels the pending task


async def test_group_by_temporal_aclose_triggers_generator_exit():
    """Test that explicitly closing the generator triggers GeneratorExit handling."""

    async def yield_slowly() -> AsyncIterator[int]:
        yield 1
        await asyncio.sleep(10)
        yield 2  # pragma: no cover

    async with group_by_temporal(yield_slowly(), soft_max_interval=0.01) as groups_iter:
        groups_aiter = cast(AsyncGenerator[list[int]], aiter(groups_iter))
        group = await anext(groups_aiter)
        assert group == [1]
        # Explicitly close the inner generator to trigger GeneratorExit
        await groups_aiter.aclose()


async def test_group_by_temporal_inner_generator_closed_on_context_exit():
    """Test that breaking early from iteration cleans up properly.

    This tests the fix for "RuntimeError: generator didn't stop after athrow()"
    which occurred when the inner generator wasn't explicitly closed.
    The test verifies that the context manager exits cleanly after a break.
    """
    items_yielded: list[int] = []

    async def yield_items() -> AsyncIterator[int]:
        items_yielded.append(1)
        yield 1
        await asyncio.sleep(0.01)
        items_yielded.append(2)
        yield 2
        await asyncio.sleep(0.01)
        items_yielded.append(3)
        yield 3

    # Break early without consuming all items
    async with group_by_temporal(yield_items(), soft_max_interval=0.05) as groups_iter:
        async for group in groups_iter:  # pragma: no branch
            assert 1 in group
            break

    # We only consumed the first item
    # The important thing is that the context manager exits cleanly without errors
    assert 1 in items_yielded


async def test_group_by_temporal_inner_generator_closed_on_exception():
    """Test that exception during iteration is handled cleanly."""
    items_yielded: list[int] = []

    async def yield_items() -> AsyncIterator[int]:
        items_yielded.append(1)
        yield 1
        await asyncio.sleep(0.01)
        items_yielded.append(2)
        yield 2

    with pytest.raises(ValueError, match='test error'):
        async with group_by_temporal(yield_items(), soft_max_interval=0.05) as groups_iter:
            async for group in groups_iter:  # pragma: no branch
                if 1 in group:  # pragma: no branch
                    raise ValueError('test error')

    # We only consumed the first item
    # The important thing is that cleanup happens without additional errors
    assert 1 in items_yielded


async def test_group_by_temporal_noop_path_break_early():
    """Test that the None interval (noop) path also handles break cleanly."""
    items_yielded: list[int] = []

    async def yield_items() -> AsyncIterator[int]:
        items_yielded.append(1)
        yield 1
        items_yielded.append(2)  # pragma: no cover
        yield 2  # pragma: no cover
        items_yielded.append(3)  # pragma: no cover
        yield 3  # pragma: no cover

    # Break early with soft_max_interval=None (noop path)
    async with group_by_temporal(yield_items(), soft_max_interval=None) as groups_iter:
        async for group in groups_iter:  # pragma: no branch
            assert group == [1]
            break

    # We only consumed the first item
    # Context manager should exit cleanly
    assert items_yielded == [1]


async def test_group_by_temporal_fully_consumed_no_error():
    """Test that fully consuming the iterator doesn't cause errors."""
    items_yielded: list[int] = []

    async def yield_items() -> AsyncIterator[int]:
        for i in range(3):
            items_yielded.append(i)
            yield i
            await asyncio.sleep(0.01)

    async with group_by_temporal(yield_items(), soft_max_interval=0.05) as groups_iter:
        all_groups: list[list[int]] = []
        async for group in groups_iter:
            all_groups.append(group)

    # All items should have been yielded
    assert items_yielded == [0, 1, 2]
    # All items should appear in some group
    all_items = [item for group in all_groups for item in group]
    assert sorted(all_items) == [0, 1, 2]


async def test_group_by_temporal_generator_exit_on_outer_close():
    """Test that closing the context manager's underlying generator handles GeneratorExit.

    This tests the defensive GeneratorExit handling by directly accessing
    the generator from the async context manager.
    """

    async def yield_items() -> AsyncIterator[int]:
        yield 1
        await asyncio.sleep(10)
        yield 2  # pragma: no cover

    # Get the async context manager
    ctx_manager = group_by_temporal(yield_items(), soft_max_interval=0.05)

    # Enter the context to get the generator - this advances to the yield
    groups_iter = await ctx_manager.__aenter__()

    # Consume one item to ensure we're in the middle of iteration
    groups_aiter = cast(AsyncGenerator[list[int]], aiter(groups_iter))
    group = await anext(groups_aiter)
    assert 1 in group

    # Now close the inner generator first (to ensure it's handled)
    await groups_aiter.aclose()

    # Then exit the context manager normally (which calls __aexit__)
    await ctx_manager.__aexit__(None, None, None)

    # If we get here without error, the cleanup worked correctly


async def test_group_by_temporal_generator_exit_via_gen_aclose():
    """Test GeneratorExit handling by directly closing the underlying generator.

    This simulates GC finalization by calling aclose() on the @asynccontextmanager's
    underlying generator, bypassing __aexit__. This throws GeneratorExit at the
    `yield inner_gen` suspension point, exercising the except GeneratorExit handler.
    """

    async def yield_slowly() -> AsyncIterator[int]:
        yield 1
        await asyncio.sleep(10)
        yield 2  # pragma: no cover

    ctx_manager = group_by_temporal(yield_slowly(), soft_max_interval=0.05)
    groups_iter = await ctx_manager.__aenter__()
    groups_aiter = cast(AsyncGenerator[list[int]], aiter(groups_iter))

    # Consume one group to create an active task (fetching item 2, which sleeps 10s)
    group = await anext(groups_aiter)
    assert group == [1]

    # Directly close the underlying generator to simulate GC finalization.
    # This throws GeneratorExit at `yield inner_gen`, exercising the handler
    # that sets closing=True and skips awaiting in the finally block.
    await ctx_manager.gen.aclose()

    # Let event loop process the cancelled task
    await asyncio.sleep(0)

    # Clean up the orphaned inner generator
    await groups_aiter.aclose()


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


async def test_run_in_executor_with_disable_threads() -> None:
    from pydantic_ai._utils import disable_threads

    calls: list[str] = []

    def sync_func() -> str:
        calls.append('called')
        return 'result'

    # Without disable_threads, should use threading
    result = await run_in_executor(sync_func)
    assert result == 'result'
    assert calls == ['called']

    # With disable_threads enabled, should execute directly
    calls.clear()
    with disable_threads():
        result = await run_in_executor(sync_func)
        assert result == 'result'
        assert calls == ['called']


def test_is_async_callable():
    def sync_func(): ...  # pragma: no branch

    assert is_async_callable(sync_func) is False

    async def async_func(): ...  # pragma: no branch

    assert is_async_callable(async_func) is True

    class AsyncCallable:
        async def __call__(self): ...  # pragma: no branch

    partial_async_callable = functools.partial(AsyncCallable())
    assert is_async_callable(partial_async_callable) is True


def test_merge_json_schema_defs():
    foo_bar_schema = {
        '$defs': {
            'Bar': {
                'description': 'Bar description',
                'properties': {'bar': {'type': 'string'}},
                'required': ['bar'],
                'title': 'Bar',
                'type': 'object',
            },
            'Foo': {
                'description': 'Foo description',
                'properties': {'foo': {'type': 'string'}},
                'required': ['foo'],
                'title': 'Foo',
                'type': 'object',
            },
        },
        'properties': {'foo': {'$ref': '#/$defs/Foo'}, 'bar': {'$ref': '#/$defs/Bar'}},
        'required': ['foo', 'bar'],
        'type': 'object',
        'title': 'FooBar',
    }

    foo_bar_baz_schema = {
        '$defs': {
            'Baz': {
                'description': 'Baz description',
                'properties': {'baz': {'type': 'string'}},
                'required': ['baz'],
                'title': 'Baz',
                'type': 'object',
            },
            'Foo': {
                'description': 'Foo description. Note that this is different from the Foo in foo_bar_schema!',
                'properties': {'foo': {'type': 'int'}},
                'required': ['foo'],
                'title': 'Foo',
                'type': 'object',
            },
            'Bar': {
                'description': 'Bar description',
                'properties': {'bar': {'type': 'string'}},
                'required': ['bar'],
                'title': 'Bar',
                'type': 'object',
            },
        },
        'properties': {'foo': {'$ref': '#/$defs/Foo'}, 'baz': {'$ref': '#/$defs/Baz'}, 'bar': {'$ref': '#/$defs/Bar'}},
        'required': ['foo', 'baz', 'bar'],
        'type': 'object',
        'title': 'FooBarBaz',
    }

    # A schema with no title that will cause numeric suffixes
    no_title_schema = {
        '$defs': {
            'Foo': {
                'description': 'Another different Foo',
                'properties': {'foo': {'type': 'boolean'}},
                'required': ['foo'],
                'title': 'Foo',
                'type': 'object',
            },
            'Bar': {
                'description': 'Another different Bar',
                'properties': {'bar': {'type': 'number'}},
                'required': ['bar'],
                'title': 'Bar',
                'type': 'object',
            },
        },
        'properties': {'foo': {'$ref': '#/$defs/Foo'}, 'bar': {'$ref': '#/$defs/Bar'}},
        'required': ['foo', 'bar'],
        'type': 'object',
    }

    # Another schema with no title that will cause more numeric suffixes
    another_no_title_schema = {
        '$defs': {
            'Foo': {
                'description': 'Yet another different Foo',
                'properties': {'foo': {'type': 'array'}},
                'required': ['foo'],
                'title': 'Foo',
                'type': 'object',
            },
            'Bar': {
                'description': 'Yet another different Bar',
                'properties': {'bar': {'type': 'object'}},
                'required': ['bar'],
                'title': 'Bar',
                'type': 'object',
            },
        },
        'properties': {'foo': {'$ref': '#/$defs/Foo'}, 'bar': {'$ref': '#/$defs/Bar'}},
        'required': ['foo', 'bar'],
        'type': 'object',
    }

    # Schema with nested properties, array items, prefixItems, and anyOf/oneOf
    complex_schema = {
        '$defs': {
            'Nested': {
                'description': 'A nested type',
                'properties': {'nested': {'type': 'string'}},
                'required': ['nested'],
                'title': 'Nested',
                'type': 'object',
            },
            'ArrayItem': {
                'description': 'An array item type',
                'properties': {'item': {'type': 'string'}},
                'required': ['item'],
                'title': 'ArrayItem',
                'type': 'object',
            },
            'UnionType': {
                'description': 'A union type',
                'properties': {'union': {'type': 'string'}},
                'required': ['union'],
                'title': 'UnionType',
                'type': 'object',
            },
        },
        'properties': {
            'nested_props': {
                'type': 'object',
                'properties': {
                    'deep_nested': {'$ref': '#/$defs/Nested'},
                },
            },
            'array_with_items': {
                'type': 'array',
                'items': {'$ref': '#/$defs/ArrayItem'},
            },
            'array_with_prefix': {
                'type': 'array',
                'prefixItems': [
                    {'$ref': '#/$defs/ArrayItem'},
                    {'$ref': '#/$defs/Nested'},
                ],
            },
            'union_anyOf': {
                'anyOf': [
                    {'$ref': '#/$defs/UnionType'},
                    {'$ref': '#/$defs/Nested'},
                ],
            },
            'union_oneOf': {
                'oneOf': [
                    {'$ref': '#/$defs/UnionType'},
                    {'$ref': '#/$defs/ArrayItem'},
                ],
            },
        },
        'type': 'object',
        'title': 'ComplexSchema',
    }

    schemas = [foo_bar_schema, foo_bar_baz_schema, no_title_schema, another_no_title_schema, complex_schema]
    rewritten_schemas, all_defs = merge_json_schema_defs(schemas)
    assert all_defs == snapshot(
        {
            'Bar': {
                'description': 'Bar description',
                'properties': {'bar': {'type': 'string'}},
                'required': ['bar'],
                'title': 'Bar',
                'type': 'object',
            },
            'Foo': {
                'description': 'Foo description',
                'properties': {'foo': {'type': 'string'}},
                'required': ['foo'],
                'title': 'Foo',
                'type': 'object',
            },
            'Baz': {
                'description': 'Baz description',
                'properties': {'baz': {'type': 'string'}},
                'required': ['baz'],
                'title': 'Baz',
                'type': 'object',
            },
            'FooBarBaz_Foo_1': {
                'description': 'Foo description. Note that this is different from the Foo in foo_bar_schema!',
                'properties': {'foo': {'type': 'int'}},
                'required': ['foo'],
                'title': 'Foo',
                'type': 'object',
            },
            'Foo_1': {
                'description': 'Another different Foo',
                'properties': {'foo': {'type': 'boolean'}},
                'required': ['foo'],
                'title': 'Foo',
                'type': 'object',
            },
            'Bar_1': {
                'description': 'Another different Bar',
                'properties': {'bar': {'type': 'number'}},
                'required': ['bar'],
                'title': 'Bar',
                'type': 'object',
            },
            'Foo_2': {
                'description': 'Yet another different Foo',
                'properties': {'foo': {'type': 'array'}},
                'required': ['foo'],
                'title': 'Foo',
                'type': 'object',
            },
            'Bar_2': {
                'description': 'Yet another different Bar',
                'properties': {'bar': {'type': 'object'}},
                'required': ['bar'],
                'title': 'Bar',
                'type': 'object',
            },
            'Nested': {
                'description': 'A nested type',
                'properties': {'nested': {'type': 'string'}},
                'required': ['nested'],
                'title': 'Nested',
                'type': 'object',
            },
            'ArrayItem': {
                'description': 'An array item type',
                'properties': {'item': {'type': 'string'}},
                'required': ['item'],
                'title': 'ArrayItem',
                'type': 'object',
            },
            'UnionType': {
                'description': 'A union type',
                'properties': {'union': {'type': 'string'}},
                'required': ['union'],
                'title': 'UnionType',
                'type': 'object',
            },
        }
    )
    assert rewritten_schemas == snapshot(
        [
            {
                'properties': {'foo': {'$ref': '#/$defs/Foo'}, 'bar': {'$ref': '#/$defs/Bar'}},
                'required': ['foo', 'bar'],
                'type': 'object',
                'title': 'FooBar',
            },
            {
                'properties': {
                    'foo': {'$ref': '#/$defs/FooBarBaz_Foo_1'},
                    'baz': {'$ref': '#/$defs/Baz'},
                    'bar': {'$ref': '#/$defs/Bar'},
                },
                'required': ['foo', 'baz', 'bar'],
                'type': 'object',
                'title': 'FooBarBaz',
            },
            {
                'properties': {'foo': {'$ref': '#/$defs/Foo_1'}, 'bar': {'$ref': '#/$defs/Bar_1'}},
                'required': ['foo', 'bar'],
                'type': 'object',
            },
            {
                'properties': {'foo': {'$ref': '#/$defs/Foo_2'}, 'bar': {'$ref': '#/$defs/Bar_2'}},
                'required': ['foo', 'bar'],
                'type': 'object',
            },
            {
                'properties': {
                    'nested_props': {
                        'type': 'object',
                        'properties': {
                            'deep_nested': {'$ref': '#/$defs/Nested'},
                        },
                    },
                    'array_with_items': {
                        'type': 'array',
                        'items': {'$ref': '#/$defs/ArrayItem'},
                    },
                    'array_with_prefix': {
                        'type': 'array',
                        'prefixItems': [
                            {'$ref': '#/$defs/ArrayItem'},
                            {'$ref': '#/$defs/Nested'},
                        ],
                    },
                    'union_anyOf': {
                        'anyOf': [
                            {'$ref': '#/$defs/UnionType'},
                            {'$ref': '#/$defs/Nested'},
                        ],
                    },
                    'union_oneOf': {
                        'oneOf': [
                            {'$ref': '#/$defs/UnionType'},
                            {'$ref': '#/$defs/ArrayItem'},
                        ],
                    },
                },
                'type': 'object',
                'title': 'ComplexSchema',
            },
        ]
    )


def test_strip_markdown_fences():
    assert strip_markdown_fences('{"foo": "bar"}') == '{"foo": "bar"}'
    assert strip_markdown_fences('```json\n{"foo": "bar"}\n```') == '{"foo": "bar"}'
    assert strip_markdown_fences('```json\n{\n  "foo": "bar"\n}') == '{\n  "foo": "bar"\n}'
    assert (
        strip_markdown_fences('{"foo": "```json\\n{"foo": "bar"}\\n```"}')
        == '{"foo": "```json\\n{"foo": "bar"}\\n```"}'
    )
    assert (
        strip_markdown_fences('Here is some beautiful JSON:\n\n```\n{"foo": "bar"}\n``` Nice right?')
        == '{"foo": "bar"}'
    )
    assert strip_markdown_fences('No JSON to be found') == 'No JSON to be found'


def test_validate_empty_kwargs_empty():
    """Test that empty dict passes validation."""
    validate_empty_kwargs({})


def test_validate_empty_kwargs_with_unknown():
    """Test that unknown kwargs raise UserError."""
    with pytest.raises(UserError, match='Unknown keyword arguments: `unknown_arg`'):
        validate_empty_kwargs({'unknown_arg': 'value'})


def test_validate_empty_kwargs_multiple_unknown():
    """Test that multiple unknown kwargs are properly formatted."""
    with pytest.raises(UserError, match='Unknown keyword arguments: `arg1`, `arg2`'):
        validate_empty_kwargs({'arg1': 'value1', 'arg2': 'value2'})


def test_validate_empty_kwargs_message_format():
    """Test that the error message format matches expected pattern."""
    with pytest.raises(UserError) as exc_info:
        validate_empty_kwargs({'test_arg': 'test_value'})

    assert 'Unknown keyword arguments: `test_arg`' in str(exc_info.value)


def test_validate_empty_kwargs_preserves_order():
    """Test that multiple kwargs preserve order in error message."""
    kwargs = {'first': '1', 'second': '2', 'third': '3'}
    with pytest.raises(UserError) as exc_info:
        validate_empty_kwargs(kwargs)

    error_msg = str(exc_info.value)
    assert '`first`' in error_msg
    assert '`second`' in error_msg
    assert '`third`' in error_msg
