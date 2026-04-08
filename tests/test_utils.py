from __future__ import annotations as _annotations

import asyncio
import contextvars
import functools
import os
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import distributions

import pytest

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
    using_thread_executor,
    validate_empty_kwargs,
)

from ._inline_snapshot import snapshot
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


async def test_run_in_executor_with_custom_executor() -> None:
    main_thread = threading.current_thread()

    def sync_func() -> threading.Thread:
        return threading.current_thread()

    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='custom-pool')
    try:
        with using_thread_executor(executor):
            result = await run_in_executor(sync_func)
            assert result is not main_thread
            assert result.name.startswith('custom-pool')
    finally:
        executor.shutdown(wait=True)


async def test_run_in_executor_custom_executor_preserves_context_vars() -> None:
    ctx_var = contextvars.ContextVar('test_var', default='default')
    ctx_var.set('custom_value')

    executor = ThreadPoolExecutor(max_workers=2)
    try:
        with using_thread_executor(executor):
            result = await run_in_executor(ctx_var.get)
            assert result == 'custom_value'
    finally:
        executor.shutdown(wait=True)


async def test_disable_threads_takes_priority_over_custom_executor() -> None:
    from pydantic_ai._utils import disable_threads

    main_thread = threading.current_thread()

    def check_thread() -> threading.Thread:
        return threading.current_thread()

    executor = ThreadPoolExecutor(max_workers=2)
    try:
        with using_thread_executor(executor):
            with disable_threads():
                result = await run_in_executor(check_thread)
                assert result is main_thread
    finally:
        executor.shutdown(wait=True)


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


def test_merge_json_schema_defs_internal_refs_in_renamed_defs():
    """When defs are renamed due to collisions, internal $refs within those defs must also be updated."""
    schema_a = {
        '$defs': {
            'Inner': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
            'Outer': {
                'type': 'object',
                'properties': {'inner': {'$ref': '#/$defs/Inner'}, 'extra_a': {'type': 'string'}},
            },
        },
        'properties': {'outer': {'$ref': '#/$defs/Outer'}},
        'type': 'object',
        'title': 'SchemaA',
    }
    schema_b = {
        '$defs': {
            'Inner': {'type': 'object', 'properties': {'x': {'type': 'integer'}}},
            'Outer': {
                'type': 'object',
                'properties': {'inner': {'$ref': '#/$defs/Inner'}, 'extra_b': {'type': 'number'}},
            },
        },
        'properties': {'outer': {'$ref': '#/$defs/Outer'}},
        'type': 'object',
        'title': 'SchemaB',
    }

    rewritten_schemas, all_defs = merge_json_schema_defs([schema_a, schema_b])

    # SchemaB's Outer was renamed to SchemaB_Outer_1, and its internal $ref to Inner
    # must now point to SchemaB_Inner_1 (not the original Inner from SchemaA)
    assert all_defs == snapshot(
        {
            'Inner': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
            'Outer': {
                'type': 'object',
                'properties': {'inner': {'$ref': '#/$defs/Inner'}, 'extra_a': {'type': 'string'}},
            },
            'SchemaB_Inner_1': {'type': 'object', 'properties': {'x': {'type': 'integer'}}},
            'SchemaB_Outer_1': {
                'type': 'object',
                'properties': {'inner': {'$ref': '#/$defs/SchemaB_Inner_1'}, 'extra_b': {'type': 'number'}},
            },
        }
    )
    assert rewritten_schemas == snapshot(
        [
            {
                'properties': {'outer': {'$ref': '#/$defs/Outer'}},
                'type': 'object',
                'title': 'SchemaA',
            },
            {
                'properties': {'outer': {'$ref': '#/$defs/SchemaB_Outer_1'}},
                'type': 'object',
                'title': 'SchemaB',
            },
        ]
    )


def test_merge_json_schema_defs_non_renamed_def_refs_renamed_def():
    """A non-renamed def that references a renamed def must also have its $refs updated."""
    schema_a = {
        '$defs': {
            'Inner': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
        },
        'properties': {'inner': {'$ref': '#/$defs/Inner'}},
        'type': 'object',
        'title': 'SchemaA',
    }
    schema_b = {
        '$defs': {
            'Inner': {'type': 'object', 'properties': {'x': {'type': 'integer'}}},
            'Wrapper': {'type': 'object', 'properties': {'inner': {'$ref': '#/$defs/Inner'}}},
        },
        'properties': {'wrapper': {'$ref': '#/$defs/Wrapper'}},
        'type': 'object',
        'title': 'SchemaB',
    }

    rewritten_schemas, all_defs = merge_json_schema_defs([schema_a, schema_b])

    # Wrapper is new (no collision), but its $ref to Inner must be rewritten
    # to SchemaB_Inner_1 because SchemaB's Inner was renamed
    assert all_defs == snapshot(
        {
            'Inner': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
            'SchemaB_Inner_1': {'type': 'object', 'properties': {'x': {'type': 'integer'}}},
            'Wrapper': {
                'type': 'object',
                'properties': {'inner': {'$ref': '#/$defs/SchemaB_Inner_1'}},
            },
        }
    )
    assert rewritten_schemas == snapshot(
        [
            {
                'properties': {'inner': {'$ref': '#/$defs/Inner'}},
                'type': 'object',
                'title': 'SchemaA',
            },
            {
                'properties': {'wrapper': {'$ref': '#/$defs/Wrapper'}},
                'type': 'object',
                'title': 'SchemaB',
            },
        ]
    )


def test_merge_json_schema_defs_additional_properties_allof_not():
    """$refs under additionalProperties, allOf, and not must be rewritten during merge."""
    schema_a = {
        '$defs': {
            'Value': {'type': 'object', 'properties': {'v': {'type': 'string'}}},
            'Base': {'type': 'object', 'properties': {'b': {'type': 'string'}}},
            'Excluded': {'type': 'object', 'properties': {'e': {'type': 'string'}}},
        },
        'properties': {
            'map': {'type': 'object', 'additionalProperties': {'$ref': '#/$defs/Value'}},
            'composed': {'allOf': [{'$ref': '#/$defs/Base'}, {'$ref': '#/$defs/Value'}]},
            'excluded': {'not': {'$ref': '#/$defs/Excluded'}},
        },
        'type': 'object',
        'title': 'SchemaA',
    }
    schema_b = {
        '$defs': {
            'Value': {'type': 'object', 'properties': {'v': {'type': 'integer'}}},
            'Base': {'type': 'object', 'properties': {'b': {'type': 'integer'}}},
            'Excluded': {'type': 'object', 'properties': {'e': {'type': 'integer'}}},
        },
        'properties': {
            'map': {'type': 'object', 'additionalProperties': {'$ref': '#/$defs/Value'}},
            'composed': {'allOf': [{'$ref': '#/$defs/Base'}, {'$ref': '#/$defs/Value'}]},
            'excluded': {'not': {'$ref': '#/$defs/Excluded'}},
        },
        'type': 'object',
        'title': 'SchemaB',
    }

    rewritten_schemas, _ = merge_json_schema_defs([schema_a, schema_b])

    # SchemaB's refs should all be rewritten to the renamed defs
    assert rewritten_schemas[1] == snapshot(
        {
            'properties': {
                'map': {'type': 'object', 'additionalProperties': {'$ref': '#/$defs/SchemaB_Value_1'}},
                'composed': {'allOf': [{'$ref': '#/$defs/SchemaB_Base_1'}, {'$ref': '#/$defs/SchemaB_Value_1'}]},
                'excluded': {'not': {'$ref': '#/$defs/SchemaB_Excluded_1'}},
            },
            'type': 'object',
            'title': 'SchemaB',
        }
    )


def test_merge_json_schema_defs_structurally_equal_with_different_ref_targets():
    """Defs that are structurally equal but whose $refs resolve to different types need separate copies."""
    schema_a = {
        '$defs': {
            'Inner': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
            'Wrapper': {'type': 'object', 'properties': {'inner': {'$ref': '#/$defs/Inner'}}},
        },
        'properties': {'wrapper': {'$ref': '#/$defs/Wrapper'}},
        'type': 'object',
        'title': 'SchemaA',
    }
    schema_b = {
        '$defs': {
            'Inner': {'type': 'object', 'properties': {'x': {'type': 'integer'}}},
            'Wrapper': {'type': 'object', 'properties': {'inner': {'$ref': '#/$defs/Inner'}}},
        },
        'properties': {'wrapper': {'$ref': '#/$defs/Wrapper'}},
        'type': 'object',
        'title': 'SchemaB',
    }

    rewritten_schemas, all_defs = merge_json_schema_defs([schema_a, schema_b])

    # Both Wrappers are structurally identical ({$ref: Inner}), but their Inner
    # defs differ, so SchemaB needs its own Wrapper copy with updated refs.
    assert all_defs == snapshot(
        {
            'Inner': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
            'Wrapper': {'type': 'object', 'properties': {'inner': {'$ref': '#/$defs/Inner'}}},
            'SchemaB_Inner_1': {'type': 'object', 'properties': {'x': {'type': 'integer'}}},
            'SchemaB_Wrapper_1': {
                'type': 'object',
                'properties': {'inner': {'$ref': '#/$defs/SchemaB_Inner_1'}},
            },
        }
    )
    assert rewritten_schemas == snapshot(
        [
            {
                'properties': {'wrapper': {'$ref': '#/$defs/Wrapper'}},
                'type': 'object',
                'title': 'SchemaA',
            },
            {
                'properties': {'wrapper': {'$ref': '#/$defs/SchemaB_Wrapper_1'}},
                'type': 'object',
                'title': 'SchemaB',
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
    # Content after closing fence with braces should not be captured (issue #4397)
    assert strip_markdown_fences('```json\n{"a": 1}\n```\nContext: {"b": 2}') == '{"a": 1}'
    assert (
        strip_markdown_fences('```json\n{"result": "pass"}\n```\nThis matches schema {"type": "object"}')
        == '{"result": "pass"}'
    )
    # Nested JSON objects should still be fully captured
    assert strip_markdown_fences('```json\n{"nested": {"key": "value"}}\n```') == '{"nested": {"key": "value"}}'
    assert strip_markdown_fences('```json\n{"a": {"b": {"c": 1}}}\n```') == '{"a": {"b": {"c": 1}}}'


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
