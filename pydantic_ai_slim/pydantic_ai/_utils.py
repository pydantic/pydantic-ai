from __future__ import annotations as _annotations

import asyncio
import copy
import functools
import inspect
import re
import sys
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Iterable, Iterator
from concurrent.futures import Executor
from contextlib import asynccontextmanager, contextmanager, suppress
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from functools import partial
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeAlias,
    TypeGuard,
    TypeVar,
    get_args,
    get_origin,
    overload,
)

import anyio
from anyio.to_thread import run_sync

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup as BaseExceptionGroup  # pragma: lax no cover
else:
    BaseExceptionGroup = BaseExceptionGroup  # pragma: lax no cover
from pydantic import BaseModel, TypeAdapter
from pydantic._internal import _decorators, _typing_extra
from pydantic.json_schema import JsonSchemaValue
from typing_extensions import (
    ParamSpec,
    TypeIs,
    is_typeddict,
)
from typing_inspection import typing_objects
from typing_inspection.introspection import is_union_origin

from pydantic_graph._utils import AbstractSpan

from . import exceptions

AbstractSpan = AbstractSpan

if TYPE_CHECKING:
    from pydantic_ai.agent import AgentRun, AgentRunResult
    from pydantic_graph import GraphRun, GraphRunResult

    from . import messages as _messages
    from .tools import ObjectJsonSchema

_P = ParamSpec('_P')
_R = TypeVar('_R')

_disable_threads: ContextVar[bool] = ContextVar('_disable_threads', default=False)
_thread_executor: ContextVar[Executor | None] = ContextVar('_thread_executor', default=None)


@contextmanager
def disable_threads() -> Iterator[None]:
    """Context manager to disable thread-based execution for sync functions.

    Inside this context, sync functions will execute inline rather than
    being sent to a thread pool via [`anyio.to_thread.run_sync`][anyio.to_thread.run_sync].

    This is useful in environments where threading is restricted, such as
    Temporal workflows which use a sandboxed event loop.

    Yields:
        None
    """
    token = _disable_threads.set(True)
    try:
        yield
    finally:
        _disable_threads.reset(token)


@contextmanager
def using_thread_executor(executor: Executor) -> Iterator[None]:
    """Context manager to use a custom executor for running sync functions in threads.

    Inside this context, sync functions will be executed using the provided executor
    via [`asyncio.get_running_loop().run_in_executor()`][asyncio.loop.run_in_executor]
    instead of the default [`anyio.to_thread.run_sync`][anyio.to_thread.run_sync].

    This is useful in long-running servers (e.g. FastAPI) where thread accumulation
    from ephemeral anyio worker threads can be a problem, and you want to use a bounded
    `ThreadPoolExecutor` instead.

    Args:
        executor: The executor to use for running sync functions.

    Yields:
        None
    """
    token = _thread_executor.set(executor)
    try:
        yield
    finally:
        _thread_executor.reset(token)


async def run_in_executor(func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
    if _disable_threads.get():
        return func(*args, **kwargs)

    wrapped_func = partial(func, *args, **kwargs)

    executor = _thread_executor.get()
    if executor is not None:
        loop = asyncio.get_running_loop()
        ctx = copy_context()
        return await loop.run_in_executor(executor, ctx.run, wrapped_func)

    return await run_sync(wrapped_func)


def is_model_like(type_: Any) -> bool:
    """Check if something is a pydantic model, dataclass or typedict.

    These should all generate a JSON Schema with `{"type": "object"}` and therefore be usable directly as
    function parameters.
    """
    return (
        isinstance(type_, type)
        and not isinstance(type_, GenericAlias)
        and (
            issubclass(type_, BaseModel)
            or is_dataclass(type_)  # pyright: ignore[reportUnknownArgumentType]
            or is_typeddict(type_)  # pyright: ignore[reportUnknownArgumentType]
            or getattr(type_, '__is_model_like__', False)  # pyright: ignore[reportUnknownArgumentType]
        )
    )


def check_object_json_schema(schema: JsonSchemaValue) -> ObjectJsonSchema:
    from .exceptions import UserError

    if schema.get('type') == 'object':
        return schema
    elif ref := schema.get('$ref'):
        prefix = '#/$defs/'
        # Return the referenced schema unless it contains additional nested references.
        if (
            ref.startswith(prefix)
            and (resolved := schema.get('$defs', {}).get(ref[len(prefix) :]))
            and resolved.get('type') == 'object'
            and not _contains_ref(resolved)
        ):
            return resolved
        return schema
    else:
        raise UserError('Schema must be an object')


def _contains_ref(obj: JsonSchemaValue | list[JsonSchemaValue]) -> bool:
    """Recursively check if an object contains any $ref keys."""
    items: Iterable[JsonSchemaValue]
    if isinstance(obj, dict):
        if '$ref' in obj:
            return True
        items = obj.values()
    else:
        items = obj
    return any(isinstance(item, dict | list) and _contains_ref(item) for item in items)  # pyright: ignore[reportUnknownArgumentType]


T = TypeVar('T')


@dataclass
class Some(Generic[T]):
    """Analogous to Rust's `Option::Some` type."""

    value: T


Option: TypeAlias = Some[T] | None
"""Analogous to Rust's `Option` type, usage: `Option[Thing]` is equivalent to `Some[Thing] | None`."""


async def gather(*coros: Awaitable[T]) -> list[T]:
    """Run awaitables concurrently via an `anyio` task group and return results in input order.

    Unlike `asyncio.gather`, a failure in one coroutine cancels the rest instead of leaving them
    as orphan background tasks. If exactly one task fails, its exception is re-raised directly to
    match `asyncio.gather`'s shape; multi-failure cases propagate as an `ExceptionGroup`.
    """
    sentinel = Unset()
    results: list[T | Unset] = [sentinel] * len(coros)

    async def _run(index: int, coro: Awaitable[T]) -> None:
        results[index] = await coro

    try:
        async with anyio.create_task_group() as tg:
            for i, coro in enumerate(coros):
                tg.start_soon(_run, i, coro)
    except BaseExceptionGroup as eg:
        if len(eg.exceptions) == 1:
            exc = eg.exceptions[0]
            exc.__suppress_context__ = True
            raise exc
        raise

    final_results: list[T] = []
    for result in results:
        assert not isinstance(result, Unset)
        final_results.append(result)
    return final_results


class Unset:
    """A singleton to represent an unset value."""

    pass


UNSET = Unset()


def is_set(t_or_unset: T | Unset) -> TypeGuard[T]:
    return t_or_unset is not UNSET


async def _cleanup_temporal_group(
    task: asyncio.Task[Any] | None,
    aiterator: AsyncIterator[Any],
) -> None:
    """Clean up pending task and async iterator after group_by_temporal exits."""
    if task:
        task.cancel('Cancelling group_by_temporal pending task')
        with suppress(asyncio.CancelledError, StopAsyncIteration):
            await task
    aclose = getattr(aiterator, 'aclose', None)
    if aclose is not None:  # pragma: no branch
        await aclose()


@asynccontextmanager
async def group_by_temporal(
    aiterable: AsyncIterable[T], soft_max_interval: float | None
) -> AsyncIterator[AsyncIterable[list[T]]]:
    """Group items from an async iterable into lists based on time interval between them.

    Effectively, this debounces the iterator.

    This returns a context manager usable as an iterator so any pending tasks can be cancelled if an error occurs
    during iteration.

    Usage:

    ```python
    async with group_by_temporal(yield_groups(), 0.1) as groups_iter:
        async for groups in groups_iter:
            print(groups)
    ```

    Args:
        aiterable: The async iterable to group.
        soft_max_interval: Maximum interval over which to group items, this should avoid a trickle of items causing
            a group to never be yielded. It's a soft max in the sense that once we're over this time, we yield items
            as soon as `anext(aiter)` returns. If `None`, no grouping/debouncing is performed

    Returns:
        A context manager usable as an async iterable of lists of items produced by the input async iterable.
    """
    # we might wait for the next item more than once, so we store the task to await next time
    task: asyncio.Task[T] | None = None
    aiterator = aiter(aiterable)

    if soft_max_interval is None:

        async def async_iter_groups() -> AsyncIterator[list[T]]:
            async for item in aiterator:
                yield [item]

    else:

        async def async_iter_groups() -> AsyncIterator[list[T]]:
            nonlocal task

            assert soft_max_interval is not None and soft_max_interval >= 0, (
                'soft_max_interval must be a positive number'
            )
            buffer: list[T] = []
            group_start_time = time.monotonic()

            while True:
                if group_start_time is None:
                    # group hasn't started, we just wait for the maximum interval
                    wait_time = soft_max_interval
                else:
                    # wait for the time remaining in the group
                    wait_time = soft_max_interval - (time.monotonic() - group_start_time)

                # if there's no current task, we get the next one
                if task is None:
                    # anext(aiter) returns an Awaitable[T], not a Coroutine which asyncio.create_task expects
                    # so far, this doesn't seem to be a problem
                    task = asyncio.create_task(anext(aiterator))  # pyright: ignore[reportArgumentType,reportUnknownVariableType]

                # we use asyncio.wait to avoid cancelling the coroutine if it's not done
                done, _ = await asyncio.wait((task,), timeout=wait_time)

                if done:
                    # the one task we waited for completed
                    try:
                        item = done.pop().result()
                    except StopAsyncIteration:
                        # if the task raised StopAsyncIteration, we're done iterating
                        if buffer:
                            yield buffer
                        task = None
                        break
                    else:
                        # we got an item, add it to the buffer and set task to None to get the next item
                        buffer.append(item)
                        task = None
                        # if this is the first item in the group, set the group start time
                        if group_start_time is None:
                            group_start_time = time.monotonic()
                elif buffer:
                    # otherwise if the task timeout expired and we have items in the buffer, yield the buffer
                    yield buffer
                    # clear the buffer and reset the group start time ready for the next group
                    buffer = []
                    group_start_time = None

    try:
        yield async_iter_groups()
    finally:
        await _cleanup_temporal_group(task, aiterator)


def sync_anext(iterator: Iterator[T]) -> T:
    """Get the next item from a sync iterator, raising `StopAsyncIteration` if it's exhausted.

    Useful when iterating over a sync iterator in an async context.
    """
    try:
        return next(iterator)
    except StopIteration as e:
        raise StopAsyncIteration() from e


def sync_async_iterator(async_iter: AsyncIterator[T]) -> Iterator[T]:
    loop = get_event_loop()
    while True:
        try:
            yield loop.run_until_complete(anext(async_iter))
        except StopAsyncIteration:
            break


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def guard_tool_call_id(
    t: _messages.ToolCallPart
    | _messages.ToolReturnPart
    | _messages.RetryPromptPart
    | _messages.BuiltinToolCallPart
    | _messages.BuiltinToolReturnPart,
) -> str:
    """Type guard that either returns the tool call id or generates a new one if it's None."""
    return t.tool_call_id or generate_tool_call_id()


TOOL_NAME_SANITIZER = re.compile(r'[^a-zA-Z0-9_-]')
"""Regex matching characters not allowed in tool names by most providers."""


def sanitize_tool_name(name: str) -> str:
    """Replace characters outside `[a-zA-Z0-9_-]` with `_`."""
    return TOOL_NAME_SANITIZER.sub('_', name)


def generate_tool_call_id() -> str:
    """Generate a tool call id.

    Ensure that the tool call id is unique.
    """
    return f'pyd_ai_{uuid.uuid4().hex}'


class PeekableAsyncStream(Generic[T]):
    """Wraps an async iterable of type T and allows peeking at the *next* item without consuming it.

    We only buffer one item at a time (the next item). Once that item is yielded, it is discarded.
    This is a single-pass stream.
    """

    def __init__(self, source: AsyncIterable[T]):
        self._source = source
        self._source_iter: AsyncIterator[T] | None = None
        self._buffer: T | Unset = UNSET
        self._exhausted = False

    async def peek(self) -> T | Unset:
        """Returns the next item that would be yielded without consuming it.

        Returns None if the stream is exhausted.
        """
        if self._exhausted:
            return UNSET

        # If we already have a buffered item, just return it.
        if not isinstance(self._buffer, Unset):
            return self._buffer

        # Otherwise, we need to fetch the next item from the underlying iterator.
        if self._source_iter is None:
            self._source_iter = aiter(self._source)

        try:
            self._buffer = await anext(self._source_iter)
        except StopAsyncIteration:
            self._exhausted = True
            return UNSET

        return self._buffer

    async def is_exhausted(self) -> bool:
        """Returns True if the stream is exhausted, False otherwise."""
        return isinstance(await self.peek(), Unset)

    def __aiter__(self) -> AsyncIterator[T]:
        # For a single-pass iteration, we can return self as the iterator.
        return self

    async def __anext__(self) -> T:
        """Yields the buffered item if present, otherwise fetches the next item from the underlying source.

        Raises StopAsyncIteration if the stream is exhausted.
        """
        if self._exhausted:
            raise StopAsyncIteration

        # If we have a buffered item, yield it.
        if not isinstance(self._buffer, Unset):
            item = self._buffer
            self._buffer = UNSET
            return item

        # Otherwise, fetch the next item from the source.
        if self._source_iter is None:
            self._source_iter = aiter(self._source)

        try:
            return await anext(self._source_iter)
        except StopAsyncIteration:
            self._exhausted = True
            raise


def get_traceparent(x: AgentRun | AgentRunResult | GraphRun | GraphRunResult) -> str:
    return x._traceparent(required=False) or ''  # type: ignore[reportPrivateUsage]


def dataclasses_no_defaults_repr(self: Any) -> str:
    """Exclude fields with values equal to the field default."""
    kv_pairs = (
        f'{f.name}={getattr(self, f.name)!r}' for f in fields(self) if f.repr and getattr(self, f.name) != f.default
    )
    return f'{self.__class__.__qualname__}({", ".join(kv_pairs)})'


_datetime_ta = TypeAdapter(datetime)


def number_to_datetime(x: int | float) -> datetime:
    return _datetime_ta.validate_python(x)


AwaitableCallable = Callable[..., Awaitable[T]]


@overload
def is_async_callable(obj: AwaitableCallable[T]) -> TypeIs[AwaitableCallable[T]]: ...


@overload
def is_async_callable(obj: Any) -> TypeIs[AwaitableCallable[Any]]: ...


def is_async_callable(obj: Any) -> Any:
    """Correctly check if a callable is async.

    This function was copied from Starlette:
    https://github.com/encode/starlette/blob/78da9b9e218ab289117df7d62aee200ed4c59617/starlette/_utils.py#L36-L40
    """
    while isinstance(obj, functools.partial):
        obj = obj.func

    return inspect.iscoroutinefunction(obj) or (callable(obj) and inspect.iscoroutinefunction(obj.__call__))


def takes_run_context(callable_obj: Callable[..., Any]) -> bool:
    """Check if a callable takes a `RunContext` as its first argument.

    Args:
        callable_obj: The callable to check.

    Returns:
        `True` if the callable takes a `RunContext` as first argument, `False` otherwise.
    """
    from ._run_context import RunContext

    first_param_type = get_first_param_type(callable_obj)
    if first_param_type is None:
        return False
    return first_param_type is RunContext or get_origin(first_param_type) is RunContext


def get_first_param_type(callable_obj: Callable[..., Any]) -> Any | None:
    """Get the type annotation of the first parameter of a callable.

    Handles regular functions, methods, and callable classes with __call__.
    Uses Pydantic internals to properly resolve type hints including forward references.

    Args:
        callable_obj: The callable to inspect.

    Returns:
        The type annotation of the first parameter, or None if it cannot be determined.
    """
    try:
        sig = inspect.signature(callable_obj)
    except ValueError:
        return None

    try:
        first_param_name = next(iter(sig.parameters.keys()))
    except StopIteration:
        return None

    # See https://github.com/pydantic/pydantic/pull/11451 for a similar implementation in Pydantic
    callable_for_hints = callable_obj
    if not isinstance(callable_obj, _decorators._function_like):  # pyright: ignore[reportPrivateUsage]
        call_func = getattr(type(callable_obj), '__call__', None)
        if call_func is not None:
            callable_for_hints = call_func
        else:
            return None  # pragma: no cover

    try:
        type_hints = _typing_extra.get_function_type_hints(_decorators.unwrap_wrapped_function(callable_for_hints))
    except (NameError, TypeError, AttributeError):
        return None

    return type_hints.get(first_param_name)


def get_function_type_hints(func: Any) -> dict[str, Any]:
    """Resolve type hints for a function, including forward references.

    Wraps `pydantic._internal._typing_extra.get_function_type_hints` so callers
    don't need to import Pydantic internals directly.
    """
    return _typing_extra.get_function_type_hints(func)


def _update_mapped_json_schema_refs(s: dict[str, Any], name_mapping: dict[str, str]) -> None:
    """Update $refs in a schema to use the new names from name_mapping."""
    if '$ref' in s:
        ref = s['$ref']
        if ref.startswith('#/$defs/'):  # pragma: no branch
            original_name = ref[8:]  # Remove '#/$defs/'
            new_name = name_mapping.get(original_name, original_name)
            s['$ref'] = f'#/$defs/{new_name}'

    # Recursively update refs in properties
    if 'properties' in s:
        props: dict[str, dict[str, Any]] = s['properties']
        for prop in props.values():
            _update_mapped_json_schema_refs(prop, name_mapping)

    # Handle arrays
    if 'items' in s and isinstance(s['items'], dict):
        items: dict[str, Any] = s['items']  # pyright: ignore[reportUnknownVariableType]
        _update_mapped_json_schema_refs(items, name_mapping)
    if 'prefixItems' in s:
        prefix_items: list[dict[str, Any]] = s['prefixItems']
        for item in prefix_items:
            _update_mapped_json_schema_refs(item, name_mapping)

    # Handle additionalProperties
    if 'additionalProperties' in s and isinstance(s['additionalProperties'], dict):
        additional_props: dict[str, Any] = s['additionalProperties']  # pyright: ignore[reportUnknownVariableType]
        _update_mapped_json_schema_refs(additional_props, name_mapping)

    # Handle unions and composition keywords
    for keyword in ['anyOf', 'oneOf', 'allOf']:
        if keyword in s:
            keyword_items: list[dict[str, Any]] = s[keyword]
            for item in keyword_items:
                _update_mapped_json_schema_refs(item, name_mapping)

    # Handle negation
    if 'not' in s and isinstance(s['not'], dict):
        not_schema: dict[str, Any] = s['not']  # pyright: ignore[reportUnknownVariableType]
        _update_mapped_json_schema_refs(not_schema, name_mapping)


def _unique_def_name(name: str, schema: dict[str, Any], all_defs: dict[str, dict[str, Any]]) -> str:
    """Generate a unique definition name by appending the schema title and/or a numeric suffix."""
    new_name = name
    if title := schema.get('title'):
        new_name = f'{title}_{name}'

    i = 1
    original_new_name = new_name
    new_name = f'{new_name}_{i}'
    while new_name in all_defs:
        i += 1
        new_name = f'{original_new_name}_{i}'
    return new_name


def merge_json_schema_defs(schemas: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Merges the `$defs` from different JSON schemas into a single deduplicated `$defs`, handling name collisions of `$defs` that are not the same, and rewrites `$ref`s to point to the new `$defs`.

    Returns a tuple of the rewritten schemas and a dictionary of the new `$defs`.
    """
    all_defs: dict[str, dict[str, Any]] = {}
    rewritten_schemas: list[dict[str, Any]] = []

    for schema in schemas:
        if '$defs' not in schema:
            rewritten_schemas.append(schema)
            continue

        schema = schema.copy()
        defs = schema.pop('$defs', None)
        schema_name_mapping: dict[str, str] = {}

        # Process definitions and build mapping
        for name, def_schema in defs.items():
            if name not in all_defs:
                all_defs[name] = def_schema
                schema_name_mapping[name] = name
            elif def_schema != all_defs[name]:
                # Different def with same name — assign a unique name
                schema_name_mapping[name] = _unique_def_name(name, schema, all_defs)
                all_defs[schema_name_mapping[name]] = def_schema
            # else: structurally equal — handled below

        # Defs that are structurally equal (same dict) may still be semantically
        # different if they contain $refs that point to defs that were renamed in
        # this schema. E.g. both schemas have Wrapper={$ref Inner}, but their
        # Inner defs differ, so Schema B's Inner was renamed to Inner_1. The shared
        # Wrapper is not actually equal — Schema B needs its own copy with updated refs.
        # Loop until stable, since creating a copy can trigger further copies
        # in defs that reference it (transitive chains).
        changed = True
        while changed:
            changed = False
            for name, def_schema in defs.items():
                if name not in schema_name_mapping:
                    updated = copy.deepcopy(def_schema)
                    _update_mapped_json_schema_refs(updated, schema_name_mapping)
                    if updated != def_schema:
                        schema_name_mapping[name] = _unique_def_name(name, schema, all_defs)
                        all_defs[schema_name_mapping[name]] = updated
                        changed = True
                    else:
                        schema_name_mapping[name] = name

        # Update refs inside definitions so internal cross-references
        # (e.g. Outer referencing Inner which was renamed to Inner_1) are corrected.
        for new_name in schema_name_mapping.values():
            _update_mapped_json_schema_refs(all_defs[new_name], schema_name_mapping)

        _update_mapped_json_schema_refs(schema, schema_name_mapping)
        rewritten_schemas.append(schema)

    return rewritten_schemas, all_defs


def validate_empty_kwargs(_kwargs: dict[str, Any]) -> None:
    """Validate that no unknown kwargs remain after processing.

    Args:
        _kwargs: Dictionary of remaining kwargs after specific ones have been processed.

    Raises:
        UserError: If any unknown kwargs remain.
    """
    if _kwargs:
        unknown_kwargs = ', '.join(f'`{k}`' for k in _kwargs.keys())
        raise exceptions.UserError(f'Unknown keyword arguments: {unknown_kwargs}')


_MARKDOWN_FENCES_PATTERN = re.compile(r'```(?:\w+)?\n(\{.*?\})\s*(?:\n?```|\Z)', flags=re.DOTALL)


def strip_markdown_fences(text: str) -> str:
    if text.startswith('{'):
        return text

    match = re.search(_MARKDOWN_FENCES_PATTERN, text)
    if match:
        return match.group(1)

    return text


def _unwrap_annotated(tp: Any) -> Any:
    origin = get_origin(tp)
    while typing_objects.is_annotated(origin):
        tp = tp.__origin__
        origin = get_origin(tp)
    return tp


def get_union_args(tp: Any) -> tuple[Any, ...]:
    """Extract the arguments of a Union type if `tp` is a union, otherwise return an empty tuple."""
    if typing_objects.is_typealiastype(tp):
        tp = tp.__value__

    tp = _unwrap_annotated(tp)
    origin = get_origin(tp)
    if is_union_origin(origin):
        return tuple(_unwrap_annotated(arg) for arg in get_args(tp))
    else:
        return ()


def get_event_loop():
    try:
        event_loop = asyncio.get_event_loop()
    except RuntimeError:  # pragma: lax no cover
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
    return event_loop


def is_str_dict(obj: Any) -> TypeGuard[dict[str, Any]]:
    """Check if obj is a dict, narrowing the type to `dict[str, Any]`."""
    return isinstance(obj, dict)


def is_text_like_media_type(media_type: str) -> bool:
    """Check if a media type represents text-like content.

    Returns True for `text/*`, JSON, XML, YAML, and their structured syntax suffixes.
    """
    return (
        media_type.startswith('text/')
        or media_type == 'application/json'
        or media_type.endswith('+json')
        or media_type == 'application/xml'
        or media_type.endswith('+xml')
        or media_type in ('application/x-yaml', 'application/yaml')
    )
