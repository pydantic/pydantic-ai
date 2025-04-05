from __future__ import annotations as _annotations

import asyncio
import inspect
from collections.abc import Awaitable, Sequence
from functools import partial
from typing import Any, Callable, TypeVar

import anyio
from typing_extensions import ParamSpec, TypeIs


class Unset:
    """A singleton to represent an unset value.

    Used to distinguish between explicitly set `None` values and values that were never set.

    Copied from pydantic_ai/_utils.py.
    """

    pass


UNSET = Unset()
"""Singleton instance of the [`Unset`][pydantic_evals._utils.Unset] class to represent unset values."""

T = TypeVar('T')


def is_set(t_or_unset: T | Unset) -> TypeIs[T]:
    """Check if a value is set (not the UNSET singleton).

    Args:
        t_or_unset: The value to check, which may be the UNSET singleton or a regular value.

    Returns:
        True if the value is not UNSET, narrowing the type to T in a type-aware way.
    """
    return t_or_unset is not UNSET


def get_unwrapped_function_name(func: Callable[..., Any]) -> str:
    """Get the name of a function, unwrapping partials and decorators.

    Args:
        func: The function to get the name of.

    Returns:
        The name of the function.

    Raises:
        AttributeError: If the function doesn't have a __name__ attribute and isn't a method.
    """

    def _unwrap(f: Callable[..., Any]) -> Callable[..., Any]:
        """Unwraps f, also unwrapping partials, for the sake of getting f's name."""
        if isinstance(f, partial):
            return _unwrap(f.func)
        return inspect.unwrap(f)

    try:
        return _unwrap(func).__name__
    except AttributeError as e:
        # Handle instances of types with `__call__` as a method
        if inspect.ismethod(getattr(func, '__call__', None)):
            return f'{type(func).__qualname__}.__call__'
        else:
            raise e


_P = ParamSpec('_P')
_R = TypeVar('_R')


def get_event_loop():
    try:
        event_loop = asyncio.get_event_loop()
    except RuntimeError:  # pragma: no cover
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
    return event_loop


async def task_group_gather(tasks: Sequence[Callable[[], Awaitable[T]]]) -> list[T]:
    """Run multiple awaitable callables concurrently using an AnyIO task group.

    Args:
        tasks: A list of no-argument callables that return awaitable objects.

    Returns:
        A list of results in the same order as the input tasks.
    """
    results: list[T] = [None] * len(tasks)  # type: ignore

    async def _run_task(tsk: Callable[[], Awaitable[T]], index: int) -> None:
        """Helper function to run a task and store the result in the correct index."""
        results[index] = await tsk()

    async with anyio.create_task_group() as tg:
        for i, task in enumerate(tasks):
            tg.start_soon(_run_task, task, i)

    return results
