from __future__ import annotations as _annotations

import asyncio
import inspect
import sys
from collections.abc import Coroutine
from functools import partial
from typing import Any, Callable, TypeVar

from typing_extensions import ParamSpec, TypeIs


class Unset:
    """A singleton to represent an unset value.

    Copied from pydantic_ai/_utils.py.
    """

    pass


UNSET = Unset()
T = TypeVar('T')


def is_set(t_or_unset: T | Unset) -> TypeIs[T]:
    return t_or_unset is not UNSET


def get_unwrapped_function_name(func: Callable[..., Any]) -> str:
    def _unwrap(f: Callable[..., Any]) -> Callable[..., Any]:
        # Unwraps f, also unwrapping partials, for the sake of getting f's name
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


def run_until_complete(coro: Coroutine[None, None, _R]) -> _R:
    if sys.version_info < (3, 11):
        try:
            loop = asyncio.new_event_loop()
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    else:
        with asyncio.runners.Runner(loop_factory=asyncio.new_event_loop) as runner:
            return runner.run(coro)
