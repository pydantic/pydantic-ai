from __future__ import annotations as _annotations

import functools
import sys
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import pytest
from _pytest.python_api import RaisesContext
from dirty_equals import HasRepr

from ..conftest import try_import

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup  # pragma: lax no cover
else:
    ExceptionGroup = ExceptionGroup  # pragma: lax no cover


with try_import() as imports_successful:
    from pydantic_evals._utils import (
        UNSET,
        Unset,
        get_unwrapped_function_name,
        is_set,
        task_group_gather,
    )

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


def test_unset():
    """Test Unset singleton."""
    assert isinstance(UNSET, Unset)
    assert UNSET is not Unset()  # note: we might want to change this and make it a true singleton..


def test_is_set():
    """Test is_set function."""
    assert is_set(42) is True
    assert is_set(None) is True
    assert is_set(UNSET) is False


def test_get_unwrapped_function_name_basic():
    """Test get_unwrapped_function_name with basic function."""

    def test_func():
        pass

    assert get_unwrapped_function_name(test_func) == 'test_func'


def test_get_unwrapped_function_name_partial():
    """Test get_unwrapped_function_name with partial function."""

    def test_func(x: int, y: int):
        raise NotImplementedError

    partial_func = partial(test_func, y=42)
    assert get_unwrapped_function_name(partial_func) == 'test_func'


def test_get_unwrapped_function_name_decorated():
    """Test get_unwrapped_function_name with decorated function."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

        return wrapper

    @decorator
    def test_func():
        pass

    assert get_unwrapped_function_name(test_func) == 'test_func'


def test_get_unwrapped_function_name_callable_class():
    """Test get_unwrapped_function_name with callable class."""

    class CallableClass:
        def __call__(self):
            pass

    assert (
        get_unwrapped_function_name(CallableClass())
        == 'test_get_unwrapped_function_name_callable_class.<locals>.CallableClass.__call__'
    )


def test_get_unwrapped_function_name_method():
    """Test get_unwrapped_function_name with method."""

    class TestClass:
        def test_method(self):
            pass

    assert get_unwrapped_function_name(TestClass().test_method) == 'test_method'


def test_get_unwrapped_function_name_error():
    """Test get_unwrapped_function_name with invalid input."""

    class InvalidCallable:
        pass

    with pytest.raises(AttributeError) as exc_info:
        get_unwrapped_function_name(InvalidCallable())  # type: ignore

    assert str(exc_info.value) == "'InvalidCallable' object has no attribute '__name__'"


async def test_task_group_gather():
    """Test task_group_gather function."""

    async def task1():
        return 1

    async def task2():
        return 2

    async def task3():
        return 3

    tasks = [task1, task2, task3]
    results = await task_group_gather(tasks)
    assert results == [1, 2, 3]


async def test_task_group_gather_with_error():
    """Test task_group_gather function with error in one task."""

    async def task1():
        return 1

    async def task2():
        raise ValueError('Task 2 failed')

    async def task3():
        return 3

    tasks = [task1, task2, task3]
    with cast(RaisesContext[ExceptionGroup[Any]], pytest.raises(ExceptionGroup)) as exc_info:
        await task_group_gather(tasks)

    assert exc_info.value == HasRepr(
        repr(ExceptionGroup('unhandled errors in a TaskGroup', [ValueError('Task 2 failed')]))
    )
