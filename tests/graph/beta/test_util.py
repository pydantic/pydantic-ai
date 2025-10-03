"""Tests for pydantic_graph.beta.util module."""

import inspect
from typing import Union

from pydantic_graph.beta.util import (
    Some,
    TypeExpression,
    get_callable_name,
    infer_name,
    unpack_type_expression,
)


def test_type_expression_unpacking():
    """Test TypeExpression wrapper and unpacking."""
    # Test with a direct type
    result = unpack_type_expression(int)
    assert result is int

    # Test with TypeExpression wrapper
    wrapped = TypeExpression[Union[str, int]]
    result = unpack_type_expression(wrapped)
    assert result == Union[str, int]


def test_some_wrapper():
    """Test Some wrapper for Maybe pattern."""
    value = Some(42)
    assert value.value == 42

    none_value = Some(None)
    assert none_value.value is None


def test_get_callable_name():
    """Test extracting names from callables."""

    def my_function():
        pass

    assert get_callable_name(my_function) == 'my_function'

    class MyClass:
        pass

    assert get_callable_name(MyClass) == 'MyClass'

    # Test with object without __name__ attribute
    obj = object()
    name = get_callable_name(obj)
    assert isinstance(name, str)
    assert 'object' in name


def test_infer_name():
    """Test inferring variable names from the calling frame."""
    my_object = object()
    # Depth 1 means we look at the frame calling infer_name
    inferred = infer_name(my_object, depth=1)
    assert inferred == 'my_object'

    # Test with object not in locals
    result = infer_name(object(), depth=1)
    assert result is None


def test_infer_name_no_frame():
    """Test infer_name when frame inspection fails."""
    # This is hard to trigger without mocking, but we can test that the function
    # returns None gracefully when it can't find the object
    some_obj = object()

    # Call with depth that would exceed the call stack
    result = infer_name(some_obj, depth=1000)
    assert result is None


def test_infer_name_from_globals():
    """Test infer_name can find names in globals."""
    # Create an object and put it in globals (simulating module-level variable)
    test_global = object()
    current_frame = inspect.currentframe()
    if current_frame is not None:
        current_frame.f_globals['test_global_obj'] = test_global
        try:
            # Use depth=1 to look in this frame
            result = infer_name(test_global, depth=1)
            assert result == 'test_global_obj'
        finally:
            # Clean up
            del current_frame.f_globals['test_global_obj']


def test_infer_name_locals_vs_globals():
    """Test infer_name prefers locals over globals."""
    test_obj = object()
    current_frame = inspect.currentframe()
    if current_frame is not None:
        # Add to both locals and globals with different names
        current_frame.f_globals['global_name'] = test_obj
        try:
            local_name = test_obj  # This creates a local binding
            result = infer_name(test_obj, depth=1)
            # Should find the local name first
            assert result in ('local_name', 'test_obj')
        finally:
            del current_frame.f_globals['global_name']
