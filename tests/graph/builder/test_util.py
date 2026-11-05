"""Tests for pydantic_graph.util module."""

from pydantic_graph.util import (
    Some,
    TypeExpression,
    get_callable_name,
    unpack_type_expression,
)


def test_type_expression_unpacking():
    """Test TypeExpression wrapper and unpacking."""
    # Test with a direct type
    result = unpack_type_expression(int)
    assert result is int

    # Test with TypeExpression wrapper
    wrapped = TypeExpression[str | int]
    result = unpack_type_expression(wrapped)
    assert result == str | int


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


def test_get_callable_name_does_not_stringify_when_name_is_available():
    """The `str()` fallback must not be evaluated when `__name__` is present."""
    calls = []

    class NamedCallable:
        __name__ = 'named_callable'

        def __call__(self):
            pass

        def __str__(self):
            calls.append('str')
            raise RuntimeError('__str__ should not be called')

        def __repr__(self):
            calls.append('repr')
            raise RuntimeError('__repr__ should not be called')

    assert get_callable_name(NamedCallable()) == 'named_callable'
    assert calls == []
