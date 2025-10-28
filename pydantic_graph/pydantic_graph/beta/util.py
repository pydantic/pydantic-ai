"""Utility types and functions for type manipulation and introspection.

This module provides helper classes and functions for working with Python's type system,
including workarounds for type checker limitations and utilities for runtime type inspection.
"""

from dataclasses import dataclass
from typing import Any, Generic, cast, get_args, get_origin

from typing_extensions import TypeAliasType, TypeVar

T = TypeVar('T', infer_variance=True)
"""Generic type variable with inferred variance."""


class TypeExpression(Generic[T]):
    """A workaround for type checker limitations when using complex type expressions.

        This class serves as a wrapper for types that cannot normally be used in positions
    requiring `type[T]`, such as `Any`, `Union[...]`, or `Literal[...]`. It provides a
        way to pass these complex type expressions to functions expecting concrete types.

    Example:
            Instead of `output_type=Union[str, int]` (which may cause type errors),
            use `output_type=TypeExpression[Union[str, int]]`.

    Note:
            This is a workaround for the lack of TypeForm in the Python type system.
    """

    pass


TypeOrTypeExpression = TypeAliasType('TypeOrTypeExpression', type[TypeExpression[T]] | type[T], type_params=(T,))
"""Type alias allowing both direct types and TypeExpression wrappers.

This alias enables functions to accept either regular types (when compatible with type checkers)
or TypeExpression wrappers for complex type expressions. The correct type should be inferred
automatically in either case.
"""


def unpack_type_expression(type_: TypeOrTypeExpression[T]) -> type[T]:
    """Extract the actual type from a TypeExpression wrapper or return the type directly.

    Args:
        type_: Either a direct type or a TypeExpression wrapper.

    Returns:
        The unwrapped type, ready for use in runtime type operations.
    """
    if get_origin(type_) is TypeExpression:
        return get_args(type_)[0]
    return cast(type[T], type_)


@dataclass
class Some(Generic[T]):
    """Container for explicitly present values in Maybe type pattern.

    This class represents a value that is definitely present, as opposed to None.
    It's part of the Maybe pattern, similar to Option/Maybe in functional programming,
    allowing distinction between "no value" (None) and "value is None" (Some(None)).
    """

    value: T
    """The wrapped value."""


Maybe = TypeAliasType('Maybe', Some[T] | None, type_params=(T,))
"""Optional-like type that distinguishes between absence and None values.

Unlike Optional[T], Maybe[T] can differentiate between:
- No value present: represented as None
- Value is None: represented as Some(None)

This is particularly useful when None is a valid value in your domain.
"""


def get_callable_name(callable_: Any) -> str:
    """Extract a human-readable name from a callable object.

    Args:
        callable_: Any callable object (function, method, class, etc.).

    Returns:
        The callable's __name__ attribute if available, otherwise its string representation.
    """
    return getattr(callable_, '__name__', str(callable_))
