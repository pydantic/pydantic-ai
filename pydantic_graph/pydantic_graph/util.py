"""Utility types and functions for type manipulation and introspection.

This module provides helper classes and functions for working with Python's type system,
including workarounds for type checker limitations and utilities for runtime type inspection.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Generic, cast, get_args, get_origin

from typing_extensions import TypeAliasType, TypeForm, TypeVar

T = TypeVar('T', infer_variance=True)
"""Generic type variable with inferred variance."""


class TypeExpression(Generic[T]):
    """Deprecated wrapper for passing complex type expressions to the graph builder.

    Before type checkers supported [PEP 747](https://peps.python.org/pep-0747/) `TypeForm`, type expressions
    that aren't classes (such as `int | str` or `Literal['a']`) could not be passed to parameters annotated
    with `type[T]` without a type error, so they had to be wrapped as `TypeExpression[int | str]`.

    Those parameters now accept any type expression directly, so this wrapper is no longer needed:
    use `g.match(Literal['a'])` instead of `g.match(TypeExpression[Literal['a']])`.

    Wrapped type expressions are still unwrapped at runtime, but subscripting this class emits a `DeprecationWarning`.
    """

    def __class_getitem__(cls, item: Any) -> Any:
        warnings.warn(
            '`TypeExpression` is deprecated, pass the type expression directly instead of wrapping it.',
            DeprecationWarning,
            stacklevel=2,
        )
        # `Generic.__class_getitem__` isn't declared in typeshed, so pyright can't resolve the `super()` call.
        return super().__class_getitem__(item)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]


TypeOrTypeExpression = TypeAliasType('TypeOrTypeExpression', TypeForm[T], type_params=(T,))
"""Deprecated alias for `TypeForm[T]`, kept for backwards compatibility.

Use `typing_extensions.TypeForm` directly instead.
"""


def unpack_type_expression(type_: TypeForm[T]) -> TypeForm[T]:
    """Unwrap a deprecated [`TypeExpression`][pydantic_graph.util.TypeExpression] wrapper, or return the type expression directly.

    Args:
        type_: A type expression, possibly wrapped in a `TypeExpression`.

    Returns:
        The unwrapped type expression, ready for use in runtime type operations.
    """
    if get_origin(type_) is TypeExpression:
        return cast(TypeForm[T], get_args(type_)[0])
    return type_


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
