import inspect
from dataclasses import dataclass
from typing import Any, Generic, cast, get_args, get_origin

from typing_extensions import TypeAliasType, TypeVar

T = TypeVar('T', infer_variance=True)


class TypeExpression(Generic[T]):
    """This is a workaround for the lack of TypeForm.

    This is used in places that require an argument of type `type[T]` when you want to use a `T` that type checkers
    don't allow in this position, such as `Any`, `Union[...]`, or `Literal[...]`. In that case, you can just use e.g.
    `output_type=TypeExpression[Union[...]]` instead of `output_type=Union[...]`.
    """

    pass


TypeOrTypeExpression = TypeAliasType('TypeOrTypeExpression', type[TypeExpression[T]] | type[T], type_params=(T,))
"""This is used to allow types directly when compatible with typecheckers, but also allow TypeExpression[T] to be used.

The correct type should get inferred either way.
"""


def unpack_type_expression(type_: TypeOrTypeExpression[T]) -> type[T]:
    """Unpack the type expression."""
    if get_origin(type_) is TypeExpression:
        return get_args(type_)[0]
    return cast(type[T], type_)


@dataclass
class Some(Generic[T]):
    """A marker that a value is present. Like a monadic version of `Optional`."""

    value: T


Maybe = TypeAliasType(
    'Maybe', Some[T] | None, type_params=(T,)
)  # like optional, but you can tell the difference between "no value" and "value is None"


def get_callable_name(callable_: Any) -> str:
    """Get the name to use for a callable."""
    # TODO(P2): Do we need to extend this logic? E.g., for instances of classes defining `__call__`?
    return getattr(callable_, '__name__', str(callable_))


# TODO(P3): Use or remove this
def infer_name(obj: Any, *, depth: int) -> str | None:
    """Infer the name of `obj` from the call frame.

    Usage should generally look like `infer_name(self, depth=2)` or similar.
    """
    target_frame = inspect.currentframe()
    if target_frame is None:
        return None
    for _ in range(depth):
        target_frame = target_frame.f_back
        if target_frame is None:
            return None

    for name, item in target_frame.f_locals.items():
        if item is obj:
            return name

    if target_frame.f_locals != target_frame.f_globals:
        # if we couldn't find the agent in locals and globals are a different dict, try globals
        for name, item in target_frame.f_globals.items():
            if item is obj:
                return name

    return None
