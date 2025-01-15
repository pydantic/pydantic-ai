from __future__ import annotations as _annotations

import sys
import types
from datetime import datetime, timezone
from typing import Annotated, Any, TypeVar, Union, get_args, get_origin

import typing_extensions


def get_union_args(tp: Any) -> tuple[Any, ...]:
    """Extract the arguments of a Union type if `response_type` is a union, otherwise return the original type."""
    # similar to `pydantic_ai_slim/pydantic_ai/_result.py:get_union_args`
    if isinstance(tp, typing_extensions.TypeAliasType):
        tp = tp.__value__

    origin = get_origin(tp)
    if origin_is_union(origin):
        return get_args(tp)
    else:
        return (tp,)


def unpack_annotated(tp: Any) -> tuple[Any, list[Any]]:
    """Strip `Annotated` from the type if present.

    Returns:
        `(tp argument, ())` if not annotated, otherwise `(stripped type, annotations)`.
    """
    origin = get_origin(tp)
    if origin is Annotated or origin is typing_extensions.Annotated:
        inner_tp, *args = get_args(tp)
        return inner_tp, args
    else:
        return tp, []


def is_never(tp: Any) -> bool:
    """Check if a type is `Never`."""
    if tp is typing_extensions.Never:
        return True
    elif typing_never := getattr(typing_extensions, 'Never', None):
        return tp is typing_never
    else:
        return False


# same as `pydantic_ai_slim/pydantic_ai/_result.py:origin_is_union`
if sys.version_info < (3, 10):

    def origin_is_union(tp: type[Any] | None) -> bool:
        return tp is Union

else:

    def origin_is_union(tp: type[Any] | None) -> bool:
        return tp is Union or tp is types.UnionType


def comma_and(items: list[str]) -> str:
    """Join with a comma and 'and' for the last item."""
    if len(items) == 1:
        return items[0]
    else:
        # oxford comma ¯\_(ツ)_/¯
        return ', '.join(items[:-1]) + ', and ' + items[-1]


def get_parent_namespace(frame: types.FrameType | None) -> dict[str, Any] | None:
    """Attempt to get the namespace where the graph was defined.

    If the graph is defined with generics `Graph[a, b]` then another frame is inserted, and we have to skip that
    to get the correct namespace.
    """
    if frame is not None:
        if back := frame.f_back:
            if back.f_code.co_filename.endswith('/typing.py'):
                return get_parent_namespace(back)
            else:
                return back.f_locals


def now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


class Unset:
    """A singleton to represent an unset value.

    Copied from pydantic_ai/_utils.py.
    """

    pass


UNSET = Unset()
T = TypeVar('T')


def is_set(t_or_unset: T | Unset) -> typing_extensions.TypeGuard[T]:
    return t_or_unset is not UNSET
