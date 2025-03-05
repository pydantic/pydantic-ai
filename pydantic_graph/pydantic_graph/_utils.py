from __future__ import annotations as _annotations

import asyncio
import types
from datetime import datetime, timezone
from typing import Any, TypeVar

from typing_extensions import TypeIs, get_args, get_origin
from typing_inspection import typing_objects
from typing_inspection.introspection import is_union_origin


def get_event_loop():
    try:
        event_loop = asyncio.get_event_loop()
    except RuntimeError:
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
    return event_loop


def get_union_args(tp: Any) -> tuple[Any, ...]:
    """Extract the arguments of a Union type if `response_type` is a union, otherwise return an empty tuple."""
    # similar to `pydantic_ai_slim/pydantic_ai/_result.py:get_union_args`
    if typing_objects.is_typealiastype(tp):
        tp = tp.__value__

    origin = get_origin(tp)
    if is_union_origin(origin):
        return get_args(tp)
    else:
        return (tp,)


def unpack_annotated(tp: Any) -> tuple[Any, list[Any]]:
    """Strip `Annotated` from the type if present.

    Returns:
        `(tp argument, ())` if not annotated, otherwise `(stripped type, annotations)`.
    """
    origin = get_origin(tp)
    if typing_objects.is_annotated(origin):
        inner_tp, *args = get_args(tp)
        return inner_tp, args
    else:
        return tp, []


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
            if back.f_globals.get('__name__') == 'typing':
                # If the class calling this function is generic, explicitly parameterizing the class
                # results in a `typing._GenericAlias` instance, which proxies instantiation calls to the
                # "real" class and thus adding an extra frame to the call. To avoid pulling anything
                # from the `typing` module, use the correct frame (the one before):
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


def is_set(t_or_unset: T | Unset) -> TypeIs[T]:
    return t_or_unset is not UNSET
