from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass, is_dataclass
from functools import partial
from types import GenericAlias
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    TypedDict,
    TypeVar,
    Union,
    cast,
    get_args,
    overload,
)

from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue
from typing_extensions import ParamSpec, TypeAlias, is_typeddict

_P = ParamSpec('_P')
_R = TypeVar('_R')


async def run_in_executor(func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
    if kwargs:
        return await asyncio.get_running_loop().run_in_executor(None, partial(func, *args, **kwargs))
    else:
        return await asyncio.get_running_loop().run_in_executor(None, func, *args)  # type: ignore


_UnionType = type(Union[int, str])


def allow_plain_str(response_type: Any) -> bool:
    """Check if the response type allows plain strings."""
    return isinstance(response_type, _UnionType) and any(t is str for t in get_args(response_type))


def is_model_like(type_: Any) -> bool:
    """Check if something is a pydantic model, dataclass or typedict.

    These should all generate a JSON Schema with `{"type": "object"}` and therefore be usable directly as
    function parameters.
    """
    return (
        isinstance(type_, type)
        and not isinstance(type_, GenericAlias)
        and (issubclass(type_, BaseModel) or is_dataclass(type_) or is_typeddict(type_))
    )


ObjectJsonSchema = TypedDict(
    'ObjectJsonSchema',
    {
        'type': Literal['object'],
        'title': str,
        'properties': dict[str, JsonSchemaValue],
        'required': list[str],
        '$defs': dict[str, Any],
    },
)


def check_object_json_schema(schema: JsonSchemaValue) -> ObjectJsonSchema:
    if schema.get('type') == 'object':
        return cast(ObjectJsonSchema, schema)
    else:
        raise ValueError('Schema must be an object')


_T = TypeVar('_T')


@dataclass
class Some(Generic[_T]):
    """Analogous to Rust's `Option::Some` type."""

    value: _T


# Analogous to Rust's `Option` type, usage: `Option[Thing]` is equivalent to `Some[Thing] | None`
Option: TypeAlias = Union[Some[_T], None]


_Left = TypeVar('_Left')
_Right = TypeVar('_Right')


class Either(Generic[_Left, _Right]):
    """Two member Union that records which member was set, this is analogous to Rust enums with two variants.

    Usage:

    ```py
    if left_thing := either.left:
        use_left(left_thing)
    else:
        use_right(either.right)
    ```
    """

    @overload
    def __init__(self, *, left: _Left) -> None: ...

    @overload
    def __init__(self, *, right: _Right) -> None: ...

    def __init__(self, **kwargs: Any) -> None:
        keys = set(kwargs.keys())
        if keys == {'left'}:
            self._left: Option[_Left] = Some(kwargs['left'])
            self._right: _Right | None = None
        elif keys == {'right'}:
            self._left = None
            self._right = kwargs['right']
        else:
            raise TypeError('Either must receive exactly one value - `left` or `right`')

    @property
    def left(self) -> Option[_Left]:
        return self._left

    @property
    def right(self) -> _Right:
        if self._right is None:
            raise TypeError('Right not set')
        return self._right

    def is_left(self) -> bool:
        return self._left is not None

    def whichever(self) -> _Left | _Right:
        return self._left.value if self._left is not None else self.right
