import asyncio
from dataclasses import is_dataclass
from functools import partial
from typing import Any, Callable, ParamSpec, TypeVar, get_args, is_typeddict

from pydantic import BaseModel

_P = ParamSpec('_P')
_R = TypeVar('_R')


async def run_in_executor(func: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R:
    if kwargs:
        return await asyncio.get_running_loop().run_in_executor(None, partial(func, *args, **kwargs))
    else:
        return await asyncio.get_running_loop().run_in_executor(None, func, *args)  # type: ignore


_UnionType = type(int | str)


def allow_plain_str(response_type: Any) -> bool:
    """Check if the response type allows plain strings."""
    return isinstance(response_type, _UnionType) and any(t is str for t in get_args(response_type))


def is_model_like(response_type: Any) -> bool:
    """Check if the response type is model-like."""
    return isinstance(response_type, type) and (
        issubclass(response_type, BaseModel) or is_dataclass(response_type) or is_typeddict(response_type)
    )
