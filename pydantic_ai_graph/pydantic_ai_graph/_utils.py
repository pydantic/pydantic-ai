import sys
import types
from typing import Any, TypeAliasType, Union, get_args, get_origin


def get_union_args(tp: Any) -> tuple[Any, ...]:
    """Extract the arguments of a Union type if `response_type` is a union, otherwise return the original type."""
    # similar to `pydantic_ai_slim/pydantic_ai/_result.py:get_union_args`
    if isinstance(tp, TypeAliasType):
        tp = tp.__value__

    origin = get_origin(tp)
    if origin_is_union(origin):
        return get_args(tp)
    else:
        return (tp,)


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


_NoneType = type(None)


def type_arg_name(arg: Any) -> str:
    if arg is _NoneType:
        return 'None'
    else:
        return arg.__name__
