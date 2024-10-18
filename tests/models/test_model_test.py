"""
This module contains tests for the testing module.
"""

from __future__ import annotations as _annotations

from typing import Annotated, Any, Literal

from annotated_types import Gt, Lt, MaxLen, MinLen
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai.models.test import _JsonSchemaTestData  # pyright: ignore[reportPrivateUsage]


def test_simple():
    class NestedModel(BaseModel):
        foo: str
        bar: int

    class TestModel(BaseModel):
        my_str: str
        my_str_long: Annotated[str, MinLen(10)]
        my_str_short: Annotated[str, MaxLen(1)]
        my_int: int
        my_int_gt: Annotated[int, Gt(5)]
        my_int_lt: Annotated[int, Lt(-5)]
        my_float: float
        my_float_gt: Annotated[float, Gt(5.0)]
        my_float_lt: Annotated[float, Lt(-5.0)]
        my_bool: bool
        my_bytes: bytes
        my_fixed_tuple: tuple[int, str]
        my_var_tuple: tuple[int, ...]
        my_list: list[str]
        my_dict: dict[str, int]
        my_set: set[str]
        my_set_min_len: Annotated[set[str], MinLen(5)]
        my_lit_int: Literal[1]
        my_lit_ints: Literal[1, 2, 3]
        my_lit_str: Literal['a']
        my_lit_strs: Literal['a', 'b', 'c']
        my_any: Any
        nested: NestedModel
        not_required: str = 'default'

    json_schema = TestModel.model_json_schema()
    data = _JsonSchemaTestData(json_schema).generate()  # pyright: ignore[reportArgumentType]
    assert data == snapshot(
        {
            'my_str': 'a',
            'my_str_long': 'aaaaaaaaaa',
            'my_str_short': 'a',
            'my_int': 0,
            'my_int_gt': 6,
            'my_int_lt': -6,
            'my_float': 0.0,
            'my_float_gt': 6.0,
            'my_float_lt': -6.0,
            'my_bool': False,
            'my_bytes': 'a',
            'my_fixed_tuple': [0, 'a'],
            'my_var_tuple': [0],
            'my_list': ['a'],
            'my_dict': {'additionalProperty': 0},
            'my_set': ['b'],
            'my_set_min_len': ['c', 'd', 'e', 'f', 'g'],
            'my_lit_int': 1,
            'my_lit_ints': 1,
            'my_lit_str': 'a',
            'my_lit_strs': 'a',
            'my_any': 'g',
            'nested': {'foo': 'g', 'bar': 6},
        }
    )
    TestModel.model_validate(data)
