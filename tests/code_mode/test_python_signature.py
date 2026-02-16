"""Tests for Python signature generation and deduplication."""

from __future__ import annotations

from typing import Optional, Union

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai._python_signature import (
    FunctionParam,
    FunctionSignature,
    GenericTypeExpr,
    TypeFieldSignature,
    TypeSignature,
    UnionTypeExpr,
    _to_pascal_case,  # pyright: ignore[reportPrivateUsage]
    collect_unique_referenced_types,
    dedup_referenced_types,
    function_to_signature,
    render_type_expr,
    schema_to_signature,
)
from pydantic_ai._run_context import RunContext
from pydantic_ai.tools import FunctionToolDefinition

pytestmark = pytest.mark.anyio


def test_dedup_referenced_types_substring_names():
    """Renaming 'User' must not corrupt 'UserMeta' in the same signature."""
    user1 = TypeSignature(
        name='User',
        fields={
            'name': TypeFieldSignature(name='name', type='str', required=True, description=None),
        },
    )
    user2 = TypeSignature(
        name='User',
        fields={
            'id': TypeFieldSignature(name='id', type='int', required=True, description=None),
        },
    )
    user_meta = TypeSignature(
        name='UserMeta',
        fields={
            'role': TypeFieldSignature(name='role', type='str', required=True, description=None),
        },
    )

    sig1 = FunctionSignature(
        name='tool_a',
        params={'user': FunctionParam(name='user', type=user1, default=None)},
        return_type='Any',
        referenced_types=[user1],
    )
    sig2 = FunctionSignature(
        name='tool_b',
        params={
            'user': FunctionParam(name='user', type=user2, default=None),
            'meta': FunctionParam(name='meta', type=user_meta, default=None),
        },
        return_type=user_meta,
        referenced_types=[user2, user_meta],
    )
    dedup_referenced_types([sig1, sig2])

    # User in sig2 was renamed to tool_b_User
    assert user2.name == 'tool_b_User'
    # UserMeta must be untouched
    assert user_meta.name == 'UserMeta'
    # Params render correctly
    assert str(sig2.params['user']) == 'user: tool_b_User'
    assert str(sig2.params['meta']) == 'meta: UserMeta'
    assert render_type_expr(sig2.return_type) == 'UserMeta'


def test_dedup_identical_types_unified():
    """Identical TypeSignatures are unified to the same object instance."""
    user1 = TypeSignature(
        name='User',
        fields={
            'name': TypeFieldSignature(name='name', type='str', required=True, description=None),
        },
    )
    user2 = TypeSignature(
        name='User',
        fields={
            'name': TypeFieldSignature(name='name', type='str', required=True, description=None),
        },
    )

    sig1 = FunctionSignature(
        name='tool_a',
        params={'user': FunctionParam(name='user', type=user1, default=None)},
        return_type='Any',
        referenced_types=[user1],
    )
    sig2 = FunctionSignature(
        name='tool_b',
        params={'user': FunctionParam(name='user', type=user2, default=None)},
        return_type='Any',
        referenced_types=[user2],
    )
    dedup_referenced_types([sig1, sig2])

    # Both sigs keep the type, but unified to the same canonical instance
    assert len(sig2.referenced_types) == 1
    assert sig2.referenced_types[0] is user1
    # sig2's param should now point to the canonical (sig1's) TypeSignature
    assert sig2.params['user'].type is user1

    # collect_unique_referenced_types emits the definition only once
    defs = collect_unique_referenced_types([sig1, sig2])
    assert len(defs) == 1


def test_dedup_mixed_identical_and_conflicting_from_schemas():
    """Two identical User $defs are unified; a third different User is renamed.

    Tests the full pipeline: JSON schema → schema_to_signature → dedup → render.
    """
    # tool_a and tool_b both have a User $def with {name: str}
    user_v1_def = {
        'type': 'object',
        'properties': {'name': {'type': 'string'}},
        'required': ['name'],
    }
    # tool_c has a User $def with {id: int} — same name, different structure
    user_v2_def = {
        'type': 'object',
        'properties': {'id': {'type': 'integer'}},
        'required': ['id'],
    }

    sig1 = schema_to_signature(
        'tool_a',
        {
            'type': 'object',
            'properties': {'user': {'$ref': '#/$defs/User'}},
            'required': ['user'],
            '$defs': {'User': user_v1_def},
        },
    )
    sig2 = schema_to_signature(
        'tool_b',
        {
            'type': 'object',
            'properties': {'user': {'$ref': '#/$defs/User'}},
            'required': ['user'],
            '$defs': {'User': user_v1_def},
        },
    )
    sig3 = schema_to_signature(
        'tool_c',
        {
            'type': 'object',
            'properties': {'user': {'$ref': '#/$defs/User'}},
            'required': ['user'],
            '$defs': {'User': user_v2_def},
        },
    )

    dedup_referenced_types([sig1, sig2, sig3])

    # sig1 and sig2 share the same canonical User instance
    assert len(sig1.referenced_types) == 1
    assert len(sig2.referenced_types) == 1
    assert sig2.referenced_types[0] is sig1.referenced_types[0]
    assert sig2.params['user'].type is sig1.referenced_types[0]

    # sig3's User was renamed to tool_c_User
    assert sig3.referenced_types[0].name == 'tool_c_User'
    assert str(sig3.params['user']) == 'user: tool_c_User'

    # collect_unique_referenced_types returns exactly 2 unique types
    unique_types = collect_unique_referenced_types([sig1, sig2, sig3])
    assert [str(t) for t in unique_types] == snapshot(
        [
            """\
class User(TypedDict):
    name: str""",
            """\
class tool_c_User(TypedDict):
    id: int""",
        ]
    )


def test_dedup_composite_type_expr_rename_propagates():
    """Renaming propagates through GenericTypeExpr references like list[User]."""
    user1 = TypeSignature(
        name='User',
        fields={
            'name': TypeFieldSignature(name='name', type='str', required=True, description=None),
        },
    )
    user2 = TypeSignature(
        name='User',
        fields={
            'id': TypeFieldSignature(name='id', type='int', required=True, description=None),
        },
    )

    sig1 = FunctionSignature(
        name='tool_a',
        params={'user': FunctionParam(name='user', type=user1, default=None)},
        return_type='Any',
        referenced_types=[user1],
    )
    # sig2 references user2 via list[User]
    sig2 = FunctionSignature(
        name='tool_b',
        params={'users': FunctionParam(name='users', type=GenericTypeExpr(base='list', args=[user2]), default=None)},
        return_type='Any',
        referenced_types=[user2],
    )
    dedup_referenced_types([sig1, sig2])

    # user2 was renamed in place
    assert user2.name == 'tool_b_User'
    # The list[User] now renders as list[tool_b_User]
    assert str(sig2.params['users']) == 'users: list[tool_b_User]'


def test_render_type_signature():
    """TypeSignature renders a valid TypedDict class definition."""
    ts = TypeSignature(
        name='User',
        fields={
            'name': TypeFieldSignature(name='name', type='str', required=True, description=None),
            'age': TypeFieldSignature(name='age', type='int', required=False, description='The age'),
        },
    )
    assert str(ts) == snapshot("""\
class User(TypedDict):
    name: str
    age: NotRequired[int]
    \"\"\"The age\"\"\"\
""")


def test_render_type_signature_empty():
    """Empty TypeSignature renders with pass."""
    ts = TypeSignature(name='Empty')
    assert str(ts) == 'class Empty(TypedDict):\n    pass'


def test_render_generic_type_expr():
    """GenericTypeExpr renders correctly."""
    user = TypeSignature(name='User')
    expr = GenericTypeExpr(base='list', args=[user])
    assert str(expr) == 'list[User]'

    dict_expr = GenericTypeExpr(base='dict', args=['str', GenericTypeExpr(base='list', args=[user])])
    assert str(dict_expr) == 'dict[str, list[User]]'


def test_render_union_type_expr():
    """UnionTypeExpr renders correctly."""
    user = TypeSignature(name='User')
    expr = UnionTypeExpr(members=[user, 'None'])
    assert str(expr) == 'User | None'


def test_render_function_param():
    """FunctionParam renders correct parameter strings."""
    p1 = FunctionParam(name='x', type='int', default=None)
    assert str(p1) == 'x: int'

    p2 = FunctionParam(name='y', type='str', default="'hello'")
    assert str(p2) == "y: str = 'hello'"

    user = TypeSignature(name='User')
    p3 = FunctionParam(name='user', type=user, default=None)
    assert str(p3) == 'user: User'


def test_structurally_equal():
    """TypeSignature.structurally_equal compares fields structurally."""
    ts1 = TypeSignature(
        name='A',
        fields={
            'x': TypeFieldSignature(name='x', type='int', required=True, description='desc1'),
        },
    )
    ts2 = TypeSignature(
        name='B',
        fields={
            'x': TypeFieldSignature(name='x', type='int', required=True, description='different desc'),
        },
    )
    # Same fields, different names and descriptions — structurally equal
    assert ts1.structurally_equal(ts2)

    ts3 = TypeSignature(
        name='C',
        fields={
            'x': TypeFieldSignature(name='x', type='str', required=True, description=None),
        },
    )
    # Different field type — not equal
    assert not ts1.structurally_equal(ts3)


def test_to_pascal_case_digit_prefix():
    """PascalCase of a name starting with digits gets a leading underscore."""
    assert _to_pascal_case('123_tool') == '_123Tool'


def test_to_pascal_case_edge_cases():
    """Edge cases: empty string, all-digits, hyphenated."""
    assert _to_pascal_case('') == ''
    assert _to_pascal_case('42') == '_42'
    assert _to_pascal_case('my-tool-name') == 'MyToolName'


def test_function_tool_definition_produces_same_signature_as_function_based():
    """FunctionToolDefinition.python_signature matches the old FunctionToolsetTool approach."""
    from pydantic_ai._python_signature import function_to_signature
    from pydantic_ai.tools import Tool

    def my_tool(x: int, y: str = 'hello') -> bool:
        """A test tool."""
        return True

    tool = Tool(my_tool)
    tool_def = tool.tool_def
    assert isinstance(tool_def, FunctionToolDefinition)

    # The cached signature should match a freshly generated one
    cached_sig = tool_def.python_signature
    fresh_sig = function_to_signature(my_tool, name='my_tool', description=tool.description)

    assert str(cached_sig) == str(fresh_sig)


def test_tool_definition_cached_property_reset_on_replace():
    """dataclasses.replace() on a ToolDefinition resets the cached python_signature."""
    from dataclasses import replace

    from pydantic_ai.tools import ToolDefinition

    td = ToolDefinition(
        name='test_tool',
        parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']},
        description='Original description',
    )
    sig1 = td.python_signature
    assert sig1.docstring == 'Original description'

    td2 = replace(td, description='New description')
    sig2 = td2.python_signature
    assert sig2.docstring == 'New description'
    # Different instances
    assert sig1 is not sig2


def test_function_tool_definition_fallback_without_original_func():
    """FunctionToolDefinition falls back to schema-based signature when original_func is None."""

    td = FunctionToolDefinition(
        name='schema_tool',
        parameters_json_schema={'type': 'object', 'properties': {'q': {'type': 'string'}}, 'required': ['q']},
        description='A schema-based tool',
        original_func=None,
    )
    sig = td.python_signature
    assert sig.name == 'schema_tool'
    assert 'q' in sig.params
    assert sig.params['q'].type == 'str'


# =============================================================================
# Function signature edge cases
# =============================================================================


def test_function_signature_special_params():
    """RunContext skipped, unannotated → Any."""

    def with_ctx(ctx: RunContext[None], x: int) -> int:
        return x

    assert str(function_to_signature(with_ctx, name='with_ctx')) == snapshot("""\
async def with_ctx(*, x: int) -> int:
    ...\
""")

    def no_annot(x):  # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
        return x  # pyright: ignore[reportUnknownVariableType]

    assert str(
        function_to_signature(no_annot, name='no_annot')  # pyright: ignore[reportUnknownArgumentType]
    ) == snapshot("""\
async def no_annot(*, x: Any) -> Any:
    ...\
""")


class _UserInfo(BaseModel):
    name: str


def test_function_signature_union_and_model_types():
    """Unions, optionals, and model types render correct signatures."""

    def complex_func(
        a: Union[int, str],  # noqa: UP007 — testing Union[] code path
        b: int | str,
        c: Optional[int] = None,  # noqa: UP045 — testing Optional[] code path
        d: _UserInfo | None = None,
    ) -> _UserInfo: ...

    sig = function_to_signature(complex_func, name='complex_func')
    assert str(sig) == snapshot("""\
async def complex_func(*, a: int | str, b: int | str, c: int | None = None, d: _UserInfo | None = None) -> _UserInfo:
    ...\
""")
    assert [str(t) for t in sig.referenced_types] == snapshot(
        [
            """\
class _UserInfo(TypedDict):
    name: str\
"""
        ]
    )


def test_type_signature_docstring_and_structural_equality():
    """Docstring rendering and structural equality with different required."""
    ts = TypeSignature(name='Documented', docstring='A documented empty type')
    assert str(ts) == snapshot('''\
class Documented(TypedDict):
    """A documented empty type"""\
''')

    ts_a = TypeSignature(
        name='A',
        fields={'x': TypeFieldSignature(name='x', type='int', required=True, description=None)},
    )
    ts_b = TypeSignature(
        name='B',
        fields={'x': TypeFieldSignature(name='x', type='int', required=False, description=None)},
    )
    assert not ts_a.structurally_equal(ts_b)


# =============================================================================
# Schema signature edge cases
# =============================================================================


def test_schema_signature_const_enum():
    """const and enum paths in schema_to_type_expr produce Literal types."""
    # const value
    sig_const = schema_to_signature(
        'tool_const',
        {
            'type': 'object',
            'properties': {'mode': {'const': 'fast'}},
            'required': ['mode'],
        },
    )
    assert str(sig_const) == snapshot("""\
async def tool_const(*, mode: Literal['fast']) -> Any:
    ...\
""")

    # enum values
    sig_enum = schema_to_signature(
        'tool_enum',
        {
            'type': 'object',
            'properties': {'color': {'enum': ['red', 'green', 'blue']}},
            'required': ['color'],
        },
    )
    assert str(sig_enum) == snapshot("""\
async def tool_enum(*, color: Literal['red', 'green', 'blue']) -> Any:
    ...\
""")


def test_collect_unique_referenced_types_empty():
    """Empty input returns empty list."""
    assert collect_unique_referenced_types([]) == []

    sig = FunctionSignature(name='no_refs', params={}, return_type='Any', referenced_types=[])
    assert collect_unique_referenced_types([sig]) == []


def test_schema_signature_union_ref_allof():
    """oneOf, allOf, $ref variants produce correct signatures."""
    sig_oneof = schema_to_signature(
        'my_tool',
        {
            'type': 'object',
            'properties': {'value': {'oneOf': [{'type': 'string'}, {'type': 'integer'}]}},
            'required': ['value'],
        },
    )
    assert str(sig_oneof) == snapshot("""\
async def my_tool(*, value: str | int) -> Any:
    ...\
""")

    sig_allof_single = schema_to_signature(
        'tool2',
        {
            'type': 'object',
            'properties': {'x': {'allOf': [{'type': 'string'}]}},
            'required': ['x'],
        },
    )
    assert str(sig_allof_single) == snapshot("""\
async def tool2(*, x: str) -> Any:
    ...\
""")

    sig_allof_multi = schema_to_signature(
        'tool3',
        {
            'type': 'object',
            'properties': {'x': {'allOf': [{'type': 'string'}, {'type': 'integer'}]}},
            'required': ['x'],
        },
    )
    assert str(sig_allof_multi) == snapshot("""\
async def tool3(*, x: Any) -> Any:
    ...\
""")

    sig_ref = schema_to_signature(
        'tool4',
        {
            'type': 'object',
            'properties': {'user': {'$ref': '#/$defs/User'}},
            'required': ['user'],
            '$defs': {'User': {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}},
        },
    )
    assert str(sig_ref) == snapshot("""\
async def tool4(*, user: User) -> Any:
    ...\
""")
    assert [str(t) for t in sig_ref.referenced_types] == snapshot(
        [
            """\
class User(TypedDict):
    name: str\
"""
        ]
    )

    sig_ref_nonobj = schema_to_signature(
        'tool5',
        {
            'type': 'object',
            'properties': {'x': {'$ref': '#/$defs/StringAlias'}},
            'required': ['x'],
            '$defs': {'StringAlias': {'type': 'string'}},
        },
    )
    assert str(sig_ref_nonobj) == snapshot("""\
async def tool5(*, x: StringAlias) -> Any:
    ...\
""")


def test_schema_signature_array_object_typelist():
    """Arrays, objects, additionalProperties, and type lists."""
    # Tuple array
    assert str(
        schema_to_signature(
            't1',
            {
                'type': 'object',
                'properties': {'coords': {'type': 'array', 'items': [{'type': 'number'}, {'type': 'number'}]}},
                'required': ['coords'],
            },
        )
    ) == snapshot("""\
async def t1(*, coords: tuple[float, float]) -> Any:
    ...\
""")

    # Empty array
    assert str(
        schema_to_signature(
            't2',
            {
                'type': 'object',
                'properties': {'data': {'type': 'array'}},
                'required': ['data'],
            },
        )
    ) == snapshot("""\
async def t2(*, data: list[Any]) -> Any:
    ...\
""")

    # additionalProperties: true
    assert str(
        schema_to_signature(
            't3',
            {
                'type': 'object',
                'properties': {'meta': {'type': 'object', 'additionalProperties': True}},
                'required': ['meta'],
            },
        )
    ) == snapshot("""\
async def t3(*, meta: dict[str, Any]) -> Any:
    ...\
""")

    # Typed additionalProperties
    assert str(
        schema_to_signature(
            't4',
            {
                'type': 'object',
                'properties': {'tags': {'type': 'object', 'additionalProperties': {'type': 'string'}}},
                'required': ['tags'],
            },
        )
    ) == snapshot("""\
async def t4(*, tags: dict[str, str]) -> Any:
    ...\
""")

    # Type list ['string', 'null']
    assert str(
        schema_to_signature(
            't5',
            {
                'type': 'object',
                'properties': {'name': {'type': ['string', 'null']}},
                'required': ['name'],
            },
        )
    ) == snapshot("""\
async def t5(*, name: str | None) -> Any:
    ...\
""")

    # Type list multi
    assert str(
        schema_to_signature(
            't6',
            {
                'type': 'object',
                'properties': {'value': {'type': ['string', 'integer', 'boolean']}},
                'required': ['value'],
            },
        )
    ) == snapshot("""\
async def t6(*, value: str | int | bool) -> Any:
    ...\
""")

    # Object type list with null
    sig = schema_to_signature(
        't7',
        {
            'type': 'object',
            'properties': {
                'config': {
                    'type': ['object', 'null'],
                    'properties': {'enabled': {'type': 'boolean'}},
                    'required': ['enabled'],
                },
            },
            'required': ['config'],
        },
    )
    assert str(sig) == snapshot("""\
async def t7(*, config: T7Config | None) -> Any:
    ...\
""")
    assert [str(t) for t in sig.referenced_types] == snapshot(
        [
            """\
class T7Config(TypedDict):
    enabled: bool\
"""
        ]
    )


def test_schema_signature_optional_params_and_return():
    """Optional params, return schema edge cases, anyOf dedup."""
    # Optional already nullable
    assert str(
        schema_to_signature(
            't1',
            {
                'type': 'object',
                'properties': {'x': {'type': ['string', 'null']}},
            },
        )
    ) == snapshot("""\
async def t1(*, x: str | None = None) -> Any:
    ...\
""")

    # Optional not nullable → adds | None
    assert str(
        schema_to_signature(
            't2',
            {
                'type': 'object',
                'properties': {'x': {'type': 'string'}},
            },
        )
    ) == snapshot("""\
async def t2(*, x: str | None = None) -> Any:
    ...\
""")

    # Unresolvable return_schema → JSON blob in description
    sig3 = schema_to_signature(
        't3',
        {'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x']},
        description='A tool',
        return_schema={},
    )
    assert str(sig3) == snapshot('''\
async def t3(*, x: str) -> Any:
    """
    A tool

    Return schema:
    {}
    """
    ...\
''')

    # Return schema with $defs
    sig4 = schema_to_signature(
        't4',
        {'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x']},
        return_schema={
            '$ref': '#/$defs/Result',
            '$defs': {'Result': {'type': 'object', 'properties': {'v': {'type': 'integer'}}, 'required': ['v']}},
        },
    )
    assert str(sig4) == snapshot("""\
async def t4(*, x: str) -> Result:
    ...\
""")
    assert [str(t) for t in sig4.referenced_types] == snapshot(
        [
            """\
class Result(TypedDict):
    v: int\
"""
        ]
    )

    # anyOf with duplicates → deduplicated
    assert str(
        schema_to_signature(
            't5',
            {
                'type': 'object',
                'properties': {'x': {'anyOf': [{'type': 'string'}, {'type': 'string'}, {'type': 'null'}]}},
                'required': ['x'],
            },
        )
    ) == snapshot("""\
async def t5(*, x: str | None) -> Any:
    ...\
""")
