"""Tests for function signature generation and deduplication."""

from __future__ import annotations

import typing
from dataclasses import replace
from enum import Enum
from typing import Optional, Union

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, RootModel

from pydantic_ai._run_context import RunContext
from pydantic_ai.function_signature import (
    FunctionParam,
    FunctionSignature,
    GenericTypeExpr,
    SimpleTypeExpr,
    TypeFieldSignature,
    TypeSignature,
    UnionTypeExpr,
)
from pydantic_ai.tools import Tool, ToolDefinition

pytestmark = pytest.mark.anyio


# Module-level types for tests that need get_type_hints() resolution
class _Color(str, Enum):
    RED = 'red'
    GREEN = 'green'


class _ConfigWithEnum(BaseModel):
    name: str
    color: _Color


def test_render_function_signature_with_no_params():
    sig = FunctionSignature(name='ping', params={}, return_type=SimpleTypeExpr('None'))
    assert str(sig) == 'def ping() -> None:\n    ...'
    # render() can override is_async
    assert sig.render('...', is_async=True) == 'async def ping() -> None:\n    ...'


def test_dedup_referenced_types_substring_names():
    """Renaming 'User' must not corrupt 'UserMeta' in the same signature."""
    user1 = TypeSignature(
        name='User',
        fields={
            'name': TypeFieldSignature(name='name', type=SimpleTypeExpr('str'), required=True, description=None),
        },
    )
    user2 = TypeSignature(
        name='User',
        fields={
            'id': TypeFieldSignature(name='id', type=SimpleTypeExpr('int'), required=True, description=None),
        },
    )
    user_meta = TypeSignature(
        name='UserMeta',
        fields={
            'role': TypeFieldSignature(name='role', type=SimpleTypeExpr('str'), required=True, description=None),
        },
    )

    sig1 = FunctionSignature(
        name='tool_a',
        params={'user': FunctionParam(name='user', type=user1, default=None)},
        return_type=SimpleTypeExpr('Any'),
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
    FunctionSignature.dedup_referenced_types([sig1, sig2])

    # User in sig2 was renamed to tool_b_User
    assert user2.name == 'tool_b_User'
    # UserMeta must be untouched
    assert user_meta.name == 'UserMeta'
    # Params render correctly
    assert str(sig2.params['user']) == 'user: tool_b_User'
    assert str(sig2.params['meta']) == 'meta: UserMeta'
    assert str(sig2.return_type) == 'UserMeta'


def test_dedup_identical_types_unified():
    """Identical TypeSignatures are unified to the same object instance."""
    user1 = TypeSignature(
        name='User',
        fields={
            'name': TypeFieldSignature(name='name', type=SimpleTypeExpr('str'), required=True, description=None),
        },
    )
    user2 = TypeSignature(
        name='User',
        fields={
            'name': TypeFieldSignature(name='name', type=SimpleTypeExpr('str'), required=True, description=None),
        },
    )

    sig1 = FunctionSignature(
        name='tool_a',
        params={'user': FunctionParam(name='user', type=user1, default=None)},
        return_type=SimpleTypeExpr('Any'),
        referenced_types=[user1],
    )
    sig2 = FunctionSignature(
        name='tool_b',
        params={'user': FunctionParam(name='user', type=user2, default=None)},
        return_type=SimpleTypeExpr('Any'),
        referenced_types=[user2],
    )
    FunctionSignature.dedup_referenced_types([sig1, sig2])

    # Both sigs keep the type, but unified to the same canonical instance
    assert len(sig2.referenced_types) == 1
    assert sig2.referenced_types[0] is user1
    # sig2's param should now point to the canonical (sig1's) TypeSignature
    assert sig2.params['user'].type is user1

    # collect_unique_referenced_types emits the definition only once
    defs = FunctionSignature.collect_unique_referenced_types([sig1, sig2])
    assert len(defs) == 1


def test_dedup_replaces_nested_generic_and_union_refs_with_canonical():
    user1 = TypeSignature(
        name='User',
        fields={
            'name': TypeFieldSignature(name='name', type=SimpleTypeExpr('str'), required=True, description=None),
        },
    )
    user2 = TypeSignature(
        name='User',
        fields={
            'name': TypeFieldSignature(name='name', type=SimpleTypeExpr('str'), required=True, description=None),
        },
    )
    wrapper = TypeSignature(
        name='Wrapper',
        fields={
            'users': TypeFieldSignature(
                name='users',
                type=GenericTypeExpr(base='list', args=[user2]),
                required=True,
                description=None,
            ),
            'maybe_user': TypeFieldSignature(
                name='maybe_user',
                type=UnionTypeExpr(members=[user2, SimpleTypeExpr('None')]),
                required=False,
                description=None,
            ),
        },
    )

    sig1 = FunctionSignature(
        name='tool_a',
        params={'user': FunctionParam(name='user', type=user1, default=None)},
        return_type=SimpleTypeExpr('Any'),
        referenced_types=[user1],
    )
    sig2 = FunctionSignature(
        name='tool_b',
        params={
            'wrapper': FunctionParam(name='wrapper', type=wrapper, default=None),
            'tags': FunctionParam(
                name='tags', type=GenericTypeExpr(base='list', args=[SimpleTypeExpr('str')]), default=None
            ),
            'label': FunctionParam(
                name='label', type=UnionTypeExpr(members=[SimpleTypeExpr('str'), SimpleTypeExpr('None')]), default=None
            ),
        },
        return_type=GenericTypeExpr(base='list', args=[user2]),
        referenced_types=[user2, wrapper],
    )

    FunctionSignature.dedup_referenced_types([sig1, sig2])

    users_expr = wrapper.fields['users'].type
    maybe_user_expr = wrapper.fields['maybe_user'].type
    return_expr = sig2.return_type

    assert isinstance(users_expr, GenericTypeExpr)
    assert isinstance(maybe_user_expr, UnionTypeExpr)
    assert isinstance(return_expr, GenericTypeExpr)
    assert users_expr.args[0] is user1
    assert maybe_user_expr.members[0] is user1
    assert return_expr.args[0] is user1


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

    sig1 = FunctionSignature.from_schema(
        name='tool_a',
        parameters_schema={
            'type': 'object',
            'properties': {'user': {'$ref': '#/$defs/User'}},
            'required': ['user'],
            '$defs': {'User': user_v1_def},
        },
    )
    sig2 = FunctionSignature.from_schema(
        name='tool_b',
        parameters_schema={
            'type': 'object',
            'properties': {'user': {'$ref': '#/$defs/User'}},
            'required': ['user'],
            '$defs': {'User': user_v1_def},
        },
    )
    sig3 = FunctionSignature.from_schema(
        name='tool_c',
        parameters_schema={
            'type': 'object',
            'properties': {'user': {'$ref': '#/$defs/User'}},
            'required': ['user'],
            '$defs': {'User': user_v2_def},
        },
    )

    FunctionSignature.dedup_referenced_types([sig1, sig2, sig3])

    # sig1 and sig2 share the same canonical User instance
    assert len(sig1.referenced_types) == 1
    assert len(sig2.referenced_types) == 1
    assert sig2.referenced_types[0] is sig1.referenced_types[0]
    assert sig2.params['user'].type is sig1.referenced_types[0]

    # sig3's User was renamed to tool_c_User
    assert sig3.referenced_types[0].name == 'tool_c_User'
    assert str(sig3.params['user']) == 'user: tool_c_User'

    # collect_unique_referenced_types returns exactly 2 unique types
    unique_types = FunctionSignature.collect_unique_referenced_types([sig1, sig2, sig3])
    assert [t.render_definition() for t in unique_types] == snapshot(
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
            'name': TypeFieldSignature(name='name', type=SimpleTypeExpr('str'), required=True, description=None),
        },
    )
    user2 = TypeSignature(
        name='User',
        fields={
            'id': TypeFieldSignature(name='id', type=SimpleTypeExpr('int'), required=True, description=None),
        },
    )

    sig1 = FunctionSignature(
        name='tool_a',
        params={'user': FunctionParam(name='user', type=user1, default=None)},
        return_type=SimpleTypeExpr('Any'),
        referenced_types=[user1],
    )
    # sig2 references user2 via list[User]
    sig2 = FunctionSignature(
        name='tool_b',
        params={'users': FunctionParam(name='users', type=GenericTypeExpr(base='list', args=[user2]), default=None)},
        return_type=SimpleTypeExpr('Any'),
        referenced_types=[user2],
    )
    FunctionSignature.dedup_referenced_types([sig1, sig2])

    # user2 was renamed in place
    assert user2.name == 'tool_b_User'
    # The list[User] now renders as list[tool_b_User]
    assert str(sig2.params['users']) == 'users: list[tool_b_User]'


def test_render_type_signature():
    """TypeSignature renders a valid TypedDict class definition."""
    ts = TypeSignature(
        name='User',
        fields={
            'name': TypeFieldSignature(name='name', type=SimpleTypeExpr('str'), required=True, description=None),
            'age': TypeFieldSignature(name='age', type=SimpleTypeExpr('int'), required=False, description='The age'),
        },
    )
    assert str(ts) == snapshot('User')


def test_render_type_signature_empty():
    """Empty TypeSignature renders with pass."""
    ts = TypeSignature(name='Empty')
    assert str(ts) == 'Empty'
    assert ts.render_definition() == 'class Empty(TypedDict):\n    pass'


def test_render_generic_type_expr():
    """GenericTypeExpr renders correctly."""
    user = TypeSignature(name='User')
    expr = GenericTypeExpr(base='list', args=[user])
    assert str(expr) == 'list[User]'

    dict_expr = GenericTypeExpr(base='dict', args=[SimpleTypeExpr('str'), GenericTypeExpr(base='list', args=[user])])
    assert str(dict_expr) == 'dict[str, list[User]]'


def test_render_union_type_expr():
    """UnionTypeExpr renders correctly."""
    user = TypeSignature(name='User')
    expr = UnionTypeExpr(members=[user, SimpleTypeExpr('None')])
    assert str(expr) == 'User | None'


def test_render_function_param():
    """FunctionParam renders correct parameter strings."""
    p1 = FunctionParam(name='x', type=SimpleTypeExpr('int'), default=None)
    assert str(p1) == 'x: int'

    p2 = FunctionParam(name='y', type=SimpleTypeExpr('str'), default="'hello'")
    assert str(p2) == "y: str = 'hello'"

    user = TypeSignature(name='User')
    p3 = FunctionParam(name='user', type=user, default=None)
    assert str(p3) == 'user: User'


def test_structurally_equal():
    """TypeSignature.structurally_equal compares fields structurally."""
    ts1 = TypeSignature(
        name='A',
        fields={
            'x': TypeFieldSignature(name='x', type=SimpleTypeExpr('int'), required=True, description='desc1'),
        },
    )
    ts2 = TypeSignature(
        name='B',
        fields={
            'x': TypeFieldSignature(name='x', type=SimpleTypeExpr('int'), required=True, description='different desc'),
        },
    )
    # Same fields, different names and descriptions — structurally equal
    assert ts1.structurally_equal(ts2)

    ts3 = TypeSignature(
        name='C',
        fields={
            'x': TypeFieldSignature(name='x', type=SimpleTypeExpr('str'), required=True, description=None),
        },
    )
    # Different field type — not equal
    assert not ts1.structurally_equal(ts3)


def test_schema_nested_object_name_with_digit_prefix():
    """Tool names starting with digits get valid PascalCase TypedDict names with leading underscore."""
    sig = FunctionSignature.from_schema(
        name='123_tool',
        parameters_schema={
            'type': 'object',
            'properties': {
                'config': {
                    'type': 'object',
                    'properties': {'key': {'type': 'string'}},
                    'required': ['key'],
                },
            },
            'required': ['config'],
        },
    )
    # The nested object should get a name like _123Tool + Config
    assert any('_123Tool' in rt.name for rt in sig.referenced_types)


def test_schema_hyphenated_tool_name():
    """Hyphenated tool names (e.g. from MCP) produce valid PascalCase TypedDict names."""
    sig = FunctionSignature.from_schema(
        name='my-tool-name',
        parameters_schema={
            'type': 'object',
            'properties': {
                'data': {
                    'type': 'object',
                    'properties': {'value': {'type': 'integer'}},
                    'required': ['value'],
                },
            },
            'required': ['data'],
        },
    )
    assert any('MyToolName' in rt.name for rt in sig.referenced_types)


def test_tool_def_function_signature_matches_function_based():
    """Tool.tool_def.function_signature matches a directly generated function signature."""

    def my_tool(x: int, y: str = 'hello') -> bool:
        """A test tool."""
        return True  # pragma: no cover

    tool = Tool(my_tool)
    tool_def = tool.tool_def

    # The cached signature should match a freshly generated one
    cached_sig = tool_def.function_signature
    fresh_sig = FunctionSignature.from_function(my_tool, name='my_tool', description=tool.description)

    assert str(cached_sig) == str(fresh_sig)


def test_tool_definition_cached_property_reset_on_replace():
    """dataclasses.replace() on a ToolDefinition resets the cached function_signature."""

    td = ToolDefinition(
        name='test_tool',
        parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']},
        description='Original description',
    )
    sig1 = td.function_signature
    assert sig1.description == 'Original description'

    td2 = replace(td, description='New description')
    sig2 = td2.function_signature
    assert sig2.description == 'New description'
    # Different instances
    assert sig1 is not sig2


def test_tool_definition_schema_based_function_signature():
    """ToolDefinition.function_signature generates a schema-based signature."""

    td = ToolDefinition(
        name='schema_tool',
        parameters_json_schema={'type': 'object', 'properties': {'q': {'type': 'string'}}, 'required': ['q']},
        description='A schema-based tool',
    )
    sig = td.function_signature
    assert sig.name == 'schema_tool'
    assert 'q' in sig.params
    assert str(sig.params['q'].type) == 'str'


def test_tool_from_schema_function_signature_uses_schema():
    async def handler(**kwargs: typing.Any) -> typing.Any:  # pragma: no cover
        return kwargs

    tool = Tool.from_schema(
        handler,
        name='search',
        description='Search documents',
        json_schema={
            'type': 'object',
            'properties': {
                'query': {'type': 'string'},
                'limit': {'type': 'integer'},
            },
            'required': ['query'],
        },
    )
    assert str(tool.tool_def.function_signature) == snapshot('''\
def search(*, query: str, limit: int | None = None) -> Any:
    """Search documents"""
    ...\
''')


# =============================================================================
# Function signature edge cases
# =============================================================================


def test_function_signature_special_params():
    """RunContext skipped, unannotated → Any."""

    def with_ctx(ctx: RunContext[None], x: int) -> int:
        return x  # pragma: no cover

    assert str(FunctionSignature.from_function(with_ctx, name='with_ctx')) == snapshot("""\
def with_ctx(*, x: int) -> int:
    ...\
""")

    def no_annot(x):  # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
        return x  # pragma: no cover  # pyright: ignore[reportUnknownVariableType]

    assert str(
        FunctionSignature.from_function(no_annot, name='no_annot')  # pyright: ignore[reportUnknownArgumentType]
    ) == snapshot("""\
def no_annot(*, x: Any) -> Any:
    ...\
""")


class _UserInfo(BaseModel):
    name: str


class _IntList(RootModel[list[int]]):
    pass


class _TreeNode(BaseModel):
    value: int
    children: list[_TreeNode] = []


def test_function_signature_union_and_model_types():
    """Unions, optionals, and model types render correct signatures."""

    def complex_func(
        a: Union[int, str],  # noqa: UP007 — testing Union[] code path
        b: int | str,
        c: Optional[int] = None,  # noqa: UP045 — testing Optional[] code path
        d: _UserInfo | None = None,
    ) -> _UserInfo: ...  # pragma: no cover

    sig = FunctionSignature.from_function(complex_func, name='complex_func')
    assert str(sig) == snapshot("""\
def complex_func(*, a: int | str, b: int | str, c: int | None = None, d: _UserInfo | None = None) -> _UserInfo:
    ...\
""")
    assert [t.render_definition() for t in sig.referenced_types] == snapshot(
        [
            """\
class _UserInfo(TypedDict):
    name: str\
"""
        ]
    )


def test_type_signature_docstring_and_structural_equality():
    """Docstring rendering and structural equality with different required."""
    ts = TypeSignature(name='Documented', description='A documented empty type')
    assert str(ts) == snapshot('Documented')

    ts_a = TypeSignature(
        name='A',
        fields={'x': TypeFieldSignature(name='x', type=SimpleTypeExpr('int'), required=True, description=None)},
    )
    ts_b = TypeSignature(
        name='B',
        fields={'x': TypeFieldSignature(name='x', type=SimpleTypeExpr('int'), required=False, description=None)},
    )
    assert not ts_a.structurally_equal(ts_b)


# =============================================================================
# Schema signature edge cases
# =============================================================================


def test_schema_signature_const_enum():
    """const and enum paths in schema_to_type_expr produce Literal types."""
    # const value
    sig_const = FunctionSignature.from_schema(
        name='tool_const',
        parameters_schema={
            'type': 'object',
            'properties': {'mode': {'const': 'fast'}},
            'required': ['mode'],
        },
    )
    assert str(sig_const) == snapshot("""\
def tool_const(*, mode: Literal['fast']) -> Any:
    ...\
""")

    # enum values
    sig_enum = FunctionSignature.from_schema(
        name='tool_enum',
        parameters_schema={
            'type': 'object',
            'properties': {'color': {'enum': ['red', 'green', 'blue']}},
            'required': ['color'],
        },
    )
    assert str(sig_enum) == snapshot("""\
def tool_enum(*, color: Literal['red', 'green', 'blue']) -> Any:
    ...\
""")


def test_collect_unique_referenced_types_empty():
    """Empty input returns empty list."""
    assert FunctionSignature.collect_unique_referenced_types([]) == []

    sig = FunctionSignature(name='no_refs', params={}, return_type=SimpleTypeExpr('Any'), referenced_types=[])
    assert FunctionSignature.collect_unique_referenced_types([sig]) == []


def test_schema_signature_union_ref_allof():
    """oneOf, allOf, $ref variants produce correct signatures."""
    sig_oneof = FunctionSignature.from_schema(
        name='my_tool',
        parameters_schema={
            'type': 'object',
            'properties': {'value': {'oneOf': [{'type': 'string'}, {'type': 'integer'}]}},
            'required': ['value'],
        },
    )
    assert str(sig_oneof) == snapshot("""\
def my_tool(*, value: str | int) -> Any:
    ...\
""")

    sig_allof_single = FunctionSignature.from_schema(
        name='tool2',
        parameters_schema={
            'type': 'object',
            'properties': {'x': {'allOf': [{'type': 'string'}]}},
            'required': ['x'],
        },
    )
    assert str(sig_allof_single) == snapshot("""\
def tool2(*, x: str) -> Any:
    ...\
""")

    sig_allof_multi = FunctionSignature.from_schema(
        name='tool3',
        parameters_schema={
            'type': 'object',
            'properties': {'x': {'allOf': [{'type': 'string'}, {'type': 'integer'}]}},
            'required': ['x'],
        },
    )
    assert str(sig_allof_multi) == snapshot("""\
def tool3(*, x: Any) -> Any:
    ...\
""")

    sig_ref = FunctionSignature.from_schema(
        name='tool4',
        parameters_schema={
            'type': 'object',
            'properties': {'user': {'$ref': '#/$defs/User'}},
            'required': ['user'],
            '$defs': {'User': {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}},
        },
    )
    assert str(sig_ref) == snapshot("""\
def tool4(*, user: User) -> Any:
    ...\
""")
    assert [t.render_definition() for t in sig_ref.referenced_types] == snapshot(
        [
            """\
class User(TypedDict):
    name: str\
"""
        ]
    )

    sig_ref_nonobj = FunctionSignature.from_schema(
        name='tool5',
        parameters_schema={
            'type': 'object',
            'properties': {'x': {'$ref': '#/$defs/StringAlias'}},
            'required': ['x'],
            '$defs': {'StringAlias': {'type': 'string'}},
        },
    )
    assert str(sig_ref_nonobj) == snapshot("""\
def tool5(*, x: StringAlias) -> Any:
    ...\
""")


def test_schema_signature_array_object_typelist():
    """Arrays, objects, additionalProperties, and type lists."""
    # Tuple array
    assert str(
        FunctionSignature.from_schema(
            name='t1',
            parameters_schema={
                'type': 'object',
                'properties': {'coords': {'type': 'array', 'items': [{'type': 'number'}, {'type': 'number'}]}},
                'required': ['coords'],
            },
        )
    ) == snapshot("""\
def t1(*, coords: tuple[float, float]) -> Any:
    ...\
""")

    # Empty array
    assert str(
        FunctionSignature.from_schema(
            name='t2',
            parameters_schema={
                'type': 'object',
                'properties': {'data': {'type': 'array'}},
                'required': ['data'],
            },
        )
    ) == snapshot("""\
def t2(*, data: list[Any]) -> Any:
    ...\
""")

    # additionalProperties: true
    assert str(
        FunctionSignature.from_schema(
            name='t3',
            parameters_schema={
                'type': 'object',
                'properties': {'meta': {'type': 'object', 'additionalProperties': True}},
                'required': ['meta'],
            },
        )
    ) == snapshot("""\
def t3(*, meta: dict[str, Any]) -> Any:
    ...\
""")

    # Typed additionalProperties
    assert str(
        FunctionSignature.from_schema(
            name='t4',
            parameters_schema={
                'type': 'object',
                'properties': {'tags': {'type': 'object', 'additionalProperties': {'type': 'string'}}},
                'required': ['tags'],
            },
        )
    ) == snapshot("""\
def t4(*, tags: dict[str, str]) -> Any:
    ...\
""")

    # Type list ['string', 'null']
    assert str(
        FunctionSignature.from_schema(
            name='t5',
            parameters_schema={
                'type': 'object',
                'properties': {'name': {'type': ['string', 'null']}},
                'required': ['name'],
            },
        )
    ) == snapshot("""\
def t5(*, name: str | None) -> Any:
    ...\
""")

    # Type list multi
    assert str(
        FunctionSignature.from_schema(
            name='t6',
            parameters_schema={
                'type': 'object',
                'properties': {'value': {'type': ['string', 'integer', 'boolean']}},
                'required': ['value'],
            },
        )
    ) == snapshot("""\
def t6(*, value: str | int | bool) -> Any:
    ...\
""")

    # Object type list with null
    sig = FunctionSignature.from_schema(
        name='t7',
        parameters_schema={
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
def t7(*, config: T7Config | None) -> Any:
    ...\
""")
    assert [t.render_definition() for t in sig.referenced_types] == snapshot(
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
        FunctionSignature.from_schema(
            name='t1',
            parameters_schema={
                'type': 'object',
                'properties': {'x': {'type': ['string', 'null']}},
            },
        )
    ) == snapshot("""\
def t1(*, x: str | None = None) -> Any:
    ...\
""")

    # Optional not nullable → adds | None
    assert str(
        FunctionSignature.from_schema(
            name='t2',
            parameters_schema={
                'type': 'object',
                'properties': {'x': {'type': 'string'}},
            },
        )
    ) == snapshot("""\
def t2(*, x: str | None = None) -> Any:
    ...\
""")

    # Optional with explicit default preserves default value
    assert str(
        FunctionSignature.from_schema(
            name='t_default',
            parameters_schema={
                'type': 'object',
                'properties': {'x': {'type': 'integer', 'default': 7}},
            },
        )
    ) == snapshot("""\
def t_default(*, x: int = 7) -> Any:
    ...\
""")

    # Unresolvable return_schema → JSON blob in description
    sig3 = FunctionSignature.from_schema(
        name='t3',
        parameters_schema={'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x']},
        description='A tool',
        return_schema={},
    )
    assert str(sig3) == snapshot('''\
def t3(*, x: str) -> Any:
    """
    A tool

    Return schema:
    {}
    """
    ...\
''')

    # Return schema with $defs
    sig4 = FunctionSignature.from_schema(
        name='t4',
        parameters_schema={'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x']},
        return_schema={
            '$ref': '#/$defs/Result',
            '$defs': {'Result': {'type': 'object', 'properties': {'v': {'type': 'integer'}}, 'required': ['v']}},
        },
    )
    assert str(sig4) == snapshot("""\
def t4(*, x: str) -> Result:
    ...\
""")
    assert [t.render_definition() for t in sig4.referenced_types] == snapshot(
        [
            """\
class Result(TypedDict):
    v: int\
"""
        ]
    )

    # anyOf with duplicates → deduplicated
    assert str(
        FunctionSignature.from_schema(
            name='t5',
            parameters_schema={
                'type': 'object',
                'properties': {'x': {'anyOf': [{'type': 'string'}, {'type': 'string'}, {'type': 'null'}]}},
                'required': ['x'],
            },
        )
    ) == snapshot("""\
def t5(*, x: str | None) -> Any:
    ...\
""")


# =============================================================================
# Additional coverage tests
# =============================================================================


def test_tool_definition_eq_non_tool():
    """ToolDefinition does not equal non-ToolDefinition objects."""
    td = ToolDefinition(name='t')
    assert td != 'not a tool'


def test_function_signature_literal_annotation():
    """Literal type annotations exercise the repr fallback in _get_type_name."""
    ns: dict[str, object] = {'typing': typing}
    exec("def func(x: typing.Literal['a', 'b']) -> None: ...", ns)
    sig = FunctionSignature.from_function(ns['func'], name='func')  # pyright: ignore[reportArgumentType]
    assert "Literal['a', 'b']" in str(sig)


def test_function_with_bare_generic_annotation():
    """Bare generic (e.g. `typing.List` without type args) renders as the base type name."""
    # Create a function with bare generic annotation (typing.List, not list)
    # via a helper module that doesn't use `from __future__ import annotations`
    import types as t

    mod = t.ModuleType('_test_bare_generic')
    mod.__dict__['List'] = getattr(typing, 'List')  # typing.List (has origin but no args)
    exec(
        compile('def func(items: List) -> None: ...', '<test>', 'exec'),
        mod.__dict__,
    )
    sig = FunctionSignature.from_function(mod.func)
    assert 'items: list' in str(sig)


def test_function_signature_nameerror_fallback():
    """Functions with unresolvable forward refs fall back to empty type hints."""
    ns: dict[str, object] = {}
    exec(
        "def func_with_fwd_ref(x: 'NonexistentType') -> None: ...",
        ns,
    )
    func = ns['func_with_fwd_ref']
    sig = FunctionSignature.from_function(func, name='func_fwd')  # pyright: ignore[reportArgumentType]
    # Should not raise — falls back to empty hints, x becomes Any
    assert 'x' in sig.params


def test_schema_bare_object():
    """Bare object type (no properties, no additionalProperties) renders as dict."""
    sig = FunctionSignature.from_schema(
        name='t_bare',
        parameters_schema={
            'type': 'object',
            'properties': {'data': {'type': 'object'}},
            'required': ['data'],
        },
    )
    assert 'dict[str, Any]' in str(sig)


def test_schema_object_type_list_no_null():
    """Object in type list without null renders without union."""
    sig = FunctionSignature.from_schema(
        name='t_obj_list',
        parameters_schema={
            'type': 'object',
            'properties': {
                'config': {
                    'type': ['object'],
                    'properties': {'x': {'type': 'string'}},
                    'required': ['x'],
                },
            },
            'required': ['config'],
        },
    )
    rendered = str(sig)
    assert 'None' not in rendered
    assert 'TObjListConfig' in rendered


def test_schema_single_type_after_filtering():
    """Single-element type list renders as plain type."""
    sig = FunctionSignature.from_schema(
        name='t_single',
        parameters_schema={
            'type': 'object',
            'properties': {'x': {'type': ['string']}},
            'required': ['x'],
        },
    )
    assert 'str' in str(sig)
    # Should not be a union
    assert '|' not in str(sig)


def test_schema_single_anyof_member():
    """Single-member anyOf returns the type directly, not a union."""
    sig = FunctionSignature.from_schema(
        name='t_anyof_single',
        parameters_schema={
            'type': 'object',
            'properties': {'x': {'anyOf': [{'type': 'string'}]}},
            'required': ['x'],
        },
    )
    rendered = str(sig)
    assert 'str' in rendered
    assert '|' not in rendered


def test_schema_defs_already_processed():
    """Second call with same $defs name finds it already processed."""
    sig = FunctionSignature.from_schema(
        name='tool_shared',
        parameters_schema={
            'type': 'object',
            'properties': {
                'a': {'$ref': '#/$defs/Shared'},
                'b': {'$ref': '#/$defs/Shared'},
            },
            'required': ['a', 'b'],
            '$defs': {
                'Shared': {
                    'type': 'object',
                    'properties': {'v': {'type': 'integer'}},
                    'required': ['v'],
                }
            },
        },
    )
    # Both params reference the same TypeSignature
    assert str(sig.params['a'].type) == 'Shared'
    assert str(sig.params['b'].type) == 'Shared'
    # Only one referenced type
    assert len(sig.referenced_types) == 1


def test_schema_return_type_dedup():
    """Param and return schemas sharing a $defs type produce one definition."""
    shared_def = {
        'type': 'object',
        'properties': {'id': {'type': 'integer'}},
        'required': ['id'],
    }
    sig = FunctionSignature.from_schema(
        name='tool_dedup',
        parameters_schema={
            'type': 'object',
            'properties': {'item': {'$ref': '#/$defs/Item'}},
            'required': ['item'],
            '$defs': {'Item': shared_def},
        },
        return_schema={
            '$ref': '#/$defs/Item',
            '$defs': {'Item': shared_def},
        },
    )
    # Both param and return reference Item
    assert str(sig.params['item'].type) == 'Item'
    assert str(sig.return_type) == 'Item'


def test_schema_object_type_name_collision():
    """Two properties generating the same path-based type name — second is reused."""
    sig = FunctionSignature.from_schema(
        name='tool_collision',
        parameters_schema={
            'type': 'object',
            'properties': {
                'data': {
                    'type': 'object',
                    'properties': {'x': {'type': 'string'}},
                    'required': ['x'],
                },
            },
            'required': ['data'],
        },
    )
    # Should have a TypedDict for the nested object
    assert len(sig.referenced_types) == 1
    assert sig.referenced_types[0].name == 'ToolCollisionData'


def test_function_signature_root_model():
    """RootModel wrapping a non-object type renders as the type name without a TypedDict."""

    def func_with_root_model(x: _IntList) -> None: ...  # pragma: no cover

    sig = FunctionSignature.from_function(func_with_root_model, name='func_with_root_model')
    # RootModel produces a non-object schema, so it's referenced by name but not as a TypedDict
    assert str(sig) == snapshot("""\
def func_with_root_model(*, x: _IntList) -> None:
    ...\
""")
    assert sig.referenced_types == []


def test_function_signature_recursive_model():
    """Recursive BaseModel with top-level $ref in schema produces correct TypedDict."""

    def func_with_tree(x: _TreeNode) -> None: ...  # pragma: no cover

    sig = FunctionSignature.from_function(func_with_tree, name='func_with_tree')
    assert str(sig) == snapshot("""\
def func_with_tree(*, x: _TreeNode) -> None:
    ...\
""")
    assert len(sig.referenced_types) == 1
    assert sig.referenced_types[0].name == '_TreeNode'
    assert 'value' in sig.referenced_types[0].fields
    assert 'children' in sig.referenced_types[0].fields


def test_schema_cross_referencing_defs():
    """$ref in a def property lazily resolves another def not yet processed."""
    sig = FunctionSignature.from_schema(
        name='tool',
        parameters_schema={
            'type': 'object',
            'properties': {'item': {'$ref': '#/$defs/Container'}},
            'required': ['item'],
            '$defs': {
                'Container': {
                    'type': 'object',
                    'properties': {'inner': {'$ref': '#/$defs/Inner'}},
                    'required': ['inner'],
                },
                'Inner': {
                    'type': 'object',
                    'properties': {'value': {'type': 'string'}},
                    'required': ['value'],
                },
            },
        },
    )
    assert str(sig) == snapshot("""\
def tool(*, item: Container) -> Any:
    ...\
""")
    # Both Container and Inner should be resolved as TypeSignatures
    type_names = {t.name for t in sig.referenced_types}
    assert type_names == {'Container', 'Inner'}
    # Container's 'inner' field references the Inner TypeSignature
    container = next(t for t in sig.referenced_types if t.name == 'Container')
    assert str(container.fields['inner'].type) == 'Inner'


def test_schema_inline_object_reuses_existing_typename():
    """Inline object property reuses a $defs type when path-based names collide."""
    sig = FunctionSignature.from_schema(
        name='tool',
        parameters_schema={
            'type': 'object',
            'properties': {
                'data': {
                    'type': 'object',
                    'properties': {'x': {'type': 'string'}},
                    'required': ['x'],
                },
            },
            'required': ['data'],
            '$defs': {
                # This $def's name matches the path-based typename for property 'data'
                # _path_to_typename('tool', 'data') == 'ToolData'
                'ToolData': {
                    'type': 'object',
                    'properties': {'y': {'type': 'integer'}},
                    'required': ['y'],
                },
            },
        },
    )
    # The $defs ToolData was registered first; the inline 'data' property reuses it
    assert str(sig.params['data'].type) == 'ToolData'


def test_schema_additional_properties_false():
    """additionalProperties: false falls through to dict[str, Any]."""
    sig = FunctionSignature.from_schema(
        name='t_ap_false',
        parameters_schema={
            'type': 'object',
            'properties': {
                'meta': {'type': 'object', 'additionalProperties': False},
            },
            'required': ['meta'],
        },
    )
    assert 'dict[str, Any]' in str(sig)


def test_schema_empty_type_list():
    """Empty type list produces Any."""
    sig = FunctionSignature.from_schema(
        name='t_empty',
        parameters_schema={
            'type': 'object',
            'properties': {'x': {'type': []}},
            'required': ['x'],
        },
    )
    assert 'Any' in str(sig)


def test_function_with_typed_dict_param():
    """TypedDict parameters use TypeAdapter path for schema extraction."""
    import types as t

    from typing_extensions import TypedDict

    # Define in a separate module to avoid `from __future__ import annotations` interference
    mod = t.ModuleType('_test_typed_dict')
    mod.__dict__['TypedDict'] = TypedDict

    exec(
        compile(
            'class SearchParams(TypedDict):\n    query: str\n    limit: int\n'
            'def search(params: SearchParams) -> str: ...',
            '<test>',
            'exec',
        ),
        mod.__dict__,
    )
    sig = FunctionSignature.from_function(mod.search, name='search')
    assert 'params: SearchParams' in str(sig)
    assert any(rt.name == 'SearchParams' for rt in sig.referenced_types)


def test_schema_optional_with_anyof_null():
    """Optional schema params with anyOf containing null are correctly handled."""
    sig = FunctionSignature.from_schema(
        name='tool',
        parameters_schema={
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'tag': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
            },
            'required': ['name'],
        },
    )
    rendered = str(sig)
    assert 'tag: str | None' in rendered


def test_schema_optional_with_oneof_null():
    """Optional schema params with oneOf containing null are correctly handled."""
    sig = FunctionSignature.from_schema(
        name='tool',
        parameters_schema={
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'tag': {'oneOf': [{'type': 'string'}, {'type': 'null'}]},
            },
            'required': ['name'],
        },
    )
    rendered = str(sig)
    assert 'tag: str | None' in rendered


def test_function_with_enum_field_in_model():
    """Model with an enum field produces non-object $defs that are skipped during type collection."""

    def configure(cfg: _ConfigWithEnum) -> str:
        return cfg.name  # pragma: no cover

    sig = FunctionSignature.from_function(configure, name='configure')
    assert 'cfg: _ConfigWithEnum' in str(sig)
    # _ConfigWithEnum should be a TypedDict, but _Color (enum) should not
    type_names = {rt.name for rt in sig.referenced_types}
    assert '_ConfigWithEnum' in type_names
    assert '_Color' not in type_names


def test_function_tool_definition_eq_with_non_tool():
    """_FunctionToolDefinition.__eq__ returns NotImplemented for non-ToolDefinition."""
    tool = Tool(lambda x: x, name='t', description='t')
    td = tool.tool_def
    assert td != 'not a tool definition'
    assert td != 42
