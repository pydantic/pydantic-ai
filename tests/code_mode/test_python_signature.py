"""Tests for Python signature generation and deduplication."""

from __future__ import annotations

from inline_snapshot import snapshot

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
    render_type_expr,
    schema_to_signature,
)


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
    assert sig2.params['user'].render() == 'user: tool_b_User'
    assert sig2.params['meta'].render() == 'meta: UserMeta'
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
    assert sig3.params['user'].render() == 'user: tool_c_User'

    # collect_unique_referenced_types returns exactly 2 unique types
    unique_types = collect_unique_referenced_types([sig1, sig2, sig3])
    assert [t.render() for t in unique_types] == snapshot(
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
    assert sig2.params['users'].render() == 'users: list[tool_b_User]'


def test_render_type_signature():
    """TypeSignature.render() produces a valid TypedDict class definition."""
    ts = TypeSignature(
        name='User',
        fields={
            'name': TypeFieldSignature(name='name', type='str', required=True, description=None),
            'age': TypeFieldSignature(name='age', type='int', required=False, description='The age'),
        },
    )
    assert ts.render() == snapshot("""\
class User(TypedDict):
    name: str
    age: NotRequired[int]
    \"\"\"The age\"\"\"\
""")


def test_render_type_signature_empty():
    """Empty TypeSignature renders with pass."""
    ts = TypeSignature(name='Empty')
    assert ts.render() == 'class Empty(TypedDict):\n    pass'


def test_render_generic_type_expr():
    """GenericTypeExpr renders correctly."""
    user = TypeSignature(name='User')
    expr = GenericTypeExpr(base='list', args=[user])
    assert expr.render() == 'list[User]'

    dict_expr = GenericTypeExpr(base='dict', args=['str', GenericTypeExpr(base='list', args=[user])])
    assert dict_expr.render() == 'dict[str, list[User]]'


def test_render_union_type_expr():
    """UnionTypeExpr renders correctly."""
    user = TypeSignature(name='User')
    expr = UnionTypeExpr(members=[user, 'None'])
    assert expr.render() == 'User | None'


def test_render_function_param():
    """FunctionParam.render() produces correct parameter strings."""
    p1 = FunctionParam(name='x', type='int', default=None)
    assert p1.render() == 'x: int'

    p2 = FunctionParam(name='y', type='str', default="'hello'")
    assert p2.render() == "y: str = 'hello'"

    user = TypeSignature(name='User')
    p3 = FunctionParam(name='user', type=user, default=None)
    assert p3.render() == 'user: User'


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
