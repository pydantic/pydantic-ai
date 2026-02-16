"""Tests for Monty-specific type checking and signature generation."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot
from typing_extensions import NotRequired, TypedDict

try:
    from pydantic_monty import Monty

    from pydantic_ai.runtime.monty import (
        MontyRuntime,
        _build_type_check_prefix,  # pyright: ignore[reportPrivateUsage]
    )
except ImportError:  # pragma: lax no cover
    pytest.skip('pydantic-monty is not installed', allow_module_level=True)

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
from pydantic_ai.exceptions import ModelRetry

from .conftest import build_code_mode_toolset, run_code_with_tools

pytestmark = pytest.mark.anyio


def add(*, x: int, y: int) -> int:
    """Add two integers."""
    return x + y


async def test_type_error_raises_model_retry():
    """Type errors in generated code raise ModelRetry so the LLM can fix them."""
    with pytest.raises(ModelRetry) as exc_info:
        await run_code_with_tools('add(x="hello", y="world")', MontyRuntime(), (add, False))

    assert str(exc_info.value) == snapshot("""\
Type error in generated code:
main.py:1:5: error[invalid-argument-type] Argument to function `add` is incorrect: Expected `int`, found `Literal["hello"]`
main.py:1:16: error[invalid-argument-type] Argument to function `add` is incorrect: Expected `int`, found `Literal["world"]`
""")


async def test_generated_signatures_are_valid_python():
    """Generated signatures must be valid Python that Monty can parse and type check."""
    _, tools = await build_code_mode_toolset(MontyRuntime(), (add, False))

    tool = tools['run_code_with_tools']
    prefix = _build_type_check_prefix(tool.cached_signatures)

    # `...` and `pass` are not valid for Monty/ty type checking — ty is intentionally
    # stricter than pyright here. See https://github.com/astral-sh/ty/issues/1922
    assert prefix == snapshot('''\
from typing import Any, TypedDict, NotRequired, Literal

async def add(*, x: int, y: int) -> int:
    """Add two integers."""
    raise NotImplementedError()\
''')
    # Verify Monty can parse and type check code using this prefix
    m = Monty('add(x=1, y=2)', external_functions=['add'])
    m.type_check(prefix_code=prefix)  # Should not raise


async def test_signatures_use_ellipsis_monty_converts_for_type_check():
    """Signatures use '...' body; Monty converts to 'raise NotImplementedError()' for type checking."""
    _code_mode, tools = await build_code_mode_toolset(MontyRuntime(), (add, False))

    tool = tools['run_code_with_tools']

    # LLM-facing description should have '...'
    description = tool.tool_def.description or ''
    assert '...' in description
    assert 'raise NotImplementedError()' not in description

    # But when Monty builds the type-check prefix, it converts to 'raise NotImplementedError()'
    prefix = _build_type_check_prefix(tool.cached_signatures)
    assert 'raise NotImplementedError()' in prefix
    assert '    ...' not in prefix


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


# --- Types and tools for test_full_description_snapshot ---


class _Tag(TypedDict):
    """A key-value tag."""

    key: str
    value: str


class _Resource(TypedDict):
    name: str
    tags: list[_Tag]
    metadata: NotRequired[dict[str, str]]
    """Extra metadata."""
    parent_id: int | None


def _find_resources(*, query: str, limit: int = 10) -> list[_Resource]:
    """Find resources matching a query."""
    return []


# Both have __name__ == 'Item' to test dedup when different tools share a type name
class _SearchItem(TypedDict):
    name: str
    price: float


class _LookupItem(TypedDict):
    id: int
    category: str


def _search_items(*, query: str) -> list[_SearchItem]:
    """Search for items by name."""
    return []


def _get_item(*, item_id: int) -> list[_LookupItem]:
    """Get items by ID."""
    return []


def _tag_resource(*, resource_name: str, tag: _Tag) -> bool:
    """Add a tag to a resource."""
    return True


async def test_full_description_snapshot():
    """Snapshot the full run_code_with_tools description with shared types, conflicts, and nesting."""
    _, tools = await build_code_mode_toolset(
        MontyRuntime(),
        (_find_resources, False),
        (_search_items, False),
        (_get_item, False),
        (_tag_resource, False),
    )
    description = tools['run_code_with_tools'].tool_def.description
    assert description == snapshot('''\

Use `run_code_with_tools` to write Python code that calls the available functions. You can make a single call or combine multiple steps in a script — use your judgment based on the task.

Execution model:
- Each `run_code_with_tools` call runs in an isolated environment — variables don't persist between calls
- Functions are async — call with `await`, e.g. `result = await get_items()`
- To run independent calls concurrently, fire them first, then await:
  ```python
  future_a = get_items()    # starts immediately
  future_b = get_users()    # starts immediately
  items = await future_a    # wait for results
  users = await future_b
  ```
- The last expression evaluated is the return value
- Return raw data when it answers the question directly; extract or transform when needed


Syntax note: the runtime uses a restricted Python subset.
- Imports are not available — use the provided functions and builtins (len, sum, str, etc.) or define your own helpers.

```python

# Available types:

class _Tag(TypedDict):
    key: str
    value: str

class _Resource(TypedDict):
    name: str
    tags: list[_Tag]
    metadata: NotRequired[dict[str, str]]
    parent_id: int | None

class Item(TypedDict):
    name: str
    price: float

class _get_item_Item(TypedDict):
    id: int
    category: str

# Available functions:

async def _find_resources(*, query: str, limit: int = 10) -> list[_Resource]:
    """Find resources matching a query."""
    ...

async def _search_items(*, query: str) -> list[Item]:
    """Search for items by name."""
    ...

async def _get_item(*, item_id: int) -> list[_get_item_Item]:
    """Get items by ID."""
    ...

async def _tag_resource(*, resource_name: str, tag: _Tag) -> bool:
    """Add a tag to a resource."""
    ...

```\
''')
