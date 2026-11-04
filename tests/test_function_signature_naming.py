"""Every type in a rendered tool catalog has one name and one definition.

These are unit tests because the names decide the text a model reads and the stubs Monty type checks, with no model
request involved. `tests/harness/code_mode/test_code_mode.py` holds Monty itself to the same cases. Schemas come from
real Pydantic models wherever the case allows, as that is how tools get them.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from itertools import product
from typing import Any

import pytest
from pydantic import BaseModel, create_model

from pydantic_ai.function_signature import FunctionSignature, SimpleTypeExpr, TypeFieldSignature, TypeSignature

from ._inline_snapshot import snapshot


def _model(name: str, /, **fields: Any) -> type[BaseModel]:
    """A Pydantic model, with every field required."""
    definitions: dict[str, Any] = {field: (annotation, ...) for field, annotation in fields.items()}
    return create_model(name, **definitions)


def _params(**params: Any) -> dict[str, Any]:
    """A tool's parameter schema, as Pydantic builds it from typed arguments."""
    schema = _model('Args', **params).model_json_schema()
    del schema['title']
    return schema


def _signature(name: str, returns: type[BaseModel] | None = None, **params: Any) -> FunctionSignature:
    return_schema = returns.model_json_schema() if returns else None
    return FunctionSignature.from_schema(name=name, parameters_schema=_params(**params), return_schema=return_schema)


def _catalog(*signatures: FunctionSignature) -> str:
    """The text code mode builds: every definition once, then every function."""
    conflicting = FunctionSignature.get_conflicting_type_names(list(signatures))
    definitions = FunctionSignature.render_type_definitions(list(signatures), conflicting)
    functions = [signature.render('...', conflicting_type_names=conflicting) for signature in signatures]
    return '\n\n'.join([*definitions, *functions])


def _user_with_name() -> type[BaseModel]:
    return _model('User', name=str)


def _user_with_id() -> type[BaseModel]:
    return _model('User', id=int)


def _order_of(user: type[BaseModel]) -> type[BaseModel]:
    return _model('Order', user=user)


# =============================================================================
# Two types with one name
# =============================================================================


def test_param_and_return_types_with_one_name_get_two_names():
    """Case 1 of #8816: both types rendered as `sync_user_User`, and `RetWrap.user` as a plain `User`."""
    ret_wrap = _model('RetWrap', user=_user_with_id())
    assert _catalog(_signature('sync_user', returns=ret_wrap, user=_user_with_name())) == snapshot("""\
class sync_user_User(TypedDict):
    name: str

class sync_user_User_2(TypedDict):
    id: int

class RetWrap(TypedDict):
    user: sync_user_User_2

def sync_user(*, user: sync_user_User) -> RetWrap:
    ...\
""")


def test_type_keeping_its_name_refers_to_a_renamed_type():
    """Case 2 of #8816: `Wrapper.user` rendered as a plain `User`, which nothing defined."""
    wrapper = _model('Wrapper', user=_user_with_name())
    assert _catalog(_signature('tool_a', wrap=wrapper), _signature('tool_b', user=_user_with_id())) == snapshot("""\
class tool_a_User(TypedDict):
    name: str

class Wrapper(TypedDict):
    user: tool_a_User

class tool_b_User(TypedDict):
    id: int

def tool_a(*, wrap: Wrapper) -> Any:
    ...

def tool_b(*, user: tool_b_User) -> Any:
    ...\
""")


def test_generated_name_does_not_take_a_real_name():
    real = _model('tool_a_User', z=str)
    catalog = _catalog(
        _signature('tool_a', user=_user_with_name(), other=real), _signature('tool_b', user=_user_with_id())
    )
    assert 'class tool_a_User(TypedDict):\n    z: str' in catalog
    assert 'class tool_a_User_2(TypedDict):\n    name: str' in catalog
    assert 'def tool_a(*, user: tool_a_User_2, other: tool_a_User) -> Any:' in catalog


def test_generated_name_does_not_take_a_name_generated_earlier():
    three_users = [_model('User', **{field: str}) for field in ('a', 'b', 'c')]
    ret_wrap = _model('RetWrap', user=three_users[1])
    catalog = _catalog(_signature('t', returns=ret_wrap, user=three_users[0]), _signature('t_', user=three_users[2]))
    assert 'class t_User(TypedDict):\n    a: str' in catalog
    assert 'class t_User_2(TypedDict):\n    b: str' in catalog
    assert 'class t__User(TypedDict):\n    c: str' in catalog


@pytest.mark.parametrize('name', ['Any', 'Literal', 'NotRequired', 'TypedDict', 'asyncio', 'str', 'list'])
def test_type_cannot_take_a_name_the_text_relies_on(name: str):
    catalog = _catalog(_signature('send', x=_model(name, a=int), y=Any))
    assert (
        catalog == f'class send_{name}(TypedDict):\n    a: int\n\ndef send(*, x: send_{name}, y: Any) -> Any:\n    ...'
    )


def test_type_cannot_take_the_name_of_a_tool():
    catalog = _catalog(_signature('User', user=_user_with_name()))
    assert catalog == 'class User_User(TypedDict):\n    name: str\n\ndef User(*, user: User_User) -> Any:\n    ...'


def test_generated_name_does_not_take_the_name_of_a_tool():
    catalog = _catalog(_signature('a', user=_user_with_name()), _signature('a_User', user=_user_with_id()))
    assert 'class a_User_2(TypedDict):\n    name: str' in catalog
    assert 'def a_User(*, user: a_User_User) -> Any:' in catalog


def test_result_is_the_set_of_names_that_changed():
    signatures = [
        _signature('a', user=_user_with_name(), x=_model('Any', z=int), kept=_model('Kept', z=int)),
        _signature('b', user=_user_with_id()),
    ]
    assert FunctionSignature.get_conflicting_type_names(signatures) == {'User', 'Any'}


# =============================================================================
# One type in two places
# =============================================================================


def test_same_type_in_two_tools_is_defined_once():
    user = _user_with_name()
    catalog = _catalog(_signature('a', user=user), _signature('b', order=_order_of(user)))
    assert catalog.count('class User(TypedDict):') == 1
    assert 'def a(*, user: User) -> Any:' in catalog


def test_same_type_in_param_and_return_is_defined_once():
    user = _user_with_name()
    catalog = _catalog(_signature('a', returns=_order_of(user), order=_order_of(user)))
    assert catalog.count('class User(TypedDict):') == 1
    assert catalog.count('class Order(TypedDict):') == 1


def test_types_differing_only_in_a_nested_type_stay_two_types():
    """The two `Order`s have the same field text, `user: User`, so comparing that text merged them."""
    assert _catalog(
        _signature('a', order=_order_of(_user_with_name())), _signature('b', order=_order_of(_user_with_id()))
    ) == snapshot("""\
class a_Order(TypedDict):
    user: a_User

class a_User(TypedDict):
    name: str

class b_Order(TypedDict):
    user: b_User

class b_User(TypedDict):
    id: int

def a(*, order: a_Order) -> Any:
    ...

def b(*, order: b_Order) -> Any:
    ...\
""")


def test_types_differing_only_in_a_required_flag_stay_two_types():
    optional_name: dict[str, Any] = {'name': (str, 'x')}
    optional_user = create_model('User', **optional_name)
    catalog = _catalog(_signature('a', user=_user_with_name()), _signature('b', user=optional_user))
    assert 'class a_User(TypedDict):\n    name: str' in catalog
    assert 'class b_User(TypedDict):\n    name: NotRequired[str]' in catalog


def test_every_tool_lists_the_merged_type_so_it_can_render_alone():
    """Dynamic workflows render part of a catalog, reusing the names of the whole."""
    user = _user_with_name()
    signatures = [_signature('a', user=user), _signature('b', user=user), _signature('c', user=_user_with_id())]
    conflicting = FunctionSignature.get_conflicting_type_names(signatures)
    assert FunctionSignature.render_type_definitions(signatures[1:2], conflicting) == [
        'class a_User(TypedDict):\n    name: str'
    ]
    assert signatures[1].render('...', conflicting_type_names=conflicting) == 'def b(*, user: a_User) -> Any:\n    ...'


# =============================================================================
# Comparing structure
# =============================================================================


def _type(name: str, **fields: Any) -> TypeSignature:
    return TypeSignature(
        name=name,
        fields={field: TypeFieldSignature(name=field, type=expr, required=True) for field, expr in fields.items()},
    )


_STR = SimpleTypeExpr('str')
_INT = SimpleTypeExpr('int')


def test_structurally_equal_looks_inside_nested_types():
    assert _type('A', x=_type('N', v=_STR)).structurally_equal(_type('A', x=_type('N', v=_STR)))
    assert not _type('A', x=_type('N', v=_STR)).structurally_equal(_type('A', x=_type('N', v=_INT)))


def test_structurally_equal_ignores_own_name_but_not_nested_names():
    assert _type('A', v=_STR).structurally_equal(_type('B', v=_STR))
    assert not _type('A', x=_type('N', v=_STR)).structurally_equal(_type('A', x=_type('M', v=_STR)))


def _cycle(last: SimpleTypeExpr) -> TypeSignature:
    """`Node -> Next -> Node`, with `last` as the only thing setting one cycle apart from another."""
    node, next_type = _type('Node'), _type('Next', leaf=last)
    node.fields['next'] = TypeFieldSignature(name='next', type=next_type, required=True)
    next_type.fields['node'] = TypeFieldSignature(name='node', type=node, required=True)
    return node


def test_structurally_equal_ends_on_cycles_and_finds_differences_in_them():
    assert _cycle(_STR).structurally_equal(_cycle(_STR))
    assert not _cycle(_STR).structurally_equal(_cycle(_INT))


def test_structurally_equal_finds_a_difference_at_the_end_of_a_chain():
    def chain(leaf: SimpleTypeExpr) -> TypeSignature:
        current = _type('Leaf', v=leaf)
        for i in range(50):
            current = _type(f'T{i}', x=current)
        return current

    assert not chain(_STR).structurally_equal(chain(_INT))


def test_recursive_type_in_two_tools_is_defined_once():
    class Node(BaseModel):
        value: str
        next: Node | None = None

    catalog = _catalog(_signature('a', node=Node, other=Any), _signature('b', node=Node, other=Any))
    assert catalog.count('class Node(TypedDict):') == 1


# =============================================================================
# Staying compatible with how signatures get used
# =============================================================================


def test_names_passed_without_resolving_still_take_a_prefix():
    """`render` and `render_definition` keep working with a set of names `get_conflicting_type_names` didn't see."""
    signature = _signature('a', wrap=_model('Wrapper', user=_user_with_name()))
    names = frozenset({'User'})
    assert FunctionSignature.render_type_definitions([signature], names) == [
        'class a_User(TypedDict):\n    name: str',
        'class Wrapper(TypedDict):\n    user: a_User',
    ]


def test_definition_rendered_without_names_keeps_its_own_name():
    signatures = [_signature('a', user=_user_with_name()), _signature('b', user=_user_with_id())]
    FunctionSignature.get_conflicting_type_names(signatures)
    assert signatures[0].referenced_types[0].render_definition() == 'class User(TypedDict):\n    name: str'


def test_resolving_again_gives_the_names_of_the_new_set():
    """The limit `get_conflicting_type_names` documents: names belong to the set resolved last."""
    a, b = _signature('a', user=_user_with_name()), _signature('b', user=_user_with_id())
    assert _catalog(a, b).startswith('class a_User(TypedDict):')
    assert _catalog(a) == 'class User(TypedDict):\n    name: str\n\ndef a(*, user: User) -> Any:\n    ...'


def test_resolving_again_clears_a_name_the_new_set_does_not_need():
    item_a, item_c = _model('Item', a=str), _model('Item', c=str)
    a = _signature('a', user=_user_with_name(), item=item_a)
    b, c = _signature('b', user=_user_with_id()), _signature('c', item=item_c)
    assert 'class a_User(TypedDict):' in _catalog(a, b)
    catalog = _catalog(a, c)
    assert 'class User(TypedDict):' in catalog
    assert 'class a_Item(TypedDict):' in catalog


def test_copied_signatures_keep_their_names():
    signatures = [_signature('a', user=_user_with_name()), _signature('b', user=_user_with_id())]
    conflicting = FunctionSignature.get_conflicting_type_names(signatures)
    expected = FunctionSignature.render_type_definitions(signatures, conflicting)
    assert FunctionSignature.render_type_definitions(deepcopy(signatures), conflicting) == expected


def test_resolved_name_is_not_part_of_the_type():
    signatures = [_signature('a', user=_user_with_name()), _signature('b', user=_user_with_id())]
    resolved = signatures[0].referenced_types[0]
    plain = _signature('a', user=_user_with_name()).referenced_types[0]
    FunctionSignature.get_conflicting_type_names(signatures)
    assert resolved == plain
    assert repr(resolved) == repr(plain)


def test_ref_that_does_not_resolve_keeps_rendering():
    """Such a `$ref` leaves a type no signature lists, which unifying must still handle."""
    schema = {'type': 'object', 'properties': {'x': {'$ref': '#/definitions/U'}}, 'required': ['x']}
    signature = FunctionSignature.from_schema(name='a', parameters_schema=schema)
    assert _catalog(signature) == 'def a(*, x: U) -> Any:\n    ...'


# =============================================================================
# The issue's two rules, over every small catalog
# =============================================================================

_TEXT_NAMES = {'Any', 'NotRequired', 'str', 'int', 'list', 'None'}


def _object_of(field: str, node: dict[str, Any]) -> dict[str, Any]:
    return {'type': 'object', 'properties': {field: node}, 'required': [field]}


def _document(names: tuple[str, str], outer_form: str, inner: tuple[str, str]) -> dict[str, Any]:
    """A parameter `p` of a type `outer` holding `inner` in one of the forms a schema can, or a plain string."""
    outer_name, inner_name = names
    inner_field, inner_type = inner
    ref = {'$ref': f'#/$defs/{inner_name}'}
    forms: dict[str, dict[str, Any]] = {
        'plain': {'type': 'string'},
        'ref': ref,
        'list': {'type': 'array', 'items': ref},
        'optional': {'anyOf': [ref, {'type': 'null'}]},
    }
    return {
        **_object_of('p', {'$ref': f'#/$defs/{outer_name}'}),
        '$defs': {
            outer_name: _object_of('f', forms[outer_form]),
            inner_name: _object_of(inner_field, {'type': inner_type}),
        },
    }


_DOCUMENTS = [
    _document(names, form, inner)
    for names, form, inner in product(
        [('User', 'Address'), ('Address', 'a_User'), ('Any', 'User')],
        ['plain', 'ref', 'list', 'optional'],
        [('g', 'string'), ('g', 'integer'), ('h', 'string')],
    )
]


def _meaning_of_schema(node: dict[str, Any], root: dict[str, Any]) -> Any:
    if '$ref' in node:
        return _meaning_of_schema(root['$defs'][node['$ref'].rsplit('/', 1)[1]], root)
    if 'anyOf' in node:
        return ('optional', _meaning_of_schema(node['anyOf'][0], root))
    if node['type'] == 'array':
        return ('list', _meaning_of_schema(node['items'], root))
    if node['type'] == 'object':
        return {field: _meaning_of_schema(value, root) for field, value in node['properties'].items()}
    return {'string': 'str', 'integer': 'int'}[node['type']]


def _meaning_of_annotation(annotation: ast.expr, classes: dict[str, ast.ClassDef]) -> Any:
    """What an annotation means in the text, failing on any name nothing defines."""
    if isinstance(annotation, ast.BinOp):
        return ('optional', _meaning_of_annotation(annotation.left, classes))
    if isinstance(annotation, ast.Subscript):
        return ('list', _meaning_of_annotation(annotation.slice, classes))
    assert isinstance(annotation, ast.Name)
    if annotation.id in _TEXT_NAMES:
        return annotation.id
    fields = [statement for statement in classes[annotation.id].body if isinstance(statement, ast.AnnAssign)]
    return {
        field.target.id: _meaning_of_annotation(field.annotation, classes)
        for field in fields
        if isinstance(field.target, ast.Name)
    }


def _assert_every_name_is_unique_and_means_its_schema(catalog: str, tools: dict[str, dict[str, Any]]) -> None:
    body = ast.parse(catalog).body
    classes = {node.name: node for node in body if isinstance(node, ast.ClassDef)}
    assert sum(isinstance(node, ast.ClassDef) for node in body) == len(classes), catalog
    assert not classes.keys() & {*_TEXT_NAMES, *tools}, catalog
    for function in body:
        if isinstance(function, ast.FunctionDef):
            schema = tools[function.name]
            annotation = function.args.kwonlyargs[0].annotation
            assert annotation is not None
            assert _meaning_of_annotation(annotation, classes) == _meaning_of_schema(
                schema['properties']['p'], schema
            ), catalog


def test_every_small_pair_of_tools():
    for params_a, params_b in product(_DOCUMENTS, repeat=2):
        signatures = [
            FunctionSignature.from_schema(name='a', parameters_schema=params_a),
            FunctionSignature.from_schema(name='a_b', parameters_schema=params_b),
        ]
        _assert_every_name_is_unique_and_means_its_schema(_catalog(*signatures), {'a': params_a, 'a_b': params_b})


def test_every_small_document_as_params_and_return():
    for params, returns in product(_DOCUMENTS, repeat=2):
        signature = FunctionSignature.from_schema(name='a', parameters_schema=params, return_schema=returns)
        catalog = _catalog(signature)
        _assert_every_name_is_unique_and_means_its_schema(catalog, {'a': params})
        function = next(node for node in ast.parse(catalog).body if isinstance(node, ast.FunctionDef))
        classes = {node.name: node for node in ast.parse(catalog).body if isinstance(node, ast.ClassDef)}
        assert function.returns is not None
        assert _meaning_of_annotation(function.returns, classes) == _meaning_of_schema(returns, returns), catalog
