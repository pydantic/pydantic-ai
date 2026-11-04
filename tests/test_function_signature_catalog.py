"""What `FunctionSignature` guarantees about a catalog: the text describing a set of tools.

Code mode shows this text to the model and type checks sandbox code against it, so a wrong name or shape makes correct
calls fail or wrong calls pass. `tests/harness/code_mode/test_code_mode.py` runs the type checker over the same forms.

Each section below is one guarantee:

1. Every type has one name and every name has one type.
2. Types merge only when they are the same all the way down.
3. A `$ref` means the schema it points at, or `Any` if this document can't answer.
4. Names Python can't write stay exact and usable.
5. Names stay right across the ways callers use the result.
6. Signatures built by hand get the same guarantees.

These are unit tests rather than recorded requests because rendering involves no model. `_catalog` also checks all the
text against the schemas with the independent checker in `tests/function_signature_invariants.py`.
"""

from __future__ import annotations

import copy
import dataclasses
import pickle
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import TypeAdapter

from pydantic_ai.function_signature import (
    FunctionParam,
    FunctionSignature,
    GenericTypeExpr,
    LiteralTypeExpr,
    SimpleTypeExpr,
    TypeExpr,
    TypeFieldSignature,
    TypeSignature,
    UnionTypeExpr,
)

from ._inline_snapshot import snapshot
from .function_signature_invariants import Tool, catalog_violations

_STR: dict[str, Any] = {'type': 'string'}
_INT: dict[str, Any] = {'type': 'integer'}


def _object(properties: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {'type': 'object', 'properties': properties, 'required': list(properties), **extra}


def _ref(name: str) -> dict[str, Any]:
    return {'$ref': f'#/$defs/{name}'}


def _with_defs(properties: dict[str, Any], **defs: Any) -> dict[str, Any]:
    return _object(properties, **{'$defs': defs})


def _signature(tool: Tool) -> FunctionSignature:
    return FunctionSignature.from_schema(name=tool.name, parameters_schema=tool.params, return_schema=tool.returns)


def _render(signatures: list[FunctionSignature]) -> str:
    conflicting = FunctionSignature.get_conflicting_type_names(signatures)
    definitions = FunctionSignature.render_type_definitions(signatures, conflicting)
    functions = [s.render('...', conflicting_type_names=conflicting) for s in signatures]
    return '\n\n'.join([*definitions, *functions])


def _catalog(*tools: Tool) -> str:
    """Render tools the way code mode does, and check the text against their schemas."""
    text = _render([_signature(t) for t in tools])
    assert catalog_violations(text, list(tools)) == [], text
    return text


def _pickled(value: Any) -> Any:
    return pickle.loads(pickle.dumps(value))


def _user_tools(**field_by_tool: str) -> list[FunctionSignature]:
    """One signature per tool, each taking a `User` with the given field."""
    return [
        _signature(Tool(name, _with_defs({'u': _ref('User')}, User=_object({field_name: _STR}))))
        for name, field_name in field_by_tool.items()
    ]


# =============================================================================
# 1. Every type has one name and every name has one type
# =============================================================================


def test_issue_8816_param_and_return_define_different_user():
    tool = Tool(
        'sync_user',
        _with_defs({'user': _ref('User')}, User=_object({'name': _STR})),
        {
            '$ref': '#/$defs/RetWrap',
            '$defs': {'User': _object({'id': _INT, 'email': _STR}), 'RetWrap': _object({'user': _ref('User')})},
        },
    )
    assert _catalog(tool) == snapshot("""\
class sync_user_User(TypedDict):
    name: str

class sync_user_User_2(TypedDict):
    id: int
    email: str

class RetWrap(TypedDict):
    user: sync_user_User_2

def sync_user(*, user: sync_user_User) -> RetWrap:
    ...\
""")


def test_issue_8816_unrenamed_type_points_at_a_renamed_one():
    tool_a = Tool(
        'tool_a', _with_defs({'w': _ref('Wrapper')}, Wrapper=_object({'user': _ref('User')}), User=_object({'a': _STR}))
    )
    tool_b = Tool('tool_b', _with_defs({'u': _ref('User')}, User=_object({'b': _INT})))
    assert _catalog(tool_a, tool_b) == snapshot("""\
class Wrapper(TypedDict):
    user: tool_a_User

class tool_a_User(TypedDict):
    a: str

class tool_b_User(TypedDict):
    b: int

def tool_a(*, w: Wrapper) -> Any:
    ...

def tool_b(*, u: tool_b_User) -> Any:
    ...\
""")


def test_generated_name_never_takes_a_real_name():
    tool_a = Tool('a', _with_defs({'x': _ref('User')}, User=_object({'p': _STR})))
    tool_a_b = Tool('a_b', _with_defs({'x': _ref('User')}, User=_object({'q': _INT})))
    tool_c = Tool('c', _with_defs({'x': _ref('a_b_User')}, a_b_User=_object({'r': _STR})))
    text = _catalog(tool_a, tool_a_b, tool_c)
    assert 'class a_b_User(TypedDict):\n    r: str' in text
    assert 'class a_b_User_2(TypedDict):\n    q: int' in text


@pytest.mark.parametrize(
    'properties,signature',
    [
        pytest.param(
            {'a': _object({'x': _STR}, title='Item'), 'b': _object({'y': _INT}, title='Item')},
            'def t(*, a: t_Item, b: t_Item_2) -> Any:',
            id='two-properties',
        ),
        pytest.param(
            {'a': {'anyOf': [_object({'x': _STR}, title='Item'), _object({'y': _INT}, title='Item')]}},
            'def t(*, a: t_Item | t_Item_2) -> Any:',
            id='union-members',
        ),
        pytest.param(
            {
                'a': {
                    'anyOf': [
                        {'type': 'array', 'items': _object({'x': _STR}, title='Item')},
                        {'type': 'array', 'items': _object({'y': _INT}, title='Item')},
                    ]
                }
            },
            'def t(*, a: list[t_Item] | list[t_Item_2]) -> Any:',
            id='union-of-lists',
        ),
    ],
)
def test_inline_objects_with_one_title_stay_two_types(properties: dict[str, Any], signature: str):
    assert signature in _catalog(Tool('t', _object(properties)))


def test_inline_objects_with_one_generated_name_stay_two_types():
    tool = Tool(
        't', _object({'home_address': _object({'x': _STR}), 'home': _object({'address': _object({'y': _INT})})})
    )
    text = _catalog(tool)
    assert 'class t_THomeAddress(TypedDict):\n    x: str' in text
    assert 'class t_THomeAddress_2(TypedDict):\n    y: int' in text


def test_inline_object_does_not_become_a_def_with_its_generated_name():
    tool = Tool('tool', _with_defs({'data': _object({'x': _STR})}, ToolData=_object({'y': _INT})))
    assert _catalog(tool) == snapshot("""\
class tool_ToolData(TypedDict):
    y: int

class tool_ToolData_2(TypedDict):
    x: str

def tool(*, data: tool_ToolData_2) -> Any:
    ...\
""")


@pytest.mark.parametrize(
    'name',
    ['Any', 'str', 'list', 'TypedDict', 'NotRequired', 'Literal', 'asyncio', 'class', 'None', '__debug__'],
)
def test_type_cannot_take_a_name_the_text_relies_on(name: str):
    text = _catalog(Tool('t', _object({'u': _ref(name)}, **{'$defs': {name: _object({'a': _STR})}})))
    assert f'class t_{name}(TypedDict):' in text
    assert f'def t(*, u: t_{name}) -> Any:' in text


def test_type_cannot_take_the_name_of_a_tool():
    text = _catalog(Tool('User', _with_defs({'u': _ref('User')}, User=_object({'a': _STR}))))
    assert 'def User(*, u: User_User) -> Any:' in text


@pytest.mark.parametrize(
    'name,ref,expected',
    [
        pytest.param('my.Type', 'my.Type', 't_my_Type', id='dot'),
        pytest.param('a b', 'a%20b', 't_a_b', id='space'),
        pytest.param('1st', '1st', 't_1st', id='leading-digit'),
        pytest.param('\ufb01le', '\ufb01le', 't_file', id='changed-by-normalization'),
    ],
)
def test_type_name_python_cannot_read_back_is_rewritten(name: str, ref: str, expected: str):
    text = _catalog(Tool('t', _object({'u': _ref(ref)}, **{'$defs': {name: _object({'a': _STR})}})))
    assert f'class {expected}(TypedDict):' in text


def _definitions(tool: Tool) -> list[str]:
    signatures = [_signature(tool)]
    return FunctionSignature.render_type_definitions(
        signatures, FunctionSignature.get_conflicting_type_names(signatures)
    )


def test_prefix_from_a_tool_name_python_cannot_write_is_rewritten():
    """Code mode renders such a tool under a sanitized name, so this renders definitions only: `def 1-t` is never written."""
    tool = Tool('1-t', _with_defs({'x': _ref('Any')}, Any=_object({'a': _STR})))
    assert _definitions(tool) == ['class _1_t_Any(TypedDict):\n    a: str']


def test_type_cannot_take_the_sanitized_name_of_a_tool():
    tool = Tool('my-tool', _with_defs({'x': _ref('my_tool')}, my_tool=_object({'a': _STR})))
    assert _definitions(tool) == ['class my_tool_my_tool(TypedDict):\n    a: str']


def test_only_names_the_text_relies_on_are_reserved():
    """Reserving every builtin would rename correct output and make it vary with the running Python's builtins."""
    assert 'class Warning(TypedDict):' in _catalog(
        Tool('t', _with_defs({'x': _ref('Warning')}, Warning=_object({'a': _STR})))
    )


# =============================================================================
# 2. Types merge only when they are the same all the way down
# =============================================================================


def test_same_type_in_two_tools_is_defined_once():
    def tool(name: str) -> Tool:
        return Tool(
            name, _with_defs({'x': _ref('Outer')}, Outer=_object({'inner': _ref('Inner')}), Inner=_object({'a': _STR}))
        )

    assert _catalog(tool('one'), tool('two')) == snapshot("""\
class Outer(TypedDict):
    inner: Inner

class Inner(TypedDict):
    a: str

def one(*, x: Outer) -> Any:
    ...

def two(*, x: Outer) -> Any:
    ...\
""")


def test_same_type_in_params_and_return_is_defined_once():
    defs = {'Order': _object({'address': _ref('Address')}), 'Address': _object({'city': _STR})}
    tool = Tool('sync', _object({'o': _ref('Order')}, **{'$defs': defs}), {'$ref': '#/$defs/Order', '$defs': defs})
    assert _catalog(tool) == snapshot("""\
class Order(TypedDict):
    address: Address

class Address(TypedDict):
    city: str

def sync(*, o: Order) -> Order:
    ...\
""")


def test_types_differing_only_in_a_nested_type_do_not_merge():
    def tool(name: str, user_field: str) -> Tool:
        return Tool(
            name,
            _with_defs({'o': _ref('Order')}, Order=_object({'user': _ref('User')}), User=_object({user_field: _STR})),
        )

    assert _catalog(tool('tool_a', 'name'), tool('tool_b', 'id')) == snapshot("""\
class tool_a_Order(TypedDict):
    user: tool_a_User

class tool_a_User(TypedDict):
    name: str

class tool_b_Order(TypedDict):
    user: tool_b_User

class tool_b_User(TypedDict):
    id: str

def tool_a(*, o: tool_a_Order) -> Any:
    ...

def tool_b(*, o: tool_b_Order) -> Any:
    ...\
""")


def test_union_members_that_merge_become_one_member():
    item = _object({'x': _STR}, title='Item')
    assert 'def t(*, a: Item) -> Any:' in _catalog(Tool('t', _object({'a': {'anyOf': [item, {**item}]}})))


def test_types_differing_only_in_a_required_flag_do_not_merge():
    optional = {'type': 'object', 'properties': {'a': _STR}}
    text = _catalog(
        Tool('tool_a', _with_defs({'x': _ref('T')}, T=_object({'a': _STR}))),
        Tool('tool_b', _with_defs({'x': _ref('T')}, T=optional)),
    )
    assert 'class tool_a_T(TypedDict):\n    a: str' in text
    assert 'class tool_b_T(TypedDict):\n    a: NotRequired[str]' in text


def _holder(field_type: TypeExpr) -> TypeSignature:
    return TypeSignature(name='H', fields={'x': TypeFieldSignature(name='x', type=field_type)})


_STR_EXPR = SimpleTypeExpr('str')


@pytest.mark.parametrize(
    'a,b,equal',
    [
        pytest.param(
            GenericTypeExpr(base='list', args=[_STR_EXPR]),
            GenericTypeExpr(base='list', args=[_STR_EXPR]),
            True,
            id='same-generic',
        ),
        pytest.param(
            GenericTypeExpr(base='list', args=[_STR_EXPR]),
            GenericTypeExpr(base='tuple', args=[_STR_EXPR]),
            False,
            id='generic-base',
        ),
        pytest.param(
            GenericTypeExpr(base='tuple', args=[_STR_EXPR]),
            GenericTypeExpr(base='tuple', args=[_STR_EXPR, _STR_EXPR]),
            False,
            id='generic-args',
        ),
        pytest.param(LiteralTypeExpr(values=[1]), LiteralTypeExpr(values=['1']), False, id='literal-value-type'),
        pytest.param(
            UnionTypeExpr(members=[_STR_EXPR]),
            GenericTypeExpr(base='list', args=[_STR_EXPR]),
            False,
            id='union-vs-generic',
        ),
        pytest.param(TypeSignature(name='Any'), SimpleTypeExpr('Any'), False, id='type-named-like-a-simple-type'),
        pytest.param(TypeSignature(name='User'), TypeSignature(name='Person'), False, id='nested-name'),
        pytest.param(
            UnionTypeExpr(members=[_STR_EXPR]),
            UnionTypeExpr(members=[_STR_EXPR, SimpleTypeExpr('None')]),
            False,
            id='union-members',
        ),
    ],
)
def test_structurally_equal_compares_expressions(a: TypeExpr, b: TypeExpr, equal: bool):
    assert _holder(a).structurally_equal(_holder(b)) is equal


def test_structurally_equal_ignores_own_name_and_description():
    a = TypeSignature(name='A', description='One.', fields={'x': TypeFieldSignature(name='x', type=_STR_EXPR)})
    b = TypeSignature(name='B', description='Two.', fields={'x': TypeFieldSignature(name='x', type=_STR_EXPR)})
    assert a.structurally_equal(b)


def test_structurally_equal_looks_inside_nested_types():
    def order(field_name: str) -> TypeSignature:
        user = TypeSignature(name='User', fields={field_name: TypeFieldSignature(name=field_name, type=_STR_EXPR)})
        return TypeSignature(name='Order', fields={'user': TypeFieldSignature(name='user', type=user)})

    assert order('a').structurally_equal(order('a'))
    assert not order('a').structurally_equal(order('b'))


def test_structurally_equal_needs_the_same_fields_in_both_directions():
    small = TypeSignature(name='H', fields={'a': TypeFieldSignature(name='a', type=_STR_EXPR)})
    large = TypeSignature(name='H', fields={**small.fields, 'b': TypeFieldSignature(name='b', type=_STR_EXPR)})
    assert not small.structurally_equal(large)
    assert not large.structurally_equal(small)


def test_structurally_equal_ends_on_cycles_and_finds_differences_in_them():
    def node(leaf: SimpleTypeExpr) -> TypeSignature:
        result = TypeSignature(name='Node')
        result.fields = {
            'next': TypeFieldSignature(name='next', type=UnionTypeExpr(members=[result, SimpleTypeExpr('None')])),
            'leaf': TypeFieldSignature(name='leaf', type=leaf),
        }
        return result

    assert node(_STR_EXPR).structurally_equal(node(_STR_EXPR))
    assert not node(_STR_EXPR).structurally_equal(node(SimpleTypeExpr('int')))


def test_structurally_equal_finds_a_difference_at_the_end_of_a_long_chain():
    def chain(leaf: str) -> TypeSignature:
        current = TypeSignature(name='T', fields={leaf: TypeFieldSignature(name=leaf, type=_STR_EXPR)})
        for _ in range(100):
            current = TypeSignature(name='T', fields={'x': TypeFieldSignature(name='x', type=current)})
        return current

    assert chain('a').structurally_equal(chain('a'))
    assert not chain('a').structurally_equal(chain('b'))


# =============================================================================
# 3. A `$ref` means the schema it points at, or `Any` if this document can't answer
# =============================================================================


@pytest.mark.parametrize(
    'schema,signature',
    [
        pytest.param(
            _object({'x': {'$ref': '#/definitions/User'}}, definitions={'User': _object({'a': _STR})}),
            'def t(*, x: User) -> Any:',
            id='draft-07-definitions',
        ),
        pytest.param(
            _with_defs({'x': {'$ref': '#/$defs/a~1b'}}, **{'a/b': _INT}), 'def t(*, x: int) -> Any:', id='escaped-slash'
        ),
        pytest.param(
            _with_defs({'x': {'$ref': '#/$defs/a~0b'}}, **{'a~b': _INT}), 'def t(*, x: int) -> Any:', id='escaped-tilde'
        ),
        pytest.param(
            _with_defs({'x': {'$ref': '#/$defs/a%20b'}}, **{'a b': _INT}),
            'def t(*, x: int) -> Any:',
            id='percent-encoded',
        ),
        pytest.param(
            _with_defs({'x': _ref('A')}, A=_ref('B'), B=_INT), 'def t(*, x: int) -> Any:', id='alias-of-scalar'
        ),
        pytest.param(
            _with_defs({'x': _ref('A')}, A=_ref('B'), B=_object({'b': _STR})),
            'def t(*, x: B) -> Any:',
            id='alias-of-object',
        ),
        pytest.param(
            _with_defs({'x': {'$ref': '#/$defs/A/anyOf/1'}}, A={'anyOf': [_STR, _INT]}),
            'def t(*, x: int) -> Any:',
            id='list-index',
        ),
        pytest.param(
            _with_defs({'x': _ref('A')}, A=_ref('B'), B=_ref('A')), 'def t(*, x: Any) -> Any:', id='alias-cycle'
        ),
        pytest.param(_object({'x': _ref('Gone')}), 'def t(*, x: Any) -> Any:', id='missing-def'),
        pytest.param(_object({'x': {'$ref': 'https://example.com/s.json'}}), 'def t(*, x: Any) -> Any:', id='remote'),
        pytest.param(_object({'x': {'$ref': '#anchor'}}), 'def t(*, x: Any) -> Any:', id='anchor'),
        pytest.param(_object({'x': {'$ref': 7}}), 'def t(*, x: Any) -> Any:', id='not-a-string'),
        pytest.param(_object({'x': {'$ref': '#/required'}}), 'def t(*, x: Any) -> Any:', id='not-a-schema'),
    ],
)
def test_ref_forms(schema: dict[str, Any], signature: str):
    assert signature in _catalog(Tool('t', schema))


@pytest.mark.parametrize('token', ['01', '٣', '1' * 5000])
def test_list_index_only_takes_what_json_pointer_allows(token: str):
    schema = _with_defs({'x': {'$ref': f'#/$defs/A/anyOf/{token}'}}, A={'anyOf': [_STR, _STR, _STR, _INT]})
    assert 'def t(*, x: Any) -> Any:' in _catalog(Tool('t', schema))


@pytest.mark.parametrize(
    'node',
    [
        pytest.param({'anyOf': []}, id='any-of'),
        pytest.param({'oneOf': []}, id='one-of'),
        pytest.param({'enum': []}, id='enum'),
    ],
)
def test_empty_schema_lists_are_any(node: dict[str, Any]):
    assert 'def t(*, x: Any) -> Any:' in _catalog(Tool('t', _object({'x': node})))


def test_ref_to_the_document_root_is_a_recursive_type():
    tool = Tool('t', {'type': 'object', 'properties': {'value': _STR, 'next': {'$ref': '#'}}, 'required': ['value']})
    assert _catalog(tool) == snapshot("""\
class T(TypedDict):
    value: str
    next: NotRequired[T]

def t(*, value: str, next: T | None = None) -> Any:
    ...\
""")


def test_ref_to_a_property_shares_its_type():
    text = _catalog(Tool('t', _object({'a': _object({'b': _STR}), 'c': {'$ref': '#/properties/a'}})))
    assert 'def t(*, a: TA, c: TA) -> Any:' in text
    assert text.count('class TA(TypedDict):') == 1


def test_recursive_alias_through_an_object_keeps_its_shape():
    schema = _with_defs(
        {'x': _ref('Tree')}, Tree={'type': 'array', 'items': _object({'kids': _ref('Tree')}, title='Node')}
    )
    assert 'kids: list[Node]' in _catalog(Tool('t', schema))


def test_parameters_can_be_a_root_ref():
    node = {'type': 'object', 'properties': {'value': _STR, 'next': _ref('Node')}, 'required': ['value']}
    schema = {'$ref': '#/$defs/Node', '$defs': {'Node': node}}
    assert 'def t(*, value: str, next: Node | None = None) -> Any:' in _catalog(Tool('t', schema))


def test_parameters_that_are_a_root_ref_cycle_are_empty():
    schema = {'$ref': '#/$defs/A', '$defs': {'A': _ref('B'), 'B': _ref('A')}}
    assert _catalog(Tool('t', schema)) == 'def t() -> Any:\n    ...'


@pytest.mark.parametrize(
    'definitions',
    [
        None,
        [],
        {'note': 'free text'},
        {'Bad': {'type': 'object', 'properties': False}},
        {'Unused': _object({'a': _STR})},
    ],
)
def test_definitions_are_only_read_through_a_ref(definitions: Any):
    """Only `$defs` is read up front, as before, so this key can hold anything until a `$ref` points into it."""
    schema = {**_object({'x': _INT}), 'definitions': definitions}
    assert _render([_signature(Tool('t', schema))]) == 'def t(*, x: int) -> Any:\n    ...'


# =============================================================================
# 4. Names Python can't write stay exact and usable
# =============================================================================


def _mail_tool(mail: dict[str, Any], **other_defs: Any) -> Tool:
    return Tool('send', _with_defs({'m': _ref('Mail')}, Mail=mail, **other_defs))


def test_field_python_cannot_write_uses_the_functional_form():
    mail = {
        'type': 'object',
        'description': 'A mail.\nSecond line.',
        'properties': {
            'from': {**_STR, 'description': 'Sender "name".\nNot the reply-to.'},
            'cc': {'type': 'array', 'items': _ref('Other')},
            'kind': {'enum': ['a']},
            'status': {'anyOf': [_ref('Other'), {'enum': ["it's"]}]},
        },
        'required': ['from', 'kind', 'status'],
    }
    assert _catalog(_mail_tool(mail, Other=_object({'x': _STR}))) == snapshot("""\
# A mail.
# Second line.
Mail = TypedDict('Mail', {
    # Sender "name".
    # Not the reply-to.
    'from': str,
    'cc': NotRequired[list[Other]],
    'kind': Literal['a'],
    'status': Other | Literal["it's"],
})

class Other(TypedDict):
    x: str

def send(*, m: Mail) -> Any:
    ...\
""")


@pytest.mark.parametrize('name', ['from', 'user-id', 'a b', "it's", '__private', '__debug__', '\ufb01le', '', '1a'])
def test_field_python_cannot_write_keeps_its_exact_name(name: str):
    assert f'{name!r}: str' in _catalog(_mail_tool(_object({name: _STR})))


@pytest.mark.parametrize(
    'mail',
    [
        pytest.param(_object({'str': _STR, 'other': _STR}), id='simple-type'),
        pytest.param(_object({'Other': _ref('Other')}), id='own-type'),
        pytest.param(_object({'Other': {'anyOf': [_STR, _ref('Other')]}}), id='union-member'),
        pytest.param(_object({'Other': {'type': 'array', 'items': _ref('Other')}}), id='generic-argument'),
        pytest.param(_object({'list': {'type': 'array', 'items': _STR}}), id='generic-base'),
        pytest.param(_object({'Literal': {'enum': ['a']}}), id='literal-helper'),
        pytest.param(
            {'type': 'object', 'properties': {'NotRequired': _STR, 'b': _STR}, 'required': ['NotRequired']},
            id='not-required-helper',
        ),
    ],
)
def test_field_named_like_a_name_its_class_uses_is_functional(mail: dict[str, Any]):
    """In a class body a type checker reads later annotations as the field, which silently turns off their checks."""
    assert "Mail = TypedDict('Mail', {" in _catalog(_mail_tool(mail, Other=_object({'a': _STR})))


@pytest.mark.parametrize(
    'mail,line',
    [
        pytest.param(_object({'list': _STR}), 'list: str', id='helper-its-class-does-not-use'),
        pytest.param(_object({'User': _STR}), 'User: str', id='type-its-class-does-not-use'),
        pytest.param(_object({'kind': {'enum': ['kind']}}), "kind: Literal['kind']", id='its-own-literal-value'),
    ],
)
def test_field_python_can_write_stays_in_a_class_body(mail: dict[str, Any], line: str):
    assert f'class Mail(TypedDict):\n    {line}' in _catalog(_mail_tool(mail))


def test_null_byte_in_a_description_stays_out_of_the_text():
    mail = {
        'type': 'object',
        'description': 'a\0b',
        'properties': {'from': {**_STR, 'description': 'c\0d'}},
        'required': ['from'],
    }
    text = _catalog(_mail_tool(mail))
    assert '# a\\x00b' in text and '# c\\x00d' in text


_KEYWORD_PARAMS = _object({'from': _STR})


def test_param_python_cannot_write_uses_kwargs():
    schema = {
        'type': 'object',
        'properties': {
            'from': {**_STR, 'description': 'Sender.'},
            'limit': {**_INT, 'default': 5},
            'note': _STR,
            'plain': _STR,
        },
        'required': ['from', 'plain'],
    }
    assert _catalog(Tool('send', schema)) == snapshot('''\
SendArgs = TypedDict('SendArgs', {
    'from': str,
    'plain': str,
    'limit': NotRequired[int],
    'note': NotRequired[str | None],
})

def send(**kwargs: Any) -> Any:
    """
    Some argument names are not valid Python keywords, so pass arguments as `send(**{...})`. `SendArgs` describes them.

    Args:
        from: Sender.
        limit: Defaults to `5`.
        note: Defaults to `None`.
    """
    ...\
''')


def test_args_types_with_one_name_stay_two_types():
    text = _catalog(Tool('send', _object({'from': _STR})), Tool('Send', _object({'from': _INT})))
    assert "send_SendArgs = TypedDict('send_SendArgs', {\n    'from': str,\n})" in text
    assert "Send_SendArgs = TypedDict('Send_SendArgs', {\n    'from': int,\n})" in text
    assert '`Send_SendArgs` describes them.' in text


def test_args_type_merges_with_an_equal_type_of_its_name():
    text = _catalog(
        Tool('send', _KEYWORD_PARAMS),
        Tool('other', _with_defs({'x': _ref('SendArgs')}, SendArgs=_KEYWORD_PARAMS)),
    )
    assert text.count("TypedDict('SendArgs', {") == 1
    assert '`SendArgs` describes them.' in text
    assert 'def other(*, x: SendArgs) -> Any:' in text


# =============================================================================
# 5. Names stay right across the ways callers use the result
# =============================================================================


def test_the_result_is_the_set_of_names_that_changed():
    assert FunctionSignature.get_conflicting_type_names(_user_tools(tool_a='a', tool_b='b')) == {'User'}
    assert FunctionSignature.get_conflicting_type_names(_user_tools(tool_a='a', tool_b='a')) == frozenset()


def test_rendering_part_of_a_catalog_keeps_the_names_of_the_whole():
    """Dynamic workflows resolve every agent they have shown, and render only the one being revealed."""
    signatures = _user_tools(tool_a='a', tool_b='b')
    conflicting = FunctionSignature.get_conflicting_type_names(signatures)
    assert FunctionSignature.render_type_definitions(signatures[1:], conflicting) == [
        'class tool_b_User(TypedDict):\n    b: str'
    ]
    assert (
        signatures[1].render('...', conflicting_type_names=conflicting)
        == 'def tool_b(*, u: tool_b_User) -> Any:\n    ...'
    )


def test_resolving_again_gives_the_names_of_the_new_set():
    """A cached signature can be in several catalogs, and each is rendered right after it is resolved."""
    signatures = _user_tools(tool_a='a', tool_b='b')
    FunctionSignature.get_conflicting_type_names(signatures)
    assert _render(signatures[:1]) == snapshot("""\
class User(TypedDict):
    a: str

def tool_a(*, u: User) -> Any:
    ...\
""")
    assert 'def tool_a(*, u: tool_a_User) -> Any:' in _render(signatures)


def test_resolving_again_is_stable():
    signatures = _user_tools(tool_a='a', tool_b='b', tool_c='a')
    assert _render(signatures) == _render(signatures)


def test_a_result_without_changes_renders_plain_names():
    signatures = _user_tools(tool_a='a', tool_b='b')
    FunctionSignature.get_conflicting_type_names(signatures)
    assert signatures[0].referenced_types[0].render_definition() == 'class User(TypedDict):\n    a: str'
    assert signatures[0].render('...') == 'def tool_a(*, u: User) -> Any:\n    ...'


def test_rendering_a_definition_with_no_owner_keeps_its_name():
    user = TypeSignature(name='User', fields={'a': TypeFieldSignature(name='a', type=SimpleTypeExpr('str'))})
    assert (
        user.render_definition(conflicting_type_names=frozenset({'User'}))
        == 'class User(TypedDict):\n    a: NotRequired[str]'
    )


def test_rendering_under_another_function_name_keeps_type_names():
    signatures = _user_tools(tool_a='a', tool_b='b')
    conflicting = FunctionSignature.get_conflicting_type_names(signatures)
    assert signatures[0].render('...', name='renamed', conflicting_type_names=conflicting) == (
        'def renamed(*, u: tool_a_User) -> Any:\n    ...'
    )


def test_set_operations_on_the_result_keep_names():
    signatures = _user_tools(tool_a='a', tool_b='b')
    conflicting = FunctionSignature.get_conflicting_type_names(signatures)
    expected = FunctionSignature.render_type_definitions(signatures, conflicting)
    assert FunctionSignature.render_type_definitions(signatures, conflicting | {'Other'}) == expected
    assert FunctionSignature.render_type_definitions(signatures, frozenset(conflicting)) == expected


@pytest.mark.parametrize('clone', [copy.deepcopy, _pickled], ids=['deepcopy', 'pickle'])
def test_copied_signatures_keep_names(clone: Callable[[Any], Any]):
    signatures = _user_tools(tool_a='a', tool_b='b')
    conflicting = FunctionSignature.get_conflicting_type_names(signatures)
    expected = FunctionSignature.render_type_definitions(signatures, conflicting)
    assert FunctionSignature.render_type_definitions(clone(signatures), conflicting) == expected


def test_unresolved_signatures_take_the_prefix_of_the_names_passed_in():
    """Callers can pass a set of their own, which was the only form before names were resolved per type."""
    signatures = _user_tools(tool_a='a', tool_b='b')
    conflicting = frozenset({'User'})
    assert FunctionSignature.render_type_definitions(signatures, conflicting) == [
        'class tool_a_User(TypedDict):\n    a: str',
        'class tool_b_User(TypedDict):\n    b: str',
    ]
    assert (
        signatures[1].render('...', conflicting_type_names=conflicting)
        == 'def tool_b(*, u: tool_b_User) -> Any:\n    ...'
    )


def test_args_type_follows_unification_into_a_later_catalog():
    """Two tools share one args type, so a later catalog that renames it has to reach both."""

    def signature(name: str, value: dict[str, Any]) -> FunctionSignature:
        return _signature(Tool(name, _object({'from': value})))

    send_x, sendx = signature('send_x', _STR), signature('Send_x', _STR)
    FunctionSignature.get_conflicting_type_names([send_x, sendx])
    conflicting = FunctionSignature.get_conflicting_type_names([send_x, sendx, signature('SEND_X', _INT)])
    assert '`send_x_SendXArgs` describes them.' in sendx.render('...', conflicting_type_names=conflicting)


# =============================================================================
# 6. Signatures built by hand get the same guarantees
# =============================================================================


def _keyword_signature(**kwargs: Any) -> FunctionSignature:
    return FunctionSignature(
        name='send',
        params={'from': FunctionParam(name='from', type=_STR_EXPR)},
        return_type=SimpleTypeExpr('Any'),
        **kwargs,
    )


def test_built_signature_with_a_keyword_param_uses_kwargs():
    sig = _keyword_signature()
    conflicting = FunctionSignature.get_conflicting_type_names([sig])
    assert FunctionSignature.render_type_definitions([sig], conflicting) == [
        "SendArgs = TypedDict('SendArgs', {\n    'from': str,\n})"
    ]
    assert sig.render('...', conflicting_type_names=conflicting).startswith('def send(**kwargs: Any) -> Any:')


def test_built_signature_defines_types_it_does_not_list():
    address = TypeSignature(name='Address', fields={'city': TypeFieldSignature(name='city', type=_STR_EXPR)})
    user = TypeSignature(name='User', fields={'home': TypeFieldSignature(name='home', type=address)})
    returned = TypeSignature(name='Out', fields={'a': TypeFieldSignature(name='a', type=_STR_EXPR)})
    sig = FunctionSignature(name='t', params={'u': FunctionParam(name='u', type=user)}, return_type=returned)
    conflicting = FunctionSignature.get_conflicting_type_names([sig])
    assert FunctionSignature.render_type_definitions([sig], conflicting) == [
        'class User(TypedDict):\n    home: NotRequired[Address]',
        'class Out(TypedDict):\n    a: NotRequired[str]',
        'class Address(TypedDict):\n    city: NotRequired[str]',
    ]


def test_constructor_leaves_the_list_it_is_given():
    given: list[TypeSignature] = []
    _keyword_signature(referenced_types=given)
    assert given == []


def test_replace_keeps_one_args_definition():
    original = _signature(Tool('send', _KEYWORD_PARAMS))
    replaced = dataclasses.replace(original)
    assert replaced == original
    assert FunctionSignature.render_type_definitions([replaced], frozenset()) == [
        "SendArgs = TypedDict('SendArgs', {\n    'from': str,\n})"
    ]


def test_replace_with_a_new_name_renames_the_args_type():
    renamed = dataclasses.replace(_signature(Tool('send', _KEYWORD_PARAMS)), name='mail')
    assert FunctionSignature.render_type_definitions([renamed], frozenset()) == [
        "MailArgs = TypedDict('MailArgs', {\n    'from': str,\n})"
    ]


def test_serialized_form_gains_no_fields():
    """Resolved names and the args type are derived, so they stay out of what `asdict` and equality see."""
    sig = _signature(Tool('send', _with_defs({'from': _ref('User')}, User=_object({'a': _STR}))))
    FunctionSignature.get_conflicting_type_names([sig])
    assert '_kwargs_type' not in dataclasses.asdict(sig)
    assert '_resolved_name' not in dataclasses.asdict(sig)['referenced_types'][0]


@pytest.mark.parametrize(
    'rebuild',
    [_pickled, TypeAdapter(FunctionSignature).validate_python],
    ids=['pickle', 'type-adapter'],
)
def test_rebuilt_signature_keeps_its_args_type(rebuild: Callable[[Any], FunctionSignature]):
    other = rebuild(_signature(Tool('send', _KEYWORD_PARAMS)))
    conflicting = FunctionSignature.get_conflicting_type_names([other])
    assert FunctionSignature.render_type_definitions([other], conflicting) == [
        "SendArgs = TypedDict('SendArgs', {\n    'from': str,\n})"
    ]
