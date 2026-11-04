"""A hand-built input that violates each code in `function_signature_invariants.py`.

A checker no committed input can fail checks nothing. Each catalog is written by hand, so these hold whatever the
renderer does.
"""

from __future__ import annotations as _annotations

import pytest

from .function_signature_invariants import Tool, catalog_violations, stability_violations

_STRING = {'type': 'string'}
_USER = {'type': 'object', 'properties': {'a': _STRING}, 'required': ['a']}


def _params(**properties: object) -> dict[str, object]:
    return {'type': 'object', 'properties': properties, 'required': list(properties)}


@pytest.mark.parametrize(
    'text,tools,code',
    [
        pytest.param(
            'def t(*, x: str -> Any:\n    ...', [Tool('t', _params(x=_STRING))], 'catalog.syntax_error', id='syntax'
        ),
        pytest.param(
            'class U(TypedDict):\n    a: str\n\nclass U(TypedDict):\n    b: str',
            [],
            'catalog.duplicate_name',
            id='duplicate',
        ),
        pytest.param('class Any(TypedDict):\n    a: str', [], 'catalog.shadowed_helper', id='shadowed-helper'),
        pytest.param('class asyncio(TypedDict):\n    a: str', [], 'catalog.shadowed_helper', id='shadowed-asyncio'),
        pytest.param('def t(*, __debug__: str) -> Any:\n    ...', [], 'catalog.syntax_error', id='compile-only-error'),
        pytest.param(
            'def t(*, x: User) -> Any:\n    ...',
            [],
            'catalog.unresolved_name',
            id='unresolved',
        ),
        pytest.param(
            "U = TypedDict('U', {'a': 'Missing'})",
            [],
            'catalog.unresolved_name',
            id='unresolved-quoted',
        ),
        pytest.param(
            'class U(TypedDict):\n    str: str\n    b: str', [], 'catalog.field_shadows_type', id='field-shadow'
        ),
        pytest.param('class U(TypedDict):\n    __a: int', [], 'catalog.unsafe_field_name', id='mangled'),
        pytest.param('', [Tool('t', _params())], 'tool.missing', id='missing'),
        pytest.param(
            'def t(*, y: str) -> Any:\n    ...', [Tool('t', _params(x=_STRING))], 'tool.wrong_params', id='param-name'
        ),
        pytest.param(
            'def t(*, x: str = None) -> Any:\n    ...',
            [Tool('t', _params(x=_STRING))],
            'tool.wrong_params',
            id='param-required',
        ),
        pytest.param(
            'def t(*, x: int) -> Any:\n    ...',
            [Tool('t', _params(x=_STRING))],
            'tool.wrong_param_shape',
            id='param-shape',
        ),
        pytest.param(
            'class U(TypedDict):\n    b: str\n\ndef t(*, x: U) -> Any:\n    ...',
            [Tool('t', _params(x=_USER))],
            'tool.wrong_param_shape',
            id='nested-shape',
        ),
        pytest.param(
            'def t() -> int:\n    ...',
            [Tool('t', _params(), _STRING)],
            'tool.wrong_return_shape',
            id='return-shape',
        ),
    ],
)
def test_catalog_violation(text: str, tools: list[Tool], code: str):
    assert catalog_violations(text, tools) == [code]


def test_not_stable():
    assert stability_violations('a', 'b') == ['catalog.not_stable']


@pytest.mark.parametrize(
    'text,tools',
    [
        pytest.param(
            'class U(TypedDict):\n    a: str\n\ndef t(*, x: U) -> Any:\n    ...',
            [Tool('t', _params(x=_USER))],
            id='class',
        ),
        pytest.param(
            "U = TypedDict('U', {'a': str})\n\ndef t(*, x: 'U') -> Any:\n    ...",
            [Tool('t', _params(x=_USER))],
            id='functional',
        ),
        pytest.param(
            'TArgs = TypedDict(\'TArgs\', {\'from\': str})\n\ndef t(**kwargs: Any) -> Any:\n    """`TArgs` describes them."""',
            [Tool('t', {'type': 'object', 'properties': {'from': _STRING}, 'required': ['from']})],
            id='kwargs',
        ),
        pytest.param(
            'def t(*, x: Any) -> Any:\n    ...',
            [Tool('t', _params(x={'$ref': '#/nowhere'}))],
            id='unresolved-ref-is-any',
        ),
    ],
)
def test_correct_catalog_has_no_violations(text: str, tools: list[Tool]):
    assert catalog_violations(text, tools) == []
