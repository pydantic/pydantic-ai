"""Every docstring section griffe parses, in each docstring style, reaches the tool definition as its kind says.

A tool's description and parameter descriptions come from its function's docstring, through griffe. Each section kind
griffe emits gets one treatment, in `TREATMENTS`, and each style has a sample docstring per kind, in `SAMPLES`, or
`None` where the style has no such section. A sample carries `MARK` in its section's text, so where `MARK` ends up
shows what happened to the section. A kind a new griffe release adds fails
`test_every_section_kind_has_a_treatment_and_a_sample_per_style` until it is classified here.

These build tools from functions rather than run agents: the claim is about the tool definition a docstring produces,
which no model request changes.
"""

from __future__ import annotations

from typing import Literal

import pytest
from griffe import Docstring, DocstringSectionKind

from pydantic_ai import Tool

Style = Literal['google', 'numpy', 'sphinx']
Treatment = Literal['description', 'parameter', 'returns', 'dropped']

STYLES: tuple[Style, ...] = ('google', 'numpy', 'sphinx')

TREATMENTS: dict[str, Treatment] = {
    'text': 'description',
    'admonition': 'description',
    'parameters': 'parameter',
    'other parameters': 'parameter',
    'returns': 'returns',
    'type parameters': 'dropped',
    'raises': 'dropped',
    'warns': 'dropped',
    'yields': 'dropped',
    'receives': 'dropped',
    'examples': 'dropped',
    'attributes': 'dropped',
    'functions': 'dropped',
    'classes': 'dropped',
    'type aliases': 'dropped',
    'modules': 'dropped',
    'deprecated': 'dropped',
}
"""What each kind of section becomes: the tool's description, a parameter's description, its `<returns>`, or nothing.

Keyed by `DocstringSectionKind` value, so an entry for a kind an older griffe lacks doesn't break the import."""

_GOOGLE: dict[str, str | None] = {
    'text': 'Summary MARK.',
    'admonition': 'Summary.\n\nNote:\n    MARK noted.',
    'parameters': 'Summary.\n\nArgs:\n    a: MARK first.',
    'other parameters': 'Summary.\n\nKeyword Args:\n    b: MARK second.',
    'returns': 'Summary.\n\nReturns:\n    MARK returned.',
    'type parameters': 'Summary.\n\nType Parameters:\n    T: MARK type.',
    'raises': 'Summary.\n\nRaises:\n    ValueError: MARK raised.',
    'warns': 'Summary.\n\nWarns:\n    UserWarning: MARK warned.',
    'yields': 'Summary.\n\nYields:\n    int: MARK yielded.',
    'receives': 'Summary.\n\nReceives:\n    int: MARK received.',
    'examples': 'Summary.\n\nExamples:\n    MARK example.\n\n    >>> f(1)\n    1',
    'attributes': 'Summary.\n\nAttributes:\n    x: MARK attribute.',
    'functions': 'Summary.\n\nFunctions:\n    g: MARK function.',
    'classes': 'Summary.\n\nClasses:\n    C: MARK class.',
    'type aliases': 'Summary.\n\nType Aliases:\n    A: MARK alias.',
    'modules': 'Summary.\n\nModules:\n    m: MARK module.',
    # Griffe's Google parser reads a `Deprecated:` header as an admonition.
    'deprecated': None,
}

_NUMPY: dict[str, str | None] = {
    'text': 'Summary MARK.',
    'admonition': 'Summary.\n\nNotes\n-----\nMARK noted.',
    'parameters': 'Summary.\n\nParameters\n----------\na : int\n    MARK first.',
    'other parameters': 'Summary.\n\nOther Parameters\n----------------\nb : int\n    MARK second.',
    'returns': 'Summary.\n\nReturns\n-------\nint\n    MARK returned.',
    'type parameters': 'Summary.\n\nType Parameters\n---------------\nT\n    MARK type.',
    'raises': 'Summary.\n\nRaises\n------\nValueError\n    MARK raised.',
    'warns': 'Summary.\n\nWarns\n-----\nUserWarning\n    MARK warned.',
    'yields': 'Summary.\n\nYields\n------\nint\n    MARK yielded.',
    'receives': 'Summary.\n\nReceives\n--------\nint\n    MARK received.',
    'examples': 'Summary.\n\nExamples\n--------\nMARK example.\n\n>>> f(1)\n1',
    'attributes': 'Summary.\n\nAttributes\n----------\nx : int\n    MARK attribute.',
    'functions': 'Summary.\n\nFunctions\n---------\ng\n    MARK function.',
    'classes': 'Summary.\n\nClasses\n-------\nC\n    MARK class.',
    'type aliases': 'Summary.\n\nType Aliases\n------------\nA\n    MARK alias.',
    'modules': 'Summary.\n\nModules\n-------\nm\n    MARK module.',
    'deprecated': 'Summary.\n\nDeprecated\n----------\n1.0\n    MARK deprecated.',
}

# Griffe's Sphinx parser emits only text, parameters, attributes, returns and raises sections.
_SPHINX: dict[str, str | None] = {
    'text': 'Summary MARK.',
    'parameters': 'Summary.\n\n:param a: MARK first.',
    'returns': 'Summary.\n\n:returns: MARK returned.',
    'raises': 'Summary.\n\n:raises ValueError: MARK raised.',
    'attributes': 'Summary.\n\n:var x: MARK attribute.',
    **dict.fromkeys(
        [
            'admonition',
            'other parameters',
            'type parameters',
            'warns',
            'yields',
            'receives',
            'examples',
            'functions',
            'classes',
            'type aliases',
            'modules',
            'deprecated',
        ]
    ),
}

SAMPLES: dict[Style, dict[str, str | None]] = {'google': _GOOGLE, 'numpy': _NUMPY, 'sphinx': _SPHINX}
"""Per style, a docstring whose section of each kind carries `MARK`, or `None` when the style has no such section."""

_KINDS = [kind.value for kind in DocstringSectionKind]


def test_every_section_kind_has_a_treatment_and_a_sample_per_style():
    for kind in _KINDS:
        assert kind in TREATMENTS, f'classify the {kind!r} docstring section in `TREATMENTS`'
        for style in STYLES:
            assert kind in SAMPLES[style], f'give the {kind!r} section a {style} sample, or `None`, in `SAMPLES`'


def _tool(docstring: str, style: Style) -> Tool[None]:
    def tool(a: int, b: int = 0) -> int:
        return a + b

    tool.__doc__ = docstring
    return Tool(tool, docstring_format=style)


@pytest.mark.parametrize(
    ('style', 'kind', 'docstring'),
    [
        pytest.param(style, kind, docstring, id=f'{style}-{kind}')
        for style in STYLES
        for kind, docstring in SAMPLES[style].items()
        if docstring is not None and kind in _KINDS
    ],
)
def test_section_reaches_the_tool_definition_as_its_kind_says(style: Style, kind: str, docstring: str):
    sections = Docstring(docstring, lineno=1, parser=style).parse(warnings=False)
    assert kind in [section.kind.value for section in sections]

    tool_def = _tool(docstring, style).tool_def
    description = tool_def.description or ''
    parameters: dict[str, dict[str, str]] = tool_def.parameters_json_schema['properties']
    marked_parameters = [name for name, schema in parameters.items() if 'MARK' in schema.get('description', '')]
    match TREATMENTS[kind]:
        case 'description':
            assert ('MARK' in description, marked_parameters) == (True, [])
        case 'parameter':
            assert ('MARK' in description, len(marked_parameters)) == (False, 1)
        case 'returns':
            assert '<returns>' in description and 'MARK' in description
        case 'dropped':
            assert ('MARK' in description, marked_parameters) == (False, [])
