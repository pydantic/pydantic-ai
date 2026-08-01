from __future__ import annotations as _annotations

import logging
import re
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from inspect import Signature
from typing import TYPE_CHECKING, Any, Literal, cast

from griffe import Docstring, DocstringSectionKind, GoogleOptions, Object as GriffeObject

if TYPE_CHECKING:
    from .tools import DocstringFormat

DocstringStyle = Literal['google', 'numpy', 'sphinx']


@dataclass
class DocstringDescriptions:
    """Structured descriptions parsed from a function docstring."""

    tool: str | None
    parameters: dict[str, str]
    summary: str | None
    returns: str | None


def doc_descriptions(
    func: Callable[..., Any],
    sig: Signature,
    *,
    docstring_format: DocstringFormat,
) -> DocstringDescriptions:
    """Extract the function description and parameter descriptions from a function's docstring.

    The function parses the docstring using the specified format (or infers it if 'auto')
    and extracts the provider-facing tool description alongside the structured pieces used by
    rendered Python signatures. If a returns section is present, the provider-facing description
    retains the existing XML format.
    """
    doc = func.__doc__
    if doc is None:
        return DocstringDescriptions(tool=None, parameters={}, summary=None, returns=None)

    # see https://github.com/mkdocstrings/griffe/issues/293
    parent = cast(GriffeObject, sig)

    docstring_style = _infer_docstring_style(doc) if docstring_format == 'auto' else docstring_format
    # These options are only valid for Google-style docstrings
    # https://mkdocstrings.github.io/griffe/reference/docstrings/#google-options
    parser_options = (
        GoogleOptions(returns_named_value=False, returns_multiple_items=False) if docstring_style == 'google' else None
    )
    docstring = Docstring(
        doc,
        lineno=1,
        parser=docstring_style,
        parent=parent,
        parser_options=parser_options,
    )
    with _disable_griffe_logging():
        sections = docstring.parse()

    params = {}
    if parameters := next((p for p in sections if p.kind == DocstringSectionKind.parameters), None):
        params = {p.name: p.description for p in parameters.value}

    main_desc = ''
    if main := next((p for p in sections if p.kind == DocstringSectionKind.text), None):
        main_desc = main.value

    summary = main_desc or None
    return_desc = None
    if return_ := next((p for p in sections if p.kind == DocstringSectionKind.returns), None):
        return_statement = return_.value[0]
        return_desc = return_statement.description
        return_type = return_statement.annotation
        type_tag = f'<type>{return_type}</type>\n' if return_type else ''
        return_xml = f'<returns>\n{type_tag}<description>{return_desc}</description>\n</returns>'

        if main_desc:
            main_desc = f'<summary>{main_desc}</summary>\n{return_xml}'
        else:
            main_desc = return_xml

    return DocstringDescriptions(tool=main_desc, parameters=params, summary=summary, returns=return_desc)


def _infer_docstring_style(doc: str) -> DocstringStyle:
    """Simplistic docstring style inference."""
    for pattern, replacements, style in _docstring_style_patterns:
        matches = (
            re.search(pattern.format(replacement), doc, re.IGNORECASE | re.MULTILINE) for replacement in replacements
        )
        if any(matches):
            return style
    # fallback to google style
    return 'google'


# See https://github.com/mkdocstrings/griffe/issues/329#issuecomment-2425017804
_docstring_style_patterns: list[tuple[str, list[str], DocstringStyle]] = [
    (
        r'\n[ \t]*:{0}([ \t]+\w+)*:([ \t]+.+)?\n',
        [
            'param',
            'parameter',
            'arg',
            'argument',
            'key',
            'keyword',
            'type',
            'var',
            'ivar',
            'cvar',
            'vartype',
            'returns',
            'return',
            'rtype',
            'raises',
            'raise',
            'except',
            'exception',
        ],
        'sphinx',
    ),
    (
        r'\n[ \t]*{0}:([ \t]+.+)?\n[ \t]+.+',
        [
            'args',
            'arguments',
            'params',
            'parameters',
            'keyword args',
            'keyword arguments',
            'other args',
            'other arguments',
            'other params',
            'other parameters',
            'raises',
            'exceptions',
            'returns',
            'yields',
            'receives',
            'examples',
            'attributes',
            'functions',
            'methods',
            'classes',
            'modules',
            'warns',
            'warnings',
        ],
        'google',
    ),
    (
        r'\n[ \t]*{0}\n[ \t]*---+\n',
        [
            'deprecated',
            'parameters',
            'other parameters',
            'returns',
            'yields',
            'receives',
            'raises',
            'warns',
            'attributes',
            'functions',
            'methods',
            'classes',
            'modules',
        ],
        'numpy',
    ),
]


@contextmanager
def _disable_griffe_logging():
    # Hacky, but suggested here: https://github.com/mkdocstrings/griffe/issues/293#issuecomment-2167668117
    old_level = logging.root.getEffectiveLevel()
    logging.root.setLevel(logging.ERROR)
    yield
    logging.root.setLevel(old_level)
