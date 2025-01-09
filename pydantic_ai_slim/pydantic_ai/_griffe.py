from __future__ import annotations as _annotations

import logging
import re
from contextlib import contextmanager
from inspect import Signature
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

from griffe import Docstring, DocstringSectionKind, Object as GriffeObject

if TYPE_CHECKING:
    from .tools import DocstringFormat

DocstringStyle = Literal['google', 'numpy', 'sphinx']


def doc_descriptions(
    func: Callable[..., Any],
    sig: Signature,
    *,
    docstring_format: DocstringFormat,
) -> tuple[str, dict[str, str]]:
    """Extract the function description and parameter descriptions from a function's docstring.

    Returns:
        A tuple of (main function description, parameter descriptions).
    """
    doc = func.__doc__
    if doc is None:
        return '', {}

    # see https://github.com/mkdocstrings/griffe/issues/293
    parent = cast(GriffeObject, sig)

    docstring_style = _infer_docstring_style(doc) if docstring_format == 'auto' else docstring_format
    docstring = Docstring(doc, lineno=1, parser=docstring_style, parent=parent)
    with _disable_griffe_logging():
        sections = docstring.parse()

    params = {}
    if parameters := next((p for p in sections if p.kind == DocstringSectionKind.parameters), None):
        params = {p.name: p.description for p in parameters.value}

    main_desc = ''
    if main := next((p for p in sections if p.kind == DocstringSectionKind.text), None):
        main_desc = main.value

    return main_desc, params


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
