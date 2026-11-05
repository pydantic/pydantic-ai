from __future__ import annotations as _annotations

import re
from collections.abc import Callable
from inspect import Signature, signature
from typing import TYPE_CHECKING, Any, Literal

from griffe import (
    Docstring,
    DocstringSection,
    DocstringSectionKind,
    DocstringSectionParameters,
    GoogleOptions,
    NumpyOptions,
    SphinxOptions,
)

if TYPE_CHECKING:
    from .tools import DocstringFormat

DocstringStyle = Literal['google', 'numpy', 'sphinx']


def doc_descriptions(
    func: Callable[..., Any],
    *,
    docstring_format: DocstringFormat,
) -> tuple[str | None, dict[str, str]]:
    """Extract the function description and parameter descriptions from a function's docstring.

    The function parses the docstring using the specified format (or infers it if 'auto')
    and extracts both the main description and parameter descriptions. Recognized admonition
    sections (e.g. `Warning:` and `Note:`) are included in the main description in source order,
    while other sections (e.g. `Raises` and `Examples`) keep their current handling. For a function
    with a Python return annotation, a Google-style returns section keeps its whole text as the
    return description instead of splitting a type off at the first colon. If a returns section
    is present in the docstring, the main description will be formatted as XML.

    Returns:
        A tuple containing:
        - str: Main description string, which may be either:
            * Plain text if no returns section is present
            * XML-formatted if returns section exists, including <summary> and <returns> tags
        - dict[str, str]: Dictionary mapping parameter names to their descriptions
    """
    doc = func.__doc__
    if doc is None:
        return None, {}

    docstring_style = _infer_docstring_style(doc) if docstring_format == 'auto' else docstring_format
    docstring = Docstring(
        doc,
        lineno=1,
        parser=docstring_style,
        parser_options=_parser_options(docstring_style),
    )
    sections = docstring.parse()

    # Google `Keyword Args:` and `Other Parameters:` sections, and numpy `Other Parameters`, parse to a subclass of
    # the parameters section (see https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#docstring-sections).
    # A parameter described in more than one section keeps its first description.
    params: dict[str, str] = {}
    for section in sections:
        if isinstance(section, DocstringSectionParameters):
            for param in section.value:
                params.setdefault(param.name.lstrip('*'), param.description)

    main_desc = _main_description(sections)

    if return_ := next((p for p in sections if p.kind == DocstringSectionKind.returns), None):
        main_desc = _main_desc_with_returns(return_, docstring_style, func, main_desc)

    return main_desc, params


def _main_description(sections: list[DocstringSection]) -> str:
    """Build the main description from the docstring's text and admonition sections, in source order.

    The first text section supplies the description; recognized admonition sections (e.g. `Warning:`) are
    appended where they appear.
    """
    main_desc_parts: list[str] = []
    text_seen = False
    for section in sections:
        if section.kind == DocstringSectionKind.text and not text_seen:
            main_desc_parts.append(section.value)
            text_seen = True
        elif section.kind == DocstringSectionKind.admonition:
            # Recognized admonition sections (e.g. `Warning:`) reach the model verbatim, in source order;
            # other sections keep their current handling.
            main_desc_parts.append(f'{section.value.kind.capitalize()}:\n{section.value.contents}')
    return '\n\n'.join(main_desc_parts)


def _main_desc_with_returns(
    return_: DocstringSection,
    docstring_style: DocstringStyle,
    func: Callable[..., Any],
    main_desc: str,
) -> str:
    """Format the returns section as the `<returns>` XML the model sees, folding in the main description."""
    return_statement = return_.value[0]
    return_desc = return_statement.description
    return_type = return_statement.annotation
    if docstring_style == 'google' and return_type and signature(func).return_annotation is not Signature.empty:
        # The Python return annotation is authoritative: keep the whole docstring text as the return
        # description instead of griffe splitting a type off at the first colon.
        return_desc = f'{return_type}: {return_desc}'
        return_type = None
    type_tag = f'<type>{return_type}</type>\n' if return_type else ''
    return_xml = f'<returns>\n{type_tag}<description>{return_desc}</description>\n</returns>'
    if main_desc:
        return f'<summary>{main_desc}</summary>\n{return_xml}'
    return return_xml


def _parser_options(docstring_style: DocstringStyle) -> GoogleOptions | NumpyOptions | SphinxOptions:
    """The griffe parser options for `docstring_style`."""
    # `warnings=False` stops griffe logging about docstrings it can't fully parse; that is the author's
    # business, not something to surface every time a tool is built.
    match docstring_style:
        case 'google':
            return GoogleOptions(returns_named_value=False, returns_multiple_items=False, warnings=False)
        case 'numpy':
            return NumpyOptions(warnings=False)
        case _:
            return SphinxOptions(warnings=False)


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
