from __future__ import annotations as _annotations

import re
from collections.abc import Callable
from inspect import Signature, cleandoc, signature
from itertools import takewhile
from typing import TYPE_CHECKING, Any, Literal

from griffe import (
    Docstring,
    DocstringSection,
    DocstringSectionAdmonition,
    DocstringSectionParameters,
    DocstringSectionReturns,
    DocstringSectionText,
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

    main_desc = _main_description(sections, docstring_style)

    if return_ := next((p for p in sections if isinstance(p, DocstringSectionReturns)), None):
        main_desc = _main_desc_with_returns(return_, docstring_style, func, doc, main_desc)

    return main_desc, params


def _main_description(sections: list[DocstringSection], docstring_style: DocstringStyle) -> str:
    """Build the main description from the docstring's text and admonition sections, in source order.

    Text sections and recognized admonition sections (e.g. `Warning:`) are joined where they appear, so prose
    after an admonition is kept too.
    """
    main_desc_parts: list[str] = []
    for section in sections:
        if isinstance(section, DocstringSectionText):
            main_desc_parts.append(section.value)
        elif isinstance(section, DocstringSectionAdmonition):
            main_desc_parts.append(f'{_admonition_header(section, docstring_style)}\n{section.value.contents}')
    return '\n\n'.join(main_desc_parts)


def _admonition_header(section: DocstringSectionAdmonition, docstring_style: DocstringStyle) -> str:
    """The admonition's header as written, e.g. `Warnings:`, `Important note:` or `Note: Mind the gap`."""
    title = section.title or ''
    kind = str(section.value.kind)
    # A NumPy admonition's title is its header. Griffe's Google parser derives the kind from the written type by
    # lowercasing it and hyphenating spaces, and uses the type as the title unless the header line gives one
    # (`Note: Mind the gap`).
    if docstring_style == 'numpy' or title.lower().replace(' ', '-') == kind:
        return f'{title}:'
    return f'{kind.replace("-", " ").capitalize()}: {title}'


def _main_desc_with_returns(
    return_: DocstringSectionReturns,
    docstring_style: DocstringStyle,
    func: Callable[..., Any],
    doc: str,
    main_desc: str,
) -> str:
    """Format the returns section as the `<returns>` XML the model sees, folding in the main description."""
    return_statement = return_.value[0]
    return_desc = return_statement.description
    return_type = return_statement.annotation
    if docstring_style == 'google' and signature(func).return_annotation is not Signature.empty:
        # The Python return annotation is authoritative: keep the whole docstring text as the return
        # description instead of griffe splitting a type off at the first colon.
        return_desc = _google_returns_text(doc, return_desc) or (
            f'{return_type}: {return_desc}' if return_type else return_desc
        )
        return_type = None
    type_tag = f'<type>{return_type}</type>\n' if return_type else ''
    return_xml = f'<returns>\n{type_tag}<description>{return_desc}</description>\n</returns>'
    if main_desc:
        return f'<summary>{main_desc}</summary>\n{return_xml}'
    return return_xml


_GOOGLE_RETURNS_HEADER = re.compile(r'^returns:(\s|$)', re.IGNORECASE)


def _google_returns_text(doc: str, parsed_description: str) -> str | None:
    """The written, dedented text of the Google-style `Returns:` section griffe parsed into `parsed_description`.

    Griffe's parsed entry can't be joined back into the written text: before the first colon, it strips parentheses
    and normalizes the text as a type expression. The section is found the way griffe's Google parser finds it: a
    `Returns:` line outside a code fence, after a blank line, with an indented line below. As a safeguard, griffe's
    description must be the tail of the text; if no section qualifies, this returns `None`.
    """
    # Clean the same text griffe parses, so trailing whitespace can't differ.
    lines = cleandoc(doc.rstrip()).splitlines()
    in_fence = False
    for i, line in enumerate(lines):
        if line.lstrip(' ').startswith('```'):
            in_fence = not in_fence
        elif (
            not in_fence
            and _GOOGLE_RETURNS_HEADER.match(line)
            and (i == 0 or not lines[i - 1].strip())
            and (block := _indented_block(lines[i + 1 :]))
            and block.endswith(parsed_description)
        ):
            return block
    return None


def _indented_block(lines: list[str]) -> str:
    """The dedented block of indented lines directly at the start of `lines`, as griffe's Google parser reads a section."""
    indent = len(lines[0]) - len(lines[0].lstrip()) if lines and lines[0].strip() else 0
    if not indent:
        return ''
    block = takewhile(lambda line: line.startswith(' ' * indent) or not line.strip(), lines)
    return '\n'.join(line[indent:] for line in block).rstrip('\n')


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
