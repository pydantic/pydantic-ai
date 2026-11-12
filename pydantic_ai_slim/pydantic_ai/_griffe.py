from __future__ import annotations as _annotations

import re
from collections.abc import Callable, Iterator
from inspect import cleandoc
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
    has_return_annotation: bool,
) -> tuple[str | None, dict[str, str]]:
    """Extract the function description and parameter descriptions from a function's docstring.

    The function parses the docstring using the specified format (or infers it if 'auto') and extracts
    both the main description and parameter descriptions. Text and recognized admonition sections
    (e.g. `Warning:` and `Note:`) make up the main description in source order; other sections (e.g.
    `Raises` and `Examples`) are dropped. When `has_return_annotation` says the function has a Python
    return annotation, the whole text of a Google-style returns section is the return description;
    otherwise any text before its first colon is the return type. If a returns section is present in
    the docstring, the main description is formatted as XML.

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
    warnings_headers: list[str] = []
    parsed_doc = doc
    if docstring_style == 'google':
        parsed_doc, warnings_headers = _google_warnings_as_admonitions(doc)
    docstring = Docstring(
        parsed_doc,
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

    main_desc = _main_description(sections, docstring_style, iter(warnings_headers))

    if return_ := next((p for p in sections if isinstance(p, DocstringSectionReturns)), None):
        main_desc = _main_desc_with_returns(return_, docstring_style, has_return_annotation, doc, main_desc)

    return main_desc, params


def _main_description(
    sections: list[DocstringSection], docstring_style: DocstringStyle, warnings_headers: Iterator[str]
) -> str:
    """Build the main description from the docstring's text and admonition sections, in source order.

    Text sections and recognized admonition sections (e.g. `Warning:`) are joined where they appear, so prose
    after an admonition is kept too. `warnings_headers` yields the written header of each Google `Warnings:`
    section, which was parsed as an admonition (see `_google_warnings_as_admonitions`), in source order.
    """
    main_desc_parts: list[str] = []
    for section in sections:
        if isinstance(section, DocstringSectionText):
            main_desc_parts.append(section.value)
        elif isinstance(section, DocstringSectionAdmonition):
            if section.value.kind == _WARNINGS_ADMONITION_KIND:
                header = next(warnings_headers)
            else:
                header = _admonition_header(section, docstring_style)
            main_desc_parts.append(f'{header}\n{section.value.contents}')
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
    has_return_annotation: bool,
    doc: str,
    main_desc: str,
) -> str:
    """Format the returns section as the `<returns>` XML the model sees, folding in the main description."""
    return_statement = return_.value[0]
    return_desc = return_statement.description
    return_type = return_statement.annotation
    if docstring_style == 'google' and has_return_annotation:
        # The Python return annotation is authoritative: keep the whole docstring text as the return
        # description instead of griffe splitting a type off at the first colon. Griffe's description is the tail of
        # the written text, which picks out the right section; if none matches, join griffe's parts back up.
        written = next(
            (text for _, _, text in _google_section_blocks(doc, _GOOGLE_RETURNS_HEADER) if text.endswith(return_desc)),
            None,
        )
        return_desc = written or (f'{return_type}: {return_desc}' if return_type else return_desc)
        return_type = None
    type_tag = f'<type>{return_type}</type>\n' if return_type else ''
    return_xml = f'<returns>\n{type_tag}<description>{return_desc}</description>\n</returns>'
    if main_desc:
        return f'<summary>{main_desc}</summary>\n{return_xml}'
    return return_xml


# The header of a Google section or admonition, as griffe's Google parser matches it.
_GOOGLE_SECTION_HEADER = re.compile(r'^[\w][\s\w-]*:(\s+[^\s].*)?\s*$')
_GOOGLE_RETURNS_HEADER = re.compile(r'^returns:(\s|$)', re.IGNORECASE)
_GOOGLE_WARNINGS_HEADER = re.compile(r'^warnings:(\s|$)', re.IGNORECASE)
_WARNINGS_ADMONITION_KIND = 'pydantic-ai-warnings'


def _google_warnings_as_admonitions(doc: str) -> tuple[str, list[str]]:
    """Rewrite each Google `Warnings:` section header so griffe parses the section as an admonition.

    Griffe parses `Warnings:` like `Warns:`, as a list of warning classes, and drops any line without a colon. Authors
    write it as prose for the reader, like a `Warning:` admonition, so it should reach the model the same way.
    Returns the rewritten docstring and the written headers, in source order.
    """
    blocks = _google_section_blocks(doc, _GOOGLE_WARNINGS_HEADER)
    if not blocks:
        return doc, []
    lines = cleandoc(doc.rstrip()).splitlines()
    for i, _, _ in blocks:
        lines[i] = f'{_WARNINGS_ADMONITION_KIND}:'
    # The lines are already cleaned; the leading newline keeps griffe's `cleandoc` from dedenting them again, which
    # it would if the first line were a section header.
    return '\n' + '\n'.join(lines), [header.rstrip() for _, header, _ in blocks]


def _google_section_blocks(doc: str, header: re.Pattern[str]) -> list[tuple[int, str, str]]:
    """Find the Google-style sections whose header matches `header`.

    Each is returned as a `(line index, written header, dedented text)` tuple, with the line index into the cleaned
    docstring. Griffe's parsed entries can't always be joined back into the written text: e.g. before the first colon
    of a returns entry, it strips parentheses and normalizes the text as a type expression. Sections are found the way
    griffe's Google parser finds them: a header line outside a code fence, after a blank line, with an indented line
    below. A section's indented block is skipped as a whole, so a code fence inside it doesn't carry past it.
    """
    # Clean the same text griffe parses, so trailing whitespace can't differ.
    lines = cleandoc(doc.rstrip()).splitlines()
    blocks: list[tuple[int, str, str]] = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.lstrip(' ').startswith('```'):
            in_fence = not in_fence
        elif (
            not in_fence
            and _GOOGLE_SECTION_HEADER.match(line)
            and (i == 1 or not lines[i - 2].strip())
            and (block := _indented_block(lines[i:]))
        ):
            if header.match(line):
                blocks.append((i - 1, line, '\n'.join(block).rstrip('\n')))
            i += len(block)
    return blocks


def _indented_block(lines: list[str]) -> list[str]:
    """The dedented lines of the indented block directly at the start of `lines`, as griffe's Google parser reads it."""
    indent = len(lines[0]) - len(lines[0].lstrip()) if lines and lines[0].strip() else 0
    if not indent:
        return []
    return [line[indent:] for line in takewhile(lambda line: line.startswith(' ' * indent) or not line.strip(), lines)]


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
