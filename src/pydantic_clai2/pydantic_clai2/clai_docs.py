"""CLAI's own docs, shipped with the package and looked up by the agent on demand."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from functools import cache
from importlib.resources import files
from pathlib import Path, PurePosixPath

from pydantic_ai import ModelRetry, Tool
from pydantic_ai.capabilities import Capability

_CLAI_DOCS: dict[str, tuple[str, str]] = {
    'readme': ('README.md', 'The product tour: what CLAI does and how to start.'),
    'reference': (
        'docs/reference.md',
        'The user reference: commands, options, settings, models and accounts, the prompt, output, '
        'the agent, sessions, and using CLAI from code.',
    ),
    'plugins': (
        'PLUGINS.md',
        'The plugin contract: MCP servers, credentials, the built-in plugins and integrations, and '
        'writing, installing, and testing plugins.',
    ),
    'customization': (
        'pydantic_clai2/customization.md',
        'How to customize CLAI: plugins, hooks, tools, settings, rendering, TUI menus, and models and '
        'providers, and what needs a source change.',
    ),
    'declarative-plugins': ('docs/declarative-plugins.md', 'Why the `Plugin` API declares instead of registering.'),
}
"""CLAI's docs by id: the path under the project root and what the doc covers.

Docs outside the package are copied into `pydantic_clai2/_docs/` by the wheel's `force-include`.
"""

_HARNESS_DOCS = (
    'ask_user',
    'code_mode',
    'coder',
    'compaction',
    'day_ai',
    'filesystem',
    'github',
    'google_workspace',
    'grain',
    'guardrails',
    'linear',
    'logfire_mcp',
    'notion',
    'ordinal',
    'posthog',
    'pylon',
    'repo_context',
    'shell',
    'skills',
    'slack',
    'step_persistence',
    'subagents',
)
"""The harness capabilities CLAI uses, whose READMEs ship in the `pydantic-ai-harness` package."""

_DOC_IDS = (*_CLAI_DOCS, *(f'harness/{name}' for name in _HARNESS_DOCS))

_READ_LIMIT = 50_000
"""Characters `read_clai_docs` returns before it answers with an outline instead."""

_SEARCH_RESULTS = 10
_SNIPPET_WIDTH = 160
_STOP_WORDS = frozenset(
    {'and', 'are', 'can', 'does', 'for', 'how', 'the', 'this', 'that', 'what', 'with', 'you', 'your'}
)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FENCE = re.compile(r'\s*(`{3,}|~{3,})')
_HEADING = re.compile(r'(#{1,6})\s+(.+?)(?:\s+#+)?\s*$')
_SLUG_DROP = re.compile(r'[^\w\- ]')
_WORD = re.compile(r'\w+')

CAPABILITY_ID = 'clai_customization'
_DESCRIPTION = (
    'Use when asked to customize CLAI itself (plugins, CLI UX/UI, commands, rendering, TUI menus, '
    'models or providers), or when asked about CLAI commands, settings, sessions, or its harness '
    'capabilities. Load to search or read the installed docs; do not assume plugin APIs exist.'
)

_HINT = (
    'For questions about CLAI itself (its commands, settings, models, sessions, plugins, or the harness '
    'capabilities it runs on), look in its docs with search_clai_docs and read_clai_docs. '
    'When asked to customize CLAI itself (plugins, CLI UX/UI, commands, rendering, TUI menus, models or '
    "providers), first call read_clai_docs with the path 'customization'. It documents supported APIs and "
    'boundaries. Do not assume plugin APIs exist.'
)


@dataclass(frozen=True, kw_only=True)
class _Section:
    path: str
    title: str
    level: int
    start: int
    """Index of the heading line."""
    body_end: int
    """End of the section's own text, where the next heading starts."""
    end: int
    """End of the section with its subsections."""


@dataclass(frozen=True, kw_only=True)
class _Doc:
    id: str
    description: str
    lines: list[str]
    sections: list[_Section]

    def text(self, start: int = 0, end: int | None = None) -> str:
        return ''.join(self.lines[start:end])


def _read(doc_id: str) -> str:
    if doc_id in _CLAI_DOCS:
        source = _CLAI_DOCS[doc_id][0]
        package = files('pydantic_clai2')
        relative = PurePosixPath(source)
        if relative.parts[0] == 'pydantic_clai2':
            return package.joinpath(*relative.parts[1:]).read_text(encoding='utf-8')
        packaged = package.joinpath('_docs').joinpath(relative.name)
        if packaged.is_file():
            return packaged.read_text(encoding='utf-8')
        # An editable install runs from the checkout, where `force-include` has not copied anything.
        return (_PROJECT_ROOT / source).read_text(encoding='utf-8')
    name = doc_id.removeprefix('harness/')
    return files('pydantic_ai_harness').joinpath(name).joinpath('README.md').read_text(encoding='utf-8')


@cache
def _doc(doc_id: str) -> _Doc:
    text = _read(doc_id)
    lines = text.splitlines(keepends=True)
    headings: list[tuple[int, int, str]] = []
    fence = ''
    for index, line in enumerate(lines):
        if match := _FENCE.match(line):
            marker = match.group(1)
            if not fence:
                fence = marker
            elif marker.startswith(fence):
                fence = ''
        elif not fence and (match := _HEADING.match(line)):
            headings.append((index, len(match.group(1)), match.group(2)))
    sections: list[_Section] = []
    seen: dict[str, int] = {}
    for position, (start, level, title) in enumerate(headings):
        later = headings[position + 1 :]
        end = next((index for index, other, _ in later if other <= level), len(lines))
        # The anchor GitHub gives a heading, so a doc's own `#links` are valid paths.
        slug = _SLUG_DROP.sub('', title.lower()).replace(' ', '-')
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        sections.append(
            _Section(
                path=f'{doc_id}#{slug}-{count}' if count else f'{doc_id}#{slug}',
                title=title,
                level=level,
                start=start,
                body_end=later[0][0] if later else len(lines),
                end=end,
            )
        )
    if doc_id in _CLAI_DOCS:
        description = _CLAI_DOCS[doc_id][1]
    else:
        # A harness README's first sentence, past its title, notes, and links.
        prose = next(block for block in text.split('\n\n') if not block.lstrip().startswith(('#', '>', '[', '!', '|')))
        description = ' '.join(prose.split()).split('. ')[0].rstrip('.') + '.'
    return _Doc(id=doc_id, description=description, lines=lines, sections=sections)


def _outline(sections: list[_Section]) -> str:
    top = min(section.level for section in sections)
    return '\n'.join(f'{"  " * (section.level - top)}- {section.path}: {section.title}' for section in sections)


def _read_clai_docs(path: str = '') -> str:
    """Read CLAI's docs: the list of docs, one doc, or one section of a doc.

    The docs ship with the installed package: CLAI's user reference, plugin contract, and
    customization guide, and the READMEs of the harness capabilities it runs on. A long doc
    returns an outline of section paths. A long section returns its own text (truncated if
    needed), followed by an outline of any subsections.

    Args:
        path: Empty to list the docs. A doc id such as `reference` or `harness/coder` for that doc.
            `doc#section` such as `reference#saved-api-keys` for one section with its subsections.
            `search_clai_docs` returns section paths.
    """
    path = path.strip()
    if not path:
        lines = ['CLAI docs. Read one with read_clai_docs(path), or find a section with search_clai_docs(query).', '']
        for doc_id in _DOC_IDS:
            doc = _doc(doc_id)
            lines.append(f'- {doc_id} ({len(doc.text()) // 1000} KB): {doc.description}')
        return '\n'.join(lines)
    doc_id, _, anchor = path.partition('#')
    if doc_id not in _DOC_IDS:
        close = get_close_matches(doc_id, _DOC_IDS, n=3, cutoff=0.5)
        hint = f' Did you mean {", ".join(close)}?' if close else ''
        raise ModelRetry(f'There is no CLAI doc {doc_id!r}.{hint} Call read_clai_docs with no path to list the docs.')
    doc = _doc(doc_id)
    if not anchor:
        text = doc.text()
        if len(text) <= _READ_LIMIT:
            return text
        return (
            f'{doc.id} is {len(text) // 1000} KB, too long to read whole. Read a section by its path:\n\n'
            f'{_outline(doc.sections)}'
        )
    by_anchor = {section.path.partition('#')[2]: section for section in doc.sections}
    section = by_anchor.get(anchor)
    if section is None:
        close = get_close_matches(anchor, list(by_anchor), n=3)
        hint = f' Did you mean {", ".join(f"{doc_id}#{other}" for other in close)}?' if close else ''
        raise ModelRetry(f'{doc_id} has no section {anchor!r}.{hint} Read {doc_id!r} for its outline.')
    text = doc.text(section.start, section.end)
    if len(text) <= _READ_LIMIT:
        return text
    own = doc.text(section.start, section.body_end)
    if len(own) > _READ_LIMIT:
        own = f'{own[:_READ_LIMIT]}\n\n[Truncated at {_READ_LIMIT} characters.]\n'
    subsections = [other for other in doc.sections if section.body_end <= other.start < section.end]
    if not subsections:
        return own
    return f'{own}\nThis section is too long to read whole. Read a subsection by its path:\n\n{_outline(subsections)}'


def _search_clai_docs(query: str) -> str:
    """Search CLAI's docs and return the best matching sections, with a short snippet of each.

    Matches words, and words that contain them, in the headings and text of every doc
    `read_clai_docs` lists.
    Read a result in full with `read_clai_docs(path)`.

    Args:
        query: Words to look for, such as `saved api keys` or `plugin settings menu`.
    """
    words = _WORD.findall(query.lower())
    terms = [word for word in words if len(word) > 2 and word not in _STOP_WORDS] or words
    if not terms:
        raise ModelRetry('Search for at least one word.')
    scored: list[tuple[int, int, int, str, str]] = []
    for doc_id in _DOC_IDS:
        doc = _doc(doc_id)
        for section in doc.sections:
            title = section.title.lower()
            body = doc.lines[section.start + 1 : section.body_end]
            text = ''.join(body).lower()
            matched = [term for term in terms if term in title or term in text]
            if matched:
                in_title = sum(term in title for term in matched)
                score = sum(text.count(term) for term in matched)
                summary = f'- {section.path}: {section.title}'
                # The first matching line, cut to a window around the term.
                for line in body:
                    excerpt = line.strip()
                    lowered = excerpt.lower()
                    if found := [lowered.find(term) for term in matched if term in lowered]:
                        start = max(0, min(found) - _SNIPPET_WIDTH // 3)
                        end = start + _SNIPPET_WIDTH
                        snippet = f'{"..." if start else ""}{excerpt[start:end]}{"..." if end < len(excerpt) else ""}'
                        summary += f'\n  {snippet}'
                        break
                scored.append((len(matched), in_title, score, section.path, summary))
    if not scored:
        return f'No CLAI docs section matches {query!r}. Call read_clai_docs with no path to list the docs.'
    # Sections that match more of the words first, then more of them in the heading, then more often.
    scored.sort(key=lambda result: (-result[0], -result[1], -result[2], result[3]))
    results = '\n'.join(summary for *_, summary in scored[:_SEARCH_RESULTS])
    return f'{results}\n\nRead a section with read_clai_docs(path).'


class ClaiDocs(Capability[None]):
    """Search and read installed CLAI docs, exposed only after the model loads this capability."""

    def __init__(self) -> None:
        """Offer the docs as a deferred capability; no doc is read until a tool is called."""
        super().__init__(
            id=CAPABILITY_ID,
            description=_DESCRIPTION,
            defer_loading=True,
            instructions=_HINT,
            tools=[Tool(_search_clai_docs, name='search_clai_docs'), Tool(_read_clai_docs, name='read_clai_docs')],
        )
