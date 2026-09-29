"""The search pipeline: discover -> shortlist -> judge -> rank -> excerpt."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.smart_grep._chunks import Chunk, discover
from pydantic_ai_harness.smart_grep._judge import JudgeModel, judge
from pydantic_ai_harness.smart_grep._retrieve import rank, terms

MAX_QUERY_CHARS = 2000
DEFAULT_CANDIDATES = 128
MAX_CANDIDATES = 256
MAX_LIMIT = 100
EXCERPT_LINES = 12
EXCERPT_CHARS = 1200
_TEST_PATH = re.compile(r'(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]*$|[._-](test|spec)\.[^/]+$|_test\.py$')


@dataclass(kw_only=True)
class SmartGrepMatch:
    """One relevant snippet, with a short excerpt focused on the evidence."""

    file_path: str
    """The file, as a path relative to the working directory (or absolute, if the search directory was)."""
    start_line: int
    """First line of `text`, 1-based."""
    end_line: int
    """Last line of `text`, 1-based and inclusive."""
    symbol: str | None = None
    """The declaration the snippet belongs to, such as `Cache.get`, when the file was parsed."""
    kind: Literal['test'] | None = None
    """`'test'` when the file looks like a test; tests are labelled, not demoted."""
    score: float
    """The judge's relevance score, in `[0, 1]`."""
    text: str
    """At most 12 lines of the snippet, around the lines most relevant to the query."""
    snippet_start_line: int
    """First line of the whole judged snippet `text` was taken from."""
    snippet_end_line: int
    """Last line of the whole judged snippet `text` was taken from."""


@dataclass(kw_only=True)
class SmartGrepCoverage:
    """How much of the directory the search actually judged."""

    files: int = 0
    """Files listed under the directory."""
    snippets: int = 0
    """Snippets those files were cut into."""
    evaluated: int = 0
    """Snippets the lexical shortlist sent to the judge."""
    skipped_files: int = 0
    """Listed files that could not be searched (binary, not UTF-8, minified, or unreadable)."""
    selection_complete: bool = False
    """Whether every snippet was judged; otherwise code with unrelated wording may have been missed."""


@dataclass(kw_only=True)
class SmartGrepResult:
    """What one `smart_grep` call found, and how thoroughly it looked."""

    matches: list[SmartGrepMatch] = field(default_factory=list[SmartGrepMatch])
    """Relevant snippets, best first, overlapping snippets collapsed into the best one."""
    coverage: SmartGrepCoverage = field(default_factory=SmartGrepCoverage)
    """How much of the directory was judged."""
    omitted_matches: int = 0
    """Relevant snippets beyond `limit`."""
    warnings: list[str] = field(default_factory=list[str])
    """Why the result may be incomplete. No matches never prove absence."""


@dataclass(frozen=True)
class _Scored:
    chunk: Chunk
    score: float


def _overlaps(a: Chunk, b: Chunk) -> bool:
    return a.path == b.path and a.line <= b.end_line and b.line <= a.end_line


def _dedupe(passing: list[_Scored]) -> list[_Scored]:
    """Best-scoring window wins; overlapping lower-ranked windows collapse into it."""
    kept: list[_Scored] = []
    for item in passing:
        if not any(_overlaps(k.chunk, item.chunk) for k in kept):
            kept.append(item)
    return kept


def _evidence(match: Chunk, passing: list[_Scored]) -> Chunk | None:
    """A narrower judged block inside `match` that also passed, if any."""
    span = match.end_line - match.line
    inside = [
        p
        for p in passing
        if p.chunk.path == match.path
        and p.chunk.line >= match.line
        and p.chunk.end_line <= match.end_line
        and p.chunk.end_line - p.chunk.line < span
    ]
    if not inside:
        return None
    best = min(inside, key=lambda p: (-p.score, p.chunk.end_line - p.chunk.line))
    return best.chunk


def _excerpt(chunk: Chunk, focus: Chunk | None, query: str) -> tuple[int, str]:
    """`(first_line, text)` of at most `EXCERPT_LINES` lines / `EXCERPT_CHARS` chars."""
    lines = chunk.text.splitlines()
    first = chunk.line
    if focus is not None:
        offset = focus.line - chunk.line
        lines, first = lines[offset : offset + focus.end_line - focus.line + 1], focus.line
    if len(lines) > EXCERPT_LINES:
        wanted = set(terms(query))
        hits = [sum(t in wanted for t in terms(line)) for line in lines]
        best = max(range(len(lines) - EXCERPT_LINES + 1), key=lambda i: (sum(hits[i : i + EXCERPT_LINES]), -i))
        lines, first = lines[best : best + EXCERPT_LINES], first + best
    while len(lines) > 1 and len('\n'.join(lines)) > EXCERPT_CHARS:
        lines = lines[:-1]
    return first, '\n'.join(lines)[:EXCERPT_CHARS]


def _to_match(item: _Scored, passing: list[_Scored], query: str) -> SmartGrepMatch:
    chunk = item.chunk
    first, text = _excerpt(chunk, _evidence(chunk, passing), query)
    return SmartGrepMatch(
        file_path=chunk.path,
        start_line=first,
        end_line=first + text.count('\n'),
        symbol=chunk.symbol,
        kind='test' if _TEST_PATH.search(chunk.path) else None,
        score=round(item.score, 3),
        text=text,
        snippet_start_line=chunk.line,
        snippet_end_line=chunk.end_line,
    )


async def search_code(
    workspace: Workspace,
    model: JudgeModel,
    query: str,
    directory: str,
    *,
    glob: str | None = None,
    limit: int = 5,
    candidates: int = DEFAULT_CANDIDATES,
    threshold: float,
    concurrency: int,
) -> SmartGrepResult:
    """Find the snippets under `directory` that `model` judges relevant to `query`."""
    query = query.strip()
    if not query or len(query) > MAX_QUERY_CHARS:
        raise ModelRetry(f'Query must contain 1-{MAX_QUERY_CHARS} characters.')
    limit = max(1, min(limit, MAX_LIMIT))
    candidates = max(1, min(candidates, MAX_CANDIDATES))

    found = await discover(workspace, directory, glob)
    selected = rank(query, found.chunks)[:candidates]
    coverage = SmartGrepCoverage(
        files=found.files,
        snippets=len(found.chunks),
        evaluated=len(selected),
        skipped_files=len(found.skipped),
        selection_complete=len(selected) == len(found.chunks),
    )

    scores = await judge(model, query, selected, concurrency=concurrency) if selected else []
    passing = sorted(
        (_Scored(c, s) for c, s in zip(selected, scores) if s >= threshold),
        key=lambda p: (-p.score, p.chunk.path, p.chunk.line),
    )
    kept = _dedupe(passing)

    warnings: list[str] = []
    if selected and not kept:
        warnings.append(
            f'No snippet reached threshold {threshold}; this does not prove absence. '
            'Try rephrasing, a narrower directory, or regular grep.'
        )
    if not coverage.selection_complete:
        warnings.append(
            f'Judged {len(selected)} of {len(found.chunks)} snippets picked by a lexical shortlist; code with '
            'unrelated wording may be missed. Raise `candidates` or narrow the directory/glob.'
        )
    if found.skipped:
        warnings.append(f'{len(found.skipped)} files skipped (binary, non-UTF-8, minified or unreadable).')

    return SmartGrepResult(
        matches=[_to_match(item, passing, query) for item in kept[:limit]],
        coverage=coverage,
        omitted_matches=max(len(kept) - limit, 0),
        warnings=warnings,
    )
