"""Memory tools over a scoped, versioned `MemoryStore`."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Literal

from opentelemetry.trace import Span
from pydantic_ai import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from typing_extensions import TypedDict

from pydantic_ai_harness.memory._store import (
    MemoryConflictError,
    MemoryMutation,
    MemoryOperation,
    MemorySearchMatch,
    MemorySearchResult,
    MemoryStore,
    SearchableMemoryStore,
)

if TYPE_CHECKING:
    from pydantic_ai_harness.memory._capability import Memory

MAIN_FILENAME = 'MEMORY.md'
"""The main notebook file injected as bounded user-role context."""

_FILENAME_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}')
_CHARS_PER_TOKEN = 4
_MAX_CAS_ATTEMPTS = 16
_MAX_SEARCH_QUERY_CHARS = 1_000
_MAX_SEARCH_TERMS = 32
_MIN_FILENAME_LIST_CHARS = 6
_READ_TRUNCATION_MARKER = (
    '\n\n[Truncated: this file exceeds `max_memory_size`; edit it externally before using `write_memory`.]'
)


class MemoryWriteResult(TypedDict):
    """Content-free result from `write_memory`."""

    file: str
    version: str
    replayed: bool
    status: Literal['created', 'appended', 'updated']


class MemoryDeleteResult(TypedDict):
    """Content-free result from `delete_memory`."""

    file: str
    version: str | None
    replayed: bool
    status: Literal['deleted', 'not_found']


class MemorySearchMatchResult(TypedDict):
    """One model-facing, scope-relative search match."""

    file: str
    snippet: str
    score: float


class MemorySearchResponse(TypedDict):
    """Bounded result from `search_memory`."""

    matches: list[MemorySearchMatchResult]
    scanned: int
    truncated: bool


def normalize_filename(file: str) -> str:
    """Validate a model-supplied memory filename and normalize it to `<name>.md`."""
    name = file.strip()
    if name and not name.endswith('.md'):
        name = f'{name}.md'
    if not _FILENAME_RE.fullmatch(name) or '..' in name:
        raise ModelRetry(
            f'{file!r} is not a valid memory filename -- use a short name like "postgres-migration.md" '
            '(letters, digits, dots, dashes; no slashes).'
        )
    return name


def _normalize_search_query(query: str) -> str:
    if len(query) > _MAX_SEARCH_QUERY_CHARS:
        raise ModelRetry(f'Search queries must be at most {_MAX_SEARCH_QUERY_CHARS} characters.')
    normalized = query.strip()
    if not normalized:
        raise ModelRetry('Pass a non-empty search query.')

    terms: list[str] = []
    seen: set[str] = set()
    for term in normalized.split():
        key = term.lower()
        if key in seen:
            continue
        if len(terms) == _MAX_SEARCH_TERMS:
            raise ModelRetry(f'Search queries must contain at most {_MAX_SEARCH_TERMS} unique terms.')
        seen.add(key)
        terms.append(term)
    return ' '.join(terms)


async def list_subfiles(store: MemoryStore, scope: str, *, limit: int) -> tuple[list[str], bool]:
    """Return a bounded filename list and whether additional paths may exist."""
    prefix = f'{scope}/'
    names: list[str] = []
    request_limit = limit + 2  # one main notebook plus one truncation sentinel
    returned_paths = await store.list_paths(prefix, limit=request_limit)
    paths = returned_paths[:request_limit]
    for path in paths:
        if not path.startswith(prefix):
            raise RuntimeError('memory backend returned a path outside the requested scope')
        name = path.removeprefix(prefix)
        if '/' not in name and name != MAIN_FILENAME and name.endswith('.md'):
            names.append(name)
    names.sort()
    return names[:limit], len(names) > limit or len(returned_paths) >= request_limit


def injection_listing_limit(max_tokens: int) -> int:
    """Derive a finite backend listing bound from the complete prompt budget."""
    return max(1, max_tokens * _CHARS_PER_TOKEN // _MIN_FILENAME_LIST_CHARS)


def render_memory_prompt(
    main_content: str,
    subfiles: list[str],
    *,
    agent_name: str,
    guidance: str,
    max_lines: int,
    max_tokens: int,
    main_truncated: bool = False,
    files_truncated: bool = False,
) -> str:
    """Render a complete memory section within the strict approximate token budget."""
    budget = max_tokens * _CHARS_PER_TOKEN
    rendered_guidance = guidance
    source_lines = main_content.rstrip().splitlines()
    kept_lines = source_lines[-max_lines:] if max_lines else []
    dropped_lines = len(source_lines) - len(kept_lines)

    max_filename_candidates = min(len(subfiles), budget // 4)
    shown_files = subfiles[:max_filename_candidates]
    omitted_files = len(subfiles) - len(shown_files)

    rendered = _render_sections(
        agent_name,
        rendered_guidance,
        kept_lines,
        dropped_lines,
        shown_files,
        omitted_files,
        main_truncated,
        files_truncated,
    )
    while len(rendered) > budget and shown_files:
        shown_files.pop()
        omitted_files += 1
        rendered = _render_sections(
            agent_name,
            rendered_guidance,
            kept_lines,
            dropped_lines,
            shown_files,
            omitted_files,
            main_truncated,
            files_truncated,
        )
    while len(rendered) > budget and kept_lines:
        kept_lines.pop(0)
        dropped_lines += 1
        rendered = _render_sections(
            agent_name,
            rendered_guidance,
            kept_lines,
            dropped_lines,
            shown_files,
            omitted_files,
            main_truncated,
            files_truncated,
        )
    if len(rendered) > budget and dropped_lines:
        dropped_lines = 0
        rendered = _render_sections(
            agent_name,
            rendered_guidance,
            kept_lines,
            dropped_lines,
            shown_files,
            omitted_files,
            main_truncated,
            files_truncated,
        )
    if len(rendered) > budget and rendered_guidance:
        rendered_guidance = rendered_guidance[: max(0, len(rendered_guidance) - (len(rendered) - budget))]
        rendered = _render_sections(
            agent_name,
            rendered_guidance,
            kept_lines,
            dropped_lines,
            shown_files,
            omitted_files,
            main_truncated,
            files_truncated,
        )
    if len(rendered) <= budget:
        return rendered
    if omitted_files or files_truncated:
        marker = '[Memory files omitted; use search_memory]'
        return marker[:budget]
    return rendered[:budget]


def _render_sections(
    agent_name: str,
    guidance: str,
    main_lines: list[str],
    dropped_lines: int,
    subfiles: list[str],
    omitted_files: int,
    main_truncated: bool,
    files_truncated: bool,
) -> str:
    sections = [f'## Agent Memory ({agent_name})']
    if guidance:
        sections.append(guidance)
    if main_lines or dropped_lines or main_truncated:
        lines = list(main_lines)
        if dropped_lines:
            lines.insert(
                0,
                f'... [{dropped_lines} earlier lines; use read_memory("{MAIN_FILENAME}") for the full notebook] ...',
            )
        if main_truncated:
            lines.append('... [notebook exceeds `max_memory_size`; bounded prefix shown] ...')
        sections.append(f'### {MAIN_FILENAME}\n\n' + '\n'.join(lines))
    if subfiles or omitted_files or files_truncated:
        listing = [f'- {name}' for name in subfiles]
        if files_truncated:
            listing.append('- ... [more files omitted; use search_memory to find relevant memory]')
        elif omitted_files:
            listing.append(f'- ... [{omitted_files} more files; use search_memory to find relevant memory]')
        sections.append('### Other memory files\n\n' + '\n'.join(listing))
    return '\n\n'.join(sections)


def _apply_write(existing: str | None, content: str, old_text: str | None, name: str) -> tuple[str, str]:
    if old_text is None:
        if existing is not None and existing.strip():
            return f'{existing.rstrip()}\n{content.rstrip()}\n', 'appended'
        return f'{content.rstrip()}\n', 'created'
    if existing is None:
        raise ModelRetry(f'There is no memory file named {name!r} to edit -- omit `old_text` to create it.')
    occurrences = existing.count(old_text) if old_text else 0
    if occurrences == 0:
        raise ModelRetry(
            f'`old_text` was not found in {name!r} -- call `read_memory("{name}")` to see its current content.'
        )
    if occurrences > 1:
        raise ModelRetry(
            f'`old_text` appears {occurrences} times in {name!r}; it must match exactly once. '
            'Add surrounding context to make it unique.'
        )
    return existing.replace(old_text, content, 1), 'updated'


class MemoryToolset(FunctionToolset[AgentDepsT]):
    """Scoped read/write/delete/search tools with CAS and durable idempotency.

    The stable `memory` ID lets Temporal and Prefect wrap this static toolset.
    DBOS does not currently turn an ordinary `FunctionToolset` into a durable
    step, so applications requiring DBOS durability must provide that wrapper.
    """

    def __init__(self, capability: Memory[AgentDepsT]) -> None:
        super().__init__(id='memory')
        self._capability = capability
        self.add_function(self.write_memory, name='write_memory')
        self.add_function(self.read_memory, name='read_memory')
        self.add_function(self.delete_memory, name='delete_memory')
        self.add_function(self.search_memory, name='search_memory')

    async def write_memory(
        self,
        ctx: RunContext[AgentDepsT],
        content: str,
        file: str = MAIN_FILENAME,
        old_text: str | None = None,
    ) -> MemoryWriteResult:
        """Write persistent memory by appending or uniquely replacing text.

        Omit `old_text` to append, creating the file when necessary. Pass
        `old_text` to replace exactly one matching passage; use an empty
        `content` to remove that passage. Keep short durable facts in
        `MEMORY.md`, and longer or evolving topics in separate files. Update
        stale entries rather than adding contradictory duplicates.

        Args:
            ctx: Framework-provided run context.
            content: Text to append, or replacement text for `old_text`.
            file: Memory filename; defaults to `MEMORY.md`.
            old_text: Exact passage to replace, which must occur once.
        """
        capability = self._capability
        name = normalize_filename(file)
        if old_text is None and not content.strip():
            raise ModelRetry('Nothing to write -- pass the text to append, or `old_text` to replace.')
        store, scope = capability.resolve_scope(ctx)
        path = f'{scope}/{name}'
        operation = _operation(ctx, scope, 'write', path, {'content': content, 'file': file, 'old_text': old_text})
        scope_hash = _scope_hash(scope)
        with ctx.tracer.start_as_current_span(
            'memory.write', record_exception=False, set_status_on_exception=False
        ) as span:
            _set_span_base(span, store, scope_hash)
            try:
                previous_mutation = await store.get_operation(operation)
                if previous_mutation is not None:
                    result = _write_result(name, previous_mutation, old_text)
                    _set_span_result(span, 'replayed', replayed=True)
                    return result

                for attempt in range(_MAX_CAS_ATTEMPTS):
                    current = await store.read(path, max_chars=capability.max_memory_size)
                    if current is not None and current.truncated:
                        raise ModelRetry(
                            f'{name!r} exceeds max_memory_size and cannot be changed from partial content; '
                            'edit or replace it through the backing store first.'
                        )
                    updated, status = _apply_write(
                        None if current is None else current.content,
                        content,
                        old_text,
                        name,
                    )
                    if len(updated) > capability.max_memory_size:
                        raise ModelRetry(
                            f'{name!r} would grow to {len(updated)} characters; the limit is '
                            f'{capability.max_memory_size}. Split the content into separate memory files.'
                        )
                    try:
                        mutation = await store.write(
                            path,
                            updated,
                            expected_version=None if current is None else current.version,
                            operation=operation,
                        )
                    except MemoryConflictError:
                        if attempt + 1 == _MAX_CAS_ATTEMPTS:
                            raise ModelRetry('Memory changed repeatedly while writing; retry the operation.')
                        continue
                    result = _write_result(name, mutation, old_text, status=status)
                    _set_span_result(span, 'ok', chars=len(updated), replayed=mutation.replayed)
                    return result
                raise RuntimeError('unreachable CAS retry state')  # pragma: no cover
            except Exception as exc:
                _set_span_error(span, exc)
                raise

    async def read_memory(self, ctx: RunContext[AgentDepsT], file: str) -> str:
        """Read a bounded prefix of one memory file.

        Memory may be stale background context, so verify volatile facts before
        relying on them.

        Args:
            ctx: Framework-provided run context.
            file: Memory filename returned by injection or search.
        """
        name = normalize_filename(file)
        store, scope = self._capability.resolve_scope(ctx)
        with ctx.tracer.start_as_current_span(
            'memory.read', record_exception=False, set_status_on_exception=False
        ) as span:
            _set_span_base(span, store, _scope_hash(scope))
            try:
                memory_file = await store.read(f'{scope}/{name}', max_chars=self._capability.max_memory_size)
                if memory_file is None:
                    _set_span_result(span, 'not_found')
                    raise ModelRetry(
                        f'There is no memory file named {name!r} -- use `search_memory` to find existing memory.'
                    )
                _set_span_result(span, 'ok', chars=len(memory_file.content))
                return memory_file.content + (_READ_TRUNCATION_MARKER if memory_file.truncated else '')
            except Exception as exc:
                _set_span_error(span, exc)
                raise

    async def delete_memory(self, ctx: RunContext[AgentDepsT], file: str) -> MemoryDeleteResult:
        """Delete a non-main memory file that is no longer useful.

        `MEMORY.md` cannot be deleted; remove or correct its text with
        `write_memory` instead.

        Args:
            ctx: Framework-provided run context.
            file: Memory filename to delete.
        """
        name = normalize_filename(file)
        if name == MAIN_FILENAME:
            raise ModelRetry(f'{MAIN_FILENAME} is the main notebook; edit it with `write_memory` instead.')
        store, scope = self._capability.resolve_scope(ctx)
        path = f'{scope}/{name}'
        operation = _operation(ctx, scope, 'delete', path, {'file': file})
        with ctx.tracer.start_as_current_span(
            'memory.delete', record_exception=False, set_status_on_exception=False
        ) as span:
            _set_span_base(span, store, _scope_hash(scope))
            try:
                previous_mutation = await store.get_operation(operation)
                if previous_mutation is not None:
                    result = _delete_result(name, previous_mutation)
                    _set_span_result(span, 'replayed', replayed=True)
                    return result

                for attempt in range(_MAX_CAS_ATTEMPTS):
                    current = await store.read(path, max_chars=1)
                    try:
                        mutation = await store.delete(
                            path,
                            expected_version=None if current is None else current.version,
                            operation=operation,
                        )
                    except MemoryConflictError:
                        if attempt + 1 == _MAX_CAS_ATTEMPTS:
                            raise ModelRetry('Memory changed repeatedly while deleting; retry the operation.')
                        continue
                    result = _delete_result(name, mutation)
                    _set_span_result(span, result['status'], replayed=mutation.replayed)
                    return result
                raise RuntimeError('unreachable CAS retry state')  # pragma: no cover
            except Exception as exc:
                _set_span_error(span, exc)
                raise

    async def search_memory(self, ctx: RunContext[AgentDepsT], query: str) -> MemorySearchResponse:
        """Search memory files in the current tenant and agent scope.

        Results contain bounded snippets; call `read_memory` when a larger
        bounded excerpt is relevant.

        Args:
            ctx: Framework-provided run context.
            query: Terms to find in memory filenames and content.
        """
        query = _normalize_search_query(query)
        capability = self._capability
        store, scope = capability.resolve_scope(ctx)
        prefix = f'{scope}/'
        with ctx.tracer.start_as_current_span(
            'memory.search', record_exception=False, set_status_on_exception=False
        ) as span:
            _set_span_base(span, store, _scope_hash(scope))
            try:
                if isinstance(store, SearchableMemoryStore):
                    result = await store.search(
                        prefix,
                        query,
                        limit=capability.max_search_results,
                        max_files=capability.max_search_files,
                        max_chars=capability.max_search_result_chars,
                        max_file_chars=capability.max_memory_size,
                    )
                else:
                    result = await _fallback_search(
                        store,
                        prefix,
                        query,
                        limit=capability.max_search_results,
                        max_files=capability.max_search_files,
                        max_chars=capability.max_search_result_chars,
                        max_file_chars=capability.max_memory_size,
                    )
                matches, bounded_truncated = _bounded_search_matches(
                    result.matches,
                    prefix,
                    limit=capability.max_search_results,
                    max_chars=capability.max_search_result_chars,
                )
                scanned = min(max(result.scanned, 0), capability.max_search_files)
                truncated = result.truncated or bounded_truncated or result.scanned > capability.max_search_files
                _set_span_result(
                    span,
                    'ok',
                    matches=len(matches),
                    scanned=scanned,
                    chars=sum(len(match['snippet']) for match in matches),
                    truncated=truncated,
                )
                return {'matches': matches, 'scanned': scanned, 'truncated': truncated}
            except Exception as exc:
                _set_span_error(span, exc)
                raise


def _operation(
    ctx: RunContext[AgentDepsT],
    scope: str,
    kind: Literal['write', 'delete'],
    path: str,
    model_args: dict[str, str | None],
) -> MemoryOperation:
    tool_call_id = ctx.tool_call_id
    if not tool_call_id:
        raise RuntimeError('memory mutations require a stable tool_call_id')
    run_id = ctx.run_id
    if not run_id:
        raise RuntimeError('memory mutations require a stable run_id')
    operation_id = _digest({'scope': scope, 'run_id': run_id, 'tool_call_id': tool_call_id})
    fingerprint = _digest({'kind': kind, 'path': path, 'model_args': model_args})
    return MemoryOperation(id=operation_id, fingerprint=fingerprint)


def _digest(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_result(
    name: str,
    mutation: MemoryMutation,
    old_text: str | None,
    *,
    status: str | None = None,
) -> MemoryWriteResult:
    if mutation.version is None:
        raise RuntimeError('memory write returned no version')  # pragma: no cover
    if status is None:
        status = 'updated' if old_text is not None else ('appended' if mutation.existed else 'created')
    if status == 'created':
        typed_status: Literal['created', 'appended', 'updated'] = 'created'
    elif status == 'appended':
        typed_status = 'appended'
    else:
        typed_status = 'updated'
    return {'file': name, 'version': mutation.version, 'replayed': mutation.replayed, 'status': typed_status}


def _delete_result(name: str, mutation: MemoryMutation) -> MemoryDeleteResult:
    status: Literal['deleted', 'not_found'] = 'deleted' if mutation.existed else 'not_found'
    return {'file': name, 'version': mutation.version, 'replayed': mutation.replayed, 'status': status}


def _search_match(match: MemorySearchMatch, prefix: str, snippet: str) -> MemorySearchMatchResult:
    name = match.path.removeprefix(prefix)
    return {'file': name, 'snippet': snippet, 'score': match.score}


async def _fallback_search(
    store: MemoryStore,
    prefix: str,
    query: str,
    *,
    limit: int,
    max_files: int,
    max_chars: int,
    max_file_chars: int,
) -> MemorySearchResult:
    paths = await store.list_paths(prefix, limit=max_files + 1)
    terms = query.lower().split()
    scored: list[tuple[float, str, str]] = []
    scanned = 0
    content_truncated = False
    for path in paths[:max_files]:
        scanned += 1
        if not path.startswith(prefix):
            raise RuntimeError('memory backend returned a path outside the requested scope')
        name = path.removeprefix(prefix)
        if '/' in name or not _FILENAME_RE.fullmatch(name) or '..' in name:
            continue
        memory_file = await store.read(path, max_chars=max_file_chars)
        if memory_file is None:
            continue
        content_truncated = content_truncated or memory_file.truncated
        lower_path = name.lower()
        lower_content = memory_file.content.lower()
        score = float(sum(lower_content.count(term) + 2 * lower_path.count(term) for term in terms))
        if score:
            scored.append((score, path, memory_file.content))
    scored.sort(key=lambda item: (-item[0], item[1]))
    matches = [
        MemorySearchMatch(path=path, snippet=_search_snippet(content, query, max_chars), score=score)
        for score, path, content in scored[:limit]
    ]
    return MemorySearchResult(
        matches=matches,
        scanned=scanned,
        truncated=len(paths) > max_files or len(scored) > len(matches) or content_truncated,
    )


def _search_snippet(content: str, query: str, max_chars: int) -> str:
    lower = content.lower()
    positions = [lower.find(term) for term in query.lower().split()]
    found = [position for position in positions if position >= 0]
    center = min(found) if found else 0
    start = max(0, center - max_chars // 3)
    end = min(len(content), start + max_chars)
    start = max(0, end - max_chars)
    snippet = content[start:end]
    if start and len(snippet) >= 3:
        snippet = f'...{snippet[3:]}'
    if end < len(content) and len(snippet) >= 3:
        snippet = f'{snippet[:-3]}...'
    return snippet


def _bounded_search_matches(
    matches: list[MemorySearchMatch],
    prefix: str,
    *,
    limit: int,
    max_chars: int,
) -> tuple[list[MemorySearchMatchResult], bool]:
    bounded: list[MemorySearchMatchResult] = []
    remaining = max_chars
    truncated = len(matches) > limit
    for match in matches[:limit]:
        if not match.path.startswith(prefix):
            raise RuntimeError('memory search backend returned a result outside the requested scope')
        name = match.path.removeprefix(prefix)
        if '/' in name or not _FILENAME_RE.fullmatch(name) or '..' in name:
            raise RuntimeError('memory search backend returned a non-file or nested result')
        remaining -= len(name)
        if remaining < 0:
            truncated = True
            break
        snippet = match.snippet[:remaining]
        if len(snippet) < len(match.snippet):
            truncated = True
        bounded.append(_search_match(match, prefix, snippet))
        remaining -= len(snippet)
    return bounded, truncated


def _scope_hash(scope: str) -> str:
    return hashlib.sha256(scope.encode()).hexdigest()[:16]


def _set_span_base(span: Span, store: MemoryStore, scope_hash: str) -> None:
    if span.is_recording():
        span.set_attributes({'memory.backend': type(store).__name__, 'memory.scope_hash': scope_hash})


def _set_span_result(span: Span, outcome: str, **attributes: str | int | bool) -> None:
    if span.is_recording():
        span.set_attributes(
            {'memory.outcome': outcome, **{f'memory.{key}': value for key, value in attributes.items()}}
        )


def _set_span_error(span: Span, exc: Exception) -> None:
    if span.is_recording():
        span.set_attributes({'memory.outcome': 'error', 'memory.exception_type': type(exc).__name__})
