"""Types, protocols, path rules, and lexical search shared by the `Memory` stores."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import anyio.lowlevel
from pydantic import TypeAdapter, ValidationError

_VALID_SEGMENT_RE = re.compile(r'[A-Za-z0-9_.-]{1,200}')
MAX_RECEIPTS = 1024
"""How many recent operation receipts `InMemoryStore` and `FileStore` keep for replay protection."""
MAX_METADATA_CHARS = 4_096
"""The longest write metadata the bundled stores accept, measured as its JSON encoding."""
_METADATA_ADAPTER = TypeAdapter(dict[str, str])
logger = logging.getLogger(__name__)


class MemoryPathEscapeError(ValueError):
    """A memory path resolves outside its store directory."""


@dataclass(frozen=True)
class MemoryFile:
    """One memory file and its opaque compare-and-set version."""

    content: str
    version: str
    operation_id: str | None
    truncated: bool
    metadata: Mapping[str, str] | None = field(default=None, hash=False)
    """What the write that produced this version recorded about itself, through a `MetadataMemoryStore`.

    `None` when that write recorded nothing, or the store does not record metadata."""


@dataclass(frozen=True)
class MemoryOperation:
    """Stable identity and argument fingerprint for an idempotent mutation."""

    id: str
    fingerprint: str


@dataclass(frozen=True)
class MemoryMutation:
    """Result of a memory write or delete."""

    version: str | None
    replayed: bool
    existed: bool


@dataclass(frozen=True)
class MemorySearchMatch:
    """One bounded lexical-search match."""

    path: str
    snippet: str
    score: float


@dataclass(frozen=True)
class MemorySearchResult:
    """Bounded search results and scan metadata."""

    matches: list[MemorySearchMatch]
    scanned: int
    truncated: bool


class MemoryConflictError(RuntimeError):
    """The stored version did not match the mutation's expected version."""


class MemoryOperationConflictError(RuntimeError):
    """An operation id was reused with a different fingerprint."""


def validate_store_path(path: str) -> None:
    r"""Reject path strings that could escape a store's root directory or name another path.

    A `.` segment is rejected because a filesystem resolves `a/./b` to `a/b`, which would let one
    namespace address another's files.
    """
    if not all(
        _VALID_SEGMENT_RE.fullmatch(segment) and '..' not in segment and segment != '.' for segment in path.split('/')
    ):
        raise ValueError(f'invalid memory path: {path!r}')


def validate_store_prefix(prefix: str) -> None:
    if prefix:
        validate_store_path(prefix.removesuffix('/'))


def encode_metadata(metadata: Mapping[str, str] | None) -> str | None:
    """Validate write metadata and return the JSON text a store keeps, or `None` when there is none.

    An empty mapping is no metadata, so a file has one representation for it.
    """
    if not metadata:
        return None
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()):
        raise TypeError('memory metadata keys and values must be strings')
    # PostgreSQL's JSONB refuses the NUL character, so no store accepts it.
    if any('\x00' in key or '\x00' in value for key, value in metadata.items()):
        raise ValueError('memory metadata keys and values must not contain NUL characters')
    encoded = json.dumps(dict(metadata), ensure_ascii=False, separators=(',', ':'), sort_keys=True)
    if len(encoded) > MAX_METADATA_CHARS:
        raise ValueError(f'memory metadata is {len(encoded)} characters as JSON; the limit is {MAX_METADATA_CHARS}')
    return encoded


def decode_metadata(encoded: object) -> dict[str, str] | None:
    """Parse metadata kept by `encode_metadata`.

    Anything else, such as a value an application wrote to the column directly, reads as no
    metadata, with a warning, rather than failing the read the agent's tools depend on.
    """
    if encoded is None:
        return None
    if isinstance(encoded, str):
        try:
            return _METADATA_ADAPTER.validate_json(encoded, strict=True) or None
        except ValidationError:
            pass
    logger.warning('Ignoring memory metadata that is not a JSON object of strings')
    return None


def is_direct_child(prefix: str, path: str) -> bool:
    """Whether `path` sits directly below `prefix` rather than in a deeper scope nested under it."""
    return '/' not in path.removeprefix(prefix)


def _snippet(content: str, query: str, max_chars: int) -> str:
    if max_chars <= 0:  # pragma: no cover - lexical_search rejects non-positive budgets
        return ''
    lower = content.lower()
    positions = [lower.find(term) for term in query.lower().split()]
    found = [position for position in positions if position >= 0]
    center = min(found) if found else 0
    start = max(0, center - max_chars // 3)
    end = min(len(content), start + max_chars)
    start = max(0, end - max_chars)
    snippet = content[start:end]
    if start:
        snippet = f'...{snippet[3:]}' if len(snippet) >= 3 else '.' * len(snippet)
    if end < len(content):
        snippet = f'{snippet[:-3]}...' if len(snippet) >= 3 else '.' * len(snippet)
    return snippet


async def lexical_search(
    files: Iterable[tuple[str, str]],
    query: str,
    *,
    limit: int,
    max_files: int,
    max_chars: int,
    score_prefix: str = '',
) -> MemorySearchResult:
    """Search a sorted file stream with deterministic scan and output bounds.

    Scoring one file at the default bounds takes milliseconds, and a search can score a thousand, so
    it yields to the event loop after each file rather than holding it for the whole scan.
    """
    terms = [term for term in query.lower().split() if term]
    if not terms or limit <= 0 or max_files <= 0 or max_chars <= 0:  # pragma: no cover - stores prevalidate
        return MemorySearchResult(matches=[], scanned=0, truncated=False)
    scored: list[tuple[float, str, str]] = []
    scanned = 0
    truncated = False
    for path, content in files:
        if scanned >= max_files:
            truncated = True
            break
        scanned += 1
        lower_path = path.removeprefix(score_prefix).lower()
        lower_content = content.lower()
        score = float(sum(lower_content.count(term) + 2 * lower_path.count(term) for term in terms))
        if score:
            scored.append((score, path, content))
        await anyio.lowlevel.checkpoint()
    scored.sort(key=lambda item: (-item[0], item[1]))
    matches: list[MemorySearchMatch] = []
    remaining = max_chars
    for score, path, content in scored[:limit]:
        visible_path = path.removeprefix(score_prefix)
        available = remaining - len(visible_path)
        if available <= 0:
            truncated = True
            break
        snippet = _snippet(content, query, available)
        matches.append(MemorySearchMatch(path=path, snippet=snippet, score=score))
        remaining -= len(visible_path) + len(snippet)
    if len(scored) > len(matches):
        truncated = True
    return MemorySearchResult(matches=matches, scanned=scanned, truncated=truncated)


@runtime_checkable
class MemoryStore(Protocol):
    """Async versioned storage for agent memories."""

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        """Return the file at `path`, or `None` if it does not exist."""
        ...  # pragma: no cover

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        """Return a prior result for `operation`, validating its fingerprint."""
        ...  # pragma: no cover

    async def write(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        """Create or replace `path` if its version equals `expected_version`."""
        ...  # pragma: no cover

    async def delete(
        self,
        path: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        """Delete `path` if its version equals `expected_version`."""
        ...  # pragma: no cover

    async def list_paths(self, prefix: str = '', *, limit: int) -> list[str]:
        """Return all stored paths starting with `prefix`, sorted."""
        ...  # pragma: no cover


@runtime_checkable
class MetadataMemoryStore(MemoryStore, Protocol):
    """A `MemoryStore` that records caller-supplied metadata with each write, such as who made it.

    `InMemoryStore`, `SqliteMemoryStore` and `PostgresMemoryStore` implement it, and a custom store
    opts in by implementing `write_with_metadata`: `Memory(write_metadata=...)` checks for it with
    `isinstance`, as `search_memory` does for `SearchableMemoryStore`.

    Metadata describes the write that produced the current version. Each write replaces what the
    previous one recorded, in the same atomic step as the content, and `write` records none, so a
    file never keeps an earlier writer's metadata. `read` returns it as `MemoryFile.metadata`, and
    `delete` removes it with the file. A replayed `operation` returns its receipt, which carries no
    metadata, and leaves the stored metadata as it is.

    The metadata write is a method of its own, rather than a parameter of `write`, so `MemoryStore`
    keeps its signatures and a subclass of a bundled store that overrides `write` keeps type-checking.
    """

    async def write_with_metadata(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        metadata: Mapping[str, str] | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        """Create or replace `path` if its version equals `expected_version`, recording `metadata` with it.

        `None` or an empty mapping records no metadata, as `write` does.
        """
        ...  # pragma: no cover


@runtime_checkable
class SearchableMemoryStore(Protocol):
    """Optional bounded search extension for a `MemoryStore`."""

    async def search(
        self,
        prefix: str,
        query: str,
        *,
        limit: int,
        max_files: int,
        max_chars: int,
        max_file_chars: int,
    ) -> MemorySearchResult:
        """Search only paths below the trusted, application-resolved `prefix`."""
        ...  # pragma: no cover
