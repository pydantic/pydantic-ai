"""Per-directory snippet indexes, refreshed file by file so repeat searches skip unchanged files.

A refresh lists the directory with `rg`, reads every listed file through the workspace and compares its
content hash with the index: only new and changed files are chunked and indexed, and files no longer
listed are retired. The workspace API has no modification times, so reading is the change check. It is
also what makes a kept index safe on any backend: a stale entry never survives a refresh. Snippet text is
not kept; the shortlist is re-read from its files.

`SnippetIndexes` keeps indexes between searches when caching is on. Parallel searches of one directory
take turns on its lock, so its index is built once and reused rather than built by every call.
"""

from __future__ import annotations

import asyncio
import heapq
import posixpath
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import blake2b

import anyio
import anyio.to_thread

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.smart_file_search._chunks import (
    MAX_TOTAL_BYTES,
    MAX_TOTAL_LINES,
    Chunk,
    LineTooLong,
    list_files,
    read_or_error,
    source_chunks,
)
from pydantic_ai_harness.smart_file_search._retrieve import Bm25

MAX_CACHED_INDEXES = 4
"""Directories (with their glob) whose index is kept between searches, least recently searched evicted first."""
COMPACT_AFTER = 10_000
"""Retired snippets tolerated before the index is compacted, once they also outnumber live ones."""
_READ_BATCH = 64
"""Files read from the workspace concurrently; each batch is indexed before the next is read."""


@dataclass(frozen=True, kw_only=True)
class Hit:
    """A shortlisted snippet's location; its text is re-read from the file."""

    path: str
    """Relative to the indexed directory."""
    line: int
    end_line: int
    symbol: str | None
    digest: bytes
    """The indexed content hash of the file, to detect a change before the text is re-read."""


@dataclass(kw_only=True)
class Shortlist:
    """The snippets worth judging, and how much of the directory they were picked from."""

    hits: list[Hit]
    files: int
    snippets: int
    skipped: list[tuple[str, str]]
    """`(path, reason)` for each listed file that could not be searched."""


@dataclass(kw_only=True)
class _File:
    digest: bytes
    lines: int
    docs: list[int] = field(default_factory=list[int])
    skipped: str | None = None
    """Why the file has no snippets: binary, not UTF-8, or a line too long to judge."""


def _digest(raw: bytes) -> bytes:
    return blake2b(raw, digest_size=16).digest()


class SnippetIndex:
    """Every snippet of one directory: file digests, snippet locations and their BM25 index.

    Not thread-safe: its owner's lock admits one caller at a time, which runs the work in a worker thread.
    """

    def __init__(self) -> None:
        self._files: dict[str, _File] = {}
        self._hits: list[Hit | None] = []
        """Snippet locations by BM25 document id; `None` once retired."""
        self._bm25 = Bm25()

    @property
    def snippets(self) -> int:
        """Snippets currently indexed."""
        return self._bm25.size

    def update(self, batch: Sequence[tuple[str, bytes]]) -> int:
        """Index the files in `batch` that are new or changed. Returns the batch's line count."""
        lines = 0
        for path, raw in batch:
            digest = _digest(raw)
            file = self._files.get(path)
            if file is None or file.digest != digest:
                if file is not None:
                    self._retire(file)
                file = self._files[path] = self._index(path, raw, digest)
            lines += file.lines
        return lines

    def _index(self, path: str, raw: bytes, digest: bytes) -> _File:
        file = _File(digest=digest, lines=raw.count(b'\n'))
        if b'\0' in raw:
            file.skipped = 'binary'
            return file
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            file.skipped = 'not UTF-8'
            return file
        # Counted the way chunking splits lines, so separators like U+2028 cannot slip past the line budget.
        file.lines = len(text.splitlines())
        try:
            chunks = source_chunks(text, path)[0]
        except LineTooLong as exc:
            file.skipped = str(exc)
            return file
        for chunk in chunks:
            file.docs.append(self._bm25.add(chunk.text, f'{path} {chunk.symbol or ""}'))
            self._hits.append(
                Hit(path=path, line=chunk.line, end_line=chunk.end_line, symbol=chunk.symbol, digest=digest)
            )
        return file

    def _retire(self, file: _File) -> None:
        for doc in file.docs:
            self._bm25.retire(doc)
            self._hits[doc] = None

    def prune(self, listed: set[str]) -> None:
        """Retire files that are no longer listed, and compact once retired snippets dominate."""
        for path in [path for path in self._files if path not in listed]:
            self._retire(self._files.pop(path))
        if self._bm25.retired > max(COMPACT_AFTER, self._bm25.size):
            renumber = self._bm25.compact()
            self._hits = [hit for hit in self._hits if hit is not None]
            for file in self._files.values():
                file.docs = [renumber[doc] for doc in file.docs]

    def skipped(self, paths: Sequence[str]) -> list[tuple[str, str]]:
        """`(path, reason)` for each of `paths` that is indexed without snippets."""
        return [(path, file.skipped) for path in paths if (file := self._files.get(path)) and file.skipped]

    def shortlist(self, query: str, k: int) -> list[Hit]:
        """The `k` best snippets for `query`: BM25 matches first, then unmatched snippets in file order."""
        scores = self._bm25.scores(query)

        def order(doc: int) -> tuple[float, str, int, int]:
            hit = self._live(doc)
            return -scores.get(doc, 0.0), hit.path, hit.line, doc

        best = heapq.nsmallest(k, scores, key=order)
        if len(best) < k:
            rest = (doc for doc, hit in enumerate(self._hits) if hit is not None and doc not in scores)
            best += heapq.nsmallest(k - len(best), rest, key=order)
        return [self._live(doc) for doc in best]

    def _live(self, doc: int) -> Hit:
        hit = self._hits[doc]
        assert hit is not None, 'only live snippets are scored or shortlisted'
        return hit


@dataclass
class _Slot:
    lock: anyio.Lock = field(default_factory=anyio.Lock)
    index: SnippetIndex = field(default_factory=SnippetIndex)


class SnippetIndexes:
    """Snippet indexes by directory and glob, kept between searches for up to `max_cached` of them.

    An index being searched is never evicted, so more may be kept while searches run in parallel; the excess
    is dropped as each search finishes.

    With `max_cached=0` every search builds a throwaway index. A search holds its directory's lock from
    refresh to shortlist, so parallel searches of one directory build its index once, in turn.
    """

    def __init__(self, max_cached: int = MAX_CACHED_INDEXES) -> None:
        self._max_cached = max_cached
        self._slots: OrderedDict[tuple[str, str | None], _Slot] = OrderedDict()

    def _slot(self, key: tuple[str, str | None]) -> _Slot:
        # Created on first use, inside a running loop: the lock binds to the loop that searches.
        slot = self._slots.pop(key, None) or _Slot()
        if self._max_cached:
            self._slots[key] = slot  # most recently searched last
            self._evict(keep=key)
        return slot

    def _evict(self, keep: tuple[str, str | None] | None = None) -> None:
        """Drop the least recently searched indexes over `max_cached`, except `keep` and any being searched."""
        # Never evict an index mid-search: a parallel search of it would then build a second one.
        idle = [other for other, kept in self._slots.items() if other != keep and not kept.lock.locked()]
        for other in idle[: len(self._slots) - self._max_cached]:
            del self._slots[other]

    async def shortlist(self, workspace: Workspace, root: str, glob: str | None, query: str, k: int) -> Shortlist:
        """Refresh `root`'s index from the workspace and shortlist `k` snippets for `query`."""
        key = (root, glob)
        slot = self._slot(key)
        async with slot.lock:
            try:
                files, skipped = await _refresh(workspace, root, glob, slot.index)
            except ModelRetry:
                # Over a search budget: keep no partial index of a directory that cannot be searched.
                if self._slots.get(key) is slot:
                    del self._slots[key]
                raise
            hits = await anyio.to_thread.run_sync(slot.index.shortlist, query, k)
        # Parallel searches can leave more than `max_cached` indexes; trim once this one is no longer busy.
        self._evict()
        return Shortlist(hits=hits, files=files, snippets=slot.index.snippets, skipped=skipped)


async def _refresh(
    workspace: Workspace, root: str, glob: str | None, index: SnippetIndex
) -> tuple[int, list[tuple[str, str]]]:
    """Bring `index` up to date with `root`. Returns the listed file count and the skipped files."""
    paths, skipped = await list_files(workspace, root, glob)
    readable: set[str] = set()
    lines = size = 0
    for offset in range(0, len(paths), _READ_BATCH):
        batch = paths[offset : offset + _READ_BATCH]
        contents = await asyncio.gather(*(read_or_error(workspace, posixpath.join(root, path)) for path in batch))
        read: list[tuple[str, bytes]] = []
        for path, raw in zip(batch, contents):
            if isinstance(raw, OSError):
                skipped.append((path, raw.strerror or type(raw).__name__))
            else:
                read.append((path, raw))
                readable.add(path)
                size += len(raw)
        if size > MAX_TOTAL_BYTES:
            raise ModelRetry(f'Search exceeds {MAX_TOTAL_BYTES >> 30} GiB of source. Narrow the directory or glob.')
        lines += await anyio.to_thread.run_sync(index.update, read)
        if lines > MAX_TOTAL_LINES:
            raise ModelRetry(f'Search exceeds {MAX_TOTAL_LINES:,} lines of source. Narrow the directory or glob.')
    await anyio.to_thread.run_sync(index.prune, readable)
    return len(paths), skipped + index.skipped(paths)


async def read_snippets(workspace: Workspace, root: str, directory: str, hits: Sequence[Hit]) -> list[Chunk]:
    """The shortlisted snippets' text, re-read from their files. Files changed since indexing are left out.

    Paths are `directory` joined with each file's path below it, so they read the way the caller spelled it.
    """
    digests = {hit.path: hit.digest for hit in hits}
    contents = await asyncio.gather(*(read_or_error(workspace, posixpath.join(root, path)) for path in digests))
    lines = {
        path: raw.decode('utf-8').splitlines()
        for path, raw in zip(digests, contents)
        if isinstance(raw, bytes) and _digest(raw) == digests[path]
    }
    return [
        Chunk(
            path=posixpath.normpath(posixpath.join(directory, hit.path)),
            line=hit.line,
            end_line=hit.end_line,
            text='\n'.join(lines[hit.path][hit.line - 1 : hit.end_line]),
            symbol=hit.symbol,
        )
        for hit in hits
        if hit.path in lines
    ]
