"""The composer's `@` picker: a session folder's files and folders, best match first for what follows `@`."""

import heapq
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import anyio
from anyio import to_thread

from pydantic_clai2.builtin_plugins.web.protocol import CompletionInfo

FILE_LIMIT = 20_000
"""Most files read from one folder, so the picker answers quickly in a huge one."""
MATCH_LIMIT = 50
"""Most suggestions one `@` offers."""
FRESH_FOR = 5.0
"""Seconds a folder's listing is reused before it is read again, so files written meanwhile show up."""
SKIPPED = frozenset({'node_modules', '__pycache__'})
"""Folders a folder outside Git never lists, with every hidden one, as `.gitignore` usually leaves them out."""


@dataclass(frozen=True, kw_only=True)
class _Listing:
    read_at: float
    paths: tuple[str, ...]


class WorkspaceFiles:
    """Each session folder's files and their folders, read once per `FRESH_FOR` however many browsers ask."""

    def __init__(self) -> None:
        self._listings: dict[Path, _Listing] = {}
        self._reading: defaultdict[Path, anyio.Lock] = defaultdict(anyio.Lock)
        """One per folder, so requests that find its listing stale wait for one read instead of starting their own."""

    async def find(self, folder: Path, query: str) -> list[CompletionInfo]:
        """The best `MATCH_LIMIT` paths in `folder` for `query`, the text after `@`, each replacing `@query`."""
        async with self._reading[folder]:
            listing = self._listings.get(folder)
            if listing is None or time.monotonic() - listing.read_at > FRESH_FOR:
                listing = _Listing(read_at=time.monotonic(), paths=await list_paths(folder))
                self._listings[folder] = listing
        found = await to_thread.run_sync(best, listing.paths, query)
        return [_completion(path, query) for path in found]


async def list_paths(folder: Path) -> tuple[str, ...]:
    """`folder`'s files, then the folders holding them (ending in `/`), relative to it, with `/` separators.

    In a Git checkout they are the files Git tracks or would add, so `.gitignore` applies; elsewhere a walk
    that skips hidden and `SKIPPED` folders. At most `FILE_LIMIT` files either way.
    """
    try:
        listed = await anyio.run_process(
            ['git', '-C', str(folder), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
            check=False,
        )
    except OSError:  # No Git installed.
        listed = None
    if listed is not None and listed.returncode == 0:
        # An unmerged file is listed once per stage.
        names = dict.fromkeys(name for name in listed.stdout.decode(errors='replace').split('\0') if name)
        files = list(names)[:FILE_LIMIT]
    else:
        files = await to_thread.run_sync(_walk, folder)
    folders = {f'{parent}/': None for name in files for parent in _parents(name)}
    return (*files, *folders)


def best(paths: tuple[str, ...], query: str) -> list[str]:
    """The best `MATCH_LIMIT` of `paths` for `query`, ignoring case.

    A name or path starting with it comes first, then a name holding it, a path holding it, and a path holding
    its characters in order; within each, shallower and shorter paths first. An empty query lists the top level,
    and the path typed in full is left out.
    """
    needle = query.lower()
    ranked: list[tuple[int, int, int, str]] = []
    for path in paths:
        lowered = path.lower()
        name = _name(lowered)
        if lowered == needle:
            continue  # Typed in full: nothing left to complete.
        if lowered.startswith(needle) or name.startswith(needle):
            rank = 0
        elif needle in name:
            rank = 1
        elif needle in lowered:
            rank = 2
        elif _scattered(needle, lowered):
            rank = 3
        else:
            continue
        ranked.append((rank, path.rstrip('/').count('/'), len(path), path))
    return [path for *_, path in heapq.nsmallest(MATCH_LIMIT, ranked)]


def _completion(path: str, query: str) -> CompletionInfo:
    directory = path.rstrip('/').rpartition('/')[0]
    return CompletionInfo(text=f'@{path}', start=len(query) + 1, display=_name(path), meta=directory or None)


def _name(path: str) -> str:
    """The last part of `path`, keeping a folder's `/`."""
    return path[path.rstrip('/').rfind('/') + 1 :]


def _scattered(needle: str, text: str) -> bool:
    """Whether `text` holds the characters of `needle` in order."""
    start = 0
    for character in needle:
        start = text.find(character, start) + 1
        if not start:
            return False
    return True


def _parents(name: str) -> list[str]:
    parts = name.split('/')[:-1]
    return ['/'.join(parts[: end + 1]) for end in range(len(parts))]


def _walk(folder: Path) -> list[str]:
    found: list[str] = []
    for directory, folders, files in os.walk(folder):
        folders[:] = sorted(name for name in folders if not name.startswith('.') and name not in SKIPPED)
        relative = Path(directory).relative_to(folder)
        for name in sorted(files):
            found.append((relative / name).as_posix())
            if len(found) == FILE_LIMIT:
                return found
    return found
