"""Discovery and chunking: turn a workspace directory into judgeable source snippets.

Source files are split along their syntax tree (see `_structure.py`): functions and methods stay whole,
long declarations split into the blocks inside them. Python uses the stdlib `ast`; fifteen more languages
use tree-sitter (`_treesitter.py`), where regions that do not parse fall back to overlapping line windows.
Other file types, and Python that does not parse, are cut into line windows entirely. Line coverage is
exact: every non-blank line lands in at least one snippet.

Files are listed with `rg --files` and read through the run's workspace, so a sandboxed workspace is
searched where it lives, never on the host.
"""

from __future__ import annotations

import asyncio
import posixpath
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.workspaces import Workspace, WorkspaceError
from pydantic_ai_harness.filesystem._ripgrep import RipgrepMissing, run_ripgrep
from pydantic_ai_harness.smart_grep._structure import Range, python_ranges
from pydantic_ai_harness.smart_grep._treesitter import treesitter_ranges

# Discovery is bounded by files and bytes only. Snippet count is not capped: only the ranked shortlist
# is judged, and local BM25 over the ~24k snippets of a repo root takes well under a second.
MAX_FILES = 20_000
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_CHUNK_CHARS = 12_000
WHOLE_DECLARATION_LINES = 160
WINDOW, OVERLAP = 60, 10
_READ_BATCH = 64
"""Files read from the workspace concurrently; the byte budget is checked between batches."""


class LineTooLong(ValueError):
    """A single line is too long to judge (minified or generated source)."""


@dataclass(frozen=True, kw_only=True)
class Chunk:
    """One judgeable snippet: a 1-based, inclusive line range of one file."""

    path: str
    line: int
    end_line: int
    text: str
    symbol: str | None = None


@dataclass(kw_only=True)
class Discovery:
    """Every snippet under a directory, and how many files were listed and skipped."""

    chunks: list[Chunk] = field(default_factory=list[Chunk])
    files: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    """`(path, reason)` for each listed file that could not be chunked."""


def windows(
    lines: Sequence[str],
    path: str,
    *,
    first_line: int = 1,
    size: int = WINDOW,
    overlap: int = OVERLAP,
    symbol: str | None = None,
) -> list[Chunk]:
    """Fixed-size line windows; giant windows are halved, never clipped."""
    out: list[Chunk] = []
    # A window starts wherever the previous one left lines uncovered.
    for i in range(0, max(len(lines) - overlap, 1), size - overlap):
        part = lines[i : i + size]
        text = '\n'.join(part)
        if not text.strip():
            continue
        if len(text) > MAX_CHUNK_CHARS:
            if size == 1:
                raise LineTooLong(f'line exceeds {MAX_CHUNK_CHARS} characters: {path}:{first_line + i}')
            out.extend(windows(part, path, first_line=first_line + i, size=max(1, size // 2), overlap=0, symbol=symbol))
        else:
            out.append(
                Chunk(path=path, line=first_line + i, end_line=first_line + i + len(part) - 1, text=text, symbol=symbol)
            )
    return out


def _chunks_from_ranges(lines: list[str], path: str, ranges: list[Range]) -> list[Chunk]:
    out: list[Chunk] = []
    next_line = 1
    for r in ranges:
        start, end = min(r.start, next_line), min(r.end, len(lines))
        if end < start:
            continue
        part = lines[start - 1 : end]
        whole = not r.window and end - start < WHOLE_DECLARATION_LINES and len('\n'.join(part)) <= MAX_CHUNK_CHARS
        size, overlap = (len(part), 0) if whole else (WINDOW, OVERLAP)
        out.extend(windows(part, path, first_line=start, size=size, overlap=overlap, symbol=r.symbol))
        next_line = end + 1
    if next_line <= len(lines):
        out.extend(windows(lines[next_line - 1 :], path, first_line=next_line))
    return out


def source_chunks(text: str, path: str) -> tuple[list[Chunk], str]:
    """Chunk one file's text. Returns `(chunks, parser name)`."""
    lines = text.splitlines()
    try:
        parsed = _parse(text, path)
    except RecursionError:  # nested deeper than the stack allows, in either parser's tree walk
        parsed = None
    if parsed:
        ranges, parser = parsed
        return _chunks_from_ranges(lines, path, ranges), parser
    return windows(lines, path), 'overlapping-lines'


def _parse(text: str, path: str) -> tuple[list[Range], str] | None:
    if not path.endswith('.py'):
        return treesitter_ranges(text, path)
    try:
        ranges = python_ranges(text)
    except (SyntaxError, ValueError):
        return None
    return (ranges, 'python') if ranges else None


async def list_files(workspace: Workspace, root: str, glob: str | None) -> tuple[list[str], list[tuple[str, str]]]:
    """Files under `root` that ripgrep would search, relative to it, and `(path, reason)` for unreadable ones.

    Honours `.gitignore`, skips hidden files and files over `MAX_FILE_BYTES`. `glob` only narrows that
    listing: ripgrep's own `--glob` overrides its ignore rules, so it would let `glob='.env'` pick a hidden,
    ignored file and send its contents to the judge.
    """
    paths, unreadable = await _ripgrep_files(workspace, root, [])
    if glob:
        matching, unreadable = await _ripgrep_files(workspace, root, ['--glob', glob])
        wanted = set(matching)
        paths = [path for path in paths if path in wanted]
    return paths, unreadable


async def _ripgrep_files(
    workspace: Workspace, root: str, arguments: list[str]
) -> tuple[list[str], list[tuple[str, str]]]:
    try:
        paths, capped, unreadable = await run_ripgrep(
            workspace,
            ['--files', '--sort', 'path', '--max-filesize', str(MAX_FILE_BYTES), *arguments],
            cwd=root,
            limit=MAX_FILES,
            listing=True,
            accept=lambda record: record.path,
        )
    except RipgrepMissing:
        raise ToolFailed(
            'smart_grep needs ripgrep (`rg`) in the workspace. Install it there '
            '(the `coder` extra does for a local workspace), or use regular file search instead.'
        ) from None
    if capped:
        raise ModelRetry(f'Search exceeds {MAX_FILES} files. Narrow the directory.')
    return paths, [(u.path, u.reason) for u in unreadable]


async def _read(workspace: Workspace, path: str) -> bytes | OSError:
    try:
        return await workspace.read_bytes(path)
    except WorkspaceError:
        raise
    except OSError as exc:  # the file vanished or became unreadable since it was listed
        return exc


async def _require_inside_working_dir(workspace: Workspace, root: str, directory: str) -> None:
    """Refuse a directory outside the working directory, symlinks followed: its source would go to the judge."""
    cwd, real = await asyncio.gather(workspace.realpath(await workspace.working_dir()), workspace.realpath(root))
    if posixpath.commonpath([cwd, real]) != cwd:
        raise ModelRetry(f'`{directory}` is outside the working directory; smart_grep only searches inside it.')


async def discover(workspace: Workspace, directory: str, glob: str | None = None) -> Discovery:
    """Read and chunk every eligible file under `directory`, a workspace path.

    Chunk paths are `directory` joined with the file's path below it, so they read the way the caller
    spelled the directory: relative to the working directory, or absolute.
    """
    root = await workspace.resolve(directory)
    await _require_inside_working_dir(workspace, root, directory)
    paths, unreadable = await list_files(workspace, root, glob)
    found = Discovery(files=len(paths), skipped=unreadable)
    texts: list[tuple[str, str]] = []
    total_bytes = 0
    for offset in range(0, len(paths), _READ_BATCH):
        batch = paths[offset : offset + _READ_BATCH]
        contents = await asyncio.gather(*(_read(workspace, posixpath.join(root, path)) for path in batch))
        for path, raw in zip(batch, contents):
            shown = posixpath.normpath(posixpath.join(directory, path))
            if isinstance(raw, OSError):
                found.skipped.append((shown, raw.strerror or type(raw).__name__))
                continue
            total_bytes += len(raw)
            if total_bytes > MAX_TOTAL_BYTES:
                raise ModelRetry('Search exceeds 32 MiB of source. Narrow the directory or glob.')
            if b'\0' in raw:
                found.skipped.append((shown, 'binary'))
                continue
            try:
                texts.append((shown, raw.decode('utf-8')))
            except UnicodeDecodeError:
                found.skipped.append((shown, 'not UTF-8'))
    for path, chunks in await asyncio.to_thread(_chunk_all, texts):
        if isinstance(chunks, LineTooLong):
            found.skipped.append((path, str(chunks)))
        else:
            found.chunks.extend(chunks)
    return found


def _chunk_all(texts: list[tuple[str, str]]) -> list[tuple[str, list[Chunk] | LineTooLong]]:
    """Chunk every file off the event loop; parsing a repo is CPU-bound."""
    out: list[tuple[str, list[Chunk] | LineTooLong]] = []
    for path, text in texts:
        try:
            out.append((path, source_chunks(text, path)[0]))
        except LineTooLong as exc:
            out.append((path, exc))
    return out
