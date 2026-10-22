"""Discovery and chunking: turn a workspace directory into judgeable snippets.

Every file is plain text cut into overlapping line windows, except Python, which is split along its stdlib
`ast` (see `_structure.py`): functions and methods stay whole, long declarations split into the blocks inside
them. Python that does not parse is cut into line windows too. Line coverage is exact: every non-blank line
lands in at least one snippet.

Files are listed with `rg --files` and read through the run's workspace (see `_index.py`), so a sandboxed
workspace is searched where it lives, never on the host.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic_ai._utils import gather
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.workspaces import Workspace, WorkspaceError
from pydantic_ai_harness.filesystem._ripgrep import RipgrepMissing, run_ripgrep
from pydantic_ai_harness.smart_search._structure import Range, python_ranges

# A search is bounded by files, lines and bytes, sized to fit the Linux kernel (~96k files, ~39M lines,
# ~1.1 GiB). Snippet count is not capped: only the ranked shortlist is judged, and the index only reads the
# query's postings.
MAX_FILES = 200_000
MAX_TOTAL_LINES = 50_000_000
MAX_TOTAL_BYTES = 2 << 30
MAX_FILE_BYTES = 1024 * 1024
MAX_CHUNK_CHARS = 12_000
WHOLE_DECLARATION_LINES = 160
WINDOW, OVERLAP = 60, 10
_LINE_BREAK = re.compile(r'\r\n|\r|\n')


def split_lines(text: str) -> list[str]:
    """`text`'s lines, numbered as editors, ripgrep and Python's tokenizer number them.

    Unlike `str.splitlines`, a form feed, U+2028 and the other separators it also splits on stay inside
    their line, so snippet line numbers match the file and the `ast` ranges of Python.
    """
    lines = _LINE_BREAK.split(text)
    if lines[-1] == '':
        lines.pop()
    return lines


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
        start, end = min(r.start, next_line), r.end
        # `ast` ranges lie within the lines `split_lines` counts, so a range never ends before it starts.
        assert start <= end <= len(lines), (path, r)
        part = lines[start - 1 : end]
        whole = end - start < WHOLE_DECLARATION_LINES and len('\n'.join(part)) <= MAX_CHUNK_CHARS
        size, overlap = (len(part), 0) if whole else (WINDOW, OVERLAP)
        out.extend(windows(part, path, first_line=start, size=size, overlap=overlap, symbol=r.symbol))
        next_line = end + 1
    if next_line <= len(lines):
        out.extend(windows(lines[next_line - 1 :], path, first_line=next_line))
    return out


def source_chunks(text: str, path: str) -> list[Chunk]:
    """Cut one file's text into snippets: along its declarations for Python that parses, else line windows."""
    lines = split_lines(text)
    if path.endswith('.py'):
        try:
            ranges = python_ranges(text)
        except (SyntaxError, ValueError, RecursionError):  # unparsable, or nested deeper than the stack allows
            ranges = None
        if ranges:
            return _chunks_from_ranges(lines, path, ranges)
    return windows(lines, path)


async def list_files(workspace: Workspace, root: str, glob: str | None) -> tuple[list[str], list[tuple[str, str]]]:
    """Files under `root` that ripgrep would search, relative to it, and `(path, reason)` for unreadable ones.

    Honors `.gitignore`, skips hidden files and files over `MAX_FILE_BYTES`. `glob` only narrows that
    listing: ripgrep's own `--glob` overrides its ignore rules, so it would let `glob='.env'` pick a hidden,
    ignored file and send its contents to the judge. The file cap applies after the glob, so a narrow glob
    can search a workspace with more than `MAX_FILES` files.
    """
    if not glob:
        return await _ripgrep_files(workspace, root, [], accept=lambda path: path, narrow='the directory')
    matching, unreadable = await _ripgrep_files(
        workspace, root, ['--glob', glob], accept=lambda path: path, narrow='the directory or glob'
    )
    wanted = set(matching)
    paths, _ = await _ripgrep_files(
        workspace, root, [], accept=lambda path: path if path in wanted else None, narrow='the directory'
    )
    return paths, unreadable


async def _ripgrep_files(
    workspace: Workspace, root: str, arguments: list[str], *, accept: Callable[[str], str | None], narrow: str
) -> tuple[list[str], list[tuple[str, str]]]:
    try:
        paths, capped, unreadable = await run_ripgrep(
            workspace,
            ['--files', '--sort', 'path', '--max-filesize', str(MAX_FILE_BYTES), *arguments],
            cwd=root,
            limit=MAX_FILES,
            listing=True,
            accept=lambda record: accept(record.path),
        )
    except RipgrepMissing:
        raise ToolFailed(
            '`smart_search` needs ripgrep (`rg`) in the workspace. Install it there '
            '(the `coder` extra does for a local workspace), or use regular file search instead.'
        ) from None
    if capped:
        raise ModelRetry(f'Search exceeds {MAX_FILES} files. Narrow {narrow}.')
    return paths, [(u.path, u.reason) for u in unreadable]


async def read_or_error(workspace: Workspace, path: str) -> bytes | OSError:
    """The file's bytes, or the `OSError` of a file that vanished or became unreadable since it was listed."""
    try:
        return await workspace.read_bytes(path)
    except WorkspaceError:
        raise
    except OSError as exc:
        return exc


async def searchable_root(workspace: Workspace, directory: str) -> str:
    """The real path of `directory`, refused outside the working directory: its source would go to the judge."""
    root = await workspace.resolve(directory)
    cwd, real = await gather(workspace.realpath(await workspace.working_dir()), workspace.realpath(root))
    if posixpath.commonpath([cwd, real]) != cwd:
        raise ModelRetry(f'{directory!r} is outside the working directory; `smart_search` only searches inside it.')
    return real
