"""Run ripgrep and split its NUL-delimited output into records.

`rg` is invoked with `--null`, so every file path it prints ends in a NUL byte
and cannot be confused with the `:`/`-` separators of the match text that
follows it. Records are streamed through the caller's `accept` filter and the
process is stopped once `limit` accepted records have been collected, so a
search over a large tree neither buffers everything before the cap applies nor
counts records the caller then drops.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import anyio
import anyio.abc
from pydantic_ai.exceptions import ModelRetry

_SEPARATOR = b'--\n'
"""What `rg` prints between non-adjacent context groups; carries no path and is dropped."""

_MAX_RECORD_BYTES = 1 << 20
"""Longest record buffered while waiting for its terminator; `rg`'s own `--max-columns` keeps lines far shorter."""

_T = TypeVar('_T')


@dataclass(kw_only=True, frozen=True)
class Record:
    """One line of ripgrep output: the file it refers to, and the rest of the line."""

    path: str
    """The path as `rg` printed it, relative to the directory it was run in."""
    text: str
    """Empty for a file listing; otherwise `<line>:<text>` for a match or `<line>-<text>` for context."""


async def run_ripgrep(
    arguments: Sequence[str],
    *,
    cwd: Path,
    limit: int,
    listing: bool = False,
    accept: Callable[[Record], _T | None],
) -> tuple[list[_T], bool]:
    """Run `rg --null` in `cwd`; return up to `limit` accepted records and whether more were cut.

    `accept` maps a record to what the caller keeps, or `None` to drop it; only
    kept records count towards `limit`. `listing` reads `--files` output, where
    each record is a bare path. Raises `ModelRetry` when `rg` is not installed
    or reports an error (an invalid pattern, say), so the model can correct the
    call or use another tool.
    """
    results: list[_T] = []
    truncated = False
    stderr = bytearray()
    terminator = b'\0' if listing else b'\n'

    async def read_stderr(stream: anyio.abc.ByteReceiveStream) -> None:
        async for chunk in stream:
            stderr.extend(chunk)

    try:
        async with await anyio.open_process(
            ['rg', '--null', '--color=never', *arguments], cwd=cwd, stderr=subprocess.PIPE
        ) as process:
            assert process.stdout is not None
            assert process.stderr is not None
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(read_stderr, process.stderr)
                pending = b''
                async for chunk in process.stdout:
                    pending += chunk
                    while not truncated and (end := pending.find(terminator)) >= 0:
                        line, pending = pending[: end + 1], pending[end + 1 :]
                        if line == _SEPARATOR:
                            continue
                        kept = accept(_record(line, listing=listing))
                        if kept is None:
                            continue
                        if len(results) >= limit:
                            truncated = True
                        else:
                            results.append(kept)
                    if truncated or len(pending) > _MAX_RECORD_BYTES:
                        truncated = True
                        with suppress(ProcessLookupError):  # a fast search may already have exited
                            process.terminate()
                        break
            await process.wait()
    except FileNotFoundError as exc:
        raise ModelRetry('ripgrep (rg) is not installed. Install it, or use the pure-Python search tools.') from exc
    if not truncated and process.returncode not in (0, 1):
        detail = stderr.decode('utf-8', errors='replace').strip()
        raise ModelRetry(f'ripgrep failed: {detail or f"exit code {process.returncode}"}')
    return results, truncated


def _record(line: bytes, *, listing: bool) -> Record:
    if listing:
        return Record(path=line[:-1].decode('utf-8', errors='replace'), text='')
    path, _, text = line.rstrip(b'\n').partition(b'\0')
    return Record(path=path.decode('utf-8', errors='replace'), text=text.decode('utf-8', errors='replace'))
