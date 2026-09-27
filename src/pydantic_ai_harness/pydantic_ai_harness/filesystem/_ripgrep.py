"""Run ripgrep inside the workspace and split its NUL-delimited output into records.

`rg` is invoked with `--null`, so every file path it prints ends in a NUL byte
and cannot be confused with the `:`/`-` separators of the match text that
follows it. The workspace returns a command's output whole, so the output is
cut inside the workspace at `_MAX_OUTPUT_BYTES` (`head -c`) and records are then
streamed through the caller's `accept` filter until `limit` accepted records
have been collected; only kept records count towards the cap.
"""

from __future__ import annotations

import shlex
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.workspaces import Workspace

_SEPARATOR = '--\n'
"""What `rg` prints between non-adjacent context groups; carries no path and is dropped."""

_MAX_RECORD_BYTES = 1 << 20
"""Longest record kept while waiting for its terminator; `rg`'s own `--max-columns` keeps lines far shorter."""

_MAX_OUTPUT_BYTES = 8 << 20
"""Bytes of `rg` output brought back from the workspace; a search that prints more is reported as truncated."""

_TIMEOUT = 120.0
"""Deadline in seconds for one search, so a search over a huge tree cannot hang the tool call."""

_STATUS_PREFIX = '__harness_rg_status='
"""Prefix of the last stderr line, which carries `rg`'s own exit status past the `head` pipe."""

_MISSING = 127

_T = TypeVar('_T')


class RipgrepMissing(Exception):
    """`rg` is not on the workspace's PATH; the caller serves the request without it."""


@dataclass(kw_only=True, frozen=True)
class Record:
    """One line of ripgrep output: the file it refers to, and the rest of the line."""

    path: str
    """The path as `rg` printed it, relative to the directory it was run in."""
    text: str
    """Empty for a file listing; otherwise `<line>:<text>` for a match or `<line>-<text>` for context."""
    real_path: str | None = None
    """POSIX search may supply the canonical path alongside each candidate."""


async def run_ripgrep(
    workspace: Workspace,
    arguments: Sequence[str],
    *,
    cwd: str,
    limit: int,
    listing: bool = False,
    accept: Callable[[Record], _T | None],
    prepare: Callable[[list[Record]], Awaitable[None]] | None = None,
) -> tuple[list[_T], bool]:
    """Run `rg --null` in `cwd` inside the workspace; return up to `limit` accepted records and whether more were cut.

    `accept` maps a record to what the caller keeps, or `None` to drop it; only
    kept records count towards `limit`. `listing` reads `--files` output, where
    each record is a bare path. Raises `RipgrepMissing` when `rg` is not on the
    workspace's PATH, and `ModelRetry` when it reports an error (an invalid
    pattern, say), so the model can correct the call.
    """
    command = shlex.join(['rg', '--null', '--color=never', *arguments])
    script = (
        f'command -v rg > /dev/null 2>&1 || exit {_MISSING}\n'
        f'{{ {command}; echo "{_STATUS_PREFIX}$?" >&2; }} | head -c {_MAX_OUTPUT_BYTES}'
    )
    result = await workspace.run(script, shell=True, cwd=cwd, timeout=_TIMEOUT)
    stderr_lines = result.stderr.rstrip('\n').split('\n')
    status_line = stderr_lines[-1] if stderr_lines else ''
    detail = '\n'.join(stderr_lines[:-1]).strip()
    if result.exit_code == _MISSING or not status_line.startswith(_STATUS_PREFIX):
        if result.exit_code == _MISSING or 'not found' in result.stderr:
            raise RipgrepMissing
        raise ModelRetry(f'ripgrep failed: {result.stderr.strip() or f"exit code {result.exit_code}"}')
    status = status_line.removeprefix(_STATUS_PREFIX)

    output = result.stdout
    output_cut = len(output.encode('utf-8', errors='surrogateescape')) >= _MAX_OUTPUT_BYTES
    results: list[_T] = []
    records: list[Record] = []
    truncated = False
    terminator = '\0' if listing else '\n'
    start = 0
    while (end := output.find(terminator, start)) >= 0:
        line, start = output[start : end + 1], end + 1
        if len(line) > _MAX_RECORD_BYTES:
            truncated = True
            break
        if line == _SEPARATOR:
            continue
        records.append(_record(line, listing=listing))
    else:
        # Anything left is a record without its terminator: cut off by the output cap, or one
        # too long to keep, which a well-formed `rg` listing never prints.
        truncated = output_cut or len(output) - start > _MAX_RECORD_BYTES
    if prepare is not None:
        await prepare(records)
    for record in records:
        kept = accept(record)
        if kept is not None:
            if len(results) >= limit:
                truncated = True
                break
            results.append(kept)
    # `rg` exits 1 for "no match"; a cut output makes `rg` see a closed pipe, which is not its error.
    if not truncated and not output_cut and status not in ('0', '1'):
        raise ModelRetry(f'ripgrep failed: {detail or f"exit code {status}"}')
    return results, truncated


def _record(line: str, *, listing: bool) -> Record:
    if listing:
        return Record(path=line[:-1], text='')
    path, _, text = line.rstrip('\n').partition('\0')
    return Record(path=path, text=text)
