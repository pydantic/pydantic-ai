"""Run ripgrep inside the workspace and split its NUL-delimited output into records.

`rg` is invoked with `--null`, so every file path it prints ends in a NUL byte
and cannot be confused with the `:`/`-` separators of the match text that
follows it. The workspace returns a command's output whole, so the output is
cut inside the workspace at `_MAX_OUTPUT_BYTES` (`head -c`) and records are then
streamed through the caller's `accept` filter until `limit` accepted records
have been collected; only kept records count towards the cap.
"""

from __future__ import annotations

import re
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

_PREPARE_BATCH = 500
"""Records handed to `prepare` at once."""

_MAX_OUTPUT_BYTES = 8 << 20
"""Bytes of `rg` output brought back from the workspace; a search that prints more is reported as truncated."""

_TIMEOUT = 120.0
"""Deadline in seconds for one search, so a search over a huge tree cannot hang the tool call."""

_STATUS_PREFIX = '__harness_rg_status='
"""Prefix of the last stderr line, which carries `rg`'s own exit status past the `head` pipe."""

_MISSING = 127

_ERROR_STATUS = '2'

_UNREADABLE = re.compile(r'rg: (?P<path>.+?): (?:IO error for operation on .+?: )?(?P<reason>[^:]+?) \(os error \d+\)')
"""An error `rg` reports for one path it could not open, after which it carries on with the rest."""

_T = TypeVar('_T')


class RipgrepMissing(Exception):
    """`rg` is not on the workspace's PATH; the caller serves the request without it."""


@dataclass(frozen=True)
class Unreadable:
    """A path the search could not read, and why; the rest of the search went on without it."""

    path: str
    """The path as the search printed it, relative to the directory it was run in."""
    reason: str
    """The operating system's reason, such as `Permission denied`."""


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
) -> tuple[list[_T], bool, list[Unreadable]]:
    """Run `rg --null` in `cwd` inside the workspace.

    Returns up to `limit` accepted records, whether more were cut, and the paths `rg` could not
    read. `accept` maps a record to what the caller keeps, or `None` to drop it; only kept records
    count towards `limit`. `listing` reads `--files` output, where each record is a bare path.
    Raises `RipgrepMissing` when `rg` is not on the workspace's PATH, and `ModelRetry` when it
    reports any other error (an invalid pattern, say), so the model can correct the call.
    """
    command = shlex.join(['rg', '--null', '--color=never', *arguments])
    script = (
        f'cd {shlex.quote(cwd)} || exit\n'
        f'command -v rg > /dev/null 2>&1 || exit {_MISSING}\n'
        f'{{ {command}; echo "{_STATUS_PREFIX}$?" >&2; }} | head -c {_MAX_OUTPUT_BYTES}'
    )
    result = await workspace.run(script, shell=True, timeout=_TIMEOUT)
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
    # Prepared a batch at a time, so a capped search stops preparing once it has enough.
    for offset in range(0, len(records), _PREPARE_BATCH):
        batch = records[offset : offset + _PREPARE_BATCH]
        # Every caller passes `prepare`.
        if prepare is not None:  # pragma: no branch
            await prepare(batch)
        if not _collect(batch, accept, results, limit):
            truncated = True
            break
    # `rg` exits 2 when a path could not be read but still searches the rest, so that is a
    # partial result; any other error line (an invalid pattern, say) fails the call.
    errors = [line for line in detail.split('\n') if line]
    unreadable = [Unreadable(m['path'], m['reason']) for line in errors if (m := _UNREADABLE.fullmatch(line))]
    partial = status == _ERROR_STATUS and len(unreadable) == len(errors) > 0
    # `rg` exits 1 for "no match"; a cut output makes `rg` see a closed pipe, which is not its error.
    if not truncated and not output_cut and status not in ('0', '1') and not partial:
        raise ModelRetry(f'ripgrep failed: {detail or f"exit code {status}"}')
    return results, truncated, unreadable


def _collect(batch: list[Record], accept: Callable[[Record], _T | None], results: list[_T], limit: int) -> bool:
    """Append what `accept` keeps from `batch` to `results`; `False` once one more than `limit` was kept."""
    for record in batch:
        kept = accept(record)
        if kept is not None:
            if len(results) >= limit:
                return False
            results.append(kept)
    return True


def _record(line: str, *, listing: bool) -> Record:
    if listing:
        return Record(path=line[:-1], text='')
    path, _, text = line.rstrip('\n').partition('\0')
    return Record(path=path, text=text)
