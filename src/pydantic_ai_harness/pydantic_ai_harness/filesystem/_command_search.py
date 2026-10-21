"""One in-workspace POSIX search for command-capable workspaces without ripgrep."""

from __future__ import annotations

import re
import shlex
from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.workspaces import Workspace

from ._ripgrep import Record, Unreadable

_T = TypeVar('_T')
_MAX_OUTPUT_BYTES = 8 << 20
_MAX_RECORD_BYTES = 1 << 20
_STATUS = '__harness_posix_status='
_UNREADABLE = '__harness_posix_unreadable='
"""Prefix of the NUL-terminated stderr record naming a file `grep` could not read."""


def validate_posix_pattern(pattern: str) -> None:
    """Reject rg syntax that POSIX ERE would interpret as different text."""
    # POSIX ERE lacks rg's \d/\w/\s classes, lazy quantifiers, named groups,
    # inline flags and Unicode properties; lookaround/backreferences are unsupported by rg too.
    if re.search(r'\\[dDsSwWpP]|\(\?|[+*?]\?|\{[0-9,]+\}\?', pattern):
        raise ValueError(
            'This regex requires ripgrep; POSIX grep does not support that syntax. Use literal=True or install rg.'
        )


async def run_posix_search(
    workspace: Workspace,
    *,
    cwd: str,
    target: str = '.',
    explicit_file: bool = False,
    pattern: str | None = None,
    literal: bool = False,
    ignore_case: bool = False,
    context: int = 0,
    include_hidden: bool = False,
    limit: int,
    accept: Callable[[Record], Awaitable[_T | None]],
    prepare: Callable[[list[Record]], Awaitable[None]] | None = None,
) -> tuple[list[_T], bool, list[Unreadable]]:
    """Enumerate sorted files and optionally grep them without transferring their contents to the host.

    Returns the accepted records, whether more were cut, and the files `grep` could not read,
    which are skipped rather than failing the search.
    """
    if pattern is not None and not literal:
        validate_posix_pattern(pattern)
    # Git applies nested .gitignore files, but --exclude-from=.ignore only reads the
    # search-root .ignore; nested .ignore rules need rg. Without git, find ignores neither.
    # --exclude-standard keeps tracked files, and even untracked dotfiles; like rg without
    # --hidden, skip every path with a dot-prefixed component before it is grepped.
    pathspec = shlex.quote(target) + ('' if include_hidden else " ':(exclude,glob)**/.*' ':(exclude,glob)**/.*/**'")
    enumeration = (
        'if [ -f .ignore ]; then extra=--exclude-from=.ignore; else extra=; fi; '
        'if command -v git >/dev/null 2>&1; then '
        'if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then '
        'git ls-files -co --exclude-standard $extra -z -- ' + pathspec + '; else tmp=$(mktemp -d) || exit 2; '
        'git init --bare -q "$tmp" || exit 2; '
        'GIT_DIR="$tmp" GIT_WORK_TREE="$PWD" git ls-files -o --exclude-standard '
        '-z $extra -- ' + pathspec + '; status=$?; rm -rf -- "$tmp"; [ "$status" -eq 0 ]; fi; '
        'else find '
        + shlex.quote(target)
        + (" ! -path . -name '.*' -prune -o " if not include_hidden else ' ')
        + '-type f -print0; fi'
    )
    if explicit_file:
        # An explicit path bypasses gitignore, but still passes through canonical authorization.
        enumeration = 'printf "%s\\0" ' + shlex.quote(target)
    # Carry the canonical path with each candidate, so a single search command can
    # authorize thousands of files without per-result sandbox round trips.
    canonical = 'real=$(realpath -- "$file" && printf .) || continue; real=${real%.}; real=${real%?}; '
    if pattern is None:
        processing = "xargs -0 sh -c 'for file do " + canonical + 'printf "%s\\0%s\\0" "$file" "$real"; done\' sh'
    else:
        flags = '-nIhF' if literal else '-nIhE'
        if ignore_case:
            flags += 'i'
        if context:
            flags += f' -C {context}'
        # Grep's 1 means no matches, while 2 (including an invalid ERE or a read error)
        # must not be turned into a plausible empty result by xargs or the output pipe.
        processing = (
            "xargs -0 sh -c 'tmp=$(mktemp) || exit 2; "
            # When head closes the pipe at the cap, remove the temp file, then die of SIGPIPE
            # again so xargs still reports the signal that marks the output as cut.
            'trap "rm -f -- \\"\\$tmp\\"; trap - PIPE; kill -PIPE \\$\\$" PIPE; '
            'pattern=$1; shift; for file do ' + canonical + f'grep {flags} -e "$pattern" -- "$file" > "$tmp"; code=$?; '
            # A file grep cannot read is skipped and reported; any other error (an invalid ERE
            # fails on every file) still ends the search.
            'if [ "$code" -gt 1 ]; then if [ -r "$file" ] || [ ! -e "$file" ]; then rm -f -- "$tmp"; exit "$code"; fi; '
            f'printf "{_UNREADABLE}%s\\0" "$file" >&2; continue; fi; '
            # With -n, every output line is numbered except the `--` between context groups.
            'while IFS= read -r line; do [ "$line" = -- ] && continue; '
            'printf "%s\\0%s\\0%s\\n" "$file" "$real" "$line"; done < "$tmp"; '
            'done; rm -f -- "$tmp"\' sh ' + shlex.quote(pattern)
        )
    # Capture enumeration's and sort's statuses separately: a failed git/find/sort must not
    # masquerade as an empty successful search through the pipeline's final xargs status.
    script = (
        f'cd {shlex.quote(cwd)} || exit\n'
        '{ list=$(mktemp) || exit 2; '
        f'{{ {enumeration}; }} > "$list"; code=$?; '
        'if [ "$code" -eq 0 ]; then LC_ALL=C sort -z -o "$list" "$list"; code=$?; fi; '
        f'if [ "$code" -eq 0 ]; then {processing} < "$list"; code=$?; fi; rm -f -- "$list"; '
        f'echo "{_STATUS}$code" >&2; }} | head -c {_MAX_OUTPUT_BYTES}'
    )
    result = await workspace.run(script, shell=True, timeout=120)
    stderr, _, status = result.stderr.rpartition(_STATUS)
    unreadable = [Unreadable(path, 'Permission denied') for path in re.findall(f'{_UNREADABLE}([^\0]*)\0', stderr)]
    stderr = re.sub(f'{_UNREADABLE}[^\0]*\0', '', stderr)
    output = result.stdout
    cut = len(output.encode('utf-8', errors='surrogateescape')) >= _MAX_OUTPUT_BYTES
    # head closes the pipe at the cap; xargs can report its child dying of SIGPIPE.
    pipe_cut = cut and ('signal 13' in stderr or status.strip() == '141')
    if result.exit_code != 0 or not status or (status.strip() != '0' and not pipe_cut):
        raise ModelRetry(f'POSIX search failed: {stderr.strip() or result.stderr.strip() or result.exit_code}')
    results: list[_T] = []
    records, incomplete = _parse_records(output, listing=pattern is None)
    cut |= incomplete
    # Every caller passes `prepare`.
    if prepare is not None:  # pragma: no branch
        await prepare(records)
    for record in records:
        kept = await accept(record)
        if kept is not None:
            if len(results) >= limit:
                cut = True
                break
            results.append(kept)
    return results, cut, unreadable


def _parse_records(output: str, *, listing: bool) -> tuple[list[Record], bool]:
    records: list[Record] = []
    start = 0
    incomplete = False
    while (end := output.find('\0', start)) >= 0:
        path = output[start:end]
        real_end = output.find('\0', end + 1)
        if real_end < 0:
            incomplete = True
            break
        real_path = output[end + 1 : real_end]
        if listing:
            text = ''
            start = real_end + 1
        else:
            line_end = output.find('\n', real_end + 1)
            if line_end < 0:
                incomplete = True
                break
            text = output[real_end + 1 : line_end]
            start = line_end + 1
        if start - end > _MAX_RECORD_BYTES:
            # Drop only the oversized record; later records are still complete.
            incomplete = True
            continue
        records.append(Record(path=path, text=text, real_path=real_path))
    return records, incomplete or start < len(output)
