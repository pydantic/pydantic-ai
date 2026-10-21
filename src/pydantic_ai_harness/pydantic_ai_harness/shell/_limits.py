"""Per-file size limits for commands, applied with the workspace shell's `ulimit -f`."""

from __future__ import annotations

import sys

_SIGXFSZ = 25
"""`SIGXFSZ` on Linux and the BSDs, read by the workspace shell rather than the host `signal` module.

A shell reports a child killed by it as `128 + 25` (`256 + 25` in ksh93); a direct child reports `-25`.
"""

LIMIT_FUNCTION = r"""__harness_limit_files() {
  __harness_probe=$(mktemp "${TMPDIR:-/tmp}/pydantic-ai-harness-ulimit.XXXXXX") || return 1
  ( trap '' XFSZ; ulimit -f 1 && printf '%01100d' 0 > "$__harness_probe" ) >/dev/null 2>&1
  __harness_unit=$(wc -c < "$__harness_probe" | tr -d ' ')
  rm -f "$__harness_probe"
  case $__harness_unit in 512|1024) ;; *) return 1 ;; esac
  ulimit -f $(( ($1 + __harness_unit - 1) / __harness_unit ))
}"""
"""Shell function that sets the per-file limit to `$1` bytes, rounded up to whole `ulimit -f` blocks.

POSIX `sh` counts `ulimit -f` in 512-byte blocks and bash counts it in 1 KiB blocks (macOS
`/bin/sh` is bash), so the function measures the block size once by writing past a one-block
limit in a subshell, then sets the limit for the calling shell and its children.
"""

_APPLY_FAILED = "{ echo 'Unable to apply max_file_bytes.' >&2; exit 1; }"


def validate_file_limit(limit: object, *, persistent: bool) -> None:
    if limit is None:
        return
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError('max_file_bytes must be a positive integer.')
    if persistent:
        raise ValueError('max_file_bytes is not supported with the persistent shell tool.')
    if limit > sys.maxsize:
        raise ValueError('max_file_bytes must not exceed sys.maxsize.')


def limited_script(command: str, limit: int | None) -> str:
    """`command` preceded by the limit, as one script for the workspace shell; unchanged without a limit."""
    if limit is None:
        return command
    return f'{LIMIT_FUNCTION}\n__harness_limit_files {limit} || {_APPLY_FAILED}\n{command}'


def file_limit_status(exit_code: int, limit: int | None) -> str:
    if limit is None or exit_code == 0:
        return ''
    if exit_code in (-_SIGXFSZ, 128 + _SIGXFSZ, 256 + _SIGXFSZ):
        return f'\n[File-size limit exceeded: max_file_bytes={limit}. Reduce file output before retrying.]'
    return f'\n[Command failed with max_file_bytes={limit}; file writes may have reached the per-file limit.]'
