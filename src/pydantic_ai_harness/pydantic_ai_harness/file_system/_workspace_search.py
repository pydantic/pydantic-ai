"""Limits and shell shared by the searches that run a command in the workspace: `rg` and the POSIX fallback."""

from __future__ import annotations

MAX_OUTPUT_BYTES = 8 << 20
"""Bytes of search output brought back from the workspace; a search that prints more is reported as truncated."""

MAX_RECORD_BYTES = 1 << 20
"""Longest record a workspace search keeps while waiting for its terminator.

`rg`'s own `--max-columns` keeps its lines far shorter; the POSIX fallback has no such option.
"""

SEARCH_TIMEOUT = 120.0
"""Deadline in seconds for one workspace search command, so a search over a huge tree cannot hang the tool call."""


def realpath_snippet(on_failure: str) -> str:
    """Shell source that sets `real` to the canonical path of `$file`, running `on_failure` when it cannot be resolved.

    `realpath` strips trailing newlines from its output, so a `.` is printed after it and both are
    stripped again, keeping a file name that ends in a newline intact.
    """
    return f'real=$(realpath -- "$file" && printf .) || {on_failure}; real=${{real%.}}; real=${{real%?}}; '
