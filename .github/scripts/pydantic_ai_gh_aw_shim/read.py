"""Claude's `Read` tool -- read a UTF-8 text file.

Backed by pydantic-ai-harness's `FileSystemToolset.read_file`: path containment,
symlink resolution, and binary-file detection come from the harness. The Claude
`Read` signature (1-based `offset`, `limit`) is preserved, and the
directory-scoped AGENTS.md / CLAUDE.md context blocks are still prepended.
"""

import re

from pydantic_ai.exceptions import ModelRetry

from ._backends import filesystem
from .shared import MAX_TOOL_OUTPUT, attach_context, clip

# When the harness truncates, it appends a continuation hint carrying *its* 0-based
# offset: `... (N more lines. Use offset=M to continue reading.)`. This tool exposes
# a 1-based offset (mirroring Claude's `Read`), so the advertised value must be
# bumped by one or a model that follows the hint literally re-reads the boundary
# line. The harness writes the hint at the start of its own line, while every
# content line is line-number-prefixed (`{n:>6}\t...`); anchoring to `^` (with
# `MULTILINE`) matches only the real hint, never a file whose own contents happen
# to reproduce the hint text.
_CONTINUE_HINT_RE = re.compile(r'^(\.\.\. \(\d+ more lines\. Use offset=)(\d+)( to continue reading\.\))', re.MULTILINE)

# The harness numbers each content line as `{lineno:>6}\t{text}` (1-based). A
# char-budget truncation reads that leading number back to advertise an exact
# continuation offset for the first line it dropped.
_LINE_NO_RE = re.compile(r' *(\d+)\t')


def _hint_to_one_based(body: str) -> str:
    """Rewrite the harness's 0-based continuation offset into this tool's 1-based one."""
    return _CONTINUE_HINT_RE.sub(lambda m: f'{m[1]}{int(m[2]) + 1}{m[3]}', body)


def _fit_to_output_budget(prefix: str, body: str) -> str:
    """Return `prefix + body`, truncated to the output cap on a whole-line boundary.

    `clip()` keeps the head and drops the tail, but the harness's continuation hint
    rides at the tail; a chunk well over the char cap (e.g. 2000 lines of code) would
    lose it, leaving the model with no next offset. Instead keep whole numbered lines
    that fit and re-advertise the 1-based offset of the first dropped line, so the
    Read round-trip stays exact even when the per-line cap and the per-char cap
    disagree.
    """
    out = prefix + body
    if len(out) <= MAX_TOOL_OUTPUT:
        return out
    size = len(prefix)
    kept: list[str] = []
    last_lineno: int | None = None
    for line in body.split('\n'):
        size += len(line) + 1
        if size > MAX_TOOL_OUTPUT and kept:
            break
        kept.append(line)
        numbered = _LINE_NO_RE.match(line)
        if numbered:
            last_lineno = int(numbered.group(1))
    if last_lineno is None:
        # Not even one whole numbered line fit (e.g. a single enormous line); fall
        # back to a plain head clip so the model still sees partial content.
        return clip(out)
    tail = f'... (truncated to fit the output limit. Use offset={last_lineno + 1} to continue reading.)'
    return prefix + '\n'.join(kept).rstrip('\n') + '\n' + tail


async def read_file(file_path: str, offset: int | None = None, limit: int | None = None) -> str:
    """Read a UTF-8 text file. Relative paths resolve under the workspace.

    Optional 1-based line `offset` and line `limit` mirror Claude's Read tool.
    """
    # Claude's `offset` is a 1-based line number; the harness uses a 0-based offset.
    zero_based = max((offset or 1) - 1, 0)
    # A non-positive `limit` is degenerate: passed straight through, `limit=0` makes
    # the harness read zero lines and emit a same-offset hint that loops forever.
    # Normalize it to `None` so it behaves exactly like an omitted `limit` (the
    # harness applies its default line cap) rather than reading nothing.
    effective_limit = limit if limit and limit > 0 else None
    try:
        body = await filesystem().read_file(file_path, offset=zero_based, limit=effective_limit)
    except (ModelRetry, OSError) as exc:
        # The harness only converts a fixed set of errors to `ModelRetry`; a bare
        # `OSError` (e.g. `ENAMETOOLONG` while resolving the path) would otherwise
        # escape and abort the whole run, where the old tool returned an error.
        return f'error: {exc}'
    return _fit_to_output_budget(attach_context(file_path), _hint_to_one_based(body))
