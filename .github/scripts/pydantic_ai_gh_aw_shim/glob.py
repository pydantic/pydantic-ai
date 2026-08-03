"""Claude's `Glob` tool -- list workspace paths matching a glob pattern.

Containment comes from a pydantic-ai-harness `FileSystemToolset.file_info`
preflight on the search `path` (the same check `Grep`/`LS` use) plus a
per-match resolved-path check, but the glob itself is hand-rolled with the stdlib
rather than delegated to `FileSystemToolset.find_files`: the harness walker hides
every dot-prefixed path, which would make `.github/**` -- where gh-aw's own
workflows live -- unmatchable. The Claude `Glob` signature is preserved, matches
are returned relative to the workspace root, and the directory-scoped AGENTS.md /
CLAUDE.md context blocks are still prepended.
"""

import glob as globlib
import os
import pathlib

from pydantic_ai.exceptions import ModelRetry

from ._backends import filesystem
from .shared import attach_context, clip, resolve, workspace


async def glob_search(pattern: str, path: str = '.') -> str:
    """Return workspace paths matching a glob `pattern` (supports `**`)."""
    # Claude's `Glob` takes a relative pattern plus a separate `path`; an absolute
    # pattern would be joined as-is and could escape the search root, so reject it.
    if os.path.isabs(pattern):
        return 'error: glob pattern must be relative to the search path'
    try:
        # `file_info` contains the search `path` (a clear error if it escapes); the
        # per-match resolve() + is_relative_to() below then contains the matches,
        # dropping anything a `..` pattern or an in-workspace symlink points to
        # outside the root (a purely lexical `relative_to` would not catch those).
        await filesystem().file_info(path)
        base = resolve(path)
        ws = pathlib.Path(workspace())
        root = ws.resolve()
        matches: list[str] = []
        for match in globlib.glob(str(base / pattern), recursive=True):
            # Resolve only to *decide* containment (collapse `..`, follow symlinks);
            # the returned path is the matched name relative to the workspace, so a
            # symlink that matched (e.g. `CLAUDE.md` -> `AGENTS.md`) is reported as
            # itself rather than silently rewritten to its target.
            if pathlib.Path(match).resolve().is_relative_to(root):
                matches.append(os.path.relpath(match, ws))
    except (ModelRetry, OSError, ValueError) as exc:
        # ModelRetry: the containment preflight rejected `path`. OSError (e.g.
        # `ENAMETOOLONG`) / ValueError: not recoverable in the harness, so catch
        # them here rather than let them abort the whole run.
        return f'error: {exc}'
    return clip(attach_context(path) + ('\n'.join(sorted(set(matches))) or 'No matches found.'))
