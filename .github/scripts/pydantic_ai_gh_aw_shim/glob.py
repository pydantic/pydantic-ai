"""Claude's `Glob` tool -- list workspace paths matching a glob pattern.

Backed by pydantic-ai-harness's `FileSystemToolset.find_files`: path containment
comes from the harness, which returns matches relative to the workspace root
(directories marked with `/`) and skips dot-prefixed paths. The Claude `Glob`
signature is preserved and the directory-scoped AGENTS.md / CLAUDE.md context
blocks are still prepended.
"""

import os

from pydantic_ai.exceptions import ModelRetry

from ._backends import filesystem
from .shared import attach_context, clip


async def glob_search(pattern: str, path: str = '.') -> str:
    """Return workspace paths matching a glob `pattern` (supports `**`)."""
    # `find_files` (via `pathlib.glob`) raises a *non-recoverable*
    # `NotImplementedError` ("Non-relative patterns are unsupported") for an
    # absolute pattern, which would abort the whole run rather than come back as
    # a string. Claude's `Glob` takes a relative pattern plus a separate `path`,
    # so reject absolute patterns up front; the `except` is a belt-and-suspenders
    # guard in case a platform's `isabs`/`glob` disagree on what is "relative".
    if os.path.isabs(pattern):
        return 'error: glob pattern must be relative to the search path'
    try:
        matches = await filesystem().find_files(pattern, path=path)
    except ModelRetry as exc:
        return f'error: {exc}'
    except NotImplementedError as exc:
        return f'error: {exc}'
    return clip(attach_context(path) + matches)
