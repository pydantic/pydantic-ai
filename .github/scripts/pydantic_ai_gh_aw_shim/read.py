"""Claude's `Read` tool -- read a UTF-8 text file.

Backed by pydantic-ai-harness's `FileSystemToolset.read_file`: path containment,
symlink resolution, and binary-file detection come from the harness. The Claude
`Read` signature (1-based `offset`, `limit`) is preserved, and the
directory-scoped AGENTS.md / CLAUDE.md context blocks are still prepended.
"""

from pydantic_ai.exceptions import ModelRetry

from ._backends import filesystem
from .shared import attach_context, clip


async def read_file(file_path: str, offset: int | None = None, limit: int | None = None) -> str:
    """Read a UTF-8 text file. Relative paths resolve under the workspace.

    Optional 1-based line `offset` and line `limit` mirror Claude's Read tool.
    """
    # Claude's `offset` is a 1-based line number; the harness uses a 0-based offset.
    zero_based = max((offset or 1) - 1, 0)
    try:
        body = await filesystem().read_file(file_path, offset=zero_based, limit=limit)
    except ModelRetry as exc:
        return f'error: {exc}'
    return clip(attach_context(file_path) + body)
