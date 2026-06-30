"""Claude's `LS` tool -- list a workspace directory's entries.

Backed by pydantic-ai-harness's `FileSystemToolset.list_directory`: path
containment and symlink resolution come from the harness, which lists entries
(relative to the workspace root, with type markers and sizes) and skips
dot-prefixed paths. The Claude `LS` signature is preserved and the
directory-scoped AGENTS.md / CLAUDE.md context blocks are still prepended.
"""

from pydantic_ai.exceptions import ModelRetry

from ._backends import filesystem
from .shared import attach_context, clip


async def list_dir(path: str = '.') -> str:
    """List a workspace directory's entries (directories marked with `/`)."""
    try:
        listing = await filesystem().list_directory(path)
    except ModelRetry as exc:
        return f'error: {exc}'
    return clip(attach_context(path) + listing)
