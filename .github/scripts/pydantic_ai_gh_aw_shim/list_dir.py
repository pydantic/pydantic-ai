"""Claude's `LS` tool -- list a workspace directory's entries.

Containment comes from a pydantic-ai-harness `FileSystemToolset.file_info`
preflight (the same check `Grep` uses), but the listing itself is hand-rolled
rather than delegated to `FileSystemToolset.list_directory`: the harness walker
hides every dot-prefixed entry, which would make `.github/` -- where gh-aw's own
workflows live -- invisible to the agent. The Claude `LS` signature is preserved
and the directory-scoped AGENTS.md / CLAUDE.md context blocks are still prepended.
"""

from pydantic_ai.exceptions import ModelRetry

from ._backends import filesystem
from .shared import attach_context, clip, resolve


async def list_dir(path: str = '.') -> str:
    """List a workspace directory's entries (directories marked with `/`)."""
    try:
        # `file_info` enforces workspace containment and rejects a missing path;
        # the enumeration then keeps the dot-prefixed entries the harness drops.
        await filesystem().file_info(path)
        entries = resolve(path).iterdir()
        listing = '\n'.join(sorted(e.name + ('/' if e.is_dir() else '') for e in entries)) or '(empty)'
    except (ModelRetry, OSError) as exc:
        # ModelRetry: the containment preflight rejected the path. OSError (e.g.
        # `NotADirectoryError` on a file, or `ENAMETOOLONG`): not recoverable in the
        # harness, so catch it here rather than let it abort the whole run.
        return f'error: {exc}'
    return clip(attach_context(path) + listing)
