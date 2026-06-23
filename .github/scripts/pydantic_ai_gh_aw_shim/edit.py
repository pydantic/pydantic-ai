"""Claude's `Edit` tool -- replace a string in a workspace file.

The single-occurrence case is backed by pydantic-ai-harness's
`FileSystemToolset.edit_file`, which requires `old_string` to match exactly once
-- the same uniqueness rule Claude Code's own `Edit` enforces. `replace_all=True`
has no harness equivalent, so it keeps the prior in-place read/replace/write.
"""

from pydantic_ai.exceptions import ModelRetry

from ._backends import filesystem
from .shared import attach_context, resolve


async def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace `old_string` with `new_string` in a workspace file.

    Replaces the single (unique) occurrence, or every occurrence when `replace_all`.
    """
    if replace_all:
        return _replace_all(file_path, old_string, new_string)
    try:
        result = await filesystem().edit_file(file_path, old_string, new_string)
    except ModelRetry as exc:
        return f'error: {exc}'
    return attach_context(file_path) + result


def _replace_all(file_path: str, old_string: str, new_string: str) -> str:
    """Replace every occurrence of `old_string`.

    The harness `edit_file` rejects non-unique matches, so replace-all stays an
    in-place rewrite (as the tool did before it was harness-backed).
    """
    try:
        p = resolve(file_path)
        text = p.read_text(encoding='utf-8')
        if old_string not in text:
            return 'error: `old_string` not found'
        p.write_text(text.replace(old_string, new_string, -1), encoding='utf-8')
        return attach_context(file_path) + f'edited {p}'
    except OSError as exc:
        return f'error: {exc}'
