"""Claude's `Read` tool — read a UTF-8 text file."""

from .shared import attach_context, clip, resolve


def read_file(file_path: str, offset: int | None = None, limit: int | None = None) -> str:
    """Read a UTF-8 text file. Relative paths resolve under the workspace.

    Optional 1-based line `offset` and line `limit` mirror Claude's Read tool.
    """
    try:
        text = resolve(file_path).read_text(encoding='utf-8', errors='replace')
    except OSError as exc:
        return f'error: {exc}'
    ctx_prefix = attach_context(file_path)
    if offset is None and limit is None:
        return clip(ctx_prefix + text)
    lines = text.splitlines()
    start = max((offset or 1) - 1, 0)
    end = start + limit if limit else len(lines)
    return clip(ctx_prefix + '\n'.join(lines[start:end]))
