"""Claude's `Glob` tool — list workspace paths matching a glob pattern."""

import glob as globlib
import pathlib

from .shared import attach_context, clip, resolve


def glob_search(pattern: str, path: str = '.') -> str:
    """Return workspace paths matching a glob `pattern` (supports `**`)."""
    base = resolve(path)
    try:
        matches = sorted(
            str(pathlib.Path(m).relative_to(base)) for m in globlib.glob(str(base / pattern), recursive=True)
        )
    except (OSError, ValueError) as exc:
        # ValueError: a match resolved outside `base` (e.g. an absolute pattern).
        return f'error: {exc}'
    return clip(attach_context(path) + ('\n'.join(matches) or '(no matches)'))
