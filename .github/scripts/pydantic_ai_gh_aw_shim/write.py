"""Claude's `Write` tool — create or overwrite a workspace text file."""

from .shared import attach_context, resolve


def write_file(file_path: str, content: str) -> str:
    """Create or overwrite a UTF-8 text file under the workspace."""
    try:
        p = resolve(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        return attach_context(file_path) + f'wrote {len(content)} chars to {p}'
    except OSError as exc:
        return f'error: {exc}'
