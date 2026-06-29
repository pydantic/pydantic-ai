"""Claude's `Edit` tool — replace a string in a workspace file."""

from .shared import attach_context, resolve


def edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace `old_string` with `new_string` in a workspace file.

    Replaces the first occurrence, or every occurrence when `replace_all`.
    """
    try:
        p = resolve(file_path)
        text = p.read_text(encoding='utf-8')
        if old_string not in text:
            return 'error: `old_string` not found'
        edited = text.replace(old_string, new_string, -1 if replace_all else 1)
        p.write_text(edited, encoding='utf-8')
        return attach_context(file_path) + f'edited {p}'
    except OSError as exc:
        return f'error: {exc}'
