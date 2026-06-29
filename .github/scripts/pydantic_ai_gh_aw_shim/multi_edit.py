"""Claude's `MultiEdit` tool — apply multiple string replacements to one file atomically."""

# pydantic requires typing_extensions.TypedDict (not typing.TypedDict) for
# schema generation on Python < 3.12; typing_extensions ships with pydantic.
from typing_extensions import NotRequired, TypedDict

from .shared import attach_context, resolve


class EditOp(TypedDict):
    """One replacement for `MultiEdit` (Claude's edit schema)."""

    old_string: str
    new_string: str
    replace_all: NotRequired[bool]


def multi_edit(file_path: str, edits: list[EditOp]) -> str:
    """Apply a sequence of string replacements to one workspace file atomically.

    Each edit replaces the first occurrence of `old_string` (or every
    occurrence when `replace_all`). If any `old_string` is missing, nothing is
    written — the file is left untouched.
    """
    try:
        p = resolve(file_path)
        text = original = p.read_text(encoding='utf-8')
    except OSError as exc:
        return f'error: {exc}'
    for i, e in enumerate(edits):
        old = e.get('old_string', '')
        if not old or old not in text:
            return f'error: edit #{i + 1} `old_string` not found (no changes written)'
        text = text.replace(old, e.get('new_string', ''), -1 if e.get('replace_all') else 1)
    if text == original:
        return 'no changes'
    try:
        p.write_text(text, encoding='utf-8')
    except OSError as exc:
        return f'error: {exc}'
    return attach_context(file_path) + f'applied {len(edits)} edit(s) to {p}'
