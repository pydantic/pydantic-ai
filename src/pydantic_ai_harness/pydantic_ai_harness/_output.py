"""Shared helpers for model-visible tool output."""


def truncate_head(text: str, max_chars: int) -> str:
    """Limit text to `max_chars`, keeping the head.

    Suits page and document text, where the lead carries the substance. The
    truncation marker counts toward the cap, so the returned text stays within
    the bound; a cap smaller than the marker returns the marker alone.
    """
    if len(text) <= max_chars:
        return text
    marker = f'\n[... page text truncated at {max_chars} characters]'
    return f'{text[: max(max_chars - len(marker), 0)]}{marker}'


def truncate_tail(text: str, max_chars: int) -> str:
    """Limit text to `max_chars`, keeping the tail.

    The truncation marker counts toward the cap and reports the exact number
    of retained chars. A cap too small to fit any marker returns a bare tail.
    """
    if max_chars <= 0:  # pragma: no cover - callers validate the cap at construction
        return ''
    if len(text) <= max_chars:
        return text

    def marker(tail_chars: int) -> str:
        return f'[... output truncated, showing last {tail_chars} chars]\n'

    tail_chars = max(0, max_chars - len(marker(max_chars)))
    if tail_chars + 1 + len(marker(tail_chars + 1)) <= max_chars:
        tail_chars += 1
    if tail_chars == 0:
        return text[-max_chars:]
    return marker(tail_chars) + text[-tail_chars:]
