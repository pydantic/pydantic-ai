"""Shared helpers for model-visible tool output."""


def validated_budget(name: str, value: object, *, limit: int | None = None) -> int:
    """Reject a budget that is not a positive integer, or that is above `limit`.

    The annotation is advisory: a config-driven caller can pass a float or a
    string, which would only fail later as a slice bound. Negative budgets are
    worse than useless here: they slice from the end, so they would return the
    wrong text rather than fail. A `bool` is an `int` too, but `True` is not a
    budget anyone means.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'`{name}` must be a positive integer, got {value!r}')
    if limit is not None and value > limit:
        raise ValueError(f'`{name}` must be at most {limit}, got {value!r}')
    return value


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


def truncate_head_within(text: str, max_chars: int) -> str:
    """Like `truncate_head`, but a cap too small for the marker ends the text with `…` instead, so it stays a ceiling."""
    truncated = truncate_head(text, max_chars)
    if len(truncated) > max_chars:
        return f'{text[: max_chars - 1]}…'
    return truncated


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
