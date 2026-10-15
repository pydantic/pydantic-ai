"""Actionable terminal messages without changing exceptions delivered to plugins."""

from typing import cast

_DETAIL = 200
"""Characters of each failed model's error to show; provider error bodies can be long."""


def error_message(error: BaseException) -> str:
    """Recognize Codex refresh failures even when the SDK wraps them as connection errors.

    A fallback chain or `PROVIDER@*` that failed on every model lists why each one failed, in order.
    """
    from pydantic_ai.providers.openai_codex import CredentialsRefreshError

    if isinstance(error, BaseExceptionGroup):
        group = cast('BaseExceptionGroup[BaseException]', error)
        lines = [str(group).splitlines()[0]]
        for number, failure in enumerate(group.exceptions, start=1):
            detail = ' '.join(error_message(failure).split())
            if len(detail) > _DETAIL:
                detail = detail[: _DETAIL - 3] + '...'
            lines.append(f'  {number}. {type(failure).__name__}: {detail}')
        return '\n'.join(lines)
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, CredentialsRefreshError):
            return (
                # The error does not say which account failed, so point to every way back in.
                'Could not refresh your Codex login. Sign in again with /login openai-codex, or press Enter on '
                'the account in /accounts, then retry your message.'
            )
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return str(error)
