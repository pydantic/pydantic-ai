"""Actionable terminal messages without changing exceptions delivered to plugins."""


def error_message(error: BaseException) -> str:
    """Recognize Codex refresh failures even when the SDK wraps them as connection errors."""
    from pydantic_ai.providers.openai_codex import CredentialsRefreshError  # noqa: PLC0415

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, CredentialsRefreshError):
            return (
                'Could not refresh your Codex login. Run /login openai-codex to sign in again, then retry your message.'
            )
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return str(error)
