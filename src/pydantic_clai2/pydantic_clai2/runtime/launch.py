"""The rules for `--resume`, `--session-id`, and `--fork-session`, shared by the CLI and the Python API."""

from uuid import UUID


def launch_session_id(*, resume: str | None, session_id: str | None, fork: bool) -> str | None:
    """Check the launch options as Claude Code takes them; return `session_id` as CLAI saves it.

    Raises `ValueError` for `fork` without `resume`, `session_id` with `resume` but without `fork`,
    or a `session_id` that is not a UUID.
    """
    # Messages name the keyword and the flag, as both `chat(...)` and the CLI raise them.
    if fork and resume is None:
        raise ValueError('`fork_session` (`--fork-session`) requires `resume` (`--resume`)')
    if session_id is None:
        return None
    if resume is not None and not fork:
        raise ValueError(
            '`session_id` (`--session-id`) can only be combined with `resume` (`--resume`)'
            ' when `fork_session` (`--fork-session`) is given'
        )
    try:
        return str(UUID(session_id))
    except ValueError:
        raise ValueError(f'`session_id` (`--session-id`) {session_id!r} is not a valid UUID') from None
