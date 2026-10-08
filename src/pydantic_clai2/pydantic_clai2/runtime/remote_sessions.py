"""Sessions stored outside this machine, which `/resume` and `--resume` can continue (hackathon: Logfire).

A plugin that stores sessions elsewhere installs a `RemoteSessions` while it is loaded. `/resume ARG` asks it
first. It either names a local session to resume as it is, or hands over the history to save and resume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.messages import ModelMessage


@dataclass(frozen=True, kw_only=True)
class RemoteResume:
    """What resuming a remote session comes down to."""

    conversation_id: str
    """The CLAI ID to resume: the remote session's own (continuing it), or a new one (a fork)."""
    messages: list[ModelMessage] | None
    """The history to save under `conversation_id` first; `None` resumes the local copy as it is."""
    title: str = ''
    subtitle: str = ''
    notice: str = ''
    """Said before the usual resume notice, such as where the history came from."""


class RemoteSessions(Protocol):
    """Somewhere sessions live besides this machine's store."""

    async def resume(self, reference: str, *, local: bool) -> RemoteResume | None:
        """How to resume `reference` (an ID or a link); `None` when it does not know it.

        `local` says whether this machine has a saved session with that ID. Raises `LookupError` for a link
        it should know but cannot find, and `ValueError` when the stored session is incomplete.
        """
        ...


_installed: list[RemoteSessions] = []


def install(source: RemoteSessions | None) -> None:
    """Install (or, with `None`, remove) the remote store `/resume` consults."""
    _installed[:] = [source] if source is not None else []


def current() -> RemoteSessions | None:
    """The installed remote store, if a loaded plugin provides one."""
    return _installed[0] if _installed else None
