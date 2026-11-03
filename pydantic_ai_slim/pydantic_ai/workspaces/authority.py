"""A policy wrapper that caps the authority of a workspace's files."""

from __future__ import annotations

from ..messages import Authority, _lower_authority  # pyright: ignore[reportPrivateUsage]
from .workspace import Workspace, WrapperWorkspace

__all__ = ('AuthorityWorkspace',)


class AuthorityWorkspace(WrapperWorkspace):
    """A [`Workspace`][pydantic_ai.workspaces.Workspace] whose files carry at most the given [authority][pydantic_ai.messages.Authority].

    Use `authority='none'` for files nobody on the application's side wrote, such as a cloned third-party
    repository or a sandbox another process filled, and `authority='user'` for the user's own files, such as
    their uploads. Every operation works as before. A capability that puts text read from the workspace in
    instructions should check [`authority`][pydantic_ai.workspaces.Workspace.authority] and send that text with
    no more authority than it has; no built-in capability reads it yet. This changes how file text reaches the
    model; it doesn't restrict what the agent can do in the workspace.
    """

    def __init__(self, wrapped: Workspace, *, authority: Authority):
        super().__init__(wrapped)
        self._authority: Authority = authority

    @property
    def authority(self) -> Authority:
        return _lower_authority(self._authority, super().authority)

    def durable_policy(self) -> tuple[object, ...]:
        return (self._authority,)
