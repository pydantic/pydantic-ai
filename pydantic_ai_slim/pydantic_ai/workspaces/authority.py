"""A policy wrapper that caps the authority of a workspace's files."""

from __future__ import annotations

from typing import Final

from ..messages import Authority
from .workspace import Workspace, WrapperWorkspace

__all__ = ('AuthorityWorkspace',)

_AUTHORITY_RANK: Final[dict[Authority, int]] = {'none': 0, 'user': 1, 'operator': 2}


class AuthorityWorkspace(WrapperWorkspace):
    """A [`Workspace`][pydantic_ai.workspaces.Workspace] whose files carry at most the given [authority][pydantic_ai.messages.Authority].

    Use `authority='none'` for files nobody on the application's side wrote, such as a cloned third-party
    repository or a sandbox another process filled, and `authority='user'` for the user's own files, such as
    their uploads. Every operation works as before. A capability that puts text read from the workspace in front
    of the model sends it with no more authority than [`authority`][pydantic_ai.workspaces.Workspace.authority],
    as a [capability part](../capabilities/custom.md#capability-parts) whose source is
    [`WorkspaceSource.from_workspace`][pydantic_ai.messages.WorkspaceSource.from_workspace]. This changes how
    file text reaches the model; it doesn't restrict what the agent can do in the workspace.
    """

    def __init__(self, wrapped: Workspace, *, authority: Authority):
        super().__init__(wrapped)
        self._authority: Authority = authority

    @property
    def authority(self) -> Authority:
        inner = super().authority
        return self._authority if _AUTHORITY_RANK[self._authority] < _AUTHORITY_RANK[inner] else inner

    def durable_policy(self) -> tuple[object, ...]:
        return (self._authority,)
