"""A policy wrapper that marks a workspace's files as not the operator's."""

from __future__ import annotations

from .workspace import WrapperWorkspace

__all__ = ('UntrustedWorkspace',)


class UntrustedWorkspace(WrapperWorkspace):
    """A [`Workspace`][pydantic_ai.workspaces.Workspace] whose files the operator doesn't control.

    Use it for a cloned third-party repository, a customer's upload, or a sandbox another process filled.
    Every operation works as before. Capabilities check
    [`trusted`][pydantic_ai.workspaces.Workspace.trusted] before putting text read from the workspace in
    instructions, and send it as tool output instead. This changes where file text reaches the model; it
    doesn't restrict what the agent can do in the workspace.
    """

    @property
    def trusted(self) -> bool:
        return False
