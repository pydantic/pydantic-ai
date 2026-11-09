"""A policy wrapper that caps the authority of a workspace's files."""

from __future__ import annotations

from collections.abc import Mapping

from ..exceptions import UserError
from ..messages import Authority
from ._authority_paths import AuthorityLayer
from .workspace import Workspace, WrapperWorkspace

__all__ = ('AuthorityWorkspace',)


class AuthorityWorkspace(WrapperWorkspace):
    """A [`Workspace`][pydantic_ai.workspaces.Workspace] whose files carry at most the given [authority][pydantic_ai.messages.Authority].

    Use `authority='none'` for files nobody on the application's side wrote, such as a cloned third-party
    repository or a sandbox another process filled, and `authority='user'` for the user's own files, such as
    their uploads. Every operation works as before. A capability that puts text read from the workspace in front
    of the model sends it with no more authority than
    [`authority_of`][pydantic_ai.workspaces.Workspace.authority_of] gives its file, as a
    [capability part](../capabilities/custom.md#capability-parts) whose source is
    [`WorkspaceSource.from_workspace`][pydantic_ai.messages.WorkspaceSource.from_workspace]. This changes how
    file text reaches the model; it doesn't restrict what the agent can do in the workspace.

    `path_authority` gives directories and files their own level, such as `'none'` for a vendored dependency in the
    operator's repository, or `'operator'` for the operator's own instruction files next to a cloned one. Paths
    resolve against the working directory (`~` is not expanded, so a path starting with it raises `UserError`), and
    the longest matching path wins. A path outside the working directory
    is part of the workspace only when some `path_authority` names it. Wrapping never raises a level: with several
    wrappers, a file gets the lowest level any of them gives it.
    """

    def __init__(
        self, wrapped: Workspace, *, authority: Authority, path_authority: Mapping[str, Authority] | None = None
    ):
        super().__init__(wrapped)
        self._authority: Authority = authority
        self._path_authority: dict[str, Authority] = dict(path_authority or {})
        if home_relative := [path for path in self._path_authority if path.startswith('~')]:
            # Read as relative, `~/x` would name `<working_dir>/~/x`, which the workspace's files can create.
            raise UserError(
                f'`path_authority` paths are workspace paths, so `~` is not expanded: got {home_relative[0]!r}. '
                'Pass an absolute path, or use `LocalWorkspace(path_authority=...)`, which expands `~` on this machine.'
            )

    def _authority_layers(self) -> list[AuthorityLayer]:
        return [*super()._authority_layers(), (self._authority, self._path_authority)]

    def durable_policy(self) -> tuple[object, ...]:
        return (self._authority, self._path_authority)
