"""Confine a path under a root, with every symlink followed.

A component that reads or writes under a root checks where a path really leads with `confine`.
`tests/harness/test_rooted_components.py` holds each such component to the same symlink cases.
"""

from __future__ import annotations as _annotations

from typing import NewType

from .workspace import Workspace

__all__ = ('ConfinedPath', 'confine')

ConfinedPath = NewType('ConfinedPath', str)
"""A real path that `confine` found inside its root."""


async def confine(workspace: Workspace, root: str, path: str, *, allow_root: bool = False) -> ConfinedPath | None:
    """Where `path` leads with every symlink followed, or `None` when that is not below `root`.

    `root` must already be a real path: whether a symlink at the root itself is followed, and whether the root
    is resolved once or on every call, is the caller's policy. With `allow_root`, the root itself is inside.

    The comparison is on the real path, so a link inside the root that leads out of it, a link mid-path and a
    `..` after a link are all refused. An error from `Workspace.realpath` propagates, so a link the workspace
    cannot read fails the call instead of passing it.
    """
    real = await workspace.realpath(path)
    sep = workspace.paths.sep
    if real.startswith(root.rstrip(sep) + sep) or (allow_root and real == root):
        return ConfinedPath(real)
    return None
