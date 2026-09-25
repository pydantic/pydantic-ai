"""Comparing two workspace selections, for the checks that a selection did not change."""

from __future__ import annotations as _annotations

from .workspace import Workspace, WrapperWorkspace


def policy_chain(workspace: Workspace) -> list[tuple[type[Workspace], dict[str, object]]]:
    """Each facade layer from the outside in, with its own state, which is where workspace policy lives.

    A layer's state is its instance attributes other than the wrapped workspace, compared with
    `==`, so two `ReadOnlyWorkspace`s match while two allowlisting wrappers with different lists do
    not. The innermost backend is left out: both sides are built from the same ref, so they name
    the same environment, and a backend's configuration has no general comparison.
    """
    chain: list[tuple[type[Workspace], dict[str, object]]] = []
    while True:
        state = {name: value for name, value in vars(workspace).items() if name != '_backend'}
        chain.append((type(workspace), state))
        if not isinstance(workspace, WrapperWorkspace):
            return chain
        workspace = workspace.wrapped


def same_workspace(a: Workspace, b: Workspace) -> bool:
    """Whether two selections name the same environment under the same policy.

    Compares the facade layers' state, the innermost backend's type, and the ref; a backend's
    own configuration has no general comparison.
    """
    return policy_chain(a) == policy_chain(b) and type(a.backend) is type(b.backend) and a.ref == b.ref
