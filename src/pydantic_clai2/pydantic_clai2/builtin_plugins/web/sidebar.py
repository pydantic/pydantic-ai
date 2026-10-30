"""The sidebar's saved sessions: each one's row, and the checkouts of a project."""

import os
from collections.abc import Mapping
from pathlib import Path, PurePath

from pydantic_ai_harness.step_persistence.conversations import ConversationSummary
from pydantic_clai2.builtin_plugins.web.protocol import Project, SessionEntry, SessionState, Worktree
from pydantic_clai2.plugins import LiveSession
from pydantic_clai2.runtime.project_identity import (
    MISSING,
    ProjectIdentity,
    git_checkout,
    project_identity,
    worktree_roots,
)

COMPLETIONS = 12
"""How many directories New session completes a typed path to; each costs a few Git calls."""


def session_entry(
    summary: ConversationSummary, *, live: LiveSession | None, busy: bool, current: bool, openable: bool
) -> SessionEntry:
    """A saved session's row: `running` from the live session holding it; `busy` only for another process's turn."""
    state: SessionState = summary.outcome
    if live is not None:
        state = 'running' if live.running else 'ready' if summary.outcome == 'running' else summary.outcome
    elif summary.outcome == 'running':
        state = 'busy' if busy else 'cancelled'
    return SessionEntry(
        id=summary.id,
        title=summary.title,
        workspace=summary.workspace,
        updated_at=summary.updated_at,
        created_at=summary.created_at or summary.updated_at,
        state=state,
        owner_pid=summary.owner_pid if state == 'busy' else None,
        messages=summary.message_count,
        model=summary.model,
        current=current,
        archived=summary.archived_at is not None,
        # Another CLAI's session shows read-only until the user forks it, as its transcript says.
        openable=openable and state != 'busy',
    )


def worktrees(
    found: Mapping[str, tuple[ProjectIdentity, list[SessionEntry]]], *, here: ProjectIdentity
) -> list[Worktree]:
    """A project's checkouts, the current one first and removed ones last, each labelled by its branch or else its folder.

    A label two of them share, such as two removed folders of one name, shows the folder's path instead.
    """
    labels = {root: identity.checkout or PurePath(root).name or root for root, (identity, _) in found.items()}
    worktrees = [
        Worktree(
            root=root,
            label=labels[root] if list(labels.values()).count(labels[root]) == 1 else root,
            current=(identity.key, root) == (here.key, here.root),
            missing=identity.missing,
            sessions=sessions,
        )
        for root, (identity, sessions) in found.items()
    ]
    return sorted(worktrees, key=lambda worktree: (not worktree.current, worktree.missing))


def project_names(identities: Mapping[str, ProjectIdentity]) -> dict[str, str]:
    """Each project's name by its key, as the sidebar heads it.

    Projects that share a name each show the shortest end of their folder's path that no other of them shares,
    as `a/scratch` and `b/scratch`, or else their key.
    """
    # A repository's key is its Git folder, as its name is: `.git` inside the main checkout, or the folder itself.
    folders = {key: PurePath(key).parent if PurePath(key).name == '.git' else PurePath(key) for key in identities}
    names = {key: identity.name for key, identity in identities.items()}
    for key, folder in folders.items():
        name = identities[key].name
        others = [other for other in folders if other != key and identities[other].name == name]
        if not others:
            continue
        for depth in range(2, len(folder.parts) + 1):
            label = folder.parts[-depth:]
            if all(folders[other].parts[-depth:] != label for other in others):
                names[key] = str(PurePath(*label))
                break
        else:
            # Equal folders, as after Git failed once while listing one, still differ by key; Removed folders has none.
            if key != MISSING:
                names[key] = key
    return names


def completions(typed: str) -> list[Project]:
    """The directories an absolute path typed in New session completes to, each repository with every worktree Git lists.

    Hidden directories only when the typed name starts with a dot; a subfolder of a checkout is offered as itself.
    """
    path = Path(typed)
    if not path.is_absolute():
        return []
    folder, prefix = (path, '') if typed.endswith(('/', os.sep)) else (path.parent, path.name)
    try:
        names = sorted(
            entry.name
            for entry in os.scandir(folder)
            if entry.name.startswith(prefix)
            and (prefix.startswith('.') or not entry.name.startswith('.'))
            and entry.is_dir()
        )
    except (OSError, ValueError):
        # `ValueError`: a typed NUL byte.
        return []
    found: dict[str, list[ProjectIdentity]] = {}
    for name in names[:COMPLETIONS]:
        directory = str(folder / name)
        identity = project_identity(directory)
        if git_checkout(identity) is None or Path(identity.root) != Path(directory).resolve():
            found[directory] = [ProjectIdentity(key=directory, name=name, root=directory)]
        elif identity.key not in found:
            found[identity.key] = [project_identity(root) for root in worktree_roots(identity)] or [identity]
    return [
        Project(
            key=key,
            name=checkouts[0].name,
            current=False,
            worktrees=[
                Worktree(
                    root=checkout.root,
                    label=checkout.checkout or PurePath(checkout.root).name,
                    current=False,
                    missing=checkout.missing,
                    sessions=[],
                )
                for checkout in checkouts
            ],
        )
        for key, checkouts in found.items()
    ]
